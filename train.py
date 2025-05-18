import argparse
import os
import copy
from test import Tester
import torch
from models import Actor, Critic
from tqdm import tqdm
from env import Environment
import torch.optim as optim
import torch.nn.functional as F
from utils import Memory, Logger
import yaml





class Trainer:
    def __init__(self, args, wandb_run=None):
        self.args = args
        self.env = Environment(args, mode="train")
        self.state_dim = self.env.get_obs_dim()
        self.action_dim = self.env.get_action_dim()
        self.discount_factor = args.discount_factor
        print(f"state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        if not hasattr(args, "device") or args.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)
        print(f"Using device: {self.device}")

        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.epsilon = args.epsilon
        self.entropy_weight = args.entropy_weight
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.memory = Memory(args, self.device)

        self.logger = Logger(args, wandb_run)
        

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)



        if hasattr(args, "ckpt_path") and args.ckpt_path is not None:
            self.load_ckpt(args.ckpt_path)

    def select_action(self, state, global_step):
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_from_actor, dist = self.actor(state_tensor)
        log_prob = dist.log_prob(action_from_actor)
        self.logger.histogram_log({
            "global_step": global_step,
        }, {
            "action": action_from_actor.cpu().detach().numpy()[0],
        })
        return action_from_actor, log_prob
    
    def get_value(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        value = self.critic(state_tensor)
        return value

    def train(self):
        best_avg_episode_reward = -float("inf")
        best_ckpt_path = None
        episode_reward_list = []
        episode_count = 0
        episode_step = 0
        global_step = 0
        pbar = tqdm(total=int(self.args.max_env_step), desc="Training Progress")

        rollout_step = 0
        state = self.env.reset()
        episode_reward = 0
        while global_step < self.args.max_env_step:
            global_step += 1
            episode_step += 1

            if global_step % self.args.num_save_step == 0:
                self.save_ckpt(mode="regular", global_step=global_step)

            if rollout_step < self.args.rollout_step:
                rollout_step += 1
                action, log_prob = self.select_action(state, global_step)
                value = self.get_value(state)
                next_state, reward, done = self.env.step(action.cpu().detach().numpy())
                next_value = self.get_value(next_state)
                episode_reward += reward
                
                self.memory.add(state, action, log_prob.detach(), reward, next_state, value, next_value, done)
                state = next_state
                if done:
                    episode_reward_list.append(episode_reward)
                    episode_count += 1
                    self.logger.log({"episode": episode_count, "global_step": global_step, "episode_step": episode_step, "episode_reward": episode_reward})
                    
                    episode_reward = 0
                    episode_step = 0
                    state = self.env.reset()
                continue

            self.update(global_step=global_step)
            rollout_step = 0
            self.memory.clear() # clear memory
            avg_reward = self.validation()
            pbar.update(self.args.rollout_step)
            pbar.set_description(f"Reward: {avg_reward:.2f}")
            if avg_reward > best_avg_episode_reward:
                best_avg_episode_reward = avg_reward
                self.save_ckpt(mode="best", episode=episode_count)
                self.logger.log({
                    "global_step": global_step,
                    "episode_count": episode_count,
                    "best_avg_episode_reward": best_avg_episode_reward,
                })

                
    def update(self, global_step):
        data_loader = self.memory.get_dataloader()
        actor_loss_list = []
        critic_loss_list = []
        entropy_bonus_list = []

        for _ in tqdm(range(self.args.update_epoch), desc="Updating"): # update_epoch times
            for batch in data_loader:
                state, action, log_prob, reward, next_state, done, advantage_gae, advantage_gae_norm, values = batch
                state = state.to(self.device)
                action = action.to(self.device)
                log_prob = log_prob.to(self.device)
                reward = reward.to(self.device)
                next_state = next_state.to(self.device)
                advantage_gae = advantage_gae.to(self.device)
                advantage_gae_norm = advantage_gae_norm.to(self.device)
                done = done.to(self.device)
                critic_loss = self.update_critic(state, reward, next_state, advantage_gae, values, done)
                actor_loss, entropy_bonus = self.update_actor(state, action, log_prob, advantage_gae_norm)
                actor_loss_list.append(actor_loss.item())
                entropy_bonus_list.append(entropy_bonus.item())
                critic_loss_list.append(critic_loss.item())

        self.logger.histogram_log({
            "global_step": global_step,
        }, {
            "actor_loss": actor_loss_list,
            "critic_loss": critic_loss_list,
            "entropy_bonus": entropy_bonus_list
        })
        
    def update_critic(self, state, reward, next_state, advantage_gae, values, done):
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            target_value = values + advantage_gae
        value_pred = self.critic(state)
        critic_loss = F.mse_loss(value_pred, target_value)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.args.grad_clip) 
        self.critic_optimizer.step()
        return critic_loss

    def update_actor(self, state, action, old_log_prob, advantage_gae_norm):
        self.actor_optimizer.zero_grad()
        _, dist = self.actor(state)

        log_prob = dist.log_prob(action)
        ratio = (log_prob - old_log_prob).exp()
        surr1 = ratio * advantage_gae_norm
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_gae_norm
        policy_loss_elementwise = -torch.min(surr1, surr2)

        entropy_bonus = dist.entropy().sum(dim=-1).mean()
        actor_loss = policy_loss_elementwise.mean() - self.entropy_weight * entropy_bonus
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.args.grad_clip) 
        self.actor_optimizer.step()
        return actor_loss, entropy_bonus




    def save_ckpt(self, mode="best", episode=None, global_step=None):
        if mode == "best":
            ckpt_path = self.get_best_ckpt_path()
        elif mode == "regular":
            ckpt_path = self.get_regular_ckpt_path(global_step)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        print(f"\n\nSaving checkpoint to {ckpt_path}\n\n")
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, ckpt_path)

    def load_ckpt(self, ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        NotImplementedError("Not implemented")

    def validation(self):
        test_args = copy.copy(self.args)
        test_args.seed = self.args.test_seed
        test_args.ckpt_path = None

        tester = Tester(test_args, 
                        actor_ckpt=self.actor.state_dict(), 
                        critic_ckpt=self.critic.state_dict())
        tester.test()
        avg_reward = tester.get_avg_reward()

        return avg_reward


    def get_best_ckpt_path(self):
        return os.path.join(self.save_dir, "best_ckpt.pt")

    def get_regular_ckpt_path(self, global_step):
        return os.path.join(self.save_dir, f"ckpt_{global_step}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--test-seed", type=int, default=42)
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--max-env-step", type=int, default=5e5)
    parser.add_argument("--entropy-weight", type=float, default=0)  # 修改為 float
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=0.2)  # 修改為 float
    parser.add_argument("--rollout-step", type=int, default=2048*8)  
    parser.add_argument("--update-epoch", type=int, default=20)  # 修改為 int
    parser.add_argument("--num-test-episodes", type=int, default=10)
    parser.add_argument("--num-save-step", type=int, default=1e5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-dir", type=str, default="result-PPO-Walker2d-fix-critic-loss")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--use-print", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="PPO-Walker2d-fix-critic-loss")
    parser.add_argument("--env-name", type=str, default="Walker2d-v4")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        args.max_env_step = 1000
        args.num_test_episodes = 1
        args.num_save_step = 500
        args.rollout_step = 100
        args.update_epoch = 1
    
    print("start train: ")
    print(args)

    

    # 創建保存目錄
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 將參數轉換為字典
    config_dict = vars(args)
    
    # 保存配置到yaml文件
    config_path = os.path.join(args.save_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    trainer = Trainer(args)
    trainer.train()

