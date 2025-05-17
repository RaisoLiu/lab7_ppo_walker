import argparse
from env import Environment
from models import Actor, Critic
import torch

class Tester:
    def __init__(self, args, actor_ckpt=None, critic_ckpt=None):
        self.args = args
        if not hasattr(args, "device") or args.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)
        self.require_video = True if hasattr(args, "require_video") and args.require_video else False
        self.env = Environment(args, mode="test", seed=args.seed, require_video=self.require_video)
        self.actor = Actor(self.env.get_obs_dim(), self.env.get_action_dim()).to(self.device)
        self.critic = Critic(self.env.get_obs_dim()).to(self.device)
        
        if actor_ckpt is not None:
            self.actor.load_state_dict(actor_ckpt)
        if critic_ckpt is not None:
            self.critic.load_state_dict(critic_ckpt)
        
        
        self.avg_reward = 0

    def get_avg_reward(self):
        return self.avg_reward
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        _, dist = self.actor(state_tensor)
        action = dist.mean
        return action, None

    def test(self):
        episode_reward_list = []
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            for episode in range(self.args.num_test_episodes):
                state = self.env.reset()
                episode_reward = 0
                while True:
                    action, _ = self.select_action(state)
                    next_state, reward, done = self.env.step(action.cpu().detach().numpy())
                    episode_reward += reward
                    state = next_state
                    if done:
                        break
                episode_reward_list.append(episode_reward)
                if self.args.use_print:
                    print(f"test episode: {episode}, reward: {episode_reward}")
        self.avg_reward = sum(episode_reward_list) / len(episode_reward_list)
        self.env.close()



    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--test-folder", type=str, default="test-PPO-Walker2d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-test-episodes", type=int, default=20, help="number of episodes for testing")
    parser.add_argument("--require-video", default=True,action="store_true")
    parser.add_argument("--env-name", type=str, default="Walker2d-v4")
    parser.add_argument("--use-print", default=True, action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    print("start test: ")
    print(args)

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    actor_ckpt  = ckpt['actor_state_dict']
    critic_ckpt = ckpt['critic_state_dict']

    tester = Tester(args, actor_ckpt, critic_ckpt)

    tester.test()
    print(f"獎勵：{tester.get_avg_reward()}")