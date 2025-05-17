import argparse
import wandb
import copy
import os
from train import Trainer

def train():
    # 從命令行參數獲取基本設置
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="Walker2d-v4")
    parser.add_argument("--sweep-folder", type=str, default="sweep-PPO-Walker2d")
    parser.add_argument("--wandb-entity", type=str, required=True, help="Weights & Biases 實體名稱（用戶名或團隊名）")
    parser.add_argument("--wandb-project", type=str, required=True, help="Weights & Biases 專案名稱")
    parser.add_argument("--sweep-id", type=str, required=True, help="Sweep ID")
    args = parser.parse_args()
    
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    config = wandb.config
    print(config)
    
    # 更新訓練參數
    train_args = copy.copy(args)
    train_args.actor_lr = config.actor_lr
    train_args.critic_lr = config.critic_lr
    train_args.test_seed = config.test_seed
    train_args.max_env_step = config.max_env_step
    train_args.discount_factor = config.discount_factor
    train_args.entropy_weight = config.entropy_weight
    train_args.tau = config.tau
    train_args.batch_size = config.batch_size
    train_args.epsilon = config.epsilon
    train_args.rollout_step = config.rollout_step
    train_args.update_epoch = config.update_epoch
    train_args.grad_clip = config.grad_clip
    train_args.num_test_episodes = config.num_test_episodes
    train_args.num_save_step = config.num_save_step
    train_args.save_dir = os.path.join(args.sweep_folder, run.name)
    train_args.use_print = False
    train_args.use_wandb = True
    train_args.device = args.device
    # 在這裡添加您的訓練代碼
    trainer = Trainer(train_args, run)
    trainer.train()

if __name__ == "__main__":
    # 從命令行參數獲取基本設置
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="Walker2d-v4")
    parser.add_argument("--sweep-folder", type=str, default="sweep-PPO-Walker2d")
    parser.add_argument("--wandb-entity", type=str, required=True, help="Weights & Biases 實體名稱（用戶名或團隊名）")
    parser.add_argument("--wandb-project", type=str, required=True, help="Weights & Biases 專案名稱")
    parser.add_argument("--sweep-id", type=str, required=True, help="Sweep ID")
    parser.add_argument("--count", type=int, default=10, help="要執行的 sweep 運行次數")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    # 使用 wandb agent 運行訓練
    wandb.agent(
        f"{args.wandb_entity}/{args.wandb_project}/{args.sweep_id}",
        function=train,
        count=args.count  # 每次運行一個實驗
    ) 