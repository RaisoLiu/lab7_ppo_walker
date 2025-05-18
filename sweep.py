import argparse
import wandb
import sys
import os

# 超參數搜索空間定義
sweep_config = {
    'method': 'bayes',  # 使用貝葉斯優化方法
    'metric': {
        'name': 'best_avg_episode_reward',  # 優化測試平均回報值
        'goal': 'maximize'  # 目標是最大化回報
    },
    'parameters': {
        'actor_lr': {
            'values': [1e-4, 3e-4, 5e-4, 7e-4, 9e-4]
        },
        'critic_lr': {
            'values': [1e-4, 3e-4, 5e-4, 7e-4, 9e-4]
        },
        'test_seed': {
            'values': [42, 77, 123, 456, 789, 1000, 1234, 1456, 1789, 2000]
        },
        'discount_factor': {
            'values': [0.9, 0.95, 0.99, 0.995]
        },
        'entropy_weight': {
            'values': [0, 0.01, 0.05, 0.1, 0.2]
        },
        'max_env_step': {
            'values': [3e6] # fixed
        },
        'rollout_step': {
            'values': [1024, 2048, 4096, 8192]
        },
        'tau': {
            'values': [0.9, 0.95, 0.99, 0.995]
        },
        'update_epoch': {
            'values': [5, 10, 20, 40]
        },
        'grad_clip': {
            'values': [0.5, 1.0, 2.0, 5.0]
        },
        'batch_size': {
            'values': [32, 64, 128, 256, 512]
        },
        'epsilon': {
            'values': [0.1, 0.2, 0.3]
        },
        'num_test_episodes': {
            'values': [5, 10, 20, 40]
        },
        'num_save_step': {
            'values': [1e5] # fixed
        }
    }
}

def init_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="要執行的 sweep 運行次數")
    parser.add_argument("--sweep-folder", type=str, default="sweep-PPO-Walker2d-2")
    parser.add_argument("--wandb-project", type=str, default="PPO-Walker2d-sweep-2")
    parser.add_argument("--wandb-entity", type=str, required=True, help="Weights & Biases 實體名稱（用戶名或團隊名）")
    parser.add_argument("--env-name", type=str, default="Walker2d-v4")
    parser.add_argument("--python-path", type=str, help="Python 解釋器路徑（如果使用虛擬環境）")
    args = parser.parse_args()
    
    print("初始化 sweep: ")
    print(args)
    
    # 設置 Python 解釋器路徑
    if args.python_path:
        sweep_config['program'] = args.python_path
    else:
        # 使用當前 Python 解釋器路徑
        sweep_config['program'] = sys.executable
    
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    print(f"\nSweep ID: {sweep_id}")
    print("\n使用以下命令在 agent 機器上運行：")
    print(f"方法 1 - 使用 wandb agent 命令：")
    print(f"wandb agent {args.wandb_entity}/{args.wandb_project}/{sweep_id}")
    print(f"\n方法 2 - 使用 python agent.py：")
    print(f"python agent.py --wandb-entity {args.wandb_entity} --wandb-project {args.wandb_project} --sweep-id {sweep_id}")

if __name__ == "__main__":
    init_sweep()







