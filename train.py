import argparse

from sweep import train_agent_with_config

# 超參數搜索空間定義
sweep_config = {
    'method': 'bayes',  # 使用貝葉斯優化方法
    'metric': {
        'name': 'avg_test_reward',  # 優化測試平均回報值
        'goal': 'maximize'  # 目標是最大化回報
    },
    'parameters': {
        'actor_lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'critic_lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'gamma': {
            'distribution': 'log_uniform_values',
            'min': 0.85,
            'max': 0.999
        },
        'entropy_beta': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'max_grad_norm': {
            'values': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]  # 測試不同的梯度裁剪值
        },
        'num_episodes': {
            'value': 1000  # 固定值，不要修改
        },
        'steps_on_memory': {
            'values': [4, 16, 64, 256, 1024]  # 記憶的步數
        },
        'save_per_epoch': {
            'value': 100  # 每 100 個回合保存一次檢查點
        },
        'seed': {
            'values': [42, 77, 123, 456, 789, 1000, 1234, 1456, 1789, 2000, 2234, 2456, 2789, 3000, 3234, 3456, 3789, 4000, 4234, 4456, 4789, 5000]  # 多個隨機種子以提高穩定性
        }
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500, help="要執行的 sweep 運行次數")
    parser.add_argument("--no-use-wandb", action="store_true", help="不使用 wandb 進行實驗追蹤")
    parser.add_argument("--test", action="store_true", help="以測試模式運行（減少回合數）")
    args = parser.parse_args()
    
    use_wandb = not args.no_use_wandb
    test_mode = args.test
    
    if use_wandb:
        import wandb
        # 初始化 sweep
        sweep_id = wandb.sweep(sweep_config, project=f"PPO-Pendulum")
        
        # 執行 sweep
        wandb.agent(sweep_id, lambda: train_agent_with_config(use_wandb=True, test_mode=test_mode), count=args.count)
    else:
        print("以無wandb模式運行單次訓練...")
        train_agent_with_config(use_wandb=False, test_mode=test_mode) 