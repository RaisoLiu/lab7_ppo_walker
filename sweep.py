import argparse

config = argparse.Namespace(
        actor_lr=3e-4,  # 較高的學習率
        critic_lr=3e-3,  # 較高的學習率
        gamma=0.95,      # 對Pendulum使用0.9的折扣因子
        entropy_beta=2.5e-4,  # 減小熵權重，增加利用率
        num_episodes=1000,
        steps_on_memory=16,  # 合適的記憶步數
        save_per_epoch=100,
        seed=42,
        max_grad_norm=10.0   # 較小的梯度裁剪閾值，增加穩定性
    )



def train_agent_with_config(use_wandb: bool, test_mode: bool):


    pass
