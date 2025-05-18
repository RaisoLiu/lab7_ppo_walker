# agent.py

import argparse
import wandb
import copy
import os
from train import Trainer

# --------------- 只在模块顶端解析一次 ---------------
parser = argparse.ArgumentParser()
parser.add_argument("--env-name", type=str, default="Walker2d-v4")
parser.add_argument("--sweep-folder",  type=str, default="sweep-PPO-Walker2d")
parser.add_argument("--wandb-entity",  type=str, required=True, help="Weights & Biases 实体名称（用户名或团队名）")
parser.add_argument("--wandb-project", type=str, required=True, help="Weights & Biases 项目名称")
parser.add_argument("--sweep-id",      type=str, required=True, help="Sweep ID")
parser.add_argument("--count",         type=int, default=10,  help="要执行的 sweep 运行次数")
parser.add_argument("--device",        type=str, default=None)
args = parser.parse_args()


def train():
    # 初始化 wandb Run
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=args  # 先把 args 传进去，后面会 override
    )
    config = wandb.config

    # 构造实际传给 Trainer 的参数
    train_args = copy.copy(args)
    # 从 sweep config 里覆盖超参数
    for attr in [
        "actor_lr","critic_lr","test_seed","max_env_step","discount_factor",
        "entropy_weight","tau","batch_size","epsilon",
        "rollout_step","update_epoch","grad_clip",
        "num_test_episodes","num_save_step"
    ]:
        setattr(train_args, attr, getattr(config, attr))

    train_args.save_dir   = os.path.join(args.sweep_folder, run.name)
    train_args.use_print  = False
    train_args.use_wandb  = True
    train_args.device     = args.device

    trainer = Trainer(train_args, run)
    trainer.train()


if __name__ == "__main__":
    # 用 wandb.agent 来启动 sweep
    sweep_str = f"{args.wandb_entity}/{args.wandb_project}/{args.sweep_id}"
    wandb.agent(
        sweep_str,
        function=train,
        count=args.count
    )
