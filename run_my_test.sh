#!/bin/bash
source ~/.bashrc
source .venv/bin/activate
echo $PWD
.venv/bin/python test.py --require-video --ckpt-path result-PPO-Pendulum/best_ckpt.pt