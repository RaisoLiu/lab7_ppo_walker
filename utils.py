from typing import List
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class Memory:
    def __init__(self, args, device):
        self.args = args
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.device = device

        self.clear()

    
    def add(self, state, action, log_prob, reward, next_state, value, next_value, done):
        state = torch.FloatTensor(state)
        reward = torch.FloatTensor(np.expand_dims(reward, axis=0))  # (1,)
        done = torch.FloatTensor(np.expand_dims(done, axis=0))  # (1,)
        next_state = torch.FloatTensor(next_state)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)
        self.next_values.append(next_value)
        self.dones.append(done)
        if done:
            self.compute_advantage_gae()
            

    def compute_advantage_gae(self):
        gae_list = []
        gae = 0
        l_index, r_index = len(self.advantage_gae), len(self.dones)
        
        for r, d, v, nv in zip(reversed(self.rewards[l_index:r_index]), reversed(self.dones[l_index:r_index]), reversed(self.values[l_index:r_index]), reversed(self.next_values[l_index:r_index])):
            r = r.to(self.device)
            d = d.to(self.device)
            v = v.to(self.device)
            nv = nv.to(self.device)
            delta = r + self.gamma * nv * (1 - d) - v
            gae = gae * self.gamma * self.tau + delta
            gae_list.append(gae.detach())

        self.advantage_gae.extend(reversed(gae_list))




    def get_dataloader(self):
        self.compute_advantage_gae()
        n = len(self.advantage_gae)
        
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        rewards = torch.stack(self.rewards)
        next_states = torch.stack(self.next_states)
        dones = torch.stack(self.dones)
        advantage_gae = torch.stack(self.advantage_gae)
        values = torch.stack(self.values)
        advantage_gae_norm = (advantage_gae - advantage_gae.mean()) / (advantage_gae.std() + 1e-8)
        advantage_gae_norm = advantage_gae_norm.detach()


        # create dataloader
        dataset = TensorDataset(states, actions, log_probs, rewards, next_states, dones, advantage_gae, advantage_gae_norm, values)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        if hasattr(self.args, "dry_run") and self.args.dry_run:
            print("="*30)
            print("Dataloader:")
            print(f"n: {n}", "len(dataloader):", len(dataloader))
            print(f"states: {states.shape}")
            print(f"actions: {actions.shape}")
            print(f"log_probs: {log_probs.shape}")
            print(f"rewards: {rewards.shape}")
            print(f"next_states: {next_states.shape}")
            print(f"dones: {dones.shape}")
            print(f"advantage_gae: {advantage_gae.shape}")
            print(f"advantage_gae_norm: {advantage_gae_norm.shape}")
            print("="*30)
        return dataloader

    def clear(self):
        self.states = []
        self.actions= []
        self.rewards= []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.next_values = []
        self.advantage_gae = []
        self.advantage_gae_norm = []


import wandb

class Logger:
    def __init__(self, args, wandb_run=None):
        self.args = args
        self.use_wandb = args.use_wandb
        self.use_print = args.use_print
        if self.use_wandb:
            self.wandb_run = wandb_run
            if self.wandb_run is None:
                self.wandb_run = wandb.init(project=args.wandb_project, config=args, save_code=True)
    
    def log(self, dict_msg):
        if self.use_wandb:
            self.wandb_run.log(dict_msg)
        elif self.use_print:
            print(dict_msg)

    def histogram_log(self, dict_msg, histogram_dist_msg):
        if self.use_wandb:
            upload_dict = {}
            for key, value in dict_msg.items():
                upload_dict[f"{key}"] = value
            for key, value in histogram_dist_msg.items():
                upload_dict[f"{key}"] = wandb.Histogram(value)
            self.wandb_run.log(upload_dict)
        elif self.use_print:
            print(dict_msg)
            print(histogram_dist_msg)
    