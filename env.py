import gymnasium as gym
import os
import numpy as np


class Environment:
    def __init__(self, args, mode="train", seed=None, require_video=False):
        self.args = args
        self.mode = mode
        self.seed = seed
        self.episode_count = 0
        if self.mode == "train":
            self.env = gym.make(self.args.env_name)
        elif self.mode == "test" and require_video:
            os.environ["MUJOCO_GL"] = "egl"
            def always_trigger_video(episode_id):
                return True
            env = gym.make(self.args.env_name, render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(env, video_folder=self.args.test_folder, episode_trigger=always_trigger_video)
        else:
            self.env = gym.make(self.args.env_name)

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

    def reset(self):
        if self.mode == "train":
            state, _ = self.env.reset()
        else:
            seed = self.seed + self.episode_count
            state, _ = self.env.reset(seed=seed)
        self.episode_count += 1
        return state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action) 
        done = terminated or truncated
        return next_state, reward, done
    
    def get_obs_dim(self):
        return self.obs_dim
    
    def get_action_dim(self):
        return self.action_dim
    
    def close(self):
        self.env.close()