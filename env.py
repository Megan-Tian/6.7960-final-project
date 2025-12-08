import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env

from reward import RewardPredictor
from oracle import make_ll_env


class RLHFEnv(gym.Wrapper):
    def __init__(self, 
                 env : gym.Env, 
                 reward_model : RewardPredictor,
                 seq_len : int = 10) -> None:
        
        super(RLHFEnv, self).__init__(env)
        self.seq_len = seq_len
        self.reward_predictor = reward_model
    
    def reset(self, **kwargs):
        self.seq_actions = np.zeros((self.seq_len, self.action_space.n), dtype=np.float32)
        self.seq_obs = np.zeros([self.seq_len] + [i for i in self.observation_space.shape], dtype=np.float32)
        
        obs, info = self.env.reset(**kwargs)
        self.seq_obs[-1] = obs
        
        return obs, info
    
    def step(self, action):
        """
        Takes a step in the underlying environment but returns the *predicted*
        reward from `self.reward_predictor`
        """
        obs, true_reward, terminated, truncated, info = self.env.step(action)
        
        # shift actions/obs one left FIFO and append most recent action/obs to the end
        self.seq_actions[:-1] = self.seq_actions[1:]
        self.seq_actions[-1] = action
        self.seq_obs[:-1] = self.seq_obs[1:]
        self.seq_obs[-1] = obs
        
        # compute predicted reward
        seq_obs = torch.tensor(self.seq_obs, dtype=torch.float32)
        seq_actions = torch.tensor(self.seq_actions, dtype=torch.float32)
        
        # FIXME ---------------------------
        pred_reward = self.reward_predictor.predict(seq_obs, seq_actions)
        self.reward_predictor.add_temp_experience(seq_obs, seq_actions, true_reward)
        # ---------------------------------
        
        info['pred_reward'] = pred_reward.item()
        info['true_reward'] = true_reward
        
        # printpred_reward))
        
        return obs, pred_reward.item(), terminated, truncated, info
    
    
    
    
if __name__ == "__main__":
    env = make_ll_env()
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    env = RLHFEnv(env, 
                  RewardPredictor(obs_shape, 
                                  action_shape, 
                                  seq_len=10,
                                  n_predictors=10,
                                  ),
                  seq_len=10)
    env.reset()
    print(env.seq_actions.shape)
    print(env.seq_obs.shape)
    
    try:
        check_env(env)
    except Exception as e:
        print(f'Failed check_env w/ exception: {e}')
    else:
        print('Success check_env(RLHFEnv)')