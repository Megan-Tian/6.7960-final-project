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
        self.seq_rewards = np.zeros(self.seq_len, dtype=np.float32)
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
        
        self.seq_rewards[:-1] = self.seq_rewards[1:]
        self.seq_rewards[-1] = true_reward
        
        # compute predicted reward
        seq_obs = torch.tensor(self.seq_obs, dtype=torch.float32)
        seq_actions = torch.tensor(self.seq_actions, dtype=torch.float32)
        seq_rewards = torch.tensor(self.seq_rewards, dtype=torch.float32)
        
        # FIXME ---------------------------
        # check how i'm implementing the reward saving - is it per step the same
        # as obs and actions? or the full reward for the trajectory?
        pred_reward = self.reward_predictor.predict(seq_obs, seq_actions)
        self.reward_predictor.add_temp_experience(seq_obs, seq_actions, seq_rewards)
        # ---------------------------------
        
        info['pred_reward'] = pred_reward.item()
        info['true_reward'] = true_reward
                
        return obs, pred_reward.item(), terminated, truncated, info
    

def check_custom_env():
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
    # print(env.seq_actions.shape)
    # print(env.seq_obs.shape)
    
    try:
        check_env(env)
    except Exception as e:
        print(f'Failed check_env w/ exception: {e}')
    else:
        print('Success check_env(RLHFEnv)')

def random_policy():
    env = make_ll_env(render_mode='human')
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    env = RLHFEnv(env, 
                  RewardPredictor(obs_shape, 
                                  action_shape, 
                                  seq_len=10,
                                  n_predictors=10,
                                  ),
                  seq_len=10)
   
    # generate two trajectory segments, each with random actions and observations
    for _ in range(3):
        obs, info = env.reset()    
        i = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            # print(info)
            
            i += 1
    
        # check recorded experience lengths
        print(f"episode length = {i}")
        print(f"env.reward_predictor.temp_experience['true_reward'] = {len(env.reward_predictor.temp_experience['true_reward'])}")
        print(f"env.reward_predictor.temp_experience['seq_obs'] = {len(env.reward_predictor.temp_experience['seq_obs'])}")
        print(f"env.reward_predictor.temp_experience['seq_actions'] = {len(env.reward_predictor.temp_experience['seq_actions'])}")
    
    print('=' * 100)
    # dataset manipulation in the reward predictor
    k = 5
    print(f'Reward predictor dataset = {env.reward_predictor.dataset}')
    print(f'Getting synthetic feedback for k = {k}')
    env.reward_predictor.get_synthetic_feedback(k)
    print(f'Reward predictor dataset len = {len(env.reward_predictor.dataset)}')
    print(f'Reward predictor dataset entries: obs1 {env.reward_predictor.dataset[0][0][0].shape} | obs2 {env.reward_predictor.dataset[0][1][0].shape} | gamma {env.reward_predictor.dataset[0][2]}')
    print('=' * 100)
    
if __name__ == "__main__":
    check_custom_env()
    random_policy()
    