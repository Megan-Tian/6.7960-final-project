import math
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy

ORACLE_NAME = 'ppo_lunarlander'

def make_ll_env(render_mode = None):
    return gym.make(
            "LunarLander-v3", 
            continuous=False, 
            gravity=-10.0,
            enable_wind=False, 
            wind_power=15.0, 
            render_mode = render_mode
        )
    
def train(save_model_name=ORACLE_NAME, train_from_checkpoint=None):
    '''
    Args:
        save_model_name: name of the saved model file
        train_from_checpoint: name of the full saved model .zip to start training from.
        If `none` a model is trained from scratch
    '''
    env = make_ll_env()
    if train_from_checkpoint is None:
        print(f'Training model from scratch')
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    else:
        print(f'Training from checkpoint {train_from_checkpoint}')
        model = PPO.load(train_from_checkpoint, env, print_system_info=True)
    model.learn(total_timesteps=100_000)
    model.save(save_model_name)
    print(f'Saved model to {save_model_name}')
    
    
def eval(saved_model_name=ORACLE_NAME):
    model = PPO.load(saved_model_name)
    env = make_ll_env(render_mode='human')
    n_eval_episodes = 5
    
    # episode is considered a "solution" if it scores >= 200
    rewards, ep_lens = evaluate_policy(
        model.policy, 
        env, 
        n_eval_episodes=n_eval_episodes, 
        deterministic=True,
        render=False,
        return_episode_rewards=True
    )
    print(f'Evaluated model from {saved_model_name} for {n_eval_episodes} episodes\n\tMean reward = {sum(rewards) / len(rewards)} | ep lens reward {sum(ep_lens) / len(rewards)}')
    
    obs, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    

    
    
if __name__ == "__main__":
    # train('ppo_lunarlander_100k', None)
    eval('ppo_lunarlander_100k')
