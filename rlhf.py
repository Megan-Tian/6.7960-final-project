import numpy as np
import torch as th

import stable_baselines3
from stable_baselines3 import PPO
from oracle import make_ll_env
from reward import TrainRewardPredictorCallback, RewardPredictor
from env import RLHFEnv
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym

def get_device():
    return "cuda" if th.cuda.is_available() else "cpu"

def load_run(model_name):
    model = PPO.load(f'{model_name}.zip')
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
    print(env.reward_predictor.predictors[0])
    obs, info = env.reset()
    terminated = truncated = False
    # # cell and hidden state of the LSTM
    # lstm_states = None
    # num_envs = 1
    # # Episode start signals are used to reset the lstm states
    # episode_starts = np.ones((num_envs,), dtype=bool)
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        # episode_starts = dones
        env.render()
        
    rewards, lens = evaluate_policy(model.policy, 
                                    env, 
                                    n_eval_episodes=10, 
                                    deterministic=True, 
                                    render=True, 
                                    return_episode_rewards=True
                                )
    print(f'Evaluated model from {model_name} for 10 episodes\n\tMean reward = {np.mean(rewards)} | Std reward {np.std(rewards)} | mean len {np.mean(lens)}')

def rp_train(model_name):    
    env = make_ll_env()
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    reward_model = RewardPredictor(obs_shape, 
                                  action_shape, 
                                  seq_len=10,
                                  n_predictors=10,
                                  )
    env = RLHFEnv(env, 
                  reward_model,
                  seq_len=10)
    
    device = get_device()
    
    # env = EnvWrapper(env=env, reward_predictor=reward_model, seq_len=30)

    # Define the custom policy with normalized_images set to False
    # policy_kwargs = dict(
    #     normalize_images=False
    # )

    model = PPO('MlpPolicy', env,
                         n_steps=512,
                         batch_size=64,
                         verbose=2,
                         learning_rate=2e-5, tensorboard_log="reward_pred_runs/rlfhp")
    #print(model.policy)

    callback = TrainRewardPredictorCallback(reward_model)
    
    try:
        model.learn(100_000, callback=callback)
    except KeyboardInterrupt as e:
        print(e)

    model.save(model_name)

# def normal_train(model_name):
#     env = gym.make("ALE/Enduro-v5", obs_type="grayscale", full_action_space=True)
    
#     env = EnvWrapper(env=env, reward_predictor=None, seq_len=30)

#     # Define the custom policy with normalized_images set to False
#     policy_kwargs = dict(
#         normalize_images=False
#     )

#     model = RecurrentPPO(CnnLstmPolicy, env, policy_kwargs=policy_kwargs,
#                          n_steps=512,
#                          batch_size=64,
#                          verbose=2,
#                          learning_rate=2e-5, tensorboard_log="reward_pred_runs/conventional_train/")
#     #print(model.policy)

    
#     callback = TrainRewardPredictorCallback(rp=None)
#     try:
#         model.learn(30_000, callback=callback)
#     except KeyboardInterrupt as e:
#         print(e)

#     model.save(model_name)

def main():
    rp_train("model_tmp_2")
    # normal_train("normal_model")
    load_run("model_tmp_2")
    

if __name__ == "__main__":
    main()