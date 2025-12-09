import numpy as np
import torch as th

import stable_baselines3
from stable_baselines3 import PPO
from oracle import make_ll_env
from reward import TrainRewardPredictorCallback, RewardPredictor
from env import RLHFEnv
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym


SEQ_LEN = 25 # Christiano et al 2017 appendix A.2 use 25 steps for Atari clips
N_PREDICTORS = 3 # also from Christiano et al

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
                                  seq_len=SEQ_LEN,
                                  n_predictors=N_PREDICTORS,
                                  ),
                  seq_len=SEQ_LEN)
    # print(env.reward_predictor.predictors[0])
    obs, info = env.reset()
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        
    rewards, lens = evaluate_policy(model.policy, 
                                    env, 
                                    n_eval_episodes=30, 
                                    deterministic=True, 
                                    # render=True, 
                                    return_episode_rewards=True
                                )
    print(f'Evaluated model from {model_name} for 30 episodes\n\tMean reward = {np.mean(rewards)} | Std reward {np.std(rewards)} | mean len {np.mean(lens)}')

def rp_train(model_name, is_feedback_continuous, kappa):
    device = get_device()  
    env = make_ll_env()
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    

    reward_model = RewardPredictor(obs_shape, 
                                  action_shape, 
                                  seq_len=SEQ_LEN,
                                  n_predictors=N_PREDICTORS,
                                  device=device,
                                  kappa=kappa
                                  )
    env = RLHFEnv(env, 
                  reward_model,
                  seq_len=SEQ_LEN)
    

    model = PPO('MlpPolicy', env,
                         n_steps=2048, # default 2048
                         batch_size=64, # default 64
                         verbose=2,
                        #  learning_rate=2e-5, 
                         tensorboard_log="reward_pred_runs/rlfhp")
    #print(model.policy)

    callback = TrainRewardPredictorCallback(reward_model, is_feedback_continuous)
    
    try:
        model.learn(200_000, callback=callback)
    except KeyboardInterrupt as e:
        print(e)

    model.save(model_name)

def main():
    is_feedback_continuous = True
    kappa = 5
    rp_train(f"model_fc={is_feedback_continuous}_kappa={kappa}", 
             is_feedback_continuous,
             kappa)
    load_run(f"model_fc={is_feedback_continuous}_kappa={kappa}")
    # load_run('results/rq2/continuous_nsteps_1024/model_fc=True')
    # load_run('results/continuous_ 200k/model_tmp_fc=True_200k')
    # load_run('results/rq2/continuous_nsteps_4096/model_fc=True')
    
    

if __name__ == "__main__":
    main()