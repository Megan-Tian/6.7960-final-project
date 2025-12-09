import math
from random import sample
import gymnasium as gym
import numpy as np
import torch
from utils import RunningStat
from stable_baselines3.common.callbacks import BaseCallback

class RewardPredictor():
    def __init__(self,
                 obs_shape : tuple,
                 action_shape : tuple,
                 seq_len : int,
                 n_predictors : int,
                #  n_action : int, # TODO idk abt this one
                 device : str = 'cuda')-> None:
        
        if torch.cuda.is_available():
            self.device = device
        else:
            self.device = 'cpu'
            if device == 'cuda': print("Defaulting to CPU as device, CUDA unavailable")
        
        self.seq_len = seq_len
        self.n_predictors = n_predictors
        self.predictors = [RewardPredictorNet(obs_shape, 
                                              action_shape, 
                                              seq_len, 
                                              device
                                              ).to(device) 
                           for _ in range(n_predictors)]
        
        self.dataset = []
        self.temp_experience = {"seq_obs" : [], "seq_actions" : [], "true_reward" : []}
        self.r_norm = RunningStat(shape=self.n_predictors)
        
        self.pct_to_sample = 0.8


    def predict(self, seq_obs : torch.Tensor, seq_actions : torch.Tensor) -> torch.Tensor:
        seq_obs = seq_obs.to(self.device)
        seq_actions = seq_actions.to(self.device)
        
        [model.eval() for model in self.predictors]
        
        with torch.no_grad():
            # print(f'seq_obs shape: {seq_obs.shape}')
            # print(f'seq_obs.unsqueeze(0) shape: {seq_obs.unsqueeze(0).shape}')
            
            # FIXME below is not batched?
            preds = torch.stack([p.forward(seq_obs, seq_actions) for p in self.predictors]).view(self.n_predictors, -1).cpu()
            preds = preds.transpose(0, 1)
        
            for p in preds:
                self.r_norm.push(p)
            
            preds -= self.r_norm.mean
            preds /= (self.r_norm.std + 1e-6)
            preds *= 1 # FIXME seems like a fudge scaling factor, og 0.05
            
            preds = preds.transpose(0, 1)
            pred_reward = torch.mean(preds, dim=0)
            
        return pred_reward[-1] # FIXME why the -1 indexing

    
    def train(self) -> float:
        [model.train() for model in self.predictors]
        dataset = []
        for data in self.dataset:
            dataset.append(data)
        
        k = round(len(dataset) * self.pct_to_sample)
        
        run_loss = 0
        
        for model in self.predictors:
            sampled_data = sample(dataset, k)
            run_loss += model.train_step(sampled_data)
            
        return run_loss / self.n_predictors
    
    
    def add_temp_experience(self, seq_obs, seq_actions, reward):
        self.temp_experience['seq_obs'].append(seq_obs)
        self.temp_experience['seq_actions'].append(seq_actions)
        self.temp_experience['true_reward'].append(reward)

    def reset_temp_experience(self):
        self.temp_experience['seq_obs'].clear()
        self.temp_experience['seq_actions'].clear()
        self.temp_experience['true_reward'].clear()
    
    def add_feedback(self, seg1: tuple[torch.Tensor, torch.Tensor], seg2: tuple[torch.Tensor, torch.Tensor], gamma: torch.Tensor):
        '''
        Args:
            seg1: tuple[torch.Tensor, torch.Tensor] --> (obs, actions)
            seg2: tuple[torch.Tensor, torch.Tensor] --> (obs, actions)
            gamma: torch.Tensor --> scalar (1,) in [0,1]
        '''
        gamma = torch.tensor(gamma, dtype=torch.float32)
        self.dataset.append((seg1, seg2, gamma))
    
    def get_synthetic_feedback(self, k : int):
        '''
        Args:        
            k: number of trajectories to add to `self.dataset` and generate synthetic 
            feedback for
        '''
        for i in range(k):
            # generate 2 random indices
            segments = sorted(sample(range(len(self.temp_experience['seq_obs'])), 2))
            
            # retrive correspoinding trajectory segments from the 2 indices above
            seq_obs_1, seq_obs_2 = self.temp_experience['seq_obs'][segments[0]], self.temp_experience['seq_obs'][segments[1]]
            seq_actions_1, seq_actions_2 = self.temp_experience['seq_actions'][segments[0]], self.temp_experience['seq_actions'][segments[1]]
            true_rewards_1, true_rewards_2 = self.temp_experience['true_reward'][segments[0]], self.temp_experience['true_reward'][segments[1]]

            true_rewards_1, true_rewards_2 = np.array(true_rewards_1), np.array(true_rewards_2)

            true_r_1_sum, true_r_2_sum = true_rewards_1.sum(), true_rewards_2.sum()
            
            # print(f'seq_obs_1 shape: {seq_obs_1.shape} | seq_obs_2 shape: {seq_obs_2.shape}')
            # print(f'seq_actions_1 shape: {seq_actions_1.shape} | seq_actions_2 shape: {seq_actions_2.shape}')
            # print(f'true_rewards_1 : {true_rewards_1} | true_rewards_2 : {true_rewards_2}')
            # raise KeyError

            # if both trajectory segments are empty / zero truereward, skip and pick new segments
            if true_r_1_sum + true_r_2_sum == 0.0:
                continue
            
            # both trajectories have same total true reward --> indicate no preference
            if true_r_1_sum == true_r_2_sum:
                gamma = 0.5
            # traj seg 2 better than 1
            elif true_r_2_sum > true_r_1_sum:
                gamma = (true_r_2_sum) / (true_r_1_sum + true_r_2_sum)
            # traj seg 1 better than 2
            else: 
                gamma = (true_r_1_sum) / (true_r_1_sum + true_r_2_sum)
            
            self.add_feedback(seg1=(seq_obs_1, seq_actions_1), 
                                  seg2=(seq_obs_2, seq_actions_2), 
                                  gamma=torch.tensor([gamma]))
            

class RewardPredictorNet(torch.nn.Module):
    def __init__(self, obs_shape, action_shape, seq_len, device='cuda', lr=5e-4, weight_decay=1e-4, kappa=10):
        super().__init__()
        self.input_flattened_shape = (math.prod(obs_shape) + 4) * seq_len
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_flattened_shape, 64),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(64, 64), 
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(64, seq_len)
        )
        
        self.device = device
        self.kappa = kappa
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, seq_obs: torch.Tensor, seq_actions: torch.Tensor):
        # add batching
        if seq_obs.dim() == 2: 
            seq_obs = seq_obs.unsqueeze(0)
            seq_actions = seq_actions.unsqueeze(0)

        # flatten starting from dimension 1 (keeping batch dimension 0)
        batch_size = seq_obs.shape[0]
        seq_obs_flat = seq_obs.reshape(batch_size, -1)
        seq_actions_flat = seq_actions.reshape(batch_size, -1)
        
        x = torch.cat((seq_obs_flat, seq_actions_flat), dim=1)
        return self.model(x)
    
    def train_step(self, dataset) -> float:
        self.optimizer.zero_grad()
        
        # vectorize training, data loading needs to be done by separating data into lists
        o1_list, a1_list, o2_list, a2_list, gamma_list = [], [], [], [], []
        
        for data in dataset:
            (o1, a1), (o2, a2), gamma = data
            o1_list.append(o1)
            a1_list.append(a1)
            o2_list.append(o2)
            a2_list.append(a2)
            gamma_list.append(gamma)
        
        # stack into tensors for batch processing
        o1_batch = torch.stack(o1_list).to(self.device)
        a1_batch = torch.stack(a1_list).to(self.device)
        o2_batch = torch.stack(o2_list).to(self.device)
        a2_batch = torch.stack(a2_list).to(self.device)
        
        # ensure gamma is the right shape (Batch_Size,)
        gamma_batch = torch.tensor(gamma_list, requires_grad=True, dtype=torch.float32).to(self.device).view(-1)

        # Single forward pass for the whole batch
        r1 = self.forward(o1_batch, a1_batch) # Shape: (Batch, Seq_Len)
        r2 = self.forward(o2_batch, a2_batch)
        
        # sum rewards across sequence dimension (dim 1) BEFORE sigmoid
        # r1_sum shape: (Batch,)
        r1_sum = r1.sum(dim=1)
        r2_sum = r2.sum(dim=1)
        
        # predict preference from trajectory segment rewards r1_sum and r2_sum
        preds = torch.tensor(torch.sigmoid(r2_sum - r1_sum), requires_grad=True).to(self.device)
        
        # self.kappa.to(self.device)
        loss = self._beta_nll_loss(gamma_batch, preds, self.kappa)
        loss.backward()
        self.optimizer.step()
        
        return loss

    def _beta_nll_loss(self, gamma, p, kappa):
        """
        Negative log-likelihood loss for Beta distribution
        
        Args:
            gamma: observed ratings in [0,1], shape (batch_size,)
            p: predicted probabilities in [0,1], shape (batch_size,)
            kappa: concentration parameter, scalar or shape (batch_size,)
        
        Returns:
            loss: scalar negative log-likelihood
        """
        kappa = torch.tensor([kappa]).to(self.device)
        assert (torch.all((gamma >= 0) & (gamma <= 1)), 
                f"ERROR observed ratings gamma must be in [0,1], gamma = {gamma}")
        assert (torch.all((p >= 0) & (p <= 1)), 
                f"ERROR predicted probabilities p must be in [0,1], p = {p}")

        # Clip to avoid numerical issues
        gamma = torch.clamp(gamma, 1e-7, 1 - 1e-7).to(self.device)
        p = torch.clamp(p, 1e-7, 1 - 1e-7).to(self.device)
        
        # Beta parameters
        alpha = torch.tensor(kappa * p).to(self.device)
        beta = torch.tensor(kappa * (1 - p)).to(self.device)
 
        # Log probability using lgamma (log of gamma function)
        log_prob = (
            torch.lgamma(kappa) # log(kappa)
            - torch.lgamma(alpha) 
            - torch.lgamma(beta)
            + (alpha - 1) * torch.log(gamma)
            + (beta - 1) * torch.log(1 - gamma)
        )
        
        return -log_prob.mean()


    


    
class TrainRewardPredictorCallback(BaseCallback):
    def __init__(self, rp: RewardPredictor, verbose=0):
        super(TrainRewardPredictorCallback, self).__init__(verbose)
        self.reward_predictor = rp
        self.true_rewards = []
        self.pred_rewards = []
        self.accumulated_true_reward = 0

    def _on_step(self) -> bool:
        true_reward = self.locals.get("infos", None)[-1].get("true_reward", None)
        pred_reward = self.locals.get("infos", None)[-1].get("pred_reward", None)
        self.true_rewards.append(true_reward)
        self.pred_rewards.append(pred_reward)
        self.accumulated_true_reward += true_reward
        
        return True

    def _on_rollout_end(self) -> None:
        ep_true_reward = sum(self.true_rewards)
        self.logger.record("custom/true_reward", ep_true_reward)
        self.logger.record_mean("custom/true_reward_mean", ep_true_reward)
        self.logger.record("custom/accumaltive_true_reward", self.accumulated_true_reward)

        if self.reward_predictor:
            ep_pred_reward = sum(self.pred_rewards)
            self.logger.record("custom/pred_reward", ep_pred_reward)
            
            # get synthetic feedback
            k = int(len(self.reward_predictor.temp_experience['seq_obs']) * 0.01) # 1%
            k = k if k > 1 else 2
            self.reward_predictor.get_synthetic_feedback(k)

            # train if we were able to successfully collect trajectory samples
            if len(self.reward_predictor.dataset) > 1:
                print(f'Tried to add {k} samples, dataset size = {len(self.reward_predictor.dataset)}')
                for i in range(50):
                    loss = self.reward_predictor.train()
                
                print(f"Reward Predict loss = {loss}")
                self.logger.record("custom/reward_predictor_loss", loss.item())
                    
            self.reward_predictor.reset_temp_experience()

            self.logger.record("custom/dataset_len", len(self.reward_predictor.dataset))
            
        self.true_rewards.clear()
        self.pred_rewards.clear()