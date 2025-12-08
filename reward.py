import math
from random import sample
import gymnasium as gym
import numpy as np
import torch
from utils import RunningStat

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
            print(f'seq_obs shape: {seq_obs.shape}')
            print(f'seq_obs.unsqueeze(0) shape: {seq_obs.unsqueeze(0).shape}')
            
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
    
    def get_synthetic_feedback(self, k : int) -> None:
        for i in range(k):
            # generate 2 random indices
            segments = sorted(sample(range(len(self.temp_experience['seq_obs'])), 2))
            
            # retrive correspoinding trajectory segments from the 2 indices above
            seq_obs_1, seq_obs_2 = self.temp_experience['seq_obs'][segments[0]], self.temp_experience['seq_obs'][segments[1]]
            seq_actions_1, seq_actions_2 = self.temp_experience['seq_actions'][segments[0]], self.temp_experience['seq_actions'][segments[1]]
            true_rewards_1, true_rewards_2 = self.temp_experience['true_reward'][segments[0]], self.temp_experience['true_reward'][segments[1]]
            
            true_rewards_1, true_rewards_2 = np.array(true_rewards_1), np.array(true_rewards_2)

            true_r_1_sum, true_r_2_sum = true_rewards_1.sum(), true_rewards_2.sum()
            
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
                                  gamma=torch.Tensor(gamma))
            

class RewardPredictorNet(torch.nn.Module):
    def __init__(self, 
                 obs_shape : tuple, 
                 action_shape : tuple, 
                 seq_len : int,
                 device : str = 'cuda',
                 lr = 5e-4,
                 weight_decay = 1e-4,
                 kappa = 10
                )-> None:
        super().__init__()
        
        # FIXME discrete actions are one-hot in gym---the +4 right now for lunar lander is hardcoded
        self.input_flattened_shape = (math.prod(obs_shape) + 4) * seq_len
        # self.output_flattened_shape = math.prod(action_shape)
        # print(f'input_flattened_shape: {self.input_flattened_shape}')
        # print(f'output_flattened_shape: {self.output_flattened_shape}')
        
        # hidden layers and activations from Appendix A.1 of Christiano et al 2017
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_flattened_shape, 64),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(64, 64), 
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(64, 1) # scalar reward dum dum
        )
        
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.kappa = kappa
        
        print(f'RewardPredictorNet initialized!')
        print(f'{self.model}')
    
    def forward(self, seq_obs: torch.Tensor, seq_actions : torch.Tensor):
        seq_obs = seq_obs.to(self.device)
        seq_actions = seq_actions.to(self.device)
        
        # flatten input obs
        seq_obs_flat = torch.flatten(seq_obs)
        print(f'seq_obs_flat shape: {seq_obs_flat.shape}')
        seq_actions_flat = torch.flatten(seq_actions)
        print(f'seq_actions_flat shape: {seq_actions_flat.shape}')
        
        print(f'torch.cat(...) shape: {torch.cat((seq_obs_flat, seq_actions_flat)).shape}')
        
        return self.model(torch.cat((seq_obs_flat, seq_actions_flat)))
    
    def train_step(self, dataset) -> float:
        '''
        `data` has data samples (traj1, traj2, mu)
        '''
        self.optimizer = torch.optim.Adam(self.parameters(), 
                                          lr=self.lr, 
                                          weight_decay=self.weight_decay,
                                          )
        
        self.optimizer.zero_grad()
        run_loss = 0
        
        labels = [] # ground truth
        preds = []
        
        for data in dataset:
            (o1, a1), (o2, a2), gamma = data # shapes o1=o2=(seg_len, obs_shape), a1=a2=(seg_len, action_shape), gamma=(seg_len,)
            
            o1 = o1.to(self.device)
            a1 = a1.to(self.device)
            o2 = o2.to(self.device)
            a2 = a2.to(self.device)
            gamma = gamma.to(self.device)
            
            r1 = self.forward(o1, a1)
            r2 = self.forward(o2, a2)
            
            labels.append(gamma)
            preds.append(torch.sigmoid(r2 - r1))
        
        # compute beta log likelihood loss
        loss = self._beta_nll_loss(torch.tensor(labels), torch.tensor(preds), self.kappa)

        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def _beta_log_likelihood(self, gamma, p, kappa):
        """
        Compute log probability of gamma under Beta(kappa*p, kappa*(1-p))
        
        Args:
            gamma: observed ratings in [0,1], shape (batch_size,)
            p: predicted probabilities in [0,1], shape (batch_size,)
            kappa: concentration parameter, scalar or shape (batch_size,)
        
        Returns:
            log_prob: log probability for each sample, shape (batch_size,)
        """
        # Clip to avoid numerical issues
        gamma = torch.clamp(gamma, 1e-7, 1 - 1e-7)
        p = torch.clamp(p, 1e-7, 1 - 1e-7)
        
        # Beta parameters
        alpha = kappa * p
        beta = kappa * (1 - p)
        
        # Log probability using lgamma (log of gamma function)
        log_prob = (
            torch.lgamma(kappa) 
            - torch.lgamma(alpha) 
            - torch.lgamma(beta)
            + (alpha - 1) * torch.log(gamma)
            + (beta - 1) * torch.log(1 - gamma)
        )
        
        return log_prob


    def _beta_nll_loss(self, gamma, p, kappa):
        """
        Negative log-likelihood loss for Beta distribution
        
        Args:
            gamma: observed ratings, shape (batch_size,)
            p: predicted probabilities, shape (batch_size,)
            kappa: concentration parameter
        
        Returns:
            loss: scalar negative log-likelihood
        """
        log_prob = self._beta_log_likelihood(gamma, p, kappa)
        return -log_prob.mean()