Variables
1. Total PPO training steps (n_steps)
2. Trajectory segment sampling frequency (this is based on n_steps)
3. Length of a trajectory segment (how many (s,a) pairs per trajectory)

Constants:
- Number of trajectory segments sampled (# of new samples every n_steps = 1% of n_steps)
- Train on random 80% of trajectories sampled thus far for 50 epochs

RQ1: How sensitive is the learned reward function to training length?
- training length: 200k vs 300k steps
- sample 20 new trajectory snippets of length 25 every n_steps (2048 default)

RQ2: How does the frequency of data sampling impact the learned policy?
- Motivation: don't want to change the reward too fast --> then the policy is chasing a moving target. This is especially dangerous because PPO is actor critic and that already has the problem of the value function/estimate chasing a moving policy

RQ3: How does the length of the trajectory segments impact the learned policy?
- 