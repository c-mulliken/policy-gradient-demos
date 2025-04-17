import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import gym

from model import Policy
from utils import compute_returns, surrogate_loss
from viz import viz_episode

print("Starting TRPO training...")

# hyperparameters
gamma = 0.99
delta = 0.01
iterations = 200
batch_size = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = Policy(state_dim, action_dim).to(device)
    old_policy = Policy(state_dim, action_dim).to(device)
    old_policy.load_state_dict(policy.state_dict())

    for iteration in range(iterations):
        states = []
        actions = []
        rewards = []
        old_log_probs = []
        total_timesteps = 0
        
        # Collect a batch of trajectories
        while total_timesteps < batch_size:
            print(f"Collecting batch {total_timesteps} / {batch_size}")
            state, _ = env.reset()
            episode_rewards = []
            done = False
            while not done:
                action, log_prob = policy.get_action(state, device)
                states.append(state)
                actions.append(action)
                old_log_probs.append(log_prob.item())
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_rewards.append(reward)
                total_timesteps += 1
                
                done = terminated or truncated
                state = next_state
            rewards.append(episode_rewards)
        
        # Flatten the rewards for advantage calculation.
        # Here we use the simple "return" as the advantage
        # For a more robust implementation, use a baseline (critic).
        all_returns = []
        for r in rewards:
            all_returns.extend(compute_returns(r, gamma))
        returns_tensor = torch.FloatTensor(all_returns)
        advantages = returns_tensor - returns_tensor.mean()  # simple normalization
        
        print("Computing loss...")
        # Compute the surrogate loss and KL divergence for diagnostics.
        surr_loss, mean_kl = surrogate_loss(policy, old_policy, old_log_probs, states, actions, advantages)
        
        # Print diagnostics
        print(f"Iteration {iteration}: surrogate loss = {surr_loss.item():.4f}, mean KL = {mean_kl.item():.6f}")
        
        # ******** TRPO update placeholder *********
        #
        # Here you would:
        # 1. Compute the gradient of the surrogate loss with respect to policy parameters.
        # 2. Use the conjugate gradient algorithm to solve Hx = g,
        #    where H is the Fisher information matrix (approximated via KL divergence second derivative)
        #    and g is the gradient of the loss.
        # 3. Compute the step size scaling factor: beta = sqrt(2 * delta / (s^T H s)).
        # 4. Perform a backtracking line search to ensure the updated policy
        #    improves the surrogate loss and the KL constraint (mean KL <= delta).
        #
        # For this starter code, we simply perform a standard gradient ascent step.
        #
        optimizer = optim.Adam(policy.parameters(), lr=1e-3)
        optimizer.zero_grad()
        # Note: maximizing the surrogate is equivalent to minimizing the negative surrogate.
        (-surr_loss).backward()
        optimizer.step()
        #
        # Update the old policy every iteration.
        old_policy.load_state_dict(policy.state_dict())
    
    env.close()
    return policy

policy = train()

# save policy
policy_path = "policies/trpo_cartpole_policy.pth"
torch.save(policy.state_dict(), policy_path)
print(f"Policy saved to {policy_path}")

for i in range(5):
    viz_episode(policy,
                env_name="CartPole-v1",
                filename=f"cartpole_play_{i}.gif",
                fps=30,
                mode="gif",
                device=device)
