from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

from model import Policy
from utils import compute_returns, surrogate_loss, flat_grad, fisher_vector_product, conjugate_gradient, flat_params, set_flat_params
from torch.distributions import Categorical
from viz import viz_episode

def train_one_epoch(
    policy: Policy,
    old_policy: Policy,
    env: gym.Env,
    batch_size: int = 5000,
    gamma: float = 0.99,
    method: str = 'reinforce',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = False,
    max_kl: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Train the policy for one epoch using the specified method.

    Args:
        policy (Policy): the current policy to be updated
        old_policy (Policy): the policy before the update for KL computations
        env (gym.Env): the environment instance
        batch_size (int): number of timesteps to collect
        gamma (float): discount factor for returns
        method (str): training method, e.g. "naive"
        device (str): computation device, e.g. "cuda"

    Returns:
        Tuple of surrogate loss, mean KL divergence, and list of episode durations
    """

    states = []
    actions = []
    old_log_probs = []
    rewards = []
    durations = []
    total_timesteps = 0

    while total_timesteps < batch_size:
        print(f"[DEBUG] Starting new episode. Total timesteps so far: {total_timesteps}") if verbose else None
        state, _ = env.reset()
        done = False
        episode_rewards = []
        duration = 0

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

            if not done:
                duration += 1

        print(f"[DEBUG] Episode finished. Length: {duration}, Rewards: {sum(episode_rewards)}") if verbose else None
        rewards.append(episode_rewards)
        durations.append(duration)

    all_returns = []
    for r in rewards:
        all_returns.extend(compute_returns(r, gamma))

    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)
    advantages = returns_tensor - returns_tensor.mean()

    print(f"[DEBUG] Calculated returns and advantages. Mean advantage: {advantages.mean().item():.4f}") if verbose else None

    surr_loss, mean_kl = surrogate_loss(
        policy,
        old_policy,
        old_log_probs,
        states,
        actions,
        advantages
    )

    print(f"[DEBUG] Surrogate loss: {surr_loss.item():.6f}, Mean KL: {mean_kl.item():.6f}") if verbose else None

    if method == "reinforce":
        print("[DEBUG] Using naive policy gradient update.") if verbose else None
        optimizer = optim.Adam(policy.parameters(), lr=1e-3)
        optimizer.zero_grad()
        (-surr_loss).backward()
        optimizer.step()
    elif method == "trpo":
        print("[DEBUG] Using TRPO update.")
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(device)
        advantages_tensor = advantages.to(device)

        surr_loss_trpo, _ = surrogate_loss(
            policy, old_policy, old_log_probs, states, actions, advantages, device
        )
        policy.zero_grad()
        grad = flat_grad(surr_loss_trpo, policy).detach()
        print(f"[DEBUG] Policy gradient norm: {grad.norm().item():.6f}") if verbose else None

        def hvp_fn(v):
            return fisher_vector_product(policy, old_policy, states_tensor, v)

        step_dir = conjugate_gradient(hvp_fn, grad)
        print(f"[DEBUG] Step direction norm: {step_dir.norm().item():.6f}") if verbose else None

        shs = 0.5 * step_dir.dot(hvp_fn(step_dir))
        step_size = torch.sqrt(max_kl / (shs + 1e-8))
        full_step = step_size * step_dir
        print(f"[DEBUG] Step size: {step_size.item():.6f}") if verbose else None

        old_params = flat_params(policy)
        def set_and_eval(step):
            set_flat_params(policy, old_params + step)
            surr, kl = surrogate_loss(
                policy, old_policy, old_log_probs, states, actions, advantages, device
            )
            return surr, kl

        for frac in [0.5 ** i for i in range(10)]:
            step = frac * full_step
            surr, kl = set_and_eval(step)
            print(f"[DEBUG] Line search frac: {frac:.5f}, KL: {kl.item():.6f}, Surrogate: {surr.item():.6f}") if verbose else None
            if kl.item() <= max_kl and surr.item() > surr_loss_trpo.item():
                break
        else:
            print("[DEBUG] Line search failed. Reverting policy parameters.") if verbose else None
            set_flat_params(policy, old_params)
    else:
        raise ValueError(f"Unknown method: {method}")

    old_policy.load_state_dict(policy.state_dict())

    return surr_loss, mean_kl, durations

def main_train(
    method: str = "reinforce",
    iterations: int = 250,
    batch_size: int = 1000,
    gamma: float = 0.99,
    device: str = None,
    verbose: bool = False,
    plot_result: bool = False,
    save_model: bool = False,
    max_kl: float = 0.01,
    hidden_neurons: int = 128,
    env_name: str = "CartPole-v1"
) -> Tuple[Policy, Dict]:
    """
    Train a policy over multiple epochs and return the trained policy and episode durations.

    Args:
        method (str): training method, e.g. "naive"
        iterations (int): number of epochs to train
        batch_size (int): timesteps per epoch
        gamma (float): discount factor
        device (str): computation device; auto-detected if None

    Returns:
        The trained policy and a list of duration lists for each epoch
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    result_dict = {}

    policy = Policy(state_dim, action_dim, hidden_neurons=hidden_neurons).to(device)
    old_policy = Policy(state_dim, action_dim, hidden_neurons=hidden_neurons).to(device)
    old_policy.load_state_dict(policy.state_dict())

    total_durations: List[List[int]] = []

    for iteration in range(iterations):
        surr_loss, mean_kl, durations = train_one_epoch(
            policy,
            old_policy,
            env,
            batch_size,
            gamma,
            method,
            device,
            verbose,
            max_kl=max_kl
        )
        print(f"[{method}] | Iteration {iteration}: surrogate loss={surr_loss.item():.4f}, mean KL={mean_kl.item():.6f}")
        print(f"Average duration: {np.mean(durations):.2f}")
        total_durations.append(durations)

    iteration_durations = [np.mean(np.array(durations)) for durations in total_durations]
    result_dict['iteration_durations'] = iteration_durations

    if plot_result:
        plt.plot(iteration_durations) 
        plt.xlabel("Iteration")
        plt.ylabel("Average Duration")
        plt.title("Average Duration per Iteration")
        plt.savefig(f"visualisations/plots/average_duration_{method}.png")

    env.close()
    torch.save(policy.state_dict(), f"policies/{env_name}_policy_{method}.pth") if save_model else None

    return policy, result_dict
