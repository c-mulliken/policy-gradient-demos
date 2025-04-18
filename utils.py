import numpy as np
import torch
from torch.distributions import Categorical

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def surrogate_loss(
        policy,
        old_policy,
        old_log_probs,
        states,
        actions,
        advantages,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
    states_tensor = torch.FloatTensor(np.array(states)).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    
    new_probs = policy(states_tensor)
    dist = Categorical(new_probs)
    new_log_probs = dist.log_prob(actions_tensor)
    
    ratio = torch.exp(new_log_probs - torch.FloatTensor(old_log_probs).to(device)).to(device)
    surrogate = torch.mean(ratio * advantages.to(device))
    
    with torch.no_grad():
        old_probs = old_policy(states_tensor)
    old_dist = Categorical(old_probs)
    
    kl_div = torch.distributions.kl_divergence(old_dist, dist)
    return surrogate, torch.mean(kl_div)
