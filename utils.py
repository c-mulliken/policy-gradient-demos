import numpy as np
import torch
from torch.distributions import Categorical

def flat_grad(y, model):
    grads = torch.autograd.grad(y, model.parameters(), create_graph=True)
    return torch.cat([g.contiguous().view(-1) for g in grads])

def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_vector):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_vector[idx:idx + numel].view(p.size()))
        idx += numel

def fisher_vector_product(policy, old_policy, states, vector, damping=1e-2):
    p = policy(states)
    with torch.no_grad():
        p_old = old_policy(states)
    dist = Categorical(p)
    dist_old = Categorical(p_old)
    kl = torch.distributions.kl_divergence(dist_old, dist).mean()
    grad_kl = flat_grad(kl, policy)
    kl_v = (grad_kl * vector).sum()
    grad2 = torch.autograd.grad(kl_v, policy.parameters(), retain_graph=True)
    flat_grad2 = torch.cat([g.contiguous().view(-1) for g in grad2])
    return flat_grad2 + damping * vector

def conjugate_gradient(hvp_fn, b, cg_iters=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = r.dot(r)
    for _ in range(cg_iters):
        Ap = hvp_fn(p)
        alpha = rsold / (p.dot(Ap) + 1e-12)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.dot(r)
        if rsnew < residual_tol:
            break
        p = r + (rsnew/rsold) * p
        rsold = rsnew
    return x

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def surrogate_loss(policy, old_policy, old_log_probs, states, actions, advantages, device='cuda' if torch.cuda.is_available() else 'cpu'):
    states_tensor = torch.FloatTensor(np.array(states)).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    new_probs = policy(states_tensor)
    dist = Categorical(new_probs)
    new_log_probs = dist.log_prob(actions_tensor)
    ratio = torch.exp(new_log_probs - torch.FloatTensor(old_log_probs).to(device))
    surrogate = torch.mean(ratio * advantages.to(device))
    with torch.no_grad():
        old_probs = old_policy(states_tensor)
    old_dist = Categorical(old_probs)
    kl_div = torch.distributions.kl_divergence(old_dist, dist)
    return surrogate, torch.mean(kl_div)

def ema(data, alpha=0.2):
    ema_data = []
    v = data[0]
    for x in data:
        v = alpha * x + (1 - alpha) * v
        ema_data.append(v)
    return np.array(ema_data)
