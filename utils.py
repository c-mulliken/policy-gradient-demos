import numpy as np
import torch
from torch.distributions import Categorical

from model import Policy

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

def kl_old_new(
    policy: torch.nn.Module,
    old_policy: torch.nn.Module,
    states: torch.Tensor
) -> torch.Tensor:
    """
    Compute mean KL divergence D_KL[ pi_old(.|s) || pi_new(.|s) ] over a batch of states.
    """
    # New and old logits
    logits_new = policy(states)
    with torch.no_grad():
        logits_old = old_policy(states)

    dist_new = Categorical(logits=logits_new)
    dist_old = Categorical(logits=logits_old)

    return torch.distributions.kl_divergence(dist_old, dist_new).mean()


def fisher_vector_product(
    policy: torch.nn.Module,
    old_policy: torch.nn.Module,
    states: torch.Tensor,
    vector: torch.Tensor,
    damping: float = 1e-2
) -> torch.Tensor:
    """
    Compute (F + damping*I) @ vector, where F is the Fisher Information Matrix
    approximated by the Hessian of KL[pi_old || pi_new].
    """
    kl = kl_old_new(policy, old_policy, states)

    # First gradient of KL
    grad_kl = flat_grad(kl, policy)
    # directional derivative: grad_kl^T vector
    kl_v = torch.dot(grad_kl, vector)
    # second gradient (Hessian-vector product)
    grad2 = torch.autograd.grad(kl_v, policy.parameters(), retain_graph=True)
    fisher_v = torch.cat([g.contiguous().view(-1) for g in grad2])

    return fisher_v + damping * vector


def conjugate_gradient(
    hvp_fn,
    b: torch.Tensor,
    cg_iters: int = 10,
    residual_tol: float = 1e-10
) -> torch.Tensor:
    """
    Solve H x = b approximately using the conjugate gradient method,
    where hvp_fn(v) returns H @ v.
    Returns x.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = r.dot(r)

    for i in range(cg_iters):
        Ap = hvp_fn(p)
        alpha = rsold / (p.dot(Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
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
