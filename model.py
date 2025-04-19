import torch
import torch.nn as nn
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x
    
    def get_action(self, state, device='cuda'):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def get_action_dist(self, state, device='cuda'):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        return dist
