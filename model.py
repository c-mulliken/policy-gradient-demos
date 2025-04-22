import torch
import torch.nn as nn
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_neurons: int = 128):
        """
        Initialize the policy network.
        Args:
            state_dim (int): dimension of the state space
            action_dim (int): dimension of the action space
            hidden_neurons (int): number of neurons in the hidden layer
        """

        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x
    
    def get_action(self, state, device='cuda'):
        """
        Get an action from the policy given a state.
        Args:
            state (np.ndarray or torch.Tensor): the current state
            device (str): computation device; auto-detected if None
        Returns:
            action (int): the action to take
            log_prob (torch.Tensor): the log probability of the action
        """

        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def get_action_dist(self, state, device='cuda'):
        """
        Get the action distribution from the policy given a state.
        Args:
            state (np.ndarray or torch.Tensor): the current state
            device (str): computation device; auto-detected if None
        Returns:
            dist (torch.distributions.Categorical): the action distribution
        """
        
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        return dist
