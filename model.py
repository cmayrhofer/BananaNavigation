import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # the first fully connected layer
        self.fc1 = nn.Linear(state_size, fc1_units)
        # the second fully connected layer
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # the third fully connected layer
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values.
           
           The network consists of three fully connected linear layers where between every two layers
           we have a rectifier activation function. The output of the network is the action-value for
           all actions for the given input state.
        """
        
        x = F.relu(self.fc1(state)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)
