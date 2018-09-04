import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, cv1_out=32, cv2_out=64, cv3_out=64, fc1_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int^4): Dimensions of each state, i.e. a duple of the number of pixels 
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # 3 input image channels (rgb), cv1_out output channels/feature maps
        # 8x8 square convolution kernel
        # with stride 4
        ## output size = (W-F)/S +1 = 
        # the output Tensor for one image, will have the dimensions: 
        self.conv1 = nn.Conv2d(3, cv1_out, 8, stride=4)        
        # dropout with p=0.4
  #      self.conv1_drop = nn.Dropout(p=0.4)
        
        self.conv2 = nn.Conv2d(cv1_out, cv2_out, 4, stride=2)
        # dropout with p=0.4
  #      self.conv2_drop = nn.Dropout(p=0.4)
        
        self.conv3 = nn.Conv2d(cv2_out, cv3_out, 3)
        # dropout with p=0.4
  #      self.conv3_drop = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(7*7*64, fc1_units)
        # dropout with p=0.4
  #      self.fc1_drop = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(state))
   #     x = self.conv1_drop(x)
        x = F.relu(self.conv2(x))
   #     x = self.conv2_drop(x)
        x = F.relu(self.conv3(x))
   #     x = self.conv3_drop(x)
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
   #     x = self.fc1_drop(x)
        return self.fc2(x)
