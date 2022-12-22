"""Model architecture highlighted at https://towardsdatascience.com/deep-q-network-with-pytorch-146bfa939dfe ."""
import torch
import torch.nn as nn   
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd 
import os
import numpy as np
import random

# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Conv_QNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        # conv layers
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(*self.input_shape))).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


