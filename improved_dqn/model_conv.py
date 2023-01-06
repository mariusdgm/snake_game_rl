"""Model architecture highlighted at https://towardsdatascience.com/deep-q-network-with-pytorch-146bfa939dfe ."""
import torch
import torch.nn as nn   
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd 
import os
import numpy as np
import random

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
        return self.features(autograd.torch.zeros(*self.input_shape)).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.num_actions = output_size

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.num_actions)

    def forward(self, x):
        """Called when calling model(x)"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


