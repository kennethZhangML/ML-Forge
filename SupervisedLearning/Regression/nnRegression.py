import numpy as np 
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(self, MLP).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias = True)
        self.fc1 = nn.Linear(hidden_dim, output_dim, bias = True)
    
    def forward(self, x):
        x = torch.nn.ReLU(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.LogSoftmax(x)
    