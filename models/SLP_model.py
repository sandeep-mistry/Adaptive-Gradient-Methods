import torch
import torch.nn as nn
import torch.nn.functional as F

class SLP(nn.Module):
    def __init__(self):
      super(SLP, self).__init__()
      self.fc1 = nn.Linear(28*28, 10)
    def forward(self, x):
      x = x.view(-1, 28*28)
      out = self.fc1(x)
      return out

def S_L_P():
    return SLP()
