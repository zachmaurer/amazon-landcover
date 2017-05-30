import torch.nn as nn
from torch.nn import init

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def initialize_weights(m):
   if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    init.xavier_uniform(m.weight.data)