import torch.nn as nn
from torch.nn import init

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image



class Conv_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
      super(Conv_BN_Relu, self).__init__()
      self.layer = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True)
                    )
    def forward(self, x):
      return self.layer(x)



def initialize_weights(m):
   if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    init.xavier_uniform(m.weight.data)