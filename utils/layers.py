import torch.nn as nn

def countParams(model, config):
  num_params = sum([p.data.nelement() for p in model.parameters()])
  config.log('Number of model parameters: {}\n'.format(num_params))
  return num_params


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image