import torch
from torch import nn


class CustomNet(nn.Module):

  def __init__(self):
    super(CustomNet, self).__init__()
    # Input features: 3 canali * 224 larghezza * 224 altezza = 150528
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(3 * 224 * 224, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 200),
    )

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.linear_relu_stack(x)
    return x