import torch
import torch.nn as nn
import torch.nn.functional as F
from basicseg.utils.registry import NET_REGISTRY

@NET_REGISTRY.register()
class My_Model(nn.Module):
    def __init__(in_c=3, out_c=1, base_dim=32):
        super().__init__()
        conv1 = nn.Conv2d(in_c, base_dim, 3)
        conv2 = nn.Conv2d(base_dim, base_dim * 2, 3)
        conv3 = nn.Conv2d(base_dim * 2, out_c)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x3
        