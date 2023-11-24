import torch
import torch.nn as nn


def normalize(in_channels, **kwargs):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def swish(x):
    return x*torch.sigmoid(x)
