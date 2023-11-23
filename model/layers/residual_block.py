import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride), bias=False)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            downsample_factor: int = 1,
            last_affine: bool = True,
            drop_final_activation: bool = False,
            norm_layer=nn.BatchNorm2d):
        """

        :param in_planes: Input features to the module
        :param out_planes: Output feature
        :param downsample_factor: Reduction factor in feature dimension
        :param drop_final_activation: if True does not pass the final output through the activation function
        """

        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride=1)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        # Enable the possibility to force alignment to normal gaussian
        self.bn2 = norm_layer(out_planes, affine=last_affine)
        self.downsample_factor = downsample_factor
        self.drop_final_activation = drop_final_activation

        self.downsample = None
        if self.downsample_factor != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, out_planes, stride=1),
                nn.AvgPool2d(downsample_factor),
                # Enable the possibility to force alignment to normal gaussian
                norm_layer(out_planes, affine=last_affine)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = F.avg_pool2d(out, self.downsample_factor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if not self.drop_final_activation:
            out = self.relu(out)

        return out
