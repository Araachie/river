import torch
import torch.nn as nn
from einops import rearrange

from lutils.configuration import Configuration
from model.layers import ResidualBlock, UpBlock
from model.vqgan.utils import swish, normalize


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 256):
        super(Decoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.pre_attn_residual = ResidualBlock(mid_channels, mid_channels, downsample_factor=1, norm_layer=normalize)
        self.attn_norm = normalize(mid_channels)
        self.attn = nn.MultiheadAttention(embed_dim=mid_channels, num_heads=1, batch_first=True)
        self.post_attn_residual = ResidualBlock(mid_channels, mid_channels, downsample_factor=1, norm_layer=normalize)

        residual_layers = []
        ch_div = [1, 2, 4, 8]
        for i in range(len(ch_div) - 1):
            in_ch = mid_channels // ch_div[i]
            out_ch = mid_channels // ch_div[i + 1]
            residual_layers.append(nn.Sequential(
                ResidualBlock(in_ch, out_ch, downsample_factor=1, norm_layer=normalize),
                UpBlock(out_ch, out_ch, scale_factor=2, upscaling_mode="nearest")))
        self.residuals = nn.Sequential(*residual_layers)

        out_ch = mid_channels // ch_div[-1]
        self.out_norm = normalize(out_ch)
        self.out_conv = nn.Conv2d(out_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """

        :param images: [b, c, h, w]
        """

        x = self.conv_in(images)

        x = self.pre_attn_residual(x)
        z = self.attn_norm(x)
        h = z.size(2)
        z = rearrange(z, "b c h w -> b (h w) c")
        z, _ = self.attn(query=z, key=z, value=z)
        z = rearrange(z, "b (h w) c -> b c h w", h=h)
        x = x + z
        x = self.post_attn_residual(x)

        x = self.residuals(x)

        x = self.out_norm(x)
        x = swish(x)
        x = self.out_conv(x)

        return torch.tanh(x)


def build_decoder(config: Configuration) -> nn.Module:
    return Decoder(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"])
