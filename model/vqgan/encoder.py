import torch
import torch.nn as nn
from einops import rearrange

from lutils.configuration import Configuration
from model.layers import ResidualBlock
from model.vqgan.utils import swish, normalize


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 32):
        super(Encoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        residual_layers = []
        ch_mult = [1, 2, 4, 8]
        for i in range(len(ch_mult) - 1):
            in_ch = ch_mult[i] * mid_channels
            out_ch = ch_mult[i + 1] * mid_channels
            residual_layers.append(ResidualBlock(
                in_ch, out_ch, downsample_factor=2, norm_layer=normalize))
        self.residuals = nn.Sequential(*residual_layers)

        attn_ch = ch_mult[-1] * mid_channels
        self.pre_attn_residual = ResidualBlock(attn_ch, attn_ch, downsample_factor=1, norm_layer=normalize)
        self.attn_norm = normalize(attn_ch)
        self.attn = nn.MultiheadAttention(embed_dim=attn_ch, num_heads=1, batch_first=True)
        self.post_attn_residual = ResidualBlock(attn_ch, attn_ch, downsample_factor=1, norm_layer=normalize)

        self.out_norm = normalize(attn_ch)
        self.out_conv = nn.Conv2d(attn_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """

        :param images: [b, c, h, w]
        """

        x = self.conv_in(images)
        x = self.residuals(x)

        x = self.pre_attn_residual(x)
        z = self.attn_norm(x)
        h = z.size(2)
        z = rearrange(z, "b c h w -> b (h w) c")
        z, _ = self.attn(query=z, key=z, value=z)
        z = rearrange(z, "b (h w) c -> b c h w", h=h)
        x = x + z
        x = self.post_attn_residual(x)

        x = self.out_norm(x)
        x = swish(x)
        x = self.out_conv(x)

        return x


def build_encoder(config: Configuration) -> nn.Module:
    return Encoder(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"])
