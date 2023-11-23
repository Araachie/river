import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from lutils.configuration import Configuration
from model.layers.position_encoding import build_position_encoding


def timestamp_embedding(timesteps, dim, scale=200, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param scale: a premultiplier of timesteps
    :param max_period: controls the minimum frequency of the embeddings.
    :param repeat_only: whether to repeat only the values in timesteps along the 2nd dim
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = scale * timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(scale * timesteps, 'b -> b d', d=dim)
    return embedding


class VectorFieldRegressor(nn.Module):
    def __init__(
            self,
            depth: int,
            mid_depth: int,
            state_size: int,
            state_res: Tuple[int, int],
            inner_dim: int,
            out_norm: str = "ln",
            reference: bool = True):
        super(VectorFieldRegressor, self).__init__()

        self.state_size = state_size
        self.state_height = state_res[0]
        self.state_width = state_res[1]
        self.inner_dim = inner_dim
        self.reference = reference

        self.position_encoding = build_position_encoding(self.inner_dim, position_embedding_name="learned")

        self.project_in = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"),
            nn.Linear(3 * self.state_size if self.reference else 2 * self.state_size, self.inner_dim)
        )

        self.time_projection = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, self.inner_dim)
        )

        def build_layer(d_model: int):
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=4 * d_model,
                dropout=0.05,
                activation="gelu",
                norm_first=True,
                batch_first=True)

        self.in_blocks = nn.ModuleList()
        self.mid_blocks = nn.Sequential(*[build_layer(self.inner_dim) for _ in range(mid_depth)])
        self.out_blocks = nn.ModuleList()
        for i in range(depth):
            self.in_blocks.append(build_layer(self.inner_dim))
            self.out_blocks.append(nn.ModuleList([
                nn.Linear(2 * self.inner_dim, self.inner_dim),
                build_layer(self.inner_dim)]))

        if out_norm == "ln":
            self.project_out = nn.Sequential(
                nn.Linear(self.inner_dim, self.inner_dim),
                nn.GELU(),
                nn.LayerNorm(self.inner_dim),
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
            )
        elif out_norm == "bn":
            self.project_out = nn.Sequential(
                nn.Linear(self.inner_dim, self.inner_dim),
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.GELU(),
                nn.BatchNorm2d(self.inner_dim),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
            )
        else:
            raise NotImplementedError

    def forward(
            self,
            input_latents: torch.Tensor,
            reference_latents: torch.Tensor,
            conditioning_latents: torch.Tensor,
            index_distances: torch.Tensor,
            timestamps: torch.Tensor) -> torch.Tensor:
        """

        :param input_latents: [b, c, h, w]
        :param reference_latents: [b, c, h, w]
        :param conditioning_latents: [b, c, h, w]
        :param index_distances: [b]
        :param timestamps: [b]
        :return: [b, c, h, w]
        """

        # Fetch timestamp tokens
        t = timestamp_embedding(timestamps, dim=self.inner_dim).unsqueeze(1)

        # Calculate position embedding
        pos = self.position_encoding(input_latents)
        pos = rearrange(pos, "b c h w -> b (h w) c")

        # Calculate distance embeddings
        dist = self.time_projection(torch.log(index_distances).unsqueeze(1)).unsqueeze(1)

        # Build input tokens
        if self.reference:
            x = self.project_in(torch.cat([input_latents, reference_latents, conditioning_latents], dim=1))
        else:
            x = self.project_in(torch.cat([input_latents, conditioning_latents], dim=1))
        x = x + pos + dist
        x = torch.cat([t, x], dim=1)

        # Propagate through the main network
        hs = []
        for block in self.in_blocks:
            x = block(x)
            hs.append(x.clone())
        x = self.mid_blocks(x)
        for i, block in enumerate(self.out_blocks):
            x = block[1](block[0](torch.cat([hs[-i - 1], x], dim=-1)))

        # Project to output
        out = self.project_out(x[:, 1:])

        return out


def build_vector_field_regressor(config: Configuration, reference: bool = True):
    return VectorFieldRegressor(
        state_size=config["state_size"],
        state_res=config["state_res"],
        inner_dim=config["inner_dim"],
        depth=config["depth"],
        mid_depth=config["mid_depth"],
        out_norm=config["out_norm"],
        reference=reference,
    )
