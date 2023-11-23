# The module was adopted from this repository https://github.com/MishaLaskin/vqvae
import torch
import torch.nn as nn
from einops import rearrange

from lutils.configuration import Configuration


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        min_encodings = min_encodings.view(z.size(0), z.size(1), z.size(2), self.n_e)

        return loss, z_q, min_encodings

    def get_latents_from_ids(self, latents_ids: torch.Tensor) -> torch.Tensor:
        """

        :param latents_ids: [b, h, w, n_e] one-hot vectors
        :return: [b, e_dim, h, w]
        """

        b, h, _, _ = latents_ids.shape

        flat_latents_ids = rearrange(latents_ids, "b h w e -> (b h w) e").to(torch.float32)
        flat_latents = torch.matmul(flat_latents_ids, self.embedding.weight)
        latents = rearrange(flat_latents, "(b h w) e -> b e h w", b=b, h=h)

        return latents


def build_vector_quantizer(config: Configuration) -> nn.Module:
    return VectorQuantizer(
        n_e=config["num_embeddings"],
        e_dim=config["embedding_dimension"],
        beta=0.25)

