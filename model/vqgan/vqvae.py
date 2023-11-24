from typing import Any

import torch
import torch.nn as nn

from lutils.configuration import Configuration
from lutils.dict_wrapper import DictWrapper
from model.layers.utils import SequenceConverter
from model.vqgan.decoder import build_decoder
from model.vqgan.encoder import build_encoder
from model.vqgan.vector_quantizer import build_vector_quantizer


class VQVAE(nn.Module):
    def __init__(self, config: Configuration):
        super(VQVAE, self).__init__()

        self.config = config

        # Configure encoder
        self.encoder = build_encoder(config=self.config["encoder"])

        # Configure decoder
        self.decoder = build_decoder(config=self.config["decoder"])

        # Configure vector quantizer
        self.vector_quantizer = build_vector_quantizer(config=self.config["vector_quantizer"])

    def load_from_ckpt(self, ckpt_path: str):
        loaded_state = torch.load(ckpt_path, map_location="cpu")

        is_ddp = False
        for k in loaded_state["model"]:
            if k.startswith("module"):
                is_ddp = True
                break
        if is_ddp:
            state = {k.replace("module.", ""): v for k, v in loaded_state["model"].items()}
        else:
            state = {f"module.{k}": v for k, v in loaded_state["model"].items()}

        dmodel = self.module if isinstance(self, torch.nn.parallel.DistributedDataParallel) else self
        dmodel.load_state_dict(state)

    def forward(self, images: torch.Tensor) -> DictWrapper[str, Any]:
        # Encode images
        latents = self.encoder(images)

        # Quantize latent codes
        vq_loss, quantized_latents, quantized_latents_ids = self.vector_quantizer(latents)

        # Decode images
        reconstructed_images = self.decoder(quantized_latents)

        return DictWrapper(
            # Input
            images=images,

            # Reconstruction
            reconstructed_images=reconstructed_images,

            # Aux output
            vq_loss=vq_loss,
            latents=latents,
            quantized_latents=quantized_latents,
            quantized_latents_ids=quantized_latents_ids,
        )

    def get_latents_from_ids(self, latents_ids: torch.Tensor) -> torch.Tensor:
        """

        :param latents_ids: [b, h, w, n_e] one-hot vectors
        :return: [b, c, h, w]
        """

        latents = self.vector_quantizer.get_latents_from_ids(latents_ids)  # [b, e_dim, h, w]

        return latents

    def decode(self, latents_ids: torch.Tensor) -> torch.Tensor:
        """

        :param latents_ids: [b, h, w, n_e] one-hot vectors
        :return: [b, c, h, w]
        """

        latents = self.get_latents_from_ids(latents_ids)  # [b, e_dim, h, w]
        decoded_images = self.decoder(latents)  # [b, c, h, w]

        return decoded_images

    def decode_from_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """

        :param latents: [b, c, h, w] latents
        :return: [b, 3, h, w]
        """

        # Quantize latent codes
        _, quantized_latents, _ = self.vector_quantizer(latents)

        # Decode images
        reconstructed_images = self.decoder(quantized_latents)

        return reconstructed_images




def build_vqvae(config: Configuration, convert_to_sequence: bool = False):
    backbone = VQVAE(config)
    if convert_to_sequence:
        return SequenceConverter(backbone)
    else:
        return backbone
