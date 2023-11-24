import importlib

import torch
import torch.nn as nn

from model.vqgan.taming.modules import Encoder, Decoder
from model.vqgan.taming.quantize import VectorQuantizer2 as VectorQuantizer


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# https://ommer-lab.com/files/latent-diffusion/vq-f8.zip
vq_f8_ddconfig = dict(embed_dim=4, n_embed=16384, double_z=False, z_channels=4, resolution=256, in_channels=3,
                      out_ch=3, ch=128, ch_mult=[1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[32], dropout=0.0)


vq_f8_small_ddconfig = dict(embed_dim=4, n_embed=16384, double_z=False, z_channels=4, resolution=64, in_channels=3,
                      out_ch=3, ch=128, ch_mult=[1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0)


# https://heibox.uni-heidelberg.de/f/0e42b04e2e904890a9b6/?dl=1
vq_f16_ddconfig = dict(embed_dim=8, n_embed=16384, double_z=False, z_channels=8, resolution=256, in_channels=3,
                       out_ch=3, ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0)


class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(ddconfig["n_embed"], ddconfig["embed_dim"], beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], ddconfig["embed_dim"], 1)
        self.post_quant_conv = torch.nn.Conv2d(ddconfig["embed_dim"], ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


class VQModelInterface(VQModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec