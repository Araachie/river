import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


class I3D(nn.Module):
    def __init__(self):
        super().__init__()
        # https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1
        self.detector = torch.jit.load('./i3d_torchscript.pt').eval()
        self.detector_args = dict(rescale=False, resize=True, return_features=True)

    def forward(self, x, target):  # perceptual loss
        with torch.no_grad():
            target = self.detector(target, **self.detector_args)  # N, 400
        x = self.detector(x, **self.detector_args)
        return F.mse_loss(x, target)

    @staticmethod
    def compute_stats(feats: np.ndarray):
        feats = feats.astype(np.float64)
        mu = feats.mean(axis=0)  # [d]
        sigma = np.cov(feats, rowvar=False)  # [d, d]
        return mu, sigma

    @torch.no_grad()
    def fvd(self, feats_fake, feats_real):
        mu_gen, sigma_gen = self.compute_stats(feats_fake)
        mu_real, sigma_real = self.compute_stats(feats_real)
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
        fvd = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fvd)
