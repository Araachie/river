import argparse
import os
from pprint import pprint

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BAIRDataset
from fvd import I3D
from inception import inception_feature_extractor
from rcan import RCAN
from ssim import SSIM
from utils import c2f, f2c
from vqvae import VQModelInterface, vq_f8_small_ddconfig


# TODO SEQUENTIAL (Autoregressive) REFINED

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    return parser.parse_args()


def eval_refiner():
    ckpt_path = os.path.join('./experiments/', parse_args().exp_name, 'checkpoint.pth')
    ckpt = torch.load(ckpt_path)
    cudnn.benchmark = True
    args = ckpt['args']
    data_loader = DataLoader(BAIRDataset(data_path=args.data_path), batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_data_loader = DataLoader(BAIRDataset(data_path=args.data_path, mode='val'), batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    fp16_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    if args.fp16:
        fp16_scaler.load_state_dict(ckpt['fp16_scaler'])
    refiner = RCAN(args.rcan_stages, args.rcan_blocks_per_stage, args.rcan_channels,
                   args.rcan_reduction, args.rcan_tanh)
    refiner.load_state_dict({k[len('module.'):]: v for k, v in ckpt['refiner'].items()})
    refiner = refiner.eval().cuda()
    perceptual = I3D().eval().cuda()
    vqae = VQModelInterface(vq_f8_small_ddconfig, ckpt_path=args.vqgan_path).eval().cuda()
    inception = inception_feature_extractor()
    for m in [refiner, perceptual, vqae]:
        for p in m.parameters():
            p.requires_grad = False

    def perceptual_fn(x):
        return perceptual.detector(x, **perceptual.detector_args).cpu().numpy()

    result = {}
    for mode, dl in zip(['train', 'val'], [data_loader, val_data_loader]):
        both = extract(True, dl, refiner, vqae, perceptual_fn, inception, fp16_scaler)
        print(f'{mode} pixel mae', both['recon_mae'], '==>', both['refined_mae'])
        print(f'{mode} pixel mse', both['recon_mse'], '==>', both['refined_mse'])
        print(f'{mode} perceptual mse', both['recon_prc'], '==>', both['refined_prc'])
        print(f'{mode} psnr', both['recon_psnr'], '==>', both['refined_psnr'])
        print(f'{mode} ssim', both['recon_ssim'], '==>', both['refined_ssim'])
        recon_paired_fvd = perceptual.fvd(both['recon_feats'], both['target_feats'])
        recon_paired_fid = perceptual.fvd(both['recon_inception_feats'], both['target_inception_feats'])
        print(mode, 'recon_paired', f'fvd: {recon_paired_fvd}, fid: {recon_paired_fid}')
        refined_paired_fvd = perceptual.fvd(both['refined_feats'], both['target_feats'])
        refined_paired_fid = perceptual.fvd(both['refined_inception_feats'], both['target_inception_feats'])
        print(mode, 'refined_paired', f'fvd: {refined_paired_fvd}, fid: {refined_paired_fid}')
        target = extract(False, dl, refiner, vqae, perceptual_fn, inception, fp16_scaler)
        real_fvd = perceptual.fvd(target['target_feats'], both['target_feats'])
        real_fid = perceptual.fvd(target['target_inception_feats'], both['target_inception_feats'])
        print(mode, 'real', f'fvd: {real_fvd}, fid: {real_fid}')
        recon_fvd = perceptual.fvd(both['recon_inception_feats'], target['target_inception_feats'])
        recon_fid = perceptual.fvd(both['recon_inception_feats'], target['target_inception_feats'])
        print(mode, 'recon', f'fvd: {recon_fvd}, fid: {recon_fid}')
        refined_fvd = perceptual.fvd(both['refined_inception_feats'], target['target_inception_feats'])
        refined_fid = perceptual.fvd(both['refined_inception_feats'], target['target_inception_feats'])
        print(mode, 'refined', f'fvd: {refined_fvd}, fid: {refined_fid}')
        result[f'{mode}_recon_paired'] = recon_paired_fvd
        result[f'{mode}_recon_paired_fid'] = recon_paired_fid
        result[f'{mode}_refined_paired'] = refined_paired_fvd
        result[f'{mode}_refined_paired_fid'] = refined_paired_fid
        result[f'{mode}_recon'] = recon_fvd
        result[f'{mode}_recon_fid'] = recon_fid
        result[f'{mode}_refined'] = refined_fvd
        result[f'{mode}_refined_fid'] = refined_fid
        result[f'{mode}_real'] = real_fvd
        result[f'{mode}_real_fid'] = real_fid
    print()
    pprint(result)


def psnr(y_pred, y, data_range=2.0):
    mse_error = torch.pow(y_pred.double() - y.double(), 2).mean(dim=tuple(range(1, y.ndim)))
    return torch.mean(10.0 * torch.log10(data_range ** 2 / (mse_error + 1e-10)))


@torch.no_grad()
def extract(both, data_loader, refiner, vqae, perceptual, inception, fp16_scaler=None):
    target_feats = []
    target_inception_features = []
    if both:
        recon_feats = []
        recon_inception_features = []
        refined_feats = []
        refined_inception_features = []
        recon_mse = 0.0
        refined_mse = 0.0
        recon_mae = 0.0
        refined_mae = 0.0
        recon_psnr = 0.0
        refined_psnr = 0.0
        recon_ssim = SSIM(data_range=2.0)
        refined_ssim = SSIM(data_range=2.0)
        count = 0
    for i, x in enumerate(tqdm(data_loader, total=100)):
        if i == 100:
            break
        x = x.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            target = x[:, :, 1:].contiguous()
            target_frames = c2f(target)
            target_feats.append(perceptual(target))
            target_inception_features.append(inception(target_frames).cpu().numpy())
            if not both:
                continue
            recon = vqae.decode(vqae.encode(target_frames))
            recon_inception_features.append(inception(recon).cpu().numpy())
            cond = c2f(x[:, :, :-1])
            refiner_input = torch.cat((cond, recon), dim=1)
            refined = refiner(refiner_input)  # BT 2C H W => BT C H W
            refined_inception_features.append(inception(refined).cpu().numpy())
            recon = f2c(recon, x.size(0))
            recon_feats.append(perceptual(recon))
            refined = f2c(refined, x.size(0))
            refined_feats.append(perceptual(refined))
            recon_mae += F.l1_loss(recon, target)
            recon_mse += F.mse_loss(recon, target)
            recon_psnr += psnr(recon, target)
            recon_ssim.update(c2f(recon), target_frames)
            refined_mae += F.l1_loss(refined, target)
            refined_mse += F.mse_loss(refined, target)
            refined_psnr += psnr(refined, target)
            refined_ssim.update(c2f(refined), target_frames)
            count += 1
    res = {'target_feats': np.concatenate(target_feats, axis=0),
           'target_inception_feats': np.concatenate(target_inception_features, axis=0)}
    if both:
        recon_feats = np.concatenate(recon_feats, axis=0)
        refined_feats = np.concatenate(refined_feats, axis=0)
        res.update({
            'recon_feats': recon_feats,
            'recon_inception_feats': np.concatenate(recon_inception_features, axis=0),
            'refined_feats': refined_feats,
            'refined_inception_feats': np.concatenate(refined_inception_features, axis=0),
            'recon_prc': ((res['target_feats'] - recon_feats) ** 2).mean(),
            'refined_prc': ((res['target_feats'] - refined_feats) ** 2).mean(),
            'recon_mse': recon_mse.item() / count,
            'refined_mse': refined_mse.item() / count,
            'recon_mae': recon_mae.item() / count,
            'refined_mae': refined_mae.item() / count,
            'recon_psnr': recon_psnr.item() / count,
            'refined_psnr': refined_psnr.item() / count,
            'recon_ssim': recon_ssim.compute(),
            'refined_ssim': refined_ssim.compute()})
    return res


if __name__ == '__main__':
    eval_refiner()
