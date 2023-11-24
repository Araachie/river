import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import BAIRDataset
from fvd import I3D
from rcan import RCAN
from utils import (restart_from_checkpoint, cosine_scheduler, save_on_master,
                   is_main_process, MetricLogger, init_distributed_mode, c2f, f2c)
from vqvae import VQModelInterface, vq_f8_small_ddconfig


# OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train.py --exp_name test --data_path path --vqgan_path path
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--vqgan_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=24, help='per gpu')
    parser.add_argument('--num_workers', type=int, default=4, help='per gpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--optim', default='adam', type=str, choices=['adam', 'sgd'])
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--perceptual_loss_weight', type=float, default=0.1)
    parser.add_argument('--rcan_stages', default=3, type=int)
    parser.add_argument('--rcan_blocks_per_stage', default=3, type=int)
    parser.add_argument('--rcan_channels', default=64, type=int)
    parser.add_argument('--rcan_reduction', default=8, type=int)
    parser.add_argument('--rcan_tanh', action='store_true')
    parser.add_argument('--l1', action='store_true')
    return parser.parse_args()


def train_refiner():
    args = parse_args()
    args.output_dir = os.path.join('./experiments/', args.exp_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)
    cudnn.benchmark = True
    dataset = BAIRDataset(data_path=args.data_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, sampler=DistributedSampler(dataset), drop_last=True)
    val_data_loader = DataLoader(BAIRDataset(data_path=args.data_path, mode='val'), batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    refiner = DistributedDataParallel(RCAN(args.rcan_stages, args.rcan_blocks_per_stage, args.rcan_channels,
                                           args.rcan_reduction, args.rcan_tanh).cuda(),
                                      device_ids=[args.gpu], gradient_as_bucket_view=True, static_graph=True)
    perceptual = I3D().eval().cuda()
    vqae = VQModelInterface(vq_f8_small_ddconfig, ckpt_path=args.vqgan_path).eval().cuda()
    for p in vqae.parameters():
        p.requires_grad = False
    lr_schedule = cosine_scheduler(
        args.lr * args.batch_size * args.world_size / 256,  # linear scaling rule
        0.0, args.epochs, len(data_loader), warmup_epochs=args.warmup_epochs)
    if args.optim == 'adam':
        optimizer = AdamW(refiner.parameters(), lr=0.0, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(refiner.parameters(), lr=0.0, momentum=0.9, weight_decay=args.weight_decay)
    fp16_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    to_restore = {'epoch': 0}
    ckpt_path = os.path.join('./experiments/', args.exp_name, 'checkpoint.pth')
    restart_from_checkpoint(
        ckpt_path, run_variables=to_restore, refiner=refiner, optimizer=optimizer, fp16_scaler=fp16_scaler)
    if is_main_process():
        wandb.init(config=args, project='river-refinement', name=args.exp_name)
    start_epoch = to_restore['epoch']
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(refiner, vqae, perceptual, data_loader, optimizer,
                                      lr_schedule, epoch, fp16_scaler, args)
        save_dict = {
            'refiner': refiner.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'args': args}
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        save_on_master(save_dict, ckpt_path)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            validate(refiner, vqae, perceptual, val_data_loader, fp16_scaler, args, (epoch + 1) * len(data_loader))


def forward(x, refiner, vqae, perceptual, perceptual_loss_weight, l1, fp16_scaler=None, verbose=True):
    x = x.cuda(non_blocking=True)  # x is B C T+1 H W and it is between -1 and 1
    res = {}
    with torch.cuda.amp.autocast(fp16_scaler is not None):
        with torch.no_grad():
            target = x[:, :, 1:].contiguous()
            recon = vqae.decode(vqae.encode(c2f(target)))
            cond = c2f(x[:, :, :-1])
            refiner_input = torch.cat((cond, recon), dim=1)
            recon = f2c(recon, x.size(0))
            if verbose:
                if l1:
                    res['recon_pixel_loss'] = F.l1_loss(recon, target).item()
                else:
                    res['recon_pixel_loss'] = F.mse_loss(recon, target).item()
                res['recon_perceptual_loss'] = perceptual(recon, target).item()
                res['recon_loss'] = res['recon_pixel_loss'] + res['recon_perceptual_loss'] * perceptual_loss_weight
        refined = refiner(refiner_input)  # BT 2C H W => BT C H W
        refined = f2c(refined, x.size(0))
        if l1:
            pixel_loss = F.l1_loss(refined, target)
        else:
            pixel_loss = F.mse_loss(refined, target)
        perceptual_loss = perceptual(refined, target)
        loss = pixel_loss + perceptual_loss * perceptual_loss_weight
    res.update({'loss': loss, 'perceptual_loss': perceptual_loss.item(), 'pixel_loss': pixel_loss.item()})
    return res, {'recon': recon[0], 'target': target[0], 'refined': refined[0]}


def train_one_epoch(refiner, vqae, perceptual, data_loader, optimizer, lr_schedule, epoch, fp16_scaler, args):
    metric_logger = MetricLogger(delimiter='  ')
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    refiner.train()
    for it, x in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]
        logs, samples = forward(x, refiner, vqae, perceptual, args.perceptual_loss_weight, args.l1, fp16_scaler)
        loss = logs['loss']
        logs['loss'] = loss.item()
        if not math.isfinite(logs['loss']):
            print('Loss is {}, stopping training'.format(logs['loss']))
            sys.exit(1)
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        logs['lr'] = optimizer.param_groups[0]['lr']
        for k, v in logs.items():
            if k.endswith('loss'):
                metric_logger.update(**{k: v})
        metric_logger.update(lr=logs['lr'])
        torch.cuda.synchronize()
        if is_main_process():
            wandb.log(logs, step=it)
            if it % 100 == 0:  # or val
                frames = samples['recon'].size(1)
                recon = samples['recon'].permute(1, 0, 2, 3)
                refined = samples['refined'].permute(1, 0, 2, 3)
                target = samples['target'].permute(1, 0, 2, 3)
                wandb.log({'AE': wandb.Image(make_grid(recon, nrow=frames, value_range=(-1, 1))),
                           'RF': wandb.Image(make_grid(refined, nrow=frames, value_range=(-1, 1))),
                           'GT': wandb.Image(make_grid(target, nrow=frames, value_range=(-1, 1))),
                           'GT/AE': wandb.Image(make_grid((recon - target).abs().mean(1, keepdim=True),
                                                          nrow=frames, value_range=(0, 2))),
                           'GT/RF': wandb.Image(make_grid((refined - target).abs().mean(1, keepdim=True),
                                                          nrow=frames, value_range=(0, 2)))},
                          step=it)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(refiner, vqae, perceptual, data_loader, fp16_scaler, args, step):
    refiner.eval()
    aggregated_log = {}
    count = 0
    for x in tqdm(data_loader):
        logs, samples = forward(x, refiner, vqae, perceptual, args.perceptual_loss_weight, args.l1, fp16_scaler)
        for k, v in logs.items():
            if k not in aggregated_log:
                aggregated_log[k] = v
            else:
                aggregated_log[k] += v
        count += 1
    aggregated_log = {k + '_val': v / count for k, v in aggregated_log.items()}
    samples = {k + '_val': v for k, v in samples.items()}
    wandb.log(aggregated_log, step=step)
    wandb.log(samples, step=step)


if __name__ == '__main__':
    train_refiner()
