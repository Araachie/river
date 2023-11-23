import albumentations
import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

from .h5 import HDF5Dataset


class Aug(nn.Module):
    def __init__(self, b: float, c: float, s: float, h: float):
        super(Aug, self).__init__()

        self.b = b
        self.c = c
        self.s = s
        self.h = h

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        im = F.adjust_brightness(im, brightness_factor=1 + self.b)
        im = F.adjust_contrast(im, contrast_factor=1 + self.c)
        im = F.adjust_saturation(im, saturation_factor=1 + self.s)
        im = F.adjust_hue(im, hue_factor=self.h)

        return im


class RandomConsistentAugFactory(nn.Module):
    def __init__(self, aug: bool = True):
        super(RandomConsistentAugFactory, self).__init__()

        self.aug = aug

    def forward(self):
        if self.aug:
            b = (torch.rand(1).item() - 0.5) / 5
            c = (torch.rand(1).item() - 0.5) / 5
            s = (torch.rand(1).item() - 0.5) / 5
            h = (torch.rand(1).item() - 0.5) / 2
            aug = Aug(b, c, s, h)

            return aug

        else:
            return T.Lambda(lambda x: x)


class VideoDataset(Dataset):

    def __init__(
            self,
            data_path,
            input_size: int,
            crop_size: int,
            frames_per_sample=5,
            skip_frames=0,
            random_time=True,
            random_horizontal_flip=True,
            aug=False,
            albumentations=False,
            total_videos=-1):

        self.data_path = data_path
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time
        self.skip_frames = skip_frames
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos

        self.albumentations = albumentations

        self.input_size = input_size
        self.crop_size = crop_size

        self.aug = RandomConsistentAugFactory(aug)

        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_path)

        print(f"Dataset length: {self.__len__()}")

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else len(self.videos_ds)

    def max_index(self):
        return len(self.videos_ds)

    def __getitem__(self, index, time_idx=0):
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

        # Setup augmentations
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        if self.albumentations:
            tr = albumentations.Compose([
                albumentations.SmallestMaxSize(max_size=self.input_size),
                albumentations.CenterCrop(height=self.crop_size, width=self.crop_size),
                albumentations.HorizontalFlip(p=flip_p)
            ])
        else:
            tr = T.Compose([
                T.Resize(size=self.input_size, antialias=True),
                T.CenterCrop(size=self.crop_size),
                T.RandomHorizontalFlip(p=flip_p)
            ])
        color_tr = self.aug()

        prefinals = []
        with h5py.File(self.videos_ds.shard_paths[shard_idx], "r") as f:
            video_len = f['len'][str(idx_in_shard)][()]
            num_frames = (self.skip_frames + 1) * (self.frames_per_sample - 1) + 1
            assert video_len >= num_frames, "The video is shorter than the desired sample size"
            if self.random_time:
                time_idx = np.random.choice(video_len - num_frames)
            assert time_idx < video_len, "Time index out of video boundary"
            for i in range(time_idx, min(time_idx + num_frames, video_len), self.skip_frames + 1):
                img = f[str(idx_in_shard)][str(i)][()]
                if self.albumentations:
                    arr = tr(image=img)["image"]
                else:
                    arr = img
                prefinals.append(torch.Tensor(arr).to(torch.uint8))

        data = torch.stack(prefinals)
        if not self.albumentations:
            data = tr(data.permute(0, 3, 1, 2))
        else:
            data = data.permute(0, 3, 1, 2)
        data = color_tr(data).to(torch.float32) / 127.5 - 1.0

        return data
