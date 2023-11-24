import glob
import os

import albumentations
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# https://github.com/fab-jul/hdf5_dataloader
class HDF5Dataset(Dataset):

    @staticmethod
    def _get_num_in_shard(shard_p):
        print(f'\rh5: Opening {shard_p}... ', end='')
        try:
            with h5py.File(shard_p, "r") as f:
                num_per_shard = len(f['len'].keys())
        except:
            print(f"h5: Could not open {shard_p}!")
            num_per_shard = -1
        return num_per_shard

    @staticmethod
    def check_shard_lengths(file_paths):
        """
        Filter away the last shard, which is assumed to be smaller. this double checks that all other shards have the
        same number of entries.
        :param file_paths: list of .hdf5 files
        :return: tuple (ps, num_per_shard) where
            ps = filtered file paths,
            num_per_shard = number of entries in all of the shards in `ps`
        """
        shard_lengths = []
        print("Checking shard_lengths in", file_paths)
        for i, p in enumerate(file_paths):
            shard_lengths.append(HDF5Dataset._get_num_in_shard(p))
        return shard_lengths

    def __init__(self, data_path,  # hdf5 file, or directory of hdf5s
                 shuffle_shards=False,
                 seed=29):
        self.data_path = data_path
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        # If `data_path` is an hdf5 file
        if os.path.splitext(self.data_path)[-1] == '.hdf5' or os.path.splitext(self.data_path)[-1] == '.h5':
            self.data_dir = os.path.dirname(self.data_path)
            self.shard_paths = [self.data_path]
        # Else, if `data_path` is a directory of hdf5s
        else:
            self.data_dir = self.data_path
            self.shard_paths = sorted(
                glob.glob(os.path.join(self.data_dir, '*.hdf5')) + glob.glob(os.path.join(self.data_dir, '*.h5')))

        assert len(self.shard_paths) > 0, "h5: Directory does not have any .hdf5 files! Dir: " + self.data_dir

        self.shard_lengths = HDF5Dataset.check_shard_lengths(self.shard_paths)
        self.num_per_shard = self.shard_lengths[0]
        self.total_num = sum(self.shard_lengths)

        assert len(
            self.shard_paths) > 0, "h5: Could not find .hdf5 files! Dir: " + self.data_dir + " ; len(self.shard_paths) = " + str(
            len(self.shard_paths))

        self.num_of_shards = len(self.shard_paths)

        print("h5: paths", len(self.shard_paths), "; shard_lengths", self.shard_lengths, "; total", self.total_num)

        # Shuffle shards
        if self.shuffle_shards:
            np.random.seed(seed)
            np.random.shuffle(self.shard_paths)

    def __len__(self):
        return self.total_num

    def get_indices(self, idx):
        shard_idx = np.digitize(idx, np.cumsum(self.shard_lengths))
        idx_in_shard = str(idx - sum(self.shard_lengths[:shard_idx]))
        return shard_idx, idx_in_shard

    def __getitem__(self, index):
        idx = index % self.total_num
        shard_idx, idx_in_shard = self.get_indices(idx)
        # Read from shard
        with h5py.File(self.shard_paths[shard_idx], "r") as f:
            data = f[idx_in_shard][()]
        return data


class HDF5Maker:

    def __init__(self, out_path, num_per_shard=100000, max_shards=None, name=None, name_fmt='shard_{:04d}.hdf5',
                 force=False, video=False):

        # `out_path` could be an hdf5 file, or a directory of hdf5s
        # If `out_path` is an hdf5 file, then `name` will be its basename
        # If `out_path` is a directory, then `name` will be used if provided else name_fmt will be used

        self.out_path = out_path
        self.num_per_shard = num_per_shard
        self.max_shards = max_shards
        self.name = name
        self.name_fmt = name_fmt
        self.force = force
        self.video = video

        # If `out_path` is an hdf5 file
        if os.path.splitext(self.out_path)[-1] == '.hdf5' or os.path.splitext(self.out_path)[-1] == '.h5':
            # If it exists, check if it should be deleted
            if os.path.isfile(self.out_path):
                if not self.force:
                    raise ValueError('{} already exists.'.format(self.out_path))
                print('Removing {}...'.format(self.out_path))
                os.remove(self.out_path)
            # Make the directory if it does not exist
            self.out_dir = os.path.dirname(self.out_path)
            os.makedirs(self.out_dir, exist_ok=True)
            # Extract its name
            self.name = os.path.basename(self.out_path)
        # Else, if `out_path` is a directory
        else:
            self.out_dir = self.out_path
            # If `out_dir` exists
            if os.path.isdir(self.out_dir):
                # Check if it should be deleted
                if not self.force:
                    raise ValueError('{} already exists.'.format(self.out_dir))
                print('Removing *.hdf5 files from {}...'.format(self.out_dir))
                files = glob.glob(os.path.join(self.out_dir, "*.hdf5"))
                files += glob.glob(os.path.join(self.out_dir, "*.h5"))
                for file in files:
                    os.remove(file)
            # Else, make the directory
            else:
                os.makedirs(self.out_dir)

        self.writer = None
        self.shard_paths = []
        self.shard_number = 0

        self.create_new_shard()

    def create_new_shard(self):
        if self.writer:
            self.writer.close()
        self.shard_number += 1
        if self.max_shards is not None and self.shard_number == self.max_shards + 1:
            print('Created {} shards, ENDING.'.format(self.max_shards))
            return
        self.shard_p = os.path.join(self.out_dir,
                                    self.name_fmt.format(self.shard_number) if self.name is None else self.name)
        assert not os.path.exists(self.shard_p), 'Record already exists! {}'.format(self.shard_p)
        self.shard_paths.append(self.shard_p)
        print('Creating shard # {}: {}...'.format(self.shard_number, self.shard_p))
        self.writer = h5py.File(self.shard_p, 'w')
        if self.video:
            self.create_video_groups()
        self.count = 0

    def create_video_groups(self):
        self.writer.create_group('len')
        self.writer.create_group('videos')

    def add_video_data(self, data, dtype=None):
        self.writer['len'].create_dataset(str(self.count), data=len(data))
        self.writer.create_group(str(self.count))
        for i, frame in enumerate(data):
            self.writer[str(self.count)].create_dataset(str(i), data=frame, dtype=dtype, compression="lzf")

    def add_data(self, data, dtype=None, return_curr_count=False):
        if self.video:
            self.add_video_data(data, dtype)
        else:
            self.writer.create_dataset(str(self.count), data=data, dtype=dtype, compression="lzf")
        curr_count = self.count
        self.count += 1
        if self.count == self.num_per_shard:
            self.create_new_shard()
        if return_curr_count:
            return curr_count

    def close(self):
        self.writer.close()
        assert len(self.shard_paths)


# https://github.com/edenton/svg/blob/master/data/bair.py
class BAIRDataset(Dataset):
    def __init__(self, data_path, mode='train', input_size=64, crop_size=64,
                 frames_per_sample=17, random_time=True, random_horizontal_flip=False, total_videos=-1):
        assert mode in {'train', 'val', 'test'}
        self.data_path = os.path.join(data_path, mode)
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos  # If we wish to restrict total number of videos (e.g. for val)
        self.input_size = input_size
        self.crop_size = crop_size
        self.videos_ds = HDF5Dataset(self.data_path)  # Read h5 files as dataset

    @staticmethod
    def window_stack(a, width=3, step=1):
        return torch.stack([a[i:1 + i - width or None:step] for i in range(width)]).transpose(0, 1)

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with h5py.File(self.videos_ds.shard_paths[shard_idx], "r") as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else len(self.videos_ds)

    def max_index(self):
        return len(self.videos_ds)

    def __getitem__(self, index, time_idx=0):
        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        clip = []
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        with h5py.File(self.videos_ds.shard_paths[shard_idx], "r") as f:
            video_len = f['len'][str(idx_in_shard)][()]
            if self.random_time and video_len > self.frames_per_sample:
                time_idx = np.random.choice(video_len - self.frames_per_sample)
            for i in range(time_idx, min(time_idx + self.frames_per_sample, video_len)):
                img = f[str(idx_in_shard)][str(i)][()]
                tr = albumentations.Compose([
                    albumentations.SmallestMaxSize(max_size=self.input_size),
                    albumentations.CenterCrop(height=self.crop_size, width=self.crop_size),
                    albumentations.HorizontalFlip(p=flip_p)
                ])
                arr = tr(image=img)["image"]
                clip.append(torch.Tensor(arr / 127.5 - 1.0).permute(2, 0, 1).to(torch.float32))
        return torch.stack(clip, dim=1)
