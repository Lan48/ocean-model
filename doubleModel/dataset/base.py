import io
import os
import bisect
from typing import Sequence
import torch
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset, Subset


GODAS2CMIP6 = {
    'pottmp': 'thetao',
    'salt': 'so',
    'sshg': 'zos',
    'sst': 'tos',
    'ucur': 'uo',
    'uflx': 'tauu',
    'vcur': 'vo',
    'vflx': 'tauv',
}

CMIP62GODAS = {
    'thetao': 'pottmp',
    'so': 'salt',
    'zos': 'sshg',
    'tos': 'sst',
    'uo': 'ucur',
    'tauu': 'uflx',
    'vo': 'vcur',
    'tauv': 'vflx',
}

MULTY_LEVEL_VARS = ['thetao', 'so', 'uo', 'vo']
SINGLE_LEVEL_VARS = ['zos', 'tos', 'tauu', 'tauv']


class BaseDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.root = args.data_dir
        self.cached = args.cached

        self.list_dir_fn = os.listdir

        self.num_depths = getattr(args, 'num_depths', None)

    def specify_split_by_time(self):

        self.start_year = int(self.times[0].split('_')[0])
        self.end_year = int(self.times[-1].split('_')[0])

        self.train_val_split_year = self.args.train_val_split_year if self.args.train_val_split_year is not None else self.end_year
        self.val_test_split_year = self.args.val_test_split_year

        if self.val_test_split_year is not None:
            assert self.train_val_split_year < self.val_test_split_year <= self.end_year

    def convert_vname_godas_to_cmip6(self, v):
        return GODAS2CMIP6[v]

    def convert_vname_cmip6_to_godas(self, v):
        return CMIP62GODAS[v]

    def _cache_sample(self, index, data):
        torch.save(data, os.path.join(self.args.cache_sample_dir, f'{index}.pt'))

    def cache_samples(self):
        if self.args.cached and not self.args.overwrite_cache:
            raise ValueError(
                "Tring to cache data, but data is already cached, set overwrite_cache to True")

        out_dir = self.args.cache_sample_dir
        os.makedirs(out_dir, exist_ok=True)

        if self.args.overwrite_cache:
            file_list = sorted(os.listdir(out_dir),
                                key=lambda x: int(x.split('.')[0]))
            if len(file_list) > 0 and int(file_list[-1].split('.')[0]) > len(self):
                num_max = int(file_list[-1].split('.')[0])
                for i in tqdm(range(len(self), num_max), desc='Removing last cache data...'):
                    if os.path.exists(os.path.join(out_dir, f'{i}.pt')):
                        os.remove(os.path.join(out_dir, f'{i}.pt'))

        bar = tqdm(total=len(self), desc='Cache samples...')
        # print(os.cpu_count())
        pool = Pool()
        for i in range(len(self)):
            pool.apply_async(self._cache_sample, (i, self[i]), callback=lambda _: bar.update(),
                                error_callback=lambda err: print(str(err)))
        pool.close()
        pool.join()

        self.cached = True


    def load_times(self, all_time=False, check=False):
        self.times = sorted([t.split('.')[0] for t in
                             self.list_dir_fn(os.path.join(self.root, self.input_var_list[0]))],
                            key=lambda x: int(x.split('_')[0]) * 12 + int(x.split('_')[1]))
        start_year = getattr(self.args, 'start_year', None)
        end_year = getattr(self.args, 'end_year', None)
        st_idx = 0
        end_idx = len(self.times)

        if not all_time:
            if start_year is not None:
                for i in range(len(self.times)):
                    if int(self.times[i].split('_')[0]) == start_year:
                        break
                st_idx = i
            if end_year is not None:
                for i in range(len(self.times)):
                    if int(self.times[i].split('_')[0]) == end_year:
                        break
                end_idx = i
        
        self.times = self.times[st_idx:end_idx]

        if check:
            self.check_times()

    def check_times(self):
        for i in range(len(self.times)-1):
            t_ = self.times[i+1].split('_')
            t = self.times[i].split('_')

            y_, m_ = int(t_[0]), int(t_[1])
            y, m = int(t[0]), int(t[1])

            if not ((y == y_ and m+1 == m_) or (y+1 == y_ and m == 12 and m_ == 1)):
                assert 0, f"prepare data error, check {self.root} {v}."

        for v in self.input_var_list:
            times = sorted([t.split('.')[0] for t in
                            self.list_dir_fn(os.path.join(self.root, v))],
                            key=lambda x: int(x.split('_')[0]) * 12 + int(x.split('_')[1]))
            if times != self.times:
                assert 0, f"prepare data error, inconsistency of times, check {self.root} {v}."

    def get_cache_sample(self, index):
        return torch.load(os.path.join(self.args.cache_sample_dir, f'{index}.pt'))

    def get_data(self, path):
        data = np.load(path)
        return torch.from_numpy(data)

    def get_values(self, base_path, var_list, time_range, cat=True):
        data = [
            torch.stack(
                [self.get_data(
                    os.path.join(base_path, v, f'{self.times[i]}.npy')
                ) for v in var_list] #[:self.num_depths]
            ) for i in time_range
        ]
        if cat:
            return torch.cat(data).float()
        else:
            return torch.stack(data).float()

    def get_subset(self, indices):
        return Subset(self, indices)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        raise NotImplementedError()