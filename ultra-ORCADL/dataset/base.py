import io
import os
import bisect
from glob import glob
from typing import Sequence
import torch
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset, Subset
try:
    import xarray as xr
except ImportError:
    xr = None
try:
    import h5py
except ImportError:
    h5py = None

from variable_config import (
    DEFAULT_MODEL_ATMO_VARS,
    get_var_channels,
    is_multi_level_var,
    is_single_level_var,
    MODEL_MULTI_LEVEL_VARS,
    MODEL_SINGLE_LEVEL_VARS,
    MODEL_TO_STAT_MAPPING,
    STAT_TO_MODEL_MAPPING,
    ensure_supported_vars,
    normalize_to_model_var,
    sort_atmo_vars,
    sort_ocean_vars,
    to_stat_var,
)

GODAS2CMIP6 = dict(STAT_TO_MODEL_MAPPING)
CMIP62GODAS = dict(MODEL_TO_STAT_MAPPING)
MULTY_LEVEL_VARS = sorted(MODEL_MULTI_LEVEL_VARS)
SINGLE_LEVEL_VARS = sorted(MODEL_SINGLE_LEVEL_VARS)


class BaseDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.root = args.data_dir
        self.cached = args.cached

        self.list_dir_fn = os.listdir

        self.num_depths = getattr(args, 'num_depths', None)
        self.default_input_shape = (128, 360)
        self._shape_cache = {}

    def specify_split_by_time(self):

        self.start_year = int(self.times[0].split('_')[0])
        self.end_year = int(self.times[-1].split('_')[0])

        self.train_val_split_year = self.args.train_val_split_year if self.args.train_val_split_year is not None else self.end_year
        self.val_test_split_year = self.args.val_test_split_year

        if self.val_test_split_year is not None:
            assert self.train_val_split_year < self.val_test_split_year <= self.end_year

    def convert_vname_godas_to_cmip6(self, v):
        return GODAS2CMIP6.get(v, v)

    def convert_vname_cmip6_to_godas(self, v):
        return CMIP62GODAS.get(v, v)

    def _normalize_var_names(self, var_list):
        return ensure_supported_vars(var_list)

    def prepare_ocean_var_list(self, available_vars, requested_vars=None):
        if requested_vars:
            requested = self._normalize_var_names(requested_vars)
        else:
            requested = [
                normalize_to_model_var(var_name)
                for var_name in available_vars
                if normalize_to_model_var(var_name) not in DEFAULT_MODEL_ATMO_VARS
            ]
        return sort_ocean_vars(requested)

    def prepare_atmo_var_list(self, available_vars, requested_vars=None):
        if requested_vars:
            requested = self._normalize_var_names(requested_vars)
        else:
            requested = [
                normalize_to_model_var(var_name)
                for var_name in available_vars
                if normalize_to_model_var(var_name) in DEFAULT_MODEL_ATMO_VARS
            ] or list(DEFAULT_MODEL_ATMO_VARS)
        return sort_atmo_vars(requested)

    def resolve_var_dir(self, base_path, var_name):
        model_var = normalize_to_model_var(var_name)
        candidates = [model_var, to_stat_var(model_var)]
        for candidate in candidates:
            candidate_path = os.path.join(base_path, candidate)
            if os.path.isdir(candidate_path):
                return candidate
        return candidates[0]

    def resolve_var_path(self, base_path, var_name, time_key, suffix=".npy"):
        var_dir = self.resolve_var_dir(base_path, var_name)
        return os.path.join(base_path, var_dir, f"{time_key}{suffix}")

    def _list_time_keys_for_var(self, base_path, var_name):
        var_dir = self.resolve_var_dir(base_path, var_name)
        var_root = os.path.join(base_path, var_dir)
        if not os.path.isdir(var_root):
            return []

        keys = set()
        for pattern in ("*.npy", "*.nc"):
            for path in glob(os.path.join(var_root, pattern)):
                keys.add(os.path.splitext(os.path.basename(path))[0])
        return sorted(keys, key=lambda x: int(x.split('_')[0]) * 12 + int(x.split('_')[1]))

    def _build_full_time_axis(self, time_keys):
        if len(time_keys) == 0:
            return []

        def parse_time_key(time_key):
            year_str, month_str = time_key.split('_')
            return int(year_str), int(month_str)

        start_year, start_month = parse_time_key(time_keys[0])
        end_year, end_month = parse_time_key(time_keys[-1])
        year, month = start_year, start_month
        full_times = []
        while (year, month) <= (end_year, end_month):
            full_times.append(f"{year}_{month}")
            month += 1
            if month == 13:
                month = 1
                year += 1
        return full_times

    def _resolve_existing_file_path(self, base_path, var_name, time_key):
        for suffix in (".npy", ".nc"):
            path = self.resolve_var_path(base_path, var_name, time_key, suffix=suffix)
            if os.path.exists(path):
                return path
        return None

    def _load_nc_data(self, path, var_name):
        model_var = normalize_to_model_var(var_name)
        stat_var = to_stat_var(model_var)
        candidates = [model_var, stat_var]

        arr = None
        if xr is not None:
            try:
                with xr.open_dataset(path) as ds:
                    data_var_name = next((name for name in candidates if name in ds.data_vars), None)
                    if data_var_name is None:
                        data_vars = list(ds.data_vars)
                        if data_vars:
                            data_var_name = data_vars[0]
                    if data_var_name is not None:
                        arr = np.asarray(ds[data_var_name].values)
            except Exception:
                arr = None

        if arr is None and h5py is not None:
            with h5py.File(path, "r") as handle:
                datasets = []

                def visitor(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets.append((name, obj))

                handle.visititems(visitor)
                selected = None
                for candidate in candidates:
                    for name, dataset in datasets:
                        if name.split("/")[-1] == candidate:
                            selected = dataset
                            break
                    if selected is not None:
                        break
                if selected is None:
                    for _, dataset in datasets:
                        if dataset.ndim >= 2:
                            selected = dataset
                            break
                if selected is None:
                    raise ValueError(f"No readable data variables found in {path}")
                arr = np.asarray(selected[()])

        if arr is None:
            raise ImportError("xarray or h5py is required to load NetCDF data")

        arr = np.squeeze(arr)
        while arr.ndim > 3:
            arr = arr[0]
        return arr

    def _sanitize_array(self, arr, var_name):
        arr = np.asarray(arr, dtype=np.float32)
        # Some reanalysis files use large finite sentinels such as 1e20 for
        # missing values. Treat them as invalid before the generic nan_to_num.
        arr = np.where(np.abs(arr) > 1e10, np.nan, arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 0:
            arr = np.full(self.default_input_shape, float(arr), dtype=np.float32)
        return arr

    def _infer_shape_from_existing_files(self, base_path, var_name):
        var_dir = self.resolve_var_dir(base_path, var_name)
        var_root = os.path.join(base_path, var_dir)
        if not os.path.isdir(var_root):
            return None

        for pattern in ("*.npy", "*.nc"):
            paths = sorted(glob(os.path.join(var_root, pattern)))
            for path in paths:
                try:
                    if path.endswith(".npy"):
                        arr = np.load(path)
                    else:
                        arr = self._load_nc_data(path, var_name)
                    arr = self._sanitize_array(arr, var_name)
                    return arr.shape
                except Exception:
                    continue
        return None

    def infer_var_shape(self, base_path, var_name):
        model_var = normalize_to_model_var(var_name)
        cache_key = (base_path, model_var)
        if cache_key in self._shape_cache:
            return self._shape_cache[cache_key]

        shape = self._infer_shape_from_existing_files(base_path, model_var)
        if shape is None:
            channels = get_var_channels(model_var)
            if is_single_level_var(model_var):
                shape = self.default_input_shape
            else:
                shape = (channels, *self.default_input_shape)

        self._shape_cache[cache_key] = shape
        return shape

    def build_zero_array(self, base_path, var_name):
        shape = self.infer_var_shape(base_path, var_name)
        return np.zeros(shape, dtype=np.float32)

    def get_var_data(self, base_path, var_name, time_key):
        file_path = self._resolve_existing_file_path(base_path, var_name, time_key)
        if file_path is None:
            return torch.from_numpy(self.build_zero_array(base_path, var_name))

        try:
            if file_path.endswith(".npy"):
                arr = np.load(file_path)
            elif file_path.endswith(".nc"):
                arr = self._load_nc_data(file_path, var_name)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            arr = self._sanitize_array(arr, var_name)
        except Exception:
            arr = self.build_zero_array(base_path, var_name)

        return torch.from_numpy(arr)

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
        candidate_vars = list(self.input_var_list) + list(getattr(self, "atmo_var_list", []))
        all_time_keys = set()
        for var_name in candidate_vars:
            all_time_keys.update(self._list_time_keys_for_var(self.root, var_name))

        self.times = self._build_full_time_axis(
            sorted(all_time_keys, key=lambda x: int(x.split('_')[0]) * 12 + int(x.split('_')[1]))
        )
        if len(self.times) == 0:
            raise ValueError(f"No .npy or .nc data files found under {self.root}")
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
                assert 0, f"prepare data error, invalid time sequence under {self.root}."

        for v in self.input_var_list:
            times = self._list_time_keys_for_var(self.root, v)
            if not set(times).issubset(set(self.times)):
                assert 0, f"prepare data error, inconsistency of times, check {self.root} {v}."

    def get_cache_sample(self, index):
        return torch.load(os.path.join(self.args.cache_sample_dir, f'{index}.pt'))

    def get_data(self, path):
        if path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".nc"):
            data = self._load_nc_data(path, os.path.basename(os.path.dirname(path)))
        else:
            raise ValueError(f"Unsupported data file: {path}")
        data = self._sanitize_array(data, os.path.basename(os.path.dirname(path)))
        return torch.from_numpy(data)

    def get_values(self, base_path, var_list, time_range, cat=True):
        data = [
            torch.stack(
                [self.get_var_data(base_path, v, self.times[i]) for v in var_list]
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
