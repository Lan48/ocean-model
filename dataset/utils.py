import io
import os
import numpy as np
import pandas as pd
import xarray as xr
from functools import lru_cache
from global_land_mask import globe


def get_stat_and_norm(indir, outdir, stat_save_dir=None, v_name=None, lat_space=None, lon_space=None, client=None, mask_land=True, time_axis_name='time'):
    
    if client is not None:
        list_fn = lambda d: [x.replace('/', '') for x in client.list(d)]
    else:
        list_fn = os.listdir

    ds = []
    f_list = []
    for name in sorted(list_fn(indir)):
        try:
            if client is not None:
                f = io.BytesIO(client.get(os.path.join(indir, name)))
                tmp = xr.open_dataset(f)
                f_list.append(f)
            else:
                tmp = xr.load_dataset(os.path.join(indir, name))
        except:
            for f in f_list:
                f.cloes()
            print(indir, name, 'error')
            return False

        ds.append(tmp)

    ds_cat = xr.concat(ds, time_axis_name)
    if v_name is None:
        v_name = ds_cat.attrs['variable_id']

    mean_path = os.path.join(stat_save_dir, f"{v_name}_mean.npy")
    std_path = os.path.join(stat_save_dir, f"{v_name}_std.npy")

    ds_grouped = ds_cat.groupby(f'{time_axis_name}.month')

    mean = ds_grouped.mean()
    std = ds_grouped.std()

    if stat_save_dir is not None:
        os.makedirs(stat_save_dir, exist_ok=True)
        np.save(mean_path, mean[v_name].values)
        np.save(std_path, std[v_name].values)
        # mean.to_netcdf(mean_path)
        # std.to_netcdf(std_path)

    os.makedirs(outdir, exist_ok=True)
    ds_normed = (ds_grouped - mean).groupby(f'{time_axis_name}.month') / std
    st_time = ds_normed[time_axis_name].values[0]

    if isinstance(st_time, np.datetime64):
        st_time = pd.Timestamp(st_time)
    y, m = st_time.year, st_time.month

    ds_np = ds_normed[v_name].values
    ds_np = np.nan_to_num(ds_np, nan=0.0, posinf=0.0, neginf=0.0)

    mask = None
    if lat_space is not None and lon_space is not None:
        mask = compute_land_mask(lat_space, lon_space)

    if mask is not None and mask_land:
        if len(ds_np.shape) == 3:
            ds_np[:, mask] = 0.0
        else:
            ds_np[:, :, mask] = 0.0

    for i in range(ds_np.shape[0]):
        np.save(os.path.join(outdir, f"{y}_{m}.npy"), ds_np[i])
        m += 1
        if m == 13:
            m = 1
            y += 1

    for f in f_list:
        f.close()

    return mean, std


def load_mean_std(stat_save_dir, var_list, predict_month=None, type='npy'):
    means = {}
    stds = {}
    for v in var_list:
        if type == 'npy':
            mean = np.load(os.path.join(stat_save_dir, f"{v}_mean.npy"))
            std = np.load(os.path.join(stat_save_dir, f"{v}_std.npy"))
        elif type == 'nc':
            mean = xr.load_dataset(os.path.join(stat_save_dir, f"{v}_mean.nc"))[v].values
            std = xr.load_dataset(os.path.join(stat_save_dir, f"{v}_std.nc"))[v].values
        else:
            assert 0

        if predict_month is not None:
            mean = mean[predict_month]
            std = std[predict_month]
            
        means[v] = mean
        stds[v] = std

    return {'mean': means, 'std': stds}

@lru_cache
def compute_land_mask(lat_space, lon_space):
    #land 1 ocean 0
    lat = np.linspace(*lat_space)
    lon = np.linspace(*lon_space)
    lon, lat = np.meshgrid(lon, lat)
    land_mask = np.roll(globe.is_land(lat, lon-180), 180, axis=1)
    return land_mask
