#!/usr/bin/env python3
"""
CMIP6 NetCDF to ORCA-DL NPY Converter

将CMIP6下载的原始NetCDF数据转换为ORCA-DL项目可用的.npy格式训练数据。

使用方法:  
    python /mnt/data/zhu.yishun/ORCA-DL-main/dataset/data_preprocess.py \
    --input_dir /mnt/data/zhu.yishun/2015- \
    --output_dir /mnt/data/zhu.yishun/ORCA-DL-main/data/train_data \
    --stat_dir /mnt/data/zhu.yishun/ORCA-DL-main/stat \
    --grid_file /mnt/data/zhu.yishun/ORCA-DL-main/grid \
    --zaxis_file /mnt/data/zhu.yishun/ORCA-DL-main/zaxis.txt


输入数据命名格式:   {var}_Omon_{model}_historical_r1i1p1f1_gr_{start}-{end}.nc
例如: so_Omon_E3SM-1-0_historical_r1i1p1f1_gr_190501-190912.nc

输出目录结构:
    E3SM-1-0/
    ├── so/
    │   ├── 1905_1.npy
    │   ├── 1905_2.npy
    │   └── ...
    ├── thetao/
    │   └── ... 
    └── ... 
"""

import os
import re
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import h5py
from glob import glob
from tqdm import tqdm
from collections import defaultdict

try:
    from netCDF4 import Dataset as NetCDF4Dataset
except ImportError:
    NetCDF4Dataset = None

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import cftime
except ImportError:
    cftime = None

# 目标深度层 (单位: 米) - 与ORCA-DL一致
TARGET_LEVELS = [10, 15, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

# 目标网格分辨率 - 与ORCA-DL grid文件一致
TARGET_LAT = 128  # 纬度点数
TARGET_LON = 360  # 经度点数
LAT_FIRST = -63.5  # 起始纬度
LON_FIRST = 0.5    # 起始经度
LAT_INC = 1.0      # 纬度间隔
LON_INC = 1.0      # 经度间隔

# 变量分类
MULTI_LEVEL_VARS = ['thetao', 'so', 'uo', 'vo']  # 3D变量
SINGLE_LEVEL_VARS = ['zos', 'tos', 'tauu', 'tauv']  # 2D变量


def require_xarray():
    """在需要旧流程时检查xarray依赖。"""
    if xr is None:
        raise ImportError("当前环境未安装 xarray，无法执行 CMIP6 插值流程；processed_new1 模式不受影响")
    if cftime is None:
        raise ImportError("当前环境未安装 cftime，无法执行 CMIP6 插值流程；processed_new1 模式不受影响")


def check_cdo_available():
    """检查CDO是否可用"""
    try:
        result = subprocess.run(['cdo', '-V'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError: 
        return False


def parse_filename(filename):
    """
    解析CMIP6文件名，提取变量名、模型名和时间范围。
    
    Args:
        filename:   文件名，如 so_Omon_E3SM-1-0_historical_r1i1p1f1_gr_190501-190912.nc
    
    Returns:  
        dict: 包含 var_name, model_name, start_date, end_date
    """
    basename = os.path.basename(filename)
    # 匹配模式:   {var}_{frequency}_{model}_{experiment}_{variant}_{grid}_{timerange}.nc
    pattern = r'([a-zA-Z]+)_([a-zA-Z]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_(\d+)-(\d+)\.nc'
    match = re.match(pattern, basename)
    
    if match:
        return {
            'var_name': match.group(1),
            'frequency': match.group(2),
            'model_name': match.group(3),
            'experiment':  match.group(4),
            'variant': match.group(5),
            'grid':   match.group(6),
            'start_date': match.group(7),
            'end_date': match.group(8)
        }
    else:  
        raise ValueError(f"无法解析文件名:  {basename}")


def cftime_to_datetime(cftime_obj):
    """
    将cftime对象转换为年月信息
    
    Args: 
        cftime_obj: cftime日期对象 (如 cftime.DatetimeNoLeap)
    
    Returns: 
        tuple: (year, month)
    """
    if hasattr(cftime_obj, 'year') and hasattr(cftime_obj, 'month'):
        return cftime_obj.year, cftime_obj.month
    elif isinstance(cftime_obj, np.datetime64):
        # 处理numpy datetime64
        import pandas as pd
        ts = pd.Timestamp(cftime_obj)
        return ts.year, ts.month
    else:
        raise TypeError(f"无法解析时间类型: {type(cftime_obj)}")


def interpolate_with_cdo(input_file, output_file, var_name, grid_file, zaxis_file, is_3d=True):
    """
    使用CDO进行网格插值
    
    Args: 
        input_file: 输入NetCDF文件
        output_file: 输出NetCDF文件
        var_name: 变量名
        grid_file: 本地已有grid描述文件路径
        zaxis_file: 本地已有zaxis描述文件路径
        is_3d: 是否是3D变量
    
    Returns:
        bool: 是否成功
    """
    try: 
        temp_dir = os.path.dirname(output_file)
        
        if is_3d: 
            # 3D变量:  先水平插值，再垂直插值
            tmp1 = os.path.join(temp_dir, 'tmp1.nc')
            tmp2 = os.path.join(temp_dir, 'tmp2.nc')
            
            # 水平插值 (双线性) - 使用本地grid文件
            cmd1 = ['cdo', '-b', 'f64', f'remapbil,{grid_file}', input_file, tmp1]
            result1 = subprocess.run(cmd1, capture_output=True, text=True)
            if result1.returncode != 0:
                print(f"CDO remapbil 失败: {result1.stderr}")
                return False
            
            # 垂直插值到目标深度层
            levels_str = ','.join(map(str, TARGET_LEVELS))
            cmd2 = ['cdo', f'intlevel,{levels_str}', tmp1, tmp2]
            result2 = subprocess.run(cmd2, capture_output=True, text=True)
            if result2.returncode != 0:
                print(f"CDO intlevel 失败: {result2.stderr}")
                # 清理临时文件
                if os.path.exists(tmp1):
                    os.remove(tmp1)
                return False
            
            # 设置z轴信息 - 使用本地zaxis文件
            cmd3 = ['cdo', f'setzaxis,{zaxis_file}', tmp2, output_file]
            result3 = subprocess.run(cmd3, capture_output=True, text=True)
            if result3.returncode != 0:
                print(f"CDO setzaxis 失败:  {result3.stderr}")
            
            # 清理临时文件
            if os.path.exists(tmp1):
                os.remove(tmp1)
            if os.path.exists(tmp2):
                os.remove(tmp2)
                
        else:
            # 2D变量: 只进行水平插值 - 使用本地grid文件
            cmd = ['cdo', '-b', 'f64', f'remapbil,{grid_file}', input_file, output_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"CDO remapbil 失败: {result.stderr}")
                return False
        
        return os.path.exists(output_file)
        
    except Exception as e:
        print(f"CDO插值异常: {e}")
        return False


def compute_monthly_stats(data_dir, var_name, output_stat_dir):
    """
    计算变量的月度统计量 (mean和std) - 参考ORCA-DL的get_stat_and_norm方法
    
    Args:
        data_dir: 包含.npy文件的目录
        var_name: 变量名
        output_stat_dir: 统计量输出目录
    
    Returns: 
        dict: {'mean': array(12, .. .), 'std': array(12, ... )}
    """
    # 按月份分组收集数据
    monthly_data = {m: [] for m in range(1, 13)}
    
    npy_files = glob(os.path.join(data_dir, '*.npy'))
    
    for npy_file in npy_files: 
        basename = os.path.basename(npy_file)
        # 解析文件名格式: {year}_{month}.npy
        match = re.match(r'(\d+)_(\d+)\.npy', basename)
        if match:
            month = int(match.group(2))
            data = np.load(npy_file)
            monthly_data[month].append(data)
    
    # 计算每月的mean和std
    shape = monthly_data[1][0].shape if monthly_data[1] else None
    
    if shape is None:
        return None
    
    means = np.zeros((12,) + shape, dtype=np.float32)
    stds = np.zeros((12,) + shape, dtype=np.float32)
    
    for m in range(1, 13):
        if monthly_data[m]:
            stacked = np.stack(monthly_data[m], axis=0)
            means[m-1] = np.nanmean(stacked, axis=0)
            stds[m-1] = np.nanstd(stacked, axis=0)
            # 避免除零
            stds[m-1] = np.where(stds[m-1] == 0, 1.0, stds[m-1])
    
    # 保存统计量
    if output_stat_dir: 
        os.makedirs(output_stat_dir, exist_ok=True)
        np.save(os.path.join(output_stat_dir, f'{var_name}_mean.npy'), means)
        np.save(os.path.join(output_stat_dir, f'{var_name}_std.npy'), stds)
    
    return {'mean': means, 'std': stds}


def normalize_data(data, month, stats, var_name):
    """
    标准化数据，适配不同维度的统计量
    
    Args:
        data: 输入数据
        month: 月份 (1-12)
        stats: 统计量字典 {'mean': array, 'std': array}
        var_name: 变量名（用于调试）
    
    Returns:
        标准化后的数据
    """
    try:
        mean = stats['mean'][month - 1]
        std = stats['std'][month - 1]
        
        # 检查维度匹配
        if data.shape != mean.shape:
            print(f"  维度不匹配警告: 数据形状 {data.shape} != 统计量形状 {mean.shape}")
            print(f"  变量: {var_name}, 月份: {month}")
            
            # 尝试广播或调整维度（根据具体情况）
            if len(data.shape) == 2 and len(mean.shape) == 3:
                # 2D数据 vs 3D统计量 - 取统计量的表层
                if mean.shape[0] == 16:  # 深度维度
                    mean = mean[0]  # 取第一个深度层
                    std = std[0]
                else:
                    mean = mean[0]  # 取第一个维度
                    std = std[0]
            elif len(data.shape) == 3 and len(mean.shape) == 4:
                # 3D数据 vs 4D统计量 - 可能需要深度维度匹配
                if data.shape[0] == mean.shape[1]:  # 深度维度匹配
                    mean = mean[0]  # 去除月份维度，保留深度、纬度、经度
                    std = std[0]
        
        # 标准化: (data - mean) / std
        normalized = (data - mean) / std
        
        # 处理NaN和无穷值
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized.astype(np.float32)
        
    except Exception as e:
        print(f"  标准化失败 {var_name} 月份 {month}: {e}")
        # 失败时返回原始数据（仅处理NaN）
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def load_existing_stats(stat_dir, var_name):
    """
    加载已有的统计量文件，支持新的目录结构和变量名映射
    
    Args:
        stat_dir: 统计量目录 (包含mean和std子目录)
        var_name: CMIP6变量名
    
    Returns:
        dict: {'mean': array, 'std': array} 或 None
    """
    # CMIP6变量名到统计量文件名的映射
    var_mapping = {
        'thetao': 'pottmp',  # 潜在温度
        'so': 'salt',        # 盐度
        'uo': 'ucur',        # 东向流速
        'vo': 'vcur',        # 北向流速
        'tos': 'sst',        # 海表温度
        'zos': 'sshg',       # 海表面高度
        'tauu': 'uflx',      # 东向风应力
        'tauv': 'vflx'       # 北向风应力
    }
    
    # 获取实际使用的统计量文件名
    actual_name = var_mapping.get(var_name)
    
    if actual_name is None:
        print(f"  警告: 变量 {var_name} 没有对应的统计量映射，跳过标准化")
        return None
    
    # 构建统计量文件路径 (mean/和std/子目录)
    mean_path = os.path.join(stat_dir, 'mean', f'{actual_name}.npy')
    std_path = os.path.join(stat_dir, 'std', f'{actual_name}.npy')
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        try:
            mean_data = np.load(mean_path)
            std_data = np.load(std_path)
            
            print(f"  加载统计量: {actual_name} (来自 {var_name})")
            print(f"    均值形状: {mean_data.shape}, 标准差形状: {std_data.shape}")
            
            return {
                'mean': mean_data,
                'std': std_data
            }
        except Exception as e:
            print(f"  加载统计量文件失败: {e}")
            return None
    else:
        print(f"  警告: 统计量文件不存在 - 均值: {mean_path}, 标准差: {std_path}")
        return None

def process_nc_file_with_cdo(nc_file, output_base_dir, grid_file, zaxis_file, 
                              stat_dir=None, compute_stats=False, normalize=True):
    """
    修改后的处理函数，使用模型名和实验名组合作为子目录名称
    """
    require_xarray()
    # 解析文件名
    file_info = parse_filename(nc_file)
    var_name = file_info['var_name']
    model_name = file_info['model_name']
    experiment = file_info['experiment']  # 新增：获取实验名
    
    print(f"\n处理文件: {os.path.basename(nc_file)}")
    print(f"  变量: {var_name}, 模型: {model_name}, 实验: {experiment}")
    
    is_3d = var_name in MULTI_LEVEL_VARS
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 使用CDO进行插值
        interpolated_file = os.path.join(temp_dir, 'interpolated.nc')
        success = interpolate_with_cdo(nc_file, interpolated_file, var_name, 
                                        grid_file, zaxis_file, is_3d)
        
        if not success:
            print(f"  CDO插值失败，尝试使用scipy插值")
            process_nc_file_scipy(nc_file, output_base_dir, stat_dir, normalize)
            return
        
        # 创建输出目录 - 修改为使用模型名和实验名的组合
        model_exp_name = f"{model_name}_{experiment}"  # 组合名称
        output_dir = os.path.join(output_base_dir, model_exp_name, var_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载统计量（在循环外加载一次）
        stats = None
        if normalize and stat_dir:
            stats = load_existing_stats(stat_dir, var_name)
            if stats is None:
                print(f"  无法加载统计量，跳过标准化")
                normalize = False
        
        # 打开插值后的数据
        ds = xr.open_dataset(interpolated_file, use_cftime=True)
        data = ds[var_name]
        
        # 获取时间坐标
        time_coord = 'time' if 'time' in data.dims else data.dims[0]
        times = ds[time_coord].values
        
        # 处理每个时间步
        for t_idx, t in enumerate(tqdm(times, desc=f"  处理时间步")):
            try:
                year, month = cftime_to_datetime(t)
            except Exception as e: 
                print(f"  时间解析警告: {e}, 尝试其他方法")
                start_date = file_info['start_date']
                start_year = int(start_date[:4])
                start_month = int(start_date[4:6])
                total_months = start_month + t_idx - 1
                year = start_year + total_months // 12
                month = total_months % 12 + 1
            
            output_file = os.path.join(output_dir, f"{year}_{month}.npy")
            
            if os.path.exists(output_file):
                continue
            
            # 提取该时间步的数据
            time_data = data.isel({time_coord: t_idx}).values
            
            # 处理数据维度
            if is_3d and len(time_data.shape) == 2:
                time_data = time_data[np.newaxis, :, :]
            
            # 标准化数据（传入var_name用于调试）
            if normalize and stats is not None:
                time_data = normalize_data(time_data, month, stats, var_name)
            else:
                time_data = np.nan_to_num(time_data, nan=0.0, posinf=0.0, neginf=0.0)
                time_data = time_data.astype(np.float32)
            
            # 保存为.npy文件
            np.save(output_file, time_data)
        
        ds.close()
        print(f"  完成! 输出目录: {output_dir}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_nc_file_scipy(nc_file, output_base_dir, stat_dir=None, normalize=True):
    """
    使用scipy处理单个NetCDF文件 - 同样修改目录结构
    """
    require_xarray()
    from scipy.interpolate import RegularGridInterpolator, interp1d
    
    # 解析文件名
    file_info = parse_filename(nc_file)
    var_name = file_info['var_name']
    model_name = file_info['model_name']
    experiment = file_info['experiment']  # 新增：获取实验名
    
    # 创建目标网格
    target_lats = np.arange(LAT_FIRST, LAT_FIRST + TARGET_LAT * LAT_INC, LAT_INC)
    target_lons = np.arange(LON_FIRST, LON_FIRST + TARGET_LON * LON_INC, LON_INC)
    
    # 创建输出目录 - 修改为使用模型名和实验名的组合
    model_exp_name = f"{model_name}_{experiment}"  # 组合名称
    output_dir = os.path.join(output_base_dir, model_exp_name, var_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开数据集 - 使用use_cftime处理非标准日历
    ds = xr.open_dataset(nc_file, use_cftime=True)
    
    # 获取变量数据
    data = ds[var_name]
    
    # 获取时间坐标
    time_coord = 'time' if 'time' in data.dims else data.dims[0]
    times = ds[time_coord].values
    
    # 获取经纬度坐标
    lat_names = ['lat', 'latitude', 'j', 'y', 'nlat']
    lon_names = ['lon', 'longitude', 'i', 'x', 'nlon']
    
    src_lat_name = None
    src_lon_name = None
    
    for name in lat_names: 
        if name in ds.coords or name in ds.dims:
            src_lat_name = name
            break
    
    for name in lon_names: 
        if name in ds.coords or name in ds.dims:
            src_lon_name = name
            break
    
    if src_lat_name is None or src_lon_name is None:
        print(f"  警告: 无法找到经纬度坐标，跳过此文件")
        ds.close()
        return
    
    src_lats = ds[src_lat_name].values
    src_lons = ds[src_lon_name].values
    
    is_3d = var_name in MULTI_LEVEL_VARS
    
    if is_3d: 
        depth_names = ['lev', 'depth', 'z', 'olevel']
        src_depth_name = None
        
        for name in depth_names:
            if name in ds.coords or name in ds.dims:
                src_depth_name = name
                break
        
        if src_depth_name is None:
            print(f"  警告: 3D变量但无法找到深度坐标，跳过此文件")
            ds.close()
            return
        
        src_depths = ds[src_depth_name].values
        if src_depths.max() > 10000:
            src_depths = src_depths / 100
    
    # 加载统计量
    stats = None
    if normalize and stat_dir:
        stats = load_existing_stats(stat_dir, var_name)
        if stats is None:
            print(f"  无法加载统计量，跳过标准化")
            normalize = False
    
    # 处理每个时间步
    for t_idx, t in enumerate(tqdm(times, desc=f"  处理时间步")):
        try:
            year, month = cftime_to_datetime(t)
        except Exception as e:
            start_date = file_info['start_date']
            start_year = int(start_date[:4])
            start_month = int(start_date[4:6])
            total_months = start_month + t_idx - 1
            year = start_year + total_months // 12
            month = total_months % 12 + 1
        
        output_file = os.path.join(output_dir, f"{year}_{month}.npy")
        
        if os.path.exists(output_file):
            continue
        
        time_data = data.isel({time_coord: t_idx}).values
        
        if is_3d: 
            result = regrid_3d_scipy(time_data, src_lats, src_lons, src_depths,
                                      target_lats, target_lons, np.array(TARGET_LEVELS))
        else: 
            if len(time_data.shape) == 3:
                time_data = time_data[0]
            result = regrid_2d_scipy(time_data, src_lats, src_lons, target_lats, target_lons)
        
        # 标准化数据（传入var_name）
        if normalize and stats is not None: 
            result = normalize_data(result, month, stats, var_name)
        else:
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            result = result.astype(np.float32)
        
        np.save(output_file, result)
    
    ds.close()
    print(f"  完成! 输出目录: {output_dir}")

def regrid_2d_scipy(data, src_lats, src_lons, target_lats, target_lons):
    """使用scipy进行2D插值"""
    from scipy.interpolate import RegularGridInterpolator
    
    data_filled = np.where(np.isnan(data), 0, data)
    
    if src_lons.min() < 0:
        src_lons = np.where(src_lons < 0, src_lons + 360, src_lons)
        sort_idx = np.argsort(src_lons)
        src_lons = src_lons[sort_idx]
        data_filled = data_filled[:, sort_idx]
    
    try:
        interp = RegularGridInterpolator(
            (src_lats, src_lons), 
            data_filled, 
            method='linear',
            bounds_error=False,
            fill_value=0
        )
        
        target_grid = np.meshgrid(target_lats, target_lons, indexing='ij')
        target_points = np.stack([target_grid[0].ravel(), target_grid[1].ravel()], axis=-1)
        
        result = interp(target_points).reshape(len(target_lats), len(target_lons))
        
    except Exception as e: 
        print(f"插值警告: {e}, 使用最近邻方法")
        from scipy.interpolate import NearestNDInterpolator
        
        src_grid = np.meshgrid(src_lats, src_lons, indexing='ij')
        points = np.stack([src_grid[0].ravel(), src_grid[1].ravel()], axis=-1)
        values = data_filled.ravel()
        
        interp = NearestNDInterpolator(points, values)
        
        target_grid = np.meshgrid(target_lats, target_lons, indexing='ij')
        target_points = np.stack([target_grid[0].ravel(), target_grid[1].ravel()], axis=-1)
        
        result = interp(target_points).reshape(len(target_lats), len(target_lons))
    
    return result.astype(np.float32)


def regrid_3d_scipy(data, src_lats, src_lons, src_depths, target_lats, target_lons, target_depths):
    """使用scipy进行3D插值"""
    from scipy.interpolate import interp1d
    
    n_depths = len(src_depths)
    temp_data = np.zeros((n_depths, len(target_lats), len(target_lons)), dtype=np.float32)
    
    for d in range(n_depths):
        temp_data[d] = regrid_2d_scipy(data[d], src_lats, src_lons, target_lats, target_lons)
    
    result = np.zeros((len(target_depths), len(target_lats), len(target_lons)), dtype=np.float32)
    
    for i in range(len(target_lats)):
        for j in range(len(target_lons)):
            profile = temp_data[:, i, j]
            valid_mask = ~np.isnan(profile) & (profile != 0)
            
            if valid_mask.sum() >= 2:
                valid_depths = src_depths[valid_mask]
                valid_values = profile[valid_mask]
                
                try:
                    f = interp1d(valid_depths, valid_values, 
                                kind='linear', 
                                bounds_error=False, 
                                fill_value='extrapolate')
                    result[:, i, j] = f(target_depths)
                except: 
                    result[:, i, j] = 0
            elif valid_mask.sum() == 1:
                result[:, i, j] = profile[valid_mask][0]
            else:
                result[:, i, j] = 0
    
    return result.astype(np.float32)


def parse_processed_new1_filename(file_path):
    """
    解析 processed_new1 文件名中的年月信息。

    Args:
        file_path: 文件路径，格式为 <year>_<month>.nc

    Returns:
        tuple: (year, month)
    """
    basename = os.path.basename(file_path)
    match = re.match(r'(\d+)_(\d+)\.nc$', basename)
    if not match:
        raise ValueError(f"无法解析 processed_new1 文件名: {basename}")

    year = int(match.group(1))
    month = int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"月份超出范围: {basename}")

    return year, month


def collect_processed_new1_files(input_root, datasets=None, attrs=None):
    """
    扫描 processed_new1 目录，按属性聚合所有 nc 文件。

    Args:
        input_root: processed_new1 根目录
        datasets: 仅处理指定数据集
        attrs: 仅处理指定属性

    Returns:
        dict: {attr: [(dataset_name, month, file_path), ...]}
    """
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"processed_new1 输入目录不存在: {input_root}")

    dataset_filter = set(datasets) if datasets else None
    attr_filter = set(attrs) if attrs else None
    collected = {}

    for dataset_name in sorted(os.listdir(input_root)):
        dataset_dir = os.path.join(input_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        if dataset_filter and dataset_name not in dataset_filter:
            continue

        for attr_name in sorted(os.listdir(dataset_dir)):
            attr_dir = os.path.join(dataset_dir, attr_name)
            if not os.path.isdir(attr_dir):
                continue
            if attr_filter and attr_name not in attr_filter:
                continue

            file_infos = collected.setdefault(attr_name, [])
            for entry in sorted(os.listdir(attr_dir)):
                if not entry.endswith('.nc'):
                    continue
                file_path = os.path.join(attr_dir, entry)
                _, month = parse_processed_new1_filename(file_path)
                file_infos.append((dataset_name, month, file_path))

    return {attr: infos for attr, infos in collected.items() if infos}


def read_processed_new1_nc(file_path, attr_name):
    """
    读取 processed_new1 中的 nc 文件主变量，并去掉前导时间维。

    Args:
        file_path: nc 文件路径
        attr_name: 属性名，同时也是数据集名

    Returns:
        np.ndarray: 2D 或 3D 数组
    """
    data = None
    read_errors = []

    if NetCDF4Dataset is not None:
        try:
            with NetCDF4Dataset(file_path, 'r') as handle:
                if attr_name not in handle.variables:
                    raise KeyError(f"{file_path} 中不存在变量 {attr_name}")
                data = np.asarray(handle.variables[attr_name][...], dtype=np.float64)
        except Exception as exc:
            read_errors.append(f"netCDF4: {exc}")

    if data is None:
        try:
            with h5py.File(file_path, 'r') as handle:
                if attr_name not in handle:
                    raise KeyError(f"{file_path} 中不存在变量 {attr_name}")
                data = np.asarray(handle[attr_name][...], dtype=np.float64)
        except Exception as exc:
            read_errors.append(f"h5py: {exc}")

    if data is None:
        raise OSError(f"{file_path} 读取失败; " + " | ".join(read_errors))

    if data.ndim >= 3 and data.shape[0] == 1:
        data = data[0]

    if data.ndim not in (2, 3):
        raise ValueError(f"{file_path} 读取后维度异常: {data.shape}")

    return data


def summarize_processed_new1_skips(attr_name, skipped, stage, max_examples=10):
    """
    打印 processed_new1 跳过文件汇总，避免单个坏样本中断整批流程。
    """
    if not skipped:
        return

    print(f"[{stage}] {attr_name} 跳过 {len(skipped)} 个异常文件")
    for file_path, reason in skipped[:max_examples]:
        print(f"  - {file_path}: {reason}")
    if len(skipped) > max_examples:
        print(f"  ... 其余 {len(skipped) - max_examples} 个异常文件已省略")


def compute_monthly_stats_from_nc(file_infos):
    """
    使用流式累加方式计算某个属性的月度 mean/std。

    Args:
        file_infos: [(dataset_name, month, file_path), ...]

    Returns:
        dict: {'mean': array, 'std': array}
    """
    if not file_infos:
        raise ValueError("file_infos 不能为空")

    first_file = file_infos[0][2]
    attr_name = os.path.basename(os.path.dirname(first_file))
    skipped = []
    data_shape = None
    sums = None
    sum_sqs = None
    counts = None

    for _, month, file_path in tqdm(file_infos, desc=f"统计 {attr_name}", unit='file'):
        try:
            data = read_processed_new1_nc(file_path, attr_name)
        except (OSError, KeyError, ValueError) as exc:
            skipped.append((file_path, str(exc)))
            continue

        if data_shape is None:
            data_shape = data.shape
            sums = np.zeros((12,) + data_shape, dtype=np.float64)
            sum_sqs = np.zeros((12,) + data_shape, dtype=np.float64)
            counts = np.zeros((12,) + data_shape, dtype=np.int64)
        elif data.shape != data_shape:
            skipped.append((file_path, f"形状不一致: {data.shape} != {data_shape}"))
            continue

        month_idx = month - 1
        valid_mask = np.isfinite(data)
        month_sum = sums[month_idx]
        month_sum_sq = sum_sqs[month_idx]
        month_count = counts[month_idx]

        month_sum[valid_mask] += data[valid_mask]
        month_sum_sq[valid_mask] += np.square(data[valid_mask], dtype=np.float64)
        month_count[valid_mask] += 1

    if data_shape is None:
        raise ValueError(f"{attr_name} 没有可用于统计的有效文件")

    means = np.zeros_like(sums, dtype=np.float64)
    stds = np.ones_like(sums, dtype=np.float64)

    valid_counts = counts > 0
    np.divide(sums, counts, out=means, where=valid_counts)

    variances = np.zeros_like(sums, dtype=np.float64)
    np.divide(sum_sqs, counts, out=variances, where=valid_counts)
    variances = np.maximum(variances - np.square(means), 0.0)

    stds[valid_counts] = np.sqrt(variances[valid_counts])
    means[~valid_counts] = 0.0
    stds[~valid_counts] = 1.0
    stds[stds == 0] = 1.0

    return {
        'mean': means.astype(np.float32),
        'std': stds.astype(np.float32)
    }, skipped


def save_monthly_stats(stat_root, attr_name, stats):
    """
    保存月度统计量到 stat/mean 与 stat/std。
    """
    mean_dir = os.path.join(stat_root, 'mean')
    std_dir = os.path.join(stat_root, 'std')
    os.makedirs(mean_dir, exist_ok=True)
    os.makedirs(std_dir, exist_ok=True)

    np.save(os.path.join(mean_dir, f'{attr_name}.npy'), stats['mean'])
    np.save(os.path.join(std_dir, f'{attr_name}.npy'), stats['std'])


def load_processed_new1_stats(stat_root, attr_name):
    """
    加载 processed_new1 属性对应的月度统计量。
    """
    mean_path = os.path.join(stat_root, 'mean', f'{attr_name}.npy')
    std_path = os.path.join(stat_root, 'std', f'{attr_name}.npy')
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"缺少统计量文件: {mean_path} 或 {std_path}")

    return {
        'mean': np.load(mean_path),
        'std': np.load(std_path)
    }


def normalize_processed_new1_files(file_infos, stats, normalized_root, overwrite=False):
    """
    按月份将 processed_new1 nc 数据归一化并保存为 npy。

    Args:
        file_infos: [(dataset_name, month, file_path), ...]
        stats: {'mean': array, 'std': array}
        normalized_root: 归一化数据输出目录
        overwrite: 是否覆盖已有输出
    """
    if not file_infos:
        return

    first_file = file_infos[0][2]
    attr_name = os.path.basename(os.path.dirname(first_file))
    skipped = []

    for dataset_name, month, file_path in tqdm(file_infos, desc=f"归一化 {attr_name}", unit='file'):
        attr_dir = os.path.join(normalized_root, dataset_name, attr_name)
        os.makedirs(attr_dir, exist_ok=True)

        output_name = os.path.splitext(os.path.basename(file_path))[0] + '.npy'
        output_path = os.path.join(attr_dir, output_name)
        if os.path.exists(output_path) and not overwrite:
            continue

        try:
            data = read_processed_new1_nc(file_path, attr_name)
        except (OSError, KeyError, ValueError) as exc:
            skipped.append((file_path, str(exc)))
            continue

        mean = stats['mean'][month - 1]
        std = stats['std'][month - 1]
        if data.shape != mean.shape or data.shape != std.shape:
            skipped.append(
                (file_path, f"形状与统计量不匹配: data={data.shape}, mean={mean.shape}, std={std.shape}")
            )
            continue

        normalized = (data - mean) / std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        np.save(output_path, normalized)

    return skipped


def run_processed_new1_pipeline(args):
    """
    执行 processed_new1 月度统计与归一化流程。
    """
    input_root = args.processed_new1_root
    stat_root = args.stat_root
    normalized_root = args.normalized_root

    if not input_root:
        raise ValueError("processed_new1 模式必须提供 --processed_new1_root")
    if not stat_root:
        raise ValueError("processed_new1 模式必须提供 --stat_root")
    if args.phase in ('normalize', 'all') and not normalized_root:
        raise ValueError("phase 为 normalize/all 时必须提供 --normalized_root")

    file_map = collect_processed_new1_files(
        input_root=input_root,
        datasets=args.datasets,
        attrs=args.attrs
    )
    if not file_map:
        print("未找到任何符合条件的 processed_new1 nc 文件")
        return

    print(f"找到 {sum(len(v) for v in file_map.values())} 个 processed_new1 文件")
    print(f"涉及 {len(file_map)} 个属性: {', '.join(sorted(file_map))}")

    computed_stats = {}
    skipped_summary = defaultdict(lambda: {'stats': 0, 'normalize': 0})
    if args.phase in ('stats', 'all'):
        for attr_name, file_infos in sorted(file_map.items()):
            mean_path = os.path.join(stat_root, 'mean', f'{attr_name}.npy')
            std_path = os.path.join(stat_root, 'std', f'{attr_name}.npy')
            
            # 检查是否应该复用已有的统计量
            stats_exist = os.path.exists(mean_path) and os.path.exists(std_path)
            should_skip = stats_exist and not args.recompute_stats and not args.overwrite
            
            if should_skip:
                print(f"复用已有统计量: {attr_name}")
                continue
            
            if stats_exist and (args.recompute_stats or args.overwrite):
                if args.recompute_stats:
                    print(f"重新计算统计量: {attr_name} (--recompute-stats 指定)")
                else:
                    print(f"重新计算统计量: {attr_name} (--overwrite 指定)")

            stats, skipped = compute_monthly_stats_from_nc(file_infos)
            save_monthly_stats(stat_root, attr_name, stats)
            computed_stats[attr_name] = stats
            skipped_summary[attr_name]['stats'] = len(skipped)
            summarize_processed_new1_skips(attr_name, skipped, stage='stats')
            print(f"已保存统计量: {attr_name}")

    if args.phase in ('normalize', 'all'):
        for attr_name, file_infos in sorted(file_map.items()):
            stats = computed_stats.get(attr_name)
            if stats is None:
                stats = load_processed_new1_stats(stat_root, attr_name)

            skipped = normalize_processed_new1_files(
                file_infos=file_infos,
                stats=stats,
                normalized_root=normalized_root,
                overwrite=args.overwrite
            )
            skipped_summary[attr_name]['normalize'] = len(skipped)
            summarize_processed_new1_skips(attr_name, skipped, stage='normalize')
            print(f"已完成归一化输出: {attr_name}")

    if skipped_summary:
        print("\nprocessed_new1 异常文件汇总:")
        for attr_name in sorted(skipped_summary):
            stats_skipped = skipped_summary[attr_name]['stats']
            normalize_skipped = skipped_summary[attr_name]['normalize']
            if stats_skipped == 0 and normalize_skipped == 0:
                continue
            print(f"  - {attr_name}: stats 跳过 {stats_skipped} 个, normalize 跳过 {normalize_skipped} 个")


def main():
    parser = argparse.ArgumentParser(
        description='ORCA-DL 数据预处理脚本，支持 CMIP6 插值流程与 processed_new1 月度统计/归一化流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:  
    python convert_cmip6_to_npy.py --input_dir ./cmip6_data --output_dir ./train_data --grid_file ./grid --zaxis_file ./zaxis.txt
    
使用ORCA-DL提供的统计量进行标准化: 
    python convert_cmip6_to_npy.py --input_dir ./cmip6_data --output_dir ./train_data --grid_file ./grid --zaxis_file ./zaxis.txt --stat_dir ./stat
    
输入目录应包含CMIP6格式的NetCDF文件，如:  
    so_Omon_E3SM-1-0_historical_r1i1p1f1_gr_190501-190912.nc
    
输出将按以下结构组织:
    E3SM-1-0/
    ├── so/
    │   ├── 1905_1.npy
    │   └── ... 
    └── ...

processed_new1 月度统计与归一化:
    python data_preprocess.py \
        --processed_new1_root /mnt/data/zhu.yishun/ORCA-DL-main/data/processed_new1 \
        --stat_root /mnt/data/zhu.yishun/ORCA-DL-main/stat \
        --normalized_root /mnt/data/zhu.yishun/ORCA-DL-main/data/processed_new1_normalized \
        --phase all

直接复用已有的统计量（如果存在）进行归一化:
    python data_preprocess.py \
        --processed_new1_root /mnt/data/zhu.yishun/ORCA-DL-main/data/processed_new1 \
        --stat_root /mnt/data/zhu.yishun/ORCA-DL-main/stat \
        --normalized_root /mnt/data/zhu.yishun/ORCA-DL-main/data/processed_new1_normalized \
        --phase normalize

强制重新计算所有统计量，不使用已有结果:
    python data_preprocess.py \
        --processed_new1_root /mnt/data/zhu.yishun/ORCA-DL-main/data/processed_new1 \
        --stat_root /mnt/data/zhu.yishun/ORCA-DL-main/stat \
        --normalized_root /mnt/data/zhu.yishun/ORCA-DL-main/data/processed_new1_normalized \
        --phase all \
        --recompute-stats
        """
    )
    
    parser.add_argument('--input_dir', type=str, default=None,
                       help='CMIP6 NetCDF数据输入目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='转换后数据输出目录')
    parser.add_argument('--grid_file', type=str, default=None,
                       help='本地已有CDO grid配置文件路径 (如./grid)')
    parser.add_argument('--zaxis_file', type=str, default=None,
                       help='本地已有CDO zaxis配置文件路径 (如./zaxis.txt)')
    parser.add_argument('--stat_dir', type=str, default=None,
                       help='统计量目录 (用于标准化，如ORCA-DL提供的stat目录)')
    parser.add_argument('--var_filter', type=str, nargs='+', default=None,
                       help='只处理指定的变量 (如: so thetao tos)')
    parser.add_argument('--no_normalize', action='store_true',
                       help='不进行数据标准化 (仅插值)')
    parser.add_argument('--force_scipy', action='store_true',
                       help='强制使用scipy插值 (不使用CDO，忽略grid和zaxis文件)')
    parser.add_argument('--processed_new1_root', type=str, default=None,
                       help='processed_new1 输入根目录')
    parser.add_argument('--stat_root', type=str, default=None,
                       help='processed_new1 模式下统计量输出目录')
    parser.add_argument('--normalized_root', type=str, default=None,
                       help='processed_new1 模式下归一化 npy 输出目录')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='processed_new1 模式下仅处理指定数据集')
    parser.add_argument('--attrs', type=str, nargs='+', default=None,
                       help='processed_new1 模式下仅处理指定属性')
    parser.add_argument('--phase', type=str, choices=['stats', 'normalize', 'all'], default='all',
                       help='processed_new1 模式下执行阶段')
    parser.add_argument('--overwrite', action='store_true',
                       help='processed_new1 模式下覆盖已有统计量或归一化输出')
    parser.add_argument('--recompute-stats', action='store_true',
                       help='重新计算统计量，即使已有统计量文件也会重新计算（不使用已有结果）')
    
    args = parser.parse_args()

    if args.processed_new1_root:
        print("运行模式: processed_new1 月度统计与归一化")
        print(f"输入目录: {args.processed_new1_root}")
        print(f"统计量目录: {args.stat_root}")
        if args.normalized_root:
            print(f"归一化输出目录: {args.normalized_root}")
        print(f"执行阶段: {args.phase}")
        print(f"覆盖已有结果: {'是' if args.overwrite else '否'}")
        print(f"重新计算统计量: {'是' if args.recompute_stats else '否 (如已有则直接复用)'}")
        if args.datasets:
            print(f"数据集过滤: {args.datasets}")
        if args.attrs:
            print(f"属性过滤: {args.attrs}")

        run_processed_new1_pipeline(args)
        print("\nprocessed_new1 处理完成!")
        return

    if not args.input_dir:
        parser.error("未指定运行模式：请提供 --processed_new1_root，或提供 --input_dir/--output_dir 运行旧的 CMIP6 流程")
    if not args.output_dir:
        parser.error("CMIP6 流程必须提供 --output_dir")

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return

    # 检查本地grid和zaxis文件（仅当使用CDO时）
    use_cdo = check_cdo_available() and not args.force_scipy
    if use_cdo:
        if not args.grid_file or not os.path.exists(args.grid_file):
            print(f"错误: 指定的grid文件不存在: {args.grid_file}")
            return
        if not args.zaxis_file or not os.path.exists(args.zaxis_file):
            print(f"错误: 指定的zaxis文件不存在: {args.zaxis_file}")
            return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 显示配置信息
    if use_cdo:
        print("检测到CDO，将使用CDO进行网格插值 (使用本地grid和zaxis文件)")
        print(f"使用本地grid文件: {args.grid_file}")
        print(f"使用本地zaxis文件: {args.zaxis_file}")
    else:
        print("CDO不可用或已禁用，将使用scipy进行插值")

    print(f"目标网格: {TARGET_LON} x {TARGET_LAT} (经度 x 纬度)")
    print(f"网格范围: 经度 [{LON_FIRST}, {LON_FIRST + TARGET_LON * LON_INC}), 纬度 [{LAT_FIRST}, {LAT_FIRST + TARGET_LAT * LAT_INC})")
    print(f"目标深度层 ({len(TARGET_LEVELS)}层): {TARGET_LEVELS}")
    print(f"数据标准化: {'否' if args.no_normalize else '是'}")
    if args.stat_dir:
        print(f"统计量目录: {args.stat_dir}")

    # 查找所有NetCDF文件
    nc_files = glob(os.path.join(args.input_dir, '*.nc'))

    if len(nc_files) == 0:
        nc_files = glob(os.path.join(args.input_dir, '**', '*.nc'), recursive=True)

    print(f"\n找到 {len(nc_files)} 个NetCDF文件")

    # 过滤变量
    if args.var_filter:
        filtered_files = []
        for f in nc_files:
            try:
                info = parse_filename(f)
                if info['var_name'] in args.var_filter:
                    filtered_files.append(f)
            except Exception:
                pass
        nc_files = filtered_files
        print(f"过滤后:  {len(nc_files)} 个文件 (变量: {args.var_filter})")

    if use_cdo:
        # 直接使用用户指定的grid和zaxis文件，无需创建临时配置目录
        grid_file = os.path.abspath(args.grid_file)
        zaxis_file = os.path.abspath(args.zaxis_file)

        for nc_file in nc_files:
            try:
                process_nc_file_with_cdo(nc_file, args.output_dir, grid_file, zaxis_file,
                                         stat_dir=args.stat_dir,
                                         normalize=not args.no_normalize)
            except Exception as e:
                print(f"处理文件失败 {nc_file}:  {e}")
                import traceback
                traceback.print_exc()
    else:
        for nc_file in nc_files:
            try:
                process_nc_file_scipy(nc_file, args.output_dir,
                                      stat_dir=args.stat_dir,
                                      normalize=not args.no_normalize)
            except Exception as e:
                print(f"处理文件失败 {nc_file}: {e}")
                import traceback
                traceback.print_exc()

    print("\n转换完成!")
    print(f"输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()
'''
python /mnt/data/zhu.yishun/ORCA-DL-main/dataset/data_preprocess.py \
  --processed_new1_root /mnt/data/zhu.yishun/ORCA-DL-main/data/processed-val-test1 \
  --stat_root /mnt/data/zhu.yishun/ORCA-DL-main/stat \
  --normalized_root /mnt/data/zhu.yishun/ORCA-DL-main/data/valid_test_data \
  --phase normalize

'''
