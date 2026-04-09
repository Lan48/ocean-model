#!/usr/bin/env python3
"""
ORCA-DL 海洋预测可视化API服务
集成 defitools-img-pred.py 和 ocean_prediction_core.py 的核心能力

功能: 
1./seasonal_mean_visual - 季节平均态模拟可视化
2./enso_forecast_visual - 季节性预测（ENSO）可视化
3./marine_heatwave_visual - 海洋热浪可视化
4./decadal_prediction_visual - 年代际预测可视化
"""
import os
import io
import base64
import logging
from typing import List, Optional, Union
from enum import Enum

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# 导入核心模块
from ocean_prediction_core import (
    ORCADLPredictor, 
    OceanPredictionResult,
    predict_ocean_state
)
from variable_config import (
    DEFAULT_MODEL_ATMO_VARS,
    DEFAULT_MODEL_OCEAN_VARS,
    MODEL_TO_STAT_MAPPING,
    get_var_channels,
)

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 配置常量（复用 defitools-img-pred.py） ====================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VAR_LIST = list(DEFAULT_MODEL_OCEAN_VARS)
OUT_CHANS = [get_var_channels(var_name) for var_name in VAR_LIST]

import numpy as np
import torch
try:
    import xarray as xr
except ImportError:
    xr = None
try:
    import h5py
except ImportError:
    h5py = None

example_dir = os.getenv(
    "ORCADL_EXAMPLE_DIR",
    "/mnt/data/zhu.yishun/ORCA-DL-main/data/godas2025-12",
)

# ocean/atmo 要素名-文件名和原有字段名的映射
variable_map = {
    var_name: MODEL_TO_STAT_MAPPING.get(var_name, var_name)
    for var_name in DEFAULT_MODEL_OCEAN_VARS + DEFAULT_MODEL_ATMO_VARS
}

# ocean, atmo 的分别归集
ocean_keys = list(DEFAULT_MODEL_OCEAN_VARS)
atmo_keys  = list(DEFAULT_MODEL_ATMO_VARS)

# 载入统计量
variables = list(variable_map.values())
stat = {"mean": {}, "std": {}}
default_stat_dir = os.getenv("ORCADL_STAT_DIR", "/mnt/data/zhu.yishun/ORCA-DL-main/stat")
for v in variables:
    mean_path = os.path.join(default_stat_dir, "mean", f"{v}.npy")
    std_path = os.path.join(default_stat_dir, "std", f"{v}.npy")
    if os.path.exists(mean_path) and os.path.exists(std_path):
        stat["mean"][v] = np.load(mean_path)
        stat["std"][v] = np.load(std_path)

month = 0  # 月份选择索引

# 完整网格范围
LAT_MIN, LAT_MAX, LAT_POINTS = -63.5, 63.5, 128
LON_MIN, LON_MAX, LON_POINTS = 0.5, 359.5, 360

# 预计算完整的经纬度数组
FULL_LATS = np.linspace(LAT_MIN, LAT_MAX, LAT_POINTS)
FULL_LONS = np.linspace(LON_MIN, LON_MAX, LON_POINTS)

# 季节定义（月份索引 0-11）
SEASON_MONTHS = {
    'DJF': [11, 0, 1],   # 12月, 1月, 2月
    'MAM': [2, 3, 4],    # 3月, 4月, 5月
    'JJA': [5, 6, 7],    # 6月, 7月, 8月
    'SON': [8, 9, 10]    # 9月, 10月, 11月
}

# Nino3.4区域定义（5°S–5°N, 170°W–120°W -> 190°E–240°E）
NINO34_LAT_MIN, NINO34_LAT_MAX = -5.0, 5.0
NINO34_LON_MIN, NINO34_LON_MAX = 190.0, 240.0  # 经度转换:  170°W=190°E, 120°W=240°E

# 变量描述字典（复用 defitools-img-pred.py）
VAR_DESCRIPTIONS = {
    "so": {
        "name_en": "Salinity",
        "name_cn":  "盐度",
        "unit": "PSU (Practical Salinity Unit)",
        "description": "海水盐度，表示每千克海水中溶解盐类的质量",
        "depth_levels": 16
    },
    "thetao": {
        "name_en": "Potential Temperature",
        "name_cn": "位温",
        "unit": "°C",
        "description": "海水位温，即海水绝热上升到海面时的温度",
        "depth_levels": 16
    },
    "tos": {
        "name_en":  "Sea Surface Temperature",
        "name_cn": "海表面温度",
        "unit":  "°C",
        "description": "海洋表面的温度",
        "depth_levels": 1
    },
    "uo": {
        "name_en": "Zonal Velocity",
        "name_cn": "纬向流速",
        "unit": "m/s",
        "description": "海流在东西方向上的速度分量，正值表示向东",
        "depth_levels": 16
    },
    "vo":  {
        "name_en":  "Meridional Velocity",
        "name_cn": "经向流速",
        "unit": "m/s",
        "description": "海流在南北方向上的速度分量，正值表示向北",
        "depth_levels":  16
    },
    "zos": {
        "name_en": "Sea Surface Height",
        "name_cn": "海面高度",
        "unit":  "m",
        "description":  "海面相对于参考面的高度",
        "depth_levels": 1
    },
    "hfds": {
        "name_en": "Downward Heat Flux",
        "name_cn": "海表向下热通量",
        "unit": "W/m^2",
        "description": "海水表面向下净热通量",
        "depth_levels": 1
    },
    "mlotst": {
        "name_en": "Mixed Layer Depth",
        "name_cn": "混合层深度",
        "unit": "m",
        "description": "海洋混合层厚度",
        "depth_levels": 1
    },
    "rsntds": {
        "name_en": "Net Shortwave Radiation",
        "name_cn": "海表净短波辐射",
        "unit": "W/m^2",
        "description": "海水表面净向下短波辐射",
        "depth_levels": 1
    },
    "sob": {
        "name_en": "Bottom Salinity",
        "name_cn": "海底盐度",
        "unit": "PSU (Practical Salinity Unit)",
        "description": "海底层海水盐度",
        "depth_levels": 1
    },
    "sos": {
        "name_en": "Sea Surface Salinity",
        "name_cn": "海表盐度",
        "unit": "PSU (Practical Salinity Unit)",
        "description": "海洋表层盐度",
        "depth_levels": 1
    },
    "tob": {
        "name_en": "Bottom Potential Temperature",
        "name_cn": "海底位温",
        "unit": "°C",
        "description": "海底层海水位温",
        "depth_levels": 1
    },
    "wfo": {
        "name_en": "Water Flux Into Sea Water",
        "name_cn": "入海水通量",
        "unit": "kg m-2 s-1",
        "description": "进入海水的净淡水通量",
        "depth_levels": 1
    },
    "wo": {
        "name_en": "Vertical Velocity",
        "name_cn": "垂向流速",
        "unit": "m/s",
        "description": "海水垂向速度分量",
        "depth_levels": 16
    }
}

# 全局预测器实例
predictor:  Optional[ORCADLPredictor] = None


# ==================== 辅助函数（复用 defitools-img-pred.py） ====================
def get_lat_lon_indices(lat_min:  float, lat_max: float, lon_min: float, lon_max: float):
    """
    根据经纬度范围获取对应的数组索引
    
    Returns:
        lat_indices: 纬度索引范围 (start, end)
        lon_indices: 经度索引范围 (start, end)
        actual_lats: 实际返回���纬度数组
        actual_lons: 实际返回的经度数组
    """
    lat_start_idx = np.searchsorted(FULL_LATS, lat_min, side='left')
    lat_end_idx = np.searchsorted(FULL_LATS, lat_max, side='right')
    
    lon_start_idx = np.searchsorted(FULL_LONS, lon_min, side='left')
    lon_end_idx = np.searchsorted(FULL_LONS, lon_max, side='right')
    
    lat_start_idx = max(0, lat_start_idx)
    lat_end_idx = min(LAT_POINTS, lat_end_idx)
    lon_start_idx = max(0, lon_start_idx)
    lon_end_idx = min(LON_POINTS, lon_end_idx)
    
    actual_lats = FULL_LATS[lat_start_idx:lat_end_idx]
    actual_lons = FULL_LONS[lon_start_idx:lon_end_idx]
    
    return (lat_start_idx, lat_end_idx), (lon_start_idx, lon_end_idx), actual_lats, actual_lons


def slice_by_lat_lon(data: np.ndarray, lat_indices: tuple, lon_indices: tuple) -> np.ndarray:
    """
    根据经纬度索引切片数据
    
    Args:
        data: 输入数据，最后两个维度是 (lat, lon)
        lat_indices:  (start, end) 纬度索引
        lon_indices: (start, end) 经度索引
    
    Returns:
        切片后的数据
    """
    lat_start, lat_end = lat_indices
    lon_start, lon_end = lon_indices
    
    if data.ndim == 2:
        return data[lat_start:lat_end, lon_start:lon_end]
    elif data.ndim == 3:
        return data[: , lat_start:lat_end, lon_start:lon_end]
    elif data.ndim == 4:
        return data[:, :, lat_start:lat_end, lon_start:lon_end]
    elif data.ndim == 5:
        return data[:, :, :, lat_start:lat_end, lon_start:lon_end]
    else:
        raise ValueError(f"Unsupported data dimension: {data.ndim}")


def sanitize_for_json(arr: np.ndarray) -> list:
    """清理数组中的 inf 和 nan 值，使其 JSON 兼容，并转换为列表"""
    arr = arr.copy()
    arr = np.nan_to_num(arr, nan=-999.0, posinf=1e38, neginf=-1e38)
    return arr.tolist()


def fig_to_base64(fig) -> str:
    """将 matplotlib 图形转换为 base64 编码的 PNG 字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_spatial_plot(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str = 'coolwarm',
    vmin: float = None,
    vmax: float = None
) -> str:
    """
    创建空间分布图并返回 base64 编码
    
    Args:
        data: 2D 数据数组 (lat, lon)
        lats: 纬度数组
        lons: 经度数组
        title: 图形标题
        colorbar_label: colorbar 标签
        cmap: colormap 名称
        vmin, vmax: 颜色范围
    
    Returns: 
        base64 编码的 PNG 图像
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    lon2d, lat2d = np.meshgrid(lons, lats)

    # 保证输入为二维数组
    plot_data = np.array(data, dtype=float)

    # 创建 mask：True 表示为陆地/无效 (NaN)
    mask = np.isnan(plot_data)

    # 使用 masked array 绘制，NaN/掩码位置不会着色
    masked = np.ma.array(plot_data, mask=mask)

    # 使用 imshow 绘制并在海/陆边界上应用透明度渐变以实现柔和边缘
    # 计算显示范围并处理无效数据
    valid = plot_data[~np.isnan(plot_data)]
    if valid.size:
        vmin_eff = vmin if vmin is not None else float(np.nanmin(plot_data))
        vmax_eff = vmax if vmax is not None else float(np.nanmax(plot_data))
    else:
        vmin_eff, vmax_eff = (vmin if vmin is not None else 0.0, vmax if vmax is not None else 1.0)

    # 构建归一化器和 colormap
    norm = mpl.colors.Normalize(vmin=vmin_eff, vmax=vmax_eff)
    cmap_inst = plt.get_cmap(cmap)

    # 生成海洋掩码（1 表示海洋有效点，0 表示陆地/无效）并尝试模糊以得到渐变 alpha
    ocean_mask = (~mask).astype(float)
    try:
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(ocean_mask, sigma=0.05)
        alpha_mask = np.clip(blurred, 0.0, 1.0)
    except Exception:
        # 若没有 scipy，可回退到硬边（无渐变）但不报错
        alpha_mask = ocean_mask

    # imshow 的 extent 需要是 [xmin, xmax, ymin, ymax]
    extent = (lons.min(), lons.max(), lats.min(), lats.max())

    # 使用 imshow 绘制色彩图，并用 alpha_mask 控制边界渐变
    im = ax.imshow(plot_data, origin='lower', extent=extent, cmap=cmap_inst,
                   norm=norm, alpha=alpha_mask, interpolation='bilinear', aspect='auto')
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(colorbar_label)

    # 在掩码边界上绘制黑色等高线以勾勒陆地轮廓
    try:
        # 将布尔掩码转换为整数场，等高线在 0.5 处绘制边界
        contour_field = mask.astype(int)
        # 绘制边界线，linewidths 可调整线宽
        ax.contour(lon2d, lat2d, contour_field, levels=[0.5], colors='k', linewidths=0.6, linestyles='-')
    except Exception:
        # 发生错误时仍然继续返回图像（不阻塞主流程）
        logger.debug("绘制陆地轮廓失败，继续返回基础图像", exc_info=True)

    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(title)
    plt.tight_layout()

    return fig_to_base64(fig)


def create_time_series_plot(
    values: list,
    x_labels: list,
    title: str,
    ylabel: str,
    xlabel: str = 'Prediction Step (Month)'
) -> str:
    """
    创建时间序列图并返回 base64 编码
    
    Args: 
        values: 数值列表
        x_labels: x 轴标签列表
        title: 图形标题
        ylabel: y 轴标签
        xlabel: x 轴标签
    
    Returns:
        base64 编码的 PNG 图像
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(range(len(values)), values, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # 降低 x 轴标签密度以避免重叠
    n = len(x_labels)
    max_labels = 12
    step = max(1, int(np.ceil(n / max_labels)))
    ticks = list(range(0, n, step))
    if ticks[-1] != n - 1:
        ticks.append(n - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([x_labels[i] for i in ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def create_enso_plot(
    anomalies: list,
    x_labels: list,
    title: str,
    ylabel: str
) -> str:
    """
    为 ENSO 绘制异常值并返回 base64 编码
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(range(len(anomalies)), anomalies, 'o-', color='tab:blue', label='Monthly Anomaly')

    # 阈值线
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=-0.5, color='gray', linestyle='--', linewidth=1)

    # 处理 x 轴标签：去掉 'Month' 前缀并减少注释以避免重叠
    cleaned_labels = []
    for lab in x_labels:
        if isinstance(lab, str) and lab.lower().startswith('month'):
            parts = lab.split()
            # 取最后一段作为月份数字或标识
            cleaned_labels.append(parts[-1])
        else:
            cleaned_labels.append(str(lab))

    n = len(cleaned_labels)
    max_labels = 10
    step = max(1, int(np.ceil(n / max_labels)))
    ticks = list(range(0, n, step))
    if ticks[-1] != n - 1:
        ticks.append(n - 1)

    ax.set_xticks(ticks)
    ax.set_xticklabels([cleaned_labels[i] for i in ticks])

    ax.set_xlabel('Prediction Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig_to_base64(fig)


def create_heatwave_plot(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    land_mask: np.ndarray = None
) -> str:
    """
    创建海洋热浪填色图并返回 base64 编码
    
    Args: 
        data: 热浪天数 2D 数据数组 (lat, lon)
        lats: 纬度数组
        lons:  经度数组
        title: 图形标题
    
    Returns:
        base64 编码的 PNG 图像
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lon2d, lat2d = np.meshgrid(lons, lats)
    
    # 若提供 land_mask（True 表示海洋），将陆地点显示为 NaN 以免着色
    plot_data = np.array(data, dtype=float)
    if land_mask is not None:
        try:
            lm = np.array(land_mask)
            if lm.shape == plot_data.shape:
                plot_data = np.where(lm, plot_data, np.nan)
            else:
                # 若形状不一致尝试裁剪/广播
                h = min(lm.shape[0], plot_data.shape[0])
                w = min(lm.shape[1], plot_data.shape[1])
                tmp = np.full_like(plot_data, np.nan)
                tmp[:h, :w] = np.where(lm[:h, :w], plot_data[:h, :w], np.nan)
                plot_data = tmp
        except Exception:
            pass

    # 确保 NaN（陆地）显示为白色，并使用蓝->红配色以增强对比
    cmap = plt.get_cmap('bwr')
    try:
        cmap.set_bad('white')
    except Exception:
        pass
    # 自动选择上限以提高对比（98百分位），若无有效数据则回退到 1
    try:
        valid = plot_data[~np.isnan(plot_data)]
        if valid.size:
            vmax = float(np.nanpercentile(valid, 98))
            if vmax <= 0:
                vmax = float(np.nanmax(valid))
        else:
            vmax = 1.0
    except Exception:
        vmax = None
    pcm = ax.pcolormesh(lon2d, lat2d, plot_data, cmap=cmap, shading='auto', vmin=0, vmax=vmax)
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Marine Heatwave Months')
    
    # 在掩码边界上绘制陆地轮廓
    try:
        # 以 NaN 掩码为基础绘制陆地轮廓，黑线勾勒
        mask = np.isnan(plot_data)
        contour_field = mask.astype(int)
        ax.contour(lon2d, lat2d, contour_field, levels=[0.5], colors='k', linewidths=0.6, linestyles='-')
    except Exception:
        logger.debug('绘制海岸轮廓失败，继续返回基础图像', exc_info=True)

    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def create_decadal_plot(
    annual_values: list,
    start_year: int,
    title: str,
    ylabel: str
) -> str:
    """
    创建年代际演化曲线图并返回 base64 编码
    
    Args:
        annual_values:  年度均值列表
        start_year: 起始年份
        title: 图形标题
        ylabel:  y 轴标签
    
    Returns:
        base64 编码的 PNG 图像
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    years = [start_year + i for i in range(len(annual_values))]
    
    ax.plot(years, annual_values, 'b-o', linewidth=2, markersize=6)
    
    # 添加趋势线
    if len(annual_values) > 1:
        z = np.polyfit(range(len(annual_values)), annual_values, 1)
        p = np.poly1d(z)
        ax.plot(years, p(range(len(annual_values))), 'r--', linewidth=1.5, 
                label=f'Trend: {z[0]:.4f}/year')
        ax.legend()
    # 减少 x 轴年份标签密度以避免重叠
    n = len(years)
    max_labels = 12
    step = max(1, int(np.ceil(n / max_labels)))
    ticks_idx = list(range(0, n, step))
    if ticks_idx[-1] != n - 1:
        ticks_idx.append(n - 1)
    ax.set_xticks([years[i] for i in ticks_idx])
    ax.set_xticklabels([str(years[i]) for i in ticks_idx])
    
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def load_example_for_key(key: str):
    """
    从 `example_dir` 中加载给定要素的示例数据，兼容 .nc 和 .npy 文件。

    返回 (varname_in_file, ndarray) 或 (None, None) 表示未找到。

    规则：
    - 优先尝试 example_dir/{mapped_name}.nc，然后 example_dir/{key}.nc
    - 如果没有 .nc，则尝试 {mapped_name}.npy 与 {key}.npy
    - 对于 2D 数据，会自动在最前面增加一维以变为 (depth, lat, lon)
    """
    fname = variable_map.get(key, key)

    # 尝试 NetCDF 文件
    nc_paths = [os.path.join(example_dir, f"{fname}.nc"), os.path.join(example_dir, f"{key}.nc")]
    for p in nc_paths:
        if os.path.exists(p):
            try:
                varname_in_ds = None
                arr = None

                if xr is not None:
                    with xr.open_dataset(p) as ds:
                        varname_in_ds = fname if fname in ds.data_vars else (key if key in ds.data_vars else None)
                        if varname_in_ds is None and len(ds.data_vars) > 0:
                            varname_in_ds = list(ds.data_vars.keys())[0]
                        if varname_in_ds is not None:
                            arr = ds[varname_in_ds].values

                if arr is None and h5py is not None:
                    with h5py.File(p, "r") as handle:
                        datasets = []

                        def visitor(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                datasets.append((name, obj))

                        handle.visititems(visitor)
                        selected = None
                        for candidate in (fname, key):
                            for name, dataset in datasets:
                                if name.split("/")[-1] == candidate:
                                    selected = (candidate, dataset)
                                    break
                            if selected is not None:
                                break
                        if selected is None:
                            for name, dataset in datasets:
                                if dataset.ndim >= 2:
                                    selected = (name.split("/")[-1], dataset)
                                    break
                        if selected is not None:
                            varname_in_ds, dataset = selected
                            arr = np.asarray(dataset[()])

                if arr is None or varname_in_ds is None:
                    raise ValueError("No readable data variable found")
                if arr.ndim == 2:
                    arr = arr[None]
                while arr.ndim > 3:
                    arr = arr[0]
                return varname_in_ds, arr.astype(np.float32)
            except Exception as e:
                logger.warning(f"加载示例 NetCDF 失败 {p} for key {key}: {e}")
                continue

    # 尝试 NPY 文件
    npy_paths = [os.path.join(example_dir, f"{fname}.npy"), os.path.join(example_dir, f"{key}.npy")]
    for p in npy_paths:
        if os.path.exists(p):
            try:
                arr = np.load(p)
                if arr.ndim == 2:
                    arr = arr[None]
                return fname, arr.astype(np.float32)
            except Exception as e:
                logger.warning(f"加载示例 NPY 失败 {p} for key {key}: {e}")
                continue

    logger.warning(f"未找到示例文件（.nc/.npy） for key {key}: checked {nc_paths + npy_paths}")
    return None, None


def build_zero_example_for_key(key: str):
    channels = get_var_channels(key)
    return np.zeros((channels, LAT_POINTS, LON_POINTS), dtype=np.float32)


def load_example_or_default(key: str):
    varname, arr = load_example_for_key(key)
    if arr is None:
        logger.warning(f"使用零值示例回填缺失变量: {key}")
        return variable_map.get(key, key), build_zero_example_for_key(key)
    return varname, arr.astype(np.float32)
    return None, None


# ==================== Pydantic 请求/响应模型 ====================
class SeasonEnum(str, Enum):
    DJF = "DJF"
    MAM = "MAM"
    JJA = "JJA"
    SON = "SON"


class VarNameEnum(str, Enum):
    so = "so"
    thetao = "thetao"
    tos = "tos"
    uo = "uo"
    vo = "vo"
    zos = "zos"
    hfds = "hfds"
    mlotst = "mlotst"
    rsntds = "rsntds"
    sob = "sob"
    sos = "sos"
    tob = "tob"
    wfo = "wfo"
    wo = "wo"


# 聚合方式：按季节/按年/按月
class AggregationMode(str, Enum):
    season = "season"
    year = "year"
    month = "month"


# 功能1: 季节平均态模拟可视化 - 请求模型
class SeasonalMeanRequest(BaseModel):
    lat_min: float = Field(..., ge=-63.5, le=63.5, description="纬度最小值（-63.5~63.5）")
    lat_max: float = Field(..., ge=-63.5, le=63.5, description="纬度最大值（-63.5~63.5，需>lat_min）")
    lon_min: float = Field(..., ge=0.5, le=359.5, description="经度最小值（0.5~359.5）")
    lon_max: float = Field(..., ge=0.5, le=359.5, description="经度最大值（0.5~359.5，需>lon_min）")
    var_name: VarNameEnum = Field(..., description="变量名（tos/zos/thetao等）")
    season: Optional[SeasonEnum] = Field(None, description="季节标识（DJF/MAM/JJA/SON），仅当 aggregation_mode='season' 时必须提供")
    predict_steps: int = Field(..., ge=1, description="预测步长（>=1，对应月份数）")

    start_month: int = Field(1, ge=1, le=12, description="起始月份（1-12）")
    aggregation_mode: AggregationMode = Field(AggregationMode.season, description="聚合方式：'season'|'year'|'month'，默认 'season'")
    occurrence_index: int = Field(1, ge=1, description="选择第几个该聚合周期（1表示第一个匹配的季节/年度/月份）")
    # 当 aggregation_mode == 'month' 时，必须指定要聚合的日历月（1-12）
    month: Optional[int] = Field(None, ge=1, le=12, description="仅当 aggregation_mode='month' 时使用，指定要聚合的日历月（1-12）")
    # 指定要可视化的深度层索引（0 为表层）。若变量无深度维则忽略
    depth_index: int = Field(0, ge=0, description="深度层索引，0为表层（可选）")

    @validator('lat_max')
    def lat_max_greater_than_min(cls, v, values):
        if 'lat_min' in values and v <= values['lat_min']:
            raise ValueError('lat_max must be greater than lat_min')
        return v

    @validator('lon_max')
    def lon_max_greater_than_min(cls, v, values):
        if 'lon_min' in values and v <= values['lon_min']:
            raise ValueError('lon_max must be greater than lon_min')
        return v

    @validator('month')
    def month_required_for_month_mode(cls, v, values):
        mode = values.get('aggregation_mode')
        if mode == AggregationMode.month and v is None:
            raise ValueError("aggregation_mode='month' 时必须提供字段 month (1-12)")
        return v

    @validator('season')
    def season_required_for_season_mode(cls, v, values):
        mode = values.get('aggregation_mode')
        if mode == AggregationMode.season and v is None:
            raise ValueError("aggregation_mode='season' 时必须指定 season")
        return v


# 功能1: 季节平均态模拟可视化 - 响应模型
class SeasonalMeanResponse(BaseModel):
    attribute_matrix: list = Field(..., description="季节平均态变量值矩阵，维度：[纬度数, 经度数]")
    attribute_description: str = Field(..., description="变量属性描述信息")
    visual_base64: str = Field(..., description="base64编码的PNG格式空间分布图字符串")


# 功能2: ENSO预测可视化 - 请求模型
class ENSOForecastRequest(BaseModel):
    predict_steps: int = Field(..., ge=1, description="预测步长（>=1，对应月份数）")
    start_month: int = Field(..., ge=1, le=12, description="起始月份（1~12）")
    # 相对于 start_month 的偏移，用于指定关注窗口 [start_month+start_offset, start_month+end_offset]
    start_offset: int = Field(0, ge=0, description="起始偏移（月），相对于 start_month 的月数，>=0")
    end_offset: Optional[int] = Field(None, ge=0, description="结束偏移（月），相对于 start_month 的月数，需>= start_offset；若为 null 则默认为 predict_steps-1")
    
    # 注意：取消上传 ocean_data/atmo_data 的功能，接口不再接收用户上传的数据。
    # 基线由模型预测优先构建，所有输入数据均从服务器端示例数据加载。


# 功能2: ENSO预测可视化 - 响应模型
class ENSOForecastResponse(BaseModel):
    attribute_matrix: list = Field(..., description="Nino3.4指数时间序列矩阵，维度：[预测步长]")
    attribute_description: str = Field(..., description="Nino3.4指数属性描述信息")
    visual_base64: str = Field(..., description="base64编码的PNG格式时间序列图字符串")
    event_labels: list = Field(..., description="每步的ENSO事件标签列表（'El Nino'/'La Nina'/'Neutral'）")


# 功能3: 海洋热浪可视化 - 请求模型
class MarineHeatwaveRequest(BaseModel):
    lat_min: float = Field(..., ge=-63.5, le=63.5, description="纬度最小值（-63.5~63.5）")
    lat_max: float = Field(..., ge=-63.5, le=63.5, description="纬度最大值（-63.5~63.5，需>lat_min）")
    lon_min: float = Field(..., ge=0.5, le=359.5, description="经度最小值（0.5~359.5）")
    lon_max: float = Field(..., ge=0.5, le=359.5, description="经度最大值（0.5~359.5，需>lon_min）")
    start_month: int = Field(..., ge=1, le=12, description="参考起始月份（1~12），作为相对偏移的基准")
    end_month: Optional[int] = Field(None, ge=1, le=12, description="兼容字段：结束月份（1~12），旧版本使用；当前建议使用 start_offset/end_offset（可选）")
    # 新增字段：相对于 start_month 的偏移（单位：月），用于选择可视化时间窗 [start_month+start_offset, start_month+end_offset]
    start_offset: int = Field(0, ge=0, description="起始偏移（月），相对于 start_month 的月数，>=0")
    end_offset: int = Field(11, ge=0, description="结束偏移（月），相对于 start_month 的月数，需>= start_offset")
    # 不再强制上传气候均值；若用户不提供阈值，则使用服务器端预计算的每月90percentile文件
    mh_clim_mean_values: Optional[Union[List[float], List[List[List[float]]]]] = Field(
        None,
        description="可选：12个数值的列表（对应1-12月）的气候均值（℃）；网格级可提供12组数组"
    )
    mh_threshold_values: Optional[Union[List[float], List[List[List[float]]]]] = Field(
        None,
        description="可选：12个数值的列表（对应1-12月）的90百分位阈值；网格级可提供12组数组。若不提供，接口将使用服务器端预计算文件 tos_p90_monthly_1993_2023.npy"
    )
    
    # 注意：取消上传 ocean_data/atmo_data 的功能，接口不再接收用户上传的数据。
    # 所有输入数据均从服务器端示例数据加载。
    
    @validator('lat_max')
    def lat_max_greater_than_min(cls, v, values):
        if 'lat_min' in values and v <= values['lat_min']:
            raise ValueError('lat_max must be greater than lat_min')
        return v
    
    @validator('lon_max')
    def lon_max_greater_than_min(cls, v, values):
        if 'lon_min' in values and v <= values['lon_min']: 
            raise ValueError('lon_max must be greater than lon_min')
        return v
    
    @validator('end_month')
    def end_month_greater_than_start(cls, v, values):
        # 保持向后兼容：end_month 现在为可选，仅在提供时进行校验
        if v is None:
            return v
        if 'start_month' in values and v < values['start_month']:
            raise ValueError('end_month must be >= start_month')
        return v

    @validator('end_offset')
    def end_offset_greater_than_start_offset(cls, v, values):
        if 'start_offset' in values and v < values['start_offset']:
            raise ValueError('end_offset must be >= start_offset')
        return v
    
    @validator('mh_clim_mean_values', 'mh_threshold_values')
    def validate_mh_values(cls, v):
        if v is None:
            return v
        if isinstance(v, list):
            if len(v) != 12:
                raise ValueError('Must contain exactly 12 values/arrays')
        return v


# 功能3: 海洋热浪可视化 - 响应模型
class MarineHeatwaveResponse(BaseModel):
    attribute_matrix: list = Field(..., description="热浪总月数矩阵，维度：[纬度数, 经度数]")
    attribute_description:  str = Field(..., description="海洋热浪属性描述信息")
    visual_base64: str = Field(..., description="base64编码的PNG格式填色图字符串")


# 功能4: 年代际预测可视化 - 请求模型
class DecadalPredictionRequest(BaseModel):
    lat_min: float = Field(..., ge=-63.5, le=63.5, description="纬度最小值（-63.5~63.5）")
    lat_max: float = Field(..., ge=-63.5, le=63.5, description="纬度最大值（-63.5~63.5，需>lat_min）")
    lon_min: float = Field(..., ge=0.5, le=359.5, description="经度最小值（0.5~359.5）")
    lon_max: float = Field(..., ge=0.5, le=359.5, description="经度最大值（0.5~359.5，需>lon_min）")
    var_name: VarNameEnum = Field(..., description="变量名（tos/zos/thetao等）")
    predict_steps: int = Field(..., ge=1, description="预测步数：当 output_granularity 为 'month' 时单位为月；当为 'year' 时单位为年（每年12个月），>=1")
    
    # 注意：取消上传 ocean_data/atmo_data 的功能，接口不再接收用户上传的数据。
    # 所有输入数据均从服务器端示例数据加载。
    start_month: int = Field(1, ge=1, le=12, description="起始月份（1-12）")
    start_year: int = Field(2020, description="起始年份（用于图表显示）")
    # 输出粒度：'month' 表示返回逐月序列；'year' 表示按年聚合（默认，保持向后兼容）
    output_granularity: str = Field('year', description="输出粒度：'month' 或 'year'，默认 'year'")
    # 指定想要可视化的深度层索引（0 为表层）。若变量无深度维则忽略此字段。
    depth_index: int = Field(0, ge=0, description="深度层索引，0为表层（可选）")
    
    @validator('lat_max')
    def lat_max_greater_than_min(cls, v, values):
        if 'lat_min' in values and v <= values['lat_min']:
            raise ValueError('lat_max must be greater than lat_min')
        return v
    
    @validator('lon_max')
    def lon_max_greater_than_min(cls, v, values):
        if 'lon_min' in values and v <= values['lon_min']:
            raise ValueError('lon_max must be greater than lon_min')
        return v

    @validator('output_granularity')
    def validate_output_granularity(cls, v):
        if v not in ('month', 'year'):
            raise ValueError("output_granularity must be 'month' or 'year'")
        return v


# 功能4: 年代际预测可视化 - 响应模型
class DecadalPredictionResponse(BaseModel):
    attribute_matrix:  list = Field(..., description="年度均值矩阵，维度：[预测年数]")
    attribute_description: str = Field(..., description="年代际预测属性描述���息")
    visual_base64: str = Field(..., description="base64编码的PNG格式演化曲线图字符串")


# 错误响应模型
class ErrorResponse(BaseModel):
    error: str = Field(..., description="错误类型")
    detail: str = Field(..., description="错误详细描述")


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="ORCA-DL Ocean Prediction Visualization API",
    description="海洋预测可视化API服务，提供季节平均态、ENSO预测、海洋热浪、年代际预测四类功能",
    version="1.0.0",
    openapi_tags=[
        {"name": "seasonal", "description": "季节平均态模拟可视化"},
        {"name": "enso", "description": "ENSO季节性预测可视化"},
        {"name": "heatwave", "description": "海洋热浪可视化"},
        {"name": "decadal", "description": "年代际预测可视化"}
    ]
)


@app.on_event("startup")
async def startup():
    """启动时加载模型"""
    global predictor
    try:
        logger.info("正在加载 ORCA-DL 预测器...")
        predictor = ORCADLPredictor(
            config_path=os.getenv("ORCADL_CONFIG_PATH", os.path.join(REPO_ROOT, "model_config.json")),
            checkpoint_path=os.getenv("ORCADL_CHECKPOINT_PATH", os.path.join(REPO_ROOT, "output-ultra", "pytorch_model.bin")),
            stat_dir=os.getenv("ORCADL_STAT_DIR", "/mnt/data/zhu.yishun/ORCA-DL-main/stat")
        )
        logger.info("ORCA-DL 预测器加载成功")
    except Exception as e:
        logger.error(f"加载预测器失败: {str(e)}")
        predictor = None


@app.get("/health")
async def health():
    """健康检查接口"""
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "model_loaded": predictor is not None
    }


# ==================== 功能1: 季节平均态模拟可视化 ====================
@app.post(
    "/seasonal_mean_visual",
    operation_id="seasonal_mean_visual",
    response_model=SeasonalMeanResponse,
    responses={
        400: {"model": ErrorResponse, "description": "参数错误"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    },
    tags=["seasonal"],
    summary="季节平均态模拟可视化",
    description="根据指定季节和经纬度范围，返回季节平均态的空间分布可视化结果"
)
async def seasonal_mean_visual(request: SeasonalMeanRequest):
    """
    季节平均态模拟可视化接口
    
    - 接收参数后，将原始数据归一化，调用模型获取指定季节对应月份的预测数据
    - 将模型输出数据反归一化，提取目标季节月度数据，计算算术平均
    - 切片指定经纬度范围内的平均数据，生成空间分布图
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="模型未加载，请稍后重试")
    
    try:
        # 参数校验
        var_name = request.var_name.value
        season = request.season.value
        
        logger.info(f"[季节平均态] 变量: {var_name}, 季节: {season}, 预测步长: {request.predict_steps}")
        
        # 校验变量名是否在映射中
        if var_name not in ORCADLPredictor.MODEL_TO_STAT_MAPPING:
            raise HTTPException(
                status_code=400,
                detail=f"变量名 {var_name} 不在支持的变量列表中:  {list(ORCADLPredictor.MODEL_TO_STAT_MAPPING.keys())}"
            )
        
        # 获取经纬度索引
        lat_indices, lon_indices, actual_lats, actual_lons = get_lat_lon_indices(
            request.lat_min, request.lat_max, request.lon_min, request.lon_max
        )
        
        logger.info(f"[季节平均态] 经纬度范围: lat[{lat_indices}], lon[{lon_indices}]")
        
        # 准备输入数据：始终从服务器端示例数据加载（取消上传数据功能）
        ocean_data = {}
        atmo_data = {}

        # 使用通用加载器，兼容 .nc 和 .npy
        for key in ocean_keys:
            varname, arr = load_example_or_default(key)
            # 保持原实现：季节接口使用映射后的变量名作为字典键
            ocean_data[varname] = arr

        for key in atmo_keys:
            varname, arr = load_example_or_default(key)
            atmo_data[varname] = arr
        
        # 调用模型预测
        logger.info(f"[季节平均态] 开始归一化输入数据...")

        # 转换用户输入月份 1-12 -> 内部 0-11
        start_month_idx = request.start_month - 1

        # 使用 predict_ocean_state 进行预测（内部会调用 _normalize_data 和 _denormalize_data）
        prediction = predict_ocean_state(
            predictor=predictor,
            ocean_data=ocean_data,
            atmo_data=atmo_data,
            start_month=start_month_idx,
            predict_steps=request.predict_steps,
            denormalize=True,  # 确保输出已反归一化
            use_model_naming=False,
            normalized=True
        )
        
        logger.info(f"[季节平均态] 模型预测完成，输出形状: {prediction.metadata['output_shape']}")
        
        # 根据用户选择的聚合方式提取并计算平均
        season_months = SEASON_MONTHS[season]
        results_by_step = prediction.metadata.get('results_by_step', {})

        agg_mode = request.aggregation_mode
        occ_idx = int(request.occurrence_index)

        # 预计算每一步对应的日历月（保持与原实现一致的偏移）
        months_seq = [ (start_month_idx + step + 1) % 12 for step in range(request.predict_steps) ]

        def _get_var_from_step(step_idx):
            step_data = results_by_step.get(step_idx, {})
            if var_name not in step_data:
                return None
            var_data = step_data[var_name]
            # 若存在深度/通道维，用请求中的 depth_index 选择对应层
            if var_data.ndim > 2:
                di = int(getattr(request, 'depth_index', 0) or 0)
                if var_data.ndim == 4:  # (B, C, H, W)
                    _, C, _, _ = var_data.shape
                    di = min(di, C - 1)
                    var_data = var_data[0, di]
                elif var_data.ndim == 3:
                    # 3D 可能为 (C, H, W) 或 (B, H, W)
                    if var_data.shape[0] > 1:
                        di = min(di, var_data.shape[0] - 1)
                        var_data = var_data[di]
                    else:
                        var_data = var_data[0]
            return var_data

        seasonal_mean = None

        if agg_mode == AggregationMode.season:
            # 查找完整的季节窗口（按顺序匹配 season_months）
            S = season_months
            starts = []
            for s in range(0, max(0, request.predict_steps - len(S) + 1)):
                if months_seq[s:s+len(S)] == S:
                    starts.append(s)
            if starts:
                if occ_idx > len(starts):
                    raise HTTPException(status_code=400, detail=f"请求的 occurrence_index 超出可用季节次数: {len(starts)}")
                start = starts[occ_idx - 1]
                mats = []
                for step in range(start, start + len(S)):
                    vd = _get_var_from_step(step)
                    if vd is not None:
                        mats.append(vd)
                if not mats:
                    raise HTTPException(status_code=500, detail="选定季节窗口内无有效预测数据")
                seasonal_mean = np.mean(np.stack(mats, axis=0), axis=0)
                logger.info(f"[季节平均态] 按季节-第{occ_idx}次聚合完成，数据形状: {seasonal_mean.shape}")
            else:
                # 若找不到完整的季节窗口, 回退为取所有匹配季节的月份并等权平均（兼容旧行为）
                mats = []
                for step in range(request.predict_steps):
                    if months_seq[step] in S:
                        vd = _get_var_from_step(step)
                        if vd is not None:
                            mats.append(vd)
                if mats:
                    seasonal_mean = np.mean(np.stack(mats, axis=0), axis=0)
                    logger.warning(f"[季节平均态] 未找到完整季节窗口，回退为对所有匹配月份等权平均，样本数={len(mats)}")
                else:
                    logger.warning(f"[季节平均态] 未找到季节 {season} 对应的任何月份数据，使用第一步预测")
                    vd = _get_var_from_step(0)
                    seasonal_mean = vd if vd is not None else np.zeros((LAT_POINTS, LON_POINTS))

        elif agg_mode == AggregationMode.year:
            # 将预测按连续的12个月块划分，第 k 年对应 steps [(k-1)*12, k*12-1]
            start_idx = (occ_idx - 1) * 12
            end_idx = start_idx + 12
            if end_idx > request.predict_steps:
                raise HTTPException(status_code=400, detail=f"请求的第 {occ_idx} 年超出预测步长，可用完整年数={request.predict_steps // 12}")
            mats = []
            for step in range(start_idx, end_idx):
                vd = _get_var_from_step(step)
                if vd is not None:
                    mats.append(vd)
            if not mats:
                raise HTTPException(status_code=500, detail="选定年度窗口内无有效预测数据")
            seasonal_mean = np.mean(np.stack(mats, axis=0), axis=0)
            logger.info(f"[季节平均态] 按年-第{occ_idx}次聚合完成，数据形状: {seasonal_mean.shape}")

        elif agg_mode == AggregationMode.month:
            # 仅取指定的日历月的第 occ_idx 次出现
            target_month = request.month
            if target_month is None:
                raise HTTPException(status_code=400, detail="aggregation_mode='month' 时必须提供 month 字段")
            indices = [i for i, m in enumerate(months_seq) if m == target_month]
            if len(indices) < occ_idx:
                raise HTTPException(status_code=400, detail=f"请求的 occurrence_index 超出可用该月次数: {len(indices)}")
            step = indices[occ_idx - 1]
            vd = _get_var_from_step(step)
            if vd is None:
                raise HTTPException(status_code=500, detail="选定月份无有效预测数据")
            seasonal_mean = vd
            logger.info(f"[季节平均态] 按月-月{target_month+1} 第{occ_idx}次聚合，使用 step={step}")

        else:
            raise HTTPException(status_code=400, detail=f"未知的 aggregation_mode: {agg_mode}")
        
        # 切片指定经纬度范围
        sliced_data = slice_by_lat_lon(seasonal_mean, lat_indices, lon_indices)
        logger.info(f"[季节平均态] 切片完成，数据形状: {sliced_data.shape}")
        
        # 生成可视化图形
        var_desc = VAR_DESCRIPTIONS.get(var_name, {})
        # 标题不包含季节/年份/月份描述，仅显示变量名与均值说明
        title = f"{var_desc.get('name_en', var_name)} - Mean"

        colorbar_label = f"{var_desc.get('name_en', var_name)} ({var_desc.get('unit', '')})"

        visual_base64 = create_spatial_plot(
            data=sliced_data,
            lats=actual_lats,
            lons=actual_lons,
            title=title,
            colorbar_label=colorbar_label
        )
        
        # 构建更详细的属性描述：包含反归一化说明、单位、聚合信息、输出经纬度范围与深度信息
        try:
            lat_min_out = float(actual_lats.min()) if getattr(actual_lats, 'size', 0) else float(request.lat_min)
            lat_max_out = float(actual_lats.max()) if getattr(actual_lats, 'size', 0) else float(request.lat_max)
            lon_min_out = float(actual_lons.min()) if getattr(actual_lons, 'size', 0) else float(request.lon_min)
            lon_max_out = float(actual_lons.max()) if getattr(actual_lons, 'size', 0) else float(request.lon_max)
        except Exception:
            lat_min_out, lat_max_out = request.lat_min, request.lat_max
            lon_min_out, lon_max_out = request.lon_min, request.lon_max

        try:
            if hasattr(request, 'depth_index') and request.depth_index is not None:
                depth_info = f"depth_index={int(request.depth_index)}"
            else:
                dl = VAR_DESCRIPTIONS.get(var_name, {}).get('depth_levels', 1)
                depth_info = 'surface' if dl == 1 else 'depth_index=0 (default surface layer)'
        except Exception:
            depth_info = 'unknown'

        # 根据聚合方式生成时间/含义描述，避免在未启用 season 时仍显示季节信息
        if agg_mode == AggregationMode.season:
            months_str = '/'.join([str(m+1) for m in season_months])
            time_desc = f"{season}季节（包含月份：{months_str}）的算术平均（已反归一化）"
        elif agg_mode == AggregationMode.year:
            time_desc = f"按年聚合：第{occ_idx}个完整年（每年12个月）的算术平均（已反归一化）"
        elif agg_mode == AggregationMode.month:
            m = getattr(request, 'month', None)
            if m is not None:
                time_desc = f"按月聚合：日历月 {m} 的第{occ_idx}次出现的值（已反归一化）"
            else:
                time_desc = f"按月聚合：第{occ_idx}次出现的该月的值（已反归一化）"
        else:
            time_desc = f"时间窗口：起始月（模型参考）={request.start_month}，预测步长={request.predict_steps}月（已反归一化）"

        attr_desc = (
            f"变量英文名：{var_desc.get('name_en', var_name)}；"
            f"中文名：{var_desc.get('name_cn', '')}；"
            f"单位：{var_desc.get('unit', '')}；"
            f"含义：{time_desc}；"
            f"输出经纬度范围：纬度 {lat_min_out:.2f}°–{lat_max_out:.2f}°，经度 {lon_min_out:.2f}°–{lon_max_out:.2f}°；"
            f"深度信息：{depth_info}；"
            f"维度说明：[纬度数({len(actual_lats)})×经度数({len(actual_lons)})，已切片为指定经纬度范围]"
        )
        
        logger.info(f"[季节平均态] 处理完成")
        
        return SeasonalMeanResponse(
            attribute_matrix=sanitize_for_json(sliced_data),
            attribute_description=attr_desc,
            visual_base64=visual_base64
        )
        
    except HTTPException: 
        raise
    except Exception as e: 
        logger.error(f"[季节平均态] 处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


# ==================== 功能2: 季节性预测（ENSO）可视化 ====================
@app.post(
    "/enso_forecast_visual",
    operation_id="enso_forecast_visual",
    response_model=ENSOForecastResponse,
    responses={
        400: {"model": ErrorResponse, "description": "参数错误"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    },
    tags=["enso"],
    summary="ENSO季节性预测可视化",
    description="预测Nino3.4指数的时间序列变化，返回可视化结果。基线按预测优先策略构建：每步使用最近三个月的模型预测值为主。接口不接收上传的 ocean_data/atmo_data。"
)
async def enso_forecast_visual(request: ENSOForecastRequest):
    """
    ENSO季节性预测可视化接口
    
    - 将原始tos数据归一化，调用模型循环预测指定步长的tos数据
    - 将模型输出的tos数据反归一化，提取Nino3.4区域的tos数据
    - 计算月度区域平均，基于预测优先构建的基线计算Nino3.4指数（不接受用户自定义气候均值）
    - 生成时间序列可视化图
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="模型未加载，请稍后重试")
    
    try:
        # 转换月份（1-12 -> 0-11）
        start_month_idx = request.start_month - 1

        # 解析相对偏移（start_offset / end_offset），end_offset 若为 None 则默认为 predict_steps-1
        start_offset = int(getattr(request, 'start_offset', 0))
        end_offset = getattr(request, 'end_offset', None)
        if end_offset is None:
            end_offset = request.predict_steps - 1
        else:
            end_offset = int(end_offset)

        if end_offset < start_offset:
            raise HTTPException(status_code=400, detail="end_offset must be >= start_offset")

        # 窗口步数（用于生成最终 ENSO 序列）和传给模型的预测步数
        window_steps = end_offset - start_offset + 1
        predict_steps_model = end_offset + 1

        logger.info(f"[ENSO预测] 起始月份: {request.start_month}, 请求预测步长: {request.predict_steps}, start_offset={start_offset}, end_offset={end_offset}, model_predict_steps={predict_steps_model}")
        
        # 获取Nino3.4区域的经纬度索引
        nino_lat_indices, nino_lon_indices, _, _ = get_lat_lon_indices(
            NINO34_LAT_MIN, NINO34_LAT_MAX, NINO34_LON_MIN, NINO34_LON_MAX
        )
        
        logger.info(f"[ENSO预测] Nino3.4区域索引: lat{nino_lat_indices}, lon{nino_lon_indices}")
        
        # 准备输入数据：始终从服务器端示例数据加载（取消上传数据功能）
        ocean_data = {}
        atmo_data = {}

        # 使用通用加载器，优先 .nc，再尝试 .npy；保留原有字典键为短名
        for key in ocean_keys:
            _, arr = load_example_or_default(key)
            ocean_data[key] = arr

        for key in atmo_keys:
            _, arr = load_example_or_default(key)
            atmo_data[key] = arr

        # 基于示例数据的 tos (sst) 构建海洋有效点掩码 (True 表示海洋)，用于在计算区域均值时排除陆地
        ocean_mask_full = None
        try:
            tos_example = ocean_data.get('tos', None)
            if tos_example is not None:
                tos_arr = np.array(tos_example)
                if tos_arr.ndim == 3:
                    tos_arr_2d = tos_arr[0]
                elif tos_arr.ndim == 2:
                    tos_arr_2d = tos_arr
                else:
                    tos_arr_2d = tos_arr.reshape((LAT_POINTS, LON_POINTS))
                ocean_mask_full = np.isfinite(tos_arr_2d)
            else:
                ocean_mask_full = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)
        except Exception:
            ocean_mask_full = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)
        
        # 基于示例数据的 tos (sst) 构建海洋有效点掩码 (True 表示海洋，可用于排除陆地)
        ocean_mask_full = None
        try:
            tos_example = ocean_data.get('tos', None)
            if tos_example is not None:
                tos_arr = np.array(tos_example)
                if tos_arr.ndim == 3:
                    tos_arr_2d = tos_arr[0]
                elif tos_arr.ndim == 2:
                    tos_arr_2d = tos_arr
                else:
                    tos_arr_2d = tos_arr.squeeze()
                ocean_mask_full = np.isfinite(tos_arr_2d)
            else:
                ocean_mask_full = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)
        except Exception:
            ocean_mask_full = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)

        # 基于示例数据的 tos (sst) 构建海洋有效点掩码 (True 表示海洋，可用于排除陆地)
        ocean_mask_full = None
        try:
            tos_example = ocean_data.get('tos', None)
            if tos_example is not None:
                tos_arr = np.array(tos_example)
                if tos_arr.ndim == 3:
                    tos_arr_2d = tos_arr[0]
                elif tos_arr.ndim == 2:
                    tos_arr_2d = tos_arr
                else:
                    tos_arr_2d = tos_arr.squeeze()
                ocean_mask_full = np.isfinite(tos_arr_2d)
            else:
                ocean_mask_full = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)
        except Exception:
            ocean_mask_full = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)
        
        # 调用模型预测
        logger.info(f"[ENSO预测] 开始归一化输入数据并预测...")
        
        prediction = predict_ocean_state(
            predictor=predictor,
            ocean_data=ocean_data,
            atmo_data=atmo_data,
            start_month=start_month_idx,
            predict_steps=predict_steps_model,
            denormalize=True,
            use_model_naming=True,
            normalized=True
        )
        
        logger.info(f"[ENSO预测] 模型预测完成，输出形状: {prediction.metadata['output_shape']}")
        
        # 提取Nino3.4区域的tos数据并计算异常（后续将计算滑动平均用于事件判定）
        results_by_step = prediction.metadata.get('results_by_step', {})
        nino_means = []
        anomalies = []
        month_labels = []
        steps_used = []

        # 计算基线：始终以模型预测的当前步及后两步（若可得）为主。
        # 已取消上传功能，不再使用请求中上传的数据作为补充基线
        obs_nino_mean = None

        # 仅遍历用户指定的相对窗口 [start_offset, end_offset]
        for step in range(start_offset, end_offset + 1):
            pred_month_idx = (start_month_idx + step + 1) % 12  # 预测月份索引（0-11）
            pred_month = pred_month_idx + 1  # 预测月份（1-12）

            step_data = results_by_step.get(step, {})
            if 'tos' not in step_data:
                continue

            tos_data = step_data['tos']
            if len(tos_data.shape) > 2:
                if len(tos_data.shape) == 4:
                    tos_data = tos_data[0, 0]
                elif len(tos_data.shape) == 3:
                    tos_data = tos_data[0]

            # 当前步的区域平均
            nino_tos = slice_by_lat_lon(tos_data, nino_lat_indices, nino_lon_indices)
            nino_mean = float(np.nanmean(nino_tos))
            nino_means.append(nino_mean)

            # 取未来一步（step+1）的预测作为三个月窗口的第二个或第三个样本（若不存在则重复当前）
            next_mean = None
            next2_mean = None
            # step+1
            s1 = results_by_step.get(step + 1, {})
            if 'tos' in s1:
                t1 = s1['tos']
                if len(t1.shape) > 2:
                    if len(t1.shape) == 4:
                        t1 = t1[0, 0]
                    elif len(t1.shape) == 3:
                        t1 = t1[0]
                n1 = float(np.nanmean(slice_by_lat_lon(t1, nino_lat_indices, nino_lon_indices)))
                next_mean = n1

            # step+2
            s2 = results_by_step.get(step + 2, {})
            if 'tos' in s2:
                t2 = s2['tos']
                if len(t2.shape) > 2:
                    if len(t2.shape) == 4:
                        t2 = t2[0, 0]
                    elif len(t2.shape) == 3:
                        t2 = t2[0]
                n2 = float(np.nanmean(slice_by_lat_lon(t2, nino_lat_indices, nino_lon_indices)))
                next2_mean = n2

                # 组装三个月样本：优先使用当前预测与后两步预测；预测不足时重复最近可用值
            samples = [nino_mean]

            # 补充 step+1：优先使用预测，若不可用则重复最近可用值
            if next_mean is not None:
                samples.append(next_mean)
            else:
                samples.append(samples[-1])

            # 补充 step+2：优先使用预测，若不可用则重复最近可用值
            if next2_mean is not None:
                samples.append(next2_mean)
            else:
                samples.append(samples[-1])

            clim_mean = float(np.nanmean(np.array(samples, dtype=float)))
            anomaly = nino_mean - clim_mean
            anomalies.append(float(anomaly))
            # 记录被使用的模型 step（用于后续绘图时标注年份）
            steps_used.append(step)
            # 使用窗口内的步序号作为备用 x 轴标签（Step 1..window_steps），不显示月份
            month_labels.append(str(step - start_offset + 1))

            logger.info(f"[ENSO预测] Step {step+1}, 月份 {pred_month}, 区域平均: {nino_mean:.3f}℃, baseline: {clim_mean:.3f}℃, Anomaly: {anomaly:.3f}℃")
        
        if not anomalies:
            raise HTTPException(status_code=500, detail="无法计算Nino3.4异常，预测数据为空")

        # 计算滑动平均（用于事件判定）
        an_arr = np.array(anomalies, dtype=float)
        rolling = []
        for i in range(len(an_arr)):
            window = an_arr[max(0, i-2): i+1]
            rolling.append(float(np.nanmean(window)))

        # 基于滑动平均判定事件
        labels = []
        en_count = 0
        la_count = 0
        for v in rolling:
            if v >= 0.5:
                labels.append('El Nino')
                en_count += 1
            elif v <= -0.5:
                labels.append('La Nina')
                la_count += 1
            else:
                labels.append('Neutral')

        # 构建用于绘图的 x 轴标签：仅在每年开头（1月）显示年份，其余位置为空字符串
        try:
            base_year = int(os.environ.get('ORCA_DL_ENSO_START_YEAR', 2026))
        except Exception:
            base_year = 2026

        x_labels_for_plot = []
        for step in steps_used:
            # 计算该 step 对应的日历月份与年份（以 base_year 为起始年）
            # month_idx: 1-12
            month_idx = ((start_month_idx + step) % 12) + 1
            year = base_year + ((start_month_idx + step) // 12)
            if month_idx == 1:
                x_labels_for_plot.append(str(year))
            else:
                x_labels_for_plot.append("")

        # 生成 ENSO 时间序列可视化（x 轴仅在每年开头标注年份）
        visual_base64 = create_enso_plot(
            anomalies=anomalies,
            x_labels=x_labels_for_plot,
            title=f"Nino3.4 Anomaly Forecast (Start Month {request.start_month})",
            ylabel="Nino3.4 Anomaly (°C)"
        )

        # 构建更详细的属性描述，包含判定摘要、时空范围与深度信息
        desc_events = f"El Nino steps: {en_count}; La Nina steps: {la_count};"
        try:
            # Nino3.4 区域为固定范围，可同时列出数值
            nlat_min, nlat_max = NINO34_LAT_MIN, NINO34_LAT_MAX
            nlon_min, nlon_max = NINO34_LON_MIN, NINO34_LON_MAX
        except Exception:
            nlat_min, nlat_max, nlon_min, nlon_max = -5.0, 5.0, 190.0, 240.0

        # ENSO 仅使用表层 SST
        depth_info = 'surface'

        attr_desc = (
            f"指标：Nino3.4 区域平均 SST 异常（5°S–5°N, 170°W–120°W，即纬度 {nlat_min}°–{nlat_max}°，经度 {nlon_min}°–{nlon_max}°）；"
            f"单位：°C；"
            f"计算方法：每步取区域月度平均 SST 并减去对应月的气候平均（基线由模型预测优先构建）；"
            f"事件判定：使用滑动平均，阈值：>=0.5°C 判定为 El Nino，<=-0.5°C 判定为 La Nina；"
            f"起始月份（输入）={request.start_month}，请求预测步长={request.predict_steps}，使用偏移窗口 start_offset={start_offset}, end_offset={end_offset}；"
            f"判定摘要：{desc_events}；深度信息：{depth_info}；维度说明：[窗口步长({len(anomalies)})]"
        )
        
        logger.info(f"[ENSO预测] 处理完成")
        
        return ENSOForecastResponse(
            attribute_matrix=anomalies,
            attribute_description=attr_desc,
            event_labels=labels,
            visual_base64=visual_base64,
        )
        
    except HTTPException:
        raise
    except Exception as e: 
        logger.error(f"[ENSO预测] 处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


# ==================== 功能3: 海洋热浪可视化 ====================
@app.post(
    "/marine_heatwave_visual",
    operation_id="marine_heatwave_visual",
    response_model=MarineHeatwaveResponse,
    responses={
        400: {"model": ErrorResponse, "description":  "参数错误"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    },
    tags=["heatwave"],
    summary="海洋热浪可视化",
    description="检测指定时段内的海洋热浪事件，返回热浪天数的空间分布可视化"
)
async def marine_heatwave_visual(request: MarineHeatwaveRequest):
    """
    海洋热浪可视化接口
    
    - 将原始thetao数据归一化，调用模型获取指定时段的thetao预测数据
    - 将模型输出的thetao数据反归一化，计算每个网格点每月的thetao异常
    - 使用传入的热浪阈值判定热浪事件
    - 统计每个网格点在指定时段内的热浪发生总天数
    - 生成填色图可视化
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="模型未加载，请稍后重试")
    
    try:
        # 转换参考起始月份（1-12 -> 0-11）并解析相对偏移
        start_month_idx = request.start_month - 1
        start_offset = int(getattr(request, 'start_offset', 0))
        end_offset = int(getattr(request, 'end_offset', 11))
        # 窗口内的步数（仅用于统计/可视化）
        window_steps = end_offset - start_offset + 1
        # 传给模型的预测步数需要覆盖到 end_offset（即请求步骤 0..end_offset），模型起始月不应跟随窗口偏移
        predict_steps_model = end_offset + 1
        logger.info(
            f"[海洋热浪] 相对时间窗: start_month({request.start_month}) + offset[{start_offset}:{end_offset}], "
            f"window_steps={window_steps}, model_predict_steps={predict_steps_model}"
        )
        
        # 获取经纬度索引
        lat_indices, lon_indices, actual_lats, actual_lons = get_lat_lon_indices(
            request.lat_min, request.lat_max, request.lon_min, request.lon_max
        )
        
        logger.info(f"[海洋热浪] 经纬度范围: lat{lat_indices}, lon{lon_indices}")
        
        # 准备输入数据：始终从服务器端示例数据加载（取消上传数据功能）
        ocean_data = {}
        atmo_data = {}

        # 使用通用加载器，优先 .nc，再尝试 .npy；保留原有字典键为短名
        for key in ocean_keys:
            _, arr = load_example_or_default(key)
            ocean_data[key] = arr

        for key in atmo_keys:
            _, arr = load_example_or_default(key)
            atmo_data[key] = arr
        
        # 基于示例数据的 tos (sst) 构建海洋有效点掩码 (True 表示海洋，可用于排除陆地)
        ocean_mask_full = None
        try:
            tos_example = ocean_data.get('tos', None)
            if tos_example is not None:
                tos_arr = np.array(tos_example)
                if tos_arr.ndim == 3:
                    tos_arr_2d = tos_arr[0]
                elif tos_arr.ndim == 2:
                    tos_arr_2d = tos_arr
                else:
                    tos_arr_2d = tos_arr.squeeze()
                ocean_mask_full = np.isfinite(tos_arr_2d)
            else:
                ocean_mask_full = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)
        except Exception:
            ocean_mask_full = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)

        # 调用模型预测
        logger.info(f"[海洋热浪] 开始归一化输入数据并预测...")
        
        # 传给模型的起始月份应为用户提供的参考起始月（与窗口偏移无关）
        # 向模型请求的步数覆盖到 end_offset（即 0..end_offset），接口内部再根据偏移只统计 [start_offset, end_offset]
        prediction = predict_ocean_state(
            predictor=predictor,
            ocean_data=ocean_data,
            atmo_data=atmo_data,
            start_month=start_month_idx,
            predict_steps=predict_steps_model,
            denormalize=True,
            use_model_naming=True,
            normalized=True
        )
        
        logger.info(f"[海洋热浪] 模型预测完成，输出形状: {prediction.metadata['output_shape']}")
        
        # 统计热浪月数（按月计数）
        results_by_step = prediction.metadata.get('results_by_step', {})
        heatwave_months = np.zeros((LAT_POINTS, LON_POINTS))

        # 解析阈值（优先使用请求中提供的阈值；否则加载服务器端预计算的每月90百分位文件）
        threshold_values = request.mh_threshold_values
        is_grid_level_threshold = False
        if threshold_values is None:
            print("未提供 mh_threshold_values，加载预计算文件")
            p90_path = '/mnt/data/zhu.yishun/ORCA-DL-main/tos_p90_monthly_1993_2023.npy'
            if not os.path.exists(p90_path):
                raise HTTPException(status_code=500, detail=f"mh_threshold_values 未提供，且预计算文件未找到: {p90_path}")
            try:
                tv = np.load(p90_path)
            except Exception as e:
                logger.error(f"加载预计算阈值文件失败: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="无法加载预计算的热浪阈值文件")
            # tv 应为 (12, H, W)
            threshold_values = tv
            is_grid_level_threshold = True
        else:
            # 如果用户提供了阈值，判断是否为网格级（每月为数组）
            try:
                is_grid_level_threshold = isinstance(threshold_values[0], (list, np.ndarray))
            except Exception:
                is_grid_level_threshold = False
        
        for step in range(predict_steps_model):
            # 预测月份索引（0-11），基于用户提供的 start_month（模型 step 0 对应 start_month_idx）
            pred_month_idx = (start_month_idx + step) % 12
            pred_month = pred_month_idx + 1

            # 仅在用户指定的偏移窗口内统计（step 对应相对于 start_month 的偏移）
            if step < start_offset or step > end_offset:
                # 跳过但可用于诊断或模型内部缓存
                continue
            
            step_data = results_by_step.get(step, {})
            if 'thetao' in step_data:
                thetao_data = step_data['thetao']
                
                # 处理维度，取表层（第一层）
                if len(thetao_data.shape) == 4:  # (B, C, H, W)
                    thetao_surface = thetao_data[0, 0]  # 表层
                elif len(thetao_data.shape) == 3:  # (C, H, W)
                    thetao_surface = thetao_data[0]  # 表层
                else:
                    thetao_surface = thetao_data
                
                # 获取该月的热浪阈值（优先网格级，否则按标量阈值比较）
                if is_grid_level_threshold:
                    try:
                        threshold = np.array(threshold_values[pred_month_idx])
                    except Exception:
                        # 支持 threshold_values 为 ndarray(12,H,W) 或 列表
                        threshold = np.array(threshold_values)[pred_month_idx]
                else:
                    # 标量阈值（每月一个数值）
                    threshold = float(threshold_values[pred_month_idx])

                # 判定热浪事件：直接比较表层温度是否超出该月90百分位阈值
                try:
                    # 排除陆地：如果存在海洋掩码，则在判定中只考虑海洋点
                    try:
                        base_mask = ocean_mask_full
                    except Exception:
                        base_mask = None

                    heatwave_mask = thetao_surface > threshold
                    if base_mask is not None:
                        # 保证形状一致：若 base_mask 与 thetao_surface 形状不一致，尝试裁剪/广播
                        try:
                            if base_mask.shape != heatwave_mask.shape:
                                bm = base_mask
                                # 选择中心裁剪或简单切片到目标形状
                                h = min(bm.shape[0], heatwave_mask.shape[0])
                                w = min(bm.shape[1], heatwave_mask.shape[1])
                                bm2 = np.zeros_like(heatwave_mask, dtype=bool)
                                bm2[:h, :w] = bm[:h, :w]
                                base_mask = bm2
                            heatwave_mask = np.logical_and(heatwave_mask, base_mask)
                        except Exception:
                            # 若处理失败则忽略掩码
                            pass
                except ValueError:
                    # 如果形状不一致，尝试广播或重采样：若 threshold 为网格且形状与模型网格不同，尝试转置/resize不做复杂处理
                    print('形状不一致，尝试广播或重采样')
                    th = np.array(threshold)
                    if th.shape != thetao_surface.shape:
                        # 尝试转置或缩放到目标形状（简单裁剪/填充）
                        out = np.zeros_like(thetao_surface)
                        h = min(out.shape[0], th.shape[0])
                        w = min(out.shape[1], th.shape[1])
                        out[:h, :w] = th[:h, :w]
                        threshold = out
                    heatwave_mask = thetao_surface > threshold

                # 统计天数（按月计，整月计30天）
                # 记录诊断信息：阈值形状/统计、thetao表层统计、超阈比例
                try:
                    th_arr = np.array(threshold)
                except Exception:
                    th_arr = None

                thetao_mean = float(np.nanmean(thetao_surface))
                thetao_std = float(np.nanstd(thetao_surface))
                if th_arr is not None:
                    try:
                        th_mean = float(np.nanmean(th_arr))
                        th_std = float(np.nanstd(th_arr))
                        th_shape = th_arr.shape
                    except Exception:
                        th_mean = None
                        th_std = None
                        th_shape = None
                else:
                    th_mean = None
                    th_std = None
                    th_shape = None

                heatwave_mask_float = heatwave_mask.astype(float)
                # 统计为月数：每个满足条件的月份计为 1（月级统计），仅对海洋点累加
                heatwave_months += heatwave_mask_float * 1.0

                # 统计时排除陆地点
                try:
                    ocean_points = int(np.sum(base_mask)) if base_mask is not None else int(np.prod(heatwave_mask.shape))
                    hot_ocean_points = int(np.sum(np.logical_and(heatwave_mask, base_mask))) if base_mask is not None else int(np.sum(heatwave_mask))
                    frac = float(hot_ocean_points) / float(ocean_points) if ocean_points > 0 else 0.0
                except Exception:
                    hot_ocean_points = int(np.sum(heatwave_mask))
                    ocean_points = int(np.prod(heatwave_mask.shape))
                    frac = float(np.sum(heatwave_mask_float) / float(ocean_points)) if ocean_points>0 else 0.0

                heatwave_count = hot_ocean_points
                total_points = ocean_points

                logger.info(
                    f"[海洋热浪-诊断] Step {step+1}, 月份 {pred_month}, "
                    f"thetao mean={thetao_mean:.3f}, std={thetao_std:.3f}; "
                    f"threshold mean={th_mean}, std={th_std}, shape={th_shape}; "
                    f"hot points={heatwave_count}/{total_points} ({frac*100:.2f}%)"
                )
        
        # 切片指定经纬度范围（结果为月数）
        sliced_heatwave_months = slice_by_lat_lon(heatwave_months, lat_indices, lon_indices)
        
        logger.info(f"[海洋热浪] 切片完成，数据形状: {sliced_heatwave_months.shape}")
        
        # 生成填色图可视化：显示具体起止年月（以 step0 = 2026-01 为基准，可通过环境变量 ORCA_DL_ENSO_START_YEAR 覆盖）
        try:
            base_year = int(os.environ.get('ORCA_DL_ENSO_START_YEAR', 2026))
        except Exception:
            base_year = 2026

        start_month_num = ((start_month_idx + start_offset) % 12) + 1
        start_year = base_year + ((start_month_idx + start_offset) // 12)
        end_month_num = ((start_month_idx + end_offset) % 12) + 1
        end_year = base_year + ((start_month_idx + end_offset) // 12)

        title = f"Marine Heatwave Months ({start_year}/{start_month_num:02d}-{end_year}/{end_month_num:02d})"
        # 切片海洋掩码并传入绘图函数，以便排除陆地并勾勒轮廓
        try:
            sliced_land_mask = slice_by_lat_lon(ocean_mask_full.astype(float), lat_indices, lon_indices).astype(bool)
        except Exception:
            sliced_land_mask = None

        visual_base64 = create_heatwave_plot(
            data=sliced_heatwave_months,
            lats=actual_lats,
            lons=actual_lons,
            title=title,
            land_mask=sliced_land_mask
        )
        
        # 构建属性描述
        attr_desc = (
            f"海洋热浪总月数：指定时段内每个网格点表层thetao（已反归一化）"
            f"异常超过传入的90百分位阈值的月份总数；"
            f"单位：月；"
            f"维度说明：[纬度数({len(actual_lats)})×经度数({len(actual_lons)})，已切片为指定经纬度范围]"
        )
        
        logger.info(f"[海洋热浪] 处理完成")
        
        return MarineHeatwaveResponse(
            attribute_matrix=sanitize_for_json(sliced_heatwave_months),
            attribute_description=attr_desc,
            visual_base64=visual_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[海洋热浪] 处理失败:  {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


# ==================== 功能4: 年代际预测可视化 ====================
@app.post(
    "/decadal_prediction_visual",
    operation_id="decadal_prediction_visual",
    response_model=DecadalPredictionResponse,
    responses={
        400: {"model":  ErrorResponse, "description": "参数错误"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    },
    tags=["decadal"],
    summary="年代际预测可视化",
    description="进行长期（10年以上）预测，返回指定变量的年度均值演化曲线"
)
async def decadal_prediction_visual(request: DecadalPredictionRequest):
    """
    年代际预测可视化接口
    
    - 将原始数据归一化，调用模型循环预测指定年数的月度数据
    - 将模型输出数据反归一化，按年聚合数据（每年12个月的算术平均）
    - 计算指定经纬度范围内的变量均值
    - 生成十年尺度的演化曲线图
    """
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="模型未加载，请稍后重试")
    
    try:
        var_name = request.var_name.value
        # 支持可选粒度：当 output_granularity == 'month' 时，request.predict_steps 表示月数；
        # 当为 'year' 时，request.predict_steps 表示年数（每年12个月）
        # 兼容旧接口名: 若存在旧字段 `predict_years` 则优先使用其值（用于向后兼容）
        predict_steps = int(getattr(request, 'predict_steps', getattr(request, 'predict_years', 0)))
        gran = getattr(request, 'output_granularity', 'year')
        if gran == 'month':
            total_months = predict_steps
            logger.info(f"[年代际预测] 变量: {var_name}, 预测步数: {predict_steps}（月）, 总月数: {total_months}")
        else:
            total_months = predict_steps * 12
            logger.info(f"[年代际预测] 变量: {var_name}, 预测步数: {predict_steps}（年）, 总月数: {total_months}")

        # 校验变量名
        if var_name not in ORCADLPredictor.MODEL_TO_STAT_MAPPING:
            raise HTTPException(
                status_code=400,
                detail=f"变量名 {var_name} 不在支持的变量列表中"
            )

        # 获取经纬度索引
        lat_indices, lon_indices, actual_lats, actual_lons = get_lat_lon_indices(
            request.lat_min, request.lat_max, request.lon_min, request.lon_max
        )
        logger.info(f"[年代际预测] 经纬度范围: lat{lat_indices}, lon{lon_indices}")

        # 准备输入数据：始终从服务器端示例数据加载（取消上传数据功能）
        ocean_data = {}
        atmo_data = {}
        # 使用通用加载器，优先 .nc，再尝试 .npy；保留原有字典键为短名
        for key in ocean_keys:
            _, arr = load_example_or_default(key)
            ocean_data[key] = arr
        for key in atmo_keys:
            _, arr = load_example_or_default(key)
            atmo_data[key] = arr

        # 构建海洋掩码（可用于后续排除陆地点的空间平均计算）
        ocean_mask_full = None
        try:
            tos_example = ocean_data.get('tos', None)
            if tos_example is not None:
                tos_arr = np.array(tos_example)
                if tos_arr.ndim == 3:
                    tos_arr_2d = tos_arr[0]
                elif tos_arr.ndim == 2:
                    tos_arr_2d = tos_arr
                else:
                    tos_arr_2d = tos_arr.squeeze()
                ocean_mask_full = np.isfinite(tos_arr_2d)
            else:
                ocean_mask_full = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)
        except Exception:
            ocean_mask_full = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)

        # 直接调用模型一次性预测所有步长
        logger.info(f"[年代际预测] 使用forward_multi_steps一次性预测{total_months}步...")

        # 转换用户输入月份 1-12 -> 内部 0-11
        start_month_idx = request.start_month - 1

        prediction = predict_ocean_state(
            predictor=predictor,
            ocean_data=ocean_data,
            atmo_data=atmo_data,
            start_month=start_month_idx,
            predict_steps=total_months,
            denormalize=True,
            use_model_naming=True,
            normalized=True
        )

        results_by_step = prediction.metadata.get('results_by_step', {})
        all_monthly_values = []
        for step in range(total_months):
            step_data = results_by_step.get(step, {})
            if var_name in step_data:
                var_data = step_data[var_name]
                # 只取第一个batch和第一个深度层（如有）
                # 支持用户指定深度层：优先使用 request.depth_index（若变量有深度维）
                if len(var_data.shape) > 2:
                    if len(var_data.shape) == 4:
                        # 形状通常为 [batch, depth, lat, lon]
                        d_idx = int(getattr(request, 'depth_index', 0))
                        d_idx = max(0, d_idx)
                        if d_idx < var_data.shape[1]:
                            var_data = var_data[0, d_idx]
                        else:
                            var_data = var_data[0, 0]
                    elif len(var_data.shape) == 3:
                        # 3D 情况可能为 [batch, lat, lon] 或 [depth, lat, lon]
                        # 如果第一个维度看上去像 depth（>1 且 depth_index 合法），按 depth 处理
                        d_idx = int(getattr(request, 'depth_index', 0))
                        if var_data.shape[0] > 1 and d_idx < var_data.shape[0]:
                            var_data = var_data[d_idx]
                        else:
                            # 否则按 batch 维处理，取第一个样本
                            var_data = var_data[0]
                sliced = slice_by_lat_lon(var_data, lat_indices, lon_indices)
                # 应用海洋掩码排除陆地：将陆地点置为 NaN，再计算空间平均
                try:
                    sliced_mask = slice_by_lat_lon(ocean_mask_full.astype(float), lat_indices, lon_indices).astype(bool)
                    sliced = np.where(sliced_mask, sliced, np.nan)
                except Exception:
                    # 若掩码切片失败，退回到原始切片
                    pass
                spatial_mean = np.nanmean(sliced)
                all_monthly_values.append(float(spatial_mean))
        # 根据请求的输出粒度决定返回月序列还是年聚合
        gran = getattr(request, 'output_granularity', 'year')
        if gran == 'month':
            # 按月返回：生成时间序列图（每步为月）
            month_labels = []
            for i in range(len(all_monthly_values)):
                # 计算相对月份标签，从 start_month 开始 (输入为1-12，已转换为 start_month_idx)
                m_idx = (start_month_idx + i) % 12
                month_labels.append(f"Month {m_idx+1}")

            visual_base64 = create_time_series_plot(
                values=all_monthly_values,
                x_labels=month_labels,
                title=f"Decadal Prediction (Monthly): {VAR_DESCRIPTIONS.get(var_name, {}).get('name_en', var_name)}",
                ylabel=f"{VAR_DESCRIPTIONS.get(var_name, {}).get('name_en', var_name)} ({VAR_DESCRIPTIONS.get(var_name, {}).get('unit', '')})"
            )

            # 组装更详细的属性描述：包含反归一化说明、单位、时间长度、输出经纬度范围与深度信息
            try:
                lat_min_out = float(actual_lats.min()) if getattr(actual_lats, 'size', 0) else float(request.lat_min)
                lat_max_out = float(actual_lats.max()) if getattr(actual_lats, 'size', 0) else float(request.lat_max)
                lon_min_out = float(actual_lons.min()) if getattr(actual_lons, 'size', 0) else float(request.lon_min)
                lon_max_out = float(actual_lons.max()) if getattr(actual_lons, 'size', 0) else float(request.lon_max)
            except Exception:
                lat_min_out, lat_max_out = request.lat_min, request.lat_max
                lon_min_out, lon_max_out = request.lon_min, request.lon_max

            # 深度信息：优先使用用户请求的 depth_index，否则根据变量描述推断
            depth_info = 'surface'
            try:
                if hasattr(request, 'depth_index') and request.depth_index is not None:
                    depth_info = f"depth_index={int(request.depth_index)}"
                else:
                    dl = VAR_DESCRIPTIONS.get(var_name, {}).get('depth_levels', 1)
                    depth_info = 'surface' if dl == 1 else 'depth_index=0 (default surface layer)'
            except Exception:
                depth_info = 'unknown'

            attr_desc = (
                f"指定经纬度范围内逐月{VAR_DESCRIPTIONS.get(var_name, {}).get('name_cn', var_name)}的时间序列（已反归一化）；"
                f"单位：{VAR_DESCRIPTIONS.get(var_name, {}).get('unit', '')}；"
                f"总月数：{len(all_monthly_values)}；"
                f"输出经纬度范围：纬度 {lat_min_out:.2f}°–{lat_max_out:.2f}°，经度 {lon_min_out:.2f}°–{lon_max_out:.2f}°；"
                f"深度信息：{depth_info}；"
                f"维度说明：[总月数({len(all_monthly_values)})]"
            )

            logger.info(f"[年代际预测] 处理完成 (按月返回)")

            return DecadalPredictionResponse(
                attribute_matrix=all_monthly_values,
                attribute_description=attr_desc,
                visual_base64=visual_base64
            )
        else:
            # 年均值分组（默认旧行为）
            annual_means = []
            for y in range(predict_steps):
                s = y * 12
                e = (y + 1) * 12
                if s >= len(all_monthly_values):
                    break
                group = all_monthly_values[s:e]
                annual_means.append(float(np.nanmean(group)) if group else float('nan'))

            visual_base64 = create_decadal_plot(
                annual_means,
                start_year=request.start_year,
                title=f"Decadal Prediction: {VAR_DESCRIPTIONS.get(var_name, {}).get('name_en', var_name)}",
                ylabel=f"{VAR_DESCRIPTIONS.get(var_name, {}).get('name_en', var_name)} Annual Mean ({VAR_DESCRIPTIONS.get(var_name, {}).get('unit', '')})"
            )

            # 提供更详细的属性描述：包括反归一化、单位、预测年数、输出经纬度范围与深度信息
            try:
                lat_min_out = float(actual_lats.min()) if getattr(actual_lats, 'size', 0) else float(request.lat_min)
                lat_max_out = float(actual_lats.max()) if getattr(actual_lats, 'size', 0) else float(request.lat_max)
                lon_min_out = float(actual_lons.min()) if getattr(actual_lons, 'size', 0) else float(request.lon_min)
                lon_max_out = float(actual_lons.max()) if getattr(actual_lons, 'size', 0) else float(request.lon_max)
            except Exception:
                lat_min_out, lat_max_out = request.lat_min, request.lat_max
                lon_min_out, lon_max_out = request.lon_min, request.lon_max

            try:
                if hasattr(request, 'depth_index') and request.depth_index is not None:
                    depth_info = f"depth_index={int(request.depth_index)}"
                else:
                    dl = VAR_DESCRIPTIONS.get(var_name, {}).get('depth_levels', 1)
                    depth_info = 'surface' if dl == 1 else 'depth_index=0 (default surface layer)'
            except Exception:
                depth_info = 'unknown'

            attr_desc = (
                f"全球平均{VAR_DESCRIPTIONS.get(var_name, {}).get('name_cn', var_name)}年度均值："
                f"指定经纬度范围内每年12个月{VAR_DESCRIPTIONS.get(var_name, {}).get('name_cn', var_name)}的算术平均（已反归一化）；"
                f"单位：{VAR_DESCRIPTIONS.get(var_name, {}).get('unit', '')}；"
                f"预测年数：{len(annual_means)}；"
                f"输出经纬度范围：纬度 {lat_min_out:.2f}°–{lat_max_out:.2f}°，经度 {lon_min_out:.2f}°–{lon_max_out:.2f}°；"
                f"深度信息：{depth_info}；"
                f"维度说明：[预测年数({len(annual_means)})]"
            )

            logger.info(f"[年代际预测] 处理完成")

            return DecadalPredictionResponse(
                attribute_matrix=annual_means,
                attribute_description=attr_desc,
                visual_base64=visual_base64
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[年代际预测] 处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


# ==================== 应用入口 ====================
if __name__ == "__main__":
    uvicorn.run("ocean_visual_api:app", host="0.0.0.0", port=51002, reload=True)
