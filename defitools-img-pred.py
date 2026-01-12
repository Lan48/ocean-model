#!/usr/bin/env python3
"""
ORCA-DL Ocean Prediction Visualization API
修改版：支持返回纯粹的图片base64编码 或 完整预测数组，支持自定义经纬度范围
"""

import io
import base64
import tempfile
import os
import json
from typing import Optional
from enum import Enum

import numpy as np
import xarray as xr
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import uvicorn

from model import ORCADLConfig, ORCADLModel

# ==================== 配置常量 ====================
VAR_LIST = ["so", "thetao", "tos", "uo", "vo", "zos"]
OUT_CHANS = [16, 16, 1, 16, 16, 1]
INPUT_VARIABLES = ['salt', 'pottmp', 'sst', 'ucur', 'vcur', 'sshg', 'uflx', 'vflx']

# 完整网格范围
LAT_MIN, LAT_MAX, LAT_POINTS = -63.5, 63.5, 128
LON_MIN, LON_MAX, LON_POINTS = 0.5, 359.5, 360

CHANNEL_OFFSETS = np.cumsum([0] + OUT_CHANS[:-1]).tolist()

# 预计算完整的经纬度数组
FULL_LATS = np.linspace(LAT_MIN, LAT_MAX, LAT_POINTS)
FULL_LONS = np.linspace(LON_MIN, LON_MAX, LON_POINTS)

# 输入变量到输出变量的映射关系
INPUT_TO_OUTPUT_MAPPING = {
    'so': 'salt',
    'thetao': 'pottmp',
    'tos':  'sst',
    'uo': 'ucur',
    'vo': 'vcur',
    'zos': 'sshg'
}

# 输出变量描述信息
VAR_DESCRIPTIONS = {
    "so": {
        "name_en": "Salinity",
        "name_cn":  "盐度",
        "unit":  "PSU (Practical Salinity Unit)",
        "description":  "海水盐度，表示每千克海水中溶解盐类的质量",
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
        "unit": "°C",
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
    "vo": {
        "name_en":  "Meridional Velocity",
        "name_cn": "经向流速",
        "unit": "m/s",
        "description": "海流在南北方向上的速度分量，正值表示向北",
        "depth_levels": 16
    },
    "zos": {
        "name_en": "Sea Surface Height",
        "name_cn":  "海表面高度",
        "unit": "m",
        "description":  "相对于参考面的海表面高度异常",
        "depth_levels": 1
    }
}

# ==================== 全局变量 ====================
model = None
stat = None
out_stat = None
device = None

def load_model_and_stats():
    global model, stat, out_stat, device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    stat = {
        'mean': {v: np.load(f"/mnt/data/zhu.yishun/ORCA-DL-main/stat/mean/{v}.npy") for v in INPUT_VARIABLES},
        'std': {v: np.load(f"/mnt/data/zhu.yishun/ORCA-DL-main/stat/std/{v}.npy") for v in INPUT_VARIABLES}
    }
    
    out_stat = {
        'mean': {},
        'std':  {}
    }
    
    for out_var, in_var in INPUT_TO_OUTPUT_MAPPING.items():
        out_stat['mean'][out_var] = stat['mean'][in_var].copy()
        out_stat['std'][out_var] = stat['std'][in_var].copy()
        print(f"Loaded stats for {out_var} from {in_var}:  mean shape={out_stat['mean'][out_var].shape}, std shape={out_stat['std'][out_var].shape}")
    
    model = ORCADLModel(ORCADLConfig.from_json_file('/mnt/data/zhu.yishun/ORCA-DL-main/model_config.json'))
    model.load_state_dict(torch.load('/mnt/data/zhu.yishun/ORCA-DL-main/ckpt/seed_1.bin', map_location='cpu'))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

# ==================== 辅助函数 ====================
def get_lat_lon_indices(lat_min:  float, lat_max: float, lon_min: float, lon_max: float):
    """
    根据经纬度范围获取对应的数组索引
    
    Returns:
        lat_indices: 纬度索引范围 (start, end)
        lon_indices: 经度索引范围 (start, end)
        actual_lats: 实际返回的纬度数组
        actual_lons: 实际返回的经度数组
    """
    # 找到最接近的纬度索引
    lat_start_idx = np.searchsorted(FULL_LATS, lat_min, side='left')
    lat_end_idx = np.searchsorted(FULL_LATS, lat_max, side='right')
    
    # 处理经度（可能跨越0度）
    lon_start_idx = np.searchsorted(FULL_LONS, lon_min, side='left')
    lon_end_idx = np.searchsorted(FULL_LONS, lon_max, side='right')
    
    # 确保索引在有效范围内
    lat_start_idx = max(0, lat_start_idx)
    lat_end_idx = min(LAT_POINTS, lat_end_idx)
    lon_start_idx = max(0, lon_start_idx)
    lon_end_idx = min(LON_POINTS, lon_end_idx)
    
    # 获取实际的经纬度值
    actual_lats = FULL_LATS[lat_start_idx: lat_end_idx]
    actual_lons = FULL_LONS[lon_start_idx:lon_end_idx]
    
    return (lat_start_idx, lat_end_idx), (lon_start_idx, lon_end_idx), actual_lats, actual_lons

def slice_by_lat_lon(data:  np.ndarray, lat_indices: tuple, lon_indices: tuple) -> np.ndarray:
    """
    根据经纬度索引切片数据
    
    Args:
        data:  输入数据，最后两个维度是 (lat, lon)
        lat_indices:  (start, end) 纬度索引
        lon_indices:  (start, end) 经度索引
    
    Returns:
        切片后的数据
    """
    lat_start, lat_end = lat_indices
    lon_start, lon_end = lon_indices
    
    # 支持任意维度的数据，只切片最后两个维度
    if data.ndim == 2:
        return data[lat_start: lat_end, lon_start:lon_end]
    elif data.ndim == 3:
        return data[: , lat_start: lat_end, lon_start:lon_end]
    elif data.ndim == 4:
        return data[:, : , lat_start: lat_end, lon_start:lon_end]
    elif data.ndim == 5:
        return data[:, : , :, lat_start:lat_end, lon_start: lon_end]
    else:
        raise ValueError(f"Unsupported data dimension: {data.ndim}")

def sanitize_for_json(arr: np.ndarray) -> np.ndarray:
    """清理数组中的 inf 和 nan 值，使其 JSON 兼容"""
    arr = arr.copy()
    arr = np.nan_to_num(arr, nan=-999.0, posinf=1e38, neginf=-1e38)
    return arr

def get_global_channel_index(var_name: str, ch_in_var: int) -> int:
    if var_name not in VAR_LIST: 
        raise ValueError(f"Unknown var_name: {var_name}")
    idx = VAR_LIST.index(var_name)
    n_ch = OUT_CHANS[idx]
    if not (0 <= ch_in_var < n_ch):
        raise ValueError(f"ch_in_var out of range: {ch_in_var}, max: {n_ch-1}")
    return CHANNEL_OFFSETS[idx] + ch_in_var

def make_lat_lon_grid():
    lon2d, lat2d = np.meshgrid(FULL_LONS, FULL_LATS)
    return lat2d, lon2d

def process_input_files(files_dict: dict, month: int):
    ocean_vars, atmo_vars = [], []
    
    for v in INPUT_VARIABLES[:-2]: 
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            tmp.write(files_dict[v])
            tmp_path = tmp.name
        try:
            ds = xr.open_dataset(tmp_path)
            data = ds[v].values
            normed = (data - stat['mean'][v][month]) / stat['std'][v][month]
            ocean_vars.append(normed if len(normed.shape) == 3 else normed[None])
            ds.close()
        finally:
            os.unlink(tmp_path)
    
    for v in INPUT_VARIABLES[-2:]: 
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            tmp.write(files_dict[v])
            tmp_path = tmp.name
        try:
            ds = xr.open_dataset(tmp_path)
            data = ds[v].values
            normed = (data - stat['mean'][v][month]) / stat['std'][v][month]
            atmo_vars.append(normed[None])
            ds.close()
        finally:
            os.unlink(tmp_path)
    
    ocean_tensor = torch.from_numpy(np.nan_to_num(np.concatenate(ocean_vars, axis=0)))[None].float().to(device)
    atmo_tensor = torch.from_numpy(np.nan_to_num(np.concatenate(atmo_vars, axis=0)))[None].float().to(device)
    return ocean_tensor, atmo_tensor

def run_prediction(ocean_vars, atmo_vars, steps, batch_size):
    if batch_size > 1:
        ocean_vars = ocean_vars.repeat(batch_size, 1, 1, 1)
        atmo_vars = atmo_vars.repeat(batch_size, 1, 1, 1)
    with torch.no_grad():
        output = model(ocean_vars=ocean_vars, atmo_vars=atmo_vars, predict_time_steps=steps)
        return output.preds.detach().cpu().numpy()

def denormalize_predictions(preds:  np.ndarray, month: int) -> dict:
    """对模型预测结果进行反归一化"""
    if preds.ndim == 4:
        preds = preds[:, np.newaxis, : , : , :]
    
    denorm_results = {}
    
    for var_idx, var_name in enumerate(VAR_LIST):
        n_channels = OUT_CHANS[var_idx]
        start_ch = CHANNEL_OFFSETS[var_idx]
        end_ch = start_ch + n_channels
        
        var_data = preds[:, :, start_ch:end_ch, : , : ].copy()
        
        mean_arr = out_stat['mean'][var_name]
        std_arr = out_stat['std'][var_name]
        
        if isinstance(mean_arr, np.ndarray):
            if mean_arr.ndim >= 1 and mean_arr.shape[0] == 12:
                mean_val = mean_arr[month]
                std_val = std_arr[month]
            else: 
                mean_val = mean_arr
                std_val = std_arr
        else:
            mean_val = mean_arr
            std_val = std_arr
        
        if isinstance(std_val, np.ndarray):
            std_val = np.where(std_val == 0, 1.0, std_val)
        elif std_val == 0:
            std_val = 1.0
        
        try:
            if isinstance(mean_val, np.ndarray) and mean_val.ndim >= 2:
                if mean_val.ndim == 3 and mean_val.shape[0] == n_channels:
                    mean_val = mean_val[np.newaxis, np.newaxis, : , : , :]
                    std_val = std_val[np.newaxis, np.newaxis, :, :, :]
                elif mean_val.ndim == 2:
                    mean_val = mean_val[np.newaxis, np.newaxis, np.newaxis, : , :]
                    std_val = std_val[np.newaxis, np.newaxis, np.newaxis, : , :]
            
            denorm_data = var_data * std_val + mean_val
        except Exception as e:
            print(f"Warning: Failed to denormalize {var_name}:  {e}")
            denorm_data = var_data
        
        denorm_results[var_name] = denorm_data
    
    return denorm_results

def compute_array_statistics(arr: np.ndarray) -> dict:
    """计算数组的统计信息，处理 inf 和 nan"""
    finite_arr = arr[np.isfinite(arr)]
    
    if len(finite_arr) == 0:
        return {
            "shape": list(arr.shape),
            "min": None,
            "max": None,
            "mean": None,
            "std":  None,
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
            "nan_percentage": 100.0
        }
    
    return {
        "shape": list(arr.shape),
        "min": float(np.min(finite_arr)),
        "max":  float(np.max(finite_arr)),
        "mean":  float(np.mean(finite_arr)),
        "std":  float(np.std(finite_arr)),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count":  int(np.isinf(arr).sum()),
        "nan_percentage": float(np.isnan(arr).sum() / arr.size * 100)
    }

def create_visualization_base64(preds, batch_idx, step_idx, var_name, ch_in_var,
                                 cmap, vmin, vmax, figsize_w, figsize_h, title_extra, steps, month):
    """生成可视化图片并返回 base64 编码"""
    # 首先对预测结果进行反归一化 [1,4](@ref)
    denorm_preds = denormalize_predictions(preds, month)
    
    if steps == 1:
        preds = preds[: , np.newaxis, : , :, :]
    
    gch = get_global_channel_index(var_name, ch_in_var)
    
    # 使用反归一化后的数据 [1,4](@ref)
    var_data = denorm_preds[var_name]
    if var_data.ndim == 5:
        field = var_data[batch_idx, step_idx, ch_in_var, : , :]
    else:
        # 兼容4维数据
        field = var_data[batch_idx, ch_in_var, : , :]
    
    lat2d, lon2d = make_lat_lon_grid()
    
    if field.shape != lat2d.shape:
        if field.shape == (lat2d.shape[1], lat2d.shape[0]):
            field = field.T
    
    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
    pcm = ax.pcolormesh(lon2d, lat2d, field, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(pcm, ax=ax)
    
    var_name_cn = {"so": "盐度", "thetao": "位温", "tos": "海表面温度",
                   "uo": "纬向流速", "vo": "经向流速", "zos": "海表面高度"}
    title = f"{var_name_cn.get(var_name, var_name)} - batch={batch_idx}, step={step_idx}, ch={ch_in_var}"
    if title_extra: 
        title += f" {title_extra}"
    #ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=50, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')





# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="ORCA-DL Ocean Prediction API",
    description="海洋状态预测与可视化工具 - 支持返回base64图片或完整预测数组，支持自定义经纬度范围",
    version="1.2.0"
)

class VarNameEnum(str, Enum):
    so = "so"
    thetao = "thetao"
    tos = "tos"
    uo = "uo"
    vo = "vo"
    zos = "zos"

class CmapEnum(str, Enum):
    viridis = "viridis"
    coolwarm = "coolwarm"
    jet = "jet"
    RdBu_r = "RdBu_r"
    plasma = "plasma"

@app.on_event("startup")
async def startup():
    load_model_and_stats()

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(
    salt_file: UploadFile = File(...),
    pottmp_file: UploadFile = File(...),
    sst_file: UploadFile = File(...),
    ucur_file: UploadFile = File(...),
    vcur_file: UploadFile = File(...),
    sshg_file: UploadFile = File(...),
    uflx_file: UploadFile = File(...),
    vflx_file: UploadFile = File(...),
    month: int = Form(0, ge=0, le=11),
    predict_time_steps:  int = Form(1, ge=1, le=12),
    batch_size:  int = Form(1, ge=1, le=16),
    batch_idx: int = Form(0, ge=0),
    step_idx: int = Form(0, ge=0),
    var_name: VarNameEnum = Form(VarNameEnum.tos),
    ch_in_var: int = Form(0, ge=0),
    cmap: CmapEnum = Form(CmapEnum.viridis),
    vmin: Optional[float] = Form(None),
    vmax:  Optional[float] = Form(None),
    figsize_w: float = Form(12.0),
    figsize_h: float = Form(5.0),
    title_extra: Optional[str] = Form(None)
):
    """运行 ORCA-DL 模型预测并返回纯粹的图片base64编码"""
    try:
        if batch_idx >= batch_size:
            raise HTTPException(status_code=400, detail=f"batch_idx ({batch_idx}) >= batch_size ({batch_size})")
        if step_idx >= predict_time_steps: 
            raise HTTPException(status_code=400, detail=f"step_idx ({step_idx}) >= predict_time_steps ({predict_time_steps})")
        
        var_idx = VAR_LIST.index(var_name.value)
        max_ch = OUT_CHANS[var_idx]
        if ch_in_var >= max_ch: 
            raise HTTPException(status_code=400, detail=f"ch_in_var ({ch_in_var}) >= max channels ({max_ch})")
        
        files_dict = {
            'salt': await salt_file.read(),
            'pottmp': await pottmp_file.read(),
            'sst': await sst_file.read(),
            'ucur': await ucur_file.read(),
            'vcur': await vcur_file.read(),
            'sshg': await sshg_file.read(),
            'uflx': await uflx_file.read(),
            'vflx': await vflx_file.read()
        }
        
        ocean_vars, atmo_vars = process_input_files(files_dict, month)
        preds = run_prediction(ocean_vars, atmo_vars, predict_time_steps, batch_size)
        
        img_base64 = create_visualization_base64(
            preds, batch_idx, step_idx, var_name.value, ch_in_var,
            cmap.value, vmin, vmax, figsize_w, figsize_h, title_extra, predict_time_steps,month
        )
        
        return Response(content=img_base64, media_type="text/plain")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict_array")
async def predict_array(
    salt_file: UploadFile = File(...),
    pottmp_file: UploadFile = File(...),
    sst_file: UploadFile = File(...),
    ucur_file: UploadFile = File(...),
    vcur_file: UploadFile = File(...),
    sshg_file: UploadFile = File(...),
    uflx_file: UploadFile = File(...),
    vflx_file: UploadFile = File(...),
    month: int = Form(0, ge=0, le=11),
    predict_time_steps: int = Form(1, ge=1, le=12),
    batch_size: int = Form(1, ge=1, le=16),
    var_name: Optional[VarNameEnum] = Form(None),
    include_raw: bool = Form(False),
    precision: int = Form(4, ge=1, le=10),
    # 经纬度范围参数
    lat_min: float = Form(LAT_MIN, ge=-90, le=90, description="纬度最小值"),
    lat_max: float = Form(LAT_MAX, ge=-90, le=90, description="纬度最大值"),
    lon_min: float = Form(LON_MIN, ge=0, le=360, description="经度最小值"),
    lon_max:  float = Form(LON_MAX, ge=0, le=360, description="经度最大值")
):
    """
    运行 ORCA-DL 模型预测并返回完整的反归一化后预测数组
    
    支持自定义经纬度范围，只返回指定区域的数据
    """
    try: 
        # 验证经纬度范围
        if lat_min >= lat_max: 
            raise HTTPException(status_code=400, detail=f"lat_min ({lat_min}) must be less than lat_max ({lat_max})")
        if lon_min >= lon_max:
            raise HTTPException(status_code=400, detail=f"lon_min ({lon_min}) must be less than lon_max ({lon_max})")
        
        # 获取经纬度索引
        lat_indices, lon_indices, actual_lats, actual_lons = get_lat_lon_indices(
            lat_min, lat_max, lon_min, lon_max
        )
        
        if len(actual_lats) == 0 or len(actual_lons) == 0:
            raise HTTPException(status_code=400, detail="指定的经纬度范围内没有数据点")
        
        files_dict = {
            'salt': await salt_file.read(),
            'pottmp': await pottmp_file.read(),
            'sst': await sst_file.read(),
            'ucur': await ucur_file.read(),
            'vcur': await vcur_file.read(),
            'sshg': await sshg_file.read(),
            'uflx': await uflx_file.read(),
            'vflx': await vflx_file.read()
        }
        
        ocean_vars, atmo_vars = process_input_files(files_dict, month)
        preds_raw = run_prediction(ocean_vars, atmo_vars, predict_time_steps, batch_size)
        
        denorm_preds = denormalize_predictions(preds_raw, month)
        
        # 构建返回的经纬度范围信息
        lat_range_info = {
            "requested":  {"min": lat_min, "max": lat_max},
            "actual":  {
                "min":  float(actual_lats[0]),
                "max": float(actual_lats[-1]),
                "points": len(actual_lats)
            }
        }
        lon_range_info = {
            "requested": {"min": lon_min, "max": lon_max},
            "actual":  {
                "min": float(actual_lons[0]),
                "max":  float(actual_lons[-1]),
                "points": len(actual_lons)
            }
        }
        
        result = {
            "metadata": {
                "month": month,
                "predict_time_steps": predict_time_steps,
                "batch_size": batch_size,
                "total_output_channels": sum(OUT_CHANS),
                "output_shape_description": "(batch, time_steps, channels, latitude, longitude)",
                "latitude_range": lat_range_info,
                "longitude_range":  lon_range_info,
                "precision":  precision,
                "full_grid_info": {
                    "latitude":  {"min": LAT_MIN, "max": LAT_MAX, "points":  LAT_POINTS},
                    "longitude": {"min":  LON_MIN, "max": LON_MAX, "points": LON_POINTS}
                }
            },
            "variables": {}
        }
        
        vars_to_process = [var_name.value] if var_name else VAR_LIST
        
        for vname in vars_to_process:
            var_data = denorm_preds[vname]
            var_desc = VAR_DESCRIPTIONS[vname]
            
            # 按经纬度范围切片
            var_data_sliced = slice_by_lat_lon(var_data, lat_indices, lon_indices)
            
            stats = compute_array_statistics(var_data_sliced)
            
            var_data_clean = sanitize_for_json(var_data_sliced)
            var_data_rounded = np.round(var_data_clean, precision)
            
            var_result = {
                "description": var_desc,
                "array_shape": list(var_data_sliced.shape),
                "statistics": stats,
                "spatial_range": {
                    "latitude": lat_range_info,
                    "longitude": lon_range_info
                },
                "data": var_data_rounded.tolist()
            }
            
            if include_raw: 
                var_idx = VAR_LIST.index(vname)
                start_ch = CHANNEL_OFFSETS[var_idx]
                end_ch = start_ch + OUT_CHANS[var_idx]
                
                if preds_raw.ndim == 4:
                    raw_data = preds_raw[:, np.newaxis, start_ch:end_ch, :, :]
                else:
                    raw_data = preds_raw[:, : , start_ch: end_ch, : , :]
                
                raw_data_sliced = slice_by_lat_lon(raw_data, lat_indices, lon_indices)
                raw_data_clean = sanitize_for_json(raw_data_sliced)
                var_result["raw_normalized_data"] = np.round(raw_data_clean, precision).tolist()
                var_result["raw_statistics"] = compute_array_statistics(raw_data_sliced)
            
            result["variables"][vname] = var_result
        
        result["grid_info"] = {
            "latitudes": [round(lat, 2) for lat in actual_lats.tolist()],
            "longitudes": [round(lon, 2) for lon in actual_lons.tolist()]
        }
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict_array_compact")
async def predict_array_compact(
    salt_file: UploadFile = File(...),
    pottmp_file: UploadFile = File(...),
    sst_file: UploadFile = File(...),
    ucur_file: UploadFile = File(...),
    vcur_file: UploadFile = File(...),
    sshg_file: UploadFile = File(...),
    uflx_file: UploadFile = File(...),
    vflx_file: UploadFile = File(...),
    month: int = Form(0, ge=0, le=11),
    predict_time_steps: int = Form(1, ge=1, le=12),
    batch_size: int = Form(1, ge=1, le=16),
    var_name: VarNameEnum = Form(VarNameEnum.tos),
    batch_idx: int = Form(0, ge=0),
    step_idx:  int = Form(0, ge=0),
    ch_in_var:  int = Form(0, ge=0),
    precision: int = Form(4, ge=1, le=10),
    # 经纬度范围参数
    lat_min: float = Form(LAT_MIN, ge=-90, le=90, description="纬度最小值"),
    lat_max:  float = Form(LAT_MAX, ge=-90, le=90, description="纬度最大值"),
    lon_min: float = Form(LON_MIN, ge=0, le=360, description="经度最小值"),
    lon_max: float = Form(LON_MAX, ge=0, le=360, description="经度最大值")
):
    """
    返回单个变量、单个时间步、单个通道的紧凑版预测数组（反归一化后）
    
    支持自定义经纬度范围，只返回指定区域的数据
    """
    try: 
        # 参数验证
        if batch_idx >= batch_size:
            raise HTTPException(status_code=400, detail=f"batch_idx ({batch_idx}) >= batch_size ({batch_size})")
        if step_idx >= predict_time_steps: 
            raise HTTPException(status_code=400, detail=f"step_idx ({step_idx}) >= predict_time_steps ({predict_time_steps})")
        
        var_idx_check = VAR_LIST.index(var_name.value)
        max_ch = OUT_CHANS[var_idx_check]
        if ch_in_var >= max_ch: 
            raise HTTPException(status_code=400, detail=f"ch_in_var ({ch_in_var}) >= max channels ({max_ch})")
        
        # 验证经纬度范围
        if lat_min >= lat_max:
            raise HTTPException(status_code=400, detail=f"lat_min ({lat_min}) must be less than lat_max ({lat_max})")
        if lon_min >= lon_max:
            raise HTTPException(status_code=400, detail=f"lon_min ({lon_min}) must be less than lon_max ({lon_max})")
        
        # 获取经纬度索引
        lat_indices, lon_indices, actual_lats, actual_lons = get_lat_lon_indices(
            lat_min, lat_max, lon_min, lon_max
        )
        
        if len(actual_lats) == 0 or len(actual_lons) == 0:
            raise HTTPException(status_code=400, detail="指定的经纬度范围内没有数据点")
        
        files_dict = {
            'salt': await salt_file.read(),
            'pottmp': await pottmp_file.read(),
            'sst': await sst_file.read(),
            'ucur':  await ucur_file.read(),
            'vcur': await vcur_file.read(),
            'sshg': await sshg_file.read(),
            'uflx':  await uflx_file.read(),
            'vflx': await vflx_file.read()
        }
        
        ocean_vars, atmo_vars = process_input_files(files_dict, month)
        preds_raw = run_prediction(ocean_vars, atmo_vars, predict_time_steps, batch_size)
        
        denorm_preds = denormalize_predictions(preds_raw, month)
        
        var_data = denorm_preds[var_name.value]
        
        if var_data.ndim == 4:
            var_data = var_data[:, np.newaxis, :, : , :]
        
        # 先提取特定的 batch, step, channel，再按经纬度切片
        slice_data = var_data[batch_idx, step_idx, ch_in_var, :, :]
        slice_data = slice_by_lat_lon(slice_data, lat_indices, lon_indices)
        
        stats = compute_array_statistics(slice_data)
        
        slice_data_clean = sanitize_for_json(slice_data)
        slice_data_rounded = np.round(slice_data_clean, precision)
        
        var_desc = VAR_DESCRIPTIONS[var_name.value]
        
        # 构建经纬度范围信息
        lat_range_info = {
            "requested": {"min": lat_min, "max": lat_max},
            "actual":  {
                "min": float(actual_lats[0]),
                "max":  float(actual_lats[-1]),
                "points": len(actual_lats)
            }
        }
        lon_range_info = {
            "requested": {"min": lon_min, "max": lon_max},
            "actual":  {
                "min": float(actual_lons[0]),
                "max":  float(actual_lons[-1]),
                "points": len(actual_lons)
            }
        }
        
        result = {
            "metadata": {
                "month":  month,
                "predict_time_steps": predict_time_steps,
                "batch_size": batch_size,
                "batch_idx": batch_idx,
                "step_idx": step_idx,
                "variable":  var_name.value,
                "channel": ch_in_var,
                "precision": precision
            },
            "variable_info": var_desc,
            "array_shape": list(slice_data.shape),
            "shape_description": "(latitude, longitude)",
            "statistics":  stats,
            "spatial_range": {
                "latitude":  lat_range_info,
                "longitude": lon_range_info,
                "full_grid":  {
                    "latitude": {"min":  LAT_MIN, "max": LAT_MAX, "points": LAT_POINTS},
                    "longitude": {"min": LON_MIN, "max": LON_MAX, "points": LON_POINTS}
                }
            },
            "grid":  {
                "latitudes": [round(lat, 2) for lat in actual_lats.tolist()],
                "longitudes": [round(lon, 2) for lon in actual_lons.tolist()]
            },
            "data":  slice_data_rounded.tolist()
        }
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.get("/variables_info")
async def get_variables_info():
    """获取所有输出变量的描述信息"""
    result = {
        "output_variables": {},
        "total_channels": sum(OUT_CHANS),
        "grid_info": {
            "latitude":  {"min": LAT_MIN, "max":  LAT_MAX, "points": LAT_POINTS},
            "longitude": {"min": LON_MIN, "max":  LON_MAX, "points": LON_POINTS}
        }
    }
    
    for i, var_name in enumerate(VAR_LIST):
        result["output_variables"][var_name] = {
            **VAR_DESCRIPTIONS[var_name],
            "channel_count": OUT_CHANS[i],
            "channel_offset": CHANNEL_OFFSETS[i]
        }
    
    return result


@app.get("/debug/stats")
async def debug_stats():
    """调试端点：查看加载的统计量信息"""
    result = {
        "input_stats": {},
        "output_stats": {}
    }
    
    for v in INPUT_VARIABLES:
        result["input_stats"][v] = {
            "mean_shape": list(stat['mean'][v].shape),
            "std_shape": list(stat['std'][v].shape),
            "mean_sample": float(stat['mean'][v].flat[0]) if stat['mean'][v].size > 0 else None,
            "std_sample": float(stat['std'][v].flat[0]) if stat['std'][v].size > 0 else None
        }
    
    for v in VAR_LIST: 
        result["output_stats"][v] = {
            "mean_shape": list(out_stat['mean'][v].shape),
            "std_shape": list(out_stat['std'][v].shape),
            "mean_sample":  float(out_stat['mean'][v].flat[0]) if out_stat['mean'][v].size > 0 else None,
            "std_sample": float(out_stat['std'][v].flat[0]) if out_stat['std'][v].size > 0 else None
        }
    
    return result
@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=51003,
        reload=False,
        workers=1
    )