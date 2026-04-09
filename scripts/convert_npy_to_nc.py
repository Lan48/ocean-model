#!/usr/bin/env python3
"""
将目录下的 .npy 文件转换为 .nc (NetCDF4) 文件的实用脚本 - 修复版

用法示例:
    python convert_npy_to_nc.py /path/to/npy/directory

可选参数:
    --outdir 输出目录（默认在源目录内创建 nc_output 子目录）
    --overwrite 覆盖已存在的 .nc 文件

脚本行为:
- 扫描目录中的所有 .npy 文件（跳过可能的坐标文件名 lats.npy/lons.npy），
  对每个数据文件尝试查找同名的坐标文件（例如 data_lat.npy / data_lon.npy 或 lats.npy / lons.npy），
  并将数组写入 NetCDF 文件，自动选择维度名称（1D/2D/3D/4D 支持）

注意: 需要安装依赖: numpy, netCDF4
    pip install numpy netCDF4
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

try:
    from netCDF4 import Dataset
except Exception as e:
    print("需要 netCDF4 库: pip install netCDF4", file=sys.stderr)
    raise


def guess_dim_names(ndim):
    # 根据维度数量返回维度名称元组
    if ndim == 1:
        return ("x",)
    if ndim == 2:
        return ("lat", "lon")
    if ndim == 3:
        return ("time", "lat", "lon")
    if ndim == 4:
        return ("time", "lat", "lon")  # 移除lev维度，简化处理
    # fallback
    return tuple(f"dim{i}" for i in range(ndim))


def load_if_exists(path: Path):
    if path.exists():
        try:
            return np.load(str(path))
        except Exception:
            return None
    return None


def safe_convert_dtype(arr):
    """安全地转换数据类型以避免HDF错误"""
    original_dtype = arr.dtype
    
    # 处理复数类型
    if np.issubdtype(original_dtype, np.complexfloating):
        print(f"警告: 转换复数类型 {original_dtype} 为 float64", file=sys.stderr)
        return arr.real.astype(np.float64)
    
    # 处理对象类型
    if original_dtype == np.object_:
        print(f"警告: 转换 object 类型为 float64", file=sys.stderr)
        return np.array(arr, dtype=np.float64)
    
    # 处理布尔类型
    if original_dtype == np.bool_:
        return arr.astype(np.uint8)
    
    # 处理大整数类型
    if np.issubdtype(original_dtype, np.integer):
        if original_dtype.itemsize > 4:  # 超过32位
            if original_dtype in [np.int64, np.uint64]:
                # 检查值范围，决定是否降级
                if np.min(arr) >= np.iinfo(np.int32).min and np.max(arr) <= np.iinfo(np.int32).max:
                    return arr.astype(np.int32)
                else:
                    return arr.astype(np.float64)
            else:
                return arr.astype(np.int32)
        else:
            return arr.astype(np.int32)
    
    # 处理浮点类型
    if np.issubdtype(original_dtype, np.floating):
        if original_dtype == np.float64:
            # 检查是否真的需要双精度
            if np.allclose(arr, arr.astype(np.float32)):
                return arr.astype(np.float32)
            else:
                return arr.astype(np.float64)
        elif original_dtype == np.float16:
            return arr.astype(np.float32)
        else:
            return arr.astype(np.float32)
    
    # 默认情况，转换为float32
    return arr.astype(np.float32)


def convert_npy_to_nc(npy_path: Path, out_path: Path, lat_arr=None, lon_arr=None, overwrite=False):
    if out_path.exists() and not overwrite:
        print(f"跳过已存在文件: {out_path}")
        return False

    try:
        arr = np.load(str(npy_path))
    except Exception as e:
        print(f"无法加载 numpy 文件 {npy_path}: {e}", file=sys.stderr)
        return False

    if not isinstance(arr, np.ndarray):
        print(f"文件不是 numpy 数组: {npy_path}", file=sys.stderr)
        return False

    # 检查数组是否为空
    if arr.size == 0:
        print(f"警告: 数组为空 {npy_path}, 跳过", file=sys.stderr)
        return False

    # 确保使用本机字节序，避免因字节序不兼容导致的底层 HDF 错误
    if arr.dtype.byteorder not in ('=', '|'):
        print(f"转换字节序: {arr.dtype} -> {arr.dtype.newbyteorder('=')}")
        arr = arr.byteswap().newbyteorder('=')

    # 安全转换数据类型
    try:
        arr = safe_convert_dtype(arr)
    except Exception as e:
        print(f"数据类型转换失败 {npy_path}: {e}", file=sys.stderr)
        return False

    dims = guess_dim_names(arr.ndim)

    # 使用 try/except 捕获 netCDF4 层面的错误并打印更多调试信息
    ds = None
    try:
        # 创建数据集
        ds = Dataset(str(out_path), "w", format="NETCDF4_CLASSIC")  # 使用CLASSIC格式更稳定
        
        # 创建维度
        for i, dim_name in enumerate(dims):
            size = arr.shape[i]
            # 对于超大维度，给出警告
            if size > 100000:
                print(f"警告: 维度 {dim_name} 大小为 {size}, 可能导致内存问题")
            ds.createDimension(dim_name, size)

        # 创建坐标变量（若提供 lat/lon）
        if "lat" in dims:
            lat_idx = dims.index("lat")
            lat_len = arr.shape[lat_idx]
            
            if lat_arr is not None and len(lat_arr) == lat_len:
                lat_var = ds.createVariable("lat", "f4", ("lat",))
                lat_var[:] = lat_arr[:lat_len]  # 确保长度匹配
                lat_var.units = "degrees_north"
                lat_var.long_name = "latitude"
            else:
                # 生成简单的索引坐标
                lat_var = ds.createVariable("lat", "f4", ("lat",))
                lat_var[:] = np.linspace(-90, 90, lat_len)  # 假设为全球数据
                lat_var.units = "degrees_north"
                lat_var.long_name = "latitude"

        if "lon" in dims:
            lon_idx = dims.index("lon")
            lon_len = arr.shape[lon_idx]
            
            if lon_arr is not None and len(lon_arr) == lon_len:
                lon_var = ds.createVariable("lon", "f4", ("lon",))
                lon_var[:] = lon_arr[:lon_len]  # 确保长度匹配
                lon_var.units = "degrees_east"
                lon_var.long_name = "longitude"
            else:
                # 生成简单的索引坐标
                lon_var = ds.createVariable("lon", "f4", ("lon",))
                lon_var[:] = np.linspace(-180, 180, lon_len)  # 假设为全球数据
                lon_var.units = "degrees_east"
                lon_var.long_name = "longitude"

        # 变量名采用 npy 文件名（不带扩展）
        varname = npy_path.stem
        
        # 创建变量 - 使用安全的数据类型
        netcdf_dtype = arr.dtype.name
        if netcdf_dtype.startswith('int') or netcdf_dtype.startswith('uint'):
            if arr.dtype.itemsize > 4:
                netcdf_dtype = 'i4'  # 强制使用32位整数
            else:
                netcdf_dtype = 'i4' if 'int' in netcdf_dtype else 'u4'
        elif netcdf_dtype.startswith('float'):
            netcdf_dtype = 'f4'  # 强制使用32位浮点
        else:
            netcdf_dtype = 'f4'  # 默认浮点
        
        v = ds.createVariable(varname, netcdf_dtype, dims, zlib=True, complevel=1)  # 启用压缩
        
        # 分块写入大数据数组以避免内存问题
        if arr.nbytes > 100 * 1024 * 1024:  # 如果大于100MB
            print(f"大数组 ({arr.nbytes / 1024 / 1024:.1f} MB)，分块写入...")
            # 对于4维数组，按第一个维度分块
            chunk_size = max(1, 100 // arr.ndim)  # 动态调整块大小
            start = 0
            while start < arr.shape[0]:
                end = min(start + chunk_size, arr.shape[0])
                if arr.ndim == 1:
                    v[start:end] = arr[start:end]
                elif arr.ndim == 2:
                    v[start:end, :] = arr[start:end, :]
                elif arr.ndim == 3:
                    v[start:end, :, :] = arr[start:end, :, :]
                elif arr.ndim == 4:
                    v[start:end, :, :, :] = arr[start:end, :, :, :]
                start = end
        else:
            v[:] = arr

        # 添加标准属性
        v.description = f"Data from {npy_path.name}"
        v.long_name = varname.replace('_', ' ').title()
        ds.Conventions = "CF-1.6"
        ds.title = f"Converted from {npy_path.name}"
        ds.institution = "Converted using convert_npy_to_nc.py"
        ds.source = "Converted from .npy by convert_npy_to_nc.py"
        ds.history = f"Created by convert_npy_to_nc.py on {np.datetime64('now')}"
        ds.original_npy = str(npy_path)

        print(f"已写入: {out_path} (shape={arr.shape}, dtype={arr.dtype})")
        return True
        
    except Exception as e:
        print(f"写入 NetCDF 失败 for {npy_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
    finally:
        if ds:
            try:
                ds.close()
            except Exception:
                pass  # 忽略关闭时的错误


def find_coord_candidates(src_dir: Path, base_stem: str):
    # 尝试找到 lat/lon 文件: base_lat.npy, base_lon.npy, lats.npy, lons.npy, lat.npy, lon.npy
    candidates = {
        "lat": [src_dir / f"{base_stem}_lat.npy", src_dir / "lats.npy", src_dir / "lat.npy"],
        "lon": [src_dir / f"{base_stem}_lon.npy", src_dir / "lons.npy", src_dir / "lon.npy"]
    }
    out = {"lat": None, "lon": None}
    for key, paths in candidates.items():
        for p in paths:
            if p.exists():
                try:
                    arr = np.load(str(p))
                    if isinstance(arr, np.ndarray) and arr.ndim == 1:  # 确保是一维坐标数组
                        out[key] = arr
                        print(f"找到坐标文件: {p}")
                        break
                except Exception as e:
                    print(f"加载坐标文件失败 {p}: {e}")
                    continue
    return out["lat"], out["lon"]


def main():
    p = argparse.ArgumentParser(description="批量将 .npy 转为 .nc - 修复版")
    p.add_argument("src_dir", help="包含 .npy 文件的目录")
    p.add_argument("--outdir", help="输出目录（默认 src_dir/nc_output）", default=None)
    p.add_argument("--overwrite", help="覆盖已存在的 .nc 文件", action="store_true")
    p.add_argument("--pattern", help="只转换匹配模式的文件（shell-style），默认 *.npy", default="*.npy")

    args = p.parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    if not src_dir.is_dir():
        print(f"目录不存在: {src_dir}")
        sys.exit(2)

    out_dir = Path(args.outdir) if args.outdir else src_dir / "nc_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted([p for p in src_dir.glob(args.pattern) if p.is_file() and p.suffix == ".npy"]) 
    
    # 过滤掉可能的坐标文件，保留数据文件
    filtered = []
    for f in npy_files:
        name = f.name.lower()
        if name in ("lats.npy", "lat.npy", "lons.npy", "lon.npy"):
            continue
        if name.endswith("_lat.npy") or name.endswith("_lon.npy"):
            continue
        filtered.append(f)

    if not filtered:
        print("未找到待转换的 .npy 数据文件（已跳过坐标文件）。")
        sys.exit(0)

    print(f"找到 {len(filtered)} 个待转换文件")
    
    success_count = 0
    for i, npy_file in enumerate(filtered):
        print(f"\n处理 {i+1}/{len(filtered)}: {npy_file.name}")
        
        base = npy_file.stem
        lat_arr, lon_arr = find_coord_candidates(src_dir, base)
        out_file = out_dir / (base + ".nc")
        
        try:
            ok = convert_npy_to_nc(npy_file, out_file, lat_arr=lat_arr, lon_arr=lon_arr, overwrite=args.overwrite)
            if ok:
                success_count += 1
        except Exception as e:
            print(f"处理失败: {npy_file}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print(f"\n完成：成功转换 {success_count}/{len(filtered)} 个文件，输出目录: {out_dir}")


if __name__ == '__main__':
    main()