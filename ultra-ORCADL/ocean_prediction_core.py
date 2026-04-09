"""
ORCA-DL 海洋预测核心功能模块
包含五大核心预测维度的实现

基于 OpenEarthLab/ORCA-DL 模型定义
"""

import os
import torch
import numpy as np
try:
    import xarray as xr
except ImportError:
    xr = None
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# 从仓库中导入模型
from model import ORCADLModel, ORCADLConfig
from variable_config import (
    ALL_SINGLE_LEVEL_VARS,
    DEFAULT_MODEL_ATMO_VARS,
    DEFAULT_MODEL_OCEAN_VARS,
    MODEL_TO_STAT_MAPPING,
    STAT_TO_MODEL_MAPPING,
    get_var_channels,
    normalize_to_model_var,
)


@dataclass
class OceanPredictionResult:
    """海洋预测结果数据类"""
    predictions: np.ndarray          # 原始预测结果
    timestamps: List[str]            # 预测时间戳
    variables: List[str]             # 变量列表
    metadata: Dict                   # 元数据


class ORCADLPredictor:
    """ORCA-DL 海洋状态预测器"""
    
    # 模型内部变量名（CMIP6命名）
    MODEL_OCEAN_VARIABLES = list(DEFAULT_MODEL_OCEAN_VARS)
    MODEL_ATMO_VARIABLES = list(DEFAULT_MODEL_ATMO_VARS)
    
    # 统计文件变量名（GODAS命名）
    STAT_OCEAN_VARIABLES = [MODEL_TO_STAT_MAPPING.get(v, v) for v in MODEL_OCEAN_VARIABLES]
    STAT_ATMO_VARIABLES = [MODEL_TO_STAT_MAPPING.get(v, v) for v in MODEL_ATMO_VARIABLES]
    
    # 模型变量 -> 统计文件变量 映射
    MODEL_TO_STAT_MAPPING = dict(MODEL_TO_STAT_MAPPING)
    
    # 统计文件变量 -> 模型变量 映射（反向）
    STAT_TO_MODEL_MAPPING = dict(STAT_TO_MODEL_MAPPING)
    
    # 每个变量的通道数
    VARIABLE_CHANNELS = [get_var_channels(v) for v in MODEL_OCEAN_VARIABLES]
    
    # 2D变量（无深度维度）
    SINGLE_LEVEL_VARS = sorted(ALL_SINGLE_LEVEL_VARS)
    
    def __init__(
        self,
        config_path: str = './model_config.json',
        checkpoint_path: str = './ckpt/seed_1.bin',
        stat_dir:  str = './stat',
        device: str = None
    ):
        """
        初始化预测器
        
        Args:
            config_path: 模型配置文件路径
            checkpoint_path: 模型权重路径
            stat_dir: 统计数据（均值/标准差）目录
            device: 计算设备 ('cuda' / 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型配置和权重
        self.config = ORCADLConfig.from_json_file(config_path)
        self.model = ORCADLModel(self.config)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()

        self.MODEL_OCEAN_VARIABLES = list(self.config.var_list)
        self.MODEL_ATMO_VARIABLES = list(getattr(self.config, "atmo_var_list", DEFAULT_MODEL_ATMO_VARS))
        self.STAT_OCEAN_VARIABLES = [self.MODEL_TO_STAT_MAPPING.get(v, v) for v in self.MODEL_OCEAN_VARIABLES]
        self.STAT_ATMO_VARIABLES = [self.MODEL_TO_STAT_MAPPING.get(v, v) for v in self.MODEL_ATMO_VARIABLES]
        self.VARIABLE_CHANNELS = [get_var_channels(v) for v in self.MODEL_OCEAN_VARIABLES]
        
        # 加载统计数据用于标准化/反标准化
        self.stat_dir = stat_dir
        self.stat = self._load_statistics(stat_dir)
        
        # 通道分割点（用于分离各变量）
        self.split_chans = list(np.cumsum(self.VARIABLE_CHANNELS))

    def _build_zero_input(self, var_name: str, use_model_naming: bool = True) -> np.ndarray:
        model_var = normalize_to_model_var(var_name if use_model_naming else self._get_model_var_name(var_name))
        channels = get_var_channels(model_var)
        h, w = tuple(self.config.input_shape)
        if channels == 1:
            return np.zeros((h, w), dtype=np.float32)
        return np.zeros((channels, h, w), dtype=np.float32)

    def _prepare_input_value(
        self,
        data_dict: Dict[str, np.ndarray],
        var_name: str,
        use_model_naming: bool = True
    ) -> np.ndarray:
        data = data_dict.get(var_name)
        if data is None:
            return self._build_zero_input(var_name, use_model_naming)

        arr = np.asarray(data, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 0:
            return self._build_zero_input(var_name, use_model_naming)
        return arr
        
    def _load_statistics(self, stat_dir: str) -> Dict:
        """
        加载均值和标准差统计数据
        使用统计文件的命名（GODAS命名）
        """
        stat = {'mean': {}, 'std':  {}}
        
        # 加载所有统计文件（使用GODAS命名）
        all_stat_vars = self.STAT_OCEAN_VARIABLES + self.STAT_ATMO_VARIABLES
        
        for stat_var in all_stat_vars:
            mean_path = os.path.join(stat_dir, 'mean', f'{stat_var}.npy')
            std_path = os.path.join(stat_dir, 'std', f'{stat_var}.npy')
            
            if os.path.exists(mean_path) and os.path.exists(std_path):
                stat['mean'][stat_var] = np.load(mean_path)
                stat['std'][stat_var] = np.load(std_path)
            else:
                print(f"Warning: Statistics not found for {stat_var}")
        
        return stat
    
    def _get_stat_var_name(self, model_var: str) -> str:
        """将模型变量名转换为统计文件变量名"""
        return self.MODEL_TO_STAT_MAPPING.get(model_var, model_var)
    
    def _get_model_var_name(self, stat_var: str) -> str:
        """将统计文件变量名转换为模型变量名"""
        return self.STAT_TO_MODEL_MAPPING.get(stat_var, stat_var)
    
    def _normalize_data(
        self,
        data: np.ndarray,
        var_name: str,
        month: Union[int, np.ndarray, List[int]],
        use_model_naming: bool = True
    ) -> np.ndarray:
        """
        标准化输入数据，支持多种维度模式
        
        Args: 
            data: 输入数据，支持以下形状: 
                  - 2D变量:  (H, W), (B, H, W), (B, T, H, W)
                  - 3D变量: (D, H, W), (B, D, H, W), (B, T, D, H, W)
            var_name: 变量名（可以是模型命名或统计文件命名）
            month: 月份索引，支持: 
                   - int: 单个月份 (0-11)
                   - array/list: 批次中每个样本的月份
            use_model_naming: True=var_name使用模型命名，False=使用统计文件命名
            
        Returns:
            标准化后的数据，形状与输入相同
        """
        # 获取统计文件变量名
        if use_model_naming:
            stat_var = self._get_stat_var_name(var_name)
        else:
            stat_var = var_name
        
        if stat_var not in self.stat['mean']:
            return data.astype(np.float32, copy=False)
        
        mean_all = self.stat['mean'][stat_var]  # shape: (12, ...) 或 (12, D, H, W)
        std_all = self.stat['std'][stat_var]
        
        # 判断变量维度类型
        is_2d_var = var_name in self.SINGLE_LEVEL_VARS or stat_var in self.SINGLE_LEVEL_VARS
        return self._apply_normalization(data, mean_all, std_all, month, is_2d_var, normalize=True)
    
    def _denormalize_data(
        self,
        data: np.ndarray,
        var_name:  str,
        month: Union[int, np.ndarray, List[int]],
        use_model_naming: bool = True
    ) -> np.ndarray:
        """
        反标准化预测数据，支持多种维度模式
        
        Args:
            data: 预测数据，支持以下形状:
                  - single step: (C, H, W) 或 (B, C, H, W)
                  - multi steps: (B, T, C, H, W)
            var_name: 变量名
            month: 月份索引，支持:
                   - int: 单个月份
                   - array/list: 每个时间步的月份
            use_model_naming: True=var_name使用模型命名
            
        Returns:
            反标准化后的数据
        """
        # 获取统计文件变量名
        if use_model_naming: 
            stat_var = self._get_stat_var_name(var_name)
        else:
            stat_var = var_name
        
        if stat_var not in self.stat['mean']:
            return data.astype(np.float32, copy=False)
        
        mean_all = self.stat['mean'][stat_var]
        std_all = self.stat['std'][stat_var]
        
        # 判断变量维度类型
        is_2d_var = var_name in self.SINGLE_LEVEL_VARS or stat_var in self.SINGLE_LEVEL_VARS
        
        return self._apply_normalization(data, mean_all, std_all, month, is_2d_var, normalize=False)
    
    def _apply_normalization(
        self,
        data: np.ndarray,
        mean_all: np.ndarray,
        std_all: np.ndarray,
        month: Union[int, np.ndarray, List[int]],
        is_2d_var: bool,
        normalize: bool = True
    ) -> np.ndarray:
        """
        应用标准化或反标准化
        
        统计数据形状: 
        - 2D变量: mean_all shape = (12, H, W)
        - 3D变量: mean_all shape = (12, D, H, W)
        
        数据形状模式:
        - 输入 2D变量: (H, W), (B, H, W), (B, T, H, W)
        - 输入 3D变量: (D, H, W), (B, D, H, W), (B, T, D, H, W)
        - 输出 single step: (C, H, W), (B, C, H, W)
        - 输出 multi steps: (B, T, C, H, W)
        """
        ndim = len(data.shape)
        
        # 统一处理 month 参数
        if isinstance(month, (list, np.ndarray)):
            months = np.array(month)
        else:
            months = np.array([month])
        
        # 确保月份在有效范围内
        months = months % 12
        
        # 根据数据维度确定处理方式
        if is_2d_var:
            # 2D变量处理
            result = self._normalize_2d_var(data, mean_all, std_all, months, normalize)
        else:
            # 3D变量处理
            result = self._normalize_3d_var(data, mean_all, std_all, months, normalize)
        
        return result
    
    def _normalize_2d_var(
        self,
        data: np.ndarray,
        mean_all: np.ndarray,
        std_all: np.ndarray,
        months: np.ndarray,
        normalize: bool
    ) -> np.ndarray:
        """
        处理2D变量的标准化/反标准化
        
        数据形状:  (H, W), (B, H, W), (B, T, H, W), (C, H, W), (B, C, H, W), (B, T, C, H, W)
        统计形状: (12, H, W)
        """
        ndim = len(data.shape)
        
        if ndim == 2:
            # (H, W) - 单个样本，单个时间步
            mean = mean_all[months[0]]
            std = std_all[months[0]]
        
        elif ndim == 3:
            # (B, H, W) 或 (C, H, W)
            if len(months) == 1:
                # 所有样本/通道使用相同月份
                mean = mean_all[months[0]]
                std = std_all[months[0]]
            else:
                # 每个样本使用不同月份
                mean = np.stack([mean_all[m] for m in months], axis=0)
                std = np.stack([std_all[m] for m in months], axis=0)
        
        elif ndim == 4:
            # (B, C, H, W) 或 (B, T, H, W)
            if len(months) == 1:
                mean = mean_all[months[0]]
                std = std_all[months[0]]
            elif len(months) == data.shape[0]:
                # 每个batch样本不同月份
                mean = np.stack([mean_all[m] for m in months], axis=0)[:, np.newaxis, : , :]
                std = np.stack([std_all[m] for m in months], axis=0)[:, np.newaxis, :, :]
            else:
                # 每个时间步不同月份
                mean = np.stack([mean_all[m] for m in months], axis=0)[np.newaxis, :, : , :]
                std = np.stack([std_all[m] for m in months], axis=0)[np.newaxis, :, :, :]
        
        elif ndim == 5:
            # (B, T, C, H, W)
            B, T = data.shape[:2]
            if len(months) == 1:
                mean = mean_all[months[0]]
                std = std_all[months[0]]
            elif len(months) == T:
                # 每个时间步不同月份
                mean = np.stack([mean_all[m] for m in months], axis=0)
                mean = mean[np.newaxis, :, np.newaxis, :, :]  # (1, T, 1, H, W)
                std = np.stack([std_all[m] for m in months], axis=0)
                std = std[np.newaxis, :, np.newaxis, :, :]
            else:
                raise ValueError(f"Month array length {len(months)} doesn't match data shape {data.shape}")
        else:
            raise ValueError(f"Unsupported data dimension: {ndim}")
        
        # 执行标准化或反标准化
        if normalize:
            return (data - mean) / (std + 1e-8)
        else:
            return data * std + mean
    
    def _normalize_3d_var(
        self,
        data:  np.ndarray,
        mean_all: np.ndarray,
        std_all: np.ndarray,
        months: np.ndarray,
        normalize: bool
    ) -> np.ndarray:
        """
        处理3D变量的标准化/反标准化（带深度维度）
        
        数据形状: (D, H, W), (B, D, H, W), (B, T, D, H, W)
        统计形状: (12, D, H, W)
        """
        ndim = len(data.shape)
        
        if ndim == 3:
            # (D, H, W) - 单个样本
            mean = mean_all[months[0]]
            std = std_all[months[0]]
        
        elif ndim == 4:
            # (B, D, H, W)
            if len(months) == 1:
                mean = mean_all[months[0]]
                std = std_all[months[0]]
            else: 
                # 每个batch样本不同月份
                mean = np.stack([mean_all[m] for m in months], axis=0)
                std = np.stack([std_all[m] for m in months], axis=0)
        
        elif ndim == 5:
            # (B, T, D, H, W)
            B, T = data.shape[:2]
            if len(months) == 1:
                mean = mean_all[months[0]]
                std = std_all[months[0]]
            elif len(months) == T:
                # 每个时间步不同月份
                mean = np.stack([mean_all[m] for m in months], axis=0)
                mean = mean[np.newaxis, :, : , : , :]  # (1, T, D, H, W)
                std = np.stack([std_all[m] for m in months], axis=0)
                std = std[np.newaxis, :, :, :, :]
            elif len(months) == B:
                # 每个batch不同起始月份
                mean = np.stack([mean_all[m] for m in months], axis=0)
                mean = mean[: , np.newaxis, :, :, :]  # (B, 1, D, H, W)
                std = np.stack([std_all[m] for m in months], axis=0)
                std = std[:, np.newaxis, :, : , :]
            else: 
                raise ValueError(f"Month array length {len(months)} doesn't match data shape {data.shape}")
        else:
            raise ValueError(f"Unsupported data dimension: {ndim}")
        
        # 执行标准化或反标准化
        if normalize: 
            return (data - mean) / (std + 1e-8)
        else:
            return data * std + mean
    
    def normalize_batch(
        self,
        ocean_data: Dict[str, np.ndarray],
        atmo_data: Dict[str, np.ndarray],
        months: Union[int, np.ndarray, List[int]],
        use_model_naming: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量标准化输入数据并转换为模型输入格式
        
        Args:
            ocean_data: 海洋变量字典
            atmo_data: 大气变量字典
            months: 月份（支持批次）
            use_model_naming: 输入字典使用模型命名
            
        Returns:
            ocean_vars, atmo_vars: 标准化后的张量
        """
        var_names = self.MODEL_OCEAN_VARIABLES if use_model_naming else self.STAT_OCEAN_VARIABLES
        atmo_names = self.MODEL_ATMO_VARIABLES if use_model_naming else self.STAT_ATMO_VARIABLES
        
        # 标准化海洋变量
        ocean_vars_list = []
        for var in var_names:
            data = self._prepare_input_value(ocean_data, var, use_model_naming)
            normed = self._normalize_data(data, var, months, use_model_naming)
            # 处理维度
            if len(normed.shape) == 2:  # (H, W) -> (1, H, W)
                normed = normed[np.newaxis, ...]
            ocean_vars_list.append(normed)
        
        ocean_vars = np.nan_to_num(np.concatenate(ocean_vars_list, axis=0))
        
        # 标准化大气变量
        atmo_vars_list = []
        for var in atmo_names:
            data = self._prepare_input_value(atmo_data, var, use_model_naming)
            normed = self._normalize_data(data, var, months, use_model_naming)
            if len(normed.shape) == 2:
                normed = normed[np.newaxis, ...]
            atmo_vars_list.append(normed)
        
        atmo_vars = np.nan_to_num(np.concatenate(atmo_vars_list, axis=0))
        
        # 添加batch维度如果需要
        if len(ocean_vars.shape) == 3: 
            ocean_vars = ocean_vars[np.newaxis, ...]
            atmo_vars = atmo_vars[np.newaxis, ...]
        
        return (
            torch.from_numpy(ocean_vars).float().to(self.device),
            torch.from_numpy(atmo_vars).float().to(self.device)
        )
    
    def reformat_batch(
        self,
        ocean_data: Dict[str, np.ndarray],
        atmo_data: Dict[str, np.ndarray],
        ocean_var_names: List[str],
        atmo_var_names: List[str],
        use_model_naming: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        只改变数据格式但不处理数据内容的批量重构函数
        
        Args:
            ocean_data: 海洋变量字典 {变量名: 数据数组}
            atmo_data: 大气变量字典 {变量名: 数据数组}
            ocean_var_names: 海洋变量名称列表，决定变量顺序
            atmo_var_names: 大气变量名称列表，决定变量顺序
            
        Returns:
            ocean_tensor, atmo_tensor: 仅格式改变后的张量，数据内容保持不变
        """
        # 重构海洋变量格式
        ocean_vars_list = []
        for var_name in ocean_var_names:
            data = self._prepare_input_value(ocean_data, var_name, use_model_naming)
            
            # 只调整维度，不改变数值
            if len(data.shape) == 2:  # (H, W) -> (1, H, W)
                data = data[np.newaxis, ...]
            elif len(data.shape) == 3:  # 已经是 (C, H, W) 或 (T, H, W)
                pass  # 保持原样
            else:
                raise ValueError(f"海洋变量 {var_name} 的维度 {data.shape} 不支持")
            
            ocean_vars_list.append(data)
        
        # 沿通道维度拼接，不进行数值处理
        ocean_vars = np.concatenate(ocean_vars_list, axis=0)
        # 将 NaN 替换为 0（与 normalize_batch 的处理保持一致）
        ocean_vars = np.nan_to_num(ocean_vars)
        
        # 重构大气变量格式
        atmo_vars_list = []
        for var_name in atmo_var_names:
            data = self._prepare_input_value(atmo_data, var_name, use_model_naming)
            
            # 只调整维度，不改变数值
            if len(data.shape) == 2:  # (H, W) -> (1, H, W)
                data = data[np.newaxis, ...]
            elif len(data.shape) == 3:  # 已经是 (C, H, W) 或 (T, H, W)
                pass  # 保持原样
            else:
                raise ValueError(f"大气变量 {var_name} 的维度 {data.shape} 不支持")
            
            atmo_vars_list.append(data)
        
        # 沿通道维度拼接，不进行数值处理
        atmo_vars = np.concatenate(atmo_vars_list, axis=0)
        # 将 NaN 替换为 0
        atmo_vars = np.nan_to_num(atmo_vars)
        
        # 添加batch维度
        if len(ocean_vars.shape) == 3: 
            ocean_vars = ocean_vars[np.newaxis, ...]  # (C, H, W) -> (1, C, H, W)
            atmo_vars = atmo_vars[np.newaxis, ...]     # (C, H, W) -> (1, C, H, W)
        
        # 转换为张量，不改变数值，并放到预测器设备上
        return (
            torch.from_numpy(ocean_vars).float().to(self.device),
            torch.from_numpy(atmo_vars).float().to(self.device)
        )

    def denormalize_output(
        self,
        predictions: np.ndarray,
        start_month: int,
        predict_steps: int = None
    ) -> Dict[str, np.ndarray]: 
        """
        反标准化模型输出
        
        Args:
            predictions: 模型输出，支持: 
                - single step: (B, C, H, W) 或 (C, H, W)
                - multi steps: (B, T, C, H, W)
            start_month: 起始月份
            predict_steps: 预测步数（multi steps模式需要）
            
        Returns: 
            Dict:  各变量反标准化后的数据
        """
        print("Denormalizing output with shape:", predictions.shape)
        ndim = len(predictions.shape)
        results = {}
        # 判断输出模式
        if ndim == 3:
            # (C, H, W) - single step, single sample
            mode = 'single'
            predict_steps = 1
        elif ndim == 4:
            # (B, C, H, W) - single step, batch
            mode = 'single_batch'
            predict_steps = 1
        elif ndim == 5:
            # (B, T, C, H, W) - multi steps
            mode = 'multi'
            predict_steps = predictions.shape[1]
        else:
            raise ValueError(f"Unsupported prediction shape: {predictions.shape}")
        
        # 按变量分割
        for i, var in enumerate(self.MODEL_OCEAN_VARIABLES):
            if i == 0:
                start_ch = 0
            else: 
                start_ch = self.split_chans[i-1]
            end_ch = self.split_chans[i]
            
            if mode == 'single':
                var_pred = predictions[start_ch:end_ch]
                pred_month = (start_month + 1) % 12
                results[var] = self._denormalize_data(var_pred, var, pred_month)
                
            elif mode == 'single_batch':
                var_pred = predictions[: , start_ch:end_ch]
                pred_month = (start_month + 1) % 12
                results[var] = self._denormalize_data(var_pred, var, pred_month)
                
            elif mode == 'multi':
                var_pred = predictions[:, : , start_ch:end_ch]
                # 每个时间步的月份
                pred_months = [(start_month + t + 1) % 12 for t in range(predict_steps)]
                results[var] = self._denormalize_data(var_pred, var, pred_months)
        
        return results


# =====================================
# 函数1: 预测未来海洋状态
# =====================================
def predict_ocean_state(
    predictor: ORCADLPredictor,
    ocean_data: Dict[str, np.ndarray],
    atmo_data: Dict[str, np.ndarray],
    start_month: int,
    predict_steps: int = 6,
    denormalize: bool = False,
    use_model_naming:  bool = True,
    batch_mode: bool = False,
    normalized: bool = False
) -> OceanPredictionResult:
    """
    读取过去海洋状态数据，使用ORCA-DL模型预测未来海洋状态
    
    支持三种输出模式: 
    - single step: predict_steps=1, 输出 (B, C, H, W)
    - multi steps: predict_steps>1, 输出 (B, T, C, H, W)
    - batch input: 输入数据带batch维度
    
    Args:
        predictor:  ORCA-DL预测器实例
        ocean_data: 海洋变量数据字典
                   - 使用模型命名:  so, thetao, tos, uo, vo, zos
                   - 使用统计命名: salt, pottmp, sst, ucur, vcur, sshg
        atmo_data: 大气变量数据字典
                   - 使用模型命名: tauu, tauv
                   - 使用统计命名: uflx, vflx
        start_month: 起始月份 (0-11)
        predict_steps: 预测步数（月数）
        denormalize: 是否反标准化输出
        use_model_naming: 输入数据是否使用模型变量命名
        batch_mode: 输入数据是否已有batch维度
        
    Returns:
        OceanPredictionResult: 包含预测结果的数据类
    """
    # 检查当前数据的键名
    print("ocean_data keys:", list(ocean_data.keys()))
    print("atmo_data keys:", list(atmo_data.keys()))

    # 对比期望的键名
    print("期望的GODAS海洋变量:", predictor.STAT_OCEAN_VARIABLES)
    print("期望的GODAS大气变量:", predictor.STAT_ATMO_VARIABLES)
    # 1.标准化并准备输入
    if normalized == False:
        ocean_vars, atmo_vars = predictor.normalize_batch(
            ocean_data, atmo_data, start_month, use_model_naming
        )
    else:
        # 当输入已标准化/只需重排格式时，按照 use_model_naming 选择变量顺序
        ocean_var_names = predictor.MODEL_OCEAN_VARIABLES if use_model_naming else predictor.STAT_OCEAN_VARIABLES
        atmo_var_names = predictor.MODEL_ATMO_VARIABLES if use_model_naming else predictor.STAT_ATMO_VARIABLES
        ocean_vars, atmo_vars = predictor.reformat_batch(
            ocean_data, atmo_data, ocean_var_names, atmo_var_names, use_model_naming
        )
    
    # 2.模型推理
    with torch.no_grad():
        output = predictor.model(
            ocean_vars=ocean_vars,
            atmo_vars=atmo_vars,
            predict_time_steps=predict_steps
        )
    
    # 3.处理输出
    preds = output.preds.detach().cpu().numpy()
    
    # 4.可选反标准化
    if denormalize:
        results_by_var = predictor.denormalize_output(preds, start_month, predict_steps)
    else:
        results_by_var = None
    
    # 5.按时间步组织结果
    results_by_step = {}
    if denormalize:
        for step in range(predict_steps):
            step_results = {}
            for var in predictor.MODEL_OCEAN_VARIABLES:
                var_data = results_by_var[var]
                if len(var_data.shape) == 5:  # (B, T, C, H, W)
                    step_results[var] = var_data[: , step]
                elif len(var_data.shape) == 4:  # (B, C, H, W)
                    step_results[var] = var_data
                else:
                    step_results[var] = var_data
            results_by_step[step] = step_results
    
    # 生成时间戳
    timestamps = [f"month_{(start_month + i + 1) % 12 + 1}" for i in range(predict_steps)]
    
    return OceanPredictionResult(
        predictions=preds,
        timestamps=timestamps,
        variables=predictor.MODEL_OCEAN_VARIABLES,
        metadata={
            'start_month': start_month,
            'predict_steps': predict_steps,
            'results_by_step': results_by_step,
            'results_by_var': results_by_var,
            'output_shape': preds.shape
        }
    )


# =====================================
# 函数2: 气候平均模拟 (Climatology)
# =====================================
def compute_climatology(
    predictions: List[OceanPredictionResult],
    target_variable: str = 'tos',
    groupby: str = 'month'
) -> Dict[str, np.ndarray]:
    """
    基于多次预测结果计算气候态（气候平均状态）
    
    气候态定义：特定时间段内（如某月份）的平均海洋状态
    用于评估预测结果与历史气候态的偏差（距平）
    
    Args: 
        predictions: 多个预测结果列表（通常来自多年初始化）
        target_variable:  目标变量名（如 'tos' 海表温度）
        groupby: 分组方式 ('month' 按月 或 'season' 按季节)
        
    Returns:
        Dict:  包含气候态均值、标准差和距平的字典
    """
    # 按月份/季节组织预测数据
    grouped_data = {}
    
    for pred_result in predictions:
        results_by_step = pred_result.metadata.get('results_by_step', {})
        start_month = pred_result.metadata['start_month']
        
        for step, step_data in results_by_step.items():
            if target_variable not in step_data: 
                continue
            
            var_data = step_data[target_variable]
            # 处理batch维度
            if len(var_data.shape) > 3:
                var_data = var_data[0]  # 取第一个batch
            
            pred_month = (start_month + step + 1) % 12
            
            if groupby == 'month':
                key = pred_month
            elif groupby == 'season':
                key = ['DJF', 'MAM', 'JJA', 'SON'][pred_month // 3]
            else: 
                raise ValueError(f"Unknown groupby:  {groupby}")
            
            if key not in grouped_data: 
                grouped_data[key] = []
            grouped_data[key].append(var_data)
    
    # 计算气候态统计
    climatology = {}
    for key, data_list in grouped_data.items():
        # 统一形状后堆叠
        stacked = np.stack([d.squeeze() for d in data_list], axis=0)
        climatology[key] = {
            'mean': np.nanmean(stacked, axis=0),
            'std': np.nanstd(stacked, axis=0),
            'count': len(data_list),
            'percentile_10': np.nanpercentile(stacked, 10, axis=0),
            'percentile_90': np.nanpercentile(stacked, 90, axis=0)
        }
    
    return {
        'climatology': climatology,
        'variable':  target_variable,
        'groupby': groupby
    }


def compute_anomaly(
    prediction: OceanPredictionResult,
    climatology: Dict,
    target_variable: str = 'tos'
) -> Dict[int, np.ndarray]:
    """
    计算预测结果相对于气候态的距平
    
    距平 = 预测值 - 气候态均值
    """
    anomalies = {}
    results_by_step = prediction.metadata.get('results_by_step', {})
    start_month = prediction.metadata['start_month']
    clim_data = climatology['climatology']
    groupby = climatology['groupby']
    
    for step, step_data in results_by_step.items():
        if target_variable not in step_data:
            continue
        
        pred_value = step_data[target_variable]
        if len(pred_value.shape) > 3:
            pred_value = pred_value[0]
        
        pred_month = (start_month + step + 1) % 12
        
        if groupby == 'month':
            key = pred_month
        elif groupby == 'season':
            key = ['DJF', 'MAM', 'JJA', 'SON'][pred_month // 3]
        
        if key in clim_data:
            clim_mean = clim_data[key]['mean']
            anomalies[step] = pred_value.squeeze() - clim_mean
    
    return anomalies


# =====================================
# 函数3: 季节性预测
# =====================================
def seasonal_prediction(
    predictor: ORCADLPredictor,
    ocean_data: Dict[str, np.ndarray],
    atmo_data: Dict[str, np.ndarray],
    start_month: int,
    target_seasons: List[str] = ['DJF', 'MAM', 'JJA', 'SON'],
    max_lead_months: int = 12,
    use_model_naming: bool = True
) -> Dict[str, Dict]: 
    """
    季节性预测 - 预测未来各季节的平均海洋状态
    
    季节定义: 
    - DJF (12-1-2月): 冬季
    - MAM (3-4-5月): 春季
    - JJA (6-7-8月): 夏季
    - SON (9-10-11月): 秋季
    """
    SEASON_MONTHS = {
        'DJF': [11, 0, 1],
        'MAM': [2, 3, 4],
        'JJA': [5, 6, 7],
        'SON': [8, 9, 10]
    }
    
    # 执行长期预测
    prediction = predict_ocean_state(
        predictor=predictor,
        ocean_data=ocean_data,
        atmo_data=atmo_data,
        start_month=start_month,
        predict_steps=max_lead_months,
        denormalize=True,
        use_model_naming=use_model_naming
    )
    
    results_by_step = prediction.metadata['results_by_step']
    
    # 按季节聚合预测结果
    seasonal_results = {}
    
    for season in target_seasons: 
        season_months = SEASON_MONTHS[season]
        season_data = {var: [] for var in predictor.MODEL_OCEAN_VARIABLES}
        lead_times = []
        
        for step in range(max_lead_months):
            pred_month = (start_month + step + 1) % 12
            
            if pred_month in season_months: 
                lead_times.append(step + 1)
                step_data = results_by_step.get(step, {})
                
                for var in predictor.MODEL_OCEAN_VARIABLES: 
                    if var in step_data:
                        var_data = step_data[var]
                        if len(var_data.shape) > 3:
                            var_data = var_data[0]
                        season_data[var].append(var_data)
        
        # 计算季节平均
        if lead_times:
            seasonal_mean = {}
            seasonal_std = {}
            
            for var in predictor.MODEL_OCEAN_VARIABLES:
                if season_data[var]:
                    stacked = np.stack([d.squeeze() for d in season_data[var]], axis=0)
                    seasonal_mean[var] = np.nanmean(stacked, axis=0)
                    seasonal_std[var] = np.nanstd(stacked, axis=0)
            
            seasonal_results[season] = {
                'mean': seasonal_mean,
                'std': seasonal_std,
                'lead_times_months': lead_times,
                'n_months_included': len(lead_times)
            }
    
    return {
        'seasonal_predictions': seasonal_results,
        'start_month': start_month,
        'max_lead_months': max_lead_months
    }


# =====================================
# 函数4: 极端气候预测
# =====================================
def extreme_event_detection(
    prediction: OceanPredictionResult,
    climatology: Dict,
    target_variable: str = 'tos',
    threshold_percentile: float = 90.0,
    extreme_type: str = 'both'
) -> Dict:
    """
    极端气候事件检测 - 识别预测中的极端海洋状态
    
    极端事件定义：
    - 海洋热浪 (Marine Heatwave): SST超过第90百分位
    - 海洋冷浪 (Marine Cold Spell): SST低于第10百分位
    """
    results_by_step = prediction.metadata.get('results_by_step', {})
    start_month = prediction.metadata['start_month']
    clim_data = climatology['climatology']
    groupby = climatology['groupby']
    
    extreme_events = []
    
    for step, step_data in results_by_step.items():
        if target_variable not in step_data: 
            continue
        
        pred_value = step_data[target_variable]
        if len(pred_value.shape) > 3:
            pred_value = pred_value[0]
        pred_value = pred_value.squeeze()
        
        pred_month = (start_month + step + 1) % 12
        
        if groupby == 'month':
            key = pred_month
        elif groupby == 'season':
            key = ['DJF', 'MAM', 'JJA', 'SON'][pred_month // 3]
        
        if key not in clim_data:
            continue
        
        clim_stats = clim_data[key]
        clim_mean = clim_stats['mean']
        clim_std = clim_stats['std']
        
        upper_threshold = clim_stats.get('percentile_90', clim_mean + 1.28 * clim_std)
        lower_threshold = clim_stats.get('percentile_10', clim_mean - 1.28 * clim_std)
        
        event_info = {
            'step': step,
            'month': pred_month + 1,
            'lead_time_months': step + 1,
            'events': []
        }
        
        if extreme_type in ['warm', 'both']:
            warm_mask = pred_value > upper_threshold
            if np.any(warm_mask):
                warm_intensity = np.where(warm_mask, pred_value - upper_threshold, 0)
                event_info['events'].append({
                    'type': 'warm_extreme',
                    'coverage_fraction': float(np.mean(warm_mask)),
                    'max_intensity': float(np.nanmax(warm_intensity)),
                    'mean_intensity': float(np.nanmean(warm_intensity[warm_mask])),
                    'mask': warm_mask
                })
        
        if extreme_type in ['cold', 'both']: 
            cold_mask = pred_value < lower_threshold
            if np.any(cold_mask):
                cold_intensity = np.where(cold_mask, lower_threshold - pred_value, 0)
                event_info['events'].append({
                    'type': 'cold_extreme',
                    'coverage_fraction':  float(np.mean(cold_mask)),
                    'max_intensity': float(np.nanmax(cold_intensity)),
                    'mean_intensity': float(np.nanmean(cold_intensity[cold_mask])),
                    'mask': cold_mask
                })
        
        if event_info['events']:
            extreme_events.append(event_info)
    
    summary = {
        'total_extreme_timesteps': len(extreme_events),
        'warm_events': sum(1 for e in extreme_events
                          for ev in e['events'] if ev['type'] == 'warm_extreme'),
        'cold_events':  sum(1 for e in extreme_events
                          for ev in e['events'] if ev['type'] == 'cold_extreme')
    }
    
    return {
        'extreme_events':  extreme_events,
        'summary': summary,
        'target_variable': target_variable,
        'threshold_percentile': threshold_percentile
    }


# =====================================
# 函数5: 长周期预测 (年代际预测)
# =====================================
def long_term_prediction(
    predictor: ORCADLPredictor,
    initial_ocean_data: Dict[str, np.ndarray],
    initial_atmo_data: Dict[str, np.ndarray],
    atmo_forcing_sequence: Optional[List[Dict[str, np.ndarray]]] = None,
    start_month: int = 0,
    predict_years: int = 5,
    use_model_naming: bool = True
) -> Dict:
    """
    长周期（年代际）预测 - 多年尺度的海洋状态预测
    
    通过迭代方式实现长期预测：
    1.使用模型预测6个月
    2.使用预测结果作为新的初始条件
    3.迭代直到达到目标年数
    """
    total_months = predict_years * 12
    all_predictions = []
    
    # 当前海洋状态
    current_ocean = initial_ocean_data.copy()
    current_atmo = initial_atmo_data.copy()
    current_month = start_month
    
    # 迭代预测
    months_predicted = 0
    iteration = 0
    max_t = predictor.config.max_t or 6
    
    while months_predicted < total_months:
        steps_this_iter = min(max_t, total_months - months_predicted)
        
        if atmo_forcing_sequence and iteration < len(atmo_forcing_sequence):
            current_atmo = atmo_forcing_sequence[iteration]
        
        prediction = predict_ocean_state(
            predictor=predictor,
            ocean_data=current_ocean,
            atmo_data=current_atmo,
            start_month=current_month,
            predict_steps=steps_this_iter,
            denormalize=True,
            use_model_naming=use_model_naming
        )
        
        all_predictions.append({
            'iteration': iteration,
            'start_month': current_month,
            'months_offset': months_predicted,
            'prediction': prediction
        })
        
        # 更新状态
        last_step = steps_this_iter - 1
        results_by_var = prediction.metadata.get('results_by_var', {})
        
        var_names = predictor.MODEL_OCEAN_VARIABLES if use_model_naming else predictor.STAT_OCEAN_VARIABLES
        for var in var_names:
            model_var = var if use_model_naming else predictor._get_model_var_name(var)
            if model_var in results_by_var:
                var_data = results_by_var[model_var]
                # 处理multi-step输出
                if len(var_data.shape) == 5:  # (B, T, C, H, W)
                    current_ocean[var] = var_data[0, last_step].squeeze()
                elif len(var_data.shape) == 4:  # (B, C, H, W)
                    current_ocean[var] = var_data[0].squeeze()
                else:
                    current_ocean[var] = var_data.squeeze()
        
        months_predicted += steps_this_iter
        current_month = (current_month + steps_this_iter) % 12
        iteration += 1
    
    # 重组为连续时间序列
    time_series = {var: [] for var in predictor.MODEL_OCEAN_VARIABLES}
    
    for pred_info in all_predictions:
        pred = pred_info['prediction']
        offset = pred_info['months_offset']
        results_by_step = pred.metadata.get('results_by_step', {})
        
        for step in sorted(results_by_step.keys()):
            month_idx = offset + step
            if month_idx < total_months:
                step_data = results_by_step[step]
                for var in predictor.MODEL_OCEAN_VARIABLES:
                    if var in step_data:
                        var_data = step_data[var]
                        if len(var_data.shape) > 3:
                            var_data = var_data[0]
                        time_series[var].append(var_data.squeeze())
    
    # 计算年平均值
    annual_means = {var: [] for var in predictor.MODEL_OCEAN_VARIABLES}
    
    for year in range(predict_years):
        year_start = year * 12
        year_end = min((year + 1) * 12, len(time_series[predictor.MODEL_OCEAN_VARIABLES[0]]))
        
        for var in predictor.MODEL_OCEAN_VARIABLES:
            if time_series[var]:
                year_data = np.stack(time_series[var][year_start:year_end], axis=0)
                annual_means[var].append(np.nanmean(year_data, axis=0))
    
    # 计算长期趋势
    trends = {}
    for var in predictor.MODEL_OCEAN_VARIABLES:
        if annual_means[var]:
            annual_stack = np.stack(annual_means[var], axis=0)
            years_arr = np.arange(len(annual_means[var]))
            spatial_mean = np.nanmean(annual_stack, axis=tuple(range(1, annual_stack.ndim)))
            
            if len(spatial_mean) > 1:
                trend_slope = np.polyfit(years_arr, spatial_mean, 1)[0]
            else:
                trend_slope = 0.0
            
            trends[var] = {
                'annual_spatial_mean': spatial_mean.tolist(),
                'trend_per_year': float(trend_slope),
                'total_change': float(trend_slope * predict_years)
            }
    
    return {
        'time_series': time_series,
        'annual_means': annual_means,
        'trends': trends,
        'predict_years': predict_years,
        'total_months': total_months,
        'start_month': start_month,
        'iterations': len(all_predictions)
    }


# =====================================
# 使用示例
# =====================================
if __name__ == "__main__":
    # 初始化预测器
    predictor = ORCADLPredictor(
        config_path='/mnt/data/zhu.yishun/ORCA-DL-main/model_config.json',
        checkpoint_path='/mnt/data/zhu.yishun/ORCA-DL-main/ckpt/seed_1.bin',
        stat_dir='/mnt/data/zhu.yishun/ORCA-DL-main/stat'
    )
    
    # 准备示例数据
    H, W = 128, 360
    DEPTH_LEVELS = 16
    
    # 方式1: 使用模型变量命名 (CMIP6)
    ocean_data_model = {
        'so':  np.random.randn(DEPTH_LEVELS, H, W),
        'thetao': np.random.randn(DEPTH_LEVELS, H, W),
        'tos': np.random.randn(H, W),
        'uo': np.random.randn(DEPTH_LEVELS, H, W),
        'vo': np.random.randn(DEPTH_LEVELS, H, W),
        'zos': np.random.randn(H, W)
    }
    atmo_data_model = {
        'tauu': np.random.randn(H, W),
        'tauv': np.random.randn(H, W)
    }
    
    # 方式2: 使用统计文件变量命名 (GODAS)
    ocean_data_stat = {
        'salt': np.random.randn(DEPTH_LEVELS, H, W),
        'pottmp': np.random.randn(DEPTH_LEVELS, H, W),
        'sst': np.random.randn(H, W),
        'ucur': np.random.randn(DEPTH_LEVELS, H, W),
        'vcur': np.random.randn(DEPTH_LEVELS, H, W),
        'sshg': np.random.randn(H, W)
    }
    atmo_data_stat = {
        'uflx': np.random.randn(H, W),
        'vflx': np.random.randn(H, W)
    }
    
    start_month = 0
    
    # ========== 测试不同输出模式 ==========
    print("=" * 60)
    print("测试模型输出模式")
    print("=" * 60)
    
    # Single step 模式
    print("\n1.Single step 模式:")
    pred_single = predict_ocean_state(
        predictor, ocean_data_model, atmo_data_model,
        start_month=0, predict_steps=1, use_model_naming=True
    )
    print(f"   输出形状: {pred_single.metadata['output_shape']}")
    
    # Multi steps 模式
    print("\n2.Multi steps 模式:")
    pred_multi = predict_ocean_state(
        predictor, ocean_data_model, atmo_data_model,
        start_month=0, predict_steps=6, use_model_naming=True
    )
    print(f"   输出形状: {pred_multi.metadata['output_shape']}")
    
    # ========== 测试变量命名映射 ==========
    print("\n" + "=" * 60)
    print("测试变量命名映射")
    print("=" * 60)
    
    print("\n使用GODAS命名:")
    pred_godas = predict_ocean_state(
        predictor, ocean_data_stat, atmo_data_stat,
        start_month=0, predict_steps=6, use_model_naming=False
    )
    print(f"   输出形状: {pred_godas.metadata['output_shape']}")
    
    # ========== 核心功能测试 ==========
    print("\n" + "=" * 60)
    print("核心预测功能测试")
    print("=" * 60)
    
    # 气候态
    print("\n3.气候平均模拟:")
    climatology = compute_climatology([pred_multi], 'tos', 'month')
    print(f"   气候态月份数: {len(climatology['climatology'])}")
    
    # 季节预测
    print("\n4.季节性预测:")
    seasonal = seasonal_prediction(
        predictor, ocean_data_model, atmo_data_model,
        start_month=0, target_seasons=['MAM', 'JJA'], max_lead_months=12
    )
    for s, d in seasonal['seasonal_predictions'].items():
        print(f"   {s}:  {d['n_months_included']} 个月")
    
    # 极端事件
    print("\n5.极端气候预测:")
    extreme = extreme_event_detection(pred_multi, climatology, 'tos')
    print(f"   检测到极端时间步: {extreme['summary']['total_extreme_timesteps']}")
    
    # 长期预测
    print("\n6.长周期预测 (2年):")
    long_term = long_term_prediction(
        predictor, ocean_data_model, atmo_data_model,
        start_month=0, predict_years=2
    )
    print(f"   总迭代次数: {long_term['iterations']}")
    print(f"   总预测月数: {long_term['total_months']}")
