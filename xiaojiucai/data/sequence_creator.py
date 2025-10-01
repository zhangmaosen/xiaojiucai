"""
序列创建工具

用于创建时间序列训练数据
"""

import numpy as np
import pandas as pd
from typing import Tuple


def create_sequences_3d(features_data: np.ndarray, returns_data: pd.DataFrame, window_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    为3D特征数据和2D回报率数据创建序列

    Args:
        features_data: 形状为 (n_timesteps, n_stocks, n_features) 的特征数据
        returns_data: 形状为 (n_timesteps, n_stocks) 的回报率数据
        window_size: 时间窗口大小

    Returns:
        X: 输入特征序列，形状为 (n_samples, window_size, n_stocks, n_features)
        y: 目标回报率序列，形状为 (n_samples, n_stocks)
    """
    sequences = []
    targets = []

    n_timesteps = features_data.shape[0]

    for i in range(n_timesteps - window_size):
        # 输入序列：过去window_size天的所有特征
        sequence = features_data[i:i+window_size]  # 形状: (window_size, n_stocks, n_features)
        # 目标：下一天的回报率
        target = returns_data.iloc[i+window_size].values  # 形状: (n_stocks,)

        sequences.append(sequence)
        targets.append(target)

    return np.array(sequences), np.array(targets)


def create_data_loaders(data_dict: dict, batch_size: int = 128, num_workers: int = 4, pin_memory: bool = True):
    """
    创建数据加载器
    
    Args:
        data_dict: 包含训练、验证、测试数据的字典
        batch_size: 批次大小
        num_workers: 数据加载进程数
        pin_memory: 是否使用锁页内存
        
    Returns:
        训练、验证、测试数据加载器
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # 创建数据集
    train_dataset = TensorDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = TensorDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = TensorDataset(data_dict['X_test'], data_dict['y_test'])

    # 创建数据加载器 - 优化性能
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader, test_loader