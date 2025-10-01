"""
数据处理器模块

包含数据加载、预处理和特征工程的核心工具
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Optional
from .feature_configs import AVAILABLE_FEATURES, validate_features


class DataProcessor:
    """时间序列数据处理器"""
    
    def __init__(self, selected_features: List[str] = None, test_mode: bool = False, test_data_size: int = 600):
        """
        初始化数据处理器
        
        Args:
            selected_features: 选择的特征列表，默认为所有特征
            test_mode: 是否启用测试模式（减少数据量）
            test_data_size: 测试模式下使用的数据行数
        """
        self.selected_features = selected_features or AVAILABLE_FEATURES
        self.test_mode = test_mode
        self.test_data_size = test_data_size
        
        # 验证特征
        validate_features(self.selected_features)
        
        print(f"数据处理器初始化完成:")
        print(f"  选择的特征: {self.selected_features}")
        print(f"  特征数量: {len(self.selected_features)}")
        if test_mode:
            print(f"  测试模式: 启用，使用后 {test_data_size} 行数据")
    
    def load_and_preprocess(self, data_path: str) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        """
        加载和预处理数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            features_data: 特征数据，形状为 (时间步, 股票数, 特征数)
            returns: 回报率数据
            tickers: 股票列表
        """
        # 加载原始数据
        data_raw = pd.read_parquet(data_path)
        
        if self.test_mode and self.test_data_size < len(data_raw):
            print(f"注意：正在测试模式下运行，仅使用后 {self.test_data_size} 行数据")
            data_raw = data_raw.iloc[-self.test_data_size:]

        print("数据形状:", data_raw.shape)
        print("数据列名:", data_raw.columns.tolist()[:10], "...")

        # 提取Close价格用于计算回报率
        close_prices = data_raw['Close']
        returns = np.log(close_prices / close_prices.shift(1))

        # 处理NaN值
        returns.iloc[0] = 0  # 第一天的回报率设为0
        returns = returns.ffill()  # 用前一个有效值填充其他NaN
        returns = returns.bfill()  # 如果数据开头有NaN，用后一个有效值填充

        # 获取股票列表
        tickers = close_prices.columns.tolist()

        # 创建特征矩阵
        n_timesteps = len(data_raw)
        n_stocks = len(tickers)
        n_features = len(self.selected_features)

        features_data = np.zeros((n_timesteps, n_stocks, n_features))

        for i, ticker in enumerate(tickers):
            for j, feature in enumerate(self.selected_features):
                features_data[:, i, j] = data_raw[feature][ticker].ffill().fillna(0)

        print(f"\n特征数据形状: {features_data.shape} (时间步, 股票数, 特征数)")
        print(f"股票列表: {tickers}")
        print(f"使用的特征: {self.selected_features}")

        # 显示特征数据的统计信息
        self._print_feature_stats(features_data)

        return features_data, returns, tickers
    
    def _print_feature_stats(self, features_data: np.ndarray):
        """打印特征数据统计信息"""
        print(f"\n各特征的统计描述:")
        for i, feature in enumerate(self.selected_features):
            feature_data = features_data[:, :, i]
            print(f"\n{feature}:")
            print(f"  均值: {np.mean(feature_data):.4f}")
            print(f"  标准差: {np.std(feature_data):.4f}")
            print(f"  最小值: {np.min(feature_data):.4f}")
            print(f"  最大值: {np.max(feature_data):.4f}")


def prepare_data_tensors(features_data: np.ndarray, returns: pd.DataFrame, 
                        window_size: int, train_ratio: float = 0.8, 
                        val_ratio: float = 0.8, device: str = 'cpu') -> dict:
    """
    准备训练用的数据张量
    
    Args:
        features_data: 特征数据
        returns: 回报率数据
        window_size: 时间窗口大小
        train_ratio: 训练+验证集占比
        val_ratio: 验证集在训练+验证集中的占比
        device: 设备类型
        
    Returns:
        包含训练、验证、测试数据的字典
    """
    from .sequence_creator import create_sequences_3d
    
    # 创建序列数据
    X, y = create_sequences_3d(features_data, returns, window_size)
    print("输入数据形状:", X.shape)  # (样本数, 时间步长, 股票数, 特征数)
    print("目标数据形状:", y.shape)  # (样本数, 股票数)

    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # 分割数据集
    train_val_size = int(train_ratio * len(X))
    train_size = int(val_ratio * train_val_size)
    
    X_test = X_tensor[train_val_size:]
    y_test = y_tensor[train_val_size:]
    X_train = X_tensor[:train_size]
    y_train = y_tensor[:train_size]
    X_val = X_tensor[train_size:train_val_size]
    y_val = y_tensor[train_size:train_val_size]

    # 将数据移至指定设备
    if device == 'cuda' and torch.cuda.is_available():
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_val = X_val.cuda()
        y_val = y_val.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()

    print("\n数据集大小:")
    print(f"训练集: {len(X_train)} 样本 ({len(X_train)/len(X_tensor):.1%})")
    print(f"验证集: {len(X_val)} 样本 ({len(X_val)/len(X_tensor):.1%})")
    print(f"测试集: {len(X_test)} 样本 ({len(X_test)/len(X_tensor):.1%})")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'input_size': X.shape[2],      # 股票数量
        'feature_size': X.shape[3],    # 特征数量
        'window_size': window_size
    }