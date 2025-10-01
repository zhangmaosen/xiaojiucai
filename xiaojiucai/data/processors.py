"""
数据处理模块

处理时间序列数据，包括加载、清理、特征工程和格式转换
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler
import warnings


class MAG7DataProcessor:
    """MAG7股票数据处理器"""
    
    def __init__(self, asset_names: List[str] = None):
        self.asset_names = asset_names or [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'
        ]
        self.scaler = StandardScaler()
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """加载MAG7数据"""
        print(f"📥 加载数据: {data_path}")
        
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")
            
        print(f"✅ 数据加载完成: {df.shape}")
        print(f"📅 时间范围: {df.index.min()} 到 {df.index.max()}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清理"""
        print("🧹 开始数据清理...")
        
        # 移除缺失值过多的行
        threshold = len(self.asset_names) * 0.8  # 至少80%的资产有数据
        df_clean = df.dropna(thresh=threshold)
        
        # 前向填充缺失值
        df_clean = df_clean.fillna(method='ffill')
        
        # 移除仍有缺失值的行
        df_clean = df_clean.dropna()
        
        print(f"✅ 数据清理完成: {df.shape} → {df_clean.shape}")
        return df_clean

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        print("⚡ 开始特征工程...")
        
        # 为每个资产计算技术指标
        feature_df_list = []
        
        for asset in self.asset_names:
            if asset not in df.columns:
                print(f"⚠️ 资产 {asset} 不在数据中，跳过")
                continue
                
            prices = df[asset]
            
            # 计算各种技术指标
            asset_features = pd.DataFrame(index=df.index)
            
            # 基础特征
            asset_features[f'{asset}_price'] = prices
            asset_features[f'{asset}_returns'] = prices.pct_change()
            asset_features[f'{asset}_log_returns'] = np.log(prices / prices.shift(1))
            
            # 标准化价格和成交量 (如果有的话)
            asset_features[f'{asset}_price_norm'] = (prices - prices.rolling(60).mean()) / prices.rolling(60).std()
            
            # 如果有成交量数据
            volume_col = f'{asset}_volume'
            if volume_col in df.columns:
                volume = df[volume_col]
                asset_features[f'{asset}_volume_norm'] = (volume - volume.rolling(60).mean()) / volume.rolling(60).std()
            else:
                # 如果没有成交量数据，使用价格变化的绝对值作为替代
                asset_features[f'{asset}_volume_norm'] = asset_features[f'{asset}_returns'].abs()
            
            feature_df_list.append(asset_features)
        
        # 合并所有资产的特征
        features_df = pd.concat(feature_df_list, axis=1)
        
        # 移除前60行（因为滚动计算）
        features_df = features_df.iloc[60:].copy()
        
        print(f"✅ 特征工程完成: {features_df.shape}")
        return features_df

    def to_deepdow_format(
        self, 
        features_df: pd.DataFrame, 
        sequence_length: int = 60, 
        prediction_horizon: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """转换为deepdow格式"""
        print(f"📊 转换为deepdow格式 (序列长度: {sequence_length}, 预测范围: {prediction_horizon})...")
        
        # 重新组织数据为 (样本, 指标, 时间窗口, 资产) 格式
        n_assets = len(self.asset_names)
        n_indicators = 4  # returns, log_returns, price_norm, volume_norm
        
        # 创建指标列表
        indicators = ['returns', 'log_returns', 'price_norm', 'volume_norm']
        
        # 提取每个资产的各个指标数据
        asset_data = {}
        for asset in self.asset_names:
            asset_data[asset] = {}
            for indicator in indicators:
                col_name = f'{asset}_{indicator}'
                if col_name in features_df.columns:
                    asset_data[asset][indicator] = features_df[col_name].values
                else:
                    print(f"⚠️ 列 {col_name} 不存在，使用零填充")
                    asset_data[asset][indicator] = np.zeros(len(features_df))
        
        # 创建滑动窗口样本
        n_samples = len(features_df) - sequence_length - prediction_horizon + 1
        
        X = np.zeros((n_samples, n_indicators, sequence_length, n_assets))
        y = np.zeros((n_samples, n_assets, prediction_horizon))
        
        for i in range(n_samples):
            for asset_idx, asset in enumerate(self.asset_names):
                for indicator_idx, indicator in enumerate(indicators):
                    # 特征窗口
                    X[i, indicator_idx, :, asset_idx] = asset_data[asset][indicator][i:i+sequence_length]
                
                # 未来收益率 (使用returns作为目标)
                future_returns = asset_data[asset]['returns'][i+sequence_length:i+sequence_length+prediction_horizon]
                y[i, asset_idx, :] = future_returns
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        print(f"✅ 格式转换完成:")
        print(f"  - 特征形状: {X_tensor.shape} (样本, 指标, 时间窗口, 资产)")
        print(f"  - 目标形状: {y_tensor.shape} (样本, 资产, 预测范围)")
        
        return X_tensor, y_tensor

    def create_data_splits(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """创建训练/验证/测试数据分割"""
        n_samples = X.shape[0]
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # 分割数据
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        print(f"📊 数据分割完成:")
        print(f"  - 训练集: {X_train.shape[0]} 样本 ({train_ratio*100:.1f}%)")
        print(f"  - 验证集: {X_val.shape[0]} 样本 ({val_ratio*100:.1f}%)")
        print(f"  - 测试集: {X_test.shape[0]} 样本 ({(1-train_ratio-val_ratio)*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class PortfolioDataset(Dataset):
    """投资组合数据集"""
    
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return self.features.shape[0]
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def create_data_loaders(
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor], 
    test_data: Tuple[torch.Tensor, torch.Tensor],
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 创建数据集
    train_dataset = PortfolioDataset(*train_data)
    val_dataset = PortfolioDataset(*val_data)
    test_dataset = PortfolioDataset(*test_data)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ 数据加载器创建完成:")
    print(f"  - 训练: {len(train_loader)} 批次")
    print(f"  - 验证: {len(val_loader)} 批次") 
    print(f"  - 测试: {len(test_loader)} 批次")
    
    return train_loader, val_loader, test_loader


def calculate_benchmark_portfolios(
    returns_data: pd.DataFrame, 
    asset_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """计算基准投资组合策略"""
    print("⚖️ 计算基准策略性能...")
    
    # 提取收益率矩阵
    if isinstance(returns_data.columns, pd.MultiIndex):
        # 多级列名情况
        returns_columns = []
        for asset in asset_names:
            for col in [(asset, 'returns'), (asset, 'log_returns')]:
                if col in returns_data.columns:
                    returns_columns.append(col)
                    break
        
        if returns_columns:
            returns_df = returns_data[returns_columns]
            returns_df.columns = [col[0] for col in returns_columns]
        else:
            raise ValueError("无法找到收益率列")
    else:
        # 单级列名情况
        returns_cols = [col for col in returns_data.columns if 'returns' in col and any(asset in col for asset in asset_names)]
        returns_df = returns_data[returns_cols[:len(asset_names)]]
        returns_df.columns = asset_names[:len(returns_cols)]
    
    returns_matrix = returns_df.values
    
    benchmarks = {}
    
    # 1. 等权重策略
    equal_weights = np.ones(len(asset_names)) / len(asset_names)
    equal_portfolio_returns = returns_matrix @ equal_weights
    benchmarks['等权重'] = {
        'weights': equal_weights,
        'returns': equal_portfolio_returns,
        'name': '等权重策略'
    }
    
    # 2. 波动率倒数加权策略
    try:
        returns_std = returns_matrix.std(axis=0)
        returns_std = np.where(returns_std == 0, 1e-8, returns_std)
        vol_weights = 1.0 / returns_std
        vol_weights = vol_weights / vol_weights.sum()
        
        vol_portfolio_returns = returns_matrix @ vol_weights
        benchmarks['波动率倒数加权'] = {
            'weights': vol_weights,
            'returns': vol_portfolio_returns,
            'name': '波动率倒数加权策略'
        }
    except Exception as e:
        print(f"⚠️ 波动率倒数加权策略失败: {e}")
        benchmarks['波动率倒数加权'] = benchmarks['等权重'].copy()
    
    # 3. 最小方差策略
    try:
        cov_matrix = np.cov(returns_matrix.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        if np.min(eigenvals) <= 0:
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-8
            
        inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
        min_var_weights = inv_vol / inv_vol.sum()
        
        min_var_portfolio_returns = returns_matrix @ min_var_weights
        benchmarks['最小方差'] = {
            'weights': min_var_weights,
            'returns': min_var_portfolio_returns,
            'name': '最小方差策略'
        }
    except Exception as e:
        print(f"⚠️ 最小方差策略失败: {e}")
        benchmarks['最小方差'] = benchmarks['等权重'].copy()
    
    # 4. 最佳资产策略
    try:
        mean_returns = returns_matrix.mean(axis=0)
        best_asset_idx = np.argmax(mean_returns)
        best_asset_name = asset_names[best_asset_idx]
        
        best_asset_weights = np.zeros(len(asset_names))
        best_asset_weights[best_asset_idx] = 1.0
        
        best_asset_portfolio_returns = returns_matrix @ best_asset_weights
        
        benchmarks['最佳资产'] = {
            'weights': best_asset_weights,
            'returns': best_asset_portfolio_returns,
            'name': f'最佳资产策略 ({best_asset_name})'
        }
    except Exception as e:
        print(f"⚠️ 最佳资产策略失败: {e}")
        benchmarks['最佳资产'] = benchmarks['等权重'].copy()
    
    print(f"✅ 成功计算 {len(benchmarks)} 个基准策略")
    return benchmarks