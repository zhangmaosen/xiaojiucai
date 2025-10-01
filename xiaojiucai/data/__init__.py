"""数据加载和处理模块。

包含了用于加载和预处理金融数据的类和工具。

主要包括:
- 数据加载器: InRAMDataset, FlexibleDataLoader, RigidDataLoader
- 数据增强: Compose, Scale, Noise等
- 数据预处理: 标准化等工具函数
- TimesFM数据处理: DataProcessor, 特征配置, 序列创建等
"""

from .load import (
    FlexibleDataLoader,
    InRAMDataset, 
    RigidDataLoader,
    prepare_standard_scaler,
    Scale
)

from .augment import (
    Compose,
    Dropout,
    Multiply,
    Noise
)

# 添加TimesFM相关的数据处理工具
from .data_processor import DataProcessor, prepare_data_tensors
from .sequence_creator import create_sequences_3d, create_data_loaders
from .feature_configs import (
    FEATURE_CONFIGS, PRICE_ONLY, PRICE_AND_VOLUME, CLOSE_ONLY, 
    OHLC, CLOSE_AND_VOLUME, AVAILABLE_FEATURES,
    validate_features, print_feature_configs
)

__all__ = [
    'Compose',
    'Dropout', 
    'FlexibleDataLoader',
    'InRAMDataset',
    'Multiply',
    'Noise',
    'RigidDataLoader',
    'Scale',
    'prepare_standard_scaler'
]