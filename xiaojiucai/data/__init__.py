"""数据加载和处理模块。

包含了用于加载和预处理金融数据的类和工具。

主要包括:
- 数据加载器: InRAMDataset, FlexibleDataLoader, RigidDataLoader
- 数据增强: Compose, Scale, Noise等
- 数据预处理: 标准化等工具函数
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