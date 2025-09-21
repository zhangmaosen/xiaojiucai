"""损失函数模块。

提供了一系列用于投资组合优化的损失函数。主要包括:
- 基于收益率的损失函数(MeanReturns, SharpeRatio等)
- 基于风险的损失函数(MaximumDrawdown等)
- 复合损失函数(通过运算符组合多个损失函数)

所有损失函数都设计为最小化目标。如果类名暗示相反(如MeanReturns)，则在计算时取负值。
"""

from .base import Loss, log2simple, simple2log
from .returns import MeanReturns, SharpeRatio
from .risk import MaximumDrawdown

__all__ = [
    'Loss',
    'MaximumDrawdown',
    'MeanReturns', 
    'SharpeRatio',
    'log2simple',
    'simple2log'
]

__all__ = [
    "Loss",
    "log2simple",
    "simple2log",
    
    # 收益相关损失函数
    "MeanReturns",
    "MeanReturnsLoss",
    "SharpeRatio",
    "SharpeRatioLoss",
    
    # 风险相关损失函数
    "MaximumDrawdown",
    "MaximumDrawdownLoss",
    "Volatility",
    "VolatilityLoss",
]
