"""神经网络层模块。

包含了用于构建深度投资组合优化网络的层。主要包括:

- allocate: 投资组合分配层(如Markowitz, NCO等)
- collapse: 维度压缩层(如Attention, Average等)
- transform: 特征变换层(如Conv, RNN等)
- misc: 其他辅助层(如协方差矩阵等)
"""

from .allocate import (
    AnalyticalMarkowitz,
    NumericalMarkowitz,
    NumericalRiskBudgeting,
    SoftmaxAllocator,
    SparsemaxAllocator
)

from .collapse import (
    AttentionCollapse,
    AverageCollapse,
    MaxCollapse,
    SumCollapse
)

from .transform import (
    Conv,
    RNN
)

from .misc import (
    CovarianceMatrix,
    MultiplyByConstant
)

__all__ = [
    'AnalyticalMarkowitz',
    'AttentionCollapse',
    'AverageCollapse',
    'Conv',
    'CovarianceMatrix',
    'MaxCollapse',
    'MultiplyByConstant',
    'NumericalMarkowitz',
    'NumericalRiskBudgeting',
    'RNN',
    'SoftmaxAllocator',
    'SparsemaxAllocator',
    'SumCollapse'
]
