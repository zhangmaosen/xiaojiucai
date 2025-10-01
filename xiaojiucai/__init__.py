"""
小韭菜投资组合优化框架

基于深度学习的智能投资组合优化系统，集成TimesFM预训练模型
"""

# 版本信息
__version__ = "0.2.0"
__author__ = "xiaojiucai"

# 导入核心模块
from . import models
from . import data
from . import utils
from . import visualize
# from . import benchmarks  # 可能有导入问题，先注释
# from . import callbacks
# from . import experiments
# from . import losses
# from . import layers
from . import training
from . import evaluation

# 导入主要类和函数
from .models.timesfm_portfolio import (
    TimesFMFeatureExtractor,
    PortfolioAttentionHead,
    TimesFMPortfolioModel,
    PortfolioLoss
)
from .data.processors import (
    MAG7DataProcessor,
    PortfolioDataset,
    create_data_loaders,
    calculate_benchmark_portfolios
)
from .training import (
    PortfolioTrainer,
    ExperimentManager,
    create_trainer_from_config
)
from .evaluation import (
    PortfolioEvaluator,
    PortfolioVisualizer
)

__all__ = [
    # 核心模型
    'TimesFMFeatureExtractor',
    'PortfolioAttentionHead', 
    'TimesFMPortfolioModel',
    'PortfolioLoss',
    
    # 数据处理
    'MAG7DataProcessor',
    'PortfolioDataset',
    'create_data_loaders',
    'calculate_benchmark_portfolios',
    
    # 训练
    'PortfolioTrainer',
    'ExperimentManager',
    'create_trainer_from_config',
    
    # 评估
    'PortfolioEvaluator',
    'PortfolioVisualizer',
    
    # 子模块
    'models',
    'data',
    'utils',
    'visualize', 
    # 'benchmarks',
    # 'callbacks',
    # 'experiments',
    # 'losses',
    # 'layers',
    'training',
    'evaluation'
]