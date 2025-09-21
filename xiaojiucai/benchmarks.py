"""
Benchmark models for portfolio optimization.
"""

import torch
import numpy as np
from .benchmarks import Benchmark, OneOverN, Random, InverseVolatility, MaximumReturn

# 为保持向后兼容性，从benchmarks包导入类
# 这样既避免了循环导入，又保持了原有的导入方式可用

__all__ = ['Benchmark', 'OneOverN', 'Random', 'InverseVolatility', 'MaximumReturn']