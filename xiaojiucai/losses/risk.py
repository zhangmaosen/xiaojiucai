"""风险相关的损失函数模块。

包含各种基于风险度量的损失函数实现。主要包括:
- MaximumDrawdown: 最大回撤损失
- Volatility: 波动率损失
- CVaR: 条件风险价值损失
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .base import Loss, log2simple, simple2log


class MaximumDrawdown(Loss):
    """最大回撤损失函数。
    
    使用投资组合的最大回撤作为风险度量。
    
    Parameters
    ----------
    returns_channel : int
        y目标张量中表示收益率的通道索引。
        
    input_type : str {'log', 'simple'}
        输入收益率类型。
        
    output_type : str {'log', 'simple'}
        输出收益率类型。
        
    normalize : bool
        是否对损失值进行标准化。
        
    Attributes
    ----------
    metrics : dict
        保存计算出的评价指标。
    """
    
    def __init__(
        self,
        returns_channel: int = 0,
        input_type: str = "log",
        output_type: str = "simple",
        normalize: bool = False
    ):
        super().__init__(
            returns_channel=returns_channel,
            input_type=input_type,
            output_type=output_type,
            normalize=normalize
        )
        self.metrics = {}
        
    def _compute_loss(self, weights: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算最大回撤损失。
        
        Parameters
        ----------
        weights : torch.Tensor
            形状为(n_samples, n_assets)的权重张量。
            
        y : torch.Tensor
            形状为(n_samples, n_channels, horizon, n_assets)的标签数据。
            
        Returns
        -------
        torch.Tensor
            形状为(n_samples,)的每个样本的最大回撤损失。
        """
        # 获取收益率数据
        returns = y[:, self.returns_channel]  # (n_samples, horizon, n_assets)
        
        # 计算投资组合收益
        portfolio_returns = torch.sum(
            weights.unsqueeze(1) * returns, dim=2
        )  # (n_samples, horizon)
        
        # 处理收益率类型转换
        if self.input_type == "log" and self.output_type == "simple":
            portfolio_returns = log2simple(portfolio_returns)
        elif self.input_type == "simple" and self.output_type == "log":
            portfolio_returns = simple2log(portfolio_returns)
            
        # 计算累积收益
        cumulative_returns = torch.cumprod(1 + portfolio_returns, dim=1)
        
        # 计算滚动最大值
        rolling_max = torch.maximum.accumulate(cumulative_returns, dim=1)
        
        # 计算每个样本的最大回撤
        drawdowns = (rolling_max - cumulative_returns) / rolling_max
        max_drawdowns = drawdowns.max(dim=1)[0]  # (n_samples,)
        
        # 标准化
        if self.normalize:
            max_drawdowns = (max_drawdowns - max_drawdowns.mean()) / (max_drawdowns.std() + 1e-8)
            
        # 更新评价指标
        self.metrics.update({
            "max_drawdown": max_drawdowns.mean().item(),
            "avg_drawdown": drawdowns.mean().item()
        })
        
        return max_drawdowns
        
    def compute_metrics(self, weights: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算评价指标。
        
        Parameters
        ----------
        weights : torch.Tensor
            形状为(n_samples, n_assets)的权重张量。
            
        y : torch.Tensor
            形状为(n_samples, n_channels, horizon, n_assets)的标签数据。
            
        Returns
        -------
        dict
            包含max_drawdown、avg_drawdown等指标的字典。
        """
        return self.metrics
        
        # 计算历史最高点
        cummax = torch.cummax(cumulative_returns, dim=1)[0]  # (n_samples, horizon)
        
        # 计算回撤
        drawdowns = (cumulative_returns / cummax) - 1  # (n_samples, horizon)
        
        # 计算最大回撤
        max_drawdown = torch.min(drawdowns, dim=1)[0]  # (n_samples,)
        
        return -max_drawdown  # 返回负值以符合最小化目标

class Volatility(Loss):
    """波动率损失函数
    
    Parameters
    ----------
    returns_channel : int
        y 目标张量中表示收益率的通道索引
        
    input_type : str, {'log', 'simple'}
        y 中收益率的类型，可以是对数收益或简单收益
        
    output_type : str, {'log', 'simple'}
        输出收益率的类型
    """
    
    def __init__(
        self,
        returns_channel=0,
        input_type="log",
        output_type="simple"
    ):
        super().__init__()
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type
        
    def forward(self, weights, y):
        """计算波动率损失
        
        Parameters
        ----------
        weights : torch.Tensor
            形状为 (n_samples, n_assets) 的权重张量
            
        y : torch.Tensor
            形状为 (n_samples, n_channels, horizon, n_assets) 的标签数据
            
        Returns
        -------
        torch.Tensor
            形状为 (n_samples,) 的每个样本的波动率损失
        """
        # 获取收益率数据
        returns = y[:, self.returns_channel]  # (n_samples, horizon, n_assets)
        
        # 计算投资组合收益
        portfolio_returns = torch.sum(weights.unsqueeze(1) * returns, dim=2)  # (n_samples, horizon)
        
        # 根据需要转换收益率类型
        if self.input_type != self.output_type:
            if self.input_type == "log" and self.output_type == "simple":
                portfolio_returns = log2simple(portfolio_returns)
            elif self.input_type == "simple" and self.output_type == "log":
                portfolio_returns = simple2log(portfolio_returns)
        
        # 计算标准差（波动率）
        volatility = torch.std(portfolio_returns, dim=1)  # (n_samples,)
        
        return volatility  # 已经是正值，无需取负

# 别名，用于向后兼容
MaximumDrawdownLoss = MaximumDrawdown
VolatilityLoss = Volatility