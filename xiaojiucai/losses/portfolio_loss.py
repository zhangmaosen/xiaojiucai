"""
投资组合优化损失函数

包含各种投资组合优化相关的损失函数
"""

import torch
import torch.nn as nn


def portfolio_loss(weights, returns, risk_aversion=1.0, sharpe_weight=0.5):
    """
    投资组合损失函数（包含夏普比率优化）
    
    Args:
        weights: 投资组合权重，形状 (batch_size, num_assets)
        returns: 资产回报率，形状 (batch_size, num_assets) 
        risk_aversion: 风险厌恶系数
        sharpe_weight: 夏普比率在损失函数中的权重
        
    Returns:
        loss: 组合损失值
    """
    # 计算投资组合收益
    portfolio_return = torch.sum(weights * returns, dim=1)
    
    # 计算平均收益和标准差
    expected_return = torch.mean(portfolio_return)
    portfolio_std = torch.std(portfolio_return) + 1e-8  # 添加小量避免除零
    
    # 计算夏普比率（负号因为要最大化）
    sharpe_ratio = expected_return / portfolio_std
    sharpe_loss = -sharpe_ratio
    
    # 计算风险（使用样本标准差的平方而不是方差）
    epsilon = 1e-8
    portfolio_risk = torch.mean((portfolio_return - expected_return) ** 2) + epsilon
    
    # 传统的风险调整收益损失
    traditional_loss = -(expected_return - risk_aversion * portfolio_risk)
    
    # 组合损失：传统损失 + 夏普比率损失
    loss = (1 - sharpe_weight) * traditional_loss + sharpe_weight * sharpe_loss
    
    # 添加正则化项以防止权重过于集中
    weight_regularization = torch.mean(torch.sum(weights ** 2, dim=1))
    regularization_factor = 0.01
    loss = loss + regularization_factor * weight_regularization
    
    return loss


class PortfolioLoss(nn.Module):
    """投资组合损失函数类"""
    
    def __init__(self, risk_aversion=1.0, sharpe_weight=0.5, regularization_factor=0.01):
        super(PortfolioLoss, self).__init__()
        self.risk_aversion = risk_aversion
        self.sharpe_weight = sharpe_weight
        self.regularization_factor = regularization_factor
    
    def forward(self, weights, returns):
        return portfolio_loss(weights, returns, self.risk_aversion, self.sharpe_weight)


class MeanVarianceLoss(nn.Module):
    """传统的均值-方差损失函数"""
    
    def __init__(self, risk_aversion=1.0):
        super(MeanVarianceLoss, self).__init__()
        self.risk_aversion = risk_aversion
    
    def forward(self, weights, returns):
        # 计算投资组合收益
        portfolio_return = torch.sum(weights * returns, dim=1)
        
        # 计算平均收益和方差
        expected_return = torch.mean(portfolio_return)
        portfolio_variance = torch.var(portfolio_return)
        
        # 均值-方差目标函数（最大化收益，最小化风险）
        loss = -(expected_return - self.risk_aversion * portfolio_variance)
        
        return loss


class SharpeRatioLoss(nn.Module):
    """夏普比率损失函数"""
    
    def __init__(self, risk_free_rate=0.0):
        super(SharpeRatioLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
    
    def forward(self, weights, returns):
        # 计算投资组合收益
        portfolio_return = torch.sum(weights * returns, dim=1)
        
        # 计算超额收益
        excess_return = portfolio_return - self.risk_free_rate
        
        # 计算夏普比率
        expected_excess = torch.mean(excess_return)
        volatility = torch.std(excess_return) + 1e-8
        sharpe_ratio = expected_excess / volatility
        
        # 返回负夏普比率（因为要最大化夏普比率）
        return -sharpe_ratio


def calculate_portfolio_metrics(weights, returns):
    """
    计算投资组合的各种指标
    
    Args:
        weights: 投资组合权重
        returns: 资产回报率
        
    Returns:
        metrics: 包含各种指标的字典
    """
    # 计算投资组合收益
    portfolio_return = torch.sum(weights * returns, dim=1)
    
    # 计算基本指标
    mean_return = torch.mean(portfolio_return).item()
    std_return = torch.std(portfolio_return).item()
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
    
    # 年化指标（假设252个交易日）
    annual_return = mean_return * 252
    annual_volatility = std_return * (252 ** 0.5)
    annual_sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    
    return {
        'mean_return': mean_return,
        'std_return': std_return, 
        'sharpe_ratio': sharpe_ratio,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'annual_sharpe': annual_sharpe
    }