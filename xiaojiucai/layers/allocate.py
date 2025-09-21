"""投资组合分配层模块。

包含了用于投资组合权重分配的层，如Markowitz优化、风险预算等。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnalyticalMarkowitz(nn.Module):
    """解析解Markowitz模型层。
    
    使用矩阵计算直接求解无约束的Markowitz均值-方差优化问题。
    
    Parameters
    ----------
    gamma : float
        风险厌恶系数。
    """
    
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, mu, sigma):
        """前向传播。
        
        Parameters
        ----------
        mu : torch.Tensor
            预期收益率向量，形状为(batch_size, n_assets)。
            
        sigma : torch.Tensor
            协方差矩阵，形状为(batch_size, n_assets, n_assets)。
            
        Returns
        -------
        w : torch.Tensor
            最优权重向量，形状为(batch_size, n_assets)。
        """
        # 计算最优权重: w = (1/gamma) * Σ^(-1) * μ
        try:
            sigma_inv = torch.linalg.inv(sigma)  # (batch_size, n_assets, n_assets)
            w = torch.bmm(sigma_inv, mu.unsqueeze(-1)).squeeze(-1)  # (batch_size, n_assets)
            w = w / self.gamma
            
            # 归一化权重
            w = w / w.sum(dim=1, keepdim=True)
            
            return w
            
        except RuntimeError as e:
            print(f"Warning: Matrix inversion failed: {e}")
            # 出错时返回等权重
            return torch.ones_like(mu) / mu.shape[1]


class NumericalMarkowitz(nn.Module):
    """数值求解Markowitz模型层。
    
    使用梯度下降等数值优化方法求解带约束的Markowitz均值-方差优化问题。
    
    Parameters
    ----------
    n_assets : int
        资产数量。
        
    max_weight : float
        单个资产的最大权重限制。
        
    gamma : float
        风险厌恶系数。
    """
    
    def __init__(self, n_assets, max_weight=1.0, gamma=1.0):
        super().__init__()
        self.n_assets = n_assets
        self.max_weight = max_weight
        self.gamma = gamma
        
    def forward(self, mu, sigma):
        """前向传播。
        
        Parameters
        ----------
        mu : torch.Tensor
            预期收益率向量，形状为(batch_size, n_assets)。
            
        sigma : torch.Tensor
            协方差矩阵，形状为(batch_size, n_assets, n_assets)。
            
        Returns
        -------
        w : torch.Tensor
            最优权重向量，形状为(batch_size, n_assets)。
        """
        batch_size = mu.shape[0]
        
        # 初始化为等权重
        w = torch.ones(batch_size, self.n_assets, device=mu.device)
        w = w / self.n_assets
        w.requires_grad = True
        
        optimizer = torch.optim.Adam([w], lr=0.01)
        
        for _ in range(100):  # 最大迭代次数
            optimizer.zero_grad()
            
            # 计算目标函数
            expected_return = torch.sum(w * mu, dim=1)  # (batch_size,)
            risk = torch.sum(w.unsqueeze(1) * sigma * w.unsqueeze(2), dim=(1,2))  # (batch_size,)
            objective = -expected_return + self.gamma * risk
            
            # 添加约束的惩罚项
            sum_constraint = torch.abs(w.sum(dim=1) - 1).mean()  # 权重和为1
            bound_constraint = F.relu(w - self.max_weight).mean() + F.relu(-w).mean()  # 权重范围约束
            
            loss = objective.mean() + 100 * (sum_constraint + bound_constraint)
            
            loss.backward()
            optimizer.step()
            
            # 投影到可行域
            with torch.no_grad():
                w.data.clamp_(min=0, max=self.max_weight)
                w.data.div_(w.data.sum(dim=1, keepdim=True))
        
        return w.detach()


class NumericalRiskBudgeting(nn.Module):
    """风险预算优化层。
    
    基于给定的风险预算分配权重。
    
    Parameters
    ----------
    n_assets : int
        资产数量。
        
    max_weight : float
        单个资产的最大权重限制。
    """
    
    def __init__(self, n_assets, max_weight=1.0):
        super().__init__()
        self.n_assets = n_assets
        self.max_weight = max_weight
        
    def forward(self, sigma, b):
        """前向传播。
        
        Parameters
        ----------
        sigma : torch.Tensor
            协方差矩阵，形状为(batch_size, n_assets, n_assets)。
            
        b : torch.Tensor
            风险预算向量，形状为(batch_size, n_assets)。
            
        Returns
        -------
        w : torch.Tensor
            最优权重向量，形状为(batch_size, n_assets)。
        """
        batch_size = sigma.shape[0]
        
        # 初始化为等权重
        w = torch.ones(batch_size, self.n_assets, device=sigma.device)
        w = w / self.n_assets
        w.requires_grad = True
        
        optimizer = torch.optim.Adam([w], lr=0.01)
        
        for _ in range(100):  # 最大迭代次数
            optimizer.zero_grad()
            
            # 计算每个资产的风险贡献
            portfolio_risk = torch.sum(w.unsqueeze(1) * sigma * w.unsqueeze(2), dim=(1,2))  # (batch_size,)
            marginal_risk = 2 * torch.sum(sigma * w.unsqueeze(2), dim=1)  # (batch_size, n_assets)
            risk_contribution = w * marginal_risk  # (batch_size, n_assets)
            
            # 目标函数：风险贡献与预算的差异
            objective = torch.sum((risk_contribution - b * portfolio_risk.unsqueeze(1))**2, dim=1)
            
            # 添加约束的惩罚项
            sum_constraint = torch.abs(w.sum(dim=1) - 1).mean()
            bound_constraint = F.relu(w - self.max_weight).mean() + F.relu(-w).mean()
            
            loss = objective.mean() + 100 * (sum_constraint + bound_constraint)
            
            loss.backward()
            optimizer.step()
            
            # 投影到可行域
            with torch.no_grad():
                w.data.clamp_(min=0, max=self.max_weight)
                w.data.div_(w.data.sum(dim=1, keepdim=True))
        
        return w.detach()


class SoftmaxAllocator(nn.Module):
    """基于Softmax的权重分配层。
    
    使用Softmax函数将任意实数映射为权重向量。
    
    Parameters
    ----------
    temperature : float
        Softmax温度参数，控制权重分布的集中度。
    """
    
    def __init__(self, temperature=None):
        super().__init__()
        if temperature is None:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.temperature = temperature
            
    def forward(self, x, temperature=None):
        """Convert scores to portfolio weights.
        
        Parameters
        ----------
        x : torch.Tensor
            Scores of shape (n_samples, n_assets).
            
        temperature : torch.Tensor, optional
            Temperature for softmax. If None, use self.temperature.
            
        Returns
        -------
        weights : torch.Tensor
            Portfolio weights of shape (n_samples, n_assets).
        """
        if temperature is None:
            temperature = self.temperature
            
        weights = F.softmax(x / temperature, dim=1)
        return weights


class LinearAllocator(nn.Module):
    """Linear allocator with weight normalization.
    
    This layer converts raw scores into portfolio weights by normalizing
    to sum to 1.
    
    Parameters
    ----------
    n_assets : int
        Number of assets.
    """
    
    def __init__(self, n_assets):
        super().__init__()
        self.n_assets = n_assets
        
    def forward(self, x):
        """Convert scores to portfolio weights.
        
        Parameters
        ----------
        x : torch.Tensor
            Scores of shape (n_samples, n_assets).
            
        Returns
        -------
        weights : torch.Tensor
            Portfolio weights of shape (n_samples, n_assets).
        """
        # Ensure weights are positive and sum to 1
        weights = F.relu(x)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights