"""其他辅助层模块。

包含了一些辅助用途的神经网络层。
"""

import numpy as np
import torch
import torch.nn as nn


class CovarianceMatrix(nn.Module):
    """协方差矩阵计算层。
    
    用于计算特征之间的协方差矩阵。
    
    Parameters
    ----------
    eps : float
        添加到对角线的小值，用于数值稳定性。
        
    sqrt : bool
        是否返回协方差矩阵的平方根。
    """
    
    def __init__(self, eps=1e-5, sqrt=False):
        super().__init__()
        self.eps = eps
        self.sqrt = sqrt
        
    def forward(self, x):
        """前向传播。
        
        Parameters
        ----------
        x : torch.Tensor
            输入tensor，形状为(batch_size, seq_len, features)。
            
        Returns
        -------
        sigma : torch.Tensor
            协方差矩阵，形状为(batch_size, features, features)。
        """
        # 计算特征均值
        mean = x.mean(dim=1, keepdim=True)  # (batch_size, 1, features)
        
        # 去均值
        x_centered = x - mean  # (batch_size, seq_len, features)
        
        # 计算协方差矩阵
        factor = 1.0 / (x.size(1) - 1)
        sigma = factor * torch.bmm(x_centered.transpose(1, 2), x_centered)  # (batch_size, features, features)
        
        # 添加eps到对角线
        sigma = sigma + torch.eye(sigma.size(-1), device=x.device) * self.eps
        
        if self.sqrt:
            # 计算平方根（通过特征值分解）
            try:
                L = torch.linalg.cholesky(sigma)  # (batch_size, features, features)
                return L
            except RuntimeError:
                # 如果Cholesky分解失败，使用特征值分解
                eigenvalues, eigenvectors = torch.linalg.eigh(sigma)  # (batch_size, features), (batch_size, features, features)
                # 处理负特征值
                sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=self.eps))  # (batch_size, features)
                sqrt_sigma = torch.bmm(
                    torch.bmm(eigenvectors, torch.diag_embed(sqrt_eigenvalues)),
                    eigenvectors.transpose(-2, -1)
                )  # (batch_size, features, features)
                return sqrt_sigma
        
        return sigma


class MultiplyByConstant(nn.Module):
    """常数乘法层。
    
    用于将输入乘以一个常数。
    
    Parameters
    ----------
    constant : float
        乘数。
        
    learnable : bool
        是否将乘数设为可学习参数。
    """
    
    def __init__(self, constant, learnable=False):
        super().__init__()
        if learnable:
            self.constant = nn.Parameter(torch.tensor(constant))
        else:
            self.register_buffer('constant', torch.tensor(constant))
            
    def forward(self, x):
        """前向传播。
        
        Parameters
        ----------
        x : torch.Tensor
            输入tensor。
            
        Returns
        -------
        y : torch.Tensor
            输出tensor = 输入 * 常数。
        """
        return x * self.constant