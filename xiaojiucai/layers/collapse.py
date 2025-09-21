"""维度压缩层模块。

包含了用于降维和特征聚合的层，如Attention、Average等。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCollapse(nn.Module):
    """基于注意力机制的维度压缩层。
    
    使用注意力机制对输入序列进行加权聚合。
    
    Parameters
    ----------
    in_features : int
        输入特征维度。
        
    hidden_dim : int
        注意力隐藏层维度。
    """
    
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """前向传播。
        
        Parameters
        ----------
        x : torch.Tensor
            输入tensor，形状为(batch_size, seq_len, in_features)。
            
        Returns
        -------
        y : torch.Tensor
            压缩后的tensor，形状为(batch_size, in_features)。
        """
        # 计算注意力权重
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        
        # 加权聚合
        y = torch.sum(x * attention_weights, dim=1)  # (batch_size, in_features)
        
        return y


class AverageCollapse(nn.Module):
    """平均池化层。
    
    对序列维度做平均池化。
    """
    
    def forward(self, x):
        """前向传播。
        
        Parameters
        ----------
        x : torch.Tensor
            输入tensor，形状为(batch_size, seq_len, features)。
            
        Returns
        -------
        y : torch.Tensor
            压缩后的tensor，形状为(batch_size, features)。
        """
        return x.mean(dim=1)


class MaxCollapse(nn.Module):
    """最大池化层。
    
    对序列维度做最大池化。
    """
    
    def forward(self, x):
        """前向传播。
        
        Parameters
        ----------
        x : torch.Tensor
            输入tensor，形状为(batch_size, seq_len, features)。
            
        Returns
        -------
        y : torch.Tensor
            压缩后的tensor，形状为(batch_size, features)。
        """
        return x.max(dim=1)[0]


class SumCollapse(nn.Module):
    """求和层。
    
    对序列维度做求和。
    """
    
    def forward(self, x):
        """前向传播。
        
        Parameters
        ----------
        x : torch.Tensor
            输入tensor，形状为(batch_size, seq_len, features)。
            
        Returns
        -------
        y : torch.Tensor
            压缩后的tensor，形状为(batch_size, features)。
        """
        return x.sum(dim=1)