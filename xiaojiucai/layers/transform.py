"""特征变换层模块。

包含了用于特征转换的层，如卷积、循环神经网络等。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """一维卷积层。
    
    用于序列数据的特征提取。
    
    Parameters
    ----------
    in_channels : int
        输入通道数。
        
    out_channels : int
        输出通道数。
        
    kernel_size : int
        卷积核大小。
        
    stride : int
        步长。
        
    padding : int or str
        填充方式。
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding if isinstance(padding, str) else padding
        )
        
    def forward(self, x):
        """前向传播。
        
        Parameters
        ----------
        x : torch.Tensor
            输入tensor，形状为(batch_size, in_channels, seq_len)。
            
        Returns
        -------
        y : torch.Tensor
            输出tensor，形状为(batch_size, out_channels, new_seq_len)。
        """
        return self.conv(x)


class RNN(nn.Module):
    """循环神经网络层。
    
    包装了PyTorch的RNN、LSTM、GRU等循环网络。
    
    Parameters
    ----------
    input_size : int
        输入特征维度。
        
    hidden_size : int
        隐藏状态维度。
        
    num_layers : int
        循环层数量。
        
    rnn_type : str
        循环网络类型，可选'rnn', 'lstm', 'gru'。
        
    dropout : float
        dropout比率。
        
    bidirectional : bool
        是否使用双向循环网络。
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 rnn_type='lstm', dropout=0, bidirectional=False):
        super().__init__()
        
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        rnn_cls = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }[self.rnn_type]
        
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
    def forward(self, x, hx=None):
        """前向传播。
        
        Parameters
        ----------
        x : torch.Tensor
            输入tensor，形状为(batch_size, seq_len, input_size)。
            
        hx : torch.Tensor or tuple[torch.Tensor]
            初始隐藏状态，对LSTM为(h0, c0)元组。
            
        Returns
        -------
        output : torch.Tensor
            输出序列，形状为(batch_size, seq_len, hidden_size * num_directions)。
            
        hn : torch.Tensor or tuple[torch.Tensor]
            最终隐藏状态，对LSTM为(hn, cn)元组。
        """
        return self.rnn(x, hx)