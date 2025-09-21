"""Return-based loss functions.

This module provides various return-based loss implementations:
- MeanReturns: Average return loss
- SharpeRatio: Sharpe ratio loss
- SortinoRatio: Sortino ratio loss
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union

from .base import Loss, log2simple, simple2log


class MeanReturns(Loss):
    """Mean returns loss.

    A loss that uses the negative expected portfolio return as the loss value.

    Args:
        returns_channel: Channel index for returns in target tensor
        input_type: Type of returns in target tensor ('log' or 'simple')
        output_type: Type of returns in output ('log' or 'simple')
        normalize: Whether to normalize returns
        eps: Small constant to prevent division by zero

    Attributes:
        metrics: Dictionary containing computed metrics
    """
    
    def __init__(
        self,
        returns_channel: int = 0,
        input_type: str = 'log',
        output_type: str = 'simple',
        normalize: bool = False,
        eps: float = 1e-8
    ):
        super().__init__()
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type
        self.normalize = normalize
        self.eps = eps
        self.metrics = {}

    def forward(
        self,
        weights: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean returns loss.

        Args:
            weights: Portfolio weights of shape (batch_size, n_assets)
            y: Target tensor of shape (batch_size, n_channels, horizon, n_assets)

        Returns:
            Loss tensor of shape (batch_size,)
        """
        # Get returns data
        returns = y[:, self.returns_channel]  # (batch_size, horizon, n_assets)
        
        # Convert returns if needed
        if self.input_type != self.output_type:
            if self.input_type == 'log' and self.output_type == 'simple':
                returns = log2simple(returns)
            else:
                returns = simple2log(returns)
                
        # Normalize returns if requested
        if self.normalize:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        # Compute portfolio returns
        portfolio_returns = torch.sum(
            weights.unsqueeze(1) * returns,
            dim=-1
        )  # (batch_size, horizon)
        
        # Compute mean return loss
        mean_returns = torch.mean(portfolio_returns, dim=-1)  # (batch_size,)
        loss = -mean_returns  # Negative because we want to maximize returns
        
        # Store metrics
        self.metrics = {
            'mean_returns': mean_returns.detach(),
            'portfolio_returns': portfolio_returns.detach()
        }
        
        return loss


class SharpeRatio(Loss):
    """Sharpe ratio loss.

    A loss that uses the negative Sharpe ratio as the loss value.

    Args:
        rf: Risk-free rate (annualized)
        returns_channel: Channel index for returns in target tensor
        input_type: Type of returns in target tensor ('log' or 'simple')
        output_type: Type of returns in output ('log' or 'simple')
        eps: Small constant to prevent division by zero

    Attributes:
        metrics: Dictionary containing computed metrics
    """
    
    def __init__(
        self,
        rf: float = 0.0,
        returns_channel: int = 0,
        input_type: str = 'log',
        output_type: str = 'simple',
        eps: float = 1e-8
    ):
        super().__init__()
        self.rf = rf
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type
        self.eps = eps
        self.metrics = {}

    def forward(
        self,
        weights: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute Sharpe ratio loss.

        Args:
            weights: Portfolio weights of shape (batch_size, n_assets)
            y: Target tensor of shape (batch_size, n_channels, horizon, n_assets)

        Returns:
            Loss tensor of shape (batch_size,)
        """
        # Get returns data
        returns = y[:, self.returns_channel]  # (batch_size, horizon, n_assets)
        
        # Convert returns if needed
        if self.input_type != self.output_type:
            if self.input_type == 'log' and self.output_type == 'simple':
                returns = log2simple(returns)
            else:
                returns = simple2log(returns)
                
        # Compute portfolio returns
        portfolio_returns = torch.sum(
            weights.unsqueeze(1) * returns,
            dim=-1
        )  # (batch_size, horizon)
        
        # Compute mean and std of returns
        mean_returns = torch.mean(portfolio_returns, dim=-1)  # (batch_size,)
        std_returns = torch.std(portfolio_returns, dim=-1)  # (batch_size,)
        
        # Compute Sharpe ratio loss
        loss = -(mean_returns - self.rf) / (std_returns + self.eps)
        
        # Store metrics
        self.metrics = {
            'mean_returns': mean_returns.detach(),
            'std_returns': std_returns.detach(),
            'sharpe_ratio': -loss.detach(),
            'portfolio_returns': portfolio_returns.detach()
        }
        
        return loss


class SortinoRatio(Loss):
    """Sortino ratio loss.

    A loss that uses the negative Sortino ratio as the loss value.

    Args:
        rf: Risk-free rate (annualized)
        returns_channel: Channel index for returns in target tensor
        input_type: Type of returns in target tensor ('log' or 'simple')
        output_type: Type of returns in output ('log' or 'simple')
        eps: Small constant to prevent division by zero

    Attributes:
        metrics: Dictionary containing computed metrics
    """
    
    def __init__(
        self,
        rf: float = 0.0,
        returns_channel: int = 0,
        input_type: str = 'log',
        output_type: str = 'simple',
        eps: float = 1e-8
    ):
        super().__init__()
        self.rf = rf
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type
        self.eps = eps
        self.metrics = {}

    def forward(
        self,
        weights: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute Sortino ratio loss.

        Args:
            weights: Portfolio weights of shape (batch_size, n_assets)
            y: Target tensor of shape (batch_size, n_channels, horizon, n_assets)

        Returns:
            Loss tensor of shape (batch_size,)
        """
        # Get returns data
        returns = y[:, self.returns_channel]  # (batch_size, horizon, n_assets)
        
        # Convert returns if needed
        if self.input_type != self.output_type:
            if self.input_type == 'log' and self.output_type == 'simple':
                returns = log2simple(returns)
            else:
                returns = simple2log(returns)
        
        # Compute portfolio returns
        portfolio_returns = torch.sum(
            weights.unsqueeze(1) * returns,
            dim=-1
        )  # (batch_size, horizon)
        
        # Compute mean returns
        mean_returns = torch.mean(portfolio_returns, dim=-1)  # (batch_size,)
        
        # Compute downside deviation
        negative_returns = torch.clamp(
            portfolio_returns - self.rf,
            max=0
        )  # (batch_size, horizon)
        
        downside_std = torch.sqrt(
            torch.mean(negative_returns ** 2, dim=-1)
        )  # (batch_size,)
        
        # Compute Sortino ratio loss
        loss = -(mean_returns - self.rf) / (downside_std + self.eps)
        
        # Store metrics
        self.metrics = {
            'mean_returns': mean_returns.detach(),
            'downside_std': downside_std.detach(),
            'sortino_ratio': -loss.detach(),
            'portfolio_returns': portfolio_returns.detach()
        }
        
        return loss