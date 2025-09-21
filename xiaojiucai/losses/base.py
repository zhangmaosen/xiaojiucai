"""Base module for loss functions.

This module defines base classes and utility functions for loss computation:
- Loss: Abstract base class for all loss functions
- ReturnConversion: Utilities for return type conversion
- MetricMixin: Mixin class for metric computation and logging
"""

import torch
import torch.nn as nn
from types import MethodType
from typing import Dict, Optional, Union


def log2simple(x: torch.Tensor) -> torch.Tensor:
    """Convert log returns to simple returns.

    Args:
        x: Log returns tensor

    Returns:
        Simple returns tensor = exp(log_returns) - 1
    """
    return torch.exp(x) - 1


def simple2log(x: torch.Tensor) -> torch.Tensor:
    """Convert simple returns to log returns.

    Args:
        x: Simple returns tensor

    Returns:
        Log returns tensor = ln(simple_returns + 1)
    """
    return torch.log(x + 1)


class MetricMixin:
    """Mixin class for metric computation and logging.

    This class provides functionality for computing and logging evaluation metrics.
    """
    
    def compute_metrics(
        self,
        weights: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics.

        Args:
            weights: Portfolio weights tensor of shape (batch_size, n_assets)
            y: Target tensor of shape (batch_size, n_channels, horizon, n_assets)

        Returns:
            Dictionary containing computed metrics
        """
        raise NotImplementedError()
        
    def log_metrics(self, metrics: Dict[str, torch.Tensor]) -> None:
        """Log computed metrics.

        Args:
            metrics: Dictionary containing metrics to log
        """
        pass


class Loss(nn.Module, MetricMixin):
    """Base class for all loss functions.

    This class implements operator overloading for arithmetic operations,
    allowing loss functions to be combined. It also inherits from MetricMixin
    for metric computation and logging.

    Args:
        returns_channel: Channel index for returns in target tensor
        input_type: Type of returns in target tensor ('log' or 'simple')
        output_type: Type of returns in output ('log' or 'simple')
        normalize: Whether to normalize loss values
    """
    
    def __init__(
        self,
        returns_channel: int = 0,
        input_type: str = "log",
        output_type: str = "simple",
        normalize: bool = False
    ):
        super().__init__()
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type
        self.normalize = normalize
    
    def forward(
        self,
        weights: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss value.

        Args:
            weights: Portfolio weights tensor of shape (batch_size, n_assets)
            y: Target tensor of shape (batch_size, n_channels, horizon, n_assets)

        Returns:
            Loss tensor of shape (batch_size,)
        """
        loss = self._compute_loss(weights, y)
        metrics = self.compute_metrics(weights, y)
        self.log_metrics(metrics)
        return loss
        
    def _compute_loss(
        self,
        weights: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute raw loss value.

        This method should be implemented by subclasses to define specific
        loss computation logic.

        Args:
            weights: Portfolio weights tensor of shape (batch_size, n_assets)
            y: Target tensor of shape (batch_size, n_channels, horizon, n_assets)

        Returns:
            Loss tensor of shape (batch_size,)
        """
        raise NotImplementedError()
        
    def __add__(self, other: Union['Loss', float]) -> 'Loss':
        """Addition operator overloading.

        Args:
            other: Another loss function or a constant value

        Returns:
            A new loss function that combines the two operands

        Raises:
            TypeError: If the operation type is not supported
        """
        if isinstance(other, Loss):
            new_instance = Loss()
            new_instance.forward = MethodType(
                lambda inst, weights, y: self(weights, y) + other(weights, y),
                new_instance
            )
            return new_instance
        elif isinstance(other, (int, float)):
            new_instance = Loss()
            new_instance.forward = MethodType(
                lambda inst, weights, y: self(weights, y) + other,
                new_instance
            )
            return new_instance
        else:
            raise TypeError(f"Unsupported operation type: '{type(other)}'")
            
    def __mul__(self, scalar: Union[int, float]) -> 'Loss':
        """Multiplication operator overloading.

        Args:
            scalar: A scalar value to multiply with

        Returns:
            A new loss function that multiplies the loss by scalar

        Raises:
            TypeError: If the operation type is not supported
        """
        if isinstance(scalar, (int, float)):
            new_instance = Loss()
            new_instance.forward = MethodType(
                lambda inst, weights, y: scalar * self(weights, y),
                new_instance
            )
            return new_instance
        else:
            raise TypeError(f"Unsupported operation type: '{type(scalar)}'")
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Loss':
        """Reverse multiplication operator overloading.

        Args:
            scalar: A scalar value to multiply with

        Returns:
            A new loss function that multiplies the loss by scalar
        """
        return self.__mul__(scalar)