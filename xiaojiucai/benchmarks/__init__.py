# 在__init__.py中定义Benchmark类，避免循环导入问题
import torch
import numpy as np


class Benchmark:
    """Base class for benchmark models."""
    
    def __init__(self):
        pass
        
    def __call__(self, X):
        """Compute portfolio weights.
        
        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples, n_channels, lookback, n_assets).
            
        Returns
        -------
        weights : torch.Tensor
            Portfolio weights of shape (n_samples, n_assets).
        """
        raise NotImplementedError


class OneOverN(Benchmark):
    """1/N portfolio - equal weight allocation.
    
    This benchmark allocates equal weights to all assets.
    """
    
    def __call__(self, X):
        """Compute equal weight portfolio.
        
        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples, n_channels, lookback, n_assets).
            
        Returns
        -------
        weights : torch.Tensor
            Portfolio weights of shape (n_samples, n_assets).
        """
        n_samples, _, _, n_assets = X.shape
        weights = torch.ones(n_samples, n_assets) / n_assets
        return weights


class Random(Benchmark):
    """Random portfolio - random weight allocation.
    
    This benchmark allocates random weights to assets and then normalizes.
    """
    
    def __call__(self, X):
        """Compute random weight portfolio.
        
        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples, n_channels, lookback, n_assets).
            
        Returns
        -------
        weights : torch.Tensor
            Portfolio weights of shape (n_samples, n_assets).
        """
        n_samples, _, _, n_assets = X.shape
        weights = torch.rand(n_samples, n_assets)
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights


class InverseVolatility(Benchmark):
    """Inverse volatility portfolio.
    
    This benchmark allocates weights inversely proportional to asset volatilities.
    """
    
    def __call__(self, X):
        """Compute inverse volatility portfolio.
        
        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples, n_channels, lookback, n_assets).
            
        Returns
        -------
        weights : torch.Tensor
            Portfolio weights of shape (n_samples, n_assets).
        """
        # Calculate volatilities (standard deviation) across lookback dimension
        volatilities = torch.std(X, dim=2)  # (n_samples, n_channels, n_assets)
        
        # Take the first channel
        volatilities = volatilities[:, 0, :]  # (n_samples, n_assets)
        
        # Calculate inverse volatilities (add small epsilon to avoid division by zero)
        inverse_volatilities = 1.0 / (volatilities + 1e-8)
        
        # Normalize weights
        weights = inverse_volatilities / inverse_volatilities.sum(dim=1, keepdim=True)
        
        return weights


class MaximumReturn(Benchmark):
    """Maximum return portfolio.
    
    This benchmark allocates weights proportional to recent asset returns.
    """
    
    def __call__(self, X):
        """Compute maximum return portfolio.
        
        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples, n_channels, lookback, n_assets).
            
        Returns
        -------
        weights : torch.Tensor
            Portfolio weights of shape (n_samples, n_assets).
        """
        # Calculate mean returns across lookback dimension
        mean_returns = torch.mean(X, dim=2)  # (n_samples, n_channels, n_assets)
        
        # Take the first channel
        mean_returns = mean_returns[:, 0, :]  # (n_samples, n_assets)
        
        # Make sure all weights are positive by shifting mean returns
        min_return = torch.min(mean_returns, dim=1, keepdim=True)[0]
        shifted_returns = mean_returns - min_return + 1e-8
        
        # Normalize weights
        weights = shifted_returns / shifted_returns.sum(dim=1, keepdim=True)
        
        return weights

__all__ = ['Benchmark', 'OneOverN', 'Random', 'InverseVolatility', 'MaximumReturn']