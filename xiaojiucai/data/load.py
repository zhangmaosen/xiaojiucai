"""
Data loading and preprocessing utilities for portfolio optimization.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class FlexibleDataLoader(DataLoader):
    """允许动态调整批次大小和抽样的数据加载器。
    
    Parameters
    ----------
    dataset : Dataset
        数据集。
        
    batch_size : int or callable
        批次大小或返回批次大小的函数。
        
    sampler : Sampler or callable, optional
        采样器或返回indices的函数。
        
    transform : callable, optional
        应用于每个批次的变换。
    """
    
    def __init__(self, dataset, batch_size, sampler=None, transform=None):
        self.dynamic_batch_size = callable(batch_size)
        self.batch_size_fn = batch_size if self.dynamic_batch_size else lambda: batch_size
        
        self.dynamic_sampler = callable(sampler)
        self.sampler_fn = sampler
        self.transform = transform
        
        super().__init__(
            dataset,
            batch_size=self.batch_size_fn() if not self.dynamic_batch_size else 1,
            sampler=sampler() if self.dynamic_sampler else sampler
        )
        
    def __iter__(self):
        if self.dynamic_sampler:
            self.sampler = self.sampler_fn()
            
        if self.dynamic_batch_size:
            self.batch_size = self.batch_size_fn()
            
        batch = []
        for idx in self.sampler:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                if self.transform:
                    yield self.transform(batch)
                else:
                    yield batch
                batch = []
        
        if batch:
            if self.transform:
                yield self.transform(batch)
            else:
                yield batch

class InRAMDataset(Dataset):
    """Dataset that stores entire dataset in RAM for fast access.
    
    Parameters
    ----------
    X : np.ndarray
        Features of shape (n_samples, n_channels, lookback, n_assets).
        
    y : np.ndarray
        Targets of shape (n_samples, n_channels, horizon, n_assets).
        
    transform : callable, optional
        Transformation to apply to the features.
    """
    
    def __init__(self, X, y, transform=None):
        # Check for NaN or Inf in input data
        self.has_nan_in_X = np.any(np.isnan(X)) or np.any(np.isinf(X))
        self.has_nan_in_y = np.any(np.isnan(y)) or np.any(np.isinf(y))
        
        if self.has_nan_in_X:
            print("Warning: NaN or Inf values detected in features (X)")
            
        if self.has_nan_in_y:
            print("Warning: NaN or Inf values detected in targets (y)")
            
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        
        # Check for NaN or Inf in individual samples (only for first few samples to reduce output)
        if idx < 5:  # Only check and warn for first 5 samples
            if np.any(np.isnan(X_sample)) or np.any(np.isinf(X_sample)):
                print(f"Warning: NaN or Inf values detected in feature sample at index {idx}")
                X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=1e6, neginf=-1e6)
                
            if np.any(np.isnan(y_sample)) or np.any(np.isinf(y_sample)):
                print(f"Warning: NaN or Inf values detected in target sample at index {idx}")
                y_sample = np.nan_to_num(y_sample, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            # Just fix NaN/Inf without warning for later samples
            if np.any(np.isnan(X_sample)) or np.any(np.isinf(X_sample)):
                X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=1e6, neginf=-1e6)
                
            if np.any(np.isnan(y_sample)) or np.any(np.isinf(y_sample)):
                y_sample = np.nan_to_num(y_sample, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if self.transform:
            X_sample = self.transform(X_sample)
            
        return X_sample, y_sample


class RigidDataLoader(DataLoader):
    """Data loader with fixed batch size and indices.
    
    Parameters
    ----------
    dataset : Dataset
        Dataset to load from.
        
    indices : list of int
        Indices to sample from.
        
    batch_size : int
        Size of each batch.
        
    shuffle : bool, default False
        Whether to shuffle the data.
        
    device : str, optional
        Device to load data to.
    """
    
    def __init__(self, dataset, indices, batch_size, shuffle=False, device=None):
        self.indices = indices
        self.device = device
        
        # Create a subset of the dataset
        subset = torch.utils.data.Subset(dataset, indices)
        
        super().__init__(
            subset,
            batch_size=batch_size,
            shuffle=shuffle
        )


def prepare_standard_scaler(X, indices=None):
    """Compute mean and standard deviation for standard scaling.
    
    Parameters
    ----------
    X : np.ndarray
        Input data.
        
    indices : list of int, optional
        Indices to compute statistics over.
        
    Returns
    -------
    means : np.ndarray
        Means for each channel.
        
    stds : np.ndarray
        Standard deviations for each channel.
    """
    if indices is not None:
        X = X[indices]
        
    # Check for NaN or Inf in input (only warn once)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: NaN or Inf values detected in input data for scaling")
        # Replace NaN with 0 and Inf with large finite numbers
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
    # Assuming X has shape (n_samples, n_channels, lookback, n_assets)
    means = np.mean(X, axis=(0, 2, 3), keepdims=True)
    stds = np.std(X, axis=(0, 2, 3), keepdims=True)
    
    # Check for NaN or Inf in computed statistics (only warn once)
    means_problem = np.any(np.isnan(means)) or np.any(np.isinf(means))
    stds_problem = np.any(np.isnan(stds)) or np.any(np.isinf(stds))
    
    if means_problem or stds_problem:
        print("Warning: NaN or Inf values detected in computed scaling statistics")
        # Replace problematic values
        if means_problem:
            means = np.nan_to_num(means, nan=0.0, posinf=1e6, neginf=-1e6)
        if stds_problem:
            stds = np.nan_to_num(stds, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Avoid division by zero
    stds = np.where(stds == 0, 1, stds)
    
    return means, stds


class Scale:
    """Scale transform for standardizing features.
    
    Parameters
    ----------
    means : np.ndarray
        Means to subtract.
        
    stds : np.ndarray
        Standard deviations to divide by.
    """
    
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds
        self.warning_shown = False
        
    def __call__(self, X):
        """Apply scaling to input.
        
        Parameters
        ----------
        X : np.ndarray
            Input to scale.
            
        Returns
        -------
        X_scaled : np.ndarray
            Scaled input.
        """
        # Check for NaN or Inf in input (only warn once)
        if (np.any(np.isnan(X)) or np.any(np.isinf(X))) and not self.warning_shown:
            print("Warning: NaN or Inf values detected in input data")
            self.warning_shown = True
            # Replace NaN with 0 and Inf with large finite numbers
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        elif np.any(np.isnan(X)) or np.any(np.isinf(X)):
            # Just fix without warning
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
        # Check for NaN or Inf in means or stds (only warn once)
        means_problem = np.any(np.isnan(self.means)) or np.any(np.isinf(self.means))
        stds_problem = np.any(np.isnan(self.stds)) or np.any(np.isinf(self.stds))
        
        if (means_problem or stds_problem) and not self.warning_shown:
            print("Warning: NaN or Inf values detected in scaling parameters")
            self.warning_shown = True
            # Replace problematic values
            means = np.nan_to_num(self.means, nan=0.0, posinf=1e6, neginf=-1e6)
            stds = np.nan_to_num(self.stds, nan=1.0, posinf=1.0, neginf=1.0)
            # Ensure stds are not zero to avoid division issues
            stds = np.where(np.abs(stds) < 1e-8, 1.0, stds)
            return (X - means) / stds
        elif means_problem or stds_problem:
            # Just fix without warning
            means = np.nan_to_num(self.means, nan=0.0, posinf=1e6, neginf=-1e6)
            stds = np.nan_to_num(self.stds, nan=1.0, posinf=1.0, neginf=1.0)
            # Ensure stds are not zero to avoid division issues
            stds = np.where(np.abs(stds) < 1e-8, 1.0, stds)
            return (X - means) / stds
            
        result = (X - self.means) / self.stds
        
        # Check for NaN or Inf in result (only warn once)
        if (np.any(np.isnan(result)) or np.any(np.isinf(result))) and not self.warning_shown:
            print("Warning: NaN or Inf values generated after scaling")
            self.warning_shown = True
            # Replace NaN with 0 and Inf with large finite numbers
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        elif np.any(np.isnan(result)) or np.any(np.isinf(result)):
            # Just fix without warning
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
        return result
