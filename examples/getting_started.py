"""
===============
Getting started
===============

This is a simple example showing how to use the portfolio optimization framework.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from xiaojiucai.data.load import InRAMDataset, RigidDataLoader, prepare_standard_scaler, Scale
from xiaojiucai.models.networks import GreatNet
from xiaojiucai.benchmarks import OneOverN, Random
from xiaojiucai.losses import MaximumDrawdown, MeanReturns, SharpeRatio
from xiaojiucai.experiments import Run
from xiaojiucai.callbacks import EarlyStoppingCallback


def sin_single(n_timesteps, freq, amplitude=1, phase=0):
    """Generate a sine wave signal.
    
    Parameters
    ----------
    n_timesteps : int
        Number of timesteps.
        
    freq : float
        Frequency.
        
    amplitude : float, default 1
        Amplitude.
        
    phase : float, default 0
        Phase.
        
    Returns
    -------
    signal : np.ndarray
        Sine wave signal.
    """
    return amplitude * np.sin(2 * np.pi * freq * np.arange(n_timesteps) + phase)


def main():
    """Main function to run the example."""
    # Set random seeds for reproducibility
    torch.manual_seed(4)
    np.random.seed(5)
    
    # Parameters
    n_timesteps, n_assets = 1000, 20
    lookback, gap, horizon = 40, 2, 20
    n_samples = n_timesteps - lookback - horizon - gap + 1
    
    # Split data into train and test
    split_ix = int(n_samples * 0.8)
    indices_train = list(range(split_ix))
    indices_test = list(range(split_ix + lookback + horizon, n_samples))
    
    print(f'Train range: {indices_train[0]}:{indices_train[-1]}')
    print(f'Test range: {indices_test[0]}:{indices_test[-1]}')
    
    # Generate synthetic asset returns
    returns = np.array([
        sin_single(n_timesteps,
                   freq=1 / np.random.randint(3, lookback),
                   amplitude=0.05,
                   phase=np.random.randint(0, lookback))
        for _ in range(n_assets)
    ]).T
    
    # Add noise
    returns += np.random.normal(scale=0.02, size=returns.shape)
    
    # Create features and targets using rolling window
    X_list, y_list = [], []
    
    for i in range(lookback, n_timesteps - horizon - gap + 1):
        X_list.append(returns[i - lookback: i, :])
        y_list.append(returns[i + gap: i + gap + horizon, :])
        
    X = np.stack(X_list, axis=0)[:, None, ...]
    y = np.stack(y_list, axis=0)[:, None, ...]
    
    print(f'X: {X.shape}, y: {y.shape}')
    
    # Scale features
    means, stds = prepare_standard_scaler(X, indices=indices_train)
    print(f'mean: {means}, std: {stds}')
    
    # Create dataset
    dataset = InRAMDataset(X, y, transform=Scale(means, stds))
    
    # Create dataloaders
    dataloader_train = RigidDataLoader(dataset,
                                       indices=indices_train,
                                       batch_size=32)
                                       
    dataloader_test = RigidDataLoader(dataset,
                                      indices=indices_test,
                                      batch_size=32)
    
    # Create network
    network = GreatNet(n_assets, lookback)
    print(network)
    
    # Set network to train mode
    network = network.train()
    
    # Define loss function
    loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
    
    # Create run
    run = Run(network,
              loss,
              dataloader_train,
              val_dataloaders={'test': dataloader_test},
              optimizer=torch.optim.Adam(network.parameters(), amsgrad=True),
              callbacks=[EarlyStoppingCallback(metric_name='loss',
                                               dataloader_name='test',
                                               patience=15)])
                                               
    # Launch training
    history = run.launch(30)
    
    print("Training completed!")


if __name__ == "__main__":
    main()