"""
Visualization utilities for portfolio optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_metrics_table(benchmarks, dataloader, metrics):
    """Generate a table of metrics for different benchmarks.
    
    Parameters
    ----------
    benchmarks : dict
        Dictionary of benchmark models.
        
    dataloader : DataLoader
        Dataloader for evaluation data.
        
    metrics : dict
        Dictionary of metric functions.
        
    Returns
    -------
    metrics_table : pd.DataFrame
        DataFrame with metrics for each benchmark.
    """
    results = []
    
    for name, model in benchmarks.items():
        for batch_x, batch_y in dataloader:
            # Compute weights
            if hasattr(model, 'eval'):
                model.eval()
                with torch.no_grad():
                    weights = model(batch_x)
            else:
                weights = model(batch_x)
                
            # Compute metrics
            for metric_name, metric_fn in metrics.items():
                metric_value = metric_fn(weights, batch_y).item()
                results.append({
                    'model': name,
                    'metric': metric_name,
                    'value': metric_value
                })
                
    return pd.DataFrame(results)


def plot_metrics(metrics_table):
    """Plot metrics for different benchmarks.
    
    Parameters
    ----------
    metrics_table : pd.DataFrame
        DataFrame with metrics for each benchmark.
    """
    metrics = metrics_table['metric'].unique()
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
    if n_metrics == 1:
        axes = [axes]
        
    for i, metric in enumerate(metrics):
        data = metrics_table[metrics_table['metric'] == metric]
        models = data['model'].unique()
        
        values = [data[data['model'] == model]['value'].mean() for model in models]
        
        axes[i].bar(models, values)
        axes[i].set_title(metric)
        axes[i].set_ylabel('Value')
        
    plt.tight_layout()
    plt.show()


def generate_weights_table(model, dataloader, asset_names=None):
    """Generate a table of portfolio weights.
    
    Parameters
    ----------
    model : callable
        Model to generate weights.
        
    dataloader : DataLoader
        Dataloader for evaluation data.
        
    asset_names : list of str, optional
        Names of assets.
        
    Returns
    -------
    weights_table : pd.DataFrame
        DataFrame with portfolio weights.
    """
    all_weights = []
    
    for batch_x, _ in dataloader:
        # Compute weights
        if hasattr(model, 'eval'):
            model.eval()
            with torch.no_grad():
                weights = model(batch_x)
        else:
            weights = model(batch_x)
            
        all_weights.append(weights)
        
    # Concatenate all weights
    weights = torch.cat(all_weights, dim=0)
    
    # Convert to numpy
    weights_np = weights.numpy()
    
    # Create DataFrame
    n_samples, n_assets = weights_np.shape
    
    if asset_names is None:
        asset_names = [f'Asset_{i}' for i in range(n_assets)]
        
    weights_table = pd.DataFrame(weights_np, columns=asset_names)
    
    return weights_table


def plot_weight_heatmap(weights_table, add_sum_column=True, time_format=None, time_skips=None):
    """Plot heatmap of portfolio weights.
    
    Parameters
    ----------
    weights_table : pd.DataFrame
        DataFrame with portfolio weights.
        
    add_sum_column : bool, default True
        Whether to add a column showing the sum of weights.
        
    time_format : str, optional
        Format for time index.
        
    time_skips : int, optional
        Number of time steps to skip in labeling.
    """
    data = weights_table.copy()
    
    if add_sum_column:
        data['Sum'] = data.sum(axis=1)
        
    plt.figure(figsize=(12, 8))
    plt.imshow(data.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight')
    
    # Set ticks
    plt.yticks(range(len(data.columns)), data.columns)
    
    if time_skips is not None:
        time_indices = range(0, len(data), time_skips)
        plt.xticks(time_indices, time_indices)
        
    plt.xlabel('Time')
    plt.ylabel('Assets')
    plt.title('Portfolio Weights Over Time')
    plt.tight_layout()
    plt.show()