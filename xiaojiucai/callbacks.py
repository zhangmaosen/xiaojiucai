"""
Callback classes for training monitoring and control.
"""

import numpy as np
import pandas as pd


class Callback:
    """Base callback class."""
    
    def on_epoch_end(self, epoch, history):
        """Called at the end of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number.
            
        history : History
            Training history.
            
        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        return False


class EarlyStoppingCallback(Callback):
    """Early stopping callback.
    
    Stops training when a monitored metric has stopped improving.
    
    Parameters
    ----------
    metric_name : str
        Name of the metric to monitor.
        
    dataloader_name : str
        Name of the dataloader to monitor.
        
    patience : int, default 10
        Number of epochs with no improvement after which training will be stopped.
        
    min_delta : float, default 0
        Minimum change in the monitored quantity to qualify as an improvement.
    """
    
    def __init__(self, metric_name, dataloader_name, patience=10, min_delta=0):
        super().__init__()
        self.metric_name = metric_name
        self.dataloader_name = dataloader_name
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, history):
        """Check if training should be stopped.
        
        Parameters
        ----------
        epoch : int
            Current epoch number.
            
        history : History
            Training history.
            
        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        # Get the latest value of the monitored metric
        mask = (
            (history.metrics['dataloader'] == self.dataloader_name) &
            (history.metrics['metric'] == self.metric_name) &
            (history.metrics['epoch'] == epoch)
        )
        
        if not mask.any():
            return False  # Continue training
            
        current_value = history.metrics[mask]['value'].iloc[0]
        
        # Print metrics for this epoch
        self._print_epoch_metrics(epoch, history)
        
        if self.best_value is None:
            self.best_value = current_value
        elif current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                print(f"Early stopping at epoch {epoch}")
                return True  # Stop training
                
        return False  # Continue training
        
    def _print_epoch_metrics(self, epoch, history):
        """Print metrics for the current epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number.
            
        history : History
            Training history.
        """
        # Get all metrics for this epoch
        mask = history.metrics['epoch'] == epoch
        epoch_metrics = history.metrics[mask]
        
        if not epoch_metrics.empty:
            print(f"Epoch {epoch+1} metrics:")
            for _, row in epoch_metrics.iterrows():
                print(f"  {row['dataloader']} {row['metric']}: {row['value']:.6f}")