# 在__init__.py中直接定义类，避免循环导入问题
import torch
import pandas as pd


class Run:
    """Experiment runner for portfolio optimization models.
    
    This class handles the training loop, validation, and metrics tracking.
    
    Parameters
    ----------
    network : torch.nn.Module
        The network to train.
        
    loss : callable
        The loss function.
        
    train_dataloader : DataLoader
        Dataloader for training data.
        
    val_dataloaders : dict of DataLoader, optional
        Dictionary of dataloaders for validation.
        
    optimizer : torch.optim.Optimizer, optional
        Optimizer for training. If None, Adam will be used.
        
    callbacks : list of callable, optional
        List of callbacks to execute during training.
    """
    
    def __init__(self, network, loss, train_dataloader, val_dataloaders=None, 
                 optimizer=None, callbacks=None):
        self.network = network
        self.loss = loss
        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders or {}
        self.optimizer = optimizer or torch.optim.Adam(network.parameters())
        self.callbacks = callbacks or []
        
        # History tracking
        self.history = History()
        
    def launch(self, n_epochs):
        """Launch the training experiment.
        
        Parameters
        ----------
        n_epochs : int
            Number of epochs to train.
            
        Returns
        -------
        history : History
            Training history.
        """
        for epoch in range(n_epochs):
            # Training phase
            train_loss = self._train_epoch()
            self.history.add_metric('train', 'loss', 'network', epoch, train_loss)
            
            # Validation phase
            for name, dataloader in self.val_dataloaders.items():
                val_loss = self._val_epoch(dataloader)
                self.history.add_metric(name, 'loss', 'network', epoch, val_loss)
                
            # Callbacks
            should_stop = False
            for callback in self.callbacks:
                if callback.on_epoch_end(epoch, self.history):
                    should_stop = True
                    
            if should_stop:
                break  # Stop training
                
        return self.history
        
    def _train_epoch(self):
        """Run one training epoch.
        
        Returns
        -------
        avg_loss : float
            Average training loss for the epoch.
        """
        self.network.train()
        total_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in self.train_dataloader:
            # Move to device and ensure correct dtype
            device = next(self.network.parameters()).device
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.float32)
            
            # Forward pass
            weights = self.network(batch_x)
            loss = self.loss(weights, batch_y)
            
            # Check for NaN in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN or Inf loss detected. Loss value: {loss}")
                print(f"Weights range: [{weights.min().item():.6f}, {weights.max().item():.6f}]")
                print(f"Batch x range: [{batch_x.min().item():.6f}, {batch_x.max().item():.6f}]")
                print(f"Batch y range: [{batch_y.min().item():.6f}, {batch_y.max().item():.6f}]")
                # Skip this batch
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        # Handle case where all batches were skipped
        if n_batches == 0:
            return float('nan')
            
        return total_loss / n_batches
        
    def _val_epoch(self, dataloader):
        """Run one validation epoch.
        
        Parameters
        ----------
        dataloader : DataLoader
            Dataloader for validation data.
            
        Returns
        -------
        avg_loss : float
            Average validation loss for the epoch.
        """
        self.network.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                # Move to device and ensure correct dtype
                device = next(self.network.parameters()).device
                batch_x = batch_x.to(device=device, dtype=torch.float32)
                batch_y = batch_y.to(device=device, dtype=torch.float32)
                
                # Forward pass
                weights = self.network(batch_x)
                loss = self.loss(weights, batch_y)
                
                # Check for NaN in loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN or Inf validation loss detected. Loss value: {loss}")
                    print(f"Validation weights range: [{weights.min().item():.6f}, {weights.max().item():.6f}]")
                    print(f"Validation batch x range: [{batch_x.min().item():.6f}, {batch_x.max().item():.6f}]")
                    print(f"Validation batch y range: [{batch_y.min().item():.6f}, {batch_y.max().item():.6f}]")
                    # Skip this batch
                    continue
                
                total_loss += loss.item()
                n_batches += 1
                
        # Handle case where all batches were skipped
        if n_batches == 0:
            return float('nan')
                
        return total_loss / n_batches


class History:
    """Training history tracker.
    
    This class tracks metrics during training.
    """
    
    def __init__(self):
        self.metrics = pd.DataFrame(columns=['dataloader', 'metric', 'model', 'epoch', 'value'])
        
    def add_metric(self, dataloader, metric, model, epoch, value):
        """Add a metric to the history.
        
        Parameters
        ----------
        dataloader : str
            Name of the dataloader.
            
        metric : str
            Name of the metric.
            
        model : str
            Name of the model.
            
        epoch : int
            Epoch number.
            
        value : float
            Metric value.
        """
        new_row = pd.DataFrame({
            'dataloader': [dataloader],
            'metric': [metric],
            'model': [model],
            'epoch': [epoch],
            'value': [value]
        })
        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True)

__all__ = ['Run']