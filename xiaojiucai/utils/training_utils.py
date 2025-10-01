"""
训练工具模块

包含模型训练、评估、保存等工具函数
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class TrainingConfig:
    """训练配置类"""
    
    def __init__(self):
        # 数据处理参数
        self.window_size = 200
        self.batch_size = 128
        
        # 训练超参数
        self.finetune_timesfm = True
        self.risk_aversion = 1.0
        self.sharpe_weight = 0.5
        
        # 训练控制参数
        self.num_epochs = 20
        self.patience = 20
        self.grad_clip_value = 1.0
        
        # 学习率参数
        self.timesfm_lr = 1e-5
        self.portfolio_lr = 1e-3
        
        # 模型保存参数
        self.checkpoint_interval = 5
        self.max_checkpoints = 5
        self.max_best_models = 3
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
    
    def print_config(self):
        """打印配置信息"""
        print("🎯 训练配置总览:")
        print(f"  数据参数: window_size={self.window_size}, batch_size={self.batch_size}")
        print(f"  训练参数: epochs={self.num_epochs}, patience={self.patience}")
        print(f"  优化参数: TimesFM_lr={self.timesfm_lr}, portfolio_lr={self.portfolio_lr}")
        print(f"  损失参数: risk_aversion={self.risk_aversion}, sharpe_weight={self.sharpe_weight}")
        print(f"  控制参数: finetune_timesfm={self.finetune_timesfm}, grad_clip={self.grad_clip_value}")


def setup_optimizer_and_scheduler(model, config: TrainingConfig):
    """
    设置优化器和学习率调度器
    
    Args:
        model: 需要训练的模型
        config: 训练配置
        
    Returns:
        optimizer: 优化器
        scheduler: 学习率调度器
    """
    # 分离参数组
    timesfm_params = []
    portfolio_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'timesfm' in name:
                timesfm_params.append(param)
            else:
                portfolio_params.append(param)
    
    # 创建分层学习率优化器
    optimizer = torch.optim.Adam([
        {'params': timesfm_params, 'lr': config.timesfm_lr, 'name': 'timesfm'},
        {'params': portfolio_params, 'lr': config.portfolio_lr, 'name': 'portfolio'}
    ])
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"🎯 优化器配置:")
    print(f"  TimesFM参数: {sum(p.numel() for p in timesfm_params):,} (lr={config.timesfm_lr})")
    print(f"  投资组合层参数: {sum(p.numel() for p in portfolio_params):,} (lr={config.portfolio_lr})")
    
    return optimizer, scheduler


def validate_gradient_flow(model, test_input, test_returns, loss_fn):
    """
    验证梯度传播
    
    Args:
        model: 模型
        test_input: 测试输入
        test_returns: 测试回报率
        loss_fn: 损失函数
        
    Returns:
        bool: 梯度是否正常传播
    """
    model.train()
    
    # 前向传播
    weights = model(test_input)
    loss = loss_fn(weights, test_returns)
    loss.backward()
    
    # 统计梯度传播
    timesfm_grad_count = 0
    timesfm_total_count = 0
    
    for name, param in model.named_parameters():
        if 'timesfm' in name and param.requires_grad:
            timesfm_total_count += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                timesfm_grad_count += 1
    
    gradient_ok = timesfm_grad_count > 0
    print(f"🔍 梯度传播验证: {'✓ 通过' if gradient_ok else '✗ 失败'} ({timesfm_grad_count}/{timesfm_total_count})")
    
    return gradient_ok


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, config: TrainingConfig, loss_fn, device='cpu', model_save_dir='models'):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.device = device
        self.model_save_dir = model_save_dir
        
        # 确保模型保存目录存在
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 生成时间戳
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置优化器和调度器
        self.optimizer, self.scheduler = setup_optimizer_and_scheduler(model, config)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_model_path = None  # 添加最佳模型路径属性
        self.patience_counter = 0
        
        # 模型保存策略
        self.saved_checkpoints = []
        self.best_models_history = []
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            weights = self.model(batch_X)
            loss = self.loss_fn(weights, batch_y)
            loss.backward()
            
            # 梯度裁剪
            if self.config.finetune_timesfm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_value)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        return epoch_loss / batch_count
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        epoch_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                weights = self.model(batch_X)
                loss = self.loss_fn(weights, batch_y)
                
                epoch_loss += loss.item()
                batch_count += 1
        
        return epoch_loss / batch_count
    
    def save_checkpoint(self, epoch, train_loss, val_loss):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.model_save_dir, 
            f'timesfm_portfolio_checkpoint_epoch_{epoch+1:02d}_{self.timestamp}.pth'
        )
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config.__dict__,
            'timestamp': self.timestamp
        }, checkpoint_path)
        
        self.saved_checkpoints.append(checkpoint_path)
        
        # 保持checkpoint数量不超过最大值
        if len(self.saved_checkpoints) > self.config.max_checkpoints:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
        
        return checkpoint_path
    
    def save_best_model(self, epoch, val_loss):
        """保存最佳模型"""
        best_model_path = os.path.join(
            self.model_save_dir,
            f'timesfm_portfolio_model_best_{self.timestamp}.pth'
        )
        
        # 保存到最佳模型历史
        best_model_info = {
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'model_state': self.model.state_dict().copy(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'timestamp': self.timestamp
        }
        self.best_models_history.append(best_model_info)
        
        # 保持最佳模型历史数量不超过最大值
        if len(self.best_models_history) > self.config.max_best_models:
            self.best_models_history.pop(0)
        
        self.best_val_loss = val_loss
        self.best_model_state = self.model.state_dict().copy()
        self.best_model_path = best_model_path  # 设置最佳模型路径
        
        # 保存全局最佳模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'timestamp': self.timestamp
        }, best_model_path)
        
        return best_model_path
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print(f"🚀 开始训练 (设备: {self.device})")
        print(f"  训练集: {len(train_loader.dataset)} 样本")
        print(f"  验证集: {len(val_loader.dataset)} 样本")
        print("=" * 80)
        
        best_model_path = None
        
        for epoch in range(self.config.num_epochs):
            # 训练和验证
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 更新学习率调度器
            self.scheduler.step(val_loss)
            
            # 获取当前学习率
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            lr_info = f"LR: {current_lrs}" if len(current_lrs) > 1 else f"LR: {current_lrs[0]:.2e}"
            
            # 模型保存逻辑
            save_info = []
            
            # 定期保存checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = self.save_checkpoint(epoch, train_loss, val_loss)
                save_info.append(f"💾 Checkpoint-{epoch+1}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                best_model_path = self.save_best_model(epoch, val_loss)
                self.patience_counter = 0
                save_info.append(f"⭐ 新最佳模型 ({self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= self.config.patience:
                save_info.append("⏹️ 早停")
                print(f'\nEarly stopping triggered after {self.config.patience} epochs without improvement')
                break
            
            # 显示训练进度
            save_status = " | ".join(save_info) if save_info else "无保存"
            status = f"继续训练" if self.patience_counter < self.config.patience else "早停触发"
            
            print(f'Epoch {epoch+1:2d}/{self.config.num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | '
                  f'Best: {self.best_val_loss:.4f} | {lr_info} | {status} | {save_status}')
        
        # 保存最终模型和训练总结
        final_model_path = self._save_final_model()
        self._save_training_summary(final_model_path, best_model_path)
        
        return {
            'best_model_path': best_model_path,
            'final_model_path': final_model_path,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def _save_final_model(self):
        """保存最终模型"""
        final_model_path = os.path.join(
            self.model_save_dir,
            f'timesfm_portfolio_model_final_{self.timestamp}.pth'
        )
        
        torch.save({
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'timestamp': self.timestamp,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'total_epochs': len(self.train_losses)
            }
        }, final_model_path)
        
        return final_model_path
    
    def _save_training_summary(self, final_model_path, best_model_path):
        """保存训练总结"""
        training_summary = {
            'timestamp': self.timestamp,
            'configuration': self.config.__dict__,
            'training_results': {
                'final_train_loss': self.train_losses[-1],
                'final_val_loss': self.val_losses[-1], 
                'best_val_loss': self.best_val_loss,
                'total_epochs': len(self.train_losses),
                'early_stopped': self.patience_counter >= self.config.patience
            },
            'saved_models': {
                'best_model': os.path.basename(best_model_path) if best_model_path else None,
                'final_model': os.path.basename(final_model_path),
                'checkpoints': [os.path.basename(cp) for cp in self.saved_checkpoints],
                'best_models_count': len(self.best_models_history)
            }
        }
        
        summary_path = os.path.join(self.model_save_dir, f'training_summary_{self.timestamp}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 训练总结已保存: {summary_path}")
        print(f"💾 最佳模型: {os.path.basename(best_model_path) if best_model_path else '无'}")
        print(f"📋 最终模型: {os.path.basename(final_model_path)}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点并恢复训练状态
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            checkpoint: 检查点信息字典
        """
        print(f"📥 加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 恢复模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器状态
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复学习率调度器状态
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练历史
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            self.train_losses = history.get('train_losses', [])
            self.val_losses = history.get('val_losses', [])
        
        # 恢复最佳验证损失
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # 更新时间戳（保持原有时间戳）
        if 'timestamp' in checkpoint:
            self.timestamp = checkpoint['timestamp']
        
        print(f"✅ 检查点加载成功:")
        print(f"  - 训练轮数: {checkpoint.get('epoch', '未知')}")
        print(f"  - 最佳验证损失: {self.best_val_loss:.4f}")
        print(f"  - 训练历史长度: {len(self.train_losses)} 轮")
        
        return checkpoint
    
    def resume_training(self, checkpoint_path, train_loader, val_loader):
        """
        从检查点恢复训练
        
        Args:
            checkpoint_path: 检查点文件路径
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练结果字典
        """
        # 加载检查点
        checkpoint = self.load_checkpoint(checkpoint_path)
        
        # 计算已完成的轮数
        completed_epochs = len(self.train_losses)
        remaining_epochs = self.config.num_epochs - completed_epochs
        
        print(f"🔄 恢复训练:")
        print(f"  - 已完成轮数: {completed_epochs}")
        print(f"  - 剩余轮数: {remaining_epochs}")
        
        if remaining_epochs <= 0:
            print("⚠️ 训练已完成，无需继续训练")
            return {
                'best_model_path': None,
                'final_model_path': None,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            }
        
        # 调整训练配置
        original_epochs = self.config.num_epochs
        self.config.num_epochs = remaining_epochs
        
        # 继续训练
        print(f"▶️ 继续训练...")
        results = self.train(train_loader, val_loader)
        
        # 恢复原始配置
        self.config.num_epochs = original_epochs
        
        return results


def load_model_checkpoint(checkpoint_path, model_class, model_args, device='cpu'):
    """
    加载模型检查点（通用版本）
    
    Args:
        checkpoint_path: 检查点文件路径
        model_class: 模型类
        model_args: 模型初始化参数
        device: 设备
        
    Returns:
        model: 加载的模型
        checkpoint: 检查点信息
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 重新创建模型
    model = model_class(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型已从 {checkpoint_path} 加载")
    print(f"检查点信息:")
    print(f"  - 训练轮数: {checkpoint.get('epoch', '未知')}")
    print(f"  - 验证损失: {checkpoint.get('val_loss', '未知')}")
    print(f"  - 时间戳: {checkpoint.get('timestamp', '未知')}")
    
    return model, checkpoint


def load_timesfm_portfolio_model(checkpoint_path, device='cpu'):
    """
    专用于加载TimesFM投资组合模型的函数
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 设备
        
    Returns:
        model: 加载的模型
        checkpoint: 检查点信息
        config: 训练配置
    """
    # 使用绝对导入
    from xiaojiucai.models import create_timesfm_model, TimesFMPortfolioModelV2
    
    print(f"📥 加载TimesFM投资组合模型: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型配置
    config_dict = checkpoint.get('config', {})
    
    # 重新创建TimesFM模型和包装器
    timesfm_model, timesfm_wrapper = create_timesfm_model()
    
    # 获取模型参数
    input_size = config_dict.get('input_size', checkpoint.get('input_size', 7))
    feature_size = config_dict.get('feature_size', checkpoint.get('feature_size', 1))
    window_size = config_dict.get('window_size', checkpoint.get('window_size', 200))
    finetune_timesfm = config_dict.get('finetune_timesfm', checkpoint.get('finetune_timesfm', True))
    
    # 重新创建投资组合模型
    portfolio_model = TimesFMPortfolioModelV2(
        input_size=input_size,
        output_size=input_size,
        feature_size=feature_size,
        timesfm_wrapper=timesfm_wrapper,
        context_len=window_size,
        finetune_timesfm=finetune_timesfm
    ).to(device)
    
    # 加载模型状态
    portfolio_model.load_state_dict(checkpoint['model_state_dict'])
    portfolio_model.eval()
    
    # 重构配置对象
    config = TrainingConfig()
    config.update(**config_dict)
    
    print(f"✅ 模型加载成功:")
    print(f"  - 模型类型: TimesFM投资组合模型V2")
    print(f"  - 训练轮数: {checkpoint.get('epoch', '未知')}")
    print(f"  - 验证损失: {checkpoint.get('val_loss', checkpoint.get('best_val_loss', '未知'))}")
    print(f"  - 股票数量: {input_size}")
    print(f"  - 特征数量: {feature_size}")
    print(f"  - 时间窗口: {window_size}")
    print(f"  - TimesFM微调: {finetune_timesfm}")
    print(f"  - 时间戳: {checkpoint.get('timestamp', '未知')}")
    
    return portfolio_model, checkpoint, config


class ModelLoader:
    """模型加载器类"""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点文件"""
        return torch.load(checkpoint_path, map_location=self.device, weights_only=False)
    
    def load_timesfm_portfolio_model(self, checkpoint_path):
        """加载TimesFM投资组合模型"""
        return load_timesfm_portfolio_model(checkpoint_path, self.device)
    
    def list_model_files(self, model_dir='models'):
        """列出模型目录中的所有模型文件"""
        import glob
        
        if not os.path.exists(model_dir):
            print(f"❌ 模型目录不存在: {model_dir}")
            return []
        
        model_files = glob.glob(os.path.join(model_dir, '*.pth'))
        
        if not model_files:
            print(f"📁 模型目录中没有找到模型文件: {model_dir}")
            return []
        
        # 按修改时间排序
        model_files.sort(key=os.path.getmtime, reverse=True)
        
        print(f"📁 找到 {len(model_files)} 个模型文件:")
        for i, model_file in enumerate(model_files):
            file_name = os.path.basename(model_file)
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            mod_time = os.path.getmtime(model_file)
            mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            
            model_type = "未知"
            if "best" in file_name:
                model_type = "最佳模型"
            elif "final" in file_name:
                model_type = "最终模型"
            elif "checkpoint" in file_name:
                model_type = "检查点"
                
            print(f"  {i+1:2d}. {file_name}")
            print(f"      类型: {model_type} | 大小: {file_size:.1f}MB | 时间: {mod_time_str}")
        
        return model_files
    
    def get_model_info(self, checkpoint_path):
        """获取模型信息（不加载模型）"""
        try:
            checkpoint = self.load_checkpoint(checkpoint_path)
            
            info = {
                'file_path': checkpoint_path,
                'file_name': os.path.basename(checkpoint_path),
                'epoch': checkpoint.get('epoch', '未知'),
                'val_loss': checkpoint.get('val_loss', checkpoint.get('best_val_loss', '未知')),
                'train_loss': checkpoint.get('train_loss', checkpoint.get('final_train_loss', '未知')),
                'timestamp': checkpoint.get('timestamp', '未知'),
                'config': checkpoint.get('config', {}),
                'model_params': {
                    'input_size': checkpoint.get('input_size', checkpoint.get('config', {}).get('input_size', '未知')),
                    'feature_size': checkpoint.get('feature_size', checkpoint.get('config', {}).get('feature_size', '未知')),
                    'window_size': checkpoint.get('window_size', checkpoint.get('config', {}).get('window_size', '未知')),
                    'finetune_timesfm': checkpoint.get('finetune_timesfm', checkpoint.get('config', {}).get('finetune_timesfm', '未知'))
                }
            }
            
            return info
            
        except Exception as e:
            print(f"❌ 获取模型信息失败: {e}")
            return None
    
    def compare_models(self, model_paths):
        """比较多个模型的信息"""
        print("🔍 模型对比:")
        print("-" * 100)
        print(f"{'文件名':<40} {'轮数':<6} {'验证损失':<12} {'训练损失':<12} {'时间戳':<20}")
        print("-" * 100)
        
        for path in model_paths:
            info = self.get_model_info(path)
            if info:
                file_name = info['file_name'][:37] + "..." if len(info['file_name']) > 40 else info['file_name']
                val_loss = f"{info['val_loss']:.6f}" if isinstance(info['val_loss'], (int, float)) else str(info['val_loss'])
                train_loss = f"{info['train_loss']:.6f}" if isinstance(info['train_loss'], (int, float)) else str(info['train_loss'])
                
                print(f"{file_name:<40} {info['epoch']:<6} {val_loss:<12} {train_loss:<12} {info['timestamp']:<20}")
        
        print("-" * 100)