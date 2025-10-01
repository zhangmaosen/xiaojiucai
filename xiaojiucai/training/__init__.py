"""
TimesFM投资组合训练模块

包含模型训练、验证、保存和实验管理功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from tqdm import tqdm


class PortfolioTrainer:
    """投资组合模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        model_save_dir: str = "models/experiments"
    ):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.model_save_dir = model_save_dir
        
        # 创建保存目录
        os.makedirs(model_save_dir, exist_ok=True)
        
    def train_epoch(
        self, 
        train_loader, 
        optimizer, 
        gradient_clip: float = 1.0
    ) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            features = batch[0].to(self.device)
            future_returns = batch[1].squeeze(2).to(self.device)
            future_returns = future_returns[:, -1, :]

            optimizer.zero_grad()
            outputs = self.model(features)
            loss_dict = self.criterion(outputs, future_returns)
            loss = loss_dict['total_loss']
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            optimizer.step()

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'avg_loss': f'{total_loss/total_samples:.4f}'
            })
            
        return total_loss / total_samples

    def validate_epoch(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_weights = []
        all_returns = []
        all_risks = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating", leave=False)
            for batch in progress_bar:
                features = batch[0].to(self.device)
                future_returns = batch[1].squeeze(2).to(self.device)
                future_returns = future_returns[:, -1, :]

                outputs = self.model(features)
                loss_dict = self.criterion(outputs, future_returns)
                loss = loss_dict['total_loss']

                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                all_weights.append(outputs['weights'].cpu())
                all_returns.append(outputs['returns'].cpu())
                all_risks.append(outputs['risks'].cpu())

                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        # 计算权重统计
        weights = torch.cat(all_weights, dim=0)
        returns = torch.cat(all_returns, dim=0)
        risks = torch.cat(all_risks, dim=0)

        weight_stats = {
            'mean_abs_weight': weights.abs().mean().item(),
            'max_weight': weights.max().item(),
            'min_weight': weights.min().item(),
            'weight_concentration': (weights**2).sum(dim=1).mean().item(),
        }

        return total_loss / total_samples, weight_stats

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        epoch: int,
        loss: float,
        timestamp: str,
        is_best: bool = False,
        experiment_name: str = None
    ) -> str:
        """保存模型检查点"""
        
        # 创建实验目录
        if experiment_name:
            experiment_dir = os.path.join(self.model_save_dir, f"{experiment_name}_{timestamp}")
        else:
            experiment_dir = os.path.join(self.model_save_dir, f"experiment_{timestamp}")
        
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 保存实验元数据 (首次保存时)
        experiment_metadata_path = os.path.join(experiment_dir, 'experiment_metadata.json')
        if not os.path.exists(experiment_metadata_path):
            experiment_metadata = {
                'experiment_name': experiment_name or 'default_experiment',
                'timestamp': timestamp,
                'experiment_dir': experiment_dir,
                'created_at': datetime.now().isoformat(),
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
            with open(experiment_metadata_path, 'w') as f:
                json.dump(experiment_metadata, f, indent=2)
        
        # 构建检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'timestamp': timestamp,
            'experiment_dir': experiment_dir,
            'is_best': is_best,
            'saved_at': datetime.now().isoformat(),
        }
        
        # 文件名
        if is_best:
            filename = f'timesfm_portfolio_model_best_{timestamp}.pth'
        else:
            filename = f'timesfm_portfolio_checkpoint_epoch_{epoch:02d}_{timestamp}.pth'
        
        checkpoint_path = os.path.join(experiment_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path

    def train(
        self,
        train_loader,
        val_loader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        config: Dict[str, Any],
        experiment_name: str = None
    ) -> Tuple[Dict[str, List], float]:
        """训练模型"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("🚀 开始训练TimesFM投资组合优化模型...")
        print(f"📊 训练集: {len(train_loader)} 批次 | 验证集: {len(val_loader)} 批次")
        
        # 创建实验目录
        if experiment_name:
            experiment_dir = os.path.join(self.model_save_dir, f"{experiment_name}_{timestamp}")
        else:
            experiment_dir = os.path.join(self.model_save_dir, f"experiment_{timestamp}")
        
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"📁 实验目录: {experiment_dir}")
        
        # 保存配置
        config_path = os.path.join(experiment_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"⚙️ 训练配置已保存: {os.path.basename(config_path)}")

        # 训练历史
        history = {
            'train_loss': [], 
            'val_loss': [], 
            'weight_stats': [], 
            'learning_rates': [], 
            'epochs': []
        }
        best_val_loss = float('inf')
        patience_counter = 0

        # 训练循环
        for epoch in range(config['epochs']):
            print(f"\n📈 Epoch [{epoch+1:3d}/{config['epochs']:3d}]")

            # 训练
            train_loss = self.train_epoch(
                train_loader, 
                optimizer, 
                gradient_clip=config.get('gradient_clip', 1.0)
            )

            # 验证
            if (epoch + 1) % config.get('validate_every', 1) == 0:
                val_loss, weight_stats = self.validate_epoch(val_loader)
                history['val_loss'].append(val_loss)
                history['weight_stats'].append(weight_stats)

                print(f"📉 验证损失: {val_loss:.6f}")
                print(f"⚖️  权重统计: 平均|权重|={weight_stats['mean_abs_weight']:.4f}, 集中度={weight_stats['weight_concentration']:.4f}")

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    checkpoint_path = self.save_checkpoint(
                        self.model, optimizer, scheduler, epoch+1, val_loss, 
                        timestamp, is_best=True, experiment_name=experiment_name
                    )
                    print(f"💾 最佳模型已保存: {os.path.basename(checkpoint_path)}")
                else:
                    patience_counter += 1
                    print(f"⏳ 验证损失未改善 ({patience_counter}/{config.get('patience', 10)})")

            # 记录历史
            history['train_loss'].append(train_loss)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            history['epochs'].append(epoch + 1)

            print(f"🔥 训练损失: {train_loss:.6f} | 学习率: {optimizer.param_groups[0]['lr']:.2e}")
            scheduler.step()

            # 定期保存检查点
            if (epoch + 1) % config.get('save_every', 5) == 0:
                checkpoint_path = self.save_checkpoint(
                    self.model, optimizer, scheduler, epoch+1, train_loss, 
                    timestamp, is_best=False, experiment_name=experiment_name
                )
                print(f"💾 检查点已保存: {os.path.basename(checkpoint_path)}")

            # 早停
            if patience_counter >= config.get('patience', 10):
                print(f"\n⏹️  早停触发！验证损失{config.get('patience', 10)}个epoch未改善")
                break

        # 保存最终模型
        final_checkpoint = self.save_checkpoint(
            self.model, optimizer, scheduler, epoch+1, train_loss, 
            timestamp, is_best=False, experiment_name=experiment_name
        )
        final_path = final_checkpoint.replace('checkpoint_epoch', 'model_final')
        os.rename(final_checkpoint, final_path)

        # 保存训练历史
        history_path = os.path.join(experiment_dir, f'training_summary_{timestamp}.json')
        with open(history_path, 'w') as f:
            # 转换numpy类型为Python类型
            json_history = {}
            for key, value in history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        json_history[key] = value
                    else:
                        json_history[key] = [
                            float(v) if isinstance(v, (np.float32, np.float64)) else v 
                            for v in value
                        ]
                else:
                    json_history[key] = value
            json.dump(json_history, f, indent=2)

        print(f"\n🎉 训练完成！")
        print(f"📈 最佳验证损失: {best_val_loss:.6f}")
        print(f"📁 实验目录: {experiment_dir}")
        print(f"📦 最终模型: {os.path.basename(final_path)}")
        print(f"📊 训练历史: {os.path.basename(history_path)}")
        
        return history, best_val_loss


class ExperimentManager:
    """实验管理工具"""
    
    def __init__(self, model_save_dir: str = "models/experiments"):
        self.model_save_dir = model_save_dir
        
    def list_experiments(self) -> List[Dict[str, Any]]:
        """列出所有实验"""
        if not os.path.exists(self.model_save_dir):
            print(f"❌ 模型保存目录不存在: {self.model_save_dir}")
            return []
        
        experiments = []
        for item in os.listdir(self.model_save_dir):
            item_path = os.path.join(self.model_save_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'experiment_metadata.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        experiments.append({
                            'dir_name': item,
                            'path': item_path,
                            'metadata': metadata
                        })
                    except:
                        print(f"⚠️ 无法读取实验元数据: {metadata_path}")
        
        if experiments:
            print(f"📁 找到 {len(experiments)} 个实验:")
            print("-" * 80)
            for exp in experiments:
                meta = exp['metadata']
                print(f"🧪 {meta.get('experiment_name', 'Unknown')}")
                print(f"   📅 创建时间: {meta.get('created_at', 'Unknown')}")
                print(f"   📁 目录: {exp['dir_name']}")
                print(f"   🏗️ 参数: {meta.get('total_parameters', 0):,}")
                print()
        else:
            print("📭 未找到任何实验")
        
        return experiments

    def load_checkpoint(
        self, 
        experiment_dir: str, 
        checkpoint_type: Union[str, int] = 'best'
    ) -> Tuple[Dict[str, Any], str]:
        """加载实验检查点"""
        if not os.path.exists(experiment_dir):
            raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")
        
        # 查找检查点文件
        checkpoint_files = []
        for file in os.listdir(experiment_dir):
            if file.endswith('.pth'):
                if checkpoint_type == 'best' and 'model_best' in file:
                    checkpoint_files.append(file)
                elif checkpoint_type == 'final' and 'model_final' in file:
                    checkpoint_files.append(file)
                elif isinstance(checkpoint_type, int) and f'epoch_{checkpoint_type:02d}' in file:
                    checkpoint_files.append(file)
        
        if not checkpoint_files:
            available_files = [f for f in os.listdir(experiment_dir) if f.endswith('.pth')]
            raise FileNotFoundError(f"未找到 {checkpoint_type} 类型的检查点。可用文件: {available_files}")
        
        # 加载最新匹配文件
        checkpoint_file = sorted(checkpoint_files)[-1]
        checkpoint_path = os.path.join(experiment_dir, checkpoint_file)
        
        print(f"📦 加载检查点: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        return checkpoint, checkpoint_path

    def get_experiment_summary(self, experiment_dir: str) -> Dict[str, Any]:
        """获取实验摘要"""
        if not os.path.exists(experiment_dir):
            raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")
        
        summary = {}
        
        # 读取实验元数据
        metadata_path = os.path.join(experiment_dir, 'experiment_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                summary['metadata'] = json.load(f)
        
        # 读取训练配置
        config_path = os.path.join(experiment_dir, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                summary['config'] = json.load(f)
        
        # 读取训练历史
        history_files = [f for f in os.listdir(experiment_dir) if f.startswith('training_summary_')]
        if history_files:
            history_path = os.path.join(experiment_dir, sorted(history_files)[-1])
            with open(history_path, 'r') as f:
                summary['history'] = json.load(f)
        
        # 列出可用检查点
        checkpoints = [f for f in os.listdir(experiment_dir) if f.endswith('.pth')]
        summary['checkpoints'] = sorted(checkpoints)
        
        return summary


def create_trainer_from_config(
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any]
) -> Tuple[PortfolioTrainer, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """从配置创建训练器和优化器"""
    
    # 创建训练器
    trainer = PortfolioTrainer(
        model=model,
        criterion=criterion,
        device=device,
        model_save_dir=config.get('model_save_dir', 'models/experiments')
    )
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 创建学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.get('epochs', 100),
        eta_min=config.get('min_lr', 1e-6)
    )
    
    return trainer, optimizer, scheduler