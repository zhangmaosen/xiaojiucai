"""
投资组合评估模块

包含模型评估、性能分析和回测功能
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error


class PortfolioEvaluator:
    """投资组合评估器"""
    
    def __init__(self, asset_names: List[str]):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        
    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_loader,
        device: torch.device,
        criterion: torch.nn.Module = None
    ) -> Dict[str, Any]:
        """评估模型性能"""
        model.eval()
        
        all_predictions = {'weights': [], 'returns': [], 'risks': []}
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch[0].to(device)
                targets = batch[1].squeeze(2).to(device)
                targets = targets[:, -1, :]  # 取最后一个时间步
                
                # 模型预测
                outputs = model(features)
                
                # 收集预测结果
                all_predictions['weights'].append(outputs['weights'].cpu().numpy())
                all_predictions['returns'].append(outputs['returns'].cpu().numpy())
                all_predictions['risks'].append(outputs['risks'].cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # 计算损失
                if criterion is not None:
                    loss_dict = criterion(outputs, targets)
                    loss = loss_dict['total_loss']
                    batch_size = features.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
        
        # 合并所有批次的结果
        predictions = {
            'weights': np.concatenate(all_predictions['weights'], axis=0),
            'returns': np.concatenate(all_predictions['returns'], axis=0),
            'risks': np.concatenate(all_predictions['risks'], axis=0)
        }
        targets = np.concatenate(all_targets, axis=0)
        
        # 计算评估指标
        evaluation_results = {
            'predictions': predictions,
            'targets': targets,
            'test_loss': total_loss / total_samples if criterion is not None else None,
            'metrics': self._calculate_metrics(predictions, targets)
        }
        
        return evaluation_results

    def _calculate_metrics(
        self, 
        predictions: Dict[str, np.ndarray], 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        # 收益预测指标
        pred_returns = predictions['returns']
        mse = mean_squared_error(targets, pred_returns)
        mae = mean_absolute_error(targets, pred_returns)
        
        metrics.update({
            'return_mse': mse,
            'return_mae': mae,
            'return_rmse': np.sqrt(mse)
        })
        
        # 权重统计
        weights = predictions['weights'] 
        metrics.update({
            'avg_weight_concentration': np.mean(np.sum(weights**2, axis=1)),
            'avg_max_weight': np.mean(np.max(weights, axis=1)),
            'avg_min_weight': np.mean(np.min(weights, axis=1)),
            'weight_turnover': np.mean(np.sum(np.abs(np.diff(weights, axis=0)), axis=1)) if len(weights) > 1 else 0
        })
        
        # 投资组合收益统计
        portfolio_returns = np.sum(weights * targets, axis=1)
        metrics.update({
            'portfolio_return_mean': np.mean(portfolio_returns),
            'portfolio_return_std': np.std(portfolio_returns),
            'portfolio_sharpe_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
        })
        
        return metrics

    def calculate_portfolio_performance(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """计算投资组合性能指标"""
        
        # 投资组合收益
        portfolio_returns = np.sum(weights * returns, axis=1)
        
        # 基础统计
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # 风险指标
        sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # VaR和CVaR (5%水平)
        var_5 = np.percentile(portfolio_returns, 5)
        cvar_5 = np.mean(portfolio_returns[portfolio_returns <= var_5])
        
        return {
            'mean_return': mean_return,
            'volatility': std_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'total_return': np.prod(1 + portfolio_returns) - 1,
            'annualized_return': (np.prod(1 + portfolio_returns) ** (252 / len(portfolio_returns))) - 1,
            'annualized_volatility': std_return * np.sqrt(252)
        }

    def compare_with_benchmarks(
        self,
        model_weights: np.ndarray,
        returns: np.ndarray,
        benchmarks: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """与基准策略比较"""
        
        results = []
        
        # 模型策略
        model_performance = self.calculate_portfolio_performance(model_weights, returns)
        model_performance['strategy'] = 'TimesFM模型'
        results.append(model_performance)
        
        # 基准策略
        for name, benchmark in benchmarks.items():
            if 'weights' in benchmark:
                # 如果基准有权重，计算其在测试期的表现
                benchmark_weights = np.tile(benchmark['weights'], (len(returns), 1))
                benchmark_performance = self.calculate_portfolio_performance(benchmark_weights, returns)
            else:
                # 如果基准已有收益率序列
                benchmark_returns = benchmark['returns']
                if len(benchmark_returns) != len(returns):
                    # 截取或填充到相同长度
                    min_len = min(len(benchmark_returns), len(returns))
                    benchmark_returns = benchmark_returns[:min_len]
                    returns_subset = returns[:min_len]
                else:
                    returns_subset = returns
                    
                benchmark_performance = {
                    'mean_return': np.mean(benchmark_returns),
                    'volatility': np.std(benchmark_returns),
                    'sharpe_ratio': np.mean(benchmark_returns) / np.std(benchmark_returns) if np.std(benchmark_returns) > 0 else 0,
                    'total_return': np.prod(1 + benchmark_returns) - 1,
                    'annualized_return': (np.prod(1 + benchmark_returns) ** (252 / len(benchmark_returns))) - 1,
                    'annualized_volatility': np.std(benchmark_returns) * np.sqrt(252),
                    'max_drawdown': 0,  # 简化处理
                    'var_5': np.percentile(benchmark_returns, 5),
                    'cvar_5': np.mean(benchmark_returns[benchmark_returns <= np.percentile(benchmark_returns, 5)])
                }
            
            benchmark_performance['strategy'] = benchmark.get('name', name)
            results.append(benchmark_performance)
        
        return pd.DataFrame(results)

    def create_performance_report(
        self,
        evaluation_results: Dict[str, Any],
        benchmarks: Dict[str, Dict[str, Any]] = None,
        save_path: str = None
    ) -> Dict[str, Any]:
        """创建完整的性能报告"""
        
        predictions = evaluation_results['predictions']
        targets = evaluation_results['targets'] 
        metrics = evaluation_results['metrics']
        
        report = {
            'model_metrics': metrics,
            'evaluation_summary': {
                'test_samples': len(targets),
                'assets': self.n_assets,
                'test_loss': evaluation_results.get('test_loss'),
                'avg_portfolio_return': metrics['portfolio_return_mean'],
                'avg_portfolio_volatility': metrics['portfolio_return_std'],
                'sharpe_ratio': metrics['portfolio_sharpe_ratio']
            }
        }
        
        # 与基准比较
        if benchmarks is not None:
            benchmark_comparison = self.compare_with_benchmarks(
                predictions['weights'], targets, benchmarks
            )
            report['benchmark_comparison'] = benchmark_comparison
        
        # 保存报告
        if save_path:
            report_df = pd.DataFrame([report['evaluation_summary']])
            if 'benchmark_comparison' in report:
                report_df = pd.concat([report_df, report['benchmark_comparison']], ignore_index=True)
            report_df.to_csv(save_path, index=False)
            print(f"📊 性能报告已保存: {save_path}")
        
        return report


class PortfolioVisualizer:
    """投资组合可视化工具"""
    
    def __init__(self, asset_names: List[str]):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        
        # 设置matplotlib样式
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_training_history(
        self, 
        history: Dict[str, List], 
        save_path: str = None
    ) -> plt.Figure:
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练历史', fontsize=16, fontweight='bold')
        
        # 损失曲线
        axes[0, 0].plot(history['epochs'], history['train_loss'], label='训练损失', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            val_epochs = history['epochs'][::len(history['epochs'])//len(history['val_loss'])][:len(history['val_loss'])]
            axes[0, 0].plot(val_epochs, history['val_loss'], label='验证损失', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 学习率曲线
        axes[0, 1].plot(history['epochs'], history['learning_rates'], color='orange', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('学习率')
        axes[0, 1].set_title('学习率调度')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # 权重统计
        if 'weight_stats' in history and history['weight_stats']:
            weight_stats = history['weight_stats']
            val_epochs = history['epochs'][::len(history['epochs'])//len(weight_stats)][:len(weight_stats)]
            
            concentration = [stats['weight_concentration'] for stats in weight_stats]
            axes[1, 0].plot(val_epochs, concentration, color='green', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('权重集中度')
            axes[1, 0].set_title('投资组合权重集中度')
            axes[1, 0].grid(True, alpha=0.3)
            
            mean_abs_weight = [stats['mean_abs_weight'] for stats in weight_stats]
            axes[1, 1].plot(val_epochs, mean_abs_weight, color='red', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('平均绝对权重')
            axes[1, 1].set_title('平均绝对权重')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 训练历史图已保存: {save_path}")
        
        return fig

    def plot_portfolio_weights(
        self, 
        weights: np.ndarray, 
        title: str = "投资组合权重分析",
        save_path: str = None
    ) -> plt.Figure:
        """绘制投资组合权重分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 权重时间序列 (stackplot)
        axes[0, 0].stackplot(range(len(weights)), 
                           *[weights[:, i] for i in range(self.n_assets)], 
                           labels=self.asset_names,
                           alpha=0.8)
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('权重')
        axes[0, 0].set_title('权重时间序列 (堆叠图)')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 权重分布箱线图
        weight_df = pd.DataFrame(weights, columns=self.asset_names)
        weight_df.boxplot(ax=axes[0, 1])
        axes[0, 1].set_title('权重分布')
        axes[0, 1].set_ylabel('权重')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 平均权重饼图
        mean_weights = np.mean(weights, axis=0)
        axes[1, 0].pie(mean_weights, labels=self.asset_names, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('平均权重分布')
        
        # 4. 权重统计表
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        stats_data = []
        for i, asset in enumerate(self.asset_names):
            asset_weights = weights[:, i]
            stats_data.append([
                asset,
                f"{np.mean(asset_weights):.4f}",
                f"{np.std(asset_weights):.4f}",
                f"{np.min(asset_weights):.4f}",
                f"{np.max(asset_weights):.4f}"
            ])
        
        table = axes[1, 1].table(
            cellText=stats_data,
            colLabels=['资产', '均值', '标准差', '最小值', '最大值'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('权重统计表')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 权重分析图已保存: {save_path}")
        
        return fig

    def plot_performance_comparison(
        self, 
        comparison_df: pd.DataFrame,
        save_path: str = None
    ) -> plt.Figure:
        """绘制性能比较图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('投资组合策略性能比较', fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('annualized_return', '年化收益率'),
            ('annualized_volatility', '年化波动率'),
            ('sharpe_ratio', '夏普比率'),
            ('max_drawdown', '最大回撤'),
            ('var_5', 'VaR (5%)'),
            ('total_return', '总收益率')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            if metric in comparison_df.columns:
                bars = ax.bar(comparison_df['strategy'], comparison_df[metric])
                
                # 高亮TimesFM模型
                for i, bar in enumerate(bars):
                    if 'TimesFM' in comparison_df.iloc[i]['strategy']:
                        bar.set_color('red')
                        bar.set_alpha(0.8)
                    else:
                        bar.set_alpha(0.6)
                
                ax.set_title(title)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 性能比较图已保存: {save_path}")
        
        return fig

    def plot_cumulative_returns(
        self,
        model_returns: np.ndarray,
        benchmarks: Dict[str, np.ndarray] = None,
        save_path: str = None
    ) -> plt.Figure:
        """绘制累积收益曲线"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 模型累积收益
        model_cumulative = np.cumprod(1 + model_returns)
        ax.plot(model_cumulative, label='TimesFM模型', linewidth=2, color='red')
        
        # 基准累积收益
        if benchmarks:
            for name, returns in benchmarks.items():
                if len(returns) == len(model_returns):
                    cumulative = np.cumprod(1 + returns)
                    ax.plot(cumulative, label=name, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('时间步')
        ax.set_ylabel('累积收益')
        ax.set_title('累积收益对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 累积收益图已保存: {save_path}")
        
        return fig