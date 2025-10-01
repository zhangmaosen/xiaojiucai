"""
投资组合可视化工具

包含投资组合分析和可视化的工具函数
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import torch


def setup_matplotlib_chinese():
    """设置matplotlib中文字体和数学符号"""
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # 设置字体和数学符号，避免警告
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.default'] = 'regular'  # 避免数学符号警告
    plt.rcParams['mathtext.fontset'] = 'dejavusans'  # 使用DejaVu Sans作为数学字体
    
    try:
        import matplotlib.font_manager as fm
        chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'Hei' in f.name or 'Kai' in f.name or 'Song' in f.name]
        if chinese_fonts:
            plt.rcParams['font.sans-serif'].insert(0, chinese_fonts[0])
            print(f"使用中文字体: {chinese_fonts[0]}")
        else:
            print("未找到中文字体，使用默认英文字体")
    except Exception as e:
        print(f"字体配置警告: {e}")
        print("使用默认字体设置")


def plot_training_curves(train_losses: List[float], val_losses: List[float], title="训练和验证损失曲线"):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_weight_allocation_over_time(predictions: np.ndarray, tickers: List[str], 
                                   title="投资组合权重分配随时间的变化"):
    """绘制投资组合权重分配随时间的变化"""
    plt.figure(figsize=(14, 8))
    plt.stackplot(range(len(predictions)),
                 predictions.T,
                 labels=tickers,
                 alpha=0.8)
    plt.xlabel('时间步')
    plt.ylabel('权重')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_average_weight_pie(predictions: np.ndarray, tickers: List[str],
                           title="平均投资组合权重分配"):
    """绘制平均权重分配饼图"""
    avg_weights = np.mean(predictions, axis=0)
    plt.figure(figsize=(10, 8))
    plt.pie(avg_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.show()


def plot_cumulative_returns(portfolio_returns: np.ndarray, individual_returns: np.ndarray, 
                          tickers: List[str], title="累积收益对比"):
    """绘制累积收益曲线对比"""
    cumulative_portfolio = np.cumprod(1 + portfolio_returns) - 1
    cumulative_individual = np.cumprod(1 + individual_returns, axis=0) - 1

    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_portfolio, label='投资组合', linewidth=3, color='red')

    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
    for i, ticker in enumerate(tickers):
        plt.plot(cumulative_individual[:, i], label=f'{ticker}个股',
                 linewidth=1.5, alpha=0.7, color=colors[i])

    plt.xlabel('时间步')
    plt.ylabel('累积收益')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_weight_stability(predictions: np.ndarray, tickers: List[str], 
                         title="投资组合权重稳定性分析"):
    """绘制权重稳定性分析"""
    weight_changes = np.abs(np.diff(predictions, axis=0))
    avg_weight_changes = np.mean(weight_changes, axis=0)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(tickers, avg_weight_changes, color='skyblue', alpha=0.7)
    plt.xlabel('股票')
    plt.ylabel('平均权重变化幅度')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)

    # 添加数值标签
    for bar, value in zip(bars, avg_weight_changes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{value:.4f}', ha='center', va='bottom', fontsize=9)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_risk_return_scatter(portfolio_metrics: Dict[str, float], 
                           individual_metrics: Dict[str, np.ndarray],
                           tickers: List[str], title="风险-收益分析"):
    """绘制风险-收益散点图"""
    plt.figure(figsize=(12, 8))

    # 绘制单个股票
    individual_returns = individual_metrics['annual_returns']
    individual_volatility = individual_metrics['annual_volatility'] 
    individual_sharpe = individual_metrics['annual_sharpe']
    
    scatter = plt.scatter(individual_volatility, individual_returns,
                         s=individual_sharpe * 100,
                         c=individual_sharpe, cmap='RdYlGn', alpha=0.7, edgecolors='black')

    # 绘制投资组合
    portfolio_vol = portfolio_metrics['annual_volatility']
    portfolio_ret = portfolio_metrics['annual_return']
    portfolio_sharpe = portfolio_metrics['annual_sharpe']
    
    plt.scatter(portfolio_vol, portfolio_ret, s=portfolio_sharpe * 100,
               c=portfolio_sharpe, cmap='RdYlGn', marker='*', 
               edgecolors='black', linewidth=2, label='投资组合')

    # 添加股票标签
    for i, stock in enumerate(tickers):
        plt.annotate(stock, (individual_volatility[i], individual_returns[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.xlabel('年化波动率')
    plt.ylabel('年化收益率')
    plt.title(title, fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('夏普比率')

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe_ratio(portfolio_returns: np.ndarray, individual_returns: np.ndarray,
                             tickers: List[str], window: int = 20,
                             title="滚动夏普比率分析"):
    """绘制滚动夏普比率"""
    def rolling_sharpe_ratio(returns, window_size):
        """计算滚动夏普比率"""
        if len(returns) < window_size:
            return np.array([])

        rolling_sharpe = []
        for i in range(window_size, len(returns) + 1):
            window_returns = returns[i-window_size:i]
            mean_ret = np.mean(window_returns)
            std_ret = np.std(window_returns)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            rolling_sharpe.append(sharpe * np.sqrt(252))  # 年化

        return np.array(rolling_sharpe)

    # 计算滚动夏普比率
    portfolio_rolling_sharpe = rolling_sharpe_ratio(portfolio_returns, window)
    individual_rolling_sharpe = np.array([rolling_sharpe_ratio(individual_returns[:, i], window)
                                        for i in range(len(tickers))])

    plt.figure(figsize=(14, 8))
    plt.plot(portfolio_rolling_sharpe, label='投资组合', linewidth=3, color='red')

    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
    for i, ticker in enumerate(tickers):
        if len(individual_rolling_sharpe[i]) > 0:
            plt.plot(individual_rolling_sharpe[i], label=f'{ticker}个股',
                     linewidth=1.5, alpha=0.7, color=colors[i])

    plt.xlabel('时间步')
    plt.ylabel(f'滚动夏普比率 ({window}期窗口)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_weight_distributions(predictions: np.ndarray, tickers: List[str],
                             title="权重分布直方图"):
    """绘制权重分布直方图"""
    n_assets = len(tickers)
    n_cols = 4
    n_rows = (n_assets + n_cols - 1) // n_cols
    
    plt.figure(figsize=(16, 4 * n_rows))

    for i, ticker in enumerate(tickers):
        plt.subplot(n_rows, n_cols, i+1)
        plt.hist(predictions[:, i], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(predictions[:, i]), color='red', linestyle='--',
                    label=f'均值: {np.mean(predictions[:, i]):.3f}')
        plt.title(f'{ticker} 权重分布')
        plt.xlabel('权重')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def calculate_performance_metrics(portfolio_returns: np.ndarray, individual_returns: np.ndarray) -> Dict[str, Any]:
    """计算绩效指标"""
    def calculate_max_drawdown(returns):
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)

    def calculate_win_rate(returns):
        """计算胜率"""
        return np.mean(returns > 0)

    # 投资组合指标
    portfolio_metrics = {
        'mean_return': np.mean(portfolio_returns),
        'std_return': np.std(portfolio_returns),
        'annual_return': np.mean(portfolio_returns) * 252,
        'annual_volatility': np.std(portfolio_returns) * np.sqrt(252),
        'max_drawdown': calculate_max_drawdown(portfolio_returns),
        'win_rate': calculate_win_rate(portfolio_returns)
    }
    portfolio_metrics['sharpe_ratio'] = portfolio_metrics['mean_return'] / portfolio_metrics['std_return']
    portfolio_metrics['annual_sharpe'] = portfolio_metrics['annual_return'] / portfolio_metrics['annual_volatility']

    # 单个股票指标
    individual_metrics = {
        'mean_returns': np.mean(individual_returns, axis=0),
        'std_returns': np.std(individual_returns, axis=0),
        'annual_returns': np.mean(individual_returns, axis=0) * 252,
        'annual_volatility': np.std(individual_returns, axis=0) * np.sqrt(252),
        'max_drawdowns': [calculate_max_drawdown(individual_returns[:, i]) for i in range(individual_returns.shape[1])],
        'win_rates': [calculate_win_rate(individual_returns[:, i]) for i in range(individual_returns.shape[1])]
    }
    individual_metrics['sharpe_ratios'] = individual_metrics['mean_returns'] / individual_metrics['std_returns']
    individual_metrics['annual_sharpe'] = individual_metrics['annual_returns'] / individual_metrics['annual_volatility']

    return portfolio_metrics, individual_metrics


def print_performance_summary(portfolio_metrics: Dict[str, float], 
                             individual_metrics: Dict[str, np.ndarray],
                             tickers: List[str]):
    """打印绩效汇总表"""
    print("\n" + "="*70)
    print("详细绩效分析汇总")
    print("="*70)
    
    print(f"{'资产':<8} {'年化收益':>10} {'年化波动':>10} {'夏普比率':>10} {'最大回撤':>10} {'胜率':>8}")
    print("-" * 70)

    # 投资组合
    print(f"{'组合':<8} {portfolio_metrics['annual_return']:>9.2%} {portfolio_metrics['annual_volatility']:>9.2%} "
          f"{portfolio_metrics['annual_sharpe']:>9.2f} {portfolio_metrics['max_drawdown']:>9.2%} "
          f"{portfolio_metrics['win_rate']:>7.1%}")

    # 单个股票
    for i, ticker in enumerate(tickers):
        print(f"{ticker:<8} {individual_metrics['annual_returns'][i]:>9.2%} "
              f"{individual_metrics['annual_volatility'][i]:>9.2%} "
              f"{individual_metrics['annual_sharpe'][i]:>9.2f} "
              f"{individual_metrics['max_drawdowns'][i]:>9.2%} "
              f"{individual_metrics['win_rates'][i]:>7.1%}")

    print("="*70)


def create_comprehensive_report(predictions: np.ndarray, portfolio_returns: np.ndarray, 
                               individual_returns: np.ndarray, tickers: List[str]):
    """创建综合分析报告"""
    setup_matplotlib_chinese()
    
    # 计算绩效指标
    portfolio_metrics, individual_metrics = calculate_performance_metrics(portfolio_returns, individual_returns)
    
    # 绘制所有图表
    plot_weight_allocation_over_time(predictions, tickers)
    plot_average_weight_pie(predictions, tickers)
    plot_cumulative_returns(portfolio_returns, individual_returns, tickers)
    plot_weight_stability(predictions, tickers)
    plot_risk_return_scatter(portfolio_metrics, individual_metrics, tickers)
    plot_rolling_sharpe_ratio(portfolio_returns, individual_returns, tickers)
    plot_weight_distributions(predictions, tickers)
    
    # 打印绩效汇总
    print_performance_summary(portfolio_metrics, individual_metrics, tickers)
    
    return portfolio_metrics, individual_metrics