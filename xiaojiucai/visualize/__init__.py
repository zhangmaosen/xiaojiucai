"""
可视化模块

包含投资组合分析和可视化工具
"""

from .portfolio_viz import (
    setup_matplotlib_chinese,
    plot_training_curves,
    plot_weight_allocation_over_time,
    plot_average_weight_pie,
    plot_cumulative_returns,
    plot_weight_stability,
    plot_risk_return_scatter,
    plot_rolling_sharpe_ratio,
    plot_weight_distributions,
    calculate_performance_metrics,
    print_performance_summary,
    create_comprehensive_report
)

__all__ = [
    'setup_matplotlib_chinese',
    'plot_training_curves',
    'plot_weight_allocation_over_time',
    'plot_average_weight_pie',
    'plot_cumulative_returns',
    'plot_weight_stability',
    'plot_risk_return_scatter',
    'plot_rolling_sharpe_ratio',
    'plot_weight_distributions',
    'calculate_performance_metrics',
    'print_performance_summary',
    'create_comprehensive_report'
]
