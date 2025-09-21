# 指数增强模型 (Index Enhanced Model) 与 投资组合优化 (Portfolio Optimization)

## 项目简介

本项目专注于开发和实现指数增强模型，通过先进的量化金融技术和机器学习算法来优化投资组合表现。项目旨在通过系统性的方法增强传统指数跟踪策略，实现超越基准指数的收益。

## 核心功能

### 1. 指数增强模型
- **因子分析**: 多因子模型构建与分析，包括价值、动量、质量、规模等因子
- **风险模型**: 高级风险模型构建，包括协方差矩阵估计和风险因子分解
- **优化算法**: 多样化的投资组合优化算法实现

### 2. 投资组合优化
- **均值方差优化**: 经典马科维茨投资组合优化
- **风险平价**: 基于风险贡献的资产配置策略
- **Black-Litterman模型**: 结合市场均衡和投资者观点的投资组合构建
- **最小方差前沿**: 最小化投资组合风险的优化方法

### 3. 回测框架
- **历史回测**: 基于历史数据的策略性能评估
- **业绩归因**: 投资组合收益来源分析
- **风险分析**: 在险价值(VaR)、最大回撤等风险指标计算

## 技术栈

- **编程语言**: Python 3.x
- **数值计算**: NumPy, SciPy
- **数据分析**: Pandas, Matplotlib, Seaborn
- **机器学习**: Scikit-learn
- **金融计算**: PyPortfolioOpt, QuantLib (可选)
- **可视化**: Plotly, Bokeh (可选)

## 安装指南

### 环境要求
- Python 3.7+
- pip 或 conda

### 安装步骤

```bash
# 克隆项目
git clone <项目地址>
cd xiaojiucai

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```python
# 示例代码
from portfolio_optimizer import PortfolioOptimizer

# 初始化优化器
optimizer = PortfolioOptimizer()

# 加载数据
data = optimizer.load_data('data/sample_data.csv')

# 执行优化
weights = optimizer.optimize(data, method='mean_variance')

# 生成报告
optimizer.generate_report(weights)
```

### 配置参数

项目支持多种配置选项，可通过配置文件或直接在代码中设置：

- 风险偏好系数
- 约束条件（权重上下限、行业暴露等）
- 交易成本模型
- 市场冲击模型

## 项目结构

```
xiaojiucai/
├── data/                  # 数据文件
├── models/                # 模型实现
├── optimization/          # 优化算法
├── backtest/              # 回测框架
├── risk_models/           # 风险模型
├── factor_models/         # 因子模型
├── reports/               # 报告输出
├── utils/                 # 工具函数
├── tests/                 # 测试代码
├── examples/              # 示例代码
├── notebooks/             # Jupyter Notebooks
├── docs/                  # 文档
├── README.md              # 项目说明
└── requirements.txt       # 依赖列表
```

## Jupyter Notebook 示例

项目包含一个完整的示例 notebook，演示如何使用 Yahoo Finance 数据和 MAG7 股票（微软、苹果、谷歌、亚马逊、特斯拉、英伟达和 Meta）来训练投资组合优化模型：

- [MAG7 Portfolio Optimization](notebooks/mag7_portfolio_optimization.ipynb) - 使用实际市场数据训练和评估投资组合优化模型

## 贡献指南

欢迎对项目进行贡献！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有任何问题或建议，请通过以下方式联系：

- 创建 Issue
- 提交 Pull Request

## 致谢

- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) - 投资组合优化库
- [QuantLib](https://www.quantlib.org/) - 量化金融库
- 相关学术论文和研究成果