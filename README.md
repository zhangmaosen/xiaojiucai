# XiaoJiuCai - 小韭菜投资组合优化框架

## 项目简介

XiaoJiuCai 是一个基于深度学习的投资组合优化框架，专注于使用神经网络和时间序列模型来优化投资组合配置。项目通过 PyTorch 实现了多种先进的网络架构，包括卷积神经网络、循环神经网络和注意力机制，以及时序基础模型 (TimesFM) 来处理金融时间序列数据。

## 核心功能

### 1. 深度学习模型
- **ConvNet**: 基于卷积神经网络的投资组合优化模型
- **AttentionNet**: 基于注意力机制的时序数据处理模型
- **HybridNet**: 混合神经网络，结合卷积、RNN和注意力机制
- **TimesFM**: 时序基础模型，用于特征提取和预测

### 2. 投资组合优化
- **权重分配**: 基于Softmax的投资组合权重优化
- **风险管理**: 协方差矩阵估计和风险度量
- **收益优化**: 多种收益和风险指标的优化目标

### 3. 损失函数与指标
- **最大回撤**: MaximumDrawdown 损失函数
- **平均收益**: MeanReturns 收益指标
- **夏普比率**: SharpeRatio 风险调整收益指标
- **自定义损失**: 可扩展的损失函数框架

### 4. 回测与基准
- **基准策略**: 等权重 (OneOverN)、随机分配 (Random) 等基准策略
- **性能评估**: 完整的回测框架和性能分析工具
- **可视化**: 投资组合表现和风险指标可视化

## 技术栈

- **深度学习**: PyTorch
- **数值计算**: NumPy
- **数据分析**: Pandas
- **可视化**: Matplotlib
- **金融数据**: Yahoo Finance (yfinance)
- **科学计算**: Scikit-learn
- **开发环境**: Jupyter Notebook

## 安装指南

### 环境要求
- Python 3.7+
- PyTorch 1.7.0+
- CUDA (可选，用于GPU加速)

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/zhangmaosen/xiaojiucai.git
cd xiaojiucai

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### GPU 支持

如果您有NVIDIA GPU并希望使用CUDA加速，请确保安装了适当版本的PyTorch：

```bash
# 安装支持CUDA的PyTorch (根据您的CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 使用方法

### 快速开始

运行示例代码来了解框架的基本用法：

```bash
# 运行基本示例
python examples/getting_started.py

# 或使用Jupyter Notebook
jupyter notebook notebooks/getting_started.ipynb
```

### 基本用法

```python
import torch
from xiaojiucai.models.networks import ConvNet
from xiaojiucai.data.load import InRAMDataset, RigidDataLoader
from xiaojiucai.experiments import Run
from xiaojiucai.losses import SharpeRatio

# 创建模型
model = ConvNet(
    n_assets=7,  # MAG7 股票数量
    lookback=60,  # 回看期
    n_channels=5,  # 特征维度
    hidden_channels=[32, 64, 128]
)

# 加载数据
dataset = InRAMDataset(
    returns=returns_data,
    features=features_data,
    targets=targets_data,
    lookback=60
)

dataloader = RigidDataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# 创建实验
experiment = Run(
    model=model,
    loaders=(train_loader, val_loader),
    loss=SharpeRatio(),
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 训练模型
results = experiment.launch()
```

### MAG7 投资组合优化示例

项目包含一个完整的 MAG7 股票（微软、苹果、谷歌、亚马逊、特斯拉、英伟达和 Meta）投资组合优化示例：

```bash
# 下载 MAG7 数据
python scripts/download_mag7_data.py

# 运行投资组合优化
jupyter notebook notebooks/mag7_portfolio_optimization.ipynb
```

## 项目结构

```
xiaojiucai/
├── xiaojiucai/                # 主要代码包
│   ├── models/                # 神经网络模型
│   │   ├── base.py           # 基础模型类
│   │   └── networks.py       # 网络模型实现
│   ├── layers/               # 神经网络层
│   │   ├── allocate.py       # 权重分配层
│   │   ├── collapse.py       # 维度压缩层
│   │   ├── misc.py          # 其他工具层
│   │   └── transform.py     # 数据变换层
│   ├── losses/               # 损失函数
│   │   ├── base.py          # 基础损失函数
│   │   ├── returns.py       # 收益相关损失
│   │   └── risk.py          # 风险相关损失
│   ├── data/                 # 数据处理
│   │   ├── load.py          # 数据加载
│   │   └── augment.py       # 数据增强
│   ├── benchmarks/           # 基准策略
│   ├── callbacks/            # 训练回调
│   ├── experiments/          # 实验框架
│   ├── utils/                # 工具函数
│   └── visualize/            # 可视化工具
├── data/                     # 数据文件
│   ├── mag7_data_raw.parquet # 原始MAG7数据
│   └── mag7_data.csv        # 处理后的数据
├── models/                   # 训练好的模型
├── notebooks/                # Jupyter Notebooks
│   ├── getting_started.ipynb
│   ├── mag7_portfolio_optimization.ipynb
│   ├── timesfm_feature_extraction.ipynb
│   └── timesfm_portfolio_optimization.ipynb
├── examples/                 # 示例代码
│   └── getting_started.py
├── scripts/                  # 实用脚本
│   └── download_mag7_data.py
├── README.md                # 项目说明
└── requirements.txt         # 依赖列表
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