#!/usr/bin/env python3
"""
TimesFM投资组合优化示例脚本

展示如何使用重构后的工具类进行投资组合优化
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np

# 导入项目工具类
from xiaojiucai.data import (
    DataProcessor, prepare_data_tensors, create_data_loaders,
    FEATURE_CONFIGS
)
from xiaojiucai.models import create_timesfm_model, TimesFMPortfolioModelV2
from xiaojiucai.losses import portfolio_loss
from xiaojiucai.utils import TrainingConfig, ModelTrainer
from xiaojiucai.visualize import setup_matplotlib_chinese, create_comprehensive_report


def main():
    """主函数"""
    print("=" * 60)
    print("TimesFM投资组合优化 - 重构版本示例")
    print("=" * 60)
    
    # 1. 环境设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"使用设备: {device}")
    
    # 2. 数据处理
    print("\n📊 数据处理...")
    processor = DataProcessor(
        selected_features=FEATURE_CONFIGS['close_only'],
        test_mode=True,  # 启用测试模式以加快运行
        test_data_size=600
    )
    
    features_data, returns, tickers = processor.load_and_preprocess('data/mag7_data_raw.parquet')
    
    # 3. 创建训练数据
    print("\n🔧 创建训练数据...")
    window_size = 100  # 测试模式使用较小窗口
    data_dict = prepare_data_tensors(
        features_data, returns, window_size,
        train_ratio=0.8, val_ratio=0.8, device=device
    )
    
    batch_size = 64  # 测试模式使用较小批次
    train_loader, val_loader, test_loader = create_data_loaders(data_dict, batch_size)
    
    # 4. 模型创建
    print("\n🤖 创建模型...")
    timesfm_model, timesfm_wrapper = create_timesfm_model()
    
    portfolio_model = TimesFMPortfolioModelV2(
        input_size=data_dict['input_size'],
        output_size=data_dict['input_size'], 
        feature_size=data_dict['feature_size'],
        timesfm_wrapper=timesfm_wrapper,
        context_len=data_dict['window_size'],
        finetune_timesfm=False  # 测试模式不启用微调以加快速度
    ).to(device)
    
    # 5. 训练配置
    print("\n⚙️ 配置训练参数...")
    config = TrainingConfig()
    config.update(
        window_size=window_size,
        batch_size=batch_size,
        num_epochs=5,  # 测试模式使用较少epochs
        patience=10,
        finetune_timesfm=False,
        risk_aversion=1.0,
        sharpe_weight=0.5
    )
    
    # 6. 模型训练
    print("\n🚀 开始训练...")
    def loss_fn(weights, returns):
        return portfolio_loss(weights, returns, config.risk_aversion, config.sharpe_weight)
    
    trainer = ModelTrainer(
        model=portfolio_model,
        config=config,
        loss_fn=loss_fn,
        device=device,
        model_save_dir='models'
    )
    
    training_results = trainer.train(train_loader, val_loader)
    
    # 7. 模型评估
    print("\n📈 模型评估...")
    portfolio_model.eval()
    test_predictions = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            weights = portfolio_model(batch_X)
            test_predictions.append(weights.cpu())
    
    test_predictions = torch.cat(test_predictions, dim=0).numpy()
    test_returns = data_dict['y_test'].cpu().numpy()
    portfolio_returns = np.sum(test_predictions * test_returns, axis=1)
    
    # 8. 显示结果
    print("\n📊 训练结果:")
    print(f"  最佳验证损失: {training_results['best_val_loss']:.4f}")
    print(f"  投资组合平均收益: {np.mean(portfolio_returns):.4f}")
    print(f"  投资组合收益波动: {np.std(portfolio_returns):.4f}")
    print(f"  夏普比率: {np.mean(portfolio_returns) / np.std(portfolio_returns):.4f}")
    
    # 显示平均权重分配
    avg_weights = np.mean(test_predictions, axis=0)
    print(f"\n💰 平均权重分配:")
    for ticker, weight in zip(tickers, avg_weights):
        print(f"  {ticker}: {weight:.3f} ({weight*100:.1f}%)")
    
    print("\n✅ 示例运行完成！")
    print("\n💡 提示:")
    print("  - 要进行完整训练，请在配置中设置 test_mode=False")
    print("  - 要启用TimesFM微调，请设置 finetune_timesfm=True")
    print("  - 要查看可视化结果，请运行notebook版本")
    
    return training_results


if __name__ == "__main__":
    try:
        results = main()
        print(f"\n🎯 程序正常结束")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)