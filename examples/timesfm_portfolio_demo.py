"""
TimesFM投资组合优化 - 使用示例

展示如何使用重构后的xiaojiucai项目进行端到端的投资组合优化
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 导入重构后的xiaojiucai模块
from xiaojiucai import (
    # 核心模型
    TimesFMPortfolioModel,
    PortfolioLoss,
    
    # 数据处理
    MAG7DataProcessor,
    create_data_loaders,
    calculate_benchmark_portfolios,
    
    # 训练
    PortfolioTrainer,
    ExperimentManager,
    create_trainer_from_config,
    
    # 评估
    PortfolioEvaluator,
    PortfolioVisualizer
)


def main():
    """主函数 - 完整的投资组合优化流程"""
    
    print("🚀 TimesFM投资组合优化系统 - 重构版本演示")
    print("=" * 60)
    
    # =================================================================
    # 1. 配置参数
    # =================================================================
    
    # 资产列表
    asset_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
    
    # 数据配置
    data_config = {
        'data_path': 'data/mag7_data.csv',  # 根据实际路径调整
        'sequence_length': 60,
        'prediction_horizon': 1,
        'train_ratio': 0.7,
        'val_ratio': 0.15
    }
    
    # 模型配置
    model_config = {
        'input_dim': 4,
        'hidden_dim': 256,
        'num_assets': len(asset_names),
        'num_heads': 8,
        'dropout': 0.1
    }
    
    # 训练配置
    training_config = {
        'epochs': 20,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'min_lr': 1e-6,
        'patience': 10,
        'gradient_clip': 1.0,
        'validate_every': 1,
        'save_every': 5,
        'model_save_dir': 'models/experiments'
    }
    
    # 损失函数配置
    loss_config = {
        'risk_aversion': 1.0,
        'weight_penalty': 0.1
    }
    
    # 实验配置
    experiment_config = {
        'name': 'timesfm_portfolio_demo',
        'description': '重构后的TimesFM投资组合优化演示'
    }
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # =================================================================
    # 2. 数据处理
    # =================================================================
    
    print("\n📊 开始数据处理...")
    
    # 初始化数据处理器
    data_processor = MAG7DataProcessor(asset_names=asset_names)
    
    # 检查数据文件是否存在
    if not Path(data_config['data_path']).exists():
        print(f"❌ 数据文件不存在: {data_config['data_path']}")
        print("请确保MAG7数据文件存在，或使用notebook中的数据加载代码")
        return
    
    try:
        # 加载和处理数据
        raw_df = data_processor.load_data(data_config['data_path'])
        clean_df = data_processor.clean_data(raw_df)
        features_df = data_processor.engineer_features(clean_df)
        
        # 转换为深度学习格式
        X, y = data_processor.to_deepdow_format(
            features_df,
            sequence_length=data_config['sequence_length'],
            prediction_horizon=data_config['prediction_horizon']
        )
        
        # 数据分割
        train_data, val_data, test_data = data_processor.create_data_splits(
            X, y,
            train_ratio=data_config['train_ratio'],
            val_ratio=data_config['val_ratio']
        )
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, val_data, test_data,
            batch_size=training_config['batch_size']
        )
        
        print("✅ 数据处理完成")
        
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        print("使用模拟数据进行演示...")
        
        # 创建模拟数据
        n_samples = 1000
        X = torch.randn(n_samples, model_config['input_dim'], data_config['sequence_length'], model_config['num_assets'])
        y = torch.randn(n_samples, model_config['num_assets'], data_config['prediction_horizon'])
        
        train_data, val_data, test_data = data_processor.create_data_splits(X, y)
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, val_data, test_data,
            batch_size=training_config['batch_size']
        )
        
        print("✅ 模拟数据创建完成")
    
    # =================================================================
    # 3. 模型创建
    # =================================================================
    
    print("\n🤖 创建TimesFM投资组合模型...")
    
    # 创建模型
    model = TimesFMPortfolioModel(**model_config).to(device)
    
    # 创建损失函数
    criterion = PortfolioLoss(**loss_config)
    
    print("✅ 模型创建完成")
    
    # =================================================================
    # 4. 训练模型
    # =================================================================
    
    print("\n🚀 开始模型训练...")
    
    # 创建训练器和优化器
    trainer, optimizer, scheduler = create_trainer_from_config(
        model, criterion, device, training_config
    )
    
    # 训练模型
    try:
        history, best_val_loss = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=training_config,
            experiment_name=experiment_config['name']
        )
        
        print("✅ 模型训练完成")
        print(f"📈 最佳验证损失: {best_val_loss:.6f}")
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        print("继续使用当前模型进行演示...")
        history = {'train_loss': [], 'val_loss': [], 'epochs': []}
    
    # =================================================================
    # 5. 模型评估
    # =================================================================
    
    print("\n📊 开始模型评估...")
    
    # 创建评估器
    evaluator = PortfolioEvaluator(asset_names=asset_names)
    
    # 评估模型
    try:
        evaluation_results = evaluator.evaluate_model(
            model, test_loader, device, criterion
        )
        
        print("✅ 模型评估完成")
        print(f"📊 测试损失: {evaluation_results.get('test_loss', 'N/A')}")
        
        # 打印关键指标
        metrics = evaluation_results['metrics']
        print(f"📈 投资组合夏普比率: {metrics['portfolio_sharpe_ratio']:.4f}")
        print(f"⚖️  平均权重集中度: {metrics['avg_weight_concentration']:.4f}")
        
    except Exception as e:
        print(f"❌ 评估过程出错: {e}")
        evaluation_results = None
    
    # =================================================================
    # 6. 可视化分析
    # =================================================================
    
    print("\n📈 生成可视化分析...")
    
    # 创建可视化工具
    visualizer = PortfolioVisualizer(asset_names=asset_names)
    
    try:
        # 训练历史可视化
        if history and history['epochs']:
            fig1 = visualizer.plot_training_history(
                history, 
                save_path='outputs/training_history.png'
            )
            print("✅ 训练历史图生成完成")
        
        # 权重分析可视化
        if evaluation_results:
            weights = evaluation_results['predictions']['weights']
            fig2 = visualizer.plot_portfolio_weights(
                weights,
                title="TimesFM模型权重分析",
                save_path='outputs/portfolio_weights.png'
            )
            print("✅ 权重分析图生成完成")
        
    except Exception as e:
        print(f"⚠️ 可视化生成部分失败: {e}")
    
    # =================================================================
    # 7. 实验管理
    # =================================================================
    
    print("\n🗂️  实验管理演示...")
    
    # 创建实验管理器
    experiment_manager = ExperimentManager(training_config['model_save_dir'])
    
    try:
        # 列出所有实验
        experiments = experiment_manager.list_experiments()
        
        if experiments:
            # 获取最新实验的摘要
            latest_exp = experiments[-1]
            summary = experiment_manager.get_experiment_summary(latest_exp['path'])
            print(f"📋 最新实验摘要: {latest_exp['dir_name']}")
            print(f"   检查点数量: {len(summary.get('checkpoints', []))}")
        
    except Exception as e:
        print(f"⚠️ 实验管理演示失败: {e}")
    
    # =================================================================
    # 8. 总结
    # =================================================================
    
    print("\n" + "=" * 60)
    print("🎉 TimesFM投资组合优化系统演示完成！")
    print("=" * 60)
    
    print("📦 重构成果:")
    print("  ✅ 模块化架构 - 清晰分离数据处理、模型、训练、评估")
    print("  ✅ TimesFM集成 - 预训练时间序列模型特征提取")
    print("  ✅ 完整训练流程 - 训练、验证、保存、实验管理")
    print("  ✅ 性能评估 - 多维度指标分析和基准比较")
    print("  ✅ 可视化工具 - 训练历史、权重分析、性能比较")
    print("  ✅ 实验管理 - 版本控制、检查点管理、配置追踪")
    
    print("\n🛠️  下一步建议:")
    print("  1. 使用真实MAG7数据进行完整训练")
    print("  2. 调优超参数以获得更好性能")
    print("  3. 扩展到更多资产和策略")
    print("  4. 集成更多基准策略进行比较")
    print("  5. 添加实时预测和部署功能")


if __name__ == "__main__":
    # 创建输出目录
    Path("outputs").mkdir(exist_ok=True)
    Path("models/experiments").mkdir(parents=True, exist_ok=True)
    
    # 运行主程序
    main()