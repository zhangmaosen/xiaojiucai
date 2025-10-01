#!/usr/bin/env python3
"""
模型管理示例脚本

展示如何加载、管理和恢复训练模型
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from xiaojiucai.utils import ModelLoader, load_timesfm_portfolio_model, ModelTrainer, TrainingConfig
from xiaojiucai.losses import portfolio_loss


def demo_model_loading():
    """演示模型加载功能"""
    print("🔧 模型加载功能演示")
    print("=" * 60)
    
    # 创建模型加载器
    device = 'cpu'  # 可以改为 'cuda' 如果有GPU
    loader = ModelLoader(device=device)
    
    # 列出所有可用的模型文件
    print("\n📁 列出模型文件:")
    model_files = loader.list_model_files('models')
    
    if not model_files:
        print("❌ 没有找到模型文件，请先训练模型")
        return False
    
    # 获取第一个模型的信息
    print(f"\n📊 获取模型信息:")
    model_info = loader.get_model_info(model_files[0])
    if model_info:
        print(f"  文件名: {model_info['file_name']}")
        print(f"  训练轮数: {model_info['epoch']}")
        print(f"  验证损失: {model_info['val_loss']}")
        print(f"  时间戳: {model_info['timestamp']}")
        print(f"  模型参数: {model_info['model_params']}")
    
    # 比较多个模型（如果有的话）
    if len(model_files) > 1:
        print(f"\n🔍 比较前3个模型:")
        loader.compare_models(model_files[:3])
    
    # 尝试加载TimesFM投资组合模型
    print(f"\n🤖 加载TimesFM投资组合模型:")
    try:
        model, checkpoint, config = load_timesfm_portfolio_model(model_files[0], device)
        
        print(f"✅ 模型加载成功!")
        print(f"  模型类型: {type(model).__name__}")
        print(f"  模型参数数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  可训练参数数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  配置: {config.finetune_timesfm=}, {config.window_size=}, {config.batch_size=}")
        
        return True, model, checkpoint, config
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def demo_resume_training():
    """演示恢复训练功能"""
    print("\n\n🔄 恢复训练功能演示")
    print("=" * 60)
    
    # 首先加载模型和配置
    success, model, checkpoint, config = demo_model_loading()
    if not success:
        print("❌ 无法加载模型，跳过恢复训练演示")
        return
    
    # 创建训练器
    device = 'cpu'
    
    def loss_fn(weights, returns):
        return portfolio_loss(weights, returns, config.risk_aversion, config.sharpe_weight)
    
    trainer = ModelTrainer(
        model=model,
        config=config,
        loss_fn=loss_fn,
        device=device,
        model_save_dir='models'
    )
    
    print(f"📋 训练器创建成功")
    print(f"  设备: {device}")
    print(f"  配置: epochs={config.num_epochs}, patience={config.patience}")
    
    # 演示检查点加载（不实际继续训练）
    print(f"\n📥 演示检查点加载功能:")
    
    # 查找检查点文件
    loader = ModelLoader(device)
    model_files = loader.list_model_files('models')
    
    checkpoint_files = [f for f in model_files if 'checkpoint' in os.path.basename(f)]
    
    if checkpoint_files:
        checkpoint_file = checkpoint_files[0]
        print(f"📋 找到检查点文件: {os.path.basename(checkpoint_file)}")
        
        # 加载检查点状态
        loaded_checkpoint = trainer.load_checkpoint(checkpoint_file)
        
        print(f"✅ 检查点状态加载成功:")
        print(f"  训练历史长度: {len(trainer.train_losses)}")
        print(f"  验证历史长度: {len(trainer.val_losses)}")
        print(f"  最佳验证损失: {trainer.best_val_loss:.6f}")
        
    else:
        print("⚠️ 没有找到检查点文件")
    
    print(f"\n💡 要实际恢复训练，请调用:")
    print(f"   trainer.resume_training(checkpoint_path, train_loader, val_loader)")


def demo_model_comparison():
    """演示模型对比功能"""
    print("\n\n📊 模型对比功能演示")
    print("=" * 60)
    
    loader = ModelLoader(device='cpu')
    model_files = loader.list_model_files('models')
    
    if len(model_files) < 2:
        print("⚠️ 需要至少2个模型文件才能进行对比")
        return
    
    # 对比所有模型
    print(f"🔍 对比所有 {len(model_files)} 个模型:")
    loader.compare_models(model_files)
    
    # 找出最佳模型
    best_models = []
    for path in model_files:
        info = loader.get_model_info(path)
        if info and isinstance(info['val_loss'], (int, float)):
            best_models.append((path, info['val_loss'], info['file_name']))
    
    if best_models:
        best_models.sort(key=lambda x: x[1])  # 按验证损失排序
        
        print(f"\n🏆 验证损失最低的前3个模型:")
        for i, (path, val_loss, name) in enumerate(best_models[:3]):
            print(f"  {i+1}. {name} (验证损失: {val_loss:.6f})")


def main():
    """主函数"""
    print("🧪 模型管理功能演示")
    print("=" * 80)
    
    try:
        # 1. 模型加载演示
        demo_model_loading()
        
        # 2. 恢复训练演示
        demo_resume_training()
        
        # 3. 模型对比演示
        demo_model_comparison()
        
        print(f"\n✅ 所有演示完成!")
        print(f"\n💡 可用的模型管理功能:")
        print(f"   - ModelLoader.list_model_files(): 列出所有模型文件")
        print(f"   - ModelLoader.get_model_info(): 获取模型信息")
        print(f"   - ModelLoader.compare_models(): 对比模型")
        print(f"   - load_timesfm_portfolio_model(): 加载TimesFM模型")
        print(f"   - ModelTrainer.load_checkpoint(): 加载检查点")
        print(f"   - ModelTrainer.resume_training(): 恢复训练")
        
    except Exception as e:
        print(f"❌ 演示过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()