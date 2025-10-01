#!/usr/bin/env python3
"""
TimesFM预训练参数加载演示

演示更新后的TimesFM模型创建流程，包含预训练参数加载功能。
"""

from xiaojiucai.models.timesfm_wrapper import create_timesfm_model

def main():
    print("🚀 TimesFM预训练参数加载演示")
    print("=" * 50)
    
    print("\n📦 创建TimesFM模型（包含预训练参数加载）...")
    try:
        # 创建TimesFM模型，现在包含预训练参数加载
        timesfm_model, timesfm_wrapper = create_timesfm_model()
        
        print(f"\n✅ 模型创建成功！")
        print(f"  模型类型: {type(timesfm_model).__name__}")
        print(f"  包装器类型: {type(timesfm_wrapper).__name__}")
        print(f"  设备: {timesfm_model.model.device}")
        print(f"  模型配置: {timesfm_model.forecast_config}")
        
        print("\n🎯 主要改进:")
        print("  ✓ 自动加载预训练参数 (model.load_checkpoint())")
        print("  ✓ 支持梯度计算的包装器")
        print("  ✓ 完整的编译和配置流程")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return
    
    print("\n📊 测试基本功能...")
    try:
        import torch
        
        # 创建测试输入
        batch_size = 2
        context_length = 100
        horizon = 20
        
        test_inputs = torch.randn(batch_size, context_length)
        test_masks = torch.ones(batch_size, context_length, dtype=torch.bool)
        
        # 推理模式测试
        timesfm_wrapper.model.eval()
        with torch.no_grad():
            forecast, quantiles, _ = timesfm_wrapper.decode_with_grad(
                horizon=horizon, 
                inputs=test_inputs, 
                masks=test_masks,
                enable_grad=False
            )
            
        print(f"  ✓ 推理模式预测形状: {forecast.shape}")
        print(f"  ✓ 分位数预测形状: {quantiles.shape}")
        
    except Exception as e:
        print(f"  ❌ 功能测试失败: {e}")
    
    print("\n🎉 演示完成！")
    print("\n💡 现在您可以使用以下代码创建包含预训练参数的TimesFM模型:")
    print("from xiaojiucai.models.timesfm_wrapper import create_timesfm_model")
    print("timesfm_model, wrapper = create_timesfm_model()")

if __name__ == "__main__":
    main()