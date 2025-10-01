"""
特征配置模块

定义了各种预设的特征配置选项
"""

# 预设配置1：仅价格特征
PRICE_ONLY = ['Close', 'High', 'Low', 'Open']

# 预设配置2：价格+成交量
PRICE_AND_VOLUME = ['Close', 'High', 'Low', 'Open', 'Volume']

# 预设配置3：仅收盘价
CLOSE_ONLY = ['Close']

# 预设配置4：高低收盘价
OHLC = ['Open', 'High', 'Low', 'Close']

# 预设配置5：收盘价+成交量
CLOSE_AND_VOLUME = ['Close', 'Volume']

# 所有可用特征
AVAILABLE_FEATURES = ['Close', 'High', 'Low', 'Open', 'Volume']

# 特征配置字典
FEATURE_CONFIGS = {
    'price_only': PRICE_ONLY,
    'price_and_volume': PRICE_AND_VOLUME, 
    'close_only': CLOSE_ONLY,
    'ohlc': OHLC,
    'close_and_volume': CLOSE_AND_VOLUME
}

def validate_features(selected_features):
    """
    验证选择的特征是否有效
    
    Args:
        selected_features: 选择的特征列表
        
    Returns:
        bool: 特征是否有效
        
    Raises:
        ValueError: 如果包含无效特征
    """
    invalid_features = [f for f in selected_features if f not in AVAILABLE_FEATURES]
    if invalid_features:
        raise ValueError(f"无效的特征: {invalid_features}. 可用特征: {AVAILABLE_FEATURES}")
    return True

def print_feature_configs():
    """打印所有可用的特征配置"""
    print("可用的特征配置:")
    for config_name, features in FEATURE_CONFIGS.items():
        print(f"  {config_name}: {features}")
    
    print("\n使用方法:")
    print("1. 选择预设配置：SELECTED_FEATURES = FEATURE_CONFIGS['price_only']")
    print("2. 自定义配置：SELECTED_FEATURES = ['Close', 'High', 'Volume']")
    print("3. 使用所有特征：SELECTED_FEATURES = ['Close', 'High', 'Low', 'Open', 'Volume']")