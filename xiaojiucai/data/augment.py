"""数据增强和预处理模块。

包含用于金融数据预处理和增强的工具类。
"""

import numpy as np


class Compose:
    """组合多个数据变换。
    
    Parameters
    ----------
    transforms : list of callable
        要按顺序应用的变换列表。
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, X):
        """应用所有变换。
        
        Parameters
        ----------
        X : np.ndarray
            输入数据。
            
        Returns
        -------
        X_transformed : np.ndarray
            变换后的数据。
        """
        for t in self.transforms:
            X = t(X)
        return X


class Noise:
    """添加随机噪声。
    
    Parameters
    ----------
    std : float
        噪声的标准差。
        
    mean : float, default 0
        噪声的均值。
    """
    
    def __init__(self, std, mean=0):
        self.std = std
        self.mean = mean
        
    def __call__(self, X):
        """添加噪声。
        
        Parameters
        ----------
        X : np.ndarray
            输入数据。
            
        Returns
        -------
        X_noisy : np.ndarray
            添加了噪声的数据。
        """
        noise = np.random.normal(self.mean, self.std, X.shape)
        return X + noise


class Dropout:
    """随机丢弃特征。
    
    Parameters
    ----------
    p : float
        丢弃概率。
    """
    
    def __init__(self, p):
        self.p = p
        
    def __call__(self, X):
        """应用dropout。
        
        Parameters
        ----------
        X : np.ndarray
            输入数据。
            
        Returns
        -------
        X_dropped : np.ndarray
            部分特征被置零的数据。
        """
        mask = np.random.binomial(1, 1-self.p, X.shape)
        return X * mask


class Multiply:
    """数据缩放变换。
    
    Parameters
    ----------
    factor : float
        缩放因子。
    """
    
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, X):
        """应用缩放。
        
        Parameters
        ----------
        X : np.ndarray
            输入数据。
            
        Returns
        -------
        X_scaled : np.ndarray
            缩放后的数据。
        """
        return X * self.factor