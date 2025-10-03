"""
Strategy Risk Analysis

Python conversion of StrategyRisk.R
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_risk_metrics(returns):
    """
    Compute risk metrics for a return series
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Return series
    
    Returns:
    --------
    dict
        Dictionary of risk metrics
    """
    metrics = {
        'volatility': np.std(returns),
        'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
        'max_drawdown': compute_max_drawdown(returns)
    }
    return metrics


def compute_max_drawdown(returns):
    """Compute maximum drawdown"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


if __name__ == "__main__":
    print("Template for strategy risk analysis")
    print("Customize this script based on your portfolio returns")
