"""
EGR by Time Scale

Python conversion of EGR_by_time_scale.R
Analysis of Excess Growth Rate by time scale
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_egr(log_caps, time_scale=1):
    """
    Compute Excess Growth Rate at a given time scale
    
    Parameters:
    -----------
    log_caps : numpy.ndarray
        Log capitalizations
    time_scale : int
        Time scale for computing differences
    
    Returns:
    --------
    numpy.ndarray
        EGR time series
    """
    n_stocks, n_times = log_caps.shape
    
    if n_times <= time_scale:
        return np.array([])
    
    egr = np.zeros(n_times - time_scale)
    
    for t in range(n_times - time_scale):
        # Compute log returns
        log_returns = log_caps[:, t + time_scale] - log_caps[:, t]
        log_returns = log_returns[~np.isnan(log_returns)]
        
        if len(log_returns) > 0:
            # Compute market average return
            mean_return = np.mean(log_returns)
            # Compute variance
            var_return = np.var(log_returns)
            # EGR is market average return minus half the variance
            egr[t] = mean_return - 0.5 * var_return
    
    return egr


# Main analysis
if __name__ == "__main__":
    # Import caps
    caps = pd.read_csv("caps_common_1962_2023_ffill.csv")
    
    # Import dates
    dates = pd.read_csv("dates_common_1962_2023.csv", parse_dates=['dates'])
    dates = dates['dates'].values
    
    # Compute log caps
    log_caps = np.log(caps.replace(0, np.nan).values)
    
    # Compute EGR at different time scales
    time_scales = [1, 5, 21, 63, 252]  # 1 day, 1 week, 1 month, 1 quarter, 1 year
    
    fig, axes = plt.subplots(len(time_scales), 1, figsize=(12, 10))
    
    for i, ts in enumerate(time_scales):
        print(f"Computing EGR at time scale {ts}...")
        egr = compute_egr(log_caps, ts)
        
        if len(egr) > 0:
            axes[i].plot(dates[:len(egr)], egr, linewidth=1)
            axes[i].set_ylabel('EGR')
            axes[i].set_title(f'EGR at Time Scale {ts} days')
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('Section_4_Intrinsic_Market_Volatility/egr_by_time_scale.png',
                dpi=150, bbox_inches='tight')
    print("Plot saved to Section_4_Intrinsic_Market_Volatility/egr_by_time_scale.png")
