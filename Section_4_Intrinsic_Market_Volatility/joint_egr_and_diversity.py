"""
Joint EGR and Diversity

Python conversion of Joint_EGRandDiversity.R
Joint analysis of EGR and diversity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../Section_3_Capital_Distribution_and_Diversity')


def compute_egr(log_caps, time_scale=1):
    """Compute Excess Growth Rate"""
    n_stocks, n_times = log_caps.shape
    if n_times <= time_scale:
        return np.array([])
    
    egr = np.zeros(n_times - time_scale)
    for t in range(n_times - time_scale):
        log_returns = log_caps[:, t + time_scale] - log_caps[:, t]
        log_returns = log_returns[~np.isnan(log_returns)]
        if len(log_returns) > 0:
            egr[t] = np.mean(log_returns) - 0.5 * np.var(log_returns)
    return egr


def diversity_index(weights):
    """Compute diversity index"""
    weights = weights[weights > 0]
    if len(weights) == 0:
        return 0
    return np.exp(-np.sum(weights * np.log(weights)))


def compute_diversity(caps):
    """Compute diversity time series"""
    if isinstance(caps, pd.DataFrame):
        caps = caps.values
    
    n_times = caps.shape[1]
    diversity = np.zeros(n_times)
    
    for t in range(n_times):
        cap_vals = caps[:, t].copy()
        cap_vals[cap_vals < 0] = 0
        cap_vals[np.isnan(cap_vals)] = 0
        total_cap = np.sum(cap_vals)
        if total_cap > 0:
            weights = cap_vals / total_cap
            diversity[t] = diversity_index(weights)
    
    return diversity


# Main analysis
if __name__ == "__main__":
    # Import caps
    caps = pd.read_csv("caps_common_1962_2023_ffill.csv")
    
    # Import dates
    dates = pd.read_csv("dates_common_1962_2023.csv", parse_dates=['dates'])
    dates = dates['dates'].values
    
    # Compute log caps
    log_caps = np.log(caps.replace(0, np.nan).values)
    
    # Compute EGR and diversity
    print("Computing EGR...")
    time_scale = 21  # 1 month
    egr = compute_egr(log_caps, time_scale)
    
    print("Computing diversity...")
    diversity = compute_diversity(caps)
    
    # Align arrays
    min_len = min(len(egr), len(diversity))
    egr = egr[:min_len]
    diversity = diversity[:min_len]
    dates_aligned = dates[:min_len]
    
    # Plot joint analysis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot EGR
    ax1.plot(dates_aligned, egr, linewidth=1)
    ax1.set_ylabel('EGR')
    ax1.set_title('Excess Growth Rate')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot Diversity
    ax2.plot(dates_aligned, diversity, linewidth=1, color='orange')
    ax2.set_ylabel('Diversity')
    ax2.set_title('Market Diversity')
    ax2.grid(True, alpha=0.3)
    
    # Scatter plot
    ax3.scatter(diversity, egr, alpha=0.3, s=10)
    ax3.set_xlabel('Diversity')
    ax3.set_ylabel('EGR')
    ax3.set_title('EGR vs Diversity')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Section_4_Intrinsic_Market_Volatility/joint_egr_diversity.png',
                dpi=150, bbox_inches='tight')
    print("Plot saved to Section_4_Intrinsic_Market_Volatility/joint_egr_diversity.png")
