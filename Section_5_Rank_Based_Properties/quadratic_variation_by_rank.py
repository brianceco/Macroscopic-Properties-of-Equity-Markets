"""
Quadratic Variation by Rank

Python conversion of QuadraticVariation_byRank.R
Analysis of quadratic variation by rank
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_qv_by_rank(log_caps, max_rank=100):
    """
    Compute quadratic variation by rank
    
    Parameters:
    -----------
    log_caps : numpy.ndarray
        Log capitalizations matrix
    max_rank : int
        Maximum rank to consider
    
    Returns:
    --------
    numpy.ndarray
        Quadratic variation for each rank
    """
    n_stocks, n_times = log_caps.shape
    qv_by_rank = np.zeros(max_rank)
    counts = np.zeros(max_rank)
    
    for t in range(1, n_times):
        # Get ranks at time t-1
        caps_prev = np.exp(log_caps[:, t-1])
        caps_prev[np.isnan(caps_prev)] = 0
        sorted_indices = np.argsort(-caps_prev)
        
        # Compute squared returns for top stocks
        for rank in range(min(max_rank, n_stocks)):
            stock_idx = sorted_indices[rank]
            if not np.isnan(log_caps[stock_idx, t]) and not np.isnan(log_caps[stock_idx, t-1]):
                ret = log_caps[stock_idx, t] - log_caps[stock_idx, t-1]
                qv_by_rank[rank] += ret**2
                counts[rank] += 1
    
    # Average quadratic variation
    qv_by_rank = np.where(counts > 0, qv_by_rank / counts, 0)
    
    return qv_by_rank


# Main analysis
if __name__ == "__main__":
    # Import caps
    caps = pd.read_csv("caps_common_1962_2023_ffill.csv")
    
    # Compute log caps
    log_caps = np.log(caps.replace(0, np.nan).values)
    
    # Compute quadratic variation by rank
    print("Computing quadratic variation by rank...")
    max_rank = 100
    qv = compute_qv_by_rank(log_caps, max_rank)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ranks = np.arange(1, max_rank + 1)
    ax.plot(ranks, qv, linewidth=2)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Quadratic Variation')
    ax.set_title('Quadratic Variation by Rank')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Section_5_Rank_Based_Properties/quadratic_variation_by_rank.png',
                dpi=150, bbox_inches='tight')
    print("Plot saved to Section_5_Rank_Based_Properties/quadratic_variation_by_rank.png")
