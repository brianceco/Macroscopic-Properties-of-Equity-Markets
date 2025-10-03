"""
Local Times Analysis

Python conversion of LocalTimes.R
Analysis of local times (time spent at each rank)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_local_times(caps, max_rank=100):
    """
    Compute local times for each rank
    
    Parameters:
    -----------
    caps : numpy.ndarray
        Capitalizations matrix
    max_rank : int
        Maximum rank to consider
    
    Returns:
    --------
    numpy.ndarray
        Local times for each rank
    """
    n_times = caps.shape[1]
    local_times = np.zeros(max_rank)
    
    for t in range(n_times):
        cap_t = caps[:, t].copy()
        cap_t[np.isnan(cap_t)] = 0
        # Rank stocks
        sorted_indices = np.argsort(-cap_t)[:max_rank]
        for rank in range(min(max_rank, len(sorted_indices))):
            local_times[rank] += 1
    
    # Normalize by total time
    local_times = local_times / n_times
    
    return local_times


# Main analysis
if __name__ == "__main__":
    # Import caps
    caps = pd.read_csv("caps_common_1962_2023_ffill.csv")
    
    # Compute local times
    print("Computing local times...")
    max_rank = 100
    caps_array = caps.fillna(0).values
    local_times = compute_local_times(caps_array, max_rank)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ranks = np.arange(1, max_rank + 1)
    ax.plot(ranks, local_times, linewidth=2)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Local Time (Fraction)')
    ax.set_title('Local Times by Rank')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Section_5_Rank_Based_Properties/local_times.png',
                dpi=150, bbox_inches='tight')
    print("Plot saved to Section_5_Rank_Based_Properties/local_times.png")
