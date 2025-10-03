"""
Rank Trajectory Analysis

Python conversion of RankTrajectory.R
Analysis of rank trajectories of stocks over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_rank_trajectory(caps, stock_indices):
    """
    Get rank trajectory for specified stocks
    
    Parameters:
    -----------
    caps : numpy.ndarray
        Capitalizations matrix
    stock_indices : list
        List of stock indices to track
    
    Returns:
    --------
    numpy.ndarray
        Rank trajectories for each stock
    """
    n_times = caps.shape[1]
    n_stocks = len(stock_indices)
    ranks = np.zeros((n_stocks, n_times))
    
    for t in range(n_times):
        cap_t = caps[:, t].copy()
        # Rank stocks (1 = highest cap)
        sorted_indices = np.argsort(-cap_t)
        rank_array = np.empty_like(sorted_indices)
        rank_array[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
        
        for i, stock_idx in enumerate(stock_indices):
            ranks[i, t] = rank_array[stock_idx]
    
    return ranks


# Main analysis
if __name__ == "__main__":
    # Import caps
    caps = pd.read_csv("caps_common_1962_2023_ffill.csv")
    
    # Import dates
    dates = pd.read_csv("dates_common_1962_2023.csv", parse_dates=['dates'])
    dates = dates['dates'].values
    
    # Replace NAs with 0
    caps_array = caps.fillna(0).values
    
    # Select some stocks to track (e.g., top 10 at start)
    initial_ranks = np.argsort(-caps_array[:, 0])[:10]
    
    print("Computing rank trajectories...")
    ranks = get_rank_trajectory(caps_array, initial_ranks)
    
    # Plot rank trajectories
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in range(len(initial_ranks)):
        ax.plot(dates, ranks[i, :], alpha=0.7, label=f'Stock {initial_ranks[i]}')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Rank')
    ax.set_title('Rank Trajectories of Top 10 Initial Stocks')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Section_3_Capital_Distribution_and_Diversity/rank_trajectory.png',
                dpi=150, bbox_inches='tight')
    print("Plot saved to Section_3_Capital_Distribution_and_Diversity/rank_trajectory.png")
