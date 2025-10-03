"""
Rank Transitions

Python conversion of RankTransitions.R
Analysis of transitions between ranks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_rank_transition_matrix(caps, n_ranks=10):
    """
    Compute transition matrix between ranks
    
    Parameters:
    -----------
    caps : numpy.ndarray
        Capitalizations matrix
    n_ranks : int
        Number of ranks to track
    
    Returns:
    --------
    numpy.ndarray
        Transition probability matrix
    """
    n_stocks, n_times = caps.shape
    transition_counts = np.zeros((n_ranks, n_ranks))
    
    for t in range(1, n_times):
        # Get ranks at t-1 and t
        caps_prev = caps[:, t-1].copy()
        caps_curr = caps[:, t].copy()
        
        caps_prev[np.isnan(caps_prev)] = 0
        caps_curr[np.isnan(caps_curr)] = 0
        
        # Get stock indices sorted by rank
        sorted_prev = np.argsort(-caps_prev)[:n_ranks]
        sorted_curr = np.argsort(-caps_curr)[:n_ranks]
        
        # Create rank mapping
        rank_prev = {stock: i for i, stock in enumerate(sorted_prev)}
        rank_curr = {stock: i for i, stock in enumerate(sorted_curr)}
        
        # Count transitions
        for stock in sorted_prev:
            if stock in rank_curr:
                from_rank = rank_prev[stock]
                to_rank = rank_curr[stock]
                transition_counts[from_rank, to_rank] += 1
    
    # Normalize to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.where(row_sums > 0, transition_counts / row_sums, 0)
    
    return transition_probs


# Main analysis
if __name__ == "__main__":
    # Import caps
    caps = pd.read_csv("caps_common_1962_2023_ffill.csv")
    caps_array = caps.fillna(0).values
    
    # Compute transition matrix
    print("Computing rank transition matrix...")
    n_ranks = 10
    transition_matrix = compute_rank_transition_matrix(caps_array, n_ranks)
    
    # Plot transition matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xlabel('To Rank')
    ax.set_ylabel('From Rank')
    ax.set_title('Rank Transition Probability Matrix')
    ax.set_xticks(np.arange(n_ranks))
    ax.set_yticks(np.arange(n_ranks))
    ax.set_xticklabels(np.arange(1, n_ranks + 1))
    ax.set_yticklabels(np.arange(1, n_ranks + 1))
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Transition Probability')
    
    # Add text annotations
    for i in range(n_ranks):
        for j in range(n_ranks):
            text = ax.text(j, i, f'{transition_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('Section_5_Rank_Based_Properties/rank_transitions.png',
                dpi=150, bbox_inches='tight')
    print("Plot saved to Section_5_Rank_Based_Properties/rank_transitions.png")
    
    # Save matrix to CSV
    transition_df = pd.DataFrame(transition_matrix,
                                 columns=[f'To_Rank_{i+1}' for i in range(n_ranks)],
                                 index=[f'From_Rank_{i+1}' for i in range(n_ranks)])
    transition_df.to_csv('Section_5_Rank_Based_Properties/rank_transition_matrix.csv')
    print("Transition matrix saved to Section_5_Rank_Based_Properties/rank_transition_matrix.csv")
