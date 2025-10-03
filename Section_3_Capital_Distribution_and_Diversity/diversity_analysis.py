"""
Diversity Analysis

Python conversion of DiversityAnalysis.R
Analysis of market diversity metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def diversity_index(weights):
    """
    Compute diversity index (effective number of components)
    
    Parameters:
    -----------
    weights : numpy.ndarray
        Array of weights (should sum to 1)
    
    Returns:
    --------
    float
        Diversity index
    """
    weights = weights[weights > 0]  # Remove zero weights
    if len(weights) == 0:
        return 0
    return np.exp(-np.sum(weights * np.log(weights)))


def compute_diversity_time_series(caps):
    """
    Compute diversity index over time
    
    Parameters:
    -----------
    caps : pandas.DataFrame or numpy.ndarray
        Market capitalizations
    
    Returns:
    --------
    numpy.ndarray
        Diversity index time series
    """
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
    
    # Compute diversity
    print("Computing diversity time series...")
    diversity = compute_diversity_time_series(caps)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, diversity, linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Diversity Index')
    ax.set_title('Market Diversity Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Section_3_Capital_Distribution_and_Diversity/diversity_analysis.png',
                dpi=150, bbox_inches='tight')
    print("Plot saved to Section_3_Capital_Distribution_and_Diversity/diversity_analysis.png")
    
    # Save diversity to CSV
    diversity_df = pd.DataFrame({'dates': dates, 'diversity': diversity})
    diversity_df.to_csv('Section_3_Capital_Distribution_and_Diversity/diversity.csv', index=False)
    print("Diversity data saved to Section_3_Capital_Distribution_and_Diversity/diversity.csv")
