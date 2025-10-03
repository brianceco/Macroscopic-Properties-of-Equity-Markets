"""
Entrants and Exits Analysis

Python conversion of Entrants_and_Exits.R
Code to plot entrances and exits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# Import caps
caps = pd.read_csv("caps_common_1962_2023_ffill.csv")

# Import dates
dates = pd.read_csv("dates_common_1962_2023.csv", parse_dates=['dates'])
dates = dates['dates'].values


def weights_entrants_ts(data, dates, window):
    """
    Determine the weights for entrants to the market
    
    Parameters:
    -----------
    data : pandas.DataFrame or numpy.ndarray
        Data frame of caps with time as columns
    dates : array-like
        Dates for the data
    window : int
        Lookback window (if a stock has entered after being absent for
        at least this number of periods we consider it a true entrance.)
    
    Returns:
    --------
    tuple
        (relevant_weights, relevant_times) - lists of weights and times
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    N = data.shape[1]
    relevant_weights = []
    relevant_times = []
    
    # Loop through columns
    for i in range(N):
        if i % 100 == 0:
            print(dates[i])
        
        if i >= window:
            # Check entrant condition
            logical = None
            for j in range(i - window, i):
                if j == i - window:
                    logical = (data[:, j] <= 0)
                else:
                    logical = (data[:, j] <= 0) & logical
            
            logical = (data[:, i] > 0) & logical
            
            # Identify rows where the criteria is met
            indices = np.where(logical)[0]
            
            # Compute weights
            v = data[:, i].copy()
            v[v <= 0] = 0
            mu = v / np.sum(v)
            weights = mu[indices]
            
            # Retrieve ranks for the identified rows
            relevant_weights.extend(weights)
            relevant_times.extend([i] * len(weights))
    
    return relevant_weights, relevant_times


def weights_exits_ts(data, dates, window):
    """
    Determine the weights for exits from the market
    
    Parameters:
    -----------
    data : pandas.DataFrame or numpy.ndarray
        Data frame of caps with time as columns
    dates : array-like
        Dates for the data
    window : int
        Lookforward window (if a stock has exited for at least
        this number of periods we consider it a true exit.)
    
    Returns:
    --------
    tuple
        (relevant_weights, relevant_times) - lists of weights and times
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    N = data.shape[1]
    relevant_weights = []
    relevant_times = []
    
    # Loop through columns
    for i in range(N - 1):
        if i % 100 == 0:
            print(dates[i])
        
        if i <= N - window - 1:
            # Check exit condition
            logical = None
            for j in range(i + 1, i + window + 1):
                if j == i + 1:
                    logical = (data[:, j] <= 0)
                else:
                    logical = (data[:, j] <= 0) & logical
            
            logical = (data[:, i] > 0) & logical
            
            # Identify rows where the criteria is met
            indices = np.where(logical)[0]
            
            # Compute weights
            v = data[:, i].copy()
            v[v <= 0] = 0
            mu = v / np.sum(v)
            weights = mu[indices]
            
            # Retrieve ranks for the identified rows
            relevant_weights.extend(weights)
            relevant_times.extend([i] * len(weights))
    
    return relevant_weights, relevant_times


# Replace NAs with 0 to be compatible with function
caps = caps.fillna(0)
wind = 60  # lookback/lookforward window length for entrants/exits

# Obtain results
print("Calculating entrants...")
entr_output = weights_entrants_ts(caps, dates, wind)
print(f"Number of entrants: {len(entr_output[0])}")

print("Calculating exits...")
exit_output = weights_exits_ts(caps, dates, wind)
print(f"Number of exits: {len(exit_output[0])}")

# Plot entrant and exit weights on log scale
entr_weights = np.array(entr_output[0])
entr_times = np.array(entr_output[1])
exit_weights = np.array(exit_output[0])
exit_times = np.array(exit_output[1])

max_val = max(np.log10(exit_weights).max(), np.log10(entr_weights).max())
min_val = min(np.log10(exit_weights).min(), np.log10(entr_weights).min())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Entrances
ax1.scatter(dates[entr_times], np.log10(entr_weights),
           c='green', alpha=0.25, s=10, marker='s')
ax1.set_ylim(min_val, max_val)
ax1.set_ylabel('Weights')
ax1.set_xlabel('Date')
ax1.set_title('Entrances')
ax1.set_yticks([-2, -4, -6, -8])
ax1.set_yticklabels(['$10^{-2}$', '$10^{-4}$', '$10^{-6}$', '$10^{-8}$'])
ax1.grid(True, alpha=0.3)

# Exits
ax2.scatter(dates[exit_times], np.log10(exit_weights),
           c='red', alpha=0.25, s=10, marker='s')
ax2.set_ylim(min_val, max_val)
ax2.set_ylabel('Weights')
ax2.set_xlabel('Date')
ax2.set_title('Exits')
ax2.set_yticks([-2, -4, -6, -8])
ax2.set_yticklabels(['$10^{-2}$', '$10^{-4}$', '$10^{-6}$', '$10^{-8}$'])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Section_3_Capital_Distribution_and_Diversity/entrants_exits.png',
            dpi=150, bbox_inches='tight')
print("Plot saved to Section_3_Capital_Distribution_and_Diversity/entrants_exits.png")
