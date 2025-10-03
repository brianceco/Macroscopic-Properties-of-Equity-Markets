"""
Process Trading Data for Test Dataset

Python conversion of ProcessTradingData_Test_Dataset.R
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('../Backtesting_and_Helper_Functions')
from portfolio_functions_crsp import caps_to_weights


# Import saved CRSP market capitalizations
caps = pd.read_csv("Test_Data/test_caps.csv")

# Import saved CRSP security returns
rets = pd.read_csv("Test_Data/test_rets.csv")

# Import saved CRSP delisting returns
dlrets = pd.read_csv("Test_Data/test_dlrets.csv")

# Import dates
dates = pd.read_csv("Test_Data/test_dates.csv", parse_dates=['dates_test'])
dates = dates['dates_test'].values

# Import permnos
permnos = pd.read_csv("Test_Data/test_permnos.csv")
permnos = permnos.iloc[:, 0].values


def topN_caps(caps, time_idx, N):
    """
    Get indices of top N stocks by capitalization at a given time
    
    Parameters:
    -----------
    caps : pandas.DataFrame
        Capitalizations
    time_idx : int
        Time index
    N : int
        Number of stocks
    
    Returns:
    --------
    tuple
        (indices, caps_N) - indices and caps of top N stocks
    """
    if isinstance(caps, pd.DataFrame):
        caps = caps.values
    
    cap_vals = caps[:, time_idx]
    indices = np.argsort(cap_vals)[::-1][:N]
    caps_N = cap_vals[indices]
    
    return indices, caps_N


# Find indices corresponding to the reduced universe
N = 3
unique_indices = []

for i in range(caps.shape[1]):
    if i % 100 == 1:
        print(f"Processing column {i}")
    
    indices, _ = topN_caps(caps, i, N)
    unique_indices = list(set(unique_indices) | set(indices))

unique_indices = sorted(unique_indices)
print(f"Number of stocks in reduced universe: {len(unique_indices)}")

# Subset Data
caps_trading = caps.iloc[unique_indices, :]
caps_trading = caps_trading.fillna(0)
rets_trading = rets.iloc[unique_indices, :]
rets_trading = rets_trading.fillna(0)
permnos_trading = permnos[unique_indices]

# Create delisting flag
delist_flag = dlrets.iloc[unique_indices, :].notna()

# Modify delisting flag format to be TRUE after stock delists
delist_flag_trading = delist_flag.values
for i in range(delist_flag_trading.shape[0]):
    if i % 100 == 1:
        print(f"Processing delist flag {i}")
    
    delist_indices = np.where(delist_flag_trading[i, :])[0]
    if len(delist_indices) > 0:
        first_delist = delist_indices[0]
        if first_delist < delist_flag_trading.shape[1]:
            delist_flag_trading[i, first_delist:] = True

# Write backtesting data to csv
caps_trading.T.to_csv('Test_Data/BACKTESTING_test_caps.csv', index=False)
rets_trading.T.to_csv('Test_Data/BACKTESTING_test_rets.csv', index=False)
pd.DataFrame(delist_flag_trading.T).to_csv('Test_Data/BACKTESTING_test_dlflg.csv', index=False)
pd.DataFrame({'permnos_trading': permnos_trading}).to_csv('Test_Data/BACKTESTING_test_permnos.csv', index=False)
pd.DataFrame({'dates': dates}).to_csv('Test_Data/BACKTESTING_test_dates.csv', index=False)

print("Backtesting data written to Test_Data/")
