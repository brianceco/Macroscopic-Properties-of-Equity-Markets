"""
CRSP Backtesting Test Data

Python conversion of CRSP_Backtesting_Test_Data.R
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('Backtesting_and_Helper_Functions')
from portfolio_functions_crsp import (
    caps_to_weights, caps_and_rets_to_rr_rd, run_portfolio
)


### IMPORT DATA FOR BACKTESTING

# Import saved CRSP market capitalizations
caps = pd.read_csv("Test_Data/BACKTESTING_test_caps.csv")

# Import saved CRSP security returns
rets = pd.read_csv("Test_Data/BACKTESTING_test_rets.csv")

# Import saved CRSP delisting flag
dlflg = pd.read_csv("Test_Data/BACKTESTING_test_dlflg.csv", dtype=bool)

# Import dates
dates = pd.read_csv("Test_Data/BACKTESTING_test_dates.csv", parse_dates=['dates'])
dates = dates['dates'].values

# Import permnos
permnos = pd.read_csv("Test_Data/BACKTESTING_test_permnos.csv")
permnos = permnos.iloc[:, 0].values


### DEFINE INPUT FUNCTIONS

# Helper Function
def topN(mu, N):
    """
    Get the indices of the top N securities
    
    Parameters:
    -----------
    mu : numpy.ndarray
        Market weights
    N : int
        Number of top securities
    
    Returns:
    --------
    tuple
        (indices, mu_N) - indices and weights of top N securities
    """
    # Get the indices of the top N securities
    indices = np.argsort(mu)[::-1][:N]
    
    # Get the weights for the top N sub-market
    mu_N = mu[indices] / np.sum(mu[indices])
    
    return indices, mu_N


### PORTFOLIO FUNCTIONS
### MODIFY THESE FUNCTIONS TO CHANGE INVESTMENT/BENCHMARK STRATEGIES

# Define portfolio map
def get_pi(mu, dl_flag):
    """
    Get target portfolio weights
    
    Parameters:
    -----------
    mu : numpy.ndarray
        Market weights
    dl_flag : numpy.ndarray
        Delisting flags
    
    Returns:
    --------
    numpy.ndarray
        Portfolio weights
    """
    # Top N stocks to trade
    N = 3
    
    # Initialize id list and portfolio
    ids = np.arange(len(mu))
    pi = np.zeros(len(mu))
    
    # Subset market weights and ids to include stocks that have not defaulted
    mask = ~dl_flag
    admissible_mu = mu[mask]
    admissible_ids = ids[mask]
    
    # Get the (sub) indices of the largest N stocks and their relative weights
    indices, muN = topN(admissible_mu, N)
    
    # Get the original indices corresponding to these stocks
    final_ids = admissible_ids[indices]
    
    # Construct a portfolio on this sub-market
    port = np.full(N, 1/N)  # Equal weight
    # Alternative: port = muN**(1/2) / np.sum(muN**(1/2))  # Diversity portfolio
    
    # Assign the target weights to the stocks
    pi[final_ids] = port
    
    return pi


# Define benchmark portfolio (Current: Index Tracking top N stocks)
def get_pi_benchmark(mu, dl_flag):
    """
    Get benchmark portfolio weights (index tracking)
    
    Parameters:
    -----------
    mu : numpy.ndarray
        Market weights
    dl_flag : numpy.ndarray
        Delisting flags
    
    Returns:
    --------
    numpy.ndarray
        Portfolio weights
    """
    # Top N stocks to trade
    N = 3
    
    # Initialize id list and portfolio
    ids = np.arange(len(mu))
    pi = np.zeros(len(mu))
    
    # Subset market weights and ids to include stocks that have not defaulted
    mask = ~dl_flag
    admissible_mu = mu[mask]
    admissible_ids = ids[mask]
    
    # Get the (sub) indices of the largest N stocks and their relative weights
    indices, muN = topN(admissible_mu, N)
    
    # Get the original indices corresponding to these stocks
    final_ids = admissible_ids[indices]
    
    # Construct a portfolio on this sub-market
    port = muN  # Index tracking top N stocks
    
    # Assign the target weights to the stocks
    pi[final_ids] = port
    
    return pi


### PREPROCESSING
# Convert caps and returns to backtesting inputs
mu = caps_to_weights(caps)  # market weights
real_ret, div_returns = caps_and_rets_to_rr_rd(caps, rets)  # returns (real & dividend)

### RUN BACKTEST
# Initial Backtest Parameters
V0 = 1000  # Initial Wealth
pi0 = get_pi(mu[0, :], dlflg.iloc[0, :].values)  # Initial Portfolio
pi0_bm = get_pi_benchmark(mu[0, :], dlflg.iloc[0, :].values)
tcb = 0.01  # Buying transaction cost
tcs = 0.01  # Selling transaction cost
freq = 3  # Trading frequency
freq_bm = 21

# Get outputs for portfolio
Term_value, Observables = run_portfolio(
    V0, pi0, tcb, tcs, freq, mu, div_returns, real_ret, dlflg, get_pi
)

# Saves the values of all portfolio quantities/statistics in [Ruf & Xie, 2020]
# for all times in the backtest
Observables.to_csv("Test_Data/test_obs.csv", index=False)

# Get outputs for benchmark portfolio
Term_value_bm, Observables_bm = run_portfolio(
    V0, pi0_bm, tcb, tcs, freq_bm, mu, div_returns, real_ret, dlflg, get_pi_benchmark
)

# Log Values
lV = np.log10(Observables['V'].values)
lV_bm = np.log10(Observables_bm['V'].values)

# Plot Log Portfolio Values
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(dates, lV, 'b-', linewidth=2, label='Portfolio')
ax1.plot(dates, lV_bm, 'r-', linewidth=2, label='Benchmark')
ax1.set_ylabel('Log Value')
ax1.set_xlabel('Dates')
ax1.set_ylim(min(np.min(lV), np.min(lV_bm)), max(np.max(lV), np.max(lV_bm)))
ax1.legend(loc='lower right')
ax1.set_title('Portfolio Performance Comparison')
ax1.grid(True, alpha=0.3)

# Plot Log Relative Value over time
ax2.plot(dates, lV - lV_bm, 'k-', linewidth=2)
ax2.set_ylabel('Log Relative Value')
ax2.set_xlabel('Dates')
ax2.set_title('Portfolio Relative Performance')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('Test_Data/backtest_results.png', dpi=150, bbox_inches='tight')
print("Plot saved to Test_Data/backtest_results.png")
print(f"Terminal Portfolio Value: {Term_value:.2f}")
print(f"Terminal Benchmark Value: {Term_value_bm:.2f}")
