"""
Make Test Dataset

Python conversion of Make_Test_Dataset.R
Create synthetic test data for backtesting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../Backtesting_and_Helper_Functions')
from atlas_model_functions import atlas_model


np.random.seed(456)

# Initialize Atlas Model Parameters
n = 5  # number of securities
N = 20  # number of time steps
T = 21/252  # terminal time
gamma = 0.1  # Atlas baseline drift
gs = ((-np.arange(n, 0, -1) + (n+1)/2) / ((n+1)/2)) * 0.1  # Atlas drifts (by rank)
sigmas = (1 + 2*np.arange(0, n) / (n-1)) * np.sqrt(0.1)  # volatilities (by rank)
times = np.linspace(0, T, N+1)  # time vector

# Simulate
X0 = 1 + np.random.rand(n) / n
X = atlas_model(n, N, T, gamma, gs, sigmas, X0)

# Compute caps and returns
caps = np.exp(X).T
rets = np.vstack([np.full(n, np.nan), caps[1:(N+1), :] / caps[0:N, :] - 1])

# Dates
dates_test = [
    "1962-01-02", "1962-01-03", "1962-01-04", "1962-01-05", "1962-01-08",
    "1962-01-09", "1962-01-10", "1962-01-11", "1962-01-12", "1962-01-15",
    "1962-01-16", "1962-01-17", "1962-01-18", "1962-01-19", "1962-01-22",
    "1962-01-23", "1962-01-24", "1962-01-25", "1962-01-26", "1962-01-29",
    "1962-01-30"
]
dates_test = pd.to_datetime(dates_test)

# Create fake permnos
permnos_test = np.array([101, 102, 103, 104, 105])

# Delisting return at time 9 for stock 1
dlret_test = np.full((N+1, 5), np.nan)
dlret_test[8, 0] = -0.1  # Time index 8 = day 9 (0-indexed)

# Stock 5 inactive for first 10 days
caps[0:10, 4] = np.nan
rets[0:11, 4] = np.nan

# Stock 1 delisting after day 9
caps[9:21, 0] = np.nan
rets[9:21, 0] = np.nan

# Add Dividend Return for Stock 3 at times 5 and 15 and Stock 5 at time 15
rets[4, 2] = rets[4, 2] + 0.03  # Time index 4 = day 5
rets[14, 2] = rets[14, 2] + 0.03  # Time index 14 = day 15
rets[14, 4] = rets[14, 4] + 0.04

# Plot Caps Data
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
colors = plt.cm.coolwarm(np.linspace(0, 1, n))

# Plot 1: Caps
ax = axes[0]
for i in range(n):
    ax.plot(dates_test, caps[:, i], color=colors[i], label=f'Stock {i+1}')
ax.set_ylabel('Caps')
ax.set_xlabel('Dates')
ax.set_ylim(2.4, 4.4)
ax.set_title('Market Capitalizations')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Returns
ax = axes[1]
for i in range(n):
    ax.scatter(dates_test, rets[:, i], color=colors[i], label=f'Stock {i+1}', alpha=0.7)
ax.set_ylabel('Return')
ax.set_xlabel('Dates')
ax.set_ylim(-0.17, 0.17)
ax.set_title('Returns')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Plot 3: Delisting Returns
ax = axes[2]
for i in range(n):
    ax.scatter(dates_test, dlret_test[:, i], color=colors[i], label=f'Stock {i+1}', alpha=0.7)
ax.set_ylabel('Delisting Return')
ax.set_xlabel('Dates')
ax.set_ylim(-0.17, 0.17)
ax.set_title('Delisting Returns')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('Test_Data/test_data_plots.png', dpi=150, bbox_inches='tight')
print("Plot saved to Test_Data/test_data_plots.png")

# Write test data to csv
caps_df = pd.DataFrame(caps.T)
rets_df = pd.DataFrame(rets.T)
dlret_df = pd.DataFrame(dlret_test.T)
dates_df = pd.DataFrame({'dates_test': dates_test})
permnos_df = pd.DataFrame({'permnos_test': permnos_test})

caps_df.to_csv('Test_Data/test_caps.csv', index=False)
rets_df.to_csv('Test_Data/test_rets.csv', index=False)
dlret_df.to_csv('Test_Data/test_dlrets.csv', index=False)
dates_df.to_csv('Test_Data/test_dates.csv', index=False)
permnos_df.to_csv('Test_Data/test_permnos.csv', index=False)

print("Test data written to Test_Data/")
