"""
Capital Distribution Analysis

Python conversion of CapitalDistributionAnalysis.R
Code to plot capital distribution curves (full market and top K stocks)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Import caps
caps = pd.read_csv("caps_common_1962_2023_ffill.csv")

# Import dates
dates = pd.read_csv("dates_common_1962_2023.csv", parse_dates=['dates'])
dates = dates['dates'].values

## Illustrate (Full) Capital Distributions in 2D

# Compute (full) market weights
mu = caps.fillna(0).values
mu = mu / mu.sum(axis=0, keepdims=True)

# Define ranks
ranks = np.arange(1, mu.shape[0] + 1)

# Create Plot
M = mu.shape[1]
colors = plt.cm.YlOrRd(np.linspace(0, 1, M))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot full capital distribution
ax1.plot(np.log10(ranks), np.log10(-np.sort(-mu[:, 0])), color=colors[0])
for i in range(0, M, 100):
    sorted_mu = -np.sort(-mu[:, i])
    ax1.plot(np.log10(ranks), np.log10(sorted_mu), color=colors[i], alpha=0.05)

ax1.set_ylim(-9.5, -0.5)
ax1.set_xlim(0, 4)
ax1.set_xlabel('Rank')
ax1.set_ylabel('Weight')
ax1.set_title('Capital distribution curves')
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xticklabels(['1', '10', '100', '1000', '10000'])
ax1.set_yticks([-2, -4, -6, -8])
ax1.set_yticklabels(['$10^{-2}$', '$10^{-4}$', '$10^{-6}$', '$10^{-8}$'])
ax1.grid(True, alpha=0.3)

## Illustrate Top N Capital Distributions in 2D

# Repeat analysis for the market composed of the largest N stocks
N = 1000
mu_N_sorted = caps.fillna(0).values
mu_N_sorted = -np.sort(-mu_N_sorted, axis=0)
mu_N_sorted = mu_N_sorted[:N, :]
mu_N_sorted = mu_N_sorted / mu_N_sorted.sum(axis=0, keepdims=True)

# Ranks corresponding to top N
ranks_N = np.arange(1, N + 1)

# Create Plot for top N
ax2.plot(np.log10(ranks_N), np.log10(-np.sort(-mu_N_sorted[:, 0])), color=colors[0])
for i in range(0, M, 100):
    sorted_mu_N = -np.sort(-mu_N_sorted[:, i])
    ax2.plot(np.log10(ranks_N), np.log10(sorted_mu_N), color=colors[i], alpha=0.05)

ax2.set_ylim(-4.5, -0.5)
ax2.set_xlim(0, 3)
ax2.set_xlabel('Rank')
ax2.set_ylabel('Weight')
ax2.set_title(f'Top {N} Capital distribution')
ax2.set_xticks([0, 1, 2, 3])
ax2.set_xticklabels(['1', '10', '100', '1000'])
ax2.set_yticks([-1, -2, -3, -4])
ax2.set_yticklabels(['$10^{-1}$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$'])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Section_3_Capital_Distribution_and_Diversity/capital_distribution.png',
            dpi=150, bbox_inches='tight')
print("Plot saved to Section_3_Capital_Distribution_and_Diversity/capital_distribution.png")

## Illustrate Top N Capital Distributions in 3D
# Extract subset of dates for plotting
date_ids = np.arange(0, len(dates), 50)
time_vector = dates[date_ids]

# Save log weights and ranks
lmu_N = np.log10(mu_N_sorted[:, date_ids])
lranks_N = np.log10(ranks_N)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for surface plot
X, Y = np.meshgrid(np.arange(len(time_vector)), lranks_N)
Z = lmu_N

# Normalize time for color gradient
time_indices = np.arange(len(time_vector))
time_norm = time_indices / (len(time_vector) - 1)

# Create color array
colors_3d = plt.cm.YlOrRd(np.tile(time_norm, (len(lranks_N), 1)))

surf = ax.plot_surface(X, Y, Z, facecolors=colors_3d, shade=False, alpha=0.8)

ax.set_xlabel('Time Index')
ax.set_ylabel('Rank (log scale)')
ax.set_zlabel('Weight (log scale)')
ax.set_title('Capital Distribution Evolution in 3D')

# Set y-axis ticks
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['1', '10', '100', '1000'])

# Set z-axis ticks
ax.set_zticks([-1, -2, -3, -4])
ax.set_zticklabels(['$10^{-1}$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$'])

plt.tight_layout()
plt.savefig('Section_3_Capital_Distribution_and_Diversity/capital_distribution_3d.png',
            dpi=150, bbox_inches='tight')
print("3D plot saved to Section_3_Capital_Distribution_and_Diversity/capital_distribution_3d.png")
