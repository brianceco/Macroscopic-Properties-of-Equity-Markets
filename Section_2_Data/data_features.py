"""
Data Features Analysis

Python conversion of DataFeatures.R
Extract and plot annual summary features of the CRSP data set
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def count_top_percent_mkt_cap(mat, percent):
    """
    Function to compute the minimum number of securities needed to
    account for a percentage of the total market capitalization.
    
    Parameters:
    -----------
    mat : numpy.ndarray
        Capitalization time series data
    percent : float
        Desired percentage (e.g., 0.95 for 95%)
    
    Returns:
    --------
    numpy.ndarray
        Array with [number_to_exceed_threshold, mkt_cap] for each time period
    """
    total_days = mat.shape[1]
    results = np.zeros((2, total_days))
    
    for i in range(total_days):
        cap_vals = mat[:, i].copy()
        cap_vals[cap_vals < 0] = 0
        cap_vals_sorted = np.sort(cap_vals)[::-1]
        mktcap = np.sum(cap_vals_sorted)
        
        if mktcap > 0:
            weights = cap_vals_sorted / mktcap
            cum_weight = np.cumsum(weights)
            is_below_per = (cum_weight < percent)
            results[0, i] = np.sum(is_below_per) + 1
        else:
            results[0, i] = 0
        
        results[1, i] = mktcap
    
    return results


def get_market_stats(stock_matrix):
    """
    Function that returns summary statistics about market data.
    
    Parameters:
    -----------
    stock_matrix : numpy.ndarray
        Matrix of capitalizations or stock prices for each day
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with daily number of active, newly listed and delisted stocks
    """
    if isinstance(stock_matrix, pd.DataFrame):
        stock_matrix = stock_matrix.values
    
    # Replace NAs with 0s
    stock_matrix = np.nan_to_num(stock_matrix, 0)
    
    num_days = stock_matrix.shape[1]
    stocks_traded = np.zeros(num_days, dtype=int)
    stocks_stopped = np.zeros(num_days, dtype=int)
    stocks_began = np.zeros(num_days, dtype=int)
    
    for i in range(1, num_days):
        # Get traded stocks
        stocks_traded[i] = np.sum(stock_matrix[:, i] > 0)
        
        # Get stopped stocks
        stopped_stocks = (stock_matrix[:, i-1] > 0) & (stock_matrix[:, i] <= 0)
        stocks_stopped[i] = np.sum(stopped_stocks)
        
        # Get started stocks
        began_stocks = (stock_matrix[:, i-1] <= 0) & (stock_matrix[:, i] > 0)
        stocks_began[i] = np.sum(began_stocks)
    
    # Make a DataFrame with market stats (excluding first day)
    market_stats = pd.DataFrame({
        'Day': np.arange(2, num_days + 1),
        'Traded': stocks_traded[1:],
        'Stopped': stocks_stopped[1:],
        'Began': stocks_began[1:]
    })
    
    return market_stats


def sum_events_every_k_days(events, k):
    """
    Function to aggregate a vector that contains daily event counts.
    
    Parameters:
    -----------
    events : numpy.ndarray
        Vector of event counts per day
    k : int
        Aggregation window (k days)
    
    Returns:
    --------
    numpy.ndarray
        Vector of total events every k days
    """
    remainder = len(events) % k
    
    # If not a multiple of k, pad with zeros
    if remainder != 0:
        events = np.concatenate([events, np.zeros(k - remainder)])
    
    # Reshape and sum
    reshaped = events.reshape(-1, k)
    summed = np.sum(reshaped, axis=1)
    
    return summed


# Main analysis
if __name__ == "__main__":
    # Import caps
    caps = pd.read_csv("Data/caps_common_1962_2023_ffill.csv")
    
    # Import dates
    dates = pd.read_csv("Data/dates_common_1962_2023.csv", parse_dates=['dates'])
    dates = dates['dates'].values
    
    # Replace NAs with 0
    caps = caps.fillna(0)
    
    # Get market stats
    print("Computing market statistics...")
    market_stats = get_market_stats(caps)
    
    # Count top percent market cap
    print("Computing top percent market cap...")
    percent = 0.95  # 95%
    results = count_top_percent_mkt_cap(caps.values, percent)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of traded stocks
    axes[0, 0].plot(dates[1:], market_stats['Traded'])
    axes[0, 0].set_title('Number of Traded Stocks Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Stocks')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Entries and Exits
    axes[0, 1].plot(dates[1:], market_stats['Began'], label='Entries', alpha=0.7)
    axes[0, 1].plot(dates[1:], market_stats['Stopped'], label='Exits', alpha=0.7)
    axes[0, 1].set_title('Market Entries and Exits')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Top N stocks to cover 95% market cap
    axes[1, 0].plot(dates, results[0, :])
    axes[1, 0].set_title(f'Number of Stocks to Cover {percent*100}% of Market Cap')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Number of Stocks')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Total market cap
    axes[1, 1].plot(dates, results[1, :])
    axes[1, 1].set_title('Total Market Capitalization')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Market Cap')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Section_2_Data/data_features.png', dpi=150, bbox_inches='tight')
    print("Plot saved to Section_2_Data/data_features.png")
    
    # Save market stats to CSV
    market_stats.to_csv('Section_2_Data/market_stats.csv', index=False)
    print("Market stats saved to Section_2_Data/market_stats.csv")
