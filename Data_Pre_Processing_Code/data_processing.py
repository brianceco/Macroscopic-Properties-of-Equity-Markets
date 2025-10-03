"""
Data Processing Script

Python conversion of DataProcessing.R
Process raw CRSP data from WRDS
"""

import pandas as pd
import numpy as np


# Import Raw CRSP data from WRDS (see README document)
data = pd.read_csv(
    "Data/CRSP_common_1962_2023.csv",
    parse_dates=['date'],
    dtype={'PERMNO': float, 'DLRET': float, 'PRC': float, 'RET': float, 'SHROUT': float}
)

print(data.head())  # Preview

## Prices
prices = data.pivot(index='PERMNO', columns='date', values='PRC')
prices = prices.sort_index(axis=1)  # Order prices by date
prices = prices.abs()  # Remove the negative sign from Bid-Ask average prices

permnos = prices.index.values  # Extract permnos
dates = prices.columns.values  # Extract dates

print(prices.iloc[:4, :4])  # Preview

## Shares
shares = data.pivot(index='PERMNO', columns='date', values='SHROUT')
shares = shares.sort_index(axis=1)  # Order shares by date

print(shares.iloc[:4, :4])  # Preview

## Market Caps
caps = prices * shares  # Compute market capitalizations

print(caps.iloc[:4, :4])  # Preview

caps.to_csv('Data/caps_common_1962_2023.csv', index=False)  # Write caps to csv

## Returns
returns = data.pivot(index='PERMNO', columns='date', values='RET')
returns = returns.sort_index(axis=1)  # Order returns by date

print(returns.iloc[:4, :4])  # Preview

returns.to_csv('Data/rets_common_1962_2023.csv', index=False)  # Write returns to csv

## Delisting Returns
dlreturns = data.pivot(index='PERMNO', columns='date', values='DLRET')
dlreturns = dlreturns.sort_index(axis=1)  # Order delisting returns by date

print(dlreturns.iloc[:4, :4])  # Preview

dlreturns.to_csv('Data/dlrets_common_1962_2023.csv', index=False)  # Write delisting returns to csv

## Dates
dates_df = pd.DataFrame({'dates': dates})
dates_df.to_csv('Data/dates_common_1962_2023.csv', index=False)  # Write dates to csv

## Permnos
permnos_df = pd.DataFrame({'permnos': permnos})
permnos_df.to_csv('Data/permnos_common_1962_2023.csv', index=False)  # Write permnos to csv
