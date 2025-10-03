"""
Data Processing Test Script

Python conversion of DataProcessing_Test.R
Code to check/test pre-processing output for AAPL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Import CRSP data for AAPL
AAPL = pd.read_csv(
    "Data/AAPL_data.csv",
    parse_dates=['date'],
    dtype={'PERMNO': float, 'DLRET': float, 'PRC': float, 'RET': float, 'SHROUT': float}
)

# Import saved CRSP market capitalizations
caps = pd.read_csv("Data/caps_common_1962_2023.csv")

# Import saved CRSP security returns
rets = pd.read_csv("Data/rets_common_1962_2023.csv")

# Import saved CRSP delisting returns
dlrets = pd.read_csv("Data/dlrets_common_1962_2023.csv")

# Import dates
dates = pd.read_csv("Data/dates_common_1962_2023.csv", parse_dates=['dates'])
dates = dates['dates'].values

# Import permnos
permnos = pd.read_csv("Data/permnos_common_1962_2023.csv")
permnos = permnos['permnos'].values

# Preview
print(AAPL.head())
print(caps.iloc[:4, :4])
print(rets.iloc[:4, :4])
print(dlrets.iloc[:4, :4])
print(dates[:5])
print(permnos[:5])

# Get identifying information for AAPL in main data set
permno_AAPL = AAPL['PERMNO'].iloc[0]
col_mask = np.isin(dates, AAPL['date'].values)
row_id = np.where(permnos == permno_AAPL)[0][0]

# Get rets and caps directly
cap_AAPL = np.abs(AAPL['PRC']) * AAPL['SHROUT']
ret_AAPL = AAPL['RET'].values
dlret_AAPL = AAPL['DLRET'].values

# Get rets and caps from data set
verify_cap_AAPL = caps.iloc[row_id, col_mask].values
verify_ret_AAPL = rets.iloc[row_id, col_mask].values
verify_dlret_AAPL = dlrets.iloc[row_id, col_mask].values

# Check if all values are equal
print("Caps equal:", np.allclose(verify_cap_AAPL, cap_AAPL, equal_nan=True))
print("Returns equal:", np.allclose(verify_ret_AAPL, ret_AAPL, equal_nan=True))
print("Delisting returns equal:", np.allclose(verify_dlret_AAPL, dlret_AAPL, equal_nan=True))

# Plot caps and returns (Note: No delisting returns for AAPL)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(AAPL['date'], cap_AAPL, 'b-', label='Original')
ax1.plot(AAPL['date'], verify_cap_AAPL, 'r-', label='Verified', alpha=0.7)
ax1.set_ylabel('Market Cap')
ax1.set_title('AAPL Market Capitalization')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(AAPL['date'], ret_AAPL, 'b-', label='Original')
ax2.plot(AAPL['date'], verify_ret_AAPL, 'r-', label='Verified', alpha=0.7)
ax2.set_ylabel('Returns')
ax2.set_xlabel('Date')
ax2.set_title('AAPL Returns')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Data_Pre_Processing_Code/aapl_verification.png', dpi=150, bbox_inches='tight')
print("Plot saved to Data_Pre_Processing_Code/aapl_verification.png")
