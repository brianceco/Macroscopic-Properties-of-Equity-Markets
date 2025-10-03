"""
Additional Data Processing

Python conversion of AdditionalDataProcessing.R
Code to eliminate dates where no security is traded and
create a file where capitalization data is forward filled
after a security is first listed and up to its delisting.
"""

import pandas as pd
import numpy as np


# Import saved CRSP market capitalizations
caps = pd.read_csv("Data/caps_common_1962_2023.csv")

# Import saved CRSP security returns
rets = pd.read_csv("Data/rets_common_1962_2023.csv")

# Import saved CRSP delisting returns
dlrets = pd.read_csv("Data/dlrets_common_1962_2023.csv")

# Import dates
dates = pd.read_csv("Data/dates_common_1962_2023.csv", parse_dates=['dates'])
dates = dates['dates'].values

## Remove dates with no data

# Identify columns with only NA entries
na_columns = caps.isna().all(axis=0)

# Check how many cases of only NA entries
print(f"Number of columns with only NAs: {na_columns.sum()}")

# Remove columns from data based on identified indices
caps = caps.loc[:, ~na_columns]
rets = rets.loc[:, ~na_columns]
dlrets = dlrets.loc[:, ~na_columns]
dates = dates[~na_columns.values]

# Save edited data
caps.to_csv('Data/caps_common_1962_2023.csv', index=False)
rets.to_csv('Data/rets_common_1962_2023.csv', index=False)
dlrets.to_csv('Data/dlrets_common_1962_2023.csv', index=False)
dates_df = pd.DataFrame({'dates': dates})
dates_df.to_csv('Data/dates_common_1962_2023.csv', index=False)


## Forward fill capitalization data between listing/delisting events

def forward_fill_interior(x):
    """
    Function to forward fill interior NAs in a vector
    
    Parameters:
    -----------
    x : pandas.Series or numpy.ndarray
        Input vector
    
    Returns:
    --------
    numpy.ndarray
        Forward filled vector
    """
    x = np.array(x)
    non_na_indices = np.where(~np.isnan(x))[0]
    
    if len(non_na_indices) == 0:
        return x
    
    first_non_na = non_na_indices[0]
    last_non_na = non_na_indices[-1]
    
    # Check if there is at least one non-NA value and that the first
    # and last non-NA values are not the same. Then, forward fill in between.
    if first_non_na != last_non_na:
        for i in range(first_non_na + 1, last_non_na + 1):
            if np.isnan(x[i]):
                x[i] = x[i - 1]
    
    return x


# Apply the forward fill operation to each row
caps_ffill = caps.apply(forward_fill_interior, axis=1)

# Convert back to a DataFrame
caps_ffill = pd.DataFrame(caps_ffill.tolist(), columns=caps.columns)

caps_ffill.to_csv('Data/caps_common_1962_2023_ffill.csv', index=False)
print("Forward filled caps saved to Data/caps_common_1962_2023_ffill.csv")
