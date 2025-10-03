"""
Create Backtesting Outputs - Frequency

Python conversion of Create_Backtesting_Outputs_Freq.R
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from portfolio_functions_crsp import run_portfolio, caps_to_weights, caps_and_rets_to_rr_rd


if __name__ == "__main__":
    print("Template for creating backtesting outputs by frequency")
    print("Customize this script based on your portfolio strategies")
