"""
Portfolio Functions for Backtesting - CRSP

Python conversion of PortfolioFunctions_CRSP.R

A NOTE ON CONVENTIONS:
Securities are indexed by the columns and days/time by the rows
We use an End of Day (EoD) convention for data:
- Returns are EoD returns
- Caps/Weights are EoD values
Portfolios are updated at EoD:
- Portfolio initialized at EoD values at time t
- First relevant return is at time t+1
- New trade time is s>=t+1
- Last relevant return is at time s
- New portfolio weights determined by observables at time s
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def caps_to_weights(caps):
    """
    Calculate mu matrix (market weights)
    
    Parameters:
    -----------
    caps : numpy.ndarray or pandas.DataFrame
        Market capitalizations
    
    Returns:
    --------
    numpy.ndarray
        Market weights matrix
    """
    if isinstance(caps, pd.DataFrame):
        caps = caps.values
    
    row_sums = np.sum(caps, axis=1, keepdims=True)
    mu = caps / row_sums
    return mu


def caps_and_rets_to_rr_rd(caps, rets):
    """
    Computes real and dividend returns from CRSP caps and rets
    
    Parameters:
    -----------
    caps : numpy.ndarray or pandas.DataFrame
        Market capitalizations
    rets : numpy.ndarray or pandas.DataFrame
        Returns
    
    Returns:
    --------
    tuple
        (real_ret, div_returns) - tuple of numpy arrays
    """
    if isinstance(caps, pd.DataFrame):
        caps = caps.values
    if isinstance(rets, pd.DataFrame):
        rets = rets.values
    
    cap_ratio = caps.copy()
    cap_ratio[0, :] = 1.0
    cap_ratio[1:, :] = caps[1:, :] / caps[:-1, :]
    
    div_returns = 1 + rets - cap_ratio
    div_returns[div_returns < 0] = 0
    div_returns[np.isnan(div_returns)] = 0
    real_ret = rets - div_returns
    
    # Correct for when both caps and rets are 0 (when a stock is not listed)
    mask = (caps == 0) & (rets == 0)
    div_returns[mask] = 0
    real_ret[mask] = 0
    
    return real_ret, div_returns


def trade_decision(days_since_trade, freq):
    """
    Trade decision function
    
    Parameters:
    -----------
    days_since_trade : int
        Days since last trade
    freq : int
        Target trading frequency
    
    Returns:
    --------
    bool
        Whether to trade
    """
    return days_since_trade == freq


def calc_psi_minus_and_D_minus(rr, rd, psi):
    """
    Calculate Psi_ and D_
    
    Parameters:
    -----------
    rr : numpy.ndarray
        Real returns (row vector)
    rd : numpy.ndarray
        Dividend returns (row vector)
    psi : numpy.ndarray
        Portfolio positions
    
    Returns:
    --------
    tuple
        (psi_, D_) - updated positions and dividends
    """
    rrp1 = 1 + rr
    psi_ = rrp1 * psi
    D_ = np.sum(psi * rd)
    return psi_, D_


def calc_D_hat(D_, V_, pi_target, psi_, tcs):
    """
    Calculate D_hat
    
    Parameters:
    -----------
    D_ : float
        Uninvested dividends
    V_ : float
        Portfolio value
    pi_target : numpy.ndarray
        Target portfolio weights
    psi_ : numpy.ndarray
        Current positions
    tcs : float
        Selling transaction cost
    
    Returns:
    --------
    float
        D_hat value
    """
    psi_sold = np.zeros(len(psi_))
    psi_sold[pi_target == 0] = 1
    D_hat = (D_ + (1 - tcs) * np.sum(psi_ * psi_sold)) / V_
    return D_hat


def get_c_vector(pi, pi_target):
    """
    Get the c vector values
    
    Parameters:
    -----------
    pi : numpy.ndarray
        Current portfolio weights
    pi_target : numpy.ndarray
        Target portfolio weights
    
    Returns:
    --------
    numpy.ndarray
        c vector
    """
    c = pi / pi_target
    c[pi_target == 0] = 0
    return c


def c_function(c, c_vec, D_hat, pi_target, tcb, tcs):
    """
    Function to solve for c
    
    Parameters:
    -----------
    c : float
        Value to optimize
    c_vec : numpy.ndarray
        c vector
    D_hat : float
        D_hat value
    pi_target : numpy.ndarray
        Target portfolio weights
    tcb : float
        Buying transaction cost
    tcs : float
        Selling transaction cost
    
    Returns:
    --------
    float
        Absolute value of the objective
    """
    c_plus_1 = c - c_vec
    c_plus_1[c_plus_1 <= 0] = 0
    c_plus_2 = c_vec - c
    c_plus_2[c_plus_2 <= 0] = 0
    c_out = (1 + tcb) * np.sum(c_plus_1 * pi_target) - (1 - tcs) * np.sum(c_plus_2 * pi_target) - D_hat
    return abs(c_out)


def run_portfolio(V0, pi0, tcb, tcs, freq, mu, div_returns, real_ret, dl_flag, get_pi):
    """
    Run portfolio on data
    
    Parameters:
    -----------
    V0 : float
        Initial wealth
    pi0 : numpy.ndarray
        Initial portfolio weights
    tcb : float
        Buying transaction cost
    tcs : float
        Selling transaction cost
    freq : int
        Trading frequency
    mu : numpy.ndarray
        Market weights
    div_returns : numpy.ndarray
        Dividend returns
    real_ret : numpy.ndarray
        Real returns
    dl_flag : numpy.ndarray
        Delisting flag
    get_pi : callable
        Function to get target portfolio
    
    Returns:
    --------
    tuple
        (terminal_value, values_dataframe)
    """
    if isinstance(mu, pd.DataFrame):
        mu = mu.values
    if isinstance(div_returns, pd.DataFrame):
        div_returns = div_returns.values
    if isinstance(real_ret, pd.DataFrame):
        real_ret = real_ret.values
    if isinstance(dl_flag, pd.DataFrame):
        dl_flag = dl_flag.values
    
    NS = mu.shape[1]  # Number of stocks
    Nper = mu.shape[0]  # Number of periods
    
    # Set up Values matrix to record data
    col_names_values = (
        [f"pi_{i}" for i in range(NS)] +
        [f"psi_{i}" for i in range(NS)] +
        ["D_"] +
        [f"pi_target_{i}" for i in range(NS)] +
        ["V_", "D_hat"] +
        [f"c_vec_{i}" for i in range(NS)] +
        ["c", "V"] +
        [f"psi_{i}_final" for i in range(NS)] +
        [f"pi_{i}_final" for i in range(NS)]
    )
    size_out = len(col_names_values)
    Values = np.zeros((Nper, size_out))
    
    # Initialize Variables
    V = V0
    V_ = V0
    pi = pi0.copy()
    psi = pi0 * V0
    psi_ = pi0 * V0
    D_ = 0  # Initial uninvested dividends
    idx = 0
    days_since_trade = 1
    
    # Initial values
    Values[0, :NS] = pi
    Values[0, NS:2*NS] = psi_
    Values[0, 2*NS] = D_
    Values[0, 2*NS+1:3*NS+1] = pi
    Values[0, 3*NS+1] = V_
    Values[0, 3*NS+2] = np.nan
    Values[0, 3*NS+3:4*NS+3] = np.nan
    Values[0, 4*NS+3] = np.nan
    Values[0, 4*NS+4] = V
    Values[0, 4*NS+5:5*NS+5] = psi
    Values[0, 5*NS+5:] = pi
    
    print("Backtest Running...")
    
    # Perform the trading until the end of the data set
    while idx < (Nper - 1):
        idx += 1
        
        if idx % max(1, int(np.ceil(Nper / 100))) == 1:
            print(f"{round(100 * idx / Nper)}% Complete.")
        
        # Get returns
        rr = real_ret[idx, :]
        rd = div_returns[idx, :]
        
        # Update psi_ and D_
        psi_, D_increment = calc_psi_minus_and_D_minus(rr, rd, psi)
        D_ = D_ + D_increment
        
        # Update V_
        V_ = np.sum(psi_)
        
        # Get current pi
        pi = psi_ / V_
        
        trade = trade_decision(days_since_trade, freq)
        
        if trade:
            # Get target portfolio
            pi_target = get_pi(mu[idx, :], dl_flag[idx, :])
            
            # Get Dhat
            D_hat = calc_D_hat(D_, V_, pi_target, psi_, tcs)
            
            # Get vector of c_i
            c_vec = get_c_vector(pi, pi_target)
            
            # Solve for value of C
            res = minimize_scalar(
                c_function,
                args=(c_vec, D_hat, pi_target, tcb, tcs),
                bounds=(0, 2),
                method='bounded'
            )
            c = res.x
            
            # Update V, psi and pi
            V = c * V_
            psi = pi_target * V
            
            # Store values
            Values[idx, :NS] = pi
            Values[idx, NS:2*NS] = psi_
            Values[idx, 2*NS] = D_
            Values[idx, 2*NS+1:3*NS+1] = pi_target
            Values[idx, 3*NS+1] = V_
            Values[idx, 3*NS+2] = D_hat
            Values[idx, 3*NS+3:4*NS+3] = c_vec
            Values[idx, 4*NS+3] = c
            Values[idx, 4*NS+4] = V
            Values[idx, 4*NS+5:5*NS+5] = psi
            Values[idx, 5*NS+5:] = pi_target
            
            # Reset uninvested dividends
            D_ = 0
            
            days_since_trade = 1
        else:
            # Update values
            V = V_ + D_
            psi = psi_.copy()
            pi_target = np.full(NS, np.nan)
            D_hat = np.nan
            c = np.nan
            c_vec = np.full(NS, np.nan)
            
            # Store values
            Values[idx, :NS] = pi
            Values[idx, NS:2*NS] = psi_
            Values[idx, 2*NS] = D_
            Values[idx, 2*NS+1:3*NS+1] = pi_target
            Values[idx, 3*NS+1] = V_
            Values[idx, 3*NS+2] = D_hat
            Values[idx, 3*NS+3:4*NS+3] = c_vec
            Values[idx, 4*NS+3] = c
            Values[idx, 4*NS+4] = V
            Values[idx, 4*NS+5:5*NS+5] = psi
            Values[idx, 5*NS+5:] = pi
            
            days_since_trade += 1
    
    print("Backtest Complete.")
    
    # Create DataFrame with column names
    values_df = pd.DataFrame(Values, columns=col_names_values)
    
    return V, values_df
