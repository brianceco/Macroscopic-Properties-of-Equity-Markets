"""
Atlas Model Functions - Portfolio Analysis Functions

Python conversion of AtlasModelFunctions.R
"""

import numpy as np


def atlas_model(n, N, T, gamma, gs, sigmas, X0):
    """
    Simulate from the (generalized) Atlas Model
    
    Parameters:
    -----------
    n : int
        System size
    N : int
        Number of time steps
    T : float
        Terminal time
    gamma : float
        Baseline drift
    gs : array-like
        Atlas drift parameter
    sigmas : array-like
        Volatilities
    X0 : array-like
        Initial values
    
    Returns:
    --------
    numpy.ndarray
        Matrix of shape (n, N+1) with simulated values
    """
    drifts = gamma + gs
    vols = sigmas
    dt = T / N
    X = np.zeros((n, N + 1))
    X[:, 0] = X0
    
    for i in range(N):
        ranks = np.argsort(-X[:, i]) + 1  # ranks (1-indexed in R style)
        # Convert to 0-indexed for Python
        ranks_idx = ranks - 1
        X[:, i + 1] = (X[:, i] + 
                       drifts[ranks_idx] * dt + 
                       vols[ranks_idx] * np.sqrt(dt) * np.random.randn(n))
    
    return X


def ranked_system(X):
    """
    Rank a system of caps (or log caps) where the input matrix has
    columns indexing time and rows indexing the market securities.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Matrix with rows as securities and columns as time
    
    Returns:
    --------
    numpy.ndarray
        Ranked matrix
    """
    N = X.shape[1]
    n = X.shape[0]
    X_R = np.zeros((n, N))
    
    for i in range(N):
        X_R[:, i] = -np.sort(-X[:, i])
    
    return X_R


def returns(X):
    """
    Compute the returns from the log caps
    
    Parameters:
    -----------
    X : numpy.ndarray
        Matrix of log capitalizations
    
    Returns:
    --------
    numpy.ndarray
        Returns matrix
    """
    R = np.diff(X, axis=1)
    return R


def returns_by_rank(X, R):
    """
    Assign returns by the capitalization rank
    
    Parameters:
    -----------
    X : numpy.ndarray
        Matrix of capitalizations
    R : numpy.ndarray
        Returns matrix
    
    Returns:
    --------
    numpy.ndarray
        Returns by rank matrix
    """
    n_cols = X.shape[1]
    ret_by_rank = np.zeros((X.shape[0], n_cols - 1))
    
    for i in range(n_cols - 1):
        r_vals = R[:, i]
        order_idx = np.argsort(-X[:, i])
        ret_by_rank[:, i] = r_vals[order_idx]
    
    return ret_by_rank


def capital_distribution(X):
    """
    Return the capital distribution from a matrix of log capitalizations
    
    Parameters:
    -----------
    X : numpy.ndarray
        Matrix of log capitalizations
    
    Returns:
    --------
    numpy.ndarray
        Capital distribution matrix
    """
    # Compute capitalizations
    caps = np.exp(X)
    
    # Rank the capitalizations
    caps_r = ranked_system(caps)
    
    # Compute the capital distribution
    col_sums = np.sum(caps_r, axis=0)
    cap_dist = caps_r / col_sums[np.newaxis, :]
    
    return cap_dist
