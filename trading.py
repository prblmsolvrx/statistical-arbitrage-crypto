# Import necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import scipy.optimize as sco  # For portfolio optimization

def mean_reversion(df, window):
    """Calculate the mean reversion signal for a given window."""
    # Calculate z-score: (current value - rolling mean) / rolling standard deviation
    zscore = (df - df.rolling(window).mean()) / df.rolling(window).std()
    # Return negative z-score as the mean reversion signal
    return -zscore

def momentum(df, window):
    """Calculate the momentum signal for a given window."""
    # Calculate percentage change of the dataframe
    returns = df.pct_change()
    # Return the rolling sum of returns for the given window
    return returns.rolling(window).sum()

def combine_signals(signals):
    """Combine multiple trading signals by taking their average."""
    # Calculate the average of all signals
    combined_signal = sum(signals) / len(signals)
    # Return the sign of the combined signal (-1, 0, or 1)
    return np.sign(combined_signal)

def backtest_portfolio(returns, positions, transaction_cost):
    """Backtest a portfolio of positions."""
    # Calculate portfolio returns considering transaction costs
    portfolio_returns = (positions.shift(1) * returns) - (np.abs(positions - positions.shift(1)) * transaction_cost)

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Calculate drawdowns
    previous_peaks = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - previous_peaks) / previous_peaks

    # Calculate Sharpe ratio (annualized)
    sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
    
    # Calculate maximum drawdown
    max_drawdown = drawdowns.min()

    return cumulative_returns, sharpe_ratio, max_drawdown

def optimize_portfolio(returns):
    """Optimize the portfolio weights to maximize the Sharpe ratio."""
    # Define the objective function to minimize (negative Sharpe ratio)
    def objective_function(weights, returns):
        portfolio_returns = np.dot(returns, weights)
        sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        return -sharpe_ratio

    # Define constraints: sum of weights must equal 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    # Define bounds: each weight must be between 0 and 1
    bounds = [(0, 1) for i in range(len(returns.columns))]

    # Initialize weights equally
    weights = np.ones(len(returns.columns)) / len(returns.columns)

    # Optimize weights using Sequential Least Squares Programming (SLSQP) method
    optimized_weights = sco.minimize(objective_function, weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

    # Return the optimized weights
    return optimized_weights.x