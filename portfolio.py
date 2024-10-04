"""
Portfolio Optimization 
"""
# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
from scipy.optimize import minimize  # For portfolio optimization

def objective(weights, returns, cov_matrix, target_return):
    """
    Objective function to minimize for portfolio optimization.

    Args:
        weights (numpy.ndarray): An array of portfolio weights.
        returns (pandas.DataFrame): A DataFrame of daily returns.
        cov_matrix (numpy.ndarray): The covariance matrix of the returns.
        target_return (float): The target daily return.

    Returns:
        float: The portfolio variance.
    """
    # Calculate the portfolio return
    portfolio_return = np.sum(returns.mean() * weights)
    
    # Calculate the portfolio variance
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Penalize for not meeting target return
    penalty = abs(portfolio_return - target_return)

    # Return the sum of portfolio variance and penalty
    return portfolio_variance + penalty


def optimize_portfolio(returns, target_return, risk_free_rate=0.0):
    """
    Optimize the portfolio weights to maximize the Sharpe ratio.

    Args:
        returns (pandas.DataFrame): A DataFrame of daily returns.
        target_return (float): The target daily return.
        risk_free_rate (float): The daily risk-free rate.

    Returns:
        dict: A dictionary containing the optimal portfolio weights and Sharpe ratio.
    """
    # Calculate the covariance matrix of returns
    cov_matrix = returns.cov()

    # Get the number of assets in the portfolio
    num_assets = len(returns.columns)

    # Define the constraint: sum of weights must equal 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    # Define bounds: each weight must be between 0 and 1
    bounds = tuple((0, 1) for i in range(num_assets))

    # Set initial weights to equal allocation
    initial_weights = num_assets * [1.0 / num_assets]

    # Perform the optimization
    optimized = minimize(objective, initial_weights, args=(returns, cov_matrix, target_return),
                         method='SLSQP', bounds=bounds, constraints=constraints)

    # Extract the optimized weights
    weights = optimized.x

    # Calculate the portfolio return using optimized weights
    portfolio_return = np.sum(returns.mean() * weights)

    # Calculate the portfolio standard deviation (volatility)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Calculate the Sharpe ratio
    # The Sharpe ratio is (portfolio return - risk-free rate) / portfolio standard deviation
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

    # Return the results as a dictionary
    return {
        'weights': weights,
        'return': portfolio_return,
        'volatility': portfolio_std_dev,
        'sharpe_ratio': sharpe_ratio
    }