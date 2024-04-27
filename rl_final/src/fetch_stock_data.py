import yfinance as yf
import pandas as pd
import numpy as np


def fetch_stock_data(tickers, period='1y'):
    stock_data = yf.download(tickers, period=period)['Adj Close']
    return stock_data

def calculate_sharpe_ratio(portfolio_returns):
    """
    Calculate the Sharpe ratio based on the portfolio returns.
    Args:
    portfolio_returns (list): The portfolio returns
    Returns:
    float: The Sharpe ratio
    """
    expected_return = np.mean(portfolio_returns)
    volatility = np.std(portfolio_returns)
    if volatility == 0:
        return 0
    return expected_return / volatility

def calculate_diversification_penalty(stock_weights, penalty_factor=1.0):
    # Calculate the Herfindahl-Hirschman Index (HHI)
    hhi = np.sum(stock_weights ** 2)
    # The closer the HHI is to 1, the less diversified the portfolio is.
    # We subtract from 1 to reverse this, so that higher values correspond to more diversification.
    diversification = 1 - hhi
    # Apply penalty
    penalty = -penalty_factor * (1 - diversification)
    return penalty

