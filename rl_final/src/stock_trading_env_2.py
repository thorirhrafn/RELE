import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

from src.fetch_stock_data import calculate_sharpe_ratio, calculate_diversification_penalty
from src.calculations import *

# utility function sem á eftir að implementa
def utility_function(x, a=1.0):
    return - np.exp(-a * x) / a

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, initial_cash=10000, risk_aversion=1.0, inflation_rate=0.001, logging=False, log_frequency=100, seed=None):
        super(StockTradingEnv, self).__init__()

        self.stock_data = stock_data
        if isinstance(self.stock_data, pd.Series):
            self.stock_data = self.stock_data.to_frame()

        self.inflation_rate = inflation_rate
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.risk_aversion = risk_aversion
        self.portfolio_returns = []

        self.logging = logging
        self.log_frequency = log_frequency

        num_stocks = len(self.stock_data.columns)
        # self.action_space = spaces.Box(low=0, high=1, shape=(num_stocks + 1,), dtype=np.float32)
        # self.action_space.seed(seed)

        num_discrete_actions = 101  # 0.00, 0.01, ..., 1.00
        self.action_space = spaces.MultiDiscrete([num_discrete_actions] * (num_stocks + 1))
        #self.action_space.seed(seed)

        observation_low = [0] * (2 * num_stocks + 2)
        observation_high = [np.inf] * (2 * num_stocks + 2)
        self.observation_space = spaces.Box(low=np.array(observation_low), high=np.array(observation_high), dtype=np.float32)

        self.window_size = 14  # A commonly used window size
        self.current_step = self.window_size  # Initialize to window size to avoid NaNs
        self.reset()

    def reset(self):
        self.window_size = 14  # A commonly used window size
        self.current_step = self.window_size  # Initialize to window size to avoid NaNs
        self.current_cash = self.initial_cash
        self.current_stock_prices = self.stock_data.iloc[self.current_step].values
        self.current_stock_held = np.zeros(len(self.stock_data.columns))
        portfolio_value = self.current_cash + np.sum(self.current_stock_held * self.current_stock_prices)
        self.portfolio_value = portfolio_value
        return self._get_observation()


    def step(self, action):
        if self.logging and (self.current_step % self.log_frequency == 0):
            logging.info(f"=== STEP {self.current_step + 1} ===")

        action = action / 100.0
        epsilon = 1e-8
        
        # Normalize action vector (ensure it sums to 1)
        action = action / (np.sum(action) + epsilon)

        # Extract the cash weight and the stock weights from the action vector
        cash_weight = action[0]
        stock_weights = action[1:]

        # Calculate new portfolio value
        new_stock_prices = self.stock_data.iloc[self.current_step + 1].values
        new_portfolio_value = self.current_cash + np.sum(self.current_stock_held * new_stock_prices)

        # Update cash in portfolio after accounting for inflation
        inflation_penalty = self.current_cash * self.inflation_rate
        new_portfolio_value -= inflation_penalty

        # Calculate the new amount of cash and stocks held in the portfolio
        cash_for_stocks = new_portfolio_value * (1 - cash_weight)
        self.current_cash = new_portfolio_value * cash_weight

        # Make sure new_stock_prices and stock_weights have the same shape
        if len(new_stock_prices) != len(stock_weights):
            raise ValueError(f"Shape mismatch: new_stock_prices {len(new_stock_prices)} vs stock_weights {len(stock_weights)}")

        new_stock_held = (cash_for_stocks * stock_weights) / new_stock_prices


        self.current_stock_held = new_stock_held
        self.current_stock_prices = new_stock_prices

        # Calculate the reward using the Sharpe ratio
        portfolio_return = new_portfolio_value - self.portfolio_value
        self.portfolio_returns.append(portfolio_return)
        sharpe_ratio = calculate_sharpe_ratio(self.portfolio_returns)
        
        reward = portfolio_return * sharpe_ratio  # Sharpe ratio as a multiplier
        reward += calculate_diversification_penalty(stock_weights, penalty_factor=10)

        self.portfolio_value = new_portfolio_value

        self.current_step += 1
        done = self.current_step == len(self.stock_data) - 1 - self.window_size

        if self.logging and (self.current_step % self.log_frequency == 0):
            logging.info(f"=== STEP {self.current_step + 1} ===")
            logging.info(f"Cash Weight: {cash_weight}, Renormalized Stock Weights: {stock_weights}")
            logging.info(f"New Stock Prices: {new_stock_prices}, Portfolio Value Before: {new_portfolio_value}")
            logging.info(f"Cash for Stocks: {cash_for_stocks}, Updated Cash: {self.current_cash}")
            logging.info(f"New Stock Held: {new_stock_held}, Reward: {reward}")

        return self._get_observation(), reward, done, {}


    # def _get_observation(self):
    #     portfolio_value = self.current_cash + np.sum(self.current_stock_held * self.current_stock_prices)
    #     moving_avg = np.mean(self.stock_data.iloc[max(0, self.current_step - 10):self.current_step + 1], axis=0)
    #     return [portfolio_value, self.current_cash] + list(moving_avg) + list(self.current_stock_held)
    #     #return [portfolio_value, self.current_cash] + list(self.current_stock_prices) + list(self.current_stock_held)

    def _get_observation(self):
        window_size = 14  # A commonly used window size
        stock_data_window = self.stock_data.iloc[max(0, self.current_step - window_size):self.current_step + 1]
        
        moving_avg = calculate_moving_average(stock_data_window, window_size)
        volatility = calculate_volatility(stock_data_window, window_size)
        momentum = calculate_momentum(stock_data_window, window_size)
        rsi = calculate_rsi(stock_data_window, window_size)
        
        # Include any other features, such as Alpha, Beta, Volume, etc.
        # ...
        
        portfolio_value = self.current_cash + np.sum(self.current_stock_held * self.current_stock_prices)
        
        return np.concatenate([
            [portfolio_value, self.current_cash],
        #    self.current_stock_prices,
            self.current_stock_held,
            moving_avg.iloc[-1].values if len(moving_avg) > 0 else np.zeros_like(self.current_stock_prices),
            volatility.iloc[-1].values if len(volatility) > 0 else np.zeros_like(self.current_stock_prices),
            momentum.iloc[-1].values if len(momentum) > 0 else np.zeros_like(self.current_stock_prices),
            rsi.iloc[-1].values if len(rsi) > 0 else np.zeros_like(self.current_stock_prices),
        ])
    
    def get_state_size(self):
        sample_state = self._get_observation()
        return len(sample_state)


    def render(self):
        pass
