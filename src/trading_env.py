"""
Gymnasium-based trading environment for reinforcement learning.
Implements Buy/Sell/Hold actions with transaction fees.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple


class TradingEnv(gym.Env):
    """
    A trading environment for reinforcement learning.
    
    Actions:
        0: Hold
        1: Buy
        2: Sell
    
    Observation Space:
        Box containing market features and current position information
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 transaction_fee: float = 0.001,
                 max_position: int = 1,
                 window_size: int = 20,
                 feature_columns: Optional[list] = None):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with market data and features
            initial_balance: Initial cash balance
            transaction_fee: Transaction fee as a fraction (0.001 = 0.1%)
            max_position: Maximum number of shares to hold (1 for simple long-only)
            window_size: Number of past observations to include in state
            feature_columns: List of feature column names to use. If None, uses all numeric columns.
        """
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position = max_position
        self.window_size = window_size
        
        # Determine feature columns
        if feature_columns is None:
            self.feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns]
        else:
            self.feature_columns = feature_columns
        
        self.n_features = len(self.feature_columns)
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features + position + cash + portfolio value
        obs_size = self.n_features * self.window_size + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.portfolio_value = initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        
        # Track history
        self.portfolio_history = []
        self.trade_history = []
        
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            Numpy array representing the current state
        """
        # Get the window of historical features
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        window_data = self.data.loc[start_idx:end_idx-1, self.feature_columns].values
        
        # Pad if necessary
        if len(window_data) < self.window_size:
            padding = np.zeros((self.window_size - len(window_data), self.n_features))
            window_data = np.vstack([padding, window_data])
        
        # Flatten the window
        features = window_data.flatten()
        
        # Add position information (normalized)
        position_info = np.array([
            self.position / self.max_position,
            self.balance / self.initial_balance,
            self.portfolio_value / self.initial_balance
        ])
        
        # Combine all features
        observation = np.concatenate([features, position_info]).astype(np.float32)
        
        return observation
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        current_price = self.data.loc[self.current_step, 'Close'] if 'Close' in self.data.columns else self.data.loc[self.current_step, self.feature_columns[0]]
        return self.balance + self.position * current_price
    
    def _execute_trade(self, action: int) -> float:
        """
        Execute a trade action and return the reward.
        
        Args:
            action: Trading action (0=Hold, 1=Buy, 2=Sell)
            
        Returns:
            Reward for the action
        """
        current_price = self.data.loc[self.current_step, 'Close'] if 'Close' in self.data.columns else self.data.loc[self.current_step, self.feature_columns[0]]
        reward = 0.0
        
        if action == 1:  # Buy
            if self.position < self.max_position and self.balance > current_price:
                # Buy one share
                cost = current_price * (1 + self.transaction_fee)
                if cost <= self.balance:
                    self.balance -= cost
                    self.position += 1
                    self.entry_price = current_price
                    self.total_trades += 1
                    self.trade_history.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'price': current_price,
                        'balance': self.balance,
                        'position': self.position
                    })
                    reward = -self.transaction_fee  # Small penalty for transaction fee
        
        elif action == 2:  # Sell
            if self.position > 0:
                # Sell one share
                proceeds = current_price * (1 - self.transaction_fee)
                self.balance += proceeds
                profit = current_price - self.entry_price
                self.position -= 1
                
                if profit > 0:
                    self.winning_trades += 1
                
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': current_price,
                    'balance': self.balance,
                    'position': self.position,
                    'profit': profit
                })
                
                # Reward based on profit
                reward = profit / self.entry_price  # Percentage profit
        
        else:  # Hold
            # Small penalty for holding to encourage action
            reward = -0.0001
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Trading action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Execute the trade
        trade_reward = self._execute_trade(action)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        self.portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history.append(self.portfolio_value)
        
        # Calculate reward based on portfolio value change
        if len(self.portfolio_history) > 1:
            value_change = (self.portfolio_value - self.portfolio_history[-2]) / self.portfolio_history[-2]
            reward = value_change + trade_reward
        else:
            reward = trade_reward
        
        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'step': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options
            
        Returns:
            Tuple of (initial_observation, info)
        """
        super().reset(seed=seed)
        
        self.current_step = self.window_size - 1  # Start after we have a full window
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.portfolio_history = [self.initial_balance]
        self.trade_history = []
        
        observation = self._get_observation()
        info = {'portfolio_value': self.portfolio_value}
        
        return observation, info
    
    def render(self):
        """Render the environment state."""
        if len(self.portfolio_history) > 0:
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {self.position}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Winning Trades: {self.winning_trades}")
            print("-" * 50)
    
    def get_portfolio_returns(self) -> float:
        """Calculate total portfolio returns."""
        return (self.portfolio_value - self.initial_balance) / self.initial_balance
