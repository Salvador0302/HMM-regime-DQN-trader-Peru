"""
Unit tests for trading environment.
"""

import unittest
import numpy as np
import pandas as pd
from src.trading_env import TradingEnv


class TestTradingEnv(unittest.TestCase):
    """Test cases for TradingEnv class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Open': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100),
            'returns': np.random.randn(100) * 0.01
        }, index=dates)
        
        self.env = TradingEnv(
            data=self.sample_data,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10
        )
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.initial_balance, 10000)
        self.assertEqual(self.env.transaction_fee, 0.001)
        self.assertEqual(self.env.action_space.n, 3)
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(self.env.balance, 10000)
        self.assertEqual(self.env.position, 0)
    
    def test_step(self):
        """Test environment step."""
        obs, info = self.env.reset()
        
        # Take a step (hold action)
        next_obs, reward, terminated, truncated, info = self.env.step(0)
        
        self.assertIsInstance(next_obs, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
    
    def test_buy_action(self):
        """Test buy action."""
        obs, info = self.env.reset()
        initial_balance = self.env.balance
        
        # Execute buy action
        next_obs, reward, terminated, truncated, info = self.env.step(1)
        
        # Balance should decrease (if trade executed)
        if self.env.position > 0:
            self.assertLess(self.env.balance, initial_balance)
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        obs, info = self.env.reset()
        initial_value = self.env._calculate_portfolio_value()
        
        self.assertEqual(initial_value, 10000)


if __name__ == '__main__':
    unittest.main()
