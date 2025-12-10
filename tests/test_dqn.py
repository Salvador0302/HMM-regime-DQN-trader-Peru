"""
Unit tests for DQN agent.
"""

import unittest
import numpy as np
import pandas as pd
from src.dqn import DQNAgent, ReplayBuffer
from src.trading_env import TradingEnv


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(self.buffer.size(), 0)
    
    def test_add(self):
        """Test adding experiences."""
        state = np.array([1, 2, 3])
        self.buffer.add(state, 0, 1.0, state, False)
        
        self.assertEqual(self.buffer.size(), 1)
    
    def test_sample(self):
        """Test sampling from buffer."""
        for i in range(10):
            state = np.array([i, i+1, i+2])
            self.buffer.add(state, 0, 1.0, state, False)
        
        states, actions, rewards, next_states, dones = self.buffer.sample(5)
        
        self.assertEqual(len(states), 5)
        self.assertEqual(len(actions), 5)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = DQNAgent(
            state_size=10,
            action_size=3,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.state_size, 10)
        self.assertEqual(self.agent.action_size, 3)
        self.assertIsNotNone(self.agent.model)
        self.assertIsNotNone(self.agent.target_model)
    
    def test_act(self):
        """Test action selection."""
        state = np.random.randn(10)
        action = self.agent.act(state, training=False)
        
        self.assertIn(action, [0, 1, 2])
    
    def test_remember(self):
        """Test experience storage."""
        state = np.random.randn(10)
        self.agent.remember(state, 0, 1.0, state, False)
        
        self.assertEqual(self.agent.replay_buffer.size(), 1)
    
    def test_update_target_model(self):
        """Test target model update."""
        # Change main model weights
        original_weights = self.agent.target_model.get_weights()
        self.agent.model.set_weights([w + 0.1 for w in original_weights])
        
        # Update target
        self.agent.update_target_model()
        
        # Check that target weights match main model
        new_weights = self.agent.target_model.get_weights()
        main_weights = self.agent.model.get_weights()
        
        for tw, mw in zip(new_weights, main_weights):
            np.testing.assert_array_almost_equal(tw, mw)


if __name__ == '__main__':
    unittest.main()
