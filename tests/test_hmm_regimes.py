"""
Unit tests for HMM regime detection module.
"""

import unittest
import numpy as np
import pandas as pd
from src.hmm_regimes import HMMRegimeDetector


class TestHMMRegimeDetector(unittest.TestCase):
    """Test cases for HMMRegimeDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = HMMRegimeDetector(n_states=3, random_state=42)
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.returns = np.random.randn(100) * 0.01
    
    def test_initialization(self):
        """Test HMMRegimeDetector initialization."""
        self.assertEqual(self.detector.n_states, 3)
        self.assertEqual(self.detector.random_state, 42)
    
    def test_fit(self):
        """Test model fitting."""
        self.detector.fit(self.X)
        
        self.assertIsNotNone(self.detector.model)
    
    def test_predict(self):
        """Test state prediction."""
        self.detector.fit(self.X)
        states = self.detector.predict(self.X)
        
        self.assertEqual(len(states), len(self.X))
        self.assertTrue(all(s in range(3) for s in states))
    
    def test_characterize_regimes(self):
        """Test regime characterization."""
        self.detector.fit(self.X)
        characteristics = self.detector.characterize_regimes(self.X, self.returns)
        
        self.assertEqual(len(characteristics), 3)
        for state, chars in characteristics.items():
            self.assertIn('mean_return', chars)
            self.assertIn('volatility', chars)
            self.assertIn('label', chars)


if __name__ == '__main__':
    unittest.main()
