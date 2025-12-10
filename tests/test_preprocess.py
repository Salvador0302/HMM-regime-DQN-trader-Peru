"""
Unit tests for data preprocessing module.
"""

import unittest
import pandas as pd
import numpy as np
from src.preprocess import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        self.assertIsNotNone(self.preprocessor)
    
    def test_add_technical_indicators(self):
        """Test technical indicator creation."""
        result = self.preprocessor.add_technical_indicators(self.sample_data, price_col='Close')
        
        # Check that indicators were added
        self.assertIn('SMA_10', result.columns)
        self.assertIn('RSI', result.columns)
        self.assertIn('MACD', result.columns)
        self.assertIn('returns', result.columns)
        self.assertIn('volatility', result.columns)
    
    def test_clean_data(self):
        """Test data cleaning."""
        # Add some NaN values
        dirty_data = self.sample_data.copy()
        dirty_data.loc[dirty_data.index[5], 'Close'] = np.nan
        
        cleaned = self.preprocessor.clean_data(dirty_data, drop_na=True)
        
        # Check that NaN values were handled
        self.assertFalse(cleaned.isnull().any().any())
    
    def test_split_train_test(self):
        """Test train/test split."""
        train, test = self.preprocessor.split_train_test(self.sample_data, train_ratio=0.8)
        
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)


if __name__ == '__main__':
    unittest.main()
