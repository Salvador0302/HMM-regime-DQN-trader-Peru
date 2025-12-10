"""
Unit tests for data ingestion module.
"""

import unittest
import pandas as pd
import numpy as np
from src.ingest import DataIngestor


class TestDataIngestor(unittest.TestCase):
    """Test cases for DataIngestor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ingestor = DataIngestor()
    
    def test_initialization(self):
        """Test DataIngestor initialization."""
        self.assertIsNotNone(self.ingestor)
    
    def test_fetch_yahoo_finance(self):
        """Test Yahoo Finance data fetching."""
        data = self.ingestor.fetch_yahoo_finance(
            tickers=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
    
    def test_get_sentiment_placeholder(self):
        """Test sentiment placeholder generation."""
        sentiment = self.ingestor.get_sentiment_placeholder(
            ticker='AAPL',
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        self.assertIsInstance(sentiment, pd.DataFrame)
        self.assertIn('sentiment_score', sentiment.columns)
        self.assertIn('sentiment_volume', sentiment.columns)


if __name__ == '__main__':
    unittest.main()
