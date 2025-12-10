"""
Data ingestion module for fetching market data from multiple sources.
Supports yfinance, FRED API, Excel files, and sentiment analysis placeholder.
"""

import pandas as pd
import yfinance as yf
from fredapi import Fred
from typing import Optional, List
import os


class DataIngestor:
    """Handles data ingestion from multiple sources."""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the data ingestor.
        
        Args:
            fred_api_key: API key for FRED (Federal Reserve Economic Data).
                         If None, will attempt to read from FRED_API_KEY env variable.
        """
        self.fred_api_key = fred_api_key or os.environ.get('FRED_API_KEY')
        self.fred = Fred(api_key=self.fred_api_key) if self.fred_api_key else None
    
    def fetch_yahoo_finance(self, 
                           tickers: List[str], 
                           start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'GOOGL'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data for the requested tickers
        """
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        return data
    
    def fetch_fred_data(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch economic data from FRED.
        
        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE', 'DFF')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Series with the requested economic indicator
        """
        if not self.fred:
            raise ValueError("FRED API key not provided. Set FRED_API_KEY environment variable.")
        
        data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        return data
    
    def load_excel_data(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to load. If None, loads the first sheet.
            
        Returns:
            DataFrame with the Excel data
        """
        return pd.read_excel(file_path, sheet_name=sheet_name)
    
    def get_sentiment_placeholder(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Placeholder for sentiment analysis data.
        
        In a production system, this would fetch sentiment scores from news APIs,
        social media, or other sentiment data providers.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with mock sentiment scores
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_data = pd.DataFrame({
            'date': date_range,
            'sentiment_score': 0.0,  # Neutral sentiment placeholder
            'sentiment_volume': 0    # Volume of sentiment mentions
        })
        sentiment_data.set_index('date', inplace=True)
        return sentiment_data
    
    def combine_data_sources(self,
                            ticker: str,
                            start_date: str,
                            end_date: str,
                            fred_series: Optional[List[str]] = None,
                            excel_path: Optional[str] = None) -> pd.DataFrame:
        """
        Combine data from multiple sources into a single DataFrame.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            fred_series: List of FRED series IDs to include
            excel_path: Path to Excel file with additional data
            
        Returns:
            Combined DataFrame with all data sources
        """
        # Fetch Yahoo Finance data
        price_data = self.fetch_yahoo_finance([ticker], start_date, end_date)
        
        # If multi-level columns, flatten them
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data.columns = ['_'.join(col).strip() for col in price_data.columns.values]
        
        # Fetch sentiment data
        sentiment_data = self.get_sentiment_placeholder(ticker, start_date, end_date)
        
        # Combine with price data
        combined_data = price_data.join(sentiment_data, how='left')
        
        # Add FRED economic indicators if requested
        if fred_series and self.fred:
            for series_id in fred_series:
                try:
                    fred_data = self.fetch_fred_data(series_id, start_date, end_date)
                    combined_data = combined_data.join(fred_data.rename(f'fred_{series_id}'), how='left')
                except Exception as e:
                    print(f"Warning: Could not fetch FRED series {series_id}: {e}")
        
        # Add Excel data if provided
        if excel_path and os.path.exists(excel_path):
            try:
                excel_data = self.load_excel_data(excel_path)
                if 'date' in excel_data.columns:
                    excel_data.set_index('date', inplace=True)
                combined_data = combined_data.join(excel_data, how='left')
            except Exception as e:
                print(f"Warning: Could not load Excel data from {excel_path}: {e}")
        
        # Forward fill missing values for economic indicators
        combined_data = combined_data.ffill()
        
        return combined_data
