"""
Data preprocessing module for feature engineering and data cleaning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, List


class DataPreprocessor:
    """Handles data preprocessing and feature engineering."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def add_technical_indicators(self, df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Name of the price column to use
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_10'] = df[price_col].rolling(window=10).mean()
        df['SMA_20'] = df[price_col].rolling(window=20).mean()
        df['SMA_50'] = df[price_col].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df[price_col].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df[price_col].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # RSI (Relative Strength Index)
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df[price_col].rolling(window=20).mean()
        bb_std = df[price_col].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # Volatility
        df['volatility'] = df[price_col].pct_change().rolling(window=20).std()
        
        # Returns
        df['returns'] = df[price_col].pct_change()
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Volume indicators (if Volume column exists)
        if 'Volume' in df.columns:
            df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features for specified columns.
        
        Args:
            df: DataFrame with data
            columns: List of column names to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with added lag features
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def clean_data(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Clean the data by handling missing values and infinities.
        
        Args:
            df: DataFrame to clean
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        if drop_na:
            # Drop rows with any NaN values
            df = df.dropna()
        else:
            # Forward fill then backward fill remaining NaN values
            df = df.ffill().bfill()
        
        return df
    
    def normalize_features(self, 
                          df: pd.DataFrame, 
                          feature_columns: Optional[List[str]] = None,
                          fit: bool = True) -> pd.DataFrame:
        """
        Normalize features using StandardScaler.
        
        Args:
            df: DataFrame with features
            feature_columns: List of columns to normalize. If None, normalizes all numeric columns.
            fit: Whether to fit the scaler or use existing fit
            
        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        
        if feature_columns is None:
            # Select all numeric columns except certain ones
            exclude_cols = ['regime', 'action', 'reward']
            feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col not in exclude_cols]
        
        self.feature_columns = feature_columns
        
        if fit:
            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        else:
            df[feature_columns] = self.scaler.transform(df[feature_columns])
        
        return df
    
    def prepare_for_hmm(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Prepare data for HMM training by selecting and reshaping features.
        
        Args:
            df: DataFrame with features
            features: List of feature column names to use for HMM
            
        Returns:
            Numpy array ready for HMM fitting
        """
        return df[features].values
    
    def split_train_test(self, 
                        df: pd.DataFrame, 
                        train_ratio: float = 0.8) -> tuple:
        """
        Split data into training and testing sets chronologically.
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def preprocess_pipeline(self,
                           df: pd.DataFrame,
                           price_col: str = 'Close',
                           add_lags: bool = True,
                           normalize: bool = True,
                           drop_na: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            price_col: Name of the price column
            add_lags: Whether to add lag features
            normalize: Whether to normalize features
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            Fully preprocessed DataFrame
        """
        # Add technical indicators
        df = self.add_technical_indicators(df, price_col)
        
        # Add lag features if requested
        if add_lags:
            lag_columns = ['returns', 'volatility', 'RSI', 'MACD']
            lags = [1, 2, 3, 5]
            df = self.add_lag_features(df, lag_columns, lags)
        
        # Clean data
        df = self.clean_data(df, drop_na=drop_na)
        
        # Normalize features if requested
        if normalize:
            df = self.normalize_features(df)
        
        return df
