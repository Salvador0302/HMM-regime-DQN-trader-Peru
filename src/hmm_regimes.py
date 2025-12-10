"""
HMM regime detection module using hmmlearn.
Detects market regimes (e.g., bull, bear, sideways) using Gaussian HMM.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Optional, List
import pickle


class HMMRegimeDetector:
    """Detects market regimes using Hidden Markov Models."""
    
    def __init__(self, n_states: int = 3, random_state: int = 42):
        """
        Initialize the HMM regime detector.
        
        Args:
            n_states: Number of hidden states (regimes). Default is 3 (bull, bear, sideways)
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.regime_characteristics = {}
        
    def create_model(self, n_features: int, covariance_type: str = 'full') -> hmm.GaussianHMM:
        """
        Create a Gaussian HMM model.
        
        Args:
            n_features: Number of features in the observation data
            covariance_type: Type of covariance parameters ('full', 'diag', 'tied', 'spherical')
            
        Returns:
            Initialized GaussianHMM model
        """
        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=covariance_type,
            n_iter=100,
            random_state=self.random_state,
            verbose=False
        )
        return model
    
    def fit(self, X: np.ndarray) -> 'HMMRegimeDetector':
        """
        Fit the HMM model to the data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        self.model = self.create_model(n_features=X.shape[1])
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely state sequence for the given observations.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Array of predicted states
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of each state for the given observations.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Array of state probabilities of shape (n_samples, n_states)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def characterize_regimes(self, X: np.ndarray, returns: np.ndarray) -> dict:
        """
        Characterize each regime based on the mean returns and volatility.
        
        Args:
            X: Feature matrix used for regime detection
            returns: Array of returns corresponding to the observations
            
        Returns:
            Dictionary with regime characteristics
        """
        if self.model is None:
            raise ValueError("Model must be fitted before characterization. Call fit() first.")
        
        states = self.predict(X)
        
        self.regime_characteristics = {}
        
        for state in range(self.n_states):
            state_mask = states == state
            state_returns = returns[state_mask]
            
            self.regime_characteristics[state] = {
                'mean_return': np.mean(state_returns),
                'volatility': np.std(state_returns),
                'count': np.sum(state_mask),
                'percentage': np.sum(state_mask) / len(states) * 100
            }
        
        # Label regimes based on mean returns
        sorted_states = sorted(self.regime_characteristics.items(), 
                              key=lambda x: x[1]['mean_return'], 
                              reverse=True)
        
        # Assign intuitive labels
        regime_labels = {
            sorted_states[0][0]: 'bull',      # Highest mean return
            sorted_states[-1][0]: 'bear',     # Lowest mean return
            sorted_states[1][0] if len(sorted_states) > 2 else sorted_states[0][0]: 'sideways'
        }
        
        for state, label in regime_labels.items():
            self.regime_characteristics[state]['label'] = label
        
        return self.regime_characteristics
    
    def get_current_regime(self, X: np.ndarray) -> tuple:
        """
        Get the current regime (last observation).
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (state_id, state_label, state_probability)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Get the last observation
        last_obs = X[-1:, :]
        
        # Predict state
        state = self.predict(last_obs)[0]
        
        # Get state probability
        state_probs = self.predict_proba(last_obs)[0]
        
        # Get label if available
        label = self.regime_characteristics.get(state, {}).get('label', f'state_{state}')
        
        return state, label, state_probs[state]
    
    def add_regimes_to_dataframe(self, df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
        """
        Add regime predictions to a DataFrame.
        
        Args:
            df: DataFrame to add regimes to
            X: Feature matrix used for regime detection
            
        Returns:
            DataFrame with added regime columns
        """
        df = df.copy()
        
        # Predict regimes
        states = self.predict(X)
        state_probs = self.predict_proba(X)
        
        # Add to dataframe
        df['regime'] = states
        
        # Add probabilities for each state
        for i in range(self.n_states):
            df[f'regime_prob_{i}'] = state_probs[:, i]
        
        # Add regime labels if available
        if self.regime_characteristics:
            df['regime_label'] = df['regime'].map(
                {k: v['label'] for k, v in self.regime_characteristics.items()}
            )
        
        return df
    
    def save_model(self, filepath: str):
        """
        Save the trained HMM model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'n_states': self.n_states,
            'random_state': self.random_state,
            'regime_characteristics': self.regime_characteristics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """
        Load a trained HMM model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.n_states = model_data['n_states']
        self.random_state = model_data['random_state']
        self.regime_characteristics = model_data['regime_characteristics']
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the state transition matrix.
        
        Returns:
            Transition matrix of shape (n_states, n_states)
        """
        if self.model is None:
            raise ValueError("Model must be fitted first.")
        
        return self.model.transmat_
    
    def print_regime_summary(self):
        """Print a summary of the detected regimes."""
        if not self.regime_characteristics:
            print("No regime characteristics available. Run characterize_regimes() first.")
            return
        
        print("\n" + "="*60)
        print("REGIME SUMMARY")
        print("="*60)
        
        for state, chars in sorted(self.regime_characteristics.items()):
            label = chars.get('label', f'State {state}')
            print(f"\n{label.upper()} (State {state}):")
            print(f"  Mean Return: {chars['mean_return']:.4f}")
            print(f"  Volatility:  {chars['volatility']:.4f}")
            print(f"  Occurrences: {chars['count']} ({chars['percentage']:.2f}%)")
        
        print("\n" + "="*60)
        print("TRANSITION MATRIX:")
        print("="*60)
        trans_matrix = self.get_transition_matrix()
        print(trans_matrix)
        print("="*60 + "\n")
