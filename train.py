"""
Training script for DQN trading agent with HMM regime detection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.ingest import DataIngestor
from src.preprocess import DataPreprocessor
from src.hmm_regimes import HMMRegimeDetector
from src.trading_env import TradingEnv
from src.dqn import DQNAgent


def train_agent(ticker: str = 'AAPL',
                start_date: str = '2020-01-01',
                end_date: str = '2023-12-31',
                n_episodes: int = 100,
                use_hmm: bool = True,
                save_models: bool = True):
    """
    Train a DQN agent for trading.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        n_episodes: Number of training episodes
        use_hmm: Whether to use HMM regime detection
        save_models: Whether to save trained models
    """
    print(f"\n{'='*70}")
    print(f"Training DQN Agent {'WITH' if use_hmm else 'WITHOUT'} HMM Regime Detection")
    print(f"Ticker: {ticker} | Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")
    
    # 1. Ingest data
    print("Step 1: Ingesting data...")
    ingestor = DataIngestor()
    raw_data = ingestor.fetch_yahoo_finance([ticker], start_date, end_date)
    
    # Flatten multi-level columns if present
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = ['_'.join(col).strip('_') for col in raw_data.columns.values]
    
    print(f"  ✓ Fetched {len(raw_data)} rows of data")
    
    # 2. Preprocess data
    print("\nStep 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Find the Close column (handle various naming conventions)
    close_col = None
    for col in raw_data.columns:
        if 'close' in col.lower():
            close_col = col
            break
    
    if close_col is None:
        raise ValueError("Could not find Close price column in data")
    
    # Add technical indicators
    processed_data = preprocessor.add_technical_indicators(raw_data, price_col=close_col)
    
    # Clean data
    processed_data = preprocessor.clean_data(processed_data, drop_na=True)
    
    print(f"  ✓ Created {len(processed_data.columns)} features")
    print(f"  ✓ {len(processed_data)} rows after cleaning")
    
    # 3. Detect HMM regimes (if enabled)
    if use_hmm:
        print("\nStep 3: Detecting HMM regimes...")
        hmm_detector = HMMRegimeDetector(n_states=3, random_state=42)
        
        # Select features for HMM
        hmm_features = ['returns', 'volatility', 'RSI', 'MACD']
        hmm_features = [f for f in hmm_features if f in processed_data.columns]
        
        # Prepare data for HMM
        X_hmm = preprocessor.prepare_for_hmm(processed_data, hmm_features)
        
        # Fit HMM
        hmm_detector.fit(X_hmm)
        
        # Characterize regimes
        returns = processed_data['returns'].values
        hmm_detector.characterize_regimes(X_hmm, returns)
        hmm_detector.print_regime_summary()
        
        # Add regimes to data
        processed_data = hmm_detector.add_regimes_to_dataframe(processed_data, X_hmm)
        
        if save_models:
            os.makedirs('models', exist_ok=True)
            hmm_detector.save_model(f'models/hmm_model_{ticker}.pkl')
            print(f"  ✓ Saved HMM model to models/hmm_model_{ticker}.pkl")
    else:
        print("\nStep 3: Skipping HMM regime detection (use_hmm=False)")
    
    # 4. Split data into train/test
    print("\nStep 4: Splitting data...")
    train_data, test_data = preprocessor.split_train_test(processed_data, train_ratio=0.8)
    print(f"  ✓ Training set: {len(train_data)} rows")
    print(f"  ✓ Test set: {len(test_data)} rows")
    
    # 5. Create trading environment
    print("\nStep 5: Creating trading environment...")
    
    # Select features for the environment (exclude some columns)
    exclude_cols = ['regime', 'regime_label'] + [col for col in processed_data.columns if 'regime_prob' in col]
    feature_cols = [col for col in processed_data.columns if col not in exclude_cols]
    
    # Normalize features
    train_data_norm = preprocessor.normalize_features(train_data, feature_columns=feature_cols, fit=True)
    test_data_norm = preprocessor.normalize_features(test_data, feature_columns=feature_cols, fit=False)
    
    env_train = TradingEnv(
        data=train_data_norm,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=20,
        feature_columns=feature_cols
    )
    
    env_test = TradingEnv(
        data=test_data_norm,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=20,
        feature_columns=feature_cols
    )
    
    print(f"  ✓ State size: {env_train.observation_space.shape[0]}")
    print(f"  ✓ Action size: {env_train.action_space.n}")
    
    # 6. Create DQN agent
    print("\nStep 6: Creating DQN agent...")
    agent = DQNAgent(
        state_size=env_train.observation_space.shape[0],
        action_size=env_train.action_space.n,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=32,
        target_update_freq=10
    )
    print("  ✓ DQN agent initialized")
    
    # 7. Train the agent
    print(f"\nStep 7: Training for {n_episodes} episodes...")
    print("-" * 70)
    
    episode_rewards = []
    episode_portfolio_values = []
    
    for episode in range(n_episodes):
        stats = agent.train_on_episode(env_train)
        
        episode_rewards.append(stats['total_reward'])
        episode_portfolio_values.append(stats['portfolio_value'])
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_portfolio = np.mean(episode_portfolio_values[-10:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Avg Portfolio: ${avg_portfolio:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    print("-" * 70)
    print("  ✓ Training completed")
    
    # 8. Save the trained agent
    if save_models:
        os.makedirs('models', exist_ok=True)
        model_suffix = 'with_hmm' if use_hmm else 'without_hmm'
        model_path = f'models/dqn_agent_{ticker}_{model_suffix}.h5'
        agent.save(model_path)
        print(f"  ✓ Saved DQN model to {model_path}")
    
    # 9. Evaluate on test set
    print("\nStep 9: Evaluating on test set...")
    test_stats = agent.test_on_episode(env_test)
    print(f"  Test Portfolio Value: ${test_stats['portfolio_value']:.2f}")
    print(f"  Test Returns: {test_stats['portfolio_returns']*100:.2f}%")
    print(f"  Total Trades: {test_stats['total_trades']}")
    print(f"  Winning Trades: {test_stats['winning_trades']}")
    
    # 10. Plot training progress
    print("\nStep 10: Saving training plots...")
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    axes[0].plot(episode_rewards)
    axes[0].set_title(f'Training Rewards ({model_suffix.replace("_", " ").title()})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)
    
    # Plot portfolio values
    axes[1].plot(episode_portfolio_values)
    axes[1].axhline(y=10000, color='r', linestyle='--', label='Initial Balance')
    axes[1].set_title(f'Portfolio Value ({model_suffix.replace("_", " ").title()})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Portfolio Value ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'results/training_{ticker}_{model_suffix}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved training plot to {plot_path}")
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}\n")
    
    return agent, env_test, test_stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN trading agent')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--use-hmm', action='store_true', default=True, help='Use HMM regime detection')
    parser.add_argument('--no-hmm', dest='use_hmm', action='store_false', help='Disable HMM regime detection')
    parser.add_argument('--save-models', action='store_true', default=True, help='Save trained models')
    
    args = parser.parse_args()
    
    train_agent(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        n_episodes=args.episodes,
        use_hmm=args.use_hmm,
        save_models=args.save_models
    )
