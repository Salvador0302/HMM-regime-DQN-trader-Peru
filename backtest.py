"""
Backtesting script for comparing DQN trading agents with and without HMM.
Computes Sharpe ratio, cumulative returns, and maximum drawdown.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.ingest import DataIngestor
from src.preprocess import DataPreprocessor
from src.hmm_regimes import HMMRegimeDetector
from src.trading_env import TradingEnv
from src.dqn import DQNAgent


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        portfolio_values: Array of portfolio values over time
        
    Returns:
        Maximum drawdown as a percentage
    """
    if len(portfolio_values) == 0:
        return 0.0
    
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdown)
    
    return max_drawdown


def calculate_cumulative_return(initial_value: float, final_value: float) -> float:
    """
    Calculate cumulative return.
    
    Args:
        initial_value: Initial portfolio value
        final_value: Final portfolio value
        
    Returns:
        Cumulative return as a percentage
    """
    return (final_value - initial_value) / initial_value


def run_backtest(agent: DQNAgent, 
                 env: TradingEnv,
                 name: str = "Agent") -> dict:
    """
    Run backtest for a given agent and environment.
    
    Args:
        agent: Trained DQN agent
        env: Trading environment
        name: Name for the backtest
        
    Returns:
        Dictionary with backtest results
    """
    print(f"\nRunning backtest: {name}")
    print("-" * 60)
    
    # Reset environment
    state, _ = env.reset()
    
    # Track portfolio values and actions
    portfolio_values = [env.initial_balance]
    actions_taken = []
    rewards = []
    
    done = False
    while not done:
        # Get action from agent (no exploration)
        action = agent.act(state, training=False)
        actions_taken.append(action)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record metrics
        portfolio_values.append(info['portfolio_value'])
        rewards.append(reward)
        
        state = next_state
    
    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    cumulative_return = calculate_cumulative_return(env.initial_balance, portfolio_values[-1])
    sharpe_ratio = calculate_sharpe_ratio(returns)
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # Count actions
    action_counts = {
        'Hold': actions_taken.count(0),
        'Buy': actions_taken.count(1),
        'Sell': actions_taken.count(2)
    }
    
    results = {
        'name': name,
        'initial_value': env.initial_balance,
        'final_value': portfolio_values[-1],
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': env.total_trades,
        'winning_trades': env.winning_trades,
        'win_rate': env.winning_trades / env.total_trades if env.total_trades > 0 else 0,
        'portfolio_values': portfolio_values,
        'returns': returns,
        'actions': action_counts
    }
    
    # Print results
    print(f"  Final Portfolio Value: ${results['final_value']:.2f}")
    print(f"  Cumulative Return: {results['cumulative_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Winning Trades: {results['winning_trades']} ({results['win_rate']*100:.2f}%)")
    print(f"  Actions - Hold: {action_counts['Hold']}, Buy: {action_counts['Buy']}, Sell: {action_counts['Sell']}")
    print("-" * 60)
    
    return results


def compare_strategies(ticker: str = 'AAPL',
                      start_date: str = '2020-01-01',
                      end_date: str = '2023-12-31'):
    """
    Compare DQN agents with and without HMM regime detection.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for backtesting
        end_date: End date for backtesting
    """
    print(f"\n{'='*70}")
    print(f"BACKTESTING COMPARISON: WITH vs WITHOUT HMM")
    print(f"Ticker: {ticker} | Period: {start_date} to {end_date}")
    print(f"{'='*70}\n")
    
    # 1. Prepare data
    print("Step 1: Preparing data...")
    ingestor = DataIngestor()
    raw_data = ingestor.fetch_yahoo_finance([ticker], start_date, end_date)
    
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = ['_'.join(col).strip('_') for col in raw_data.columns.values]
    
    preprocessor = DataPreprocessor()
    
    # Find Close column
    close_col = None
    for col in raw_data.columns:
        if 'close' in col.lower():
            close_col = col
            break
    
    processed_data = preprocessor.add_technical_indicators(raw_data, price_col=close_col)
    processed_data = preprocessor.clean_data(processed_data, drop_na=True)
    
    # Split data (use only test set for backtesting)
    _, test_data = preprocessor.split_train_test(processed_data, train_ratio=0.8)
    print(f"  ✓ Test set: {len(test_data)} rows")
    
    # 2. Prepare environments and agents
    exclude_cols = ['regime', 'regime_label'] + [col for col in processed_data.columns if 'regime_prob' in col]
    feature_cols = [col for col in processed_data.columns if col not in exclude_cols]
    
    test_data_norm = preprocessor.normalize_features(test_data, feature_columns=feature_cols, fit=True)
    
    # 3. Load and backtest agent WITH HMM
    results_with_hmm = None
    model_path_with = f'models/dqn_agent_{ticker}_with_hmm.h5'
    
    if os.path.exists(model_path_with):
        print("\nStep 2: Backtesting agent WITH HMM...")
        
        env_with_hmm = TradingEnv(
            data=test_data_norm,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=20,
            feature_columns=feature_cols
        )
        
        agent_with_hmm = DQNAgent(
            state_size=env_with_hmm.observation_space.shape[0],
            action_size=env_with_hmm.action_space.n
        )
        agent_with_hmm.load(model_path_with)
        
        results_with_hmm = run_backtest(agent_with_hmm, env_with_hmm, "DQN WITH HMM")
    else:
        print(f"\n  ⚠ Model not found: {model_path_with}")
        print("  Run train.py with --use-hmm first")
    
    # 4. Load and backtest agent WITHOUT HMM
    results_without_hmm = None
    model_path_without = f'models/dqn_agent_{ticker}_without_hmm.h5'
    
    if os.path.exists(model_path_without):
        print("\nStep 3: Backtesting agent WITHOUT HMM...")
        
        env_without_hmm = TradingEnv(
            data=test_data_norm,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=20,
            feature_columns=feature_cols
        )
        
        agent_without_hmm = DQNAgent(
            state_size=env_without_hmm.observation_space.shape[0],
            action_size=env_without_hmm.action_space.n
        )
        agent_without_hmm.load(model_path_without)
        
        results_without_hmm = run_backtest(agent_without_hmm, env_without_hmm, "DQN WITHOUT HMM")
    else:
        print(f"\n  ⚠ Model not found: {model_path_without}")
        print("  Run train.py with --no-hmm first")
    
    # 5. Create buy-and-hold baseline
    print("\nStep 4: Computing Buy-and-Hold baseline...")
    initial_price = test_data_norm.iloc[0][close_col] if close_col in test_data_norm.columns else test_data_norm.iloc[0]['Close']
    final_price = test_data_norm.iloc[-1][close_col] if close_col in test_data_norm.columns else test_data_norm.iloc[-1]['Close']
    
    buy_hold_return = (final_price - initial_price) / initial_price
    buy_hold_final_value = 10000 * (1 + buy_hold_return)
    
    # Calculate buy-and-hold portfolio values
    prices = test_data_norm[close_col].values if close_col in test_data_norm.columns else test_data_norm['Close'].values
    buy_hold_portfolio = 10000 * (prices / prices[0])
    buy_hold_returns = np.diff(buy_hold_portfolio) / buy_hold_portfolio[:-1]
    
    results_buy_hold = {
        'name': 'Buy and Hold',
        'initial_value': 10000,
        'final_value': buy_hold_final_value,
        'cumulative_return': buy_hold_return,
        'sharpe_ratio': calculate_sharpe_ratio(buy_hold_returns),
        'max_drawdown': calculate_max_drawdown(buy_hold_portfolio),
        'portfolio_values': buy_hold_portfolio
    }
    
    print(f"  Final Portfolio Value: ${results_buy_hold['final_value']:.2f}")
    print(f"  Cumulative Return: {results_buy_hold['cumulative_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {results_buy_hold['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {results_buy_hold['max_drawdown']*100:.2f}%")
    
    # 6. Create comparison visualizations
    print("\nStep 5: Creating comparison plots...")
    os.makedirs('results', exist_ok=True)
    
    # Prepare data for plotting
    all_results = [r for r in [results_with_hmm, results_without_hmm, results_buy_hold] if r is not None]
    
    if len(all_results) > 0:
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Portfolio value over time
        ax1 = fig.add_subplot(gs[0, :])
        for result in all_results:
            ax1.plot(result['portfolio_values'], label=result['name'], linewidth=2)
        ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Balance')
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative returns comparison
        ax2 = fig.add_subplot(gs[1, 0])
        names = [r['name'] for r in all_results]
        returns = [r['cumulative_return'] * 100 for r in all_results]
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax2.bar(names, returns, color=colors, alpha=0.7)
        ax2.set_title('Cumulative Returns (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Sharpe ratio comparison
        ax3 = fig.add_subplot(gs[1, 1])
        sharpe_ratios = [r['sharpe_ratio'] for r in all_results]
        colors = ['green' if s > 0 else 'red' for s in sharpe_ratios]
        ax3.bar(names, sharpe_ratios, color=colors, alpha=0.7)
        ax3.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Max drawdown comparison
        ax4 = fig.add_subplot(gs[2, 0])
        drawdowns = [r['max_drawdown'] * 100 for r in all_results]
        ax4.bar(names, drawdowns, color='red', alpha=0.7)
        ax4.set_title('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 5. Summary metrics table
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('tight')
        ax5.axis('off')
        
        table_data = []
        for result in all_results:
            row = [
                result['name'],
                f"${result['final_value']:.2f}",
                f"{result['cumulative_return']*100:.2f}%",
                f"{result['sharpe_ratio']:.4f}",
                f"{result['max_drawdown']*100:.2f}%"
            ]
            table_data.append(row)
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Strategy', 'Final Value', 'Return', 'Sharpe', 'Max DD'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle(f'Backtest Comparison: {ticker}', fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        plot_path = f'results/backtest_comparison_{ticker}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved comparison plot to {plot_path}")
        
        plt.close()
    
    # 7. Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if results_with_hmm and results_without_hmm:
        improvement = (results_with_hmm['cumulative_return'] - results_without_hmm['cumulative_return']) * 100
        sharpe_improvement = results_with_hmm['sharpe_ratio'] - results_without_hmm['sharpe_ratio']
        
        print(f"\nHMM Impact:")
        print(f"  Return Improvement: {improvement:+.2f}%")
        print(f"  Sharpe Improvement: {sharpe_improvement:+.4f}")
        print(f"  Better Strategy: {'WITH HMM' if improvement > 0 else 'WITHOUT HMM'}")
    
    print(f"\n{'='*70}\n")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest DQN trading agents')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    compare_strategies(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date
    )
