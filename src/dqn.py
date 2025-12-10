"""
Deep Q-Network (DQN) agent with experience replay and target network.
Implements epsilon-greedy exploration strategy.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
from typing import Tuple, Optional


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def size(self) -> int:
        """Get the current size of the buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class DQNAgent:
    """Deep Q-Network agent for trading."""
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 10):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_capacity: Capacity of replay buffer
            batch_size: Size of training batches
            target_update_freq: Frequency of target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Build networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Training metrics
        self.training_step = 0
        self.losses = []
    
    def _build_model(self) -> keras.Model:
        """
        Build the neural network model for Q-value approximation.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights from main model to target model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_size)
        
        # Exploit: best action according to Q-values
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def replay(self) -> Optional[float]:
        """
        Train the model on a batch of experiences from replay buffer.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if self.replay_buffer.size() < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Predict Q-values for current states
        q_values = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values using Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.model.fit(states, q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.losses.append(loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_model()
        
        return loss
    
    def train_on_episode(self, env, max_steps: Optional[int] = None) -> dict:
        """
        Train the agent on one episode.
        
        Args:
            env: Trading environment
            max_steps: Maximum number of steps per episode
            
        Returns:
            Dictionary with episode statistics
        """
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        done = False
        while not done:
            # Select and perform action
            action = self.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.remember(state, action, reward, next_state, done)
            
            # Train on replay buffer
            loss = self.replay()
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            if max_steps and steps >= max_steps:
                break
        
        episode_stats = {
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'portfolio_value': info.get('portfolio_value', 0),
            'total_trades': info.get('total_trades', 0),
            'winning_trades': info.get('winning_trades', 0)
        }
        
        return episode_stats
    
    def test_on_episode(self, env, max_steps: Optional[int] = None) -> dict:
        """
        Test the agent on one episode (no exploration).
        
        Args:
            env: Trading environment
            max_steps: Maximum number of steps per episode
            
        Returns:
            Dictionary with episode statistics
        """
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        done = False
        while not done:
            # Select action (no exploration)
            action = self.act(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            if max_steps and steps >= max_steps:
                break
        
        episode_stats = {
            'total_reward': total_reward,
            'steps': steps,
            'portfolio_value': info.get('portfolio_value', 0),
            'total_trades': info.get('total_trades', 0),
            'winning_trades': info.get('winning_trades', 0),
            'portfolio_returns': env.get_portfolio_returns()
        }
        
        return episode_stats
    
    def save(self, filepath: str):
        """Save the model weights."""
        self.model.save_weights(filepath)
    
    def load(self, filepath: str):
        """Load the model weights."""
        self.model.load_weights(filepath)
        self.update_target_model()
