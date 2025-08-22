"""
Training script for Game 28 bidding policy using PPO
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple
import random
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from rl_bidding.env_adapter import Game28Env
from game28.constants import *


class BiddingPolicy(nn.Module):
    """
    Neural network policy for bidding decisions
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_actions: int = len(BID_RANGE) + 1):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.feature_extractor(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


# Use standard PPO with MultiInputPolicy for dictionary observations


def create_env(player_id: int = 0):
    """Create environment function for vectorization"""
    def _create():
        return Game28Env(player_id=player_id)
    return _create


def train_bidding_policy(
    num_episodes: int = 10000,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    save_dir: str = "models/bidding_policy",
    log_dir: str = "logs/bidding_training"
):
    """
    Train a bidding policy using PPO
    
    Args:
        num_episodes: Number of training episodes
        learning_rate: Learning rate for the optimizer
        batch_size: Batch size for training
        save_dir: Directory to save the trained model
        log_dir: Directory for tensorboard logs
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set random seeds
    set_random_seed(42)
    
    # Create vectorized environment
    env = DummyVecEnv([create_env(i) for i in range(4)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([create_env(0)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best_model",
        log_path=log_dir,
        eval_freq=max(num_episodes // 10, 1),
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(num_episodes // 5, 1),
        save_path=save_dir,
        name_prefix="bidding_model"
    )
    
    # Create and train model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=1024,  # Reduced to get more frequent updates
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1
    )
    
    # Train the model
    print(f"Starting training for {num_episodes} episodes...")
    
    # Calculate total timesteps more accurately
    # Each episode typically has 4-8 bidding rounds, so estimate 6 steps per episode
    total_timesteps = num_episodes * 6
    
    print(f"Training for {total_timesteps} timesteps...")
    print(f"PPO n_steps: {model.n_steps}")
    print(f"Expected updates: {total_timesteps // model.n_steps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f"{save_dir}/final_model")
    env.save(f"{save_dir}/vec_normalize.pkl")
    
    print(f"Training completed. Model saved to {save_dir}")
    
    return model


def evaluate_policy(model_path: str, num_games: int = 100) -> Dict[str, float]:
    """
    Evaluate a trained bidding policy
    
    Args:
        model_path: Path to the trained model
        num_games: Number of games to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Load model
    model = PPO.load(model_path)
    
    # Create evaluation environment
    env = Game28Env(player_id=0)
    
    # Evaluation metrics
    wins = 0
    total_reward = 0
    bid_success_rate = 0
    avg_bid = 0
    
    for _ in tqdm(range(num_games), desc="Evaluating"):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
        # Record metrics
        if episode_reward > 0:
            wins += 1
        total_reward += episode_reward
        
        # Calculate bid success rate
        if env.game_state.bidder is not None:
            bidder_team = 'A' if env.game_state.bidder in TEAM_A else 'B'
            team_score = env.game_state.team_scores[bidder_team]
            winning_bid = env.game_state.winning_bid
            
            if team_score >= winning_bid:
                bid_success_rate += 1
            
            avg_bid += winning_bid
    
    # Calculate final metrics
    win_rate = wins / num_games
    avg_reward = total_reward / num_games
    bid_success_rate = bid_success_rate / num_games
    avg_bid = avg_bid / num_games
    
    results = {
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'bid_success_rate': bid_success_rate,
        'avg_bid': avg_bid
    }
    
    print(f"Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.3f}")
    
    return results


if __name__ == "__main__":
    # Train the bidding policy
    model = train_bidding_policy(
        num_episodes=5000,
        learning_rate=3e-4,
        batch_size=64
    )
    
    # Evaluate the trained model
    results = evaluate_policy("models/bidding_policy/final_model", num_games=100)
