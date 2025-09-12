"""
28Bot v2 Configuration
Central configuration file for all system components
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"

# Model paths
BELIEF_MODEL_PATH = MODELS_DIR / "belief_network.pt"
RL_MODEL_PATH = MODELS_DIR / "rl_agent.pt"
POINT_PREDICTION_PATH = MODELS_DIR / "point_prediction.pt"

# Training configuration
TRAINING_CONFIG = {
    "belief_network": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_split": 0.2
    },
    "rl_agent": {
        "episodes": 10000,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995
    }
}

# Game configuration
GAME_CONFIG = {
    "num_players": 4,
    "cards_per_player": 8,
    "total_cards": 32,
    "min_bid": 16,
    "max_bid": 28,
    "point_values": {
        "J": 3,
        "9": 2,
        "A": 1,
        "10": 1
    }
}

# System configuration
SYSTEM_CONFIG = {
    "decision_time_limit": 25,  # milliseconds
    "enable_hybrid_mode": True,
    "enable_uncertainty_quantification": True,
    "log_level": "INFO"
}

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)
