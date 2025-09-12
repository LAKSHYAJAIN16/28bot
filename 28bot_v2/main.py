#!/usr/bin/env python3
"""
28Bot v2 - Main Entry Point
Advanced Multi-Agent AI System for Imperfect Information Games

This is the main entry point for the 28Bot system. It provides easy access
to all major functionalities including training, evaluation, and gameplay.
"""

import argparse
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(
        description="28Bot v2 - Advanced AI System for Game 28",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py quick-start          # Run quick demonstration
  python main.py train-belief         # Train belief network
  python main.py train-rl             # Train RL agent
  python main.py play-game            # Play a game simulation
  python main.py evaluate             # Run comprehensive evaluation
  python main.py analyze              # Analyze model performance
        """
    )
    
    parser.add_argument(
        'command',
        choices=[
            'quick-start', 'train-belief', 'train-rl', 'play-game', 
            'evaluate', 'analyze', 'help'
        ],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.command == 'quick-start':
        print("Starting 28Bot v2 Quick Start...")
        from quick_start import main as quick_start_main
        quick_start_main()
        
    elif args.command == 'train-belief':
        print("Training Belief Network...")
        from scripts.train_improved_belief import main as train_belief_main
        train_belief_main()
        
    elif args.command == 'train-rl':
        print("Training RL Agent...")
        from rl_bidding.train_policy import main as train_rl_main
        train_rl_main()
        
    elif args.command == 'play-game':
        print("Starting Game Simulation...")
        from scripts.main_game_simulation import main as play_game_main
        play_game_main()
        
    elif args.command == 'evaluate':
        print("Running Comprehensive Evaluation...")
        from tests.test_all_improvements import main as evaluate_main
        evaluate_main()
        
    elif args.command == 'analyze':
        print("🔍 Analyzing Model Performance...")
        from utils.analyze_belief_model import main as analyze_main
        analyze_main()
        
    elif args.command == 'help':
        print("""
28Bot v2 - Advanced Multi-Agent AI System for Imperfect Information Games

Available Commands:
  quick-start    Run a quick demonstration of the system
  train-belief   Train the belief network for opponent modeling
  train-rl       Train the reinforcement learning agent
  play-game      Run a game simulation with AI agents
  evaluate       Run comprehensive evaluation of all components
  analyze        Analyze model performance and generate reports

For more detailed help on any command, use:
  python main.py <command> --help

Documentation is available in the docs/ folder.
        """)

if __name__ == "__main__":
    main()
