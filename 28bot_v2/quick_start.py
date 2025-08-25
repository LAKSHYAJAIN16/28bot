#!/usr/bin/env python3
"""
Quick Start Script for 28Bot v2
This script demonstrates the new organized structure and helps you get started.
"""

import sys
import os

def print_structure():
    """Print the new organized structure"""
    print("="*60)
    print("28BOT V2 - ORGANIZED STRUCTURE")
    print("="*60)
    print()
    print("📁 Project Structure:")
    print("├── 📚 docs/                    # Documentation and guides")
    print("├── 🔧 scripts/                 # Training and analysis scripts")
    print("├── 💡 examples/                # Usage examples and demos")
    print("├── 🧪 tests/                   # Testing and debugging")
    print("├── 📊 data/                    # Data files")
    print("├── 🤖 models/                  # Trained models")
    print("├── 📈 logs/                    # Training logs")
    print("├── 🎮 game28/                  # Core game logic")
    print("├── 🧠 rl_bidding/             # RL environment")
    print("├── 🌳 ismcts/                 # MCTS implementation")
    print("└── 🧮 belief_model/           # Belief networks")
    print()

def print_quick_commands():
    """Print quick commands to get started"""
    print("🚀 Quick Start Commands:")
    print()
    print("1. Train the improved bidding model:")
    print("   python scripts/improved_bidding_trainer.py")
    print()
    print("2. Use the improved model:")
    print("   python examples/use_improved_bidding_model.py")
    print()
    print("3. Run a simple example:")
    print("   python examples/simple_improved_bidding_example.py")
    print()
    print("4. Test the environment:")
    print("   python tests/test_improved_env.py")
    print()
    print("5. Analyze MCTS data:")
    print("   python scripts/analyze_mcts_data.py")
    print()

def print_documentation():
    """Print documentation information"""
    print("📖 Documentation:")
    print()
    print("• Main README: README.md")
    print("• Quick Start: docs/QUICKSTART.md")
    print("• Improved Model Usage: docs/IMPROVED_MODEL_USAGE.md")
    print("• First 4 Cards Analysis: docs/FIRST_4_CARDS_ANALYSIS.md")
    print("• Point Prediction: docs/POINT_PREDICTION_APPROACH.md")
    print()

def print_key_features():
    """Print key features"""
    print("🎯 Key Features:")
    print()
    print("• 🤖 Improved Bidding Model (MCTS-enhanced RL)")
    print("• 🧠 Point Prediction Model")
    print("• 🌳 MCTS Bot Implementation")
    print("• 📊 First 4 Cards Analysis (Game 28 specific)")
    print("• 🎮 Complete Game 28 Implementation")
    print("• 📈 Comprehensive Evaluation Tools")
    print()

def main():
    """Main function"""
    print_structure()
    print_quick_commands()
    print_documentation()
    print_key_features()
    
    print("="*60)
    print("🎉 You're all set! The project is now organized and ready to use.")
    print("="*60)
    
    # Check if key files exist
    print("\n🔍 Checking key files...")
    
    key_files = [
        "scripts/improved_bidding_trainer.py",
        "examples/use_improved_bidding_model.py",
        "data/mcts_bidding_analysis.json",
        "models/improved_bidding_model.zip"
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (not found)")
    
    print("\n💡 Tip: If models are missing, run the training script first!")
    print("   python scripts/improved_bidding_trainer.py")

if __name__ == "__main__":
    main()
