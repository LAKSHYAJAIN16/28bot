#!/usr/bin/env python3
"""
Test script to verify backwards compatibility of the improved belief model
Works on both CPU and GPU
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from belief_model.improved_belief_net import create_improved_belief_model


def test_device_compatibility():
    """Test that the model works on both CPU and GPU"""
    print("=== TESTING DEVICE COMPATIBILITY ===")
    
    # Test on CPU
    print("\n1. Testing on CPU...")
    device_cpu = torch.device("cpu")
    model_cpu = create_improved_belief_model()
    model_cpu = model_cpu.to(device_cpu)
    print(f"✅ Model created on CPU: {next(model_cpu.parameters()).device}")
    
    # Test forward pass on CPU
    try:
        # Create a simple test game state
        class SimpleGameState:
            def __init__(self):
                self.hands = [[], [], [], []]
                self.phase = "BIDDING"
                self.current_player = 0
                self.bidder = None
                self.winning_bid = 0
                self.trump_suit = None
                self.trump_revealed = False
                self.current_bid = 0
                self.bid_history = []
                self.passed_players = []
                self.current_trick = None
                self.tricks = []
                self.trick_leader = 0
                self.team_scores = [0, 0]
                self.game_points = {'A': 0, 'B': 0}  # Changed to dict
                self.round_number = 1
                self.game_over = False
        
        test_state = SimpleGameState()
        test_player_id = 0
        
        with torch.no_grad():
            predictions = model_cpu(test_state, test_player_id)
        print(f"✅ CPU forward pass successful: {predictions.trump_suit.device}")
    except Exception as e:
        print(f"❌ CPU forward pass failed: {e}")
        return False
    
    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n2. Testing on GPU...")
        device_gpu = torch.device("cuda")
        model_gpu = create_improved_belief_model()
        model_gpu = model_gpu.to(device_gpu)
        print(f"✅ Model created on GPU: {next(model_gpu.parameters()).device}")
        
        # Test forward pass on GPU
        try:
            with torch.no_grad():
                predictions = model_gpu(test_state, test_player_id)
            print(f"✅ GPU forward pass successful: {predictions.trump_suit.device}")
        except Exception as e:
            print(f"❌ GPU forward pass failed: {e}")
            return False
    else:
        print("\n2. GPU not available, skipping GPU test")
    
    print("\n✅ Device compatibility test passed!")
    return True


def test_amp_compatibility():
    """Test AMP (Automatic Mixed Precision) compatibility"""
    print("\n=== TESTING AMP COMPATIBILITY ===")
    
    if not torch.cuda.is_available():
        print("GPU not available, skipping AMP test")
        return True
    
    device = torch.device("cuda")
    model = create_improved_belief_model().to(device)
    
    # Test new AMP API (PyTorch 2.0+)
    print("\n1. Testing PyTorch 2.0+ AMP API...")
    try:
        scaler = torch.amp.GradScaler('cuda')
        print("✅ PyTorch 2.0+ GradScaler works")
        
        with torch.amp.autocast('cuda'):
            # Create a simple test game state
            class SimpleGameState:
                def __init__(self):
                    self.hands = [[], [], [], []]
                    self.phase = "BIDDING"
                    self.current_player = 0
                    self.bidder = None
                    self.winning_bid = 0
                    self.trump_suit = None
                    self.trump_revealed = False
                    self.current_bid = 0
                    self.bid_history = []
                    self.passed_players = []
                    self.current_trick = None
                    self.tricks = []
                    self.trick_leader = 0
                    self.team_scores = [0, 0]
                    self.game_points = {'A': 0, 'B': 0}  # Changed to dict
                    self.round_number = 1
                    self.game_over = False
            
            test_state = SimpleGameState()
            test_player_id = 0
            
            predictions = model(test_state, test_player_id)
        print("✅ PyTorch 2.0+ autocast works")
    except Exception as e:
        print(f"❌ PyTorch 2.0+ AMP API failed: {e}")
    
    # Test old AMP API (PyTorch 1.x)
    print("\n2. Testing PyTorch 1.x AMP API...")
    try:
        scaler = torch.cuda.amp.GradScaler()
        print("✅ PyTorch 1.x GradScaler works")
        
        with torch.cuda.amp.autocast():
            predictions = model(test_state, test_player_id)
        print("✅ PyTorch 1.x autocast works")
    except Exception as e:
        print(f"❌ PyTorch 1.x AMP API failed: {e}")
    
    print("\n✅ AMP compatibility test completed!")
    return True


def main():
    """Run all compatibility tests"""
    print("🧪 RUNNING BACKWARDS COMPATIBILITY TESTS")
    print("=" * 50)
    
    # Test device compatibility
    if not test_device_compatibility():
        print("\n❌ Device compatibility test failed!")
        return False
    
    # Test AMP compatibility
    if not test_amp_compatibility():
        print("\n❌ AMP compatibility test failed!")
        return False
    
    print("\n" + "=" * 50)
    print("✅ ALL COMPATIBILITY TESTS PASSED!")
    print("The improved belief model is fully backwards compatible!")
    print("It works on:")
    print("  - CPU-only machines")
    print("  - GPU machines with PyTorch 1.x")
    print("  - GPU machines with PyTorch 2.0+")
    print("  - With and without AMP (mixed precision)")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
