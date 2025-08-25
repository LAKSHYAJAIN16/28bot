#!/usr/bin/env python3
"""
Retrain the point prediction model with correct dimensions
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.point_prediction_model import PointPredictionTrainer
import torch

def retrain_point_prediction():
    """Retrain the point prediction model"""
    print("🔄 Retraining Point Prediction Model")
    print("=" * 50)
    
    try:
        # Create trainer
        print("Creating trainer...")
        trainer = PointPredictionTrainer()
        
        # Train the model
        print("Training model...")
        trainer.train(epochs=50)  # Quick training
        
        # Save the model
        print("Saving model...")
        model_path = "models/point_prediction_model.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(trainer.model.state_dict(), model_path)
        
        print(f"✅ Point prediction model saved to {model_path}")
        print("Model now has correct dimensions:")
        print("  - Input: 4 cards × 13 ranks")
        print("  - Conv1d: 4 input channels")
        
        return True
        
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    retrain_point_prediction()
