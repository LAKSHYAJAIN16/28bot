# Backwards Compatibility for Improved Belief Model

This document explains how the improved belief model has been made backwards compatible to work on both CPU and GPU environments.

## Overview

The improved belief model now supports:
- **CPU-only machines** (no GPU required)
- **GPU machines with PyTorch 1.x** (older versions)
- **GPU machines with PyTorch 2.0+** (latest versions)
- **Automatic Mixed Precision (AMP)** with fallback support

## Key Changes Made

### 1. Device-Aware Tensor Creation
All tensors in the model are now created on the correct device from the start:
```python
# Get device from model parameters
device = next(self.parameters()).device

# Create tensors on the correct device
hand_tensor = torch.zeros(8, 8, device=device)
features = torch.zeros(50, device=device)
trick_tensor = torch.zeros(8, 16, device=device)
```

### 2. Backwards Compatible AMP (Automatic Mixed Precision)
The training functions now support both old and new PyTorch AMP APIs:

```python
# Backwards compatible AMP setup
if use_amp and device.type == 'cuda':
    try:
        # Try new API first (PyTorch 2.0+)
        scaler = torch.amp.GradScaler('cuda')
        print("Using PyTorch 2.0+ AMP API")
    except:
        # Fallback to old API (PyTorch 1.x)
        scaler = torch.cuda.amp.GradScaler()
        print("Using PyTorch 1.x AMP API")
else:
    scaler = None
    print("AMP disabled (CPU training or not available)")
```

### 3. Backwards Compatible Autocast
Forward passes use the appropriate autocast API:

```python
# Forward pass with backwards compatible autocast
if use_amp and device.type == 'cuda':
    try:
        # Try new API first (PyTorch 2.0+)
        with torch.amp.autocast('cuda'):
            predictions = model(game_state, player_id)
            loss = _calculate_loss(predictions, target_beliefs, criterion, device)
    except:
        # Fallback to old API (PyTorch 1.x)
        with torch.cuda.amp.autocast():
            predictions = model(game_state, player_id)
            loss = _calculate_loss(predictions, target_beliefs, criterion, device)
else:
    # CPU training
    predictions = model(game_state, player_id)
    loss = _calculate_loss(predictions, target_beliefs, criterion, device)
```

## Usage Options

### Option 1: Local Training (CPU or GPU)
Use the local training script:
```bash
# Train on CPU
python train_improved_belief_local.py --epochs 20 --lr 0.001 --no-amp

# Train on GPU with AMP
python train_improved_belief_local.py --epochs 20 --lr 0.001

# Train with custom parameters
python train_improved_belief_local.py --epochs 50 --lr 0.0005 --save-dir my_models
```

### Option 2: Google Colab Training (GPU)
Use the backwards compatible Colab notebook:
- `colab_train_improved_belief_backwards_compatible.ipynb`

This notebook automatically detects your environment and uses the appropriate APIs.

### Option 3: Test Compatibility
Run the compatibility test to verify everything works:
```bash
python test_backwards_compatibility.py
```

## Environment Requirements

### Minimum Requirements
- Python 3.7+
- PyTorch 1.8+ (for CPU training)
- PyTorch 1.8+ with CUDA (for GPU training)

### Recommended Requirements
- Python 3.8+
- PyTorch 2.0+ (for latest features and performance)
- CUDA 11.0+ (for GPU training)

## Troubleshooting

### Common Issues

1. **"Expected all tensors to be on the same device"**
   - This should be fixed with the device-aware tensor creation
   - If you still see this, run the compatibility test

2. **"Module not found" errors**
   - Make sure you're running from the `28bot_v2` directory
   - Check that all required files are present

3. **AMP errors on older PyTorch versions**
   - The code automatically falls back to the old API
   - If you still have issues, use `--no-amp` flag

4. **CUDA out of memory**
   - Reduce batch size or use CPU training
   - Use `--no-amp` to disable mixed precision

### Testing Your Setup

Run the compatibility test to verify your environment:
```bash
python test_backwards_compatibility.py
```

This will test:
- CPU compatibility
- GPU compatibility (if available)
- PyTorch 1.x AMP API
- PyTorch 2.0+ AMP API

## Performance Notes

- **CPU Training**: Slower but works everywhere
- **GPU Training**: Much faster, especially with AMP
- **AMP**: Reduces memory usage and speeds up training on GPU
- **PyTorch 2.0+**: Better performance and newer features

## File Structure

```
28bot_v2/
├── belief_model/
│   └── improved_belief_net.py          # Main model (backwards compatible)
├── train_improved_belief_local.py      # Local training script
├── test_backwards_compatibility.py     # Compatibility test
├── colab_train_improved_belief_backwards_compatible.ipynb  # Colab notebook
└── BACKWARDS_COMPATIBILITY_README.md   # This file
```

## Support

If you encounter any issues:
1. Run the compatibility test first
2. Check that you're using a supported PyTorch version
3. Try running on CPU first to isolate GPU issues
4. Use the `--no-amp` flag if AMP is causing problems
