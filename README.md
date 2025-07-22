# Multi-Fusion-Net for Autonomous Driving

## 🎯 Project Overview
Multi-modal (Camera + LiDAR) BEV-based perception model with adaptive camera configuration support.

### Key Features
- **Dynamic Multi-Camera**: Supports 2-6 cameras with automatic adaptation
- **LiDAR Fallback**: Works even when cameras are unavailable  
- **Multi-Task**: 3D Detection + Lane Segmentation + Occupancy mapping
- **Real-time**: Optimized for Jetson Orin (TensorRT INT8 ≥20 FPS)

### Dataset
- **PandaSet**: 39 sequences, 6-cam + 64ch LiDAR → 32ch downsampled
- **Training**: Random camera dropout for robustness

## 🏗️ Project Structure
```
multi_fusion_net/
├── configs/           # Configuration files
├── src/              # Source code
│   ├── dataset/      # Data loaders
│   ├── models/       # Model architectures
│   ├── utils/        # Utility functions
│   └── losses/       # Loss functions
├── scripts/          # Training/evaluation scripts
├── data/            # Dataset symlink
└── checkpoints/     # Model weights
```

## 🔧 Development Phases
1. **Infrastructure**: Project setup, data loaders, coordinate transforms
2. **Model Architecture**: Camera backbone, LiDAR encoder, fusion module, heads
3. **Training System**: Loss functions, training scripts
4. **Deployment**: TensorRT optimization, real-time inference

## 📊 Target Performance
- **mAP**: >0.4 for 3D detection
- **Latency**: <50ms inference time
- **Memory**: <8GB GPU memory during training