#!/usr/bin/env python3
"""
Simple test to verify basic functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing imports...")
    
    # Basic imports
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    import yaml
    print("✅ YAML")
    
    # Try our modules
    from src.utils.transforms import CoordinateTransforms
    print("✅ CoordinateTransforms")
    
    # Test config loading
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Config loaded")
    
    # Test coordinate transforms
    coord_transforms = CoordinateTransforms(config['bev_grid'])
    print("✅ CoordinateTransforms initialized")
    
    # Test basic tensor operations
    test_tensor = torch.randn(100, 3)
    bev_points = coord_transforms.ego_to_bev(test_tensor.numpy())
    print(f"✅ BEV projection: {bev_points.shape}")
    
    print("\n🎉 All basic tests passed!")
    print("Ready to proceed with data loading...")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)