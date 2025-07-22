#!/usr/bin/env python3
"""
Quick data loading test
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset.pandaset import PandaSetDataset

def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating dataset with first 3 samples...")
    
    # Modify config for quick test
    config['dataset']['sequences']['train'] = ["001"]
    
    # Create dataset
    dataset = PandaSetDataset(
        config=config,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first sample
    print("\nTesting first sample...")
    try:
        sample = dataset[0]
        
        print("✅ Sample loaded successfully!")
        print(f"  Sequence: {sample['sequence_id']}")
        print(f"  Frame: {sample['frame_idx']}")
        print(f"  Valid cameras: {len(sample['camera_valid'])}")
        print(f"  LiDAR points: {sample['lidar_points'].shape}")
        print(f"  3D boxes: {sample['boxes_3d'].shape}")
        
        # Check camera images
        for cam_name, image in sample['images'].items():
            print(f"  Camera {cam_name}: {image.shape}")
            
    except Exception as e:
        print(f"❌ Error loading sample: {e}")
        return False
    
    print("\n🎉 Data loading test passed!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("Ready to start training!")
    else:
        print("Please fix data loading issues first.")
        sys.exit(1)