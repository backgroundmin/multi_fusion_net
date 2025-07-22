"""
Test script for PandaSet DataLoader
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.dataset.pandaset import PandaSetDataset


def visualize_sample(sample):
    """Visualize a single data sample"""
    print(f"Sample info: Seq {sample['sequence_id']}, Frame {sample['frame_idx']}")
    print(f"Valid cameras: {sample['camera_valid']}")
    print(f"LiDAR points: {len(sample['lidar_points'])}")
    print(f"3D boxes: {len(sample['boxes_3d'])}")
    
    # Visualize camera images
    num_cameras = len(sample['images'])
    if num_cameras > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (cam_name, image) in enumerate(sample['images'].items()):
            if idx < 6:
                axes[idx].imshow(image)
                axes[idx].set_title(f"{cam_name}")
                axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(num_cameras, 6):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig('camera_sample.png')
        plt.close()
    
    # Visualize LiDAR point cloud (top view)
    if len(sample['lidar_points']) > 0:
        points = sample['lidar_points']
        
        plt.figure(figsize=(10, 8))
        plt.scatter(points[:, 0], points[:, 1], c=points[:, 3], s=1, cmap='viridis')
        plt.xlabel('X (forward)')
        plt.ylabel('Y (right)')
        plt.title(f'LiDAR Point Cloud ({len(points)} points)')
        plt.colorbar(label='Intensity')
        plt.axis('equal')
        plt.savefig('lidar_sample.png')
        plt.close()


def main():
    """Test PandaSet DataLoader"""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating PandaSet dataset...")
    
    # Create dataset
    dataset = PandaSetDataset(
        config=config,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        collate_fn=None  # We'll need a custom collate function later
    )
    
    # Test loading a few samples
    print("\nTesting data loading...")
    for i, sample in enumerate(dataloader):
        if i >= 3:  # Test first 3 samples
            break
            
        print(f"\n--- Sample {i+1} ---")
        
        # Since batch_size=1, sample is a list with one element
        sample = sample[0] if isinstance(sample, list) else sample
        
        visualize_sample(sample)
        
        # Print data shapes and info
        for cam_name, image in sample['images'].items():
            print(f"{cam_name}: {image.shape}")
            
        print(f"LiDAR points shape: {sample['lidar_points'].shape}")
        print(f"3D boxes shape: {sample['boxes_3d'].shape}")
    
    print(f"\nVisualization saved to camera_sample.png and lidar_sample.png")


if __name__ == "__main__":
    main()