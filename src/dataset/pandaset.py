"""
PandaSet Dataset Loader for Multi-Fusion-Net
Dynamic Multi-Camera + LiDAR support with camera dropout
"""

import os
import json
import pickle
import random
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

from ..utils.transforms import CoordinateTransforms, create_transformation_matrix


class PandaSetDataset(Dataset):
    """
    PandaSet Dataset with Dynamic Multi-Camera Support
    
    Features:
    - Variable camera count (2-6 cameras)
    - Camera dropout for robustness training
    - 64ch → 32ch LiDAR downsampling
    - Multi-task annotations (3D boxes, lane seg, occupancy)
    """
    
    CAMERA_NAMES = [
        "front_camera", "front_left_camera", "front_right_camera",
        "left_camera", "right_camera", "back_camera"
    ]
    
    def __init__(self, 
                 config: Dict,
                 split: str = "train",
                 transform: Optional[Any] = None):
        """
        Args:
            config: Dataset configuration from config.yaml
            split: Dataset split ("train", "val", "test")
            transform: Data augmentation transforms
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # Dataset paths
        self.root_path = Path(config['dataset']['root_path'])
        self.sequences = config['dataset']['sequences'][split]
        
        # Camera configuration
        self.enabled_cameras = config['dataset']['cameras']['enabled']
        self.image_size = config['dataset']['cameras']['image_size']
        self.resize_to = config['dataset']['cameras']['resize_to']
        
        # LiDAR configuration
        self.lidar_config = config['dataset']['lidar']
        
        # Initialize coordinate transforms
        self.coord_transforms = CoordinateTransforms(config['bev_grid'])
        
        # Training-specific settings
        self.camera_dropout_prob = 0.0
        if split == "train" and 'training' in config:
            self.camera_dropout_prob = config['training']['augmentation'].get('camera_dropout_prob', 0.0)
        
        # Build dataset index
        self.data_index = self._build_index()
        
        print(f"Loaded {len(self.data_index)} samples for {split} split")
        print(f"Camera dropout probability: {self.camera_dropout_prob}")
    
    def _build_index(self) -> List[Dict]:
        """Build index of all data samples"""
        data_index = []
        
        for seq_id in self.sequences:
            seq_path = self.root_path / seq_id
            
            if not seq_path.exists():
                print(f"Warning: Sequence {seq_id} not found at {seq_path}")
                continue
            
            # Get number of frames (assuming 80 frames per sequence)
            camera_path = seq_path / "camera" / self.CAMERA_NAMES[0]
            if camera_path.exists():
                frame_count = len(list(camera_path.glob("*.jpg")))
            else:
                frame_count = 80  # Default
            
            # Add all frames from this sequence
            for frame_idx in range(frame_count):
                data_index.append({
                    'sequence_id': seq_id,
                    'frame_idx': frame_idx
                })
        
        return data_index
    
    def _load_camera_data(self, seq_path: Path, frame_idx: int) -> Dict:
        """Load camera images and calibration data"""
        camera_data = {
            'images': {},
            'intrinsics': {},
            'extrinsics': {},
            'valid_cameras': []
        }
        
        # Apply camera dropout during training
        available_cameras = self.enabled_cameras.copy()
        if self.split == "train" and random.random() < self.camera_dropout_prob:
            # Randomly disable some cameras
            num_to_keep = random.randint(0, len(available_cameras))
            available_cameras = random.sample(available_cameras, num_to_keep)
        
        for camera_name in self.CAMERA_NAMES:
            camera_path = seq_path / "camera" / camera_name
            image_path = camera_path / f"{frame_idx:02d}.jpg"
            intrinsics_path = camera_path / "intrinsics.json"
            poses_path = camera_path / "poses.json"
            
            # Check if camera is enabled and available
            is_available = (camera_name in available_cameras and 
                          image_path.exists() and 
                          intrinsics_path.exists() and 
                          poses_path.exists())
            
            if is_available:
                # Load image
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize image if specified
                if self.resize_to:
                    image = cv2.resize(image, tuple(self.resize_to))
                
                camera_data['images'][camera_name] = image
                
                # Load intrinsics
                with open(intrinsics_path, 'r') as f:
                    intrinsics_data = json.load(f)
                
                K = np.array([
                    [intrinsics_data['fx'], 0, intrinsics_data['cx']],
                    [0, intrinsics_data['fy'], intrinsics_data['cy']],
                    [0, 0, 1]
                ])
                
                # Scale intrinsics if image was resized
                if self.resize_to:
                    scale_x = self.resize_to[0] / self.image_size[0]
                    scale_y = self.resize_to[1] / self.image_size[1]
                    K[0, :] *= scale_x
                    K[1, :] *= scale_y
                
                camera_data['intrinsics'][camera_name] = K
                
                # Load extrinsics (camera pose)
                with open(poses_path, 'r') as f:
                    poses_data = json.load(f)
                
                frame_pose = poses_data[frame_idx]  # poses_data is a list, not dict
                T_cam_to_ego = create_transformation_matrix(
                    translation=[frame_pose['position']['x'], frame_pose['position']['y'], frame_pose['position']['z']],
                    quaternion=[frame_pose['heading']['w'], frame_pose['heading']['x'], frame_pose['heading']['y'], frame_pose['heading']['z']]
                )
                
                camera_data['extrinsics'][camera_name] = T_cam_to_ego
                camera_data['valid_cameras'].append(camera_name)
        
        return camera_data
    
    def _load_lidar_data(self, seq_path: Path, frame_idx: int) -> Dict:
        """Load LiDAR point cloud and calibration data"""
        lidar_path = seq_path / "lidar" / f"{frame_idx:02d}.pkl"
        poses_path = seq_path / "lidar" / "poses.json"
        
        lidar_data = {
            'points': np.array([]),
            'extrinsics': np.eye(4)
        }
        
        if not lidar_path.exists():
            return lidar_data
        
        # Load point cloud
        with open(lidar_path, 'rb') as f:
            points_df = pickle.load(f)
        
        # Extract point coordinates and intensity
        points = np.column_stack([
            points_df['x'].values,
            points_df['y'].values, 
            points_df['z'].values,
            points_df['i'].values  # intensity
        ])
        
        # Downsample from 64 to 32 channels
        points = self.coord_transforms.downsample_lidar_channels(
            points,
            original_channels=self.lidar_config['original_channels'],
            target_channels=self.lidar_config['target_channels'],
            method=self.lidar_config['downsample_method']
        )
        
        # Filter by range
        ranges = np.linalg.norm(points[:, :3], axis=1)
        range_mask = (ranges >= self.lidar_config['min_range']) & \
                    (ranges <= self.lidar_config['max_range'])
        points = points[range_mask]
        
        lidar_data['points'] = points
        
        # Load extrinsics
        if poses_path.exists():
            with open(poses_path, 'r') as f:
                poses_data = json.load(f)
            
            frame_pose = poses_data[frame_idx]  # poses_data is a list, not dict
            T_lidar_to_ego = create_transformation_matrix(
                translation=[frame_pose['position']['x'], frame_pose['position']['y'], frame_pose['position']['z']],
                quaternion=[frame_pose['heading']['w'], frame_pose['heading']['x'], frame_pose['heading']['y'], frame_pose['heading']['z']]
            )
            lidar_data['extrinsics'] = T_lidar_to_ego
        
        return lidar_data
    
    def _load_annotations(self, seq_path: Path, frame_idx: int) -> Dict:
        """Load 3D bounding boxes and segmentation annotations"""
        annotations = {
            'boxes_3d': [],
            'labels': [],
            'semseg_points': np.array([])
        }
        
        # Load 3D bounding boxes
        cuboids_path = seq_path / "annotations" / "cuboids" / f"{frame_idx:02d}.pkl"
        if cuboids_path.exists():
            with open(cuboids_path, 'rb') as f:
                cuboids_df = pickle.load(f)
            
            boxes = []
            labels = []
            
            for _, row in cuboids_df.iterrows():
                # Extract 3D box parameters
                box = [
                    row['position.x'], row['position.y'], row['position.z'],
                    row['dimensions.x'], row['dimensions.y'], row['dimensions.z'],
                    row['yaw']
                ]
                boxes.append(box)
                
                # Map label to class index (simplified)
                label = self._map_label_to_class(row['label'])
                labels.append(label)
            
            annotations['boxes_3d'] = np.array(boxes) if boxes else np.empty((0, 7))
            annotations['labels'] = np.array(labels) if labels else np.array([])
        
        # Load semantic segmentation
        semseg_path = seq_path / "annotations" / "semseg" / f"{frame_idx:02d}.pkl"
        if semseg_path.exists():
            with open(semseg_path, 'rb') as f:
                semseg_df = pickle.load(f)
            annotations['semseg_points'] = semseg_df.values
        
        return annotations
    
    def _map_label_to_class(self, label: str) -> int:
        """Map PandaSet label string to class index"""
        # Simplified class mapping - you may want to expand this
        label_map = {
            'Car': 0, 'Pickup Truck': 1, 'Medium-sized Truck': 2,
            'Semi-truck': 3, 'Towed Object': 4, 'Motorcycle': 5,
            'Other Vehicle - Construction Vehicle': 6, 'Other Vehicle - Uncommon': 7,
            'Other Vehicle - Pedicab': 8, 'Emergency Vehicle': 9,
            'Bus': 10, 'Personal Mobility Device': 11, 'Motorized Scooter': 12,
            'Bicycle': 13, 'Train': 14, 'Trolley': 15, 'Tram / Subway': 16,
            'Pedestrian': 17, 'Pedestrian with Object': 18, 'Animals - Bird': 19,
            'Animals - Ground Animal': 20, 'Pylons': 21, 'Road Barriers': 22,
            'Signs': 23, 'Cones': 24, 'Construction Signs': 25,
            'Temporary Construction Barriers': 26, 'Rolling Containers': 27
        }
        return label_map.get(label, 0)  # Default to class 0
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample
        
        Returns:
            sample: Dictionary containing:
                - images: Dict of camera images
                - camera_intrinsics: Dict of camera K matrices
                - camera_extrinsics: Dict of camera extrinsics
                - camera_valid: List of valid camera names
                - lidar_points: LiDAR point cloud
                - lidar_extrinsics: LiDAR extrinsics
                - annotations: Ground truth annotations
        """
        data_info = self.data_index[idx]
        seq_path = self.root_path / data_info['sequence_id']
        frame_idx = data_info['frame_idx']
        
        # Load camera data
        camera_data = self._load_camera_data(seq_path, frame_idx)
        
        # Load LiDAR data
        lidar_data = self._load_lidar_data(seq_path, frame_idx)
        
        # Load annotations
        annotations = self._load_annotations(seq_path, frame_idx)
        
        # Prepare sample
        sample = {
            'sequence_id': data_info['sequence_id'],
            'frame_idx': frame_idx,
            
            # Camera data
            'images': camera_data['images'],
            'camera_intrinsics': camera_data['intrinsics'],
            'camera_extrinsics': camera_data['extrinsics'],
            'camera_valid': camera_data['valid_cameras'],
            
            # LiDAR data
            'lidar_points': lidar_data['points'],
            'lidar_extrinsics': lidar_data['extrinsics'],
            
            # Annotations
            'boxes_3d': annotations['boxes_3d'],
            'labels': annotations['labels'],
            'semseg_points': annotations['semseg_points']
        }
        
        # Apply transforms if specified
        if self.transform:
            sample = self.transform(sample)
        
        return sample