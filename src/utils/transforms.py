"""
Coordinate transformation utilities for Multi-Fusion-Net
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
from pyquaternion import Quaternion


class CoordinateTransforms:
    """Coordinate transformation utilities for multi-modal sensor fusion"""
    
    def __init__(self, bev_config: Dict):
        """
        Args:
            bev_config: BEV grid configuration from config.yaml
        """
        self.x_range = bev_config['x_range']
        self.y_range = bev_config['y_range'] 
        self.z_range = bev_config['z_range']
        self.resolution = bev_config['resolution']
        
        # BEV grid dimensions
        self.bev_h = int((self.x_range[1] - self.x_range[0]) / self.resolution)  # 500
        self.bev_w = int((self.y_range[1] - self.y_range[0]) / self.resolution)  # 250
        
    def ego_to_bev(self, points_ego: np.ndarray) -> np.ndarray:
        """
        Transform points from ego vehicle coordinate to BEV grid coordinate
        
        Args:
            points_ego: (N, 3) points in ego coordinate [x, y, z]
            
        Returns:
            points_bev: (N, 2) points in BEV grid coordinate [u, v]
        """
        x, y = points_ego[:, 0], points_ego[:, 1]
        
        # Convert to BEV grid indices
        u = (x - self.x_range[0]) / self.resolution  # Forward axis
        v = (y - self.y_range[0]) / self.resolution  # Right axis
        
        return np.stack([u, v], axis=1)
    
    def bev_to_ego(self, points_bev: np.ndarray) -> np.ndarray:
        """
        Transform points from BEV grid coordinate to ego vehicle coordinate
        
        Args:
            points_bev: (N, 2) points in BEV grid coordinate [u, v]
            
        Returns:
            points_ego: (N, 2) points in ego coordinate [x, y]
        """
        u, v = points_bev[:, 0], points_bev[:, 1]
        
        x = u * self.resolution + self.x_range[0]
        y = v * self.resolution + self.y_range[0]
        
        return np.stack([x, y], axis=1)
    
    def camera_to_ego(self, points_cam: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """
        Transform points from camera coordinate to ego coordinate
        
        Args:
            points_cam: (N, 3) points in camera coordinate
            extrinsics: (4, 4) transformation matrix T_cam→ego
            
        Returns:
            points_ego: (N, 3) points in ego coordinate
        """
        # Add homogeneous coordinate
        points_homo = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
        
        # Transform to ego coordinate
        points_ego_homo = (extrinsics @ points_homo.T).T
        
        return points_ego_homo[:, :3]
    
    def lidar_to_ego(self, points_lidar: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """
        Transform points from LiDAR coordinate to ego coordinate
        
        Args:
            points_lidar: (N, 3) points in LiDAR coordinate
            extrinsics: (4, 4) transformation matrix T_lidar→ego
            
        Returns:
            points_ego: (N, 3) points in ego coordinate
        """
        return self.camera_to_ego(points_lidar, extrinsics)
    
    def project_to_image(self, points_3d: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D image plane
        
        Args:
            points_3d: (N, 3) 3D points in camera coordinate
            intrinsics: (3, 3) camera intrinsic matrix K
            
        Returns:
            points_2d: (N, 2) 2D points in image coordinate [u, v]
            valid_mask: (N,) boolean mask for points in front of camera
        """
        # Filter points in front of camera
        valid_mask = points_3d[:, 2] > 0.1
        
        # Perspective projection
        points_homo = (intrinsics @ points_3d.T).T
        points_2d = points_homo[:, :2] / (points_homo[:, 2:3] + 1e-6)
        
        return points_2d, valid_mask
    
    def downsample_lidar_channels(self, points: np.ndarray, 
                                original_channels: int = 64,
                                target_channels: int = 32,
                                method: str = "uniform") -> np.ndarray:
        """
        Downsample LiDAR from 64 channels to 32 channels
        
        Args:
            points: (N, 4) LiDAR points [x, y, z, intensity]
            original_channels: Original number of channels (64)
            target_channels: Target number of channels (32)  
            method: Downsampling method ("uniform" or "density_based")
            
        Returns:
            downsampled_points: (M, 4) downsampled LiDAR points
        """
        if len(points) == 0:
            return points
            
        # Calculate vertical angles for channel assignment
        ranges = np.linalg.norm(points[:, :2], axis=1)
        vertical_angles = np.arctan2(points[:, 2], ranges)
        
        # Assign points to channels based on vertical angles
        angle_min, angle_max = vertical_angles.min(), vertical_angles.max()
        channel_indices = ((vertical_angles - angle_min) / (angle_max - angle_min) * 
                          (original_channels - 1)).astype(int)
        
        if method == "uniform":
            # Select every 2nd channel (64 → 32)
            selected_channels = np.arange(0, original_channels, 2)
        elif method == "density_based":
            # Keep channels with higher point density (near horizon)
            channel_counts = np.bincount(channel_indices, minlength=original_channels)
            selected_channels = np.argsort(channel_counts)[-target_channels:]
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
        
        # Filter points belonging to selected channels
        mask = np.isin(channel_indices, selected_channels)
        return points[mask]


def create_transformation_matrix(translation: List[float], 
                               quaternion: List[float]) -> np.ndarray:
    """
    Create 4x4 transformation matrix from translation and quaternion
    
    Args:
        translation: [x, y, z] translation vector
        quaternion: [w, x, y, z] quaternion (w first)
        
    Returns:
        T: (4, 4) transformation matrix
    """
    T = np.eye(4)
    
    # Set rotation part
    q = Quaternion(quaternion)
    T[:3, :3] = q.rotation_matrix
    
    # Set translation part
    T[:3, 3] = translation
    
    return T