"""
BEV Projection utilities for Multi-Fusion-Net
Camera to BEV and LiDAR to BEV transformations
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import cv2


class BEVProjector:
    """BEV projection utilities for camera and LiDAR data"""
    
    def __init__(self, bev_config: Dict):
        """
        Args:
            bev_config: BEV grid configuration
        """
        self.x_range = bev_config['x_range']
        self.y_range = bev_config['y_range'] 
        self.z_range = bev_config['z_range']
        self.resolution = bev_config['resolution']
        
        # BEV grid dimensions
        self.bev_h = int((self.x_range[1] - self.x_range[0]) / self.resolution)  # 500
        self.bev_w = int((self.y_range[1] - self.y_range[0]) / self.resolution)  # 250
        
    def camera_to_bev_projection(self, 
                                image_features: torch.Tensor,
                                depth_map: torch.Tensor, 
                                intrinsics: torch.Tensor,
                                extrinsics: torch.Tensor,
                                image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Project camera features to BEV space using depth estimation
        
        Args:
            image_features: (C, H, W) camera feature map
            depth_map: (H, W) estimated depth map
            intrinsics: (3, 3) camera intrinsic matrix K
            extrinsics: (4, 4) camera extrinsic matrix T_cam→ego
            image_size: (width, height) original image size
            
        Returns:
            bev_features: (C, bev_h, bev_w) projected BEV features
        """
        C, H, W = image_features.shape
        device = image_features.device
        
        # Generate 2D pixel coordinates
        u_coords = torch.arange(W, device=device).float()
        v_coords = torch.arange(H, device=device).float()
        u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')
        
        # Flatten coordinates
        u_flat = u_grid.flatten()  # (H*W,)
        v_flat = v_grid.flatten()  # (H*W,)
        depth_flat = depth_map.flatten()  # (H*W,)
        
        # Filter valid depths
        valid_mask = (depth_flat > 0.1) & (depth_flat < 100.0)
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]
        
        if len(u_valid) == 0:
            return torch.zeros(C, self.bev_h, self.bev_w, device=device)
        
        # Unproject to 3D camera coordinates
        # Scale coordinates to original image size
        scale_u = image_size[0] / W
        scale_v = image_size[1] / H
        u_scaled = u_valid * scale_u
        v_scaled = v_valid * scale_v
        
        # Camera coordinates
        x_cam = (u_scaled - intrinsics[0, 2]) * depth_valid / intrinsics[0, 0]
        y_cam = (v_scaled - intrinsics[1, 2]) * depth_valid / intrinsics[1, 1]
        z_cam = depth_valid
        
        # Stack to homogeneous coordinates
        points_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(x_cam)], dim=1)
        
        # Transform to ego coordinates
        points_ego = (extrinsics @ points_cam.T).T[:, :3]
        
        # Project to BEV grid
        x_ego, y_ego, z_ego = points_ego[:, 0], points_ego[:, 1], points_ego[:, 2]
        
        # Filter points within BEV range
        bev_mask = (x_ego >= self.x_range[0]) & (x_ego <= self.x_range[1]) & \
                   (y_ego >= self.y_range[0]) & (y_ego <= self.y_range[1]) & \
                   (z_ego >= self.z_range[0]) & (z_ego <= self.z_range[1])
        
        if bev_mask.sum() == 0:
            return torch.zeros(C, self.bev_h, self.bev_w, device=device)
        
        x_bev = x_ego[bev_mask]
        y_bev = y_ego[bev_mask]
        u_indices = u_valid[bev_mask].long()
        v_indices = v_valid[bev_mask].long()
        
        # Convert to BEV grid indices
        grid_x = ((x_bev - self.x_range[0]) / self.resolution).long()
        grid_y = ((y_bev - self.y_range[0]) / self.resolution).long()
        
        # Clamp indices
        grid_x = torch.clamp(grid_x, 0, self.bev_h - 1)
        grid_y = torch.clamp(grid_y, 0, self.bev_w - 1)
        
        # Initialize BEV feature map
        bev_features = torch.zeros(C, self.bev_h, self.bev_w, device=device)
        
        # Aggregate features to BEV grid
        for i in range(len(grid_x)):
            bev_x, bev_y = grid_x[i], grid_y[i]
            u_idx, v_idx = u_indices[i], v_indices[i]
            bev_features[:, bev_x, bev_y] += image_features[:, v_idx, u_idx]
        
        return bev_features
    
    def lidar_to_bev_voxelization(self, 
                                 points: torch.Tensor,
                                 features: Optional[torch.Tensor] = None,
                                 voxel_size: List[float] = [0.2, 0.2, 8.0]) -> torch.Tensor:
        """
        Convert LiDAR point cloud to BEV voxel representation
        
        Args:
            points: (N, 3) LiDAR points in ego coordinate [x, y, z]
            features: (N, C) point features (e.g., intensity, time)
            voxel_size: [x, y, z] voxel dimensions in meters
            
        Returns:
            bev_voxels: (C, bev_h, bev_w) BEV voxel features
        """
        device = points.device
        N = len(points)
        
        if N == 0:
            return torch.zeros(1, self.bev_h, self.bev_w, device=device)
        
        # Default features (intensity/density)
        if features is None:
            features = torch.ones(N, 1, device=device)
        
        C = features.shape[1]
        
        # Filter points within BEV range
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        range_mask = (x >= self.x_range[0]) & (x <= self.x_range[1]) & \
                    (y >= self.y_range[0]) & (y <= self.y_range[1]) & \
                    (z >= self.z_range[0]) & (z <= self.z_range[1])
        
        if range_mask.sum() == 0:
            return torch.zeros(C, self.bev_h, self.bev_w, device=device)
        
        valid_points = points[range_mask]
        valid_features = features[range_mask]
        
        # Compute voxel indices
        grid_x = ((valid_points[:, 0] - self.x_range[0]) / self.resolution).long()
        grid_y = ((valid_points[:, 1] - self.y_range[0]) / self.resolution).long()
        
        # Clamp indices
        grid_x = torch.clamp(grid_x, 0, self.bev_h - 1)
        grid_y = torch.clamp(grid_y, 0, self.bev_w - 1)
        
        # Initialize BEV voxel map
        bev_voxels = torch.zeros(C, self.bev_h, self.bev_w, device=device)
        
        # Aggregate point features to voxels
        for c in range(C):
            bev_voxels[c].index_put_(
                indices=(grid_x, grid_y),
                values=valid_features[:, c],
                accumulate=True
            )
        
        return bev_voxels
    
    def create_occupancy_map(self, 
                           camera_points: List[torch.Tensor],
                           lidar_points: torch.Tensor,
                           camera_valid: List[bool]) -> torch.Tensor:
        """
        Create occupancy map from camera and LiDAR data
        
        Args:
            camera_points: List of camera point clouds in ego coordinate
            lidar_points: (N, 3) LiDAR points in ego coordinate
            camera_valid: List of valid camera flags
            
        Returns:
            occupancy_map: (3, bev_h, bev_w) [free, occupied, unknown]
        """
        device = lidar_points.device
        occupancy_map = torch.zeros(3, self.bev_h, self.bev_w, device=device)
        
        # LiDAR-based occupancy (most reliable)
        if len(lidar_points) > 0:
            lidar_bev = self.lidar_to_bev_voxelization(lidar_points)
            lidar_mask = lidar_bev[0] > 0  # Points exist
            
            # Mark occupied cells
            occupancy_map[1][lidar_mask] = 1.0
            
            # Mark free space (ray tracing from ego to points)
            ego_pos = torch.tensor([0.0, 0.0], device=device)
            for i in range(len(lidar_points)):
                point = lidar_points[i, :2]  # x, y
                free_cells = self._trace_ray(ego_pos, point)
                if len(free_cells) > 0:
                    occupancy_map[0][free_cells[:, 0], free_cells[:, 1]] = 1.0
        
        # Camera-based occupancy (complementary)
        for i, (cam_points, is_valid) in enumerate(zip(camera_points, camera_valid)):
            if is_valid and len(cam_points) > 0:
                cam_bev = self.lidar_to_bev_voxelization(cam_points)
                cam_mask = cam_bev[0] > 0
                
                # Add to occupied if not already marked as free
                occupancy_map[1][cam_mask & (occupancy_map[0] == 0)] = 1.0
        
        # Mark unknown regions (not observed by any sensor)
        observed_mask = (occupancy_map[0] > 0) | (occupancy_map[1] > 0)
        occupancy_map[2] = (~observed_mask).float()
        
        # Normalize to probabilities
        occupancy_sum = occupancy_map.sum(dim=0, keepdim=True)
        occupancy_map = occupancy_map / (occupancy_sum + 1e-6)
        
        return occupancy_map
    
    def _trace_ray(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """
        Trace ray from start to end point and return BEV grid cells
        Simple Bresenham-like algorithm for ray tracing
        
        Args:
            start: (2,) start point [x, y] in ego coordinate
            end: (2,) end point [x, y] in ego coordinate
            
        Returns:
            cells: (M, 2) BEV grid cells along the ray
        """
        device = start.device
        
        # Convert to BEV grid coordinates
        start_grid = ((start - torch.tensor([self.x_range[0], self.y_range[0]], device=device)) / self.resolution).long()
        end_grid = ((end - torch.tensor([self.x_range[0], self.y_range[0]], device=device)) / self.resolution).long()
        
        # Clamp to valid range
        start_grid = torch.clamp(start_grid, torch.tensor([0, 0], device=device), 
                                torch.tensor([self.bev_h-1, self.bev_w-1], device=device))
        end_grid = torch.clamp(end_grid, torch.tensor([0, 0], device=device),
                              torch.tensor([self.bev_h-1, self.bev_w-1], device=device))
        
        # Simple linear interpolation (can be improved with Bresenham)
        num_steps = max(abs(end_grid[0] - start_grid[0]), abs(end_grid[1] - start_grid[1]))
        if num_steps == 0:
            return start_grid.unsqueeze(0)
        
        steps = torch.linspace(0, 1, num_steps + 1, device=device)
        ray_points = start_grid.unsqueeze(0) + steps.unsqueeze(1) * (end_grid - start_grid).unsqueeze(0)
        ray_cells = ray_points.long()
        
        # Remove duplicates and return unique cells
        unique_cells = torch.unique(ray_cells, dim=0)
        
        return unique_cells


def create_depth_map_from_lidar(lidar_points: np.ndarray,
                               intrinsics: np.ndarray,
                               extrinsics: np.ndarray,
                               image_size: Tuple[int, int]) -> np.ndarray:
    """
    Create depth map by projecting LiDAR points to camera image
    
    Args:
        lidar_points: (N, 3) LiDAR points in ego coordinate
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (4, 4) camera extrinsic matrix T_cam→ego
        image_size: (width, height) image dimensions
        
    Returns:
        depth_map: (height, width) depth map in meters
    """
    if len(lidar_points) == 0:
        return np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    # Transform LiDAR points to camera coordinate
    T_ego_to_cam = np.linalg.inv(extrinsics)
    points_homo = np.concatenate([lidar_points, np.ones((len(lidar_points), 1))], axis=1)
    points_cam = (T_ego_to_cam @ points_homo.T).T[:, :3]
    
    # Filter points in front of camera
    valid_mask = points_cam[:, 2] > 0.1
    if valid_mask.sum() == 0:
        return np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    valid_points = points_cam[valid_mask]
    
    # Project to image plane
    points_2d_homo = (intrinsics @ valid_points.T).T
    points_2d = points_2d_homo[:, :2] / (points_2d_homo[:, 2:3] + 1e-6)
    depths = valid_points[:, 2]
    
    # Filter points within image bounds
    u_coords = points_2d[:, 0]
    v_coords = points_2d[:, 1]
    
    image_mask = (u_coords >= 0) & (u_coords < image_size[0]) & \
                 (v_coords >= 0) & (v_coords < image_size[1])
    
    if image_mask.sum() == 0:
        return np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    u_valid = u_coords[image_mask].astype(int)
    v_valid = v_coords[image_mask].astype(int)
    depths_valid = depths[image_mask]
    
    # Create depth map
    depth_map = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    # Fill depth values (take minimum depth for overlapping pixels)
    for i in range(len(u_valid)):
        u, v, d = u_valid[i], v_valid[i], depths_valid[i]
        if depth_map[v, u] == 0 or depth_map[v, u] > d:
            depth_map[v, u] = d
    
    return depth_map