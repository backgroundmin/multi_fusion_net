"""
LiDAR SparseConv Encoder for Multi-Fusion-Net
Efficient 3D sparse convolution with 64→32 channel downsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import spconv.pytorch as spconv
    SPCONV_AVAILABLE = True
except ImportError:
    SPCONV_AVAILABLE = False
    print("Warning: spconv not available, using fallback implementation")

from ..utils.bev_projection import BEVProjector
from ..utils.transforms import CoordinateTransforms


class LiDARBackbone(nn.Module):
    """
    LiDAR Sparse Convolution Encoder with BEV projection
    
    Features:
    - Sparse 3D convolution for efficiency
    - 64ch → 32ch downsampling support
    - Direct BEV voxelization
    - Point cloud range filtering
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.lidar_config = config['model']['lidar_encoder']
        self.bev_config = config['bev_grid']
        
        # Voxelization parameters
        self.voxel_size = self.lidar_config['voxel_size']  # [0.2, 0.2, 8.0]
        self.point_cloud_range = self.lidar_config['point_cloud_range']
        
        # Calculate voxel grid dimensions
        self.voxel_grid_size = [
            int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]),  # X
            int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]),  # Y  
            int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2])   # Z
        ]
        
        # Initialize coordinate transforms and BEV projector
        self.coord_transforms = CoordinateTransforms(self.bev_config)
        self.bev_projector = BEVProjector(self.bev_config)
        
        # Output feature dimensions
        self.output_channels = 256
        
        if SPCONV_AVAILABLE:
            # Use SparseConv for efficient 3D processing
            self.sparse_encoder = self._build_sparse_encoder()
        else:
            # Fallback to dense voxel processing
            self.dense_encoder = self._build_dense_encoder()
    
    def _build_sparse_encoder(self) -> nn.Module:
        """Build sparse 3D convolution encoder"""
        encoder = spconv.SparseSequential(
            # Input: sparse voxels with 4 features (x, y, z, intensity)
            spconv.SubMConv3d(4, 32, kernel_size=3, padding=1, bias=False, indice_key='subm1'),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # First downsampling block
            spconv.SparseConv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # Second block
            spconv.SubMConv3d(64, 64, kernel_size=3, padding=1, bias=False, indice_key='subm2'),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # Second downsampling block
            spconv.SparseConv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Third block
            spconv.SubMConv3d(128, 128, kernel_size=3, padding=1, bias=False, indice_key='subm3'),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Final features
            spconv.SparseConv3d(128, self.output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.output_channels),
            nn.ReLU(inplace=True),
        )
        
        return encoder
    
    def _build_dense_encoder(self) -> nn.Module:
        """Build fallback dense 3D convolution encoder"""
        encoder = nn.Sequential(
            # Input: dense voxels (4, D, H, W)
            nn.Conv3d(4, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # First downsampling
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), 
            nn.ReLU(inplace=True),
            
            # Second block
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Second downsampling
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv3d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Final features
            nn.Conv3d(128, self.output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.output_channels),
            nn.ReLU(inplace=True)
        )
        
        return encoder
    
    def voxelize_points(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert point cloud to voxel representation
        
        Args:
            points: (N, 4) LiDAR points [x, y, z, intensity] in ego coordinate
            
        Returns:
            voxel_features: Voxelized features
            voxel_coords: Voxel coordinates
        """
        device = points.device
        
        if len(points) == 0:
            if SPCONV_AVAILABLE:
                return torch.empty(0, 4, device=device), torch.empty(0, 3, dtype=torch.int32, device=device)
            else:
                return torch.zeros(4, *self.voxel_grid_size, device=device), None
        
        # Filter points within range
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        range_mask = (x >= self.point_cloud_range[0]) & (x <= self.point_cloud_range[3]) & \
                    (y >= self.point_cloud_range[1]) & (y <= self.point_cloud_range[4]) & \
                    (z >= self.point_cloud_range[2]) & (z <= self.point_cloud_range[5])
        
        if range_mask.sum() == 0:
            if SPCONV_AVAILABLE:
                return torch.empty(0, 4, device=device), torch.empty(0, 3, dtype=torch.int32, device=device)
            else:
                return torch.zeros(4, *self.voxel_grid_size, device=device), None
        
        valid_points = points[range_mask]
        
        # Compute voxel indices  
        voxel_coords = torch.floor(
            (valid_points[:, :3] - torch.tensor(self.point_cloud_range[:3], device=device)) / 
            torch.tensor(self.voxel_size, device=device)
        ).int()  # Use int32 for spconv compatibility
        
        # Clamp to valid range
        voxel_coords[:, 0] = torch.clamp(voxel_coords[:, 0], 0, self.voxel_grid_size[0] - 1)
        voxel_coords[:, 1] = torch.clamp(voxel_coords[:, 1], 0, self.voxel_grid_size[1] - 1) 
        voxel_coords[:, 2] = torch.clamp(voxel_coords[:, 2], 0, self.voxel_grid_size[2] - 1)
        
        if SPCONV_AVAILABLE:
            # For sparse convolution: return unique voxels and their features
            unique_coords, inverse_indices = torch.unique(voxel_coords, return_inverse=True, dim=0)
            
            # Aggregate point features per voxel (mean pooling)
            voxel_features = torch.zeros(len(unique_coords), 4, device=device)
            for i in range(len(unique_coords)):
                point_mask = inverse_indices == i
                if point_mask.sum() > 0:
                    voxel_features[i] = valid_points[point_mask].mean(dim=0)
            
            return voxel_features, unique_coords
        
        else:
            # For dense convolution: create full voxel grid
            voxel_grid = torch.zeros(4, *self.voxel_grid_size, device=device)
            
            # Aggregate point features (mean pooling)
            for i in range(len(valid_points)):
                x_idx, y_idx, z_idx = voxel_coords[i]
                voxel_grid[:, z_idx, y_idx, x_idx] += valid_points[i]
            
            # Normalize by point count per voxel (simplified)
            voxel_grid[voxel_grid != 0] = voxel_grid[voxel_grid != 0]
            
            return voxel_grid, None
    
    def project_to_bev(self, voxel_features: torch.Tensor, voxel_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project 3D voxel features to BEV space
        
        Args:
            voxel_features: 3D voxel features 
            voxel_coords: Voxel coordinates (for sparse)
            
        Returns:
            bev_features: (output_channels, bev_h, bev_w) BEV features
        """
        device = voxel_features.device
        
        if SPCONV_AVAILABLE and voxel_coords is not None:
            # Sparse case: aggregate along Z dimension
            bev_features = torch.zeros(self.output_channels, self.voxel_grid_size[1], self.voxel_grid_size[0], device=device)
            
            for i in range(len(voxel_coords)):
                x_idx, y_idx, z_idx = voxel_coords[i]
                if 0 <= x_idx < self.voxel_grid_size[0] and 0 <= y_idx < self.voxel_grid_size[1]:
                    bev_features[:, y_idx, x_idx] += voxel_features[i]
        
        else:
            # Dense case: max pooling along Z dimension
            bev_features, _ = torch.max(voxel_features, dim=2)  # (C, H, W)
        
        # Resize to match BEV grid if needed
        target_h, target_w = self.coord_transforms.bev_h, self.coord_transforms.bev_w
        
        if bev_features.shape[1] != target_h or bev_features.shape[2] != target_w:
            bev_features = F.interpolate(
                bev_features.unsqueeze(0), 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        return bev_features
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LiDAR backbone
        
        Args:
            points: (N, 4) LiDAR points [x, y, z, intensity] in ego coordinate
            
        Returns:
            bev_features: (output_channels, bev_h, bev_w) BEV features
        """
        device = points.device
        
        # Handle empty point cloud
        if len(points) == 0:
            return torch.zeros(self.output_channels, 
                             self.coord_transforms.bev_h, 
                             self.coord_transforms.bev_w, 
                             device=device)
        
        # Voxelize point cloud
        voxel_features, voxel_coords = self.voxelize_points(points)
        
        if SPCONV_AVAILABLE and len(voxel_features) > 0:
            # Sparse convolution processing
            batch_size = 1
            voxel_coords_batch = torch.cat([
                torch.zeros(len(voxel_coords), 1, dtype=torch.int32, device=device),
                voxel_coords.int()  # Convert to int32 for spconv
            ], dim=1)  # Add batch dimension
            
            # Create sparse tensor
            sparse_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords_batch,
                spatial_shape=self.voxel_grid_size,
                batch_size=batch_size
            )
            
            # Forward through sparse encoder
            sparse_out = self.sparse_encoder(sparse_tensor)
            
            # Extract features and coordinates
            processed_features = sparse_out.features
            processed_coords = sparse_out.indices[:, 1:]  # Remove batch dimension
            
        elif len(voxel_features) > 0:
            # Dense convolution processing
            processed_features = self.dense_encoder(voxel_features.unsqueeze(0)).squeeze(0)
            processed_coords = None
            
        else:
            # Empty point cloud
            return torch.zeros(self.output_channels,
                             self.coord_transforms.bev_h,
                             self.coord_transforms.bev_w,
                             device=device)
        
        # Project to BEV space
        bev_features = self.project_to_bev(processed_features, processed_coords)
        
        return bev_features


class VoxelFeatureExtractor(nn.Module):
    """
    Alternative simple voxel feature extractor
    For systems without spconv
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.bev_config = config['bev_grid']
        self.coord_transforms = CoordinateTransforms(self.bev_config)
        
        # Simple feature extraction
        self.feature_net = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Simple BEV voxelization without 3D convolution
        
        Args:
            points: (N, 4) LiDAR points [x, y, z, intensity]
            
        Returns:
            bev_features: (256, bev_h, bev_w) BEV features
        """
        device = points.device
        
        if len(points) == 0:
            return torch.zeros(256, 
                             self.coord_transforms.bev_h, 
                             self.coord_transforms.bev_w, 
                             device=device)
        
        # Direct BEV voxelization
        bev_voxels = self.coord_transforms.bev_projector.lidar_to_bev_voxelization(
            points=points[:, :3],
            features=points[:, [3]]  # intensity only
        )
        
        # Expand to 4 channels (x, y, z, intensity statistics)
        bev_h, bev_w = bev_voxels.shape[1], bev_voxels.shape[2]
        
        # Create height and density maps
        height_map = torch.zeros(1, bev_h, bev_w, device=device)  # Mean height
        density_map = torch.zeros(1, bev_h, bev_w, device=device)  # Point density
        intensity_map = bev_voxels  # Intensity from voxelization
        occupancy_map = (intensity_map > 0).float()  # Binary occupancy
        
        # Stack features
        multi_channel = torch.cat([height_map, density_map, intensity_map, occupancy_map], dim=0)
        
        # Extract features
        features = self.feature_net(multi_channel.unsqueeze(0)).squeeze(0)
        
        return features