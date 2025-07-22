"""
Dynamic Multi-Camera Backbone for Multi-Fusion-Net
Supports variable camera configurations with Lift-Splat-Shoot BEV projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import torchvision.models as models

from ..utils.bev_projection import BEVProjector, create_depth_map_from_lidar


class CameraBackbone(nn.Module):
    """
    Dynamic Multi-Camera Backbone with BEV Projection
    
    Features:
    - ResNet-based feature extraction
    - Dynamic camera count support (0-6 cameras)
    - Lift-Splat-Shoot BEV projection
    - Camera dropout robustness
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.backbone_config = config['model']['camera_backbone']
        self.bev_config = config['bev_grid']
        
        # Initialize BEV projector
        self.bev_projector = BEVProjector(self.bev_config)
        
        # Camera feature extractor (shared across all cameras)
        self.feature_extractor = self._build_backbone()
        
        # Depth estimation head (for lift-splat-shoot)
        self.depth_head = self._build_depth_head()
        
        # BEV feature dimensions
        self.bev_channels = 256  # Output BEV feature channels
        
        # Camera position encoding (ResNet50 outputs 2048 features)
        self.camera_position_embed = nn.ModuleDict({
            'front_camera': nn.Linear(2048, 64),
            'front_left_camera': nn.Linear(2048, 64),  
            'front_right_camera': nn.Linear(2048, 64),
            'left_camera': nn.Linear(2048, 64),
            'right_camera': nn.Linear(2048, 64),
            'back_camera': nn.Linear(2048, 64)
        })
        
        # BEV feature projection (2048 from ResNet50 + 64 from position encoding)
        self.bev_projection = nn.Sequential(
            nn.Conv2d(2048 + 64, self.bev_channels, 3, padding=1),
            nn.BatchNorm2d(self.bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bev_channels, self.bev_channels, 3, padding=1),
            nn.BatchNorm2d(self.bev_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism for multi-camera fusion
        self.camera_attention = nn.MultiheadAttention(
            embed_dim=self.bev_channels,
            num_heads=8,
            batch_first=True
        )
    
    def _build_backbone(self) -> nn.Module:
        """Build camera feature extractor backbone"""
        backbone_type = self.backbone_config['type']
        pretrained = self.backbone_config['pretrained']
        frozen_stages = self.backbone_config.get('frozen_stages', 0)
        
        if backbone_type == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            # Remove classification head
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            
            # Freeze early stages if specified
            if frozen_stages > 0:
                for i, child in enumerate(backbone.children()):
                    if i < frozen_stages:
                        for param in child.parameters():
                            param.requires_grad = False
        
        elif backbone_type == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        return backbone
    
    def _build_depth_head(self) -> nn.Module:
        """Build depth estimation head for lift-splat-shoot"""
        # Simple depth estimation head
        depth_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),  # ResNet50 output: 2048
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 3, padding=1),
            nn.Sigmoid()  # Output depth in [0, 1], will be scaled
        )
        return depth_head
    
    def forward(self, 
                camera_data: Dict,
                lidar_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with dynamic camera configuration
        
        Args:
            camera_data: Dictionary containing:
                - 'images': Dict of camera images {cam_name: (3, H, W)}
                - 'intrinsics': Dict of K matrices {cam_name: (3, 3)}
                - 'extrinsics': Dict of T matrices {cam_name: (4, 4)}
                - 'valid_cameras': List of active camera names
            lidar_points: Optional LiDAR points for depth supervision (N, 3)
            
        Returns:
            bev_features: (bev_channels, bev_h, bev_w) fused BEV features
        """
        device = next(self.parameters()).device
        valid_cameras = camera_data['valid_cameras']
        
        if len(valid_cameras) == 0:
            # No cameras available - return zero BEV features
            return torch.zeros(self.bev_channels, 
                             self.bev_projector.bev_h, 
                             self.bev_projector.bev_w, 
                             device=device)
        
        # Process each camera
        camera_bev_features = []
        camera_positions = []
        
        for cam_name in valid_cameras:
            if cam_name not in camera_data['images']:
                continue
                
            # Extract camera data
            image = camera_data['images'][cam_name]  # (3, H, W)
            K = camera_data['intrinsics'][cam_name]  # (3, 3)
            T = camera_data['extrinsics'][cam_name]  # (4, 4)
            
            # Extract image features
            image = image.unsqueeze(0)  # Add batch dimension
            image_features = self.feature_extractor(image)  # (1, 2048, H', W')
            image_features = image_features.squeeze(0)  # Remove batch dimension
            
            # Estimate depth
            depth_raw = self.depth_head(image_features.unsqueeze(0)).squeeze(0).squeeze(0)
            depth_map = depth_raw * 100.0  # Scale to [0, 100] meters
            
            # Optionally use LiDAR for depth supervision/initialization
            if lidar_points is not None and self.training:
                lidar_depth = self._create_lidar_depth_map(
                    lidar_points, K, T, (image.shape[3], image.shape[2])
                )
                if lidar_depth is not None:
                    # Use LiDAR depth where available
                    lidar_mask = lidar_depth > 0
                    depth_map[lidar_mask] = lidar_depth[lidar_mask]
            
            # Add positional encoding
            C, H_feat, W_feat = image_features.shape
            pos_embed = self.camera_position_embed[cam_name](
                image_features.view(C, -1).mean(dim=1)  # Global average pooling
            )  # (64,)
            pos_embed = pos_embed.view(64, 1, 1).expand(64, H_feat, W_feat)
            
            # Combine features with position encoding
            enhanced_features = torch.cat([image_features, pos_embed], dim=0)  # (2048+64, H', W')
            
            # Project to BEV
            bev_feat = self.bev_projector.camera_to_bev_projection(
                image_features=enhanced_features,
                depth_map=depth_map,
                intrinsics=K,
                extrinsics=T,
                image_size=(image.shape[3], image.shape[2])
            )
            
            camera_bev_features.append(bev_feat)
            camera_positions.append(cam_name)
        
        if len(camera_bev_features) == 0:
            return torch.zeros(self.bev_channels,
                             self.bev_projector.bev_h,
                             self.bev_projector.bev_w,
                             device=device)
        
        # Stack camera BEV features
        stacked_features = torch.stack(camera_bev_features, dim=0)  # (N_cam, C, H, W)
        
        # Apply BEV projection to get final features
        projected_features = []
        for i, feat in enumerate(stacked_features):
            proj_feat = self.bev_projection(feat.unsqueeze(0)).squeeze(0)
            projected_features.append(proj_feat)
        
        projected_stack = torch.stack(projected_features, dim=0)  # (N_cam, bev_channels, H, W)
        
        # Simple multi-camera fusion (avoid attention for memory efficiency)
        if len(projected_features) == 1:
            # Single camera
            fused_bev = projected_features[0]
        else:
            # Simple mean fusion across cameras
            fused_bev = torch.stack(projected_features, dim=0).mean(dim=0)  # (C, H, W)
        
        return fused_bev
    
    def _create_lidar_depth_map(self,
                               lidar_points: torch.Tensor,
                               intrinsics: torch.Tensor,
                               extrinsics: torch.Tensor,
                               image_size: Tuple[int, int]) -> Optional[torch.Tensor]:
        """Create depth map from LiDAR points for supervision"""
        try:
            # Convert to numpy for processing
            lidar_np = lidar_points.cpu().numpy()
            K_np = intrinsics.cpu().numpy()
            T_np = extrinsics.cpu().numpy()
            
            # Create depth map
            depth_map_np = create_depth_map_from_lidar(
                lidar_points=lidar_np,
                intrinsics=K_np,
                extrinsics=T_np,
                image_size=image_size
            )
            
            # Convert back to tensor
            device = lidar_points.device
            return torch.from_numpy(depth_map_np).to(device)
            
        except Exception as e:
            # Fallback to estimated depth if LiDAR projection fails
            return None


class LiftSplatShoot(nn.Module):
    """
    Lift-Splat-Shoot BEV transformation
    Alternative implementation focusing on explicit geometric transformation
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.bev_projector = BEVProjector(config['bev_grid'])
        
        # Depth distribution prediction (discretized depth bins)
        self.depth_bins = 64
        self.depth_min = 1.0
        self.depth_max = 100.0
        
        # Depth classifier
        self.depth_classifier = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.depth_bins, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, 
                image_features: torch.Tensor,
                intrinsics: torch.Tensor,
                extrinsics: torch.Tensor) -> torch.Tensor:
        """
        Explicit Lift-Splat-Shoot transformation
        
        Args:
            image_features: (C, H, W) image features
            intrinsics: (3, 3) camera intrinsics
            extrinsics: (4, 4) camera extrinsics
            
        Returns:
            bev_features: (C, bev_h, bev_w) BEV features
        """
        # Predict depth distribution
        depth_probs = self.depth_classifier(image_features.unsqueeze(0)).squeeze(0)  # (depth_bins, H, W)
        
        # Create depth bins
        device = image_features.device
        depth_bins = torch.linspace(self.depth_min, self.depth_max, self.depth_bins, device=device)
        
        # Expected depth
        depth_map = (depth_probs * depth_bins.view(-1, 1, 1)).sum(dim=0)  # (H, W)
        
        # Project to BEV using expected depth
        bev_features = self.bev_projector.camera_to_bev_projection(
            image_features=image_features,
            depth_map=depth_map,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_size=(image_features.shape[2], image_features.shape[1])
        )
        
        return bev_features