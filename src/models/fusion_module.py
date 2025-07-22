"""
Adaptive Fusion Module for Multi-Fusion-Net
Dynamic Camera + LiDAR feature fusion with camera dropout handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class AdaptiveFusionModule(nn.Module):
    """
    Adaptive multi-modal fusion with dynamic camera configuration support
    
    Features:
    - Camera + LiDAR feature fusion
    - Handles variable camera counts (0-6)
    - Attention-based adaptive fusion
    - Camera dropout robustness
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.fusion_config = config['model']['fusion']
        self.bev_config = config['bev_grid']
        
        # Feature dimensions
        self.camera_channels = 256  # From camera backbone
        self.lidar_channels = 256   # From LiDAR backbone
        self.hidden_dim = self.fusion_config['hidden_dim']
        self.output_channels = self.hidden_dim
        
        # BEV dimensions
        self.bev_h = int((self.bev_config['x_range'][1] - self.bev_config['x_range'][0]) / self.bev_config['resolution'])
        self.bev_w = int((self.bev_config['y_range'][1] - self.bev_config['y_range'][0]) / self.bev_config['resolution'])
        
        # Fusion type
        self.fusion_type = self.fusion_config['type']
        
        if self.fusion_type == "concat":
            self.fusion_layer = self._build_concat_fusion()
        elif self.fusion_type == "attention":
            self.fusion_layer = self._build_attention_fusion()
        elif self.fusion_type == "cross_attention":
            self.fusion_layer = self._build_cross_attention_fusion()
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        # Modality-specific feature processors
        self.camera_processor = nn.Sequential(
            nn.Conv2d(self.camera_channels, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.lidar_processor = nn.Sequential(
            nn.Conv2d(self.lidar_channels, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive weighting based on modality availability
        self.modality_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim // 4, 2, 1),  # 2 modalities
            nn.Sigmoid()
        )
    
    def _build_concat_fusion(self) -> nn.Module:
        """Build simple concatenation-based fusion"""
        return nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.output_channels, 3, padding=1),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_attention_fusion(self) -> nn.Module:
        """Build self-attention based fusion"""
        return SpatialAttentionFusion(self.hidden_dim, self.output_channels)
    
    def _build_cross_attention_fusion(self) -> nn.Module:
        """Build cross-attention based fusion"""
        return CrossModalAttentionFusion(self.hidden_dim, self.output_channels)
    
    def forward(self, 
                camera_features: Optional[torch.Tensor] = None,
                lidar_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with adaptive multi-modal fusion
        
        Args:
            camera_features: (C, H, W) camera BEV features (None if no cameras)
            lidar_features: (C, H, W) LiDAR BEV features (None if no LiDAR)
            
        Returns:
            fused_features: (output_channels, H, W) fused BEV features
        """
        device = next(self.parameters()).device
        
        # Handle missing modalities
        if camera_features is None and lidar_features is None:
            # No sensor data available
            return torch.zeros(self.output_channels, self.bev_h, self.bev_w, device=device)
        
        elif camera_features is None:
            # LiDAR-only mode
            return self._lidar_only_mode(lidar_features)
        
        elif lidar_features is None:
            # Camera-only mode
            return self._camera_only_mode(camera_features)
        
        else:
            # Multi-modal fusion
            return self._multi_modal_fusion(camera_features, lidar_features)
    
    def _camera_only_mode(self, camera_features: torch.Tensor) -> torch.Tensor:
        """Process camera-only features"""
        # Ensure correct spatial dimensions
        camera_features = self._resize_to_bev(camera_features)
        
        # Process camera features
        processed_camera = self.camera_processor(camera_features.unsqueeze(0)).squeeze(0)
        
        # Simple projection to output dimensions
        if processed_camera.shape[0] != self.output_channels:
            projection = nn.Conv2d(processed_camera.shape[0], self.output_channels, 1).to(camera_features.device)
            processed_camera = projection(processed_camera.unsqueeze(0)).squeeze(0)
        
        return processed_camera
    
    def _lidar_only_mode(self, lidar_features: torch.Tensor) -> torch.Tensor:
        """Process LiDAR-only features"""
        # Ensure correct spatial dimensions
        lidar_features = self._resize_to_bev(lidar_features)
        
        # Process LiDAR features
        processed_lidar = self.lidar_processor(lidar_features.unsqueeze(0)).squeeze(0)
        
        # Simple projection to output dimensions
        if processed_lidar.shape[0] != self.output_channels:
            projection = nn.Conv2d(processed_lidar.shape[0], self.output_channels, 1).to(lidar_features.device)
            processed_lidar = projection(processed_lidar.unsqueeze(0)).squeeze(0)
        
        return processed_lidar
    
    def _multi_modal_fusion(self, 
                           camera_features: torch.Tensor, 
                           lidar_features: torch.Tensor) -> torch.Tensor:
        """Perform multi-modal fusion"""
        # Ensure correct spatial dimensions
        camera_features = self._resize_to_bev(camera_features)
        lidar_features = self._resize_to_bev(lidar_features)
        
        # Process each modality
        processed_camera = self.camera_processor(camera_features.unsqueeze(0)).squeeze(0)
        processed_lidar = self.lidar_processor(lidar_features.unsqueeze(0)).squeeze(0)
        
        # Apply fusion strategy
        if self.fusion_type == "concat":
            # Concatenate features
            concat_features = torch.cat([processed_camera, processed_lidar], dim=0)
            fused_features = self.fusion_layer(concat_features.unsqueeze(0)).squeeze(0)
            
        elif self.fusion_type == "attention":
            # Self-attention fusion
            stacked_features = torch.stack([processed_camera, processed_lidar], dim=0)
            fused_features = self.fusion_layer(stacked_features)
            
        elif self.fusion_type == "cross_attention":
            # Cross-attention fusion
            fused_features = self.fusion_layer(processed_camera, processed_lidar)
        
        # Adaptive modality weighting
        fused_features = self._apply_modality_gating(
            fused_features, processed_camera, processed_lidar
        )
        
        return fused_features
    
    def _apply_modality_gating(self, 
                              fused_features: torch.Tensor,
                              camera_features: torch.Tensor,
                              lidar_features: torch.Tensor) -> torch.Tensor:
        """Apply adaptive modality weighting"""
        # Stack modality features for gating
        modality_stack = torch.cat([camera_features, lidar_features], dim=0)
        
        # Compute modality weights
        weights = self.modality_gate(modality_stack.unsqueeze(0)).squeeze(0)  # (2, 1, 1)
        camera_weight = weights[0:1]  # (1, 1, 1)
        lidar_weight = weights[1:2]   # (1, 1, 1)
        
        # Weighted combination
        weighted_features = (
            camera_weight * camera_features +
            lidar_weight * lidar_features
        )
        
        # Combine with fused features
        final_features = 0.7 * fused_features + 0.3 * weighted_features
        
        return final_features
    
    def _resize_to_bev(self, features: torch.Tensor) -> torch.Tensor:
        """Resize features to match BEV grid dimensions"""
        if features.shape[1] != self.bev_h or features.shape[2] != self.bev_w:
            features = F.interpolate(
                features.unsqueeze(0),
                size=(self.bev_h, self.bev_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return features


class SpatialAttentionFusion(nn.Module):
    """Spatial attention-based fusion for multi-modal features"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, stacked_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stacked_features: (N_modalities, C, H, W) stacked modality features
            
        Returns:
            fused_features: (C_out, H, W) fused features
        """
        N, C, H, W = stacked_features.shape
        
        # Reshape for attention: (N, H*W, C)
        tokens = stacked_features.permute(0, 2, 3, 1).reshape(N, H*W, C)
        
        # Apply self-attention across modalities
        attended_tokens, _ = self.attention(tokens, tokens, tokens)  # (N, H*W, C)
        
        # Aggregate across modalities (mean)
        fused_tokens = attended_tokens.mean(dim=0)  # (H*W, C)
        
        # Reshape back to spatial
        fused_spatial = fused_tokens.view(H, W, C).permute(2, 0, 1)  # (C, H, W)
        
        # Project to output dimensions
        output = self.output_proj(fused_spatial.unsqueeze(0)).squeeze(0)
        
        return output


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention fusion between camera and LiDAR"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Cross-attention layers
        self.cam_to_lidar_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.lidar_to_cam_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(input_dim * 2, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, 
                camera_features: torch.Tensor, 
                lidar_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_features: (C, H, W) camera features
            lidar_features: (C, H, W) LiDAR features
            
        Returns:
            fused_features: (C_out, H, W) cross-attended features
        """
        C, H, W = camera_features.shape
        
        # Reshape to tokens: (H*W, C)
        cam_tokens = camera_features.permute(1, 2, 0).reshape(H*W, C).unsqueeze(0)  # (1, H*W, C)
        lidar_tokens = lidar_features.permute(1, 2, 0).reshape(H*W, C).unsqueeze(0)  # (1, H*W, C)
        
        # Cross-attention: Camera attends to LiDAR
        cam_attended, _ = self.cam_to_lidar_attn(
            cam_tokens, lidar_tokens, lidar_tokens
        )  # (1, H*W, C)
        
        # Cross-attention: LiDAR attends to Camera
        lidar_attended, _ = self.lidar_to_cam_attn(
            lidar_tokens, cam_tokens, cam_tokens
        )  # (1, H*W, C)
        
        # Reshape back to spatial
        cam_spatial = cam_attended.squeeze(0).view(H, W, C).permute(2, 0, 1)    # (C, H, W)
        lidar_spatial = lidar_attended.squeeze(0).view(H, W, C).permute(2, 0, 1) # (C, H, W)
        
        # Concatenate cross-attended features
        cross_features = torch.cat([cam_spatial, lidar_spatial], dim=0)  # (2*C, H, W)
        
        # Project to output
        output = self.output_proj(cross_features.unsqueeze(0)).squeeze(0)
        
        return output


class FeatureAlignment(nn.Module):
    """Align features from different modalities to common representation"""
    
    def __init__(self, camera_dim: int, lidar_dim: int, output_dim: int):
        super().__init__()
        
        self.camera_align = nn.Conv2d(camera_dim, output_dim, 1)
        self.lidar_align = nn.Conv2d(lidar_dim, output_dim, 1)
        
        self.norm = nn.BatchNorm2d(output_dim)
    
    def forward(self, camera_feat: torch.Tensor, lidar_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align camera and LiDAR features to same dimensions
        
        Returns:
            aligned_camera: (output_dim, H, W)
            aligned_lidar: (output_dim, H, W)
        """
        aligned_camera = self.norm(self.camera_align(camera_feat.unsqueeze(0))).squeeze(0)
        aligned_lidar = self.norm(self.lidar_align(lidar_feat.unsqueeze(0))).squeeze(0)
        
        return aligned_camera, aligned_lidar