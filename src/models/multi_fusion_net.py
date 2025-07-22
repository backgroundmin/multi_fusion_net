"""
Multi-Fusion-Net Main Model
Complete end-to-end multi-modal perception model
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from .camera_backbone import CameraBackbone
from .lidar_backbone import LiDARBackbone, VoxelFeatureExtractor
from .fusion_module import AdaptiveFusionModule
from .detection_heads import MultiTaskHead, NMSPostProcessor


class MultiFusionNet(nn.Module):
    """
    Complete Multi-Fusion-Net model for autonomous driving perception
    
    Features:
    - Dynamic multi-camera support (0-6 cameras)
    - LiDAR sparse convolution processing  
    - Adaptive multi-modal fusion
    - Multi-task outputs (3D detection + Lane seg + Occupancy)
    - Camera dropout robustness
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Complete model configuration
        """
        super().__init__()
        
        self.config = config
        self.training_config = config.get('training', {})
        
        # Model components
        self.camera_backbone = CameraBackbone(config)
        
        # Choose LiDAR backbone based on spconv availability
        try:
            self.lidar_backbone = LiDARBackbone(config)
        except Exception:
            print("Warning: Using fallback LiDAR encoder (spconv not available)")
            self.lidar_backbone = VoxelFeatureExtractor(config)
        
        self.fusion_module = AdaptiveFusionModule(config)
        self.multi_task_head = MultiTaskHead(config)
        
        # Post-processing
        self.nms_processor = NMSPostProcessor(config)
        
        # Model state
        self.current_epoch = 0
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model
        
        Args:
            batch: Input batch containing:
                - camera_data: Dict with images, intrinsics, extrinsics, valid_cameras
                - lidar_points: (N, 4) LiDAR points
                - (targets): Ground truth for training
                
        Returns:
            outputs: Model predictions
        """
        # Extract input data
        camera_data = batch.get('camera_data', {})
        lidar_points = batch.get('lidar_points', torch.empty(0, 4))
        
        # Process camera branch
        camera_features = None
        if len(camera_data.get('valid_cameras', [])) > 0:
            camera_features = self.camera_backbone(
                camera_data=camera_data,
                lidar_points=lidar_points if self.training else None
            )
        
        # Process LiDAR branch  
        lidar_features = None
        if len(lidar_points) > 0:
            lidar_features = self.lidar_backbone(lidar_points)
        
        # Fusion
        fused_features = self.fusion_module(
            camera_features=camera_features,
            lidar_features=lidar_features
        )
        
        # Multi-task heads
        raw_outputs = self.multi_task_head(fused_features)
        
        # Post-processing during inference
        if not self.training:
            # Apply NMS to detection outputs
            detection_outputs = {
                'boxes_3d': raw_outputs['boxes_3d'],
                'scores': raw_outputs['scores'],
                'objectness': raw_outputs.get('objectness', torch.ones_like(raw_outputs['scores'][:, 0]))
            }
            
            processed_detections = self.nms_processor(detection_outputs)
            raw_outputs.update(processed_detections)
        
        return raw_outputs
    
    def set_epoch(self, epoch: int):
        """Set current training epoch (for epoch-dependent behaviors)"""
        self.current_epoch = epoch
        
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def create_model(config: Dict) -> MultiFusionNet:
    """Create Multi-Fusion-Net model from configuration"""
    model = MultiFusionNet(config)
    
    print(f"Created Multi-Fusion-Net with {model.get_num_parameters():,} parameters")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    return model


def load_pretrained_weights(model: MultiFusionNet, checkpoint_path: str):
    """Load pretrained weights from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle potential key mismatches
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    # Load matching weights
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {checkpoint_path}")


def save_checkpoint(model: MultiFusionNet, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int,
                   loss: float,
                   save_path: str,
                   best_metric: Optional[float] = None):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'best_metric': best_metric,
        'model_config': model.config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by module"""
    param_counts = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if param_count > 0:
                param_counts[name] = param_count
    
    return param_counts