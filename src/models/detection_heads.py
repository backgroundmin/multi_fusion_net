"""
Multi-task Detection Heads for Multi-Fusion-Net
3D Object Detection + Lane Segmentation + Occupancy Mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class MultiTaskHead(nn.Module):
    """
    Multi-task head combining 3D detection, lane segmentation, and occupancy mapping
    
    Features:
    - 3D bounding box detection (28 classes)
    - Lane segmentation (10 lane types)  
    - Occupancy mapping (free/occupied/unknown)
    - Shared feature extraction with task-specific heads
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.bev_config = config['bev_grid']
        
        # Input feature dimensions (from fusion module)
        self.input_channels = config['model']['fusion']['hidden_dim']  # 256
        
        # BEV grid dimensions
        self.bev_h = int((self.bev_config['x_range'][1] - self.bev_config['x_range'][0]) / self.bev_config['resolution'])
        self.bev_w = int((self.bev_config['y_range'][1] - self.bev_config['y_range'][0]) / self.bev_config['resolution'])
        
        # Shared feature extractor
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Task-specific heads
        self.detection_head = Detection3DHead(config)
        self.lane_seg_head = LaneSegmentationHead(config)
        self.occupancy_head = OccupancyHead(config)
    
    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task heads
        
        Args:
            bev_features: (input_channels, bev_h, bev_w) fused BEV features
            
        Returns:
            outputs: Dictionary containing task-specific outputs
        """
        # Shared feature extraction
        shared_features = self.shared_conv(bev_features.unsqueeze(0)).squeeze(0)  # (256, H, W)
        
        # Task-specific processing
        detection_outputs = self.detection_head(shared_features)
        lane_outputs = self.lane_seg_head(shared_features)
        occupancy_outputs = self.occupancy_head(shared_features)
        
        outputs = {
            # 3D Detection outputs
            'boxes_3d': detection_outputs['boxes_3d'],
            'scores': detection_outputs['scores'],
            'labels': detection_outputs['labels'],
            'objectness': detection_outputs['objectness'],  # Missing key added
            
            # Lane Segmentation outputs
            'lane_seg': lane_outputs['lane_seg'],
            'lane_confidence': lane_outputs['confidence'],
            
            # Occupancy outputs
            'occupancy': occupancy_outputs['occupancy'],
            'occupancy_confidence': occupancy_outputs['confidence']
        }
        
        return outputs


class Detection3DHead(nn.Module):
    """3D Object Detection Head using anchor-based approach"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.detection_config = config['model']['detection_head']
        self.bev_config = config['bev_grid']
        
        # Detection parameters
        self.num_classes = self.detection_config['num_classes']  # 28
        self.anchor_sizes = self.detection_config['anchor_sizes']
        self.num_anchors = len(self.anchor_sizes)
        
        # Feature dimensions
        self.input_channels = 256
        self.hidden_channels = 128
        
        # Detection backbone
        self.det_backbone = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, 3, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.cls_head = nn.Conv2d(
            self.hidden_channels, 
            self.num_anchors * self.num_classes, 
            3, padding=1
        )
        
        # Regression head (x, y, z, w, l, h, yaw)
        self.reg_head = nn.Conv2d(
            self.hidden_channels,
            self.num_anchors * 7,  # 7 box parameters
            3, padding=1
        )
        
        # Object confidence head
        self.obj_head = nn.Conv2d(
            self.hidden_channels,
            self.num_anchors,
            3, padding=1
        )
        
        # Generate anchors
        self.anchors = self._generate_anchors()
    
    def _generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes for each BEV grid cell"""
        # Create grid coordinates
        x_range = torch.linspace(
            self.bev_config['x_range'][0] + self.bev_config['resolution'] / 2,
            self.bev_config['x_range'][1] - self.bev_config['resolution'] / 2,
            int((self.bev_config['x_range'][1] - self.bev_config['x_range'][0]) / self.bev_config['resolution'])
        )
        y_range = torch.linspace(
            self.bev_config['y_range'][0] + self.bev_config['resolution'] / 2,
            self.bev_config['y_range'][1] - self.bev_config['resolution'] / 2,
            int((self.bev_config['y_range'][1] - self.bev_config['y_range'][0]) / self.bev_config['resolution'])
        )
        
        grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
        
        # Create anchors for each grid cell and anchor size
        anchors_list = []
        
        for anchor_size in self.anchor_sizes:
            w, l, h = anchor_size
            
            # Create anchors at each grid location
            anchors = torch.zeros(len(x_range), len(y_range), 7)
            anchors[:, :, 0] = grid_x  # x center
            anchors[:, :, 1] = grid_y  # y center
            anchors[:, :, 2] = 0.0     # z center (ground level)
            anchors[:, :, 3] = w       # width
            anchors[:, :, 4] = l       # length
            anchors[:, :, 5] = h       # height
            anchors[:, :, 6] = 0.0     # yaw
            
            anchors_list.append(anchors)
        
        # Stack anchors: (num_anchors, H, W, 7)
        anchors = torch.stack(anchors_list, dim=0)
        
        return anchors
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for 3D detection
        
        Args:
            features: (256, H, W) BEV features
            
        Returns:
            outputs: Detection results
        """
        # Extract detection features
        det_features = self.det_backbone(features.unsqueeze(0)).squeeze(0)  # (128, H, W)
        
        # Prediction heads
        cls_pred = self.cls_head(det_features.unsqueeze(0)).squeeze(0)  # (num_anchors * num_classes, H, W)
        reg_pred = self.reg_head(det_features.unsqueeze(0)).squeeze(0)  # (num_anchors * 7, H, W)
        obj_pred = self.obj_head(det_features.unsqueeze(0)).squeeze(0)  # (num_anchors, H, W)
        
        # Reshape predictions
        H, W = det_features.shape[1], det_features.shape[2]
        
        cls_pred = cls_pred.view(self.num_anchors, self.num_classes, H, W)
        reg_pred = reg_pred.view(self.num_anchors, 7, H, W)
        obj_pred = obj_pred.view(self.num_anchors, H, W)
        
        # Apply activations
        cls_scores = torch.sigmoid(cls_pred)  # Class probabilities
        obj_scores = torch.sigmoid(obj_pred)  # Objectness scores
        
        # Decode boxes from regression predictions
        decoded_boxes = self._decode_boxes(reg_pred)
        
        outputs = {
            'boxes_3d': decoded_boxes,           # (num_anchors, H, W, 7)
            'scores': cls_scores,                # (num_anchors, num_classes, H, W)
            'labels': torch.argmax(cls_scores, dim=1),  # (num_anchors, H, W)
            'objectness': obj_scores             # (num_anchors, H, W)
        }
        
        return outputs
    
    def _decode_boxes(self, reg_pred: torch.Tensor) -> torch.Tensor:
        """
        Decode regression predictions to actual box coordinates
        
        Args:
            reg_pred: (num_anchors, 7, H, W) regression predictions
            
        Returns:
            decoded_boxes: (num_anchors, H, W, 7) decoded boxes
        """
        device = reg_pred.device
        anchors = self.anchors.to(device)  # (num_anchors, H, W, 7)
        
        # Decode center coordinates
        dx = reg_pred[:, 0]  # (num_anchors, H, W)
        dy = reg_pred[:, 1]
        dz = reg_pred[:, 2]
        
        decoded_x = anchors[:, :, :, 0] + dx * anchors[:, :, :, 3]  # x = anchor_x + dx * anchor_w
        decoded_y = anchors[:, :, :, 1] + dy * anchors[:, :, :, 4]  # y = anchor_y + dy * anchor_l
        decoded_z = anchors[:, :, :, 2] + dz * anchors[:, :, :, 5]  # z = anchor_z + dz * anchor_h
        
        # Decode dimensions (log scale)
        dw = reg_pred[:, 3]
        dl = reg_pred[:, 4]
        dh = reg_pred[:, 5]
        
        decoded_w = anchors[:, :, :, 3] * torch.exp(dw)
        decoded_l = anchors[:, :, :, 4] * torch.exp(dl)
        decoded_h = anchors[:, :, :, 5] * torch.exp(dh)
        
        # Decode rotation
        dyaw = reg_pred[:, 6]
        decoded_yaw = anchors[:, :, :, 6] + dyaw
        
        # Stack decoded parameters
        decoded_boxes = torch.stack([
            decoded_x, decoded_y, decoded_z,
            decoded_w, decoded_l, decoded_h,
            decoded_yaw
        ], dim=-1)  # (num_anchors, H, W, 7)
        
        return decoded_boxes


class LaneSegmentationHead(nn.Module):
    """Lane Segmentation Head for BEV lane detection"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.lane_config = config['model']['lane_seg_head']
        
        # Lane parameters
        self.num_classes = self.lane_config['num_classes']  # 10 lane types
        self.input_channels = 256
        
        # Lane segmentation network
        self.lane_net = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_classes, 3, padding=1)
        )
        
        # Confidence estimation
        self.confidence_net = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for lane segmentation
        
        Args:
            features: (256, H, W) BEV features
            
        Returns:
            outputs: Lane segmentation results
        """
        # Lane segmentation
        lane_logits = self.lane_net(features.unsqueeze(0)).squeeze(0)  # (num_classes, H, W)
        lane_seg = torch.softmax(lane_logits, dim=0)
        
        # Confidence estimation
        confidence = self.confidence_net(features.unsqueeze(0)).squeeze(0).squeeze(0)  # (H, W)
        
        outputs = {
            'lane_seg': lane_seg,         # (num_classes, H, W)
            'confidence': confidence      # (H, W)
        }
        
        return outputs


class OccupancyHead(nn.Module):
    """Occupancy Mapping Head for free/occupied/unknown classification"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.occupancy_config = config['model']['occupancy_head']
        
        # Occupancy parameters
        self.num_classes = self.occupancy_config['num_classes']  # 3: free, occupied, unknown
        self.input_channels = 256
        
        # Occupancy network
        self.occupancy_net = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_classes, 3, padding=1)
        )
        
        # Uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for occupancy mapping
        
        Args:
            features: (256, H, W) BEV features
            
        Returns:
            outputs: Occupancy mapping results
        """
        # Occupancy prediction
        occupancy_logits = self.occupancy_net(features.unsqueeze(0)).squeeze(0)  # (3, H, W)
        occupancy = torch.softmax(occupancy_logits, dim=0)
        
        # Uncertainty estimation
        confidence = self.uncertainty_net(features.unsqueeze(0)).squeeze(0).squeeze(0)  # (H, W)
        
        outputs = {
            'occupancy': occupancy,       # (3, H, W) [free, occupied, unknown]
            'confidence': confidence      # (H, W) prediction confidence
        }
        
        return outputs


class NMSPostProcessor:
    """Non-Maximum Suppression for 3D detection results"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.score_threshold = 0.3
        self.iou_threshold = 0.5
        self.max_detections = 100
    
    def __call__(self, detection_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply NMS to detection outputs
        
        Args:
            detection_outputs: Raw detection outputs
            
        Returns:
            filtered_outputs: NMS-filtered detections
        """
        boxes = detection_outputs['boxes_3d']      # (num_anchors, H, W, 7)
        scores = detection_outputs['scores']       # (num_anchors, num_classes, H, W)
        objectness = detection_outputs['objectness']  # (num_anchors, H, W)
        
        # Get dimensions
        num_anchors, num_classes, H, W = scores.shape
        
        # Combine objectness and class scores
        combined_scores = scores * objectness.unsqueeze(1)  # (num_anchors, num_classes, H, W)
        
        # Find maximum class score for each anchor
        max_scores, max_classes = torch.max(combined_scores, dim=1)  # (num_anchors, H, W)
        
        # Filter by score threshold
        valid_mask = max_scores > self.score_threshold
        
        if not valid_mask.any():
            # No valid detections
            return {
                'boxes_3d': torch.empty(0, 7),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            }
        
        # Extract valid detections
        valid_boxes = boxes[valid_mask]      # (N_valid, 7)
        valid_scores = max_scores[valid_mask]  # (N_valid,)
        valid_labels = max_classes[valid_mask]  # (N_valid,)
        
        # Sort by score
        sorted_indices = torch.argsort(valid_scores, descending=True)
        sorted_boxes = valid_boxes[sorted_indices]
        sorted_scores = valid_scores[sorted_indices]
        sorted_labels = valid_labels[sorted_indices]
        
        # Apply NMS (simplified - you might want to use torchvision.ops.nms)
        keep_indices = self._nms_3d(sorted_boxes, sorted_scores)
        
        final_boxes = sorted_boxes[keep_indices]
        final_scores = sorted_scores[keep_indices]
        final_labels = sorted_labels[keep_indices]
        
        # Limit maximum detections
        if len(final_boxes) > self.max_detections:
            final_boxes = final_boxes[:self.max_detections]
            final_scores = final_scores[:self.max_detections]
            final_labels = final_labels[:self.max_detections]
        
        return {
            'boxes_3d': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        }
    
    def _nms_3d(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Simplified 3D NMS using 2D IoU on BEV projection
        
        Args:
            boxes: (N, 7) 3D boxes [x, y, z, w, l, h, yaw]
            scores: (N,) confidence scores
            
        Returns:
            keep_indices: Indices of boxes to keep
        """
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Convert to BEV boxes for IoU computation
        bev_boxes = self._boxes_3d_to_bev(boxes)  # (N, 4) [x1, y1, x2, y2]
        
        keep = []
        remaining = torch.arange(len(boxes))
        
        while len(remaining) > 0:
            # Pick box with highest score
            best_idx = remaining[0]
            keep.append(best_idx)
            
            if len(remaining) == 1:
                break
            
            # Compute IoU with remaining boxes
            best_box = bev_boxes[best_idx:best_idx+1]  # (1, 4)
            other_boxes = bev_boxes[remaining[1:]]      # (M, 4)
            
            ious = self._compute_iou_2d(best_box, other_boxes)  # (M,)
            
            # Keep boxes with IoU < threshold
            keep_mask = ious < self.iou_threshold
            remaining = remaining[1:][keep_mask]
        
        return torch.tensor(keep, dtype=torch.long)
    
    def _boxes_3d_to_bev(self, boxes_3d: torch.Tensor) -> torch.Tensor:
        """Convert 3D boxes to BEV 2D boxes"""
        x, y, w, l = boxes_3d[:, 0], boxes_3d[:, 1], boxes_3d[:, 3], boxes_3d[:, 4]
        
        x1 = x - w / 2
        y1 = y - l / 2
        x2 = x + w / 2
        y2 = y + l / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _compute_iou_2d(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Compute 2D IoU between boxes"""
        # box1: (1, 4), box2: (N, 4)
        x1_max = torch.max(box1[:, 0], box2[:, 0])
        y1_max = torch.max(box1[:, 1], box2[:, 1])
        x2_min = torch.min(box1[:, 2], box2[:, 2])
        y2_min = torch.min(box1[:, 3], box2[:, 3])
        
        intersection = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)
        
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)