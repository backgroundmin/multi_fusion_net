"""
Multi-task Loss Functions for Multi-Fusion-Net
3D Detection + Lane Segmentation + Occupancy Mapping losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss with adaptive weighting
    
    Features:
    - 3D detection loss (classification + regression + objectness)
    - Lane segmentation loss (cross-entropy + focal)
    - Occupancy mapping loss (cross-entropy + IoU)
    - Adaptive task weighting
    - Camera dropout compensation
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Training configuration
        """
        super().__init__()
        
        self.config = config
        self.loss_weights = config['training']['loss_weights']
        
        # Individual loss components
        self.detection_loss = Detection3DLoss(config)
        self.lane_loss = LaneSegmentationLoss(config)
        self.occupancy_loss = OccupancyLoss(config)
        
        # Adaptive weighting (learned parameters)
        self.task_weights = nn.Parameter(torch.ones(3))  # [detection, lane, occupancy]
        
        # Camera dropout compensation
        self.camera_dropout_compensation = config['training']['augmentation'].get('camera_dropout_prob', 0.0)
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                num_valid_cameras: int = 6) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            num_valid_cameras: Number of active cameras (for dropout compensation)
            
        Returns:
            loss_dict: Dictionary of loss values
        """
        # Compute individual task losses
        det_loss_dict = self.detection_loss(
            predictions={
                'boxes_3d': predictions['boxes_3d'],
                'scores': predictions['scores'],
                'objectness': predictions['objectness']
            },
            targets={
                'boxes_3d': targets['boxes_3d'],
                'labels': targets['labels']
            }
        )
        
        lane_loss_dict = self.lane_loss(
            predictions={'lane_seg': predictions['lane_seg']},
            targets={'lane_seg': targets.get('lane_seg', None)}
        )
        
        occupancy_loss_dict = self.occupancy_loss(
            predictions={'occupancy': predictions['occupancy']},
            targets={'occupancy': targets.get('occupancy', None)}
        )
        
        # Apply adaptive task weighting
        task_weights_normalized = F.softmax(self.task_weights, dim=0)
        
        # Camera dropout compensation (increase weight when fewer cameras)
        camera_compensation = 1.0 + (6 - num_valid_cameras) * self.camera_dropout_compensation
        
        # Weighted task losses
        weighted_det_loss = (
            task_weights_normalized[0] * 
            camera_compensation * 
            det_loss_dict['total_loss']
        )
        
        weighted_lane_loss = (
            task_weights_normalized[1] * 
            self.loss_weights['lane_seg'] * 
            lane_loss_dict['total_loss']
        )
        
        weighted_occupancy_loss = (
            task_weights_normalized[2] * 
            self.loss_weights['occupancy'] * 
            occupancy_loss_dict['total_loss']
        )
        
        # Total loss
        total_loss = weighted_det_loss + weighted_lane_loss + weighted_occupancy_loss
        
        # Compile loss dictionary
        loss_dict = {
            'total_loss': total_loss,
            'detection_loss': weighted_det_loss,
            'lane_loss': weighted_lane_loss,
            'occupancy_loss': weighted_occupancy_loss,
            
            # Individual components
            'det_cls_loss': det_loss_dict['cls_loss'],
            'det_reg_loss': det_loss_dict['reg_loss'],
            'det_obj_loss': det_loss_dict['obj_loss'],
            'lane_seg_loss': lane_loss_dict.get('seg_loss', torch.tensor(0.0)),
            'occupancy_seg_loss': occupancy_loss_dict.get('seg_loss', torch.tensor(0.0)),
            
            # Task weights (for monitoring)
            'task_weights': task_weights_normalized,
            'camera_compensation': camera_compensation
        }
        
        return loss_dict


class Detection3DLoss(nn.Module):
    """3D Object Detection Loss (Classification + Regression + Objectness)"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.num_classes = config['model']['detection_head']['num_classes']
        
        # Loss weights
        self.cls_weight = 1.0
        self.reg_weight = 2.0
        self.obj_weight = 1.0
        
        # IoU threshold for positive/negative assignment
        self.pos_iou_threshold = 0.6
        self.neg_iou_threshold = 0.4
        
        # Focal loss parameters
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute 3D detection loss
        
        Args:
            predictions: {
                'boxes_3d': (num_anchors, H, W, 7),
                'scores': (num_anchors, num_classes, H, W),
                'objectness': (num_anchors, H, W)
            }
            targets: {
                'boxes_3d': (N, 7) ground truth boxes,
                'labels': (N,) ground truth labels
            }
            
        Returns:
            loss_dict: Detection loss components
        """
        pred_boxes = predictions['boxes_3d']      # (num_anchors, H, W, 7)
        pred_scores = predictions['scores']       # (num_anchors, num_classes, H, W)  
        pred_objectness = predictions['objectness'] # (num_anchors, H, W)
        
        gt_boxes = targets['boxes_3d']           # (N, 7)
        gt_labels = targets['labels']            # (N,)
        
        device = pred_boxes.device
        
        if len(gt_boxes) == 0:
            # No ground truth objects
            cls_loss = torch.tensor(0.0, device=device)
            reg_loss = torch.tensor(0.0, device=device)
            obj_loss = pred_objectness.mean()  # Encourage low objectness
            
            total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss + self.obj_weight * obj_loss
            
            return {
                'total_loss': total_loss,
                'cls_loss': cls_loss,
                'reg_loss': reg_loss,
                'obj_loss': obj_loss
            }
        
        # Assign anchors to ground truth boxes
        anchor_assignments = self._assign_anchors_to_gt(pred_boxes, gt_boxes, gt_labels)
        
        pos_mask = anchor_assignments['pos_mask']      # (num_anchors, H, W)
        neg_mask = anchor_assignments['neg_mask']      # (num_anchors, H, W)
        assigned_gt_boxes = anchor_assignments['assigned_gt_boxes']  # (num_anchors, H, W, 7)
        assigned_gt_labels = anchor_assignments['assigned_gt_labels'] # (num_anchors, H, W)
        
        # Classification loss (focal loss)
        cls_loss = self._compute_classification_loss(
            pred_scores, assigned_gt_labels, pos_mask, neg_mask
        )
        
        # Regression loss (smooth L1 loss)
        reg_loss = self._compute_regression_loss(
            pred_boxes, assigned_gt_boxes, pos_mask
        )
        
        # Objectness loss (binary cross entropy)
        obj_loss = self._compute_objectness_loss(
            pred_objectness, pos_mask, neg_mask
        )
        
        # Total detection loss
        total_loss = (
            self.cls_weight * cls_loss + 
            self.reg_weight * reg_loss + 
            self.obj_weight * obj_loss
        )
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'obj_loss': obj_loss
        }
    
    def _assign_anchors_to_gt(self, 
                             pred_boxes: torch.Tensor,
                             gt_boxes: torch.Tensor,
                             gt_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Assign anchors to ground truth boxes based on IoU"""
        num_anchors, H, W, _ = pred_boxes.shape
        device = pred_boxes.device
        
        # Flatten anchor boxes for IoU computation
        flat_pred_boxes = pred_boxes.view(-1, 7)  # (num_anchors * H * W, 7)
        
        # Compute IoU between all anchors and all GT boxes
        ious = self._compute_iou_3d(flat_pred_boxes, gt_boxes)  # (num_anchors * H * W, num_gt)
        
        # Find best GT for each anchor
        max_ious, max_gt_indices = torch.max(ious, dim=1)  # (num_anchors * H * W,)
        
        # Create masks
        pos_mask_flat = max_ious >= self.pos_iou_threshold
        neg_mask_flat = max_ious <= self.neg_iou_threshold
        
        # Reshape masks
        pos_mask = pos_mask_flat.view(num_anchors, H, W)
        neg_mask = neg_mask_flat.view(num_anchors, H, W)
        
        # Assign GT boxes and labels to anchors
        assigned_gt_boxes = torch.zeros_like(pred_boxes)
        assigned_gt_labels = torch.zeros(num_anchors, H, W, dtype=torch.long, device=device)
        
        # Fill assigned targets for positive anchors
        if pos_mask.sum() > 0:
            pos_indices = pos_mask_flat.nonzero().squeeze(1)
            assigned_gt_boxes.view(-1, 7)[pos_indices] = gt_boxes[max_gt_indices[pos_indices]]
            assigned_gt_labels.view(-1)[pos_indices] = gt_labels[max_gt_indices[pos_indices]]
        
        return {
            'pos_mask': pos_mask,
            'neg_mask': neg_mask,
            'assigned_gt_boxes': assigned_gt_boxes,
            'assigned_gt_labels': assigned_gt_labels
        }
    
    def _compute_iou_3d(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D IoU between two sets of boxes
        Simplified to 2D BEV IoU for efficiency
        """
        # Convert to BEV boxes (x1, y1, x2, y2)
        def to_bev(boxes):
            x, y, w, l = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]
            x1 = x - w / 2
            y1 = y - l / 2  
            x2 = x + w / 2
            y2 = y + l / 2
            return torch.stack([x1, y1, x2, y2], dim=1)
        
        bev_boxes1 = to_bev(boxes1)  # (N1, 4)
        bev_boxes2 = to_bev(boxes2)  # (N2, 4)
        
        # Compute 2D IoU
        N1, N2 = len(bev_boxes1), len(bev_boxes2)
        ious = torch.zeros(N1, N2, device=boxes1.device)
        
        for i in range(N2):
            box2 = bev_boxes2[i:i+1]  # (1, 4)
            
            # Intersection
            x1_max = torch.max(bev_boxes1[:, 0], box2[:, 0])
            y1_max = torch.max(bev_boxes1[:, 1], box2[:, 1])
            x2_min = torch.min(bev_boxes1[:, 2], box2[:, 2])
            y2_min = torch.min(bev_boxes1[:, 3], box2[:, 3])
            
            intersection = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)
            
            # Union
            area1 = (bev_boxes1[:, 2] - bev_boxes1[:, 0]) * (bev_boxes1[:, 3] - bev_boxes1[:, 1])
            area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
            union = area1 + area2 - intersection
            
            ious[:, i] = intersection / (union + 1e-6)
        
        return ious
    
    def _compute_classification_loss(self, 
                                   pred_scores: torch.Tensor,
                                   assigned_labels: torch.Tensor,
                                   pos_mask: torch.Tensor,
                                   neg_mask: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for classification"""
        num_anchors, num_classes, H, W = pred_scores.shape
        device = pred_scores.device
        
        # Create one-hot targets
        targets = torch.zeros_like(pred_scores)  # (num_anchors, num_classes, H, W)
        
        # Fill positive targets
        if pos_mask.sum() > 0:
            pos_anchors, pos_h, pos_w = pos_mask.nonzero(as_tuple=True)
            pos_labels = assigned_labels[pos_anchors, pos_h, pos_w]
            targets[pos_anchors, pos_labels, pos_h, pos_w] = 1.0
        
        # Compute focal loss
        pred_sigmoid = torch.sigmoid(pred_scores)
        ce_loss = F.binary_cross_entropy_with_logits(pred_scores, targets, reduction='none')
        
        # Focal loss weighting
        pt = torch.where(targets == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_weight = torch.where(targets == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # Only compute loss for positive and negative anchors
        valid_mask = pos_mask.unsqueeze(1) | neg_mask.unsqueeze(1)  # (num_anchors, 1, H, W)
        focal_loss = focal_loss * valid_mask
        
        return focal_loss.sum() / (valid_mask.sum() + 1e-6)
    
    def _compute_regression_loss(self,
                               pred_boxes: torch.Tensor,
                               assigned_gt_boxes: torch.Tensor,
                               pos_mask: torch.Tensor) -> torch.Tensor:
        """Compute smooth L1 loss for box regression"""
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # Extract positive predictions and targets
        pos_pred_boxes = pred_boxes[pos_mask]      # (num_pos, 7)
        pos_gt_boxes = assigned_gt_boxes[pos_mask]  # (num_pos, 7)
        
        # Compute smooth L1 loss
        reg_loss = F.smooth_l1_loss(pos_pred_boxes, pos_gt_boxes, reduction='mean')
        
        return reg_loss
    
    def _compute_objectness_loss(self,
                               pred_objectness: torch.Tensor,
                               pos_mask: torch.Tensor,
                               neg_mask: torch.Tensor) -> torch.Tensor:
        """Compute binary cross entropy for objectness"""
        # Create objectness targets
        obj_targets = torch.zeros_like(pred_objectness)
        obj_targets[pos_mask] = 1.0  # Positive anchors should have high objectness
        
        # Valid mask (positive + negative anchors)
        valid_mask = pos_mask | neg_mask
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_objectness.device)
        
        # Extract valid predictions and targets
        valid_pred = pred_objectness[valid_mask]
        valid_targets = obj_targets[valid_mask]
        
        # Binary cross entropy loss
        obj_loss = F.binary_cross_entropy_with_logits(valid_pred, valid_targets, reduction='mean')
        
        return obj_loss


class LaneSegmentationLoss(nn.Module):
    """Lane Segmentation Loss (Cross-entropy + Focal)"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.num_classes = config['model']['lane_seg_head']['num_classes']
        
        # Focal loss parameters
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        
        # Class weights (background vs lane classes)
        self.class_weights = torch.ones(self.num_classes)
        self.class_weights[0] = 0.1  # Reduce background weight
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute lane segmentation loss
        
        Args:
            predictions: {'lane_seg': (num_classes, H, W)}
            targets: {'lane_seg': (H, W)} or None
            
        Returns:
            loss_dict: Lane segmentation loss
        """
        pred_lane_seg = predictions['lane_seg']  # (num_classes, H, W)
        
        if targets.get('lane_seg') is None:
            # No lane annotation available
            return {
                'total_loss': torch.tensor(0.0, device=pred_lane_seg.device),
                'seg_loss': torch.tensor(0.0, device=pred_lane_seg.device)
            }
        
        gt_lane_seg = targets['lane_seg']  # (H, W)
        
        # Move class weights to device
        class_weights = self.class_weights.to(pred_lane_seg.device)
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            pred_lane_seg.unsqueeze(0),  # Add batch dimension
            gt_lane_seg.unsqueeze(0).long(),
            weight=class_weights,
            reduction='mean'
        )
        
        # Focal loss for hard examples
        pred_softmax = F.softmax(pred_lane_seg, dim=0)
        targets_onehot = F.one_hot(gt_lane_seg.long(), num_classes=self.num_classes).permute(2, 0, 1).float()
        
        ce_loss_pixel = F.cross_entropy(
            pred_lane_seg.unsqueeze(0),
            gt_lane_seg.unsqueeze(0).long(),
            weight=class_weights,
            reduction='none'
        ).squeeze(0)
        
        pt = torch.sum(targets_onehot * pred_softmax, dim=0)
        focal_weight = (1 - pt) ** self.focal_gamma
        focal_loss = (focal_weight * ce_loss_pixel).mean()
        
        # Combined loss
        total_loss = 0.7 * ce_loss + 0.3 * focal_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': total_loss
        }


class OccupancyLoss(nn.Module):
    """Occupancy Mapping Loss (Cross-entropy + IoU)"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.num_classes = config['model']['occupancy_head']['num_classes']  # 3
        
        # Class weights [free, occupied, unknown]
        self.class_weights = torch.tensor([1.0, 2.0, 0.5])  # Emphasize occupied regions
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute occupancy mapping loss
        
        Args:
            predictions: {'occupancy': (3, H, W)}
            targets: {'occupancy': (3, H, W)} or None
            
        Returns:
            loss_dict: Occupancy loss
        """
        pred_occupancy = predictions['occupancy']  # (3, H, W)
        
        if targets.get('occupancy') is None:
            # No occupancy annotation - can be derived from LiDAR
            return {
                'total_loss': torch.tensor(0.0, device=pred_occupancy.device),
                'seg_loss': torch.tensor(0.0, device=pred_occupancy.device)
            }
        
        gt_occupancy = targets['occupancy']  # (3, H, W)
        
        # Convert to class indices
        gt_labels = torch.argmax(gt_occupancy, dim=0)  # (H, W)
        
        # Move class weights to device
        class_weights = self.class_weights.to(pred_occupancy.device)
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            pred_occupancy.unsqueeze(0),
            gt_labels.unsqueeze(0).long(),
            weight=class_weights,
            reduction='mean'
        )
        
        # IoU loss for better boundary prediction
        iou_loss = self._compute_soft_iou_loss(pred_occupancy, gt_occupancy)
        
        # Combined loss
        total_loss = 0.6 * ce_loss + 0.4 * iou_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': total_loss
        }
    
    def _compute_soft_iou_loss(self,
                              pred: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
        """Compute soft IoU loss for occupancy"""
        pred_soft = F.softmax(pred, dim=0)  # (3, H, W)
        
        # Compute IoU for each class
        iou_losses = []
        
        for c in range(self.num_classes):
            pred_c = pred_soft[c]    # (H, W)
            target_c = target[c]     # (H, W)
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            
            iou = intersection / (union + 1e-6)
            iou_loss_c = 1 - iou
            
            iou_losses.append(iou_loss_c)
        
        # Average IoU loss across classes
        return torch.stack(iou_losses).mean()