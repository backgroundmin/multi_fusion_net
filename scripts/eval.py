"""
Evaluation Script for Multi-Fusion-Net
Comprehensive evaluation with metrics and visualization
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.pandaset import PandaSetDataset
from src.models.multi_fusion_net import create_model, load_pretrained_weights
from src.utils.transforms import CoordinateTransforms


class Evaluator:
    """Multi-Fusion-Net Evaluator with comprehensive metrics"""
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup model
        self.model = create_model(config)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint
        load_pretrained_weights(self.model, checkpoint_path)
        
        # Setup dataset
        self.test_dataset = PandaSetDataset(
            config=config,
            split='test'
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=self._custom_collate_fn
        )
        
        # Coordinate transforms for visualization
        self.coord_transforms = CoordinateTransforms(config['bev_grid'])
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Test dataset: {len(self.test_dataset)} samples")
    
    def _custom_collate_fn(self, batch):
        """Custom collate function (same as training)"""
        if len(batch) == 1:
            sample = batch[0]
            
            # Prepare camera data
            camera_data = {
                'images': {},
                'intrinsics': {},
                'extrinsics': {},
                'valid_cameras': sample['camera_valid']
            }
            
            # Convert images and matrices to tensors
            for cam_name in sample['images']:
                camera_data['images'][cam_name] = torch.from_numpy(
                    sample['images'][cam_name].transpose(2, 0, 1)
                ).float() / 255.0
                
                camera_data['intrinsics'][cam_name] = torch.from_numpy(
                    sample['camera_intrinsics'][cam_name]
                ).float()
                
                camera_data['extrinsics'][cam_name] = torch.from_numpy(
                    sample['camera_extrinsics'][cam_name]
                ).float()
            
            # Prepare LiDAR data
            lidar_points = torch.from_numpy(sample['lidar_points']).float()
            
            # Prepare targets
            targets = {
                'boxes_3d': torch.from_numpy(sample['boxes_3d']).float(),
                'labels': torch.from_numpy(sample['labels']).long(),
            }
            
            return {
                'camera_data': camera_data,
                'lidar_points': lidar_points,
                'targets': targets,
                'sequence_id': sample['sequence_id'],
                'frame_idx': sample['frame_idx']
            }
        
        else:
            raise NotImplementedError("Batch size > 1 not supported")
    
    def evaluate(self, save_visualizations: bool = True) -> Dict[str, float]:
        """Run comprehensive evaluation"""
        print("Starting evaluation...")
        
        all_detections = []
        all_ground_truths = []
        camera_configs = []
        inference_times = []
        
        vis_count = 0
        max_visualizations = 10
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Move to device
                batch = self._move_batch_to_device(batch)
                
                # Record camera configuration
                num_cameras = len(batch['camera_data']['valid_cameras'])
                camera_configs.append(num_cameras)
                
                # Measure inference time
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                
                # Forward pass
                outputs = self.model(batch)
                
                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)  # milliseconds
                else:
                    inference_time = 0.0
                
                inference_times.append(inference_time)
                
                # Collect results
                all_detections.append({
                    'boxes_3d': outputs['boxes_3d'].cpu(),
                    'scores': outputs['scores'].cpu(),
                    'labels': outputs['labels'].cpu()
                })
                
                all_ground_truths.append({
                    'boxes_3d': batch['targets']['boxes_3d'].cpu(),
                    'labels': batch['targets']['labels'].cpu()
                })
                
                # Save visualizations
                if save_visualizations and vis_count < max_visualizations:
                    self._save_visualization(
                        batch, outputs, 
                        save_path=f"eval_results/sample_{batch_idx}.png"
                    )
                    vis_count += 1
                
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx + 1}/{len(self.test_loader)} samples")
        
        # Compute metrics
        metrics = self._compute_metrics(all_detections, all_ground_truths)
        
        # Add inference metrics
        metrics['avg_inference_time_ms'] = np.mean(inference_times)
        metrics['fps'] = 1000.0 / metrics['avg_inference_time_ms']
        
        # Camera configuration analysis
        camera_analysis = self._analyze_camera_configs(camera_configs, all_detections)
        metrics.update(camera_analysis)
        
        # Print results
        self._print_results(metrics)
        
        return metrics
    
    def _move_batch_to_device(self, batch):
        """Move batch to device"""
        # Move camera data
        for cam_name in batch['camera_data']['images']:
            batch['camera_data']['images'][cam_name] = batch['camera_data']['images'][cam_name].to(self.device)
            batch['camera_data']['intrinsics'][cam_name] = batch['camera_data']['intrinsics'][cam_name].to(self.device)
            batch['camera_data']['extrinsics'][cam_name] = batch['camera_data']['extrinsics'][cam_name].to(self.device)
        
        # Move LiDAR data
        batch['lidar_points'] = batch['lidar_points'].to(self.device)
        
        # Move targets
        for key in batch['targets']:
            batch['targets'][key] = batch['targets'][key].to(self.device)
        
        return batch
    
    def _compute_metrics(self, detections: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        total_tp, total_fp, total_fn = 0, 0, 0
        total_detections = 0
        total_ground_truths = 0
        
        iou_threshold = 0.5
        
        for det, gt in zip(detections, ground_truths):
            det_boxes = det['boxes_3d']
            gt_boxes = gt['boxes_3d']
            
            total_detections += len(det_boxes)
            total_ground_truths += len(gt_boxes)
            
            if len(det_boxes) == 0 and len(gt_boxes) == 0:
                continue
            elif len(det_boxes) == 0:
                total_fn += len(gt_boxes)
                continue
            elif len(gt_boxes) == 0:
                total_fp += len(det_boxes)
                continue
            
            # Compute IoU matrix
            ious = self._compute_iou_matrix(det_boxes, gt_boxes)
            
            # Hungarian matching (simplified)
            matched_dets, matched_gts = self._simple_matching(ious, iou_threshold)
            
            tp = len(matched_dets)
            fp = len(det_boxes) - tp
            fn = len(gt_boxes) - len(matched_gts)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Compute metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_detections': total_detections,
            'total_ground_truths': total_ground_truths,
            'avg_detections_per_frame': total_detections / len(detections),
            'avg_ground_truths_per_frame': total_ground_truths / len(ground_truths)
        }
        
        return metrics
    
    def _compute_iou_matrix(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU matrix between detection and ground truth boxes"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return torch.zeros(len(boxes1), len(boxes2))
        
        # Convert to BEV boxes for IoU computation
        def to_bev(boxes):
            x, y, w, l = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]
            x1 = x - w / 2
            y1 = y - l / 2
            x2 = x + w / 2
            y2 = y + l / 2
            return torch.stack([x1, y1, x2, y2], dim=1)
        
        bev_boxes1 = to_bev(boxes1)
        bev_boxes2 = to_bev(boxes2)
        
        # Compute IoU
        N1, N2 = len(bev_boxes1), len(bev_boxes2)
        ious = torch.zeros(N1, N2)
        
        for i in range(N2):
            box2 = bev_boxes2[i:i+1]
            
            x1_max = torch.max(bev_boxes1[:, 0], box2[:, 0])
            y1_max = torch.max(bev_boxes1[:, 1], box2[:, 1])
            x2_min = torch.min(bev_boxes1[:, 2], box2[:, 2])
            y2_min = torch.min(bev_boxes1[:, 3], box2[:, 3])
            
            intersection = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)
            
            area1 = (bev_boxes1[:, 2] - bev_boxes1[:, 0]) * (bev_boxes1[:, 3] - bev_boxes1[:, 1])
            area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
            union = area1 + area2 - intersection
            
            ious[:, i] = intersection / (union + 1e-6)
        
        return ious
    
    def _simple_matching(self, ious: torch.Tensor, threshold: float):
        """Simple greedy matching"""
        matched_dets = []
        matched_gts = []
        
        while True:
            # Find best match
            max_iou, max_idx = torch.max(ious.flatten(), dim=0)
            
            if max_iou < threshold:
                break
            
            det_idx = max_idx // ious.shape[1]
            gt_idx = max_idx % ious.shape[1]
            
            matched_dets.append(det_idx.item())
            matched_gts.append(gt_idx.item())
            
            # Remove matched pairs
            ious[det_idx, :] = 0
            ious[:, gt_idx] = 0
        
        return matched_dets, matched_gts
    
    def _analyze_camera_configs(self, camera_configs: List[int], detections: List[Dict]) -> Dict[str, float]:
        """Analyze performance by camera configuration"""
        config_performance = {}
        
        for num_cams in range(7):  # 0-6 cameras
            indices = [i for i, c in enumerate(camera_configs) if c == num_cams]
            if len(indices) == 0:
                continue
            
            # Average number of detections for this configuration
            avg_detections = np.mean([len(detections[i]['boxes_3d']) for i in indices])
            config_performance[f'avg_detections_cam_{num_cams}'] = avg_detections
        
        return config_performance
    
    def _save_visualization(self, batch: Dict, outputs: Dict, save_path: str):
        """Save visualization of results"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: LiDAR points (top view)
        ax = axes[0, 0]
        lidar_points = batch['lidar_points'].cpu().numpy()
        if len(lidar_points) > 0:
            ax.scatter(lidar_points[:, 0], lidar_points[:, 1], c=lidar_points[:, 3], s=1, cmap='viridis')
            ax.set_title('LiDAR Points (Top View)')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.axis('equal')
        
        # Plot 2: Ground truth boxes
        ax = axes[0, 1]
        gt_boxes = batch['targets']['boxes_3d'].cpu().numpy()
        if len(gt_boxes) > 0:
            for box in gt_boxes:
                x, y, w, l = box[0], box[1], box[3], box[4]
                rect = plt.Rectangle((x - w/2, y - l/2), w, l, fill=False, color='green', linewidth=2)
                ax.add_patch(rect)
        ax.set_title('Ground Truth Boxes')
        ax.set_xlim(self.config['bev_grid']['x_range'])
        ax.set_ylim(self.config['bev_grid']['y_range'])
        ax.set_aspect('equal')
        
        # Plot 3: Predicted boxes
        ax = axes[1, 0]
        pred_boxes = outputs['boxes_3d'].cpu()
        if len(pred_boxes) > 0:
            pred_boxes_np = pred_boxes.numpy()
            for box in pred_boxes_np:
                x, y, w, l = box[0], box[1], box[3], box[4]
                rect = plt.Rectangle((x - w/2, y - l/2), w, l, fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
        ax.set_title('Predicted Boxes')
        ax.set_xlim(self.config['bev_grid']['x_range'])
        ax.set_ylim(self.config['bev_grid']['y_range'])
        ax.set_aspect('equal')
        
        # Plot 4: Occupancy map
        ax = axes[1, 1]
        if 'occupancy' in outputs:
            occupancy = outputs['occupancy'].cpu().numpy()  # (3, H, W)
            # Show occupied regions
            occupied = occupancy[1]  # Occupied channel
            ax.imshow(occupied, cmap='Reds', alpha=0.7, origin='lower')
        ax.set_title('Occupancy Map')
        
        # Add metadata
        sequence_id = batch['sequence_id'][0] if isinstance(batch['sequence_id'], list) else batch['sequence_id']
        frame_idx = batch['frame_idx'][0] if isinstance(batch['frame_idx'], list) else batch['frame_idx']
        num_cameras = len(batch['camera_data']['valid_cameras'])
        
        fig.suptitle(f'Seq: {sequence_id}, Frame: {frame_idx}, Cameras: {num_cameras}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _print_results(self, metrics: Dict[str, float]):
        """Print evaluation results"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"Detection Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        print(f"\nInference Performance:")
        print(f"  Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
        print(f"  Average FPS: {metrics['fps']:.2f}")
        
        print(f"\nDataset Statistics:")
        print(f"  Total Detections: {metrics['total_detections']}")
        print(f"  Total Ground Truths: {metrics['total_ground_truths']}")
        print(f"  Avg Detections/Frame: {metrics['avg_detections_per_frame']:.2f}")
        print(f"  Avg Ground Truths/Frame: {metrics['avg_ground_truths_per_frame']:.2f}")
        
        # Camera configuration analysis
        print(f"\nCamera Configuration Analysis:")
        for key, value in metrics.items():
            if key.startswith('avg_detections_cam_'):
                num_cams = key.split('_')[-1]
                print(f"  {num_cams} cameras: {value:.2f} avg detections")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multi-Fusion-Net')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint)
    
    # Run evaluation
    metrics = evaluator.evaluate(save_visualizations=args.visualize)
    
    # Save results
    results_path = "eval_results/metrics.yaml"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()