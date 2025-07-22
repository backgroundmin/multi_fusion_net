"""
Training Script for Multi-Fusion-Net
AMP + Dynamic Camera Dropout + Multi-task Learning
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.pandaset import PandaSetDataset
from src.models.multi_fusion_net import create_model, save_checkpoint, load_pretrained_weights
from src.losses.multi_task_loss import MultiTaskLoss
from src.utils.transforms import CoordinateTransforms


class Trainer:
    """Multi-Fusion-Net Trainer with AMP and multi-task learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        
        # AMP setup
        self.use_amp = config['training']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging setup
        self.use_wandb = config['logging']['use_wandb']
        self.log_interval = config['logging']['log_interval']
        
        # Validation parameters
        self.eval_interval = config['validation']['eval_interval']
        self.save_interval = config['validation']['save_interval']
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
    def _setup_data(self):
        """Setup datasets and data loaders"""
        print("Setting up datasets...")
        
        # Create datasets
        self.train_dataset = PandaSetDataset(
            config=self.config,
            split='train'
        )
        
        self.val_dataset = PandaSetDataset(
            config=self.config,
            split='val'
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config['device']['num_workers'],
            pin_memory=self.config['device']['pin_memory'],
            collate_fn=self._custom_collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # Validation with batch size 1 for simplicity
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=self._custom_collate_fn
        )
        
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Val dataset: {len(self.val_dataset)} samples")
        
    def _custom_collate_fn(self, batch):
        """Custom collate function for variable-sized data"""
        # For simplicity, process batch size 1
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
                ).float() / 255.0  # Normalize to [0, 1]
                
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
            
            # Add synthetic occupancy map from LiDAR (simplified)
            coord_transforms = CoordinateTransforms(self.config['bev_grid'])
            if len(lidar_points) > 0:
                # Create simple occupancy from LiDAR points
                occupancy = self._create_occupancy_from_lidar(lidar_points, coord_transforms)
                targets['occupancy'] = occupancy
            
            return {
                'camera_data': camera_data,
                'lidar_points': lidar_points,
                'targets': targets,
                'sequence_id': sample['sequence_id'],
                'frame_idx': sample['frame_idx']
            }
        
        else:
            # Handle larger batches (more complex implementation needed)
            raise NotImplementedError("Batch size > 1 requires more complex collate function")
    
    def _create_occupancy_from_lidar(self, lidar_points, coord_transforms):
        """Create simple occupancy map from LiDAR points"""
        bev_h, bev_w = coord_transforms.bev_h, coord_transforms.bev_w
        occupancy = torch.zeros(3, bev_h, bev_w)  # [free, occupied, unknown]
        
        if len(lidar_points) > 0:
            # Project LiDAR points to BEV grid
            points_ego = lidar_points[:, :3].numpy()
            points_bev = coord_transforms.ego_to_bev(points_ego)
            
            # Filter points within BEV bounds
            valid_mask = (points_bev[:, 0] >= 0) & (points_bev[:, 0] < bev_h) & \
                        (points_bev[:, 1] >= 0) & (points_bev[:, 1] < bev_w)
            
            if valid_mask.sum() > 0:
                valid_points = points_bev[valid_mask].astype(int)
                
                # Mark occupied cells
                occupancy[1, valid_points[:, 0], valid_points[:, 1]] = 1.0
                
                # Mark remaining as unknown (simplified)
                occupancy[2] = 1.0 - occupancy[1]
        
        else:
            # No LiDAR data - mark all as unknown
            occupancy[2] = 1.0
        
        return occupancy
    
    def _setup_model(self):
        """Setup model and loss function"""
        print("Setting up model...")
        
        # Create model
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.criterion = MultiTaskLoss(self.config)
        self.criterion = self.criterion.to(self.device)
        
        # Load pretrained weights if specified
        pretrained_path = self.config.get('model', {}).get('pretrained_path')
        if pretrained_path and os.path.exists(pretrained_path):
            load_pretrained_weights(self.model, pretrained_path)
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        print("Setting up optimizer...")
        
        # Optimizer
        optimizer_type = self.config['training']['optimizer']
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Scheduler
        scheduler_type = self.config['training']['scheduler']
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.num_epochs // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
    
    def _setup_logging(self):
        """Setup logging with wandb"""
        if self.use_wandb:
            wandb.init(
                project=self.config['logging']['project_name'],
                config=self.config,
                name=f"multi_fusion_net_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            wandb.watch(self.model, log='all', log_freq=1000)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(batch)
                
                # Compute number of valid cameras for dropout compensation
                num_valid_cameras = len(batch['camera_data']['valid_cameras'])
                
                # Compute loss
                loss_dict = self.criterion(
                    predictions=outputs,
                    targets=batch['targets'],
                    num_valid_cameras=num_valid_cameras
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss_dict['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total_loss'].backward()
                self.optimizer.step()
            
            # Update metrics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    # Handle both scalar and non-scalar tensors
                    if value.numel() == 1:
                        value = value.item()
                    else:
                        value = value.mean().item()  # Take mean for multi-element tensors
                
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)
            
            # Log batch metrics
            if batch_idx % self.log_interval == 0:
                self._log_batch_metrics(batch_idx, loss_dict, num_valid_cameras)
            
            self.global_step += 1
        
        # Compute epoch averages
        for key, values in epoch_losses.items():
            if isinstance(values[0], (int, float)):
                epoch_metrics[f'train_{key}'] = sum(values) / len(values)
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {}
        val_metrics = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute number of valid cameras
                num_valid_cameras = len(batch['camera_data']['valid_cameras'])
                
                # Compute loss
                loss_dict = self.criterion(
                    predictions=outputs,
                    targets=batch['targets'],
                    num_valid_cameras=num_valid_cameras
                )
                
                # Update metrics
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value)
        
        # Compute validation averages
        for key, values in val_losses.items():
            if isinstance(values[0], (int, float)):
                val_metrics[f'val_{key}'] = sum(values) / len(values)
        
        return val_metrics
    
    def _move_batch_to_device(self, batch):
        """Move batch tensors to device"""
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
    
    def _log_batch_metrics(self, batch_idx, loss_dict, num_valid_cameras):
        """Log batch metrics"""
        lr = self.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {self.current_epoch}, Batch {batch_idx}: "
              f"Loss {loss_dict['total_loss']:.4f}, "
              f"Cameras {num_valid_cameras}, "
              f"LR {lr:.2e}")
        
        if self.use_wandb:
            log_dict = {
                'batch/total_loss': loss_dict['total_loss'],
                'batch/detection_loss': loss_dict['detection_loss'],
                'batch/lane_loss': loss_dict['lane_loss'],
                'batch/occupancy_loss': loss_dict['occupancy_loss'],
                'batch/num_cameras': num_valid_cameras,
                'batch/learning_rate': lr,
                'batch/step': self.global_step
            }
            wandb.log(log_dict)
    
    def _log_epoch_metrics(self, train_metrics, val_metrics=None):
        """Log epoch metrics"""
        print(f"\nEpoch {self.current_epoch} Results:")
        print(f"Train Loss: {train_metrics.get('train_total_loss', 0):.4f}")
        
        if val_metrics:
            print(f"Val Loss: {val_metrics.get('val_total_loss', 0):.4f}")
        
        if self.use_wandb:
            log_dict = {**train_metrics}
            if val_metrics:
                log_dict.update(val_metrics)
            log_dict['epoch'] = self.current_epoch
            wandb.log(log_dict)
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            self.model.set_epoch(epoch)
            
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = None
            if epoch % self.eval_interval == 0:
                print("Running validation...")
                val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            if epoch % self.save_interval == 0:
                checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
                os.makedirs("checkpoints", exist_ok=True)
                
                current_metric = val_metrics.get('val_total_loss', train_metrics.get('train_total_loss', float('inf')))
                is_best = current_metric < self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=current_metric,
                    save_path=checkpoint_path,
                    best_metric=self.best_metric
                )
                
                if is_best:
                    best_path = "checkpoints/best_model.pth"
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        loss=current_metric,
                        save_path=best_path,
                        best_metric=self.best_metric
                    )
        
        print("\nTraining completed!")
        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Fusion-Net')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_metric = checkpoint.get('best_metric', float('inf'))
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()