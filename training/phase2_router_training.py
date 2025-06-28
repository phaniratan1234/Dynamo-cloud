"""
Phase 2 Training: Dynamic Router Training
Trains the dynamic router with frozen LoRA adapters using complex loss functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Any
import os
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

from model import DynamoModel
from data import DatasetLoader, create_mixed_task_dataset, create_mixed_task_dataloader
from training.losses import DynamoLoss, create_loss_function
from utils.config import Config
from utils.logger import get_logger, TrainingLogger
from utils.helpers import (
    set_seed, count_parameters, move_to_device, AverageMeter, 
    EarlyStopping, gumbel_softmax
)

logger = get_logger(__name__)


class Phase2Trainer:
    """
    Trainer for Phase 2: Dynamic router training.
    """
    
    def __init__(self, config: Config, model: DynamoModel):
        """
        Initialize Phase 2 trainer.
        
        Args:
            config: Training configuration
            model: DYNAMO model
        """
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        
        # GPU Optimizations
        if self.device.type == 'cuda':
            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Enable mixed precision training
            self.use_amp = True
            self.scaler = GradScaler()
            logger.info("🚀 Phase 2 GPU optimizations enabled: cuDNN benchmark, mixed precision")
        else:
            self.use_amp = False
            self.scaler = None
            logger.info("⚠️  Phase 2 running on CPU - mixed precision disabled")
        
        # CRITICAL: Load Phase 1 checkpoints before setting Phase 2 mode
        self._load_phase1_checkpoints()
        
        # Set model to Phase 2 mode (this will freeze the loaded adapters)
        self.model.set_training_phase("phase2")
        self.model.to(self.device)
        
        # Initialize data loader
        self.data_loader = DatasetLoader(config.__dict__)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Router training specifics
        self.gumbel_temperature = config.training.gumbel_temperature
        self.temperature_decay = config.training.temperature_decay
        self.min_temperature = config.training.min_temperature
        
        # Curriculum learning
        self.curriculum_start_ratio = config.training.curriculum_start_ratio
        self.curriculum_end_ratio = config.training.curriculum_end_ratio
        
        # Logging
        self.training_logger = TrainingLogger(config.log_dir)
        
        logger.info("Phase 2 trainer initialized")
    
    def train_router(
        self,
        mixed_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = None,
        learning_rate: float = None
    ) -> Dict[str, float]:
        """
        Train the dynamic router.
        
        Args:
            mixed_dataloader: Mixed task training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            Training metrics
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        if learning_rate is None:
            learning_rate = self.config.training.router_lr
        
        logger.info(f"Training router for {num_epochs} epochs")
        
        # Setup optimizer and scheduler
        optimizer = self._setup_optimizer(learning_rate)
        scheduler = self._setup_scheduler(optimizer, len(mixed_dataloader) * num_epochs)
        
        # Setup loss function
        loss_fn = create_loss_function(
            self.config.training.__dict__,
            self.model.task_names,
            self.model.adapters.get_num_tasks()
        )
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=self.config.training.get('patience', 5),
            min_delta=0.001
        )
        
        # Training metrics
        best_val_loss = float('inf')
        training_metrics = {
            'train_loss': [],
            'val_loss': [],
            'routing_accuracy': [],
            'load_balance_loss': [],
            'efficiency_loss': [],
            'consistency_loss': [],
            'best_val_loss': float('inf')
        }
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.training_logger.log_epoch_start(epoch)
            
            # Update curriculum and temperature
            self._update_curriculum_and_temperature(epoch, num_epochs)
            
            # Training phase
            epoch_metrics = self._train_epoch(
                mixed_dataloader, optimizer, scheduler, loss_fn
            )
            
            # Update training metrics
            for key, value in epoch_metrics.items():
                if key in training_metrics:
                    training_metrics[key].append(value)
            
            # Validation phase
            if val_dataloader is not None:
                val_metrics = self._validate_epoch(val_dataloader, loss_fn)
                
                val_loss = val_metrics['total_loss']
                training_metrics['val_loss'].append(val_loss)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    training_metrics['best_val_loss'] = best_val_loss
                    self._save_router_checkpoint(epoch, val_loss, val_metrics)
                
                # Early stopping check
                if early_stopping(val_loss, self.model.router):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                self.training_logger.log_epoch_end(
                    epoch, epoch_metrics['train_loss'], 
                    val_loss=val_loss,
                    routing_accuracy=val_metrics.get('routing_accuracy', 0.0)
                )
            else:
                self.training_logger.log_epoch_end(epoch, epoch_metrics['train_loss'])
        
        logger.info(f"Completed router training. Best val loss: {best_val_loss:.4f}")
        return training_metrics
    
    def train_with_mixed_data(self, resume: bool = True) -> Dict[str, float]:
        """
        Train router using mixed task data.
        
        Args:
            resume: Whether to resume from existing checkpoints
            
        Returns:
            Training metrics
        """
        logger.info("Starting Phase 2: Training dynamic router")
        
        # Check for existing checkpoint
        if resume and self._check_phase2_checkpoint():
            logger.info("✅ Phase 2 already completed - loading checkpoint")
            return self._load_phase2_checkpoint()
        elif not resume:
            logger.info("Skipping checkpoint loading (resume=False)")
        
        # Load single task datasets for mixed data generation
        train_datasets = self.data_loader.create_datasets('train')
        val_datasets = self.data_loader.create_datasets('validation')
        
        # Create mixed task datasets
        mixed_train_dataset = create_mixed_task_dataset(
            train_datasets,
            self.config.__dict__,
            num_examples=self.config.data.mixed_task_size
        )
        
        mixed_val_dataset = create_mixed_task_dataset(
            val_datasets,
            self.config.__dict__,
            num_examples=self.config.data.mixed_task_size // 5  # Smaller validation set
        )
        
        # Create data loaders
        train_dataloader = create_mixed_task_dataloader(
            mixed_train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        
        val_dataloader = create_mixed_task_dataloader(
            mixed_val_dataset,
            batch_size=self.config.evaluation.eval_batch_size,
            shuffle=False
        )
        
        # Train router
        metrics = self.train_router(train_dataloader, val_dataloader)
        
        # Save final checkpoint
        self._save_phase2_checkpoint(metrics)
        
        # Log to wandb if enabled
        if self.config.use_wandb:
            self._log_wandb_metrics(metrics)
        
        logger.info("Phase 2 training completed")
        return metrics
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Any,
        loss_fn: DynamoLoss
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Metrics tracking
        loss_meters = {
            'train_loss': AverageMeter(),
            'load_balance_loss': AverageMeter(),
            'efficiency_loss': AverageMeter(),
            'consistency_loss': AverageMeter(),
            'routing_accuracy': AverageMeter()
        }
        
        progress_bar = tqdm(dataloader, desc="Training Router")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = move_to_device(batch, self.device)
            
            # Forward pass
            with autocast():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_routing_info=True
                )
            
            # Get routing information
            routing_probs = outputs['routing_probs']
            routing_info = outputs['routing_info']
            
            # Prepare targets for loss computation
            task_targets = self._prepare_task_targets(batch)
            
            # Get backbone embeddings for consistency loss
            backbone_outputs = self.model.backbone(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            cls_embeddings = backbone_outputs.last_hidden_state[:, 0, :]
            
            # Compute losses
            losses = loss_fn(
                task_outputs=outputs['task_outputs'],
                task_targets=task_targets,
                routing_probs=routing_probs,
                input_embeddings=cls_embeddings,
                temperature=routing_info.get('temperature'),
                training_phase="phase2"
            )
            
            total_loss = losses['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.router.parameters(), 
                max_norm=1.0
            )
            
            optimizer.step()
            scheduler.step()
            
            # Compute routing accuracy
            routing_accuracy = self._compute_routing_accuracy(batch, routing_probs)
            
            # Update metrics
            batch_size = batch['input_ids'].size(0)
            loss_meters['train_loss'].update(total_loss.item(), batch_size)
            loss_meters['routing_accuracy'].update(routing_accuracy, batch_size)
            
            for loss_name, loss_value in losses.items():
                if loss_name in loss_meters and loss_name != 'total_loss':
                    loss_meters[loss_name].update(loss_value.item(), batch_size)
            
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.evaluation.logging_steps == 0:
                self.training_logger.log_step(
                    total_loss.item(),
                    scheduler.get_last_lr()[0],
                    routing_accuracy=routing_accuracy,
                    temperature=self.model.router.get_temperature()
                )
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_meters['train_loss'].avg:.4f}",
                'acc': f"{loss_meters['routing_accuracy'].avg:.3f}",
                'temp': f"{self.model.router.get_temperature():.3f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Return epoch metrics
        return {key: meter.avg for key, meter in loss_meters.items()}
    
    def _validate_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: DynamoLoss
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        loss_meters = {
            'total_loss': AverageMeter(),
            'routing_accuracy': AverageMeter()
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating Router"):
                batch = move_to_device(batch, self.device)
                
                # Forward pass
                with autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        return_routing_info=True
                    )
                
                routing_probs = outputs['routing_probs']
                
                # Prepare targets
                task_targets = self._prepare_task_targets(batch)
                
                # Get backbone embeddings
                backbone_outputs = self.model.backbone(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                cls_embeddings = backbone_outputs.last_hidden_state[:, 0, :]
                
                # Compute losses
                losses = loss_fn(
                    task_outputs=outputs['task_outputs'],
                    task_targets=task_targets,
                    routing_probs=routing_probs,
                    input_embeddings=cls_embeddings,
                    training_phase="phase2"
                )
                
                # Compute routing accuracy
                routing_accuracy = self._compute_routing_accuracy(batch, routing_probs)
                
                # Update metrics
                batch_size = batch['input_ids'].size(0)
                loss_meters['total_loss'].update(losses['total_loss'].item(), batch_size)
                loss_meters['routing_accuracy'].update(routing_accuracy, batch_size)
        
        return {key: meter.avg for key, meter in loss_meters.items()}
    
    def _prepare_task_targets(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare task targets from mixed task batch."""
        task_targets = {}
        
        # Extract expected outputs for each task
        expected_outputs = batch['expected_outputs']
        
        for task_name in self.model.task_names:
            if task_name in expected_outputs:
                targets = expected_outputs[task_name]
                
                # Convert to appropriate tensor format
                if isinstance(targets, list):
                    if task_name == 'qa':
                        # QA targets are [start, end] positions
                        task_targets[task_name] = torch.stack([
                            torch.tensor(t, dtype=torch.long, device=self.device) 
                            for t in targets
                        ])
                    else:
                        task_targets[task_name] = torch.tensor(
                            targets, dtype=torch.long, device=self.device
                        )
                elif isinstance(targets, torch.Tensor):
                    task_targets[task_name] = targets.to(self.device)
        
        return task_targets
    
    def _compute_routing_accuracy(
        self, 
        batch: Dict[str, Any], 
        routing_probs: torch.Tensor
    ) -> float:
        """Compute routing accuracy based on task labels."""
        # Get predicted tasks
        predicted_tasks = torch.argmax(routing_probs, dim=-1)
        
        # Get true task labels (multi-hot to single label for accuracy)
        task_labels = batch['task_labels']  # [batch_size, num_tasks]
        true_tasks = torch.argmax(task_labels, dim=-1)
        
        # Compute accuracy
        correct = (predicted_tasks == true_tasks).float()
        accuracy = correct.mean().item()
        
        return accuracy
    
    def _update_curriculum_and_temperature(self, epoch: int, total_epochs: int):
        """Update curriculum learning and temperature annealing."""
        # Temperature annealing
        self.model.router.update_temperature(
            decay_factor=self.temperature_decay,
            min_temp=self.min_temperature
        )
        
        # Curriculum learning (could be implemented here)
        # For now, we use a simple linear schedule
        progress = epoch / total_epochs
        current_ratio = (
            self.curriculum_start_ratio * (1 - progress) + 
            self.curriculum_end_ratio * progress
        )
        
        logger.debug(f"Epoch {epoch}: Temperature={self.model.router.get_temperature():.3f}, "
                    f"Curriculum ratio={current_ratio:.3f}")
    
    def _setup_optimizer(self, learning_rate: float) -> optim.Optimizer:
        """Setup optimizer for router training."""
        # Only optimize router parameters
        optimizer = optim.AdamW(
            self.model.router.parameters(),
            lr=learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        return optimizer
    
    def _setup_scheduler(self, optimizer: optim.Optimizer, total_steps: int) -> Any:
        """Setup learning rate scheduler."""
        warmup_steps = int(total_steps * self.config.training.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return scheduler
    
    def _save_router_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]):
        """Save router checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "phase2")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'metrics': metrics,
            'router_state_dict': self.model.router.state_dict(),
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, "best_router.pt")
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved router checkpoint to {checkpoint_path}")
    
    def _save_phase2_checkpoint(self, metrics: Dict[str, float]):
        """Save complete Phase 2 checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "phase2")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save complete model state
        model_path = os.path.join(checkpoint_dir, "phase2_model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'metrics': metrics
        }, model_path)
        
        # Save training metrics
        metrics_path = os.path.join(checkpoint_dir, "training_metrics.pt")
        torch.save(metrics, metrics_path)
        
        logger.info(f"Saved Phase 2 checkpoint to {checkpoint_dir}")
    
    def _log_wandb_metrics(self, metrics: Dict[str, float]):
        """Log metrics to Weights & Biases."""
        if not self.config.use_wandb:
            return
        
        wandb_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, list) and value:
                wandb_metrics[f"phase2/{key}"] = value[-1]
            elif isinstance(value, (int, float)):
                wandb_metrics[f"phase2/{key}"] = value
        
        wandb.log(wandb_metrics, step=self.global_step)

    def _check_phase2_checkpoint(self) -> bool:
        """Check if Phase 2 checkpoint exists."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "phase2")
        checkpoint_path = os.path.join(checkpoint_dir, "phase2_model.pt")
        return os.path.exists(checkpoint_path)
    
    def _load_phase2_checkpoint(self) -> Dict[str, float]:
        """Load Phase 2 checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "phase2")
        checkpoint_path = os.path.join(checkpoint_dir, "phase2_model.pt")
        metrics_path = os.path.join(checkpoint_dir, "training_metrics.pt")
        
        try:
            # Load model state
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load metrics
            if os.path.exists(metrics_path):
                metrics = torch.load(metrics_path, map_location='cpu')
            else:
                metrics = checkpoint.get('metrics', {"status": "loaded"})
            
            logger.info(f"✅ Loaded Phase 2 checkpoint from {checkpoint_path}")
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to load Phase 2 checkpoint: {e}")
            return {"status": "failed_to_load"}

    def _load_phase1_checkpoints(self):
        """Load Phase 1 adapter checkpoints before starting Phase 2."""
        phase1_dir = os.path.join(self.config.checkpoint_dir, "phase1")
        
        # Check if Phase 1 checkpoints exist
        required_files = [
            "sentiment_adapter.pt",
            "qa_adapter.pt", 
            "summarization_adapter.pt",
            "code_generation_adapter.pt",
            "translation_adapter.pt"
        ]
        
        missing_files = []
        for filename in required_files:
            if not os.path.exists(os.path.join(phase1_dir, filename)):
                missing_files.append(filename)
        
        if missing_files:
            logger.error("❌ Phase 1 checkpoints missing! Phase 2 requires trained adapters from Phase 1.")
            logger.error(f"Missing files: {missing_files}")
            logger.error("Please complete Phase 1 training first by running:")
            logger.error("  python train_optimized.py --phase 1")
            raise FileNotFoundError(f"Phase 1 checkpoints missing: {missing_files}")
        
        # Load individual adapter checkpoints
        logger.info("📂 Loading Phase 1 adapter checkpoints...")
        adapters_loaded = 0
        
        for task_name in self.model.task_names:
            adapter_path = os.path.join(phase1_dir, f"{task_name}_adapter.pt")
            
            try:
                # Load adapter checkpoint
                adapter_checkpoint = torch.load(adapter_path, map_location=self.device)
                
                # Get the adapter and load its state
                adapter = self.model.adapters.get_adapter(task_name)
                adapter.load_state_dict(adapter_checkpoint['adapter_state_dict'])
                
                adapters_loaded += 1
                logger.info(f"  ✅ Loaded {task_name} adapter (epoch {adapter_checkpoint.get('epoch', 'unknown')})")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to load {task_name} adapter: {e}")
                raise RuntimeError(f"Failed to load {task_name} adapter from {adapter_path}")
        
        logger.info(f"🎉 Successfully loaded {adapters_loaded}/{len(self.model.task_names)} Phase 1 adapters!")
        logger.info("✅ Phase 2 is ready to train the router with frozen, trained adapters")
        
        # Verify adapters are properly loaded by checking if they have non-random weights
        self._verify_adapters_loaded()

    def _verify_adapters_loaded(self):
        """Verify that adapters have been properly loaded (not random initialization)."""
        logger.info("🔍 Verifying adapter weights are loaded correctly...")
        
        for task_name in self.model.task_names:
            adapter = self.model.adapters.get_adapter(task_name)
            
            # Check task head weights (should not be near zero if trained)
            if hasattr(adapter.task_head, '0'):  # Sequential with Linear as first layer
                first_layer = adapter.task_head[0]
            elif hasattr(adapter.task_head, 'weight'):  # Direct Linear layer
                first_layer = adapter.task_head
            else:
                continue  # Skip verification for complex heads
            
            if hasattr(first_layer, 'weight'):
                weight_mean = first_layer.weight.data.abs().mean().item()
                weight_std = first_layer.weight.data.std().item()
                
                # Random initialization usually has small variance, trained weights should have more structure
                if weight_std < 0.01:  # Very small variance suggests untrained
                    logger.warning(f"⚠️  {task_name} adapter may not be properly trained (low weight variance: {weight_std:.6f})")
                else:
                    logger.info(f"  ✅ {task_name} adapter verified (weight std: {weight_std:.4f})")


def run_phase2_training(config: Config, model: DynamoModel, resume: bool = True) -> Dict[str, float]:
    """
    Run Phase 2 training.
    
    Args:
        config: Training configuration
        model: DYNAMO model
        resume: Whether to resume from checkpoints (default: True)
    
    Returns:
        Training metrics
    """
    # Set random seed
    set_seed(config.seed)
    
    # Initialize trainer
    trainer = Phase2Trainer(config, model)
    
    # Run training
    metrics = trainer.train_with_mixed_data(resume=resume)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from utils.config import get_config
    from model import DynamoModel
    
    # Load configuration
    config = get_config()
    
    # Initialize model
    model = DynamoModel(config.__dict__)
    
    # Run Phase 2 training
    metrics = run_phase2_training(config, model)
    
    print("Phase 2 training completed!")
    print(f"Best validation loss: {metrics['best_val_loss']:.4f}")
    print(f"Final routing accuracy: {metrics['routing_accuracy'][-1]:.3f}")

