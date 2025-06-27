"""
Phase 1 Training: Individual LoRA Adapter Training
Trains each LoRA adapter independently on task-specific data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Any
import os
from tqdm import tqdm
import wandb
from pathlib import Path

from model import DynamoModel
from data import DatasetLoader
from training.losses import TaskSpecificLoss
from utils.config import Config
from utils.logger import get_logger, TrainingLogger
from utils.helpers import (
    set_seed, count_parameters, move_to_device, AverageMeter, EarlyStopping
)

logger = get_logger(__name__)


class Phase1Trainer:
    """
    Trainer for Phase 1: Individual LoRA adapter training.
    """
    
    def __init__(self, config: Config, model: DynamoModel):
        """
        Initialize Phase 1 trainer.
        
        Args:
            config: Training configuration
            model: DYNAMO model
        """
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        
        # GPU Optimizations
        if self.device.type == 'cuda' and torch.cuda.is_available():
            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Enable mixed precision training
            self.use_amp = True
            self.scaler = GradScaler()
            logger.info("🚀 GPU optimizations enabled: cuDNN benchmark, mixed precision")
        else:
            self.use_amp = False
            self.scaler = None
            logger.info("⚠️  Running on CPU - mixed precision disabled")
        
        # Set model to Phase 1 mode
        self.model.set_training_phase("phase1")
        self.model.to(self.device)
        
        # Initialize data loader
        self.data_loader = DatasetLoader(config.__dict__)
        
        # Training state
        self.current_task = None
        self.current_epoch = 0
        self.global_step = 0
        
        # Logging
        self.training_logger = TrainingLogger(config.log_dir)
        
        logger.info("Phase 1 trainer initialized")
    
    def train_single_adapter(
        self,
        task_name: str,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = None,
        learning_rate: float = None
    ) -> Dict[str, float]:
        """
        Train a single LoRA adapter for a specific task.
        
        Args:
            task_name: Name of the task
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            Training metrics
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        if learning_rate is None:
            learning_rate = self.config.training.lora_lr
        
        self.current_task = task_name
        logger.info(f"Training {task_name} adapter for {num_epochs} epochs")
        
        # Get the specific adapter
        adapter = self.model.adapters.get_adapter(task_name)
        
        # Freeze all other adapters
        self.model.adapters.freeze_adapters()
        self.model.adapters.unfreeze_adapters([task_name])
        
        # Setup optimizer and scheduler
        optimizer = self._setup_optimizer(adapter, learning_rate)
        scheduler = self._setup_scheduler(optimizer, len(train_dataloader) * num_epochs)
        
        # Setup loss function
        loss_fn = TaskSpecificLoss(task_name)
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=getattr(self.config.training, 'patience', 3),
            min_delta=0.001
        )
        
        # Training metrics
        best_val_loss = float('inf')
        training_metrics = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf')
        }
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.training_logger.log_epoch_start(epoch)
            
            # Training phase
            train_loss = self._train_epoch(
                train_dataloader, optimizer, scheduler, loss_fn, task_name
            )
            training_metrics['train_loss'].append(train_loss)
            
            # Validation phase
            if val_dataloader is not None:
                val_loss = self._validate_epoch(val_dataloader, loss_fn, task_name)
                training_metrics['val_loss'].append(val_loss)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    training_metrics['best_val_loss'] = best_val_loss
                    self._save_adapter_checkpoint(task_name, epoch, val_loss)
                
                # Early stopping check
                if early_stopping(val_loss, adapter):
                    logger.info(f"Early stopping triggered for {task_name} at epoch {epoch}")
                    break
                
                self.training_logger.log_epoch_end(epoch, train_loss, val_loss=val_loss)
            else:
                self.training_logger.log_epoch_end(epoch, train_loss)
        
        logger.info(f"Completed training {task_name} adapter. Best val loss: {best_val_loss:.4f}")
        return training_metrics
    
    def train_all_adapters(self, resume: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Train all LoRA adapters sequentially.
        
        Args:
            resume: Whether to resume from existing checkpoints
            
        Returns:
            Training metrics for all adapters
        """
        logger.info("Starting Phase 1: Training all LoRA adapters")
        
        # Check for existing checkpoints only if resume is enabled
        completed_adapters = []
        if resume:
            completed_adapters = self._check_existing_checkpoints()
            if completed_adapters:
                logger.info(f"Found existing checkpoints for: {completed_adapters}")
                self._load_existing_checkpoints(completed_adapters)
            else:
                logger.info("No existing checkpoints found")
        else:
            logger.info("Skipping checkpoint loading (resume=False)")
        
        # Load datasets
        train_datasets = self.data_loader.create_datasets('train')
        val_datasets = self.data_loader.create_datasets('validation')
        
        # Create data loaders
        train_dataloaders = self.data_loader.create_dataloaders(
            train_datasets,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        val_dataloaders = self.data_loader.create_dataloaders(
            val_datasets,
            batch_size=self.config.evaluation.eval_batch_size,
            shuffle=False
        )
        
        all_metrics = {}
        
        # Train each adapter (skip completed ones)
        for task_name in self.model.task_names:
            if task_name in completed_adapters:
                logger.info(f"⏭️  Skipping {task_name} (already trained)")
                all_metrics[task_name] = {"status": "skipped", "checkpoint_loaded": True}
                continue
                
            logger.info(f"{'='*50}")
            logger.info(f"Training {task_name} adapter")
            logger.info(f"{'='*50}")
            
            if task_name in train_dataloaders:
                train_dl = train_dataloaders[task_name]
                val_dl = val_dataloaders.get(task_name, None)
                
                metrics = self.train_single_adapter(
                    task_name, train_dl, val_dl
                )
                all_metrics[task_name] = metrics
                
                # Log to wandb if enabled
                if self.config.use_wandb:
                    self._log_wandb_metrics(task_name, metrics)
            else:
                logger.warning(f"No training data found for task: {task_name}")
        
        # Save final model
        self._save_phase1_checkpoint(all_metrics)
        
        logger.info("Phase 1 training completed")
        return all_metrics
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Any,
        loss_fn: nn.Module,
        task_name: str
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()
        
        # Gradient accumulation setup
        gradient_accumulation_steps = getattr(self.config.training, 'gradient_accumulation_steps', 1)
        
        progress_bar = tqdm(dataloader, desc=f"Training {task_name}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = move_to_device(batch, self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        task_labels=torch.full((batch['input_ids'].size(0),), 
                                             self.model.task_to_idx[task_name], 
                                             device=self.device)
                    )
                    
                    # Get task-specific output
                    if task_name in outputs['task_outputs']:
                        predictions = outputs['task_outputs'][task_name]
                        targets = batch['target']
                        
                        # Compute loss and normalize by accumulation steps
                        loss = loss_fn(predictions, targets) / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation: only step every N batches
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping with scaling
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step with scaling
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # Standard forward pass (for CPU)
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    task_labels=torch.full((batch['input_ids'].size(0),), 
                                         self.model.task_to_idx[task_name], 
                                         device=self.device)
                )
                
                # Get task-specific output
                if task_name in outputs['task_outputs']:
                    predictions = outputs['task_outputs'][task_name]
                    targets = batch['target']
                    
                    # Compute loss and normalize by accumulation steps
                    loss = loss_fn(predictions, targets) / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient accumulation: only step every N batches
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
            
            # Update metrics (use original loss for logging)
            actual_loss = loss.item() * gradient_accumulation_steps
            loss_meter.update(actual_loss, batch['input_ids'].size(0))
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.evaluation.logging_steps == 0:
                self.training_logger.log_step(
                    actual_loss,
                    scheduler.get_last_lr()[0]
                )
            
            # Update progress bar
            mem_info = "N/A"
            if torch.cuda.is_available():
                try:
                    mem_info = f"{torch.cuda.memory_reserved()/1024**3:.1f}GB"
                except:
                    mem_info = "N/A"
            
            progress_bar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'mem': mem_info
            })
        
        # Handle any remaining gradients
        if self.use_amp:
            if len(dataloader) % gradient_accumulation_steps != 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
        else:
            if len(dataloader) % gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        return loss_meter.avg
    
    def _validate_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        task_name: str
    ) -> float:
        """Validate for one epoch."""
        self.model.eval()
        loss_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validating {task_name}"):
                batch = move_to_device(batch, self.device)
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            task_labels=torch.full((batch['input_ids'].size(0),), 
                                                 self.model.task_to_idx[task_name], 
                                                 device=self.device)
                        )
                        
                        # Get task-specific output
                        if task_name in outputs['task_outputs']:
                            predictions = outputs['task_outputs'][task_name]
                            targets = batch['target']
                            
                            # Compute loss
                            loss = loss_fn(predictions, targets)
                            loss_meter.update(loss.item(), batch['input_ids'].size(0))
                else:
                    # Standard forward pass (for CPU)
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        task_labels=torch.full((batch['input_ids'].size(0),), 
                                             self.model.task_to_idx[task_name], 
                                             device=self.device)
                    )
                    
                    # Get task-specific output
                    if task_name in outputs['task_outputs']:
                        predictions = outputs['task_outputs'][task_name]
                        targets = batch['target']
                        
                        # Compute loss
                        loss = loss_fn(predictions, targets)
                        loss_meter.update(loss.item(), batch['input_ids'].size(0))
        
        return loss_meter.avg
    
    def _setup_optimizer(self, adapter: nn.Module, learning_rate: float) -> optim.Optimizer:
        """Setup optimizer for adapter training."""
        # Only optimize the current adapter parameters
        optimizer = optim.AdamW(
            adapter.parameters(),
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
    
    def _save_adapter_checkpoint(self, task_name: str, epoch: int, val_loss: float):
        """Save adapter checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir) / "phase1"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get adapter
        adapter = self.model.adapters.get_adapter(task_name)
        
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'adapter_state_dict': adapter.state_dict(),
            'task_name': task_name,
            'global_step': self.global_step
        }
        
        # Save as {task_name}_adapter.pt for consistency with loading
        checkpoint_path = checkpoint_dir / f"{task_name}_adapter.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved {task_name} adapter checkpoint to {checkpoint_path}")
    
    def _save_phase1_checkpoint(self, all_metrics: Dict[str, Dict[str, float]]):
        """Save complete Phase 1 checkpoint."""
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, "phase1")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save all adapters
        self.model.adapters.save_adapters(checkpoint_dir)
        
        # Save training metrics
        metrics_path = os.path.join(checkpoint_dir, "training_metrics.pt")
        torch.save(all_metrics, metrics_path)
        
        logger.info(f"Saved Phase 1 checkpoint to {checkpoint_dir}")
    
    def _log_wandb_metrics(self, task_name: str, metrics: Dict[str, float]):
        """Log metrics to Weights & Biases."""
        if not self.config.use_wandb:
            return
        
        wandb_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, list) and value:
                wandb_metrics[f"phase1/{task_name}/{key}"] = value[-1]
            elif isinstance(value, (int, float)):
                wandb_metrics[f"phase1/{task_name}/{key}"] = value
        
        wandb.log(wandb_metrics, step=self.global_step)

    def _check_existing_checkpoints(self) -> list:
        """Check which LoRA adapters have already been trained."""
        checkpoint_dir = Path(self.config.checkpoint_dir) / "phase1"
        
        if not checkpoint_dir.exists():
            return []
        
        completed_adapters = []
        for task_name in self.model.task_names:
            adapter_file = checkpoint_dir / f"{task_name}_adapter.pt"
            if adapter_file.exists():
                completed_adapters.append(task_name)
        
        return completed_adapters

    def _load_existing_checkpoints(self, completed_adapters: list):
        """Load existing adapter checkpoints."""
        checkpoint_dir = Path(self.config.checkpoint_dir) / "phase1"
        
        for task_name in completed_adapters:
            adapter_file = checkpoint_dir / f"{task_name}_adapter.pt"
            try:
                checkpoint = torch.load(adapter_file, map_location=self.device)
                
                # Load adapter state dict
                adapter = self.model.adapters.task_adapters[task_name]
                adapter.load_state_dict(checkpoint['adapter_state_dict'])
                
                logger.info(f"✅ Loaded checkpoint for {task_name} adapter")
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint for {task_name}: {e}")


def run_phase1_training(config: Config, model: DynamoModel, resume: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Run Phase 1 training.
    
    Args:
        config: Training configuration
        model: DYNAMO model
        resume: Whether to resume from checkpoints (default: True)
    
    Returns:
        Training metrics for all adapters
    """
    # Set random seed
    set_seed(config.seed)
    
    # Initialize trainer
    trainer = Phase1Trainer(config, model)
    
    # Run training
    metrics = trainer.train_all_adapters(resume=resume)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from utils.config import get_config
    from model import DynamoModel
    
    # Load configuration
    config = get_config()
    
    # Initialize model
    model = DynamoModel(config.__dict__)
    
    # Run Phase 1 training
    metrics = run_phase1_training(config, model)
    
    print("Phase 1 training completed!")
    for task_name, task_metrics in metrics.items():
        print(f"{task_name}: Best val loss = {task_metrics['best_val_loss']:.4f}")

