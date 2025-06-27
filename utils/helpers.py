"""
Helper functions for DYNAMO project.
Contains utility functions used across different modules.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
import os
from collections import defaultdict


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_parameters(model: torch.nn.Module):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_parameters(model: torch.nn.Module):
    """Unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a batch of data to the specified device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Apply Gumbel-Softmax to logits.
    
    Args:
        logits: Input logits
        temperature: Temperature parameter
        hard: Whether to use hard (discrete) or soft (continuous) sampling
    
    Returns:
        Gumbel-Softmax output
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = (logits + gumbel_noise) / temperature
    y_soft = F.softmax(y, dim=-1)
    
    if hard:
        # Straight-through estimator
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def compute_load_balance_loss(routing_probs: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute load balancing loss to prevent router collapse.
    
    Args:
        routing_probs: Router output probabilities [batch_size, num_experts]
        num_experts: Number of experts/adapters
    
    Returns:
        Load balancing loss
    """
    # Compute the fraction of tokens assigned to each expert
    expert_usage = routing_probs.mean(dim=0)  # [num_experts]
    
    # Ideal usage would be 1/num_experts for each expert
    ideal_usage = 1.0 / num_experts
    
    # Compute the coefficient of variation (std/mean) as load balance loss
    balance_loss = torch.var(expert_usage) / (ideal_usage ** 2)
    
    return balance_loss


def compute_consistency_loss(routing_probs_1: torch.Tensor, routing_probs_2: torch.Tensor) -> torch.Tensor:
    """
    Compute consistency loss between two routing decisions.
    Used to encourage similar inputs to use similar routing.
    
    Args:
        routing_probs_1: First routing probabilities
        routing_probs_2: Second routing probabilities
    
    Returns:
        Consistency loss
    """
    return F.kl_div(
        F.log_softmax(routing_probs_1, dim=-1),
        F.softmax(routing_probs_2, dim=-1),
        reduction='batchmean'
    )


def compute_efficiency_loss(routing_probs: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Compute efficiency loss to encourage sparse routing.
    
    Args:
        routing_probs: Router output probabilities
        threshold: Threshold below which probabilities are considered sparse
    
    Returns:
        Efficiency loss
    """
    # Encourage sparsity by penalizing non-zero probabilities below threshold
    small_probs = torch.where(routing_probs < threshold, routing_probs, torch.zeros_like(routing_probs))
    return small_probs.sum(dim=-1).mean()


def save_json(data: Dict, filepath: str):
    """Save dictionary as JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """Load JSON file as dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str):
    """Save data as pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """Load pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_mixed_task_examples(examples: List[Dict], task_combinations: List[Tuple[str, str]]) -> List[Dict]:
    """
    Create mixed-task examples by combining single-task examples.
    
    Args:
        examples: List of single-task examples
        task_combinations: List of task pairs to combine
    
    Returns:
        List of mixed-task examples
    """
    mixed_examples = []
    
    # Group examples by task
    task_examples = defaultdict(list)
    for example in examples:
        task_examples[example['task']].append(example)
    
    for task1, task2 in task_combinations:
        if task1 in task_examples and task2 in task_examples:
            # Sample examples from each task
            examples1 = random.sample(task_examples[task1], min(100, len(task_examples[task1])))
            examples2 = random.sample(task_examples[task2], min(100, len(task_examples[task2])))
            
            for ex1, ex2 in zip(examples1, examples2):
                # Create mixed instruction
                mixed_instruction = f"{ex1['instruction']} Also, {ex2['instruction'].lower()}"
                
                mixed_example = {
                    'instruction': mixed_instruction,
                    'input': ex1['input'],  # Use input from first task
                    'tasks': [task1, task2],
                    'expected_outputs': {
                        task1: ex1['output'],
                        task2: ex2['output']
                    }
                }
                mixed_examples.append(mixed_example)
    
    return mixed_examples


def calculate_routing_entropy(routing_probs: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy of routing probabilities.
    Higher entropy indicates more uncertain/balanced routing.
    
    Args:
        routing_probs: Router output probabilities
    
    Returns:
        Entropy values
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    routing_probs = routing_probs + eps
    entropy = -torch.sum(routing_probs * torch.log(routing_probs), dim=-1)
    return entropy


def get_top_k_routing(routing_probs: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-k routing probabilities and indices.
    
    Args:
        routing_probs: Router output probabilities
        k: Number of top experts to select
    
    Returns:
        Tuple of (top_k_probs, top_k_indices)
    """
    top_k_probs, top_k_indices = torch.topk(routing_probs, k, dim=-1)
    return top_k_probs, top_k_indices


def normalize_routing_probs(routing_probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Normalize routing probabilities with temperature scaling.
    
    Args:
        routing_probs: Raw routing logits
        temperature: Temperature for scaling
    
    Returns:
        Normalized probabilities
    """
    return F.softmax(routing_probs / temperature, dim=-1)


def interpolate_configs(config1: Dict, config2: Dict, alpha: float) -> Dict:
    """
    Interpolate between two configurations for curriculum learning.
    
    Args:
        config1: First configuration
        config2: Second configuration
        alpha: Interpolation factor (0 = config1, 1 = config2)
    
    Returns:
        Interpolated configuration
    """
    interpolated = {}
    for key in config1:
        if isinstance(config1[key], (int, float)):
            interpolated[key] = (1 - alpha) * config1[key] + alpha * config2[key]
        else:
            # For non-numeric values, use threshold-based selection
            interpolated[key] = config2[key] if alpha > 0.5 else config1[key]
    return interpolated


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        # Debug: ensure val is numeric
        if not isinstance(val, (int, float)):
            if hasattr(val, 'item'):
                val = val.item()
            else:
                raise TypeError(f"AverageMeter.update() expects numeric value, got {type(val)}: {val}")
        
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
        
        Returns:
            True if training should be stopped
        """
        # Ensure val_loss is a float
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
        elif not isinstance(val_loss, (int, float)):
            raise TypeError(f"val_loss must be a number, got {type(val_loss)}: {val_loss}")
        
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        return False
    
    def save_checkpoint(self, model: torch.nn.Module):
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def restore_checkpoint(self, model: torch.nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict({k: v.to(model.device) for k, v in self.best_weights.items()})

