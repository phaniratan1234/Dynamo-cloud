"""
Custom loss functions for DYNAMO training.
Implements load balancing, efficiency, consistency, and other specialized losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from utils.logger import get_logger

logger = get_logger(__name__)


class LoadBalanceLoss(nn.Module):
    """
    Load balancing loss to prevent router collapse.
    Encourages uniform distribution of samples across adapters.
    """
    
    def __init__(self, num_experts: int, weight: float = 1.0):
        """
        Initialize load balance loss.
        
        Args:
            num_experts: Number of experts/adapters
            weight: Loss weight
        """
        super().__init__()
        self.num_experts = num_experts
        self.weight = weight
        self.ideal_usage = 1.0 / num_experts
    
    def forward(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balance loss.
        
        Args:
            routing_probs: Routing probabilities [batch_size, num_experts]
        
        Returns:
            Load balance loss
        """
        # Compute expert usage across the batch
        expert_usage = routing_probs.mean(dim=0)  # [num_experts]
        
        # Compute coefficient of variation (std/mean)
        usage_variance = torch.var(expert_usage)
        balance_loss = usage_variance / (self.ideal_usage ** 2)
        
        return self.weight * balance_loss


class EfficiencyLoss(nn.Module):
    """
    Efficiency loss to encourage sparse routing.
    Penalizes non-zero probabilities below a threshold.
    """
    
    def __init__(self, threshold: float = 0.1, weight: float = 1.0):
        """
        Initialize efficiency loss.
        
        Args:
            threshold: Threshold below which probabilities are considered inefficient
            weight: Loss weight
        """
        super().__init__()
        self.threshold = threshold
        self.weight = weight
    
    def forward(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute efficiency loss.
        
        Args:
            routing_probs: Routing probabilities [batch_size, num_experts]
        
        Returns:
            Efficiency loss
        """
        # Penalize small but non-zero probabilities
        small_probs = torch.where(
            routing_probs < self.threshold,
            routing_probs,
            torch.zeros_like(routing_probs)
        )
        
        efficiency_loss = small_probs.sum(dim=-1).mean()
        return self.weight * efficiency_loss


class ConsistencyLoss(nn.Module):
    """
    Consistency loss to encourage similar inputs to use similar routing.
    Uses KL divergence between routing decisions of similar samples.
    """
    
    def __init__(self, weight: float = 1.0, similarity_threshold: float = 0.8):
        """
        Initialize consistency loss.
        
        Args:
            weight: Loss weight
            similarity_threshold: Threshold for considering samples similar
        """
        super().__init__()
        self.weight = weight
        self.similarity_threshold = similarity_threshold
    
    def forward(
        self,
        routing_probs: torch.Tensor,
        input_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss.
        
        Args:
            routing_probs: Routing probabilities [batch_size, num_experts]
            input_embeddings: Input embeddings [batch_size, hidden_size]
        
        Returns:
            Consistency loss
        """
        batch_size = routing_probs.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=routing_probs.device)
        
        # Compute pairwise similarities
        normalized_embeddings = F.normalize(input_embeddings, p=2, dim=-1)
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Find similar pairs
        similar_pairs = similarity_matrix > self.similarity_threshold
        similar_pairs = similar_pairs & ~torch.eye(batch_size, dtype=torch.bool, device=routing_probs.device)
        
        if not similar_pairs.any():
            return torch.tensor(0.0, device=routing_probs.device)
        
        # Compute KL divergence for similar pairs
        consistency_loss = 0.0
        num_pairs = 0
        
        for i in range(batch_size):
            similar_indices = similar_pairs[i].nonzero(as_tuple=True)[0]
            
            if len(similar_indices) > 0:
                for j in similar_indices:
                    kl_div = F.kl_div(
                        F.log_softmax(routing_probs[i:i+1], dim=-1),
                        F.softmax(routing_probs[j:j+1], dim=-1),
                        reduction='sum'
                    )
                    consistency_loss += kl_div
                    num_pairs += 1
        
        if num_pairs > 0:
            consistency_loss = consistency_loss / num_pairs
        
        return self.weight * consistency_loss


class EntropyRegularizationLoss(nn.Module):
    """
    Entropy regularization loss to encourage diversity in routing decisions.
    """
    
    def __init__(self, weight: float = 1.0, target_entropy: Optional[float] = None):
        """
        Initialize entropy regularization loss.
        
        Args:
            weight: Loss weight
            target_entropy: Target entropy value (None for maximum entropy)
        """
        super().__init__()
        self.weight = weight
        self.target_entropy = target_entropy
    
    def forward(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization loss.
        
        Args:
            routing_probs: Routing probabilities [batch_size, num_experts]
        
        Returns:
            Entropy regularization loss
        """
        # Compute entropy
        eps = 1e-8
        entropy = -torch.sum(routing_probs * torch.log(routing_probs + eps), dim=-1)
        
        if self.target_entropy is not None:
            # Penalize deviation from target entropy
            entropy_loss = F.mse_loss(entropy, torch.full_like(entropy, self.target_entropy))
        else:
            # Maximize entropy (minimize negative entropy)
            entropy_loss = -entropy.mean()
        
        return self.weight * entropy_loss


class TemperatureRegularizationLoss(nn.Module):
    """
    Temperature regularization loss to prevent temperature from becoming too extreme.
    """
    
    def __init__(self, weight: float = 1.0, target_temperature: float = 1.0):
        """
        Initialize temperature regularization loss.
        
        Args:
            weight: Loss weight
            target_temperature: Target temperature value
        """
        super().__init__()
        self.weight = weight
        self.target_temperature = target_temperature
    
    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature regularization loss.
        
        Args:
            temperature: Current temperature value (tensor or float)
        
        Returns:
            Temperature regularization loss
        """
        # Handle both tensor and float inputs
        if isinstance(temperature, (int, float)):
            # Convert float to tensor
            temperature_tensor = torch.tensor(temperature, dtype=torch.float32)
            target_tensor = torch.tensor(self.target_temperature, dtype=torch.float32)
        else:
            # Temperature is already a tensor
            temperature_tensor = temperature
            target_tensor = torch.tensor(self.target_temperature, dtype=temperature.dtype, device=temperature.device)
        
        temp_loss = F.mse_loss(temperature_tensor, target_tensor)
        return self.weight * temp_loss


class TaskSpecificLoss(nn.Module):
    """
    Task-specific loss functions for different tasks.
    """
    
    def __init__(self, task_name: str, weight: float = 1.0):
        """
        Initialize task-specific loss.
        
        Args:
            task_name: Name of the task
            weight: Loss weight
        """
        super().__init__()
        self.task_name = task_name
        self.weight = weight
        
        # Define task-specific loss functions
        if task_name == "sentiment":
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_name == "qa":
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_name in ["summarization", "code_generation", "translation"]:
            # For generation tasks, use CrossEntropy loss for vocabulary prediction
            # Ignore padding tokens (token_id = 1 for RoBERTa)
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=1)  # RoBERTa pad_token_id = 1
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute task-specific loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Task-specific loss
        """
        if self.task_name == "qa":
            # QA has start and end positions
            # predictions: [batch_size, seq_len, 2] or [batch_size, 2]
            # targets: [batch_size, 2] where targets[:, 0] = start_pos, targets[:, 1] = end_pos
            
            if predictions.dim() == 3:
                # predictions: [batch_size, seq_len, 2]
                start_logits = predictions[:, :, 0]  # [batch_size, seq_len]
                end_logits = predictions[:, :, 1]    # [batch_size, seq_len]
            else:
                # predictions: [batch_size, 2] - this shouldn't happen for QA but handle it
                start_logits = predictions[:, 0].unsqueeze(1)  # [batch_size, 1]
                end_logits = predictions[:, 1].unsqueeze(1)    # [batch_size, 1]
            
            start_targets = targets[:, 0]  # [batch_size]
            end_targets = targets[:, 1]    # [batch_size]
            
            # Ensure targets are within valid range
            seq_len = start_logits.size(1)
            start_targets = torch.clamp(start_targets, 0, seq_len - 1)
            end_targets = torch.clamp(end_targets, 0, seq_len - 1)
            
            start_loss = self.loss_fn(start_logits, start_targets)
            end_loss = self.loss_fn(end_logits, end_targets)
            
            return self.weight * (start_loss + end_loss) / 2
        
        elif self.task_name in ["summarization", "code_generation", "translation"]:
            # For generation tasks: 
            # predictions: [batch_size, seq_len, vocab_size] - vocabulary logits for each position
            # targets: [batch_size, seq_len] - target token IDs
            
            batch_size, seq_len, vocab_size = predictions.shape
            target_seq_len = targets.shape[1]
            
            # Handle sequence length mismatch
            if seq_len != target_seq_len:
                # Take minimum length to avoid index errors
                min_len = min(seq_len, target_seq_len)
                predictions = predictions[:, :min_len, :]  # [batch_size, min_len, vocab_size]
                targets = targets[:, :min_len]  # [batch_size, min_len]
                seq_len = min_len
            
            # Reshape for CrossEntropy loss
            # predictions: [batch_size * seq_len, vocab_size]
            # targets: [batch_size * seq_len]
            predictions_flat = predictions.contiguous().view(-1, vocab_size)
            targets_flat = targets.contiguous().view(-1)
            
            # Compute sequence-to-sequence loss
            loss = self.loss_fn(predictions_flat, targets_flat)
            
            return self.weight * loss
        
        else:
            return self.weight * self.loss_fn(predictions, targets)


class RouterSupervisionLoss(nn.Module):
    """
    Supervised loss for training the router to correctly identify task types.
    This is the CRITICAL missing component for router learning.
    """
    
    def __init__(self, weight: float = 1.0, temperature: float = 1.0):
        """
        Initialize router supervision loss.
        
        Args:
            weight: Loss weight
            temperature: Temperature for softmax (higher = softer targets)
        """
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        routing_logits: torch.Tensor, 
        true_task_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised routing loss.
        
        Args:
            routing_logits: Raw routing logits [batch_size, num_tasks]
            true_task_labels: True task labels [batch_size] (single task per example)
        
        Returns:
            Supervised routing loss
        """
        # Apply temperature scaling
        scaled_logits = routing_logits / self.temperature
        
        # Compute cross-entropy loss
        loss = self.cross_entropy(scaled_logits, true_task_labels)
        
        return self.weight * loss


class RouterConfidenceLoss(nn.Module):
    """
    Loss to encourage confident routing decisions when the router is correct.
    """
    
    def __init__(self, weight: float = 0.1, margin: float = 0.2):
        """
        Initialize router confidence loss.
        
        Args:
            weight: Loss weight
            margin: Minimum margin between correct and incorrect predictions
        """
        super().__init__()
        self.weight = weight
        self.margin = margin
    
    def forward(
        self, 
        routing_probs: torch.Tensor, 
        true_task_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute confidence loss.
        
        Args:
            routing_probs: Routing probabilities [batch_size, num_tasks]
            true_task_labels: True task labels [batch_size]
        
        Returns:
            Confidence loss
        """
        batch_size, num_tasks = routing_probs.shape
        
        # Get probabilities for correct tasks
        correct_probs = routing_probs.gather(1, true_task_labels.unsqueeze(1)).squeeze(1)
        
        # Get maximum probability for incorrect tasks
        mask = torch.ones_like(routing_probs)
        mask.scatter_(1, true_task_labels.unsqueeze(1), 0)
        incorrect_probs = (routing_probs * mask).max(dim=1)[0]
        
        # Margin loss: encourage correct_prob > incorrect_prob + margin
        loss = torch.clamp(incorrect_probs - correct_probs + self.margin, min=0.0)
        
        return self.weight * loss.mean()


class DynamoLoss(nn.Module):
    """
    Combined loss function for DYNAMO training.
    Includes task-specific losses and routing losses.
    """
    
    def __init__(
        self,
        task_names: List[str],
        num_experts: int,
        load_balance_weight: float = 0.1,
        efficiency_weight: float = 0.05,
        consistency_weight: float = 0.1,
        entropy_weight: float = 0.01,
        temperature_weight: float = 0.01,
        router_supervision_weight: float = 2.0,  # NEW: High weight for supervision
        router_confidence_weight: float = 0.5    # NEW: Confidence loss weight
    ):
        """
        Initialize DYNAMO loss function.
        
        Args:
            task_names: List of task names
            num_experts: Number of experts/adapters
            load_balance_weight: Weight for load balancing loss
            efficiency_weight: Weight for efficiency loss
            consistency_weight: Weight for consistency loss
            entropy_weight: Weight for entropy regularization
            temperature_weight: Weight for temperature regularization
            router_supervision_weight: Weight for router supervision loss (CRITICAL)
            router_confidence_weight: Weight for router confidence loss
        """
        super().__init__()
        
        self.task_names = task_names
        self.num_experts = num_experts
        
        # Task-specific losses
        self.task_losses = nn.ModuleDict()
        for task_name in task_names:
            self.task_losses[task_name] = TaskSpecificLoss(task_name)
        
        # Routing losses
        self.load_balance_loss = LoadBalanceLoss(num_experts, load_balance_weight)
        self.efficiency_loss = EfficiencyLoss(weight=efficiency_weight)
        self.consistency_loss = ConsistencyLoss(weight=consistency_weight)
        self.entropy_loss = EntropyRegularizationLoss(weight=entropy_weight)
        self.temperature_loss = TemperatureRegularizationLoss(weight=temperature_weight)
        
        # NEW: Critical supervised learning components
        self.router_supervision_loss = RouterSupervisionLoss(weight=router_supervision_weight)
        self.router_confidence_loss = RouterConfidenceLoss(weight=router_confidence_weight)
    
    def forward(
        self,
        task_outputs: Dict[str, torch.Tensor],
        task_targets: Dict[str, torch.Tensor],
        routing_probs: torch.Tensor,
        routing_logits: torch.Tensor,  # NEW: Need raw logits for supervision
        true_task_labels: torch.Tensor,  # NEW: Ground truth task labels
        input_embeddings: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        training_phase: str = "phase3"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined DYNAMO loss.
        
        Args:
            task_outputs: Dictionary of task predictions
            task_targets: Dictionary of task targets
            routing_probs: Routing probabilities
            routing_logits: Raw routing logits (for supervision)
            true_task_labels: Ground truth task labels [batch_size]
            input_embeddings: Input embeddings (for consistency loss)
            temperature: Current temperature (for temperature regularization)
            training_phase: Current training phase
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Task-specific losses
        task_loss_sum = 0.0
        num_tasks_with_targets = 0
        
        for task_name in self.task_names:
            if task_name in task_outputs and task_name in task_targets:
                task_loss = self.task_losses[task_name](
                    task_outputs[task_name],
                    task_targets[task_name]
                )
                losses[f"{task_name}_loss"] = task_loss
                task_loss_sum += task_loss
                num_tasks_with_targets += 1
        
        if num_tasks_with_targets > 0:
            avg_task_loss = task_loss_sum / num_tasks_with_targets
            losses["task_loss"] = avg_task_loss
            total_loss += avg_task_loss
        
        # Routing losses (only in phase 2 and 3)
        if training_phase in ["phase2", "phase3"]:
            # CRITICAL: Router supervision loss - teaches router to identify tasks
            router_supervision = self.router_supervision_loss(routing_logits, true_task_labels)
            losses["router_supervision_loss"] = router_supervision
            total_loss += router_supervision
            
            # Router confidence loss - encourages confident correct decisions
            router_confidence = self.router_confidence_loss(routing_probs, true_task_labels)
            losses["router_confidence_loss"] = router_confidence
            total_loss += router_confidence
            
            # Load balance loss
            load_balance = self.load_balance_loss(routing_probs)
            losses["load_balance_loss"] = load_balance
            total_loss += load_balance
            
            # Efficiency loss
            efficiency = self.efficiency_loss(routing_probs)
            losses["efficiency_loss"] = efficiency
            total_loss += efficiency
            
            # Entropy regularization (reduced weight when we have supervision)
            entropy_reg = self.entropy_loss(routing_probs)
            losses["entropy_loss"] = entropy_reg
            total_loss += entropy_reg
            
            # Consistency loss (if input embeddings provided)
            if input_embeddings is not None:
                consistency = self.consistency_loss(routing_probs, input_embeddings)
                losses["consistency_loss"] = consistency
                total_loss += consistency
            
            # Temperature regularization (if temperature provided)
            if temperature is not None:
                temp_reg = self.temperature_loss(temperature)
                losses["temperature_loss"] = temp_reg
                total_loss += temp_reg
        
        losses["total_loss"] = total_loss
        return losses


class CurriculumLoss(nn.Module):
    """
    Curriculum learning loss that gradually increases task complexity.
    """
    
    def __init__(self, base_loss: nn.Module, curriculum_schedule: Dict[str, float]):
        """
        Initialize curriculum loss.
        
        Args:
            base_loss: Base loss function
            curriculum_schedule: Schedule for curriculum weights
        """
        super().__init__()
        self.base_loss = base_loss
        self.curriculum_schedule = curriculum_schedule
        self.current_step = 0
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute curriculum-weighted loss.
        
        Args:
            *args: Arguments for base loss
            **kwargs: Keyword arguments for base loss
        
        Returns:
            Dictionary of losses with curriculum weighting
        """
        # Get base losses
        losses = self.base_loss(*args, **kwargs)
        
        # Apply curriculum weighting
        curriculum_weight = self._get_curriculum_weight()
        
        # Weight the losses based on curriculum
        for key, loss in losses.items():
            if key != "total_loss":
                losses[key] = loss * curriculum_weight
        
        # Recompute total loss
        total_loss = sum(loss for key, loss in losses.items() if key != "total_loss")
        losses["total_loss"] = total_loss
        
        return losses
    
    def _get_curriculum_weight(self) -> float:
        """Get curriculum weight for current step."""
        # Simple linear schedule
        max_steps = max(self.curriculum_schedule.keys())
        progress = min(self.current_step / max_steps, 1.0)
        
        # Interpolate between schedule points
        for step, weight in sorted(self.curriculum_schedule.items()):
            if self.current_step <= step:
                return weight
        
        return list(self.curriculum_schedule.values())[-1]
    
    def step(self):
        """Advance curriculum step."""
        self.current_step += 1


def create_loss_function(
    config: Dict[str, Any],
    task_names: List[str],
    num_experts: int
) -> DynamoLoss:
    """
    Create DYNAMO loss function from configuration.
    
    Args:
        config: Training configuration (dictionary or config object)
        task_names: List of task names
        num_experts: Number of experts/adapters
    
    Returns:
        Configured DYNAMO loss function
    """
    # Handle both dictionary and config object formats
    if isinstance(config, dict):
        # Dictionary format
        load_balance_weight = config.get("load_balance_weight", 0.1)
        efficiency_weight = config.get("efficiency_weight", 0.05)
        consistency_weight = config.get("consistency_weight", 0.1)
        entropy_weight = config.get("entropy_weight", 0.01)
        temperature_weight = config.get("temperature_weight", 0.01)
        router_supervision_weight = config.get("router_supervision_weight", 2.0)  # NEW
        router_confidence_weight = config.get("router_confidence_weight", 0.5)    # NEW
    else:
        # Config object format
        load_balance_weight = getattr(config, "load_balance_weight", 0.1)
        efficiency_weight = getattr(config, "efficiency_weight", 0.05)
        consistency_weight = getattr(config, "consistency_weight", 0.1)
        entropy_weight = getattr(config, "entropy_weight", 0.01)
        temperature_weight = getattr(config, "temperature_weight", 0.01)
        router_supervision_weight = getattr(config, "router_supervision_weight", 2.0)  # NEW
        router_confidence_weight = getattr(config, "router_confidence_weight", 0.5)    # NEW
    
    return DynamoLoss(
        task_names=task_names,
        num_experts=num_experts,
        load_balance_weight=load_balance_weight,
        efficiency_weight=efficiency_weight,
        consistency_weight=consistency_weight,
        entropy_weight=entropy_weight,
        temperature_weight=temperature_weight,
        router_supervision_weight=router_supervision_weight,  # NEW
        router_confidence_weight=router_confidence_weight     # NEW
    )

