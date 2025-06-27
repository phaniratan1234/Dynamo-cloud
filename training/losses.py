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
            temperature: Current temperature value
        
        Returns:
            Temperature regularization loss
        """
        temp_loss = F.mse_loss(temperature, torch.tensor(self.target_temperature, device=temperature.device))
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
            self.loss_fn = nn.MSELoss()  # Simplified for hidden state matching
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
        else:
            return self.weight * self.loss_fn(predictions, targets)


class DynamoLoss(nn.Module):
    """
    Combined loss function for DYNAMO training.
    Integrates task-specific losses with routing losses.
    """
    
    def __init__(
        self,
        task_names: List[str],
        num_experts: int,
        load_balance_weight: float = 0.1,
        efficiency_weight: float = 0.05,
        consistency_weight: float = 0.1,
        entropy_weight: float = 0.01,
        temperature_weight: float = 0.01
    ):
        """
        Initialize DYNAMO loss.
        
        Args:
            task_names: List of task names
            num_experts: Number of experts/adapters
            load_balance_weight: Weight for load balance loss
            efficiency_weight: Weight for efficiency loss
            consistency_weight: Weight for consistency loss
            entropy_weight: Weight for entropy regularization
            temperature_weight: Weight for temperature regularization
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
    
    def forward(
        self,
        task_outputs: Dict[str, torch.Tensor],
        task_targets: Dict[str, torch.Tensor],
        routing_probs: torch.Tensor,
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
            # Load balance loss
            load_balance = self.load_balance_loss(routing_probs)
            losses["load_balance_loss"] = load_balance
            total_loss += load_balance
            
            # Efficiency loss
            efficiency = self.efficiency_loss(routing_probs)
            losses["efficiency_loss"] = efficiency
            total_loss += efficiency
            
            # Entropy regularization
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
        config: Training configuration
        task_names: List of task names
        num_experts: Number of experts/adapters
    
    Returns:
        Configured DYNAMO loss function
    """
    return DynamoLoss(
        task_names=task_names,
        num_experts=num_experts,
        load_balance_weight=config.get("load_balance_weight", 0.1),
        efficiency_weight=config.get("efficiency_weight", 0.05),
        consistency_weight=config.get("consistency_weight", 0.1),
        entropy_weight=config.get("entropy_weight", 0.01),
        temperature_weight=config.get("temperature_weight", 0.01)
    )

