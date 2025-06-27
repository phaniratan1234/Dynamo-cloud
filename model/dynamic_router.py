"""
Dynamic Router implementation for DYNAMO.
Routes input embeddings to appropriate LoRA adapters based on task detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from utils.logger import get_logger
from utils.helpers import gumbel_softmax, count_parameters

logger = get_logger(__name__)


class DynamicRouter(nn.Module):
    """
    Dynamic routing network that determines which LoRA adapters to use
    based on input embeddings from the RoBERTa backbone.
    """
    
    def __init__(
        self,
        input_size: int = 768,
        hidden_sizes: List[int] = [512, 256],
        num_tasks: int = 5,
        dropout: float = 0.1,
        temperature_init: float = 1.0,
        temperature_learnable: bool = True,
        use_batch_norm: bool = False,
        activation: str = "relu"
    ):
        """
        Initialize dynamic router.
        
        Args:
            input_size: Size of input embeddings (RoBERTa hidden size)
            hidden_sizes: List of hidden layer sizes
            num_tasks: Number of tasks/adapters
            dropout: Dropout rate
            temperature_init: Initial temperature for softmax
            temperature_learnable: Whether temperature is learnable
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ("relu", "gelu", "tanh")
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_tasks = num_tasks
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Temperature parameter for routing
        if temperature_learnable:
            self.temperature = nn.Parameter(torch.tensor(temperature_init))
        else:
            self.register_buffer('temperature', torch.tensor(temperature_init))
        self.temperature_learnable = temperature_learnable
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (routing logits)
        self.output_layer = nn.Linear(prev_size, num_tasks)
        
        # Additional components for advanced routing
        self.confidence_head = nn.Linear(prev_size, 1)  # Routing confidence
        self.entropy_regularizer = nn.Parameter(torch.tensor(0.1))  # Entropy regularization weight
        
        # Initialize weights
        self.reset_parameters()
        
        logger.info(f"Created dynamic router with {count_parameters(self):,} parameters")
    
    def reset_parameters(self):
        """Initialize router parameters."""
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            nn.init.zeros_(layer.bias)
        
        # Initialize output layer with smaller weights for stable training
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        
        # Initialize confidence head
        nn.init.normal_(self.confidence_head.weight, std=0.02)
        nn.init.zeros_(self.confidence_head.bias)
    
    def forward(
        self,
        input_embeddings: torch.Tensor,
        return_confidence: bool = False,
        return_entropy: bool = False,
        hard_routing: bool = False,
        gumbel_temperature: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the router.
        
        Args:
            input_embeddings: Input embeddings [batch_size, input_size]
            return_confidence: Whether to return routing confidence
            return_entropy: Whether to return routing entropy
            hard_routing: Whether to use hard (discrete) routing
            gumbel_temperature: Temperature for Gumbel-Softmax (if hard_routing)
        
        Returns:
            Dictionary containing:
                - routing_probs: Routing probabilities [batch_size, num_tasks]
                - routing_logits: Raw routing logits [batch_size, num_tasks]
                - confidence: Routing confidence [batch_size, 1] (if requested)
                - entropy: Routing entropy [batch_size] (if requested)
        """
        batch_size = input_embeddings.size(0)
        
        # Forward through MLP layers
        x = input_embeddings
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_batch_norm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropouts[i](x)
        
        # Get routing logits
        routing_logits = self.output_layer(x)  # [batch_size, num_tasks]
        
        # Apply temperature scaling
        scaled_logits = routing_logits / torch.clamp(self.temperature, min=0.1)
        
        # Get routing probabilities
        if hard_routing:
            # Use Gumbel-Softmax for hard routing
            if gumbel_temperature is None:
                gumbel_temperature = self.temperature.item()
            routing_probs = gumbel_softmax(scaled_logits, gumbel_temperature, hard=True)
        else:
            # Soft routing with temperature-scaled softmax
            routing_probs = F.softmax(scaled_logits, dim=-1)
        
        # Prepare output
        output = {
            'routing_probs': routing_probs,
            'routing_logits': routing_logits
        }
        
        # Routing confidence (how confident the router is in its decision)
        if return_confidence:
            confidence = torch.sigmoid(self.confidence_head(x))  # [batch_size, 1]
            output['confidence'] = confidence
        
        # Routing entropy (measure of uncertainty)
        if return_entropy:
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            entropy = -torch.sum(routing_probs * torch.log(routing_probs + eps), dim=-1)
            output['entropy'] = entropy
        
        return output
    
    def get_top_k_routing(
        self,
        input_embeddings: torch.Tensor,
        k: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k routing decisions.
        
        Args:
            input_embeddings: Input embeddings [batch_size, input_size]
            k: Number of top adapters to select
        
        Returns:
            Tuple of (top_k_probs, top_k_indices)
        """
        output = self.forward(input_embeddings)
        routing_probs = output['routing_probs']
        
        top_k_probs, top_k_indices = torch.topk(routing_probs, k, dim=-1)
        return top_k_probs, top_k_indices
    
    def route_with_threshold(
        self,
        input_embeddings: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route with confidence threshold.
        Only route if confidence is above threshold, otherwise use uniform routing.
        
        Args:
            input_embeddings: Input embeddings
            threshold: Confidence threshold
        
        Returns:
            Tuple of (routing_probs, confidence_mask)
        """
        output = self.forward(input_embeddings, return_confidence=True)
        routing_probs = output['routing_probs']
        confidence = output['confidence'].squeeze(-1)  # [batch_size]
        
        # Create confidence mask
        confidence_mask = confidence > threshold
        
        # Use uniform routing for low-confidence samples
        uniform_probs = torch.ones_like(routing_probs) / self.num_tasks
        final_routing_probs = torch.where(
            confidence_mask.unsqueeze(-1),
            routing_probs,
            uniform_probs
        )
        
        return final_routing_probs, confidence_mask
    
    def compute_routing_loss(
        self,
        routing_probs: torch.Tensor,
        target_tasks: Optional[torch.Tensor] = None,
        load_balance_weight: float = 0.1,
        entropy_weight: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        Compute routing-specific losses.
        
        Args:
            routing_probs: Routing probabilities [batch_size, num_tasks]
            target_tasks: Target task indices [batch_size] (for supervised routing)
            load_balance_weight: Weight for load balancing loss
            entropy_weight: Weight for entropy regularization
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Load balancing loss (prevent router collapse)
        expert_usage = routing_probs.mean(dim=0)  # [num_tasks]
        ideal_usage = 1.0 / self.num_tasks
        load_balance_loss = torch.var(expert_usage) / (ideal_usage ** 2)
        losses['load_balance'] = load_balance_weight * load_balance_loss
        
        # Entropy regularization (encourage diversity)
        eps = 1e-8
        entropy = -torch.sum(routing_probs * torch.log(routing_probs + eps), dim=-1)
        entropy_loss = -entropy.mean()  # Negative because we want to maximize entropy
        losses['entropy'] = entropy_weight * entropy_loss
        
        # Supervised routing loss (if target tasks are provided)
        if target_tasks is not None:
            routing_loss = F.cross_entropy(
                torch.log(routing_probs + eps),
                target_tasks
            )
            losses['routing'] = routing_loss
        
        # Total routing loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def get_routing_statistics(
        self,
        input_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get routing statistics for analysis.
        
        Args:
            input_embeddings: Input embeddings
        
        Returns:
            Dictionary of routing statistics
        """
        with torch.no_grad():
            output = self.forward(
                input_embeddings,
                return_confidence=True,
                return_entropy=True
            )
            
            routing_probs = output['routing_probs']
            confidence = output['confidence']
            entropy = output['entropy']
            
            # Compute statistics
            stats = {
                'mean_confidence': confidence.mean().item(),
                'std_confidence': confidence.std().item(),
                'mean_entropy': entropy.mean().item(),
                'std_entropy': entropy.std().item(),
                'max_routing_prob': routing_probs.max().item(),
                'min_routing_prob': routing_probs.min().item(),
                'routing_sparsity': (routing_probs < 0.1).float().mean().item(),
                'temperature': self.temperature.item()
            }
            
            # Per-task usage
            task_usage = routing_probs.mean(dim=0)
            for i, usage in enumerate(task_usage):
                stats[f'task_{i}_usage'] = usage.item()
        
        return stats
    
    def update_temperature(self, decay_factor: float = 0.999, min_temp: float = 0.1):
        """
        Update temperature with decay (for curriculum learning).
        
        Args:
            decay_factor: Temperature decay factor
            min_temp: Minimum temperature value
        """
        if self.temperature_learnable:
            with torch.no_grad():
                self.temperature.data = torch.clamp(
                    self.temperature.data * decay_factor,
                    min=min_temp
                )
    
    def set_temperature(self, temperature: float):
        """Set temperature value."""
        with torch.no_grad():
            self.temperature.data.fill_(temperature)
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()


class HierarchicalRouter(DynamicRouter):
    """
    Hierarchical router that first determines task family, then specific task.
    Useful for handling related tasks (e.g., different types of QA).
    """
    
    def __init__(
        self,
        input_size: int = 768,
        hidden_sizes: List[int] = [512, 256],
        task_families: Dict[str, List[str]] = None,
        **kwargs
    ):
        """
        Initialize hierarchical router.
        
        Args:
            input_size: Size of input embeddings
            hidden_sizes: Hidden layer sizes
            task_families: Dictionary mapping family names to task lists
            **kwargs: Additional arguments for base router
        """
        if task_families is None:
            task_families = {
                "classification": ["sentiment"],
                "qa": ["qa"],
                "generation": ["summarization", "code_generation", "translation"]
            }
        
        self.task_families = task_families
        self.family_names = list(task_families.keys())
        self.num_families = len(self.family_names)
        
        # Initialize base router for family selection
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_tasks=self.num_families,
            **kwargs
        )
        
        # Create task-specific routers for each family
        self.task_routers = nn.ModuleDict()
        for family_name, tasks in task_families.items():
            if len(tasks) > 1:
                self.task_routers[family_name] = DynamicRouter(
                    input_size=input_size,
                    hidden_sizes=[hidden_sizes[-1] // 2],  # Smaller router
                    num_tasks=len(tasks),
                    **{k: v for k, v in kwargs.items() if k not in ['num_tasks']}
                )
    
    def forward(
        self,
        input_embeddings: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical forward pass.
        
        Args:
            input_embeddings: Input embeddings
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with hierarchical routing results
        """
        # First level: family selection
        family_output = super().forward(input_embeddings, **kwargs)
        family_probs = family_output['routing_probs']  # [batch_size, num_families]
        
        # Second level: task selection within families
        batch_size = input_embeddings.size(0)
        all_task_probs = []
        task_names = []
        
        for family_idx, (family_name, tasks) in enumerate(self.task_families.items()):
            family_weight = family_probs[:, family_idx]  # [batch_size]
            
            if len(tasks) == 1:
                # Single task in family
                task_probs = family_weight.unsqueeze(-1)  # [batch_size, 1]
            else:
                # Multiple tasks: use task-specific router
                task_router = self.task_routers[family_name]
                task_output = task_router.forward(input_embeddings, **kwargs)
                task_probs = task_output['routing_probs']  # [batch_size, num_tasks_in_family]
                
                # Weight by family probability
                task_probs = task_probs * family_weight.unsqueeze(-1)
            
            all_task_probs.append(task_probs)
            task_names.extend(tasks)
        
        # Concatenate all task probabilities
        final_task_probs = torch.cat(all_task_probs, dim=-1)  # [batch_size, total_tasks]
        
        return {
            'routing_probs': final_task_probs,
            'family_probs': family_probs,
            'task_names': task_names
        }

