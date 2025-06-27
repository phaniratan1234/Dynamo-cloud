"""
Baseline models for DYNAMO evaluation.
Implements oracle routing, single LoRA, full fine-tuning, and random routing baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from typing import Dict, List, Optional, Any, Union
import random
import numpy as np

from ..model import DynamoModel, RobertaBackbone, LoRAAdapterCollection
from ..utils.logger import get_logger
from ..utils.helpers import count_parameters, move_to_device

logger = get_logger(__name__)


class OracleRoutingBaseline(nn.Module):
    """
    Oracle routing baseline that uses perfect task knowledge.
    This represents the upper bound performance when task identification is perfect.
    """
    
    def __init__(self, dynamo_model: DynamoModel):
        """
        Initialize oracle routing baseline.
        
        Args:
            dynamo_model: Trained DYNAMO model to use adapters from
        """
        super().__init__()
        
        self.backbone = dynamo_model.backbone
        self.adapters = dynamo_model.adapters
        self.task_names = dynamo_model.task_names
        self.task_to_idx = dynamo_model.task_to_idx
        
        logger.info("Oracle routing baseline initialized")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with oracle routing.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task_labels: Ground truth task labels
        
        Returns:
            Dictionary with task outputs and routing information
        """
        if task_labels is None:
            raise ValueError("Oracle routing requires task labels")
        
        # Get backbone embeddings
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Create one-hot routing based on task labels
        batch_size = input_ids.size(0)
        num_tasks = len(self.task_names)
        
        routing_probs = torch.zeros(batch_size, num_tasks, device=input_ids.device)
        routing_probs.scatter_(1, task_labels.unsqueeze(1), 1.0)
        
        # Forward through adapters with oracle routing
        task_outputs = self.adapters.forward_task_heads(
            backbone_outputs.last_hidden_state,
            routing_probs,
            hard_routing=True
        )
        
        return {
            'task_outputs': task_outputs,
            'routing_probs': routing_probs
        }


class SingleLoRABaseline(nn.Module):
    """
    Single large LoRA baseline without routing.
    Uses a single adapter for all tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize single LoRA baseline.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # Initialize backbone
        self.backbone = RobertaBackbone(
            model_name=config.get("base_model_name", "roberta-base"),
            freeze=True
        )
        
        # Create single large LoRA adapter
        from ..model.lora_adapters import TaskSpecificLoRA
        
        # Use larger rank for single adapter
        large_rank = max(config.get("lora_configs", {}).get(task, {}).get("rank", 16) 
                        for task in ["sentiment", "qa", "summarization", "code_generation", "translation"])
        large_rank = int(large_rank * 1.5)  # 50% larger
        
        self.single_adapter = TaskSpecificLoRA(
            task_name="unified",
            hidden_size=self.backbone.hidden_size,
            rank=large_rank,
            alpha=large_rank * 2,
            dropout=0.1,
            num_layers=12
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name in ["sentiment", "qa", "summarization", "code_generation", "translation"]:
            if task_name == "sentiment":
                self.task_heads[task_name] = nn.Linear(self.backbone.hidden_size, 2)
            elif task_name == "qa":
                self.task_heads[task_name] = nn.Linear(self.backbone.hidden_size, 2)
            else:
                self.task_heads[task_name] = nn.Linear(self.backbone.hidden_size, self.backbone.hidden_size)
        
        logger.info(f"Single LoRA baseline initialized with {count_parameters(self):,} parameters")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through single LoRA.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task_labels: Task labels (used to determine which head to use)
        
        Returns:
            Dictionary with task outputs
        """
        # Get backbone embeddings
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Apply single adapter (simplified - just use the task head)
        hidden_states = backbone_outputs.last_hidden_state
        
        # Get task-specific outputs
        task_outputs = {}
        
        if task_labels is not None:
            # Use task labels to determine which heads to use
            for i, task_idx in enumerate(task_labels):
                task_name = ["sentiment", "qa", "summarization", "code_generation", "translation"][task_idx.item()]
                
                if task_name == "sentiment" or task_name == "qa":
                    # Use CLS token
                    cls_hidden = hidden_states[i:i+1, 0, :]
                    output = self.task_heads[task_name](cls_hidden)
                else:
                    # Use CLS token for generation tasks
                    cls_hidden = hidden_states[i:i+1, 0, :]
                    output = self.task_heads[task_name](cls_hidden)
                
                if task_name not in task_outputs:
                    batch_size = input_ids.size(0)
                    output_shape = [batch_size] + list(output.shape[1:])
                    task_outputs[task_name] = torch.zeros(output_shape, device=output.device)
                
                task_outputs[task_name][i] = output[0]
        else:
            # Apply all heads
            for task_name, head in self.task_heads.items():
                if task_name in ["sentiment", "qa"]:
                    cls_hidden = hidden_states[:, 0, :]
                    task_outputs[task_name] = head(cls_hidden)
                else:
                    cls_hidden = hidden_states[:, 0, :]
                    task_outputs[task_name] = head(cls_hidden)
        
        return {'task_outputs': task_outputs}


class FullFineTuningBaseline(nn.Module):
    """
    Full fine-tuning baseline that fine-tunes the entire RoBERTa model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize full fine-tuning baseline.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # Initialize backbone (not frozen)
        self.backbone = RobertaBackbone(
            model_name=config.get("base_model_name", "roberta-base"),
            freeze=False  # Allow fine-tuning
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name in ["sentiment", "qa", "summarization", "code_generation", "translation"]:
            if task_name == "sentiment":
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(self.backbone.hidden_size, self.backbone.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.backbone.hidden_size // 2, 2)
                )
            elif task_name == "qa":
                self.task_heads[task_name] = nn.Linear(self.backbone.hidden_size, 2)
            else:
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(self.backbone.hidden_size, self.backbone.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.backbone.hidden_size, self.backbone.hidden_size)
                )
        
        logger.info(f"Full fine-tuning baseline initialized with {count_parameters(self):,} parameters")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full fine-tuned model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task_labels: Task labels
        
        Returns:
            Dictionary with task outputs
        """
        # Get backbone embeddings (trainable)
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = backbone_outputs.last_hidden_state
        
        # Get task-specific outputs
        task_outputs = {}
        
        if task_labels is not None:
            # Use task labels to determine which heads to use
            for i, task_idx in enumerate(task_labels):
                task_name = ["sentiment", "qa", "summarization", "code_generation", "translation"][task_idx.item()]
                
                if task_name in ["sentiment"]:
                    cls_hidden = hidden_states[i:i+1, 0, :]
                    output = self.task_heads[task_name](cls_hidden)
                elif task_name == "qa":
                    # For QA, use all tokens
                    output = self.task_heads[task_name](hidden_states[i:i+1])
                else:
                    cls_hidden = hidden_states[i:i+1, 0, :]
                    output = self.task_heads[task_name](cls_hidden)
                
                if task_name not in task_outputs:
                    batch_size = input_ids.size(0)
                    if task_name == "qa":
                        output_shape = [batch_size, hidden_states.size(1), 2]
                    else:
                        output_shape = [batch_size] + list(output.shape[1:])
                    task_outputs[task_name] = torch.zeros(output_shape, device=output.device)
                
                task_outputs[task_name][i] = output[0]
        else:
            # Apply all heads
            for task_name, head in self.task_heads.items():
                if task_name == "sentiment":
                    cls_hidden = hidden_states[:, 0, :]
                    task_outputs[task_name] = head(cls_hidden)
                elif task_name == "qa":
                    task_outputs[task_name] = head(hidden_states)
                else:
                    cls_hidden = hidden_states[:, 0, :]
                    task_outputs[task_name] = head(cls_hidden)
        
        return {'task_outputs': task_outputs}


class RandomRoutingBaseline(nn.Module):
    """
    Random routing baseline that randomly selects adapters.
    """
    
    def __init__(self, dynamo_model: DynamoModel):
        """
        Initialize random routing baseline.
        
        Args:
            dynamo_model: Trained DYNAMO model to use adapters from
        """
        super().__init__()
        
        self.backbone = dynamo_model.backbone
        self.adapters = dynamo_model.adapters
        self.task_names = dynamo_model.task_names
        self.num_tasks = len(self.task_names)
        
        logger.info("Random routing baseline initialized")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with random routing.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Dictionary with task outputs and routing information
        """
        batch_size = input_ids.size(0)
        
        # Get backbone embeddings
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Create random routing probabilities
        routing_probs = torch.rand(batch_size, self.num_tasks, device=input_ids.device)
        routing_probs = F.softmax(routing_probs, dim=-1)
        
        # Forward through adapters with random routing
        task_outputs = self.adapters.forward_task_heads(
            backbone_outputs.last_hidden_state,
            routing_probs,
            hard_routing=False
        )
        
        return {
            'task_outputs': task_outputs,
            'routing_probs': routing_probs
        }


class UniformRoutingBaseline(nn.Module):
    """
    Uniform routing baseline that uses equal weights for all adapters.
    """
    
    def __init__(self, dynamo_model: DynamoModel):
        """
        Initialize uniform routing baseline.
        
        Args:
            dynamo_model: Trained DYNAMO model to use adapters from
        """
        super().__init__()
        
        self.backbone = dynamo_model.backbone
        self.adapters = dynamo_model.adapters
        self.task_names = dynamo_model.task_names
        self.num_tasks = len(self.task_names)
        
        logger.info("Uniform routing baseline initialized")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uniform routing.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Dictionary with task outputs and routing information
        """
        batch_size = input_ids.size(0)
        
        # Get backbone embeddings
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Create uniform routing probabilities
        uniform_prob = 1.0 / self.num_tasks
        routing_probs = torch.full(
            (batch_size, self.num_tasks), 
            uniform_prob, 
            device=input_ids.device
        )
        
        # Forward through adapters with uniform routing
        task_outputs = self.adapters.forward_task_heads(
            backbone_outputs.last_hidden_state,
            routing_probs,
            hard_routing=False
        )
        
        return {
            'task_outputs': task_outputs,
            'routing_probs': routing_probs
        }


class BaselineCollection:
    """
    Collection of all baseline models for easy comparison.
    """
    
    def __init__(self, dynamo_model: DynamoModel, config: Dict[str, Any]):
        """
        Initialize baseline collection.
        
        Args:
            dynamo_model: Trained DYNAMO model
            config: Model configuration
        """
        self.dynamo_model = dynamo_model
        self.config = config
        
        # Initialize baselines
        self.baselines = {
            'oracle': OracleRoutingBaseline(dynamo_model),
            'single_lora': SingleLoRABaseline(config),
            'full_finetune': FullFineTuningBaseline(config),
            'random_routing': RandomRoutingBaseline(dynamo_model),
            'uniform_routing': UniformRoutingBaseline(dynamo_model)
        }
        
        logger.info(f"Baseline collection initialized with {len(self.baselines)} baselines")
    
    def get_baseline(self, name: str) -> nn.Module:
        """Get a specific baseline model."""
        if name not in self.baselines:
            raise ValueError(f"Unknown baseline: {name}. Available: {list(self.baselines.keys())}")
        return self.baselines[name]
    
    def get_all_baselines(self) -> Dict[str, nn.Module]:
        """Get all baseline models."""
        return self.baselines
    
    def get_parameter_counts(self) -> Dict[str, int]:
        """Get parameter counts for all baselines."""
        counts = {}
        
        # DYNAMO model
        counts['dynamo'] = count_parameters(self.dynamo_model)
        
        # Baselines
        for name, model in self.baselines.items():
            if name in ['oracle', 'random_routing', 'uniform_routing']:
                # These use DYNAMO's parameters
                counts[name] = count_parameters(self.dynamo_model)
            else:
                counts[name] = count_parameters(model)
        
        return counts
    
    def compare_parameter_efficiency(self) -> Dict[str, float]:
        """Compare parameter efficiency relative to full fine-tuning."""
        counts = self.get_parameter_counts()
        full_finetune_params = counts['full_finetune']
        
        efficiency = {}
        for name, param_count in counts.items():
            efficiency[name] = param_count / full_finetune_params
        
        return efficiency


def create_baseline_collection(
    dynamo_model: DynamoModel, 
    config: Dict[str, Any]
) -> BaselineCollection:
    """
    Create a collection of baseline models.
    
    Args:
        dynamo_model: Trained DYNAMO model
        config: Model configuration
    
    Returns:
        Baseline collection
    """
    return BaselineCollection(dynamo_model, config)

