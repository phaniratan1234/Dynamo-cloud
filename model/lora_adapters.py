"""
LoRA (Low-Rank Adaptation) adapters for DYNAMO.
Implements task-specific adapters for sentiment analysis, QA, summarization, 
code generation, and translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from utils.logger import get_logger
from utils.helpers import count_parameters

logger = get_logger(__name__)


class LoRALayer(nn.Module):
    """
    Single LoRA layer implementation.
    Applies low-rank adaptation to a linear layer.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        bias: bool = False
    ):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Rank of the low-rank adaptation
            alpha: Scaling factor for LoRA
            dropout: Dropout rate
            bias: Whether to include bias
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.
        
        Args:
            x: Input tensor [batch_size, ..., in_features]
        
        Returns:
            Output tensor [batch_size, ..., out_features]
        """
        # Apply LoRA: x @ (A @ B) * scaling
        lora_output = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        
        if self.bias is not None:
            lora_output = lora_output + self.bias
        
        return lora_output


class TaskSpecificLoRA(nn.Module):
    """
    Task-specific LoRA adapter that can be applied to multiple layers.
    """
    
    def __init__(
        self,
        task_name: str,
        hidden_size: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        target_modules: List[str] = None,
        num_layers: int = 12
    ):
        """
        Initialize task-specific LoRA adapter.
        
        Args:
            task_name: Name of the task
            hidden_size: Hidden size of the base model
            rank: Rank of LoRA adaptation
            alpha: Scaling factor
            dropout: Dropout rate
            target_modules: List of module names to adapt
            num_layers: Number of transformer layers
        """
        super().__init__()
        
        self.task_name = task_name
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Default target modules (attention and feed-forward)
        if target_modules is None:
            target_modules = ["query", "key", "value", "dense", "intermediate", "output"]
        self.target_modules = target_modules
        
        # Create LoRA layers for each target module in each layer
        self.lora_layers = nn.ModuleDict()
        
        for layer_idx in range(num_layers):
            layer_dict = nn.ModuleDict()
            
            for module_name in target_modules:
                if module_name in ["query", "key", "value", "dense"]:
                    # Attention modules
                    layer_dict[module_name] = LoRALayer(
                        hidden_size, hidden_size, rank, alpha, dropout
                    )
                elif module_name == "intermediate":
                    # Feed-forward intermediate layer (usually 4x hidden size)
                    layer_dict[module_name] = LoRALayer(
                        hidden_size, hidden_size * 4, rank, alpha, dropout
                    )
                elif module_name == "output":
                    # Feed-forward output layer
                    layer_dict[module_name] = LoRALayer(
                        hidden_size * 4, hidden_size, rank, alpha, dropout
                    )
            
            self.lora_layers[f"layer_{layer_idx}"] = layer_dict
        
        # Task-specific head (for final predictions)
        self.task_head = self._create_task_head()
        
        logger.info(f"Created {task_name} LoRA adapter with {count_parameters(self):,} parameters")
    
    def _create_task_head(self) -> nn.Module:
        """Create task-specific prediction head."""
        if self.task_name == "sentiment":
            # Binary sentiment classification
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size // 2, 2)  # Positive/Negative
            )
        
        elif self.task_name == "qa":
            # Question answering (start/end positions)
            return nn.Sequential(
                nn.Linear(self.hidden_size, 2)  # Start and end logits
            )
        
        elif self.task_name in ["summarization", "code_generation", "translation"]:
            # Generation tasks: predict vocabulary logits with positional encoding
            return nn.ModuleDict({
                'vocab_head': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_size, 50265)  # RoBERTa vocab size
                ),
                'pos_projection': nn.Linear(self.hidden_size + 1, self.hidden_size)
            })
        
        else:
            # Generic classification head
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size // 2, 1)
            )
    
    def apply_to_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        module_name: str
    ) -> torch.Tensor:
        """
        Apply LoRA adaptation to a specific layer and module.
        
        Args:
            hidden_states: Input hidden states
            layer_idx: Layer index
            module_name: Module name
        
        Returns:
            Adapted hidden states
        """
        layer_key = f"layer_{layer_idx}"
        
        if layer_key in self.lora_layers and module_name in self.lora_layers[layer_key]:
            lora_layer = self.lora_layers[layer_key][module_name]
            return lora_layer(hidden_states)
        else:
            return torch.zeros_like(hidden_states)
    
    def forward(self, hidden_states: torch.Tensor, task_type: str = None) -> torch.Tensor:
        """
        Forward pass through task head.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            task_type: Type of task output needed
        
        Returns:
            Task-specific predictions
        """
        # Use CLS token for classification tasks
        if self.task_name in ["sentiment"]:
            cls_hidden = hidden_states[:, 0, :]  # [batch_size, hidden_size]
            return self.task_head(cls_hidden)
        
        # Use all tokens for sequence labeling tasks
        elif self.task_name == "qa":
            return self.task_head(hidden_states)  # [batch_size, seq_len, 2]
        
        # Use proper sequence-to-sequence prediction for generation tasks
        elif self.task_name in ["summarization", "code_generation", "translation"]:
            # Use all input tokens for context-aware generation
            batch_size, input_seq_len, hidden_size = hidden_states.shape
            
            # Define target sequence length for each task
            if self.task_name == "summarization":
                target_seq_len = 128
            else:  # code_generation, translation
                target_seq_len = 256
            
            # Use mean-pooled representation for each target position
            pooled_hidden = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
            
            # Get the ModuleDict components
            pos_projection = self.task_head['pos_projection']
            vocab_head = self.task_head['vocab_head']
            
            # Generate position-aware embeddings
            vocab_logits_list = []
            for pos in range(target_seq_len):
                # Add position-specific information
                pos_encoding = torch.full_like(pooled_hidden[:, :1], pos / target_seq_len)
                position_hidden = torch.cat([pooled_hidden, pos_encoding], dim=1)
                
                # Project to match original hidden size
                position_hidden_projected = pos_projection(position_hidden)
                vocab_logits = vocab_head(position_hidden_projected)
                vocab_logits_list.append(vocab_logits)
            
            # Stack to create [batch_size, target_seq_len, vocab_size]
            vocab_logits_seq = torch.stack(vocab_logits_list, dim=1)
            return vocab_logits_seq
        
        # Use CLS token for other tasks
        else:
            cls_hidden = hidden_states[:, 0, :]  # [batch_size, hidden_size]
            return self.task_head(cls_hidden)


class LoRAAdapterCollection(nn.Module):
    """
    Collection of all LoRA adapters for different tasks.
    Manages the five task-specific adapters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize collection of LoRA adapters.
        
        Args:
            config: Configuration dictionary containing adapter settings
        """
        super().__init__()
        
        # Handle both Config objects and dictionaries
        if hasattr(config, '__dict__'):
            # If it's a Config object, convert to dict
            self.config = config.__dict__ if hasattr(config, '__dict__') else config
        else:
            self.config = config
            
        self.hidden_size = self.config.get("hidden_size", 768)
        self.num_layers = self.config.get("num_layers", 12)
        
        # Task configurations
        default_configs = {
            "sentiment": {"rank": 16, "alpha": 32, "dropout": 0.1},
            "qa": {"rank": 32, "alpha": 64, "dropout": 0.1},
            "summarization": {"rank": 24, "alpha": 48, "dropout": 0.1},
            "code_generation": {"rank": 20, "alpha": 40, "dropout": 0.1},
            "translation": {"rank": 28, "alpha": 56, "dropout": 0.1}
        }
        
        lora_configs = self.config.get("lora_configs", default_configs)
        
        # Handle SimpleConfig objects
        if hasattr(lora_configs, '__dict__'):
            # Convert SimpleConfig to dict
            self.task_configs = {}
            for attr_name in dir(lora_configs):
                if not attr_name.startswith('_'):
                    attr_value = getattr(lora_configs, attr_name)
                    if hasattr(attr_value, '__dict__'):
                        # Convert nested SimpleConfig to dict
                        self.task_configs[attr_name] = attr_value.__dict__
                    else:
                        self.task_configs[attr_name] = attr_value
        else:
            # Regular dictionary
            self.task_configs = lora_configs
        
        # Create adapters
        self.adapters = nn.ModuleDict()
        self.task_names = list(self.task_configs.keys())
        
        for task_name, task_config in self.task_configs.items():
            self.adapters[task_name] = TaskSpecificLoRA(
                task_name=task_name,
                hidden_size=self.hidden_size,
                rank=task_config["rank"],
                alpha=task_config["alpha"],
                dropout=task_config["dropout"],
                num_layers=self.num_layers
            )
        
        logger.info(f"Created LoRA adapter collection with {len(self.adapters)} adapters")
        logger.info(f"Total LoRA parameters: {count_parameters(self):,}")
    
    def get_adapter(self, task_name: str) -> TaskSpecificLoRA:
        """Get adapter for a specific task."""
        if task_name not in self.adapters:
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {self.task_names}")
        return self.adapters[task_name]
    
    def apply_weighted_adapters(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        layer_idx: int,
        module_name: str
    ) -> torch.Tensor:
        """
        Apply weighted combination of adapters.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            routing_weights: Routing weights [batch_size, num_tasks]
            layer_idx: Layer index
            module_name: Module name
        
        Returns:
            Weighted adapter outputs [batch_size, seq_len, hidden_size]
        """
        batch_size = hidden_states.size(0)
        weighted_output = torch.zeros_like(hidden_states)
        
        for i, (task_name, adapter) in enumerate(self.adapters.items()):
            # Apply adapter
            adapter_output = adapter.apply_to_layer(hidden_states, layer_idx, module_name)
            
            # Weight by routing probability
            task_weight = routing_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
            weighted_output += task_weight * adapter_output
        
        return weighted_output
    
    def forward_task_heads(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        hard_routing: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through task heads with routing.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            routing_weights: Routing weights [batch_size, num_tasks]
            hard_routing: Whether to use hard (discrete) routing
        
        Returns:
            Dictionary of task predictions
        """
        task_outputs = {}
        
        if hard_routing:
            # Use only the highest-weighted adapter
            selected_task_idx = torch.argmax(routing_weights, dim=-1)  # [batch_size]
            
            for batch_idx in range(hidden_states.size(0)):
                task_idx = selected_task_idx[batch_idx].item()
                task_name = self.task_names[task_idx]
                adapter = self.adapters[task_name]
                
                # Get prediction for this sample
                sample_hidden = hidden_states[batch_idx:batch_idx+1]  # [1, seq_len, hidden_size]
                prediction = adapter.forward(sample_hidden)
                
                if task_name not in task_outputs:
                    # Initialize output tensor
                    output_shape = [hidden_states.size(0)] + list(prediction.shape[1:])
                    task_outputs[task_name] = torch.zeros(output_shape, device=prediction.device)
                
                task_outputs[task_name][batch_idx] = prediction[0]
        
        else:
            # Soft routing: weighted combination
            for i, (task_name, adapter) in enumerate(self.adapters.items()):
                prediction = adapter.forward(hidden_states)
                task_weight = routing_weights[:, i]  # [batch_size]
                
                # Apply weight
                if prediction.dim() == 2:  # [batch_size, num_classes]
                    weighted_prediction = prediction * task_weight.unsqueeze(-1)
                else:  # [batch_size, seq_len, num_classes]
                    weighted_prediction = prediction * task_weight.unsqueeze(-1).unsqueeze(-1)
                
                if task_name not in task_outputs:
                    task_outputs[task_name] = weighted_prediction
                else:
                    task_outputs[task_name] += weighted_prediction
        
        return task_outputs
    
    def get_task_names(self) -> List[str]:
        """Get list of task names."""
        return self.task_names
    
    def get_num_tasks(self) -> int:
        """Get number of tasks."""
        return len(self.task_names)
    
    def freeze_adapters(self, task_names: Optional[List[str]] = None):
        """
        Freeze specific adapters.
        
        Args:
            task_names: List of task names to freeze (None for all)
        """
        if task_names is None:
            task_names = self.task_names
        
        for task_name in task_names:
            if task_name in self.adapters:
                for param in self.adapters[task_name].parameters():
                    param.requires_grad = False
                logger.info(f"Frozen {task_name} adapter")
    
    def unfreeze_adapters(self, task_names: Optional[List[str]] = None):
        """
        Unfreeze specific adapters.
        
        Args:
            task_names: List of task names to unfreeze (None for all)
        """
        if task_names is None:
            task_names = self.task_names
        
        for task_name in task_names:
            if task_name in self.adapters:
                for param in self.adapters[task_name].parameters():
                    param.requires_grad = True
                logger.info(f"Unfrozen {task_name} adapter")
    
    def save_adapters(self, save_dir: str):
        """Save all adapters."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for task_name, adapter in self.adapters.items():
            adapter_path = os.path.join(save_dir, f"{task_name}_adapter.pt")
            torch.save(adapter.state_dict(), adapter_path)
            logger.info(f"Saved {task_name} adapter to {adapter_path}")
    
    def load_adapters(self, save_dir: str):
        """Load all adapters."""
        import os
        
        for task_name, adapter in self.adapters.items():
            adapter_path = os.path.join(save_dir, f"{task_name}_adapter.pt")
            if os.path.exists(adapter_path):
                adapter.load_state_dict(torch.load(adapter_path))
                logger.info(f"Loaded {task_name} adapter from {adapter_path}")
            else:
                logger.warning(f"Adapter file not found: {adapter_path}")

