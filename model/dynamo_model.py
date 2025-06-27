"""
DYNAMO: Dynamic Neural Adapter Mixture Optimization
Main model that integrates RoBERTa backbone, LoRA adapters, and dynamic router.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

from .roberta_backbone import RobertaBackbone
from .lora_adapters import LoRAAdapterCollection
from .dynamic_router import DynamicRouter
from utils.logger import get_logger
from utils.helpers import count_parameters, move_to_device

logger = get_logger(__name__)


class DynamoModel(nn.Module):
    """
    DYNAMO: Dynamic Neural Adapter Mixture Optimization
    
    A multi-task NLP system that automatically detects task requirements from raw input text
    and dynamically routes to appropriate LoRA adapters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DYNAMO model.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__()
        
        self.config = config
        self.model_config = config.get("model", {})
        
        # Initialize components
        logger.info("Initializing DYNAMO model components...")
        
        # 1. RoBERTa Backbone (frozen)
        self.backbone = RobertaBackbone(
            model_name=self.model_config.get("base_model_name", "roberta-base"),
            freeze=True
        )
        
        # 2. LoRA Adapter Collection
        adapter_config = {
            "hidden_size": self.backbone.hidden_size,
            "num_layers": 12,  # RoBERTa-base has 12 layers
            "lora_configs": self.model_config.get("lora_configs", {})
        }
        self.adapters = LoRAAdapterCollection(adapter_config)
        
        # 3. Dynamic Router
        router_config = self.model_config
        self.router = DynamicRouter(
            input_size=self.backbone.hidden_size,
            hidden_sizes=router_config.get("router_hidden_sizes", [512, 256]),
            num_tasks=self.adapters.get_num_tasks(),
            dropout=router_config.get("router_dropout", 0.1),
            temperature_init=router_config.get("temperature_init", 1.0),
            temperature_learnable=router_config.get("temperature_learnable", True)
        )
        
        # Task mapping
        self.task_names = self.adapters.get_task_names()
        self.task_to_idx = {task: idx for idx, task in enumerate(self.task_names)}
        self.idx_to_task = {idx: task for task, idx in self.task_to_idx.items()}
        
        # Training mode flags
        self.training_phase = "phase1"  # phase1, phase2, phase3
        self.hard_routing = False
        self.gumbel_temperature = 1.0
        
        logger.info(f"DYNAMO model initialized with {count_parameters(self):,} total parameters")
        logger.info(f"  - Backbone (frozen): {count_parameters(self.backbone):,}")
        logger.info(f"  - Adapters: {count_parameters(self.adapters):,}")
        logger.info(f"  - Router: {count_parameters(self.router):,}")
        logger.info(f"  - Tasks: {self.task_names}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_labels: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DYNAMO model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            task_labels: Ground truth task labels [batch_size] (for supervised routing)
            return_routing_info: Whether to return detailed routing information
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing:
                - task_outputs: Dictionary of task-specific predictions
                - routing_probs: Routing probabilities [batch_size, num_tasks]
                - routing_info: Additional routing information (if requested)
                - backbone_outputs: RoBERTa outputs (if requested)
        """
        batch_size = input_ids.size(0)
        
        # 1. Get embeddings from RoBERTa backbone
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract CLS embeddings for routing
        cls_embeddings = backbone_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # 2. Dynamic routing decision
        routing_output = self.router(
            input_embeddings=cls_embeddings,
            return_confidence=return_routing_info,
            return_entropy=return_routing_info,
            hard_routing=self.hard_routing,
            gumbel_temperature=self.gumbel_temperature
        )
        
        routing_probs = routing_output['routing_probs']  # [batch_size, num_tasks]
        
        # 3. Apply LoRA adapters based on routing
        if self.training_phase == "phase1":
            # Phase 1: Train individual adapters (oracle routing)
            task_outputs = self._forward_phase1(
                backbone_outputs.last_hidden_state,
                task_labels,
                routing_probs
            )
        
        elif self.training_phase == "phase2":
            # Phase 2: Train router (frozen adapters)
            task_outputs = self._forward_phase2(
                backbone_outputs.last_hidden_state,
                routing_probs
            )
        
        else:  # phase3 or inference
            # Phase 3: Joint training or inference
            task_outputs = self._forward_phase3(
                backbone_outputs.last_hidden_state,
                routing_probs
            )
        
        # Prepare output
        output = {
            'task_outputs': task_outputs,
            'routing_probs': routing_probs
        }
        
        if return_routing_info:
            routing_info = {
                'routing_logits': routing_output['routing_logits'],
                'temperature': self.router.get_temperature()
            }
            if 'confidence' in routing_output:
                routing_info['confidence'] = routing_output['confidence']
            if 'entropy' in routing_output:
                routing_info['entropy'] = routing_output['entropy']
            
            output['routing_info'] = routing_info
        
        return output
    
    def _forward_phase1(
        self,
        hidden_states: torch.Tensor,
        task_labels: Optional[torch.Tensor],
        routing_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Phase 1: Individual LoRA training.
        Uses oracle routing based on task labels.
        """
        if task_labels is None:
            raise ValueError("Task labels required for Phase 1 training")
        
        task_outputs = {}
        
        # Use oracle routing (one-hot based on task labels)
        oracle_routing = torch.zeros_like(routing_probs)
        oracle_routing.scatter_(1, task_labels.unsqueeze(1), 1.0)
        
        # Forward through adapters with oracle routing
        task_outputs = self.adapters.forward_task_heads(
            hidden_states,
            oracle_routing,
            hard_routing=True
        )
        
        return task_outputs
    
    def _forward_phase2(
        self,
        hidden_states: torch.Tensor,
        routing_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Phase 2: Router training.
        Uses frozen adapters with learned routing.
        """
        # Forward through adapters with learned routing
        task_outputs = self.adapters.forward_task_heads(
            hidden_states,
            routing_probs,
            hard_routing=self.hard_routing
        )
        
        return task_outputs
    
    def _forward_phase3(
        self,
        hidden_states: torch.Tensor,
        routing_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Phase 3: Joint fine-tuning or inference.
        Uses learned routing with trainable adapters.
        """
        # Forward through adapters with learned routing
        task_outputs = self.adapters.forward_task_heads(
            hidden_states,
            routing_probs,
            hard_routing=self.hard_routing
        )
        
        return task_outputs
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_routing_info: bool = True,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Make predictions on input texts.
        
        Args:
            texts: Input text(s)
            return_routing_info: Whether to return routing information
            device: Device to run inference on
        
        Returns:
            Dictionary with predictions and routing information
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if device is None:
            device = next(self.parameters()).device
        
        # Tokenize inputs
        tokenized = self.backbone.tokenize(texts)
        tokenized = move_to_device(tokenized, device)
        
        # Forward pass
        with torch.no_grad():
            self.eval()
            outputs = self.forward(
                input_ids=tokenized['input_ids'],
                attention_mask=tokenized['attention_mask'],
                return_routing_info=return_routing_info
            )
        
        # Process predictions
        predictions = {}
        task_outputs = outputs['task_outputs']
        routing_probs = outputs['routing_probs']
        
        for task_name, task_output in task_outputs.items():
            if task_name == "sentiment":
                # Binary classification
                probs = F.softmax(task_output, dim=-1)
                predicted_labels = torch.argmax(probs, dim=-1)
                predictions[task_name] = {
                    'labels': predicted_labels.cpu().tolist(),
                    'probabilities': probs.cpu().tolist(),
                    'predictions': ['negative' if label == 0 else 'positive' 
                                  for label in predicted_labels.cpu().tolist()]
                }
            
            elif task_name == "qa":
                # Question answering (start/end positions)
                start_logits, end_logits = task_output.split(1, dim=-1)
                start_probs = F.softmax(start_logits.squeeze(-1), dim=-1)
                end_probs = F.softmax(end_logits.squeeze(-1), dim=-1)
                
                predictions[task_name] = {
                    'start_positions': torch.argmax(start_probs, dim=-1).cpu().tolist(),
                    'end_positions': torch.argmax(end_probs, dim=-1).cpu().tolist(),
                    'start_probabilities': start_probs.cpu().tolist(),
                    'end_probabilities': end_probs.cpu().tolist()
                }
            
            else:
                # Generation tasks (simplified)
                predictions[task_name] = {
                    'hidden_states': task_output.cpu().tolist()
                }
        
        # Add routing information
        if return_routing_info:
            routing_info = outputs.get('routing_info', {})
            selected_tasks = torch.argmax(routing_probs, dim=-1)
            
            predictions['routing'] = {
                'probabilities': routing_probs.cpu().tolist(),
                'selected_tasks': [self.idx_to_task[idx.item()] for idx in selected_tasks],
                'task_names': self.task_names,
                'temperature': routing_info.get('temperature', self.router.get_temperature())
            }
            
            if 'confidence' in routing_info:
                predictions['routing']['confidence'] = routing_info['confidence'].cpu().tolist()
            if 'entropy' in routing_info:
                predictions['routing']['entropy'] = routing_info['entropy'].cpu().tolist()
        
        return predictions
    
    def set_training_phase(self, phase: str):
        """
        Set training phase and configure model accordingly.
        
        Args:
            phase: Training phase ("phase1", "phase2", "phase3")
        """
        self.training_phase = phase
        
        if phase == "phase1":
            # Phase 1: Train individual adapters
            self.adapters.unfreeze_adapters()
            self.router.requires_grad_(False)  # Freeze router
            logger.info("Set to Phase 1: Training individual LoRA adapters")
        
        elif phase == "phase2":
            # Phase 2: Train router
            self.adapters.freeze_adapters()
            self.router.requires_grad_(True)  # Unfreeze router
            logger.info("Set to Phase 2: Training dynamic router")
        
        elif phase == "phase3":
            # Phase 3: Joint training
            self.adapters.unfreeze_adapters()
            self.router.requires_grad_(True)
            logger.info("Set to Phase 3: Joint fine-tuning")
        
        else:
            raise ValueError(f"Unknown training phase: {phase}")
    
    def set_routing_mode(self, hard_routing: bool = False, gumbel_temperature: float = 1.0):
        """
        Set routing mode.
        
        Args:
            hard_routing: Whether to use hard (discrete) routing
            gumbel_temperature: Temperature for Gumbel-Softmax
        """
        self.hard_routing = hard_routing
        self.gumbel_temperature = gumbel_temperature
        
        mode = "hard" if hard_routing else "soft"
        logger.info(f"Set routing mode to {mode} (temperature: {gumbel_temperature})")
    
    def get_routing_statistics(self, dataloader) -> Dict[str, float]:
        """
        Get routing statistics over a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
        
        Returns:
            Dictionary of routing statistics
        """
        self.eval()
        all_stats = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = move_to_device(batch, next(self.parameters()).device)
                
                # Get backbone embeddings
                backbone_outputs = self.backbone(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                cls_embeddings = backbone_outputs.last_hidden_state[:, 0, :]
                
                # Get routing statistics
                stats = self.router.get_routing_statistics(cls_embeddings)
                all_stats.append(stats)
        
        # Aggregate statistics
        aggregated_stats = {}
        for key in all_stats[0].keys():
            values = [stats[key] for stats in all_stats]
            aggregated_stats[key] = sum(values) / len(values)
        
        return aggregated_stats
    
    def save_model(self, save_dir: str):
        """
        Save the complete DYNAMO model.
        
        Args:
            save_dir: Directory to save the model
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        model_state = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'task_names': self.task_names,
            'training_phase': self.training_phase
        }
        torch.save(model_state, os.path.join(save_dir, "dynamo_model.pt"))
        
        # Save backbone separately
        self.backbone.save_pretrained(os.path.join(save_dir, "backbone"))
        
        # Save adapters separately
        self.adapters.save_adapters(os.path.join(save_dir, "adapters"))
        
        logger.info(f"DYNAMO model saved to {save_dir}")
    
    @classmethod
    def load_model(cls, save_dir: str, device: Optional[torch.device] = None):
        """
        Load a saved DYNAMO model.
        
        Args:
            save_dir: Directory containing the saved model
            device: Device to load the model on
        
        Returns:
            Loaded DYNAMO model
        """
        import os
        
        # Load model state
        model_path = os.path.join(save_dir, "dynamo_model.pt")
        model_state = torch.load(model_path, map_location=device)
        
        # Create model instance
        model = cls(model_state['config'])
        model.load_state_dict(model_state['model_state_dict'])
        model.training_phase = model_state.get('training_phase', 'phase3')
        
        # Load adapters
        adapter_dir = os.path.join(save_dir, "adapters")
        if os.path.exists(adapter_dir):
            model.adapters.load_adapters(adapter_dir)
        
        if device is not None:
            model = model.to(device)
        
        logger.info(f"DYNAMO model loaded from {save_dir}")
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'total_parameters': count_parameters(self),
            'backbone_parameters': count_parameters(self.backbone),
            'adapter_parameters': count_parameters(self.adapters),
            'router_parameters': count_parameters(self.router),
            'task_names': self.task_names,
            'training_phase': self.training_phase,
            'routing_mode': 'hard' if self.hard_routing else 'soft',
            'temperature': self.router.get_temperature(),
            'hidden_size': self.backbone.hidden_size,
            'num_tasks': len(self.task_names)
        }

