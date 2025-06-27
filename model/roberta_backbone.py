"""
RoBERTa backbone implementation for DYNAMO.
Provides the frozen base model that generates embeddings for both routing and task execution.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from typing import Dict, Optional, Tuple
import warnings

from utils.logger import get_logger
from utils.helpers import freeze_parameters, count_parameters

logger = get_logger(__name__)


class RobertaBackbone(nn.Module):
    """
    Frozen RoBERTa backbone for DYNAMO.
    
    This module wraps a pre-trained RoBERTa model and provides:
    1. Frozen embeddings for routing decisions
    2. Frozen embeddings for task-specific processing
    3. Utilities for tokenization and preprocessing
    """
    
    def __init__(self, model_name: str = "roberta-base", freeze: bool = True):
        """
        Initialize RoBERTa backbone.
        
        Args:
            model_name: Name of the pre-trained RoBERTa model
            freeze: Whether to freeze the model parameters
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Load pre-trained model and tokenizer
        logger.info(f"Loading RoBERTa model: {model_name}")
        self.config = RobertaConfig.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Freeze parameters if specified
        if freeze:
            freeze_parameters(self.model)
            logger.info(f"Frozen RoBERTa parameters: {count_parameters(self.model):,}")
        else:
            logger.info(f"Trainable RoBERTa parameters: {count_parameters(self.model):,}")
        
        # Store important dimensions
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.max_position_embeddings = self.config.max_position_embeddings
        
        logger.info(f"RoBERTa backbone initialized - Hidden size: {self.hidden_size}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through RoBERTa.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (not used in RoBERTa)
            return_dict: Whether to return a dictionary
            output_hidden_states: Whether to output hidden states
            output_attentions: Whether to output attention weights
        
        Returns:
            Dictionary containing:
                - last_hidden_state: [batch_size, seq_len, hidden_size]
                - pooler_output: [batch_size, hidden_size] (CLS token representation)
                - hidden_states: List of hidden states (if requested)
                - attentions: List of attention weights (if requested)
        """
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict
            )
        
        return outputs
    
    def get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get CLS token embedding for routing decisions.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            CLS embeddings [batch_size, hidden_size]
        """
        outputs = self.forward(input_ids, attention_mask)
        
        # Extract CLS token (first token) embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        return cls_embeddings
    
    def get_pooled_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling_strategy: str = "cls"
    ) -> torch.Tensor:
        """
        Get pooled embeddings using different strategies.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            pooling_strategy: Pooling strategy ("cls", "mean", "max", "attention")
        
        Returns:
            Pooled embeddings [batch_size, hidden_size]
        """
        outputs = self.forward(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        if pooling_strategy == "cls":
            return last_hidden_state[:, 0, :]
        
        elif pooling_strategy == "mean":
            # Mean pooling with attention mask
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(last_hidden_state, dim=1)
        
        elif pooling_strategy == "max":
            # Max pooling with attention mask
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                last_hidden_state = last_hidden_state * mask_expanded + (1 - mask_expanded) * (-1e9)
            return torch.max(last_hidden_state, dim=1)[0]
        
        elif pooling_strategy == "attention":
            # Attention-weighted pooling
            attention_weights = torch.softmax(
                torch.sum(last_hidden_state, dim=-1), dim=-1
            )  # [batch_size, seq_len]
            
            if attention_mask is not None:
                attention_weights = attention_weights * attention_mask.float()
                attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-9)
            
            weighted_embeddings = torch.sum(
                last_hidden_state * attention_weights.unsqueeze(-1), dim=1
            )
            return weighted_embeddings
        
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
    
    def tokenize(
        self,
        texts: list,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Format of returned tensors
        
        Returns:
            Dictionary with tokenized inputs
        """
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> list:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_hidden_size(self) -> int:
        """Get hidden size."""
        return self.hidden_size
    
    def get_max_length(self) -> int:
        """Get maximum sequence length."""
        return self.max_position_embeddings
    
    def save_pretrained(self, save_directory: str):
        """Save the model and tokenizer."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"RoBERTa backbone saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, freeze: bool = True):
        """
        Load a pre-trained RoBERTa backbone.
        
        Args:
            model_path: Path to the saved model
            freeze: Whether to freeze parameters
        
        Returns:
            RobertaBackbone instance
        """
        instance = cls.__new__(cls)
        super(RobertaBackbone, instance).__init__()
        
        instance.model_name = model_path
        instance.freeze = freeze
        
        # Load model and tokenizer
        instance.config = RobertaConfig.from_pretrained(model_path)
        instance.model = RobertaModel.from_pretrained(model_path)
        instance.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        
        # Freeze if specified
        if freeze:
            freeze_parameters(instance.model)
        
        # Store dimensions
        instance.hidden_size = instance.config.hidden_size
        instance.vocab_size = instance.config.vocab_size
        instance.max_position_embeddings = instance.config.max_position_embeddings
        
        logger.info(f"RoBERTa backbone loaded from {model_path}")
        return instance
    
    def train(self, mode: bool = True):
        """Set training mode."""
        if self.freeze:
            # Keep the model in eval mode if frozen
            self.model.eval()
            return self
        else:
            return super().train(mode)
    
    def eval(self):
        """Set evaluation mode."""
        return super().eval()


class RobertaEmbeddingExtractor:
    """
    Utility class for extracting embeddings from RoBERTa backbone.
    Useful for analysis and visualization.
    """
    
    def __init__(self, backbone: RobertaBackbone):
        """
        Initialize embedding extractor.
        
        Args:
            backbone: RoBERTa backbone model
        """
        self.backbone = backbone
    
    def extract_layer_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[list] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from specific layers.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            layers: List of layer indices to extract (None for all)
        
        Returns:
            Dictionary mapping layer index to embeddings
        """
        outputs = self.backbone.forward(
            input_ids, attention_mask, output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states  # Tuple of tensors
        
        if layers is None:
            layers = list(range(len(hidden_states)))
        
        layer_embeddings = {}
        for layer_idx in layers:
            if 0 <= layer_idx < len(hidden_states):
                layer_embeddings[layer_idx] = hidden_states[layer_idx]
        
        return layer_embeddings
    
    def extract_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[list] = None,
        heads: Optional[list] = None
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Extract attention weights from specific layers and heads.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            layers: List of layer indices
            heads: List of head indices
        
        Returns:
            Dictionary mapping (layer, head) to attention weights
        """
        outputs = self.backbone.forward(
            input_ids, attention_mask, output_attentions=True
        )
        
        attentions = outputs.attentions  # Tuple of tensors
        
        if layers is None:
            layers = list(range(len(attentions)))
        if heads is None:
            heads = list(range(attentions[0].size(1)))  # Number of heads
        
        attention_weights = {}
        for layer_idx in layers:
            if 0 <= layer_idx < len(attentions):
                layer_attention = attentions[layer_idx]  # [batch, heads, seq, seq]
                for head_idx in heads:
                    if 0 <= head_idx < layer_attention.size(1):
                        attention_weights[(layer_idx, head_idx)] = layer_attention[:, head_idx, :, :]
        
        return attention_weights

