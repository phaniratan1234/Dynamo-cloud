"""
DYNAMO: Dynamic Neural Adapter Multi-task Optimization

A PyTorch implementation of a novel multi-task learning framework that uses 
dynamic routing to select task-specific LoRA adapters based on input characteristics.

Main Components:
- model: Core DYNAMO model implementation
- training: Three-phase training pipeline
- data: Dataset loading and processing
- evaluation: Metrics, baselines, and analysis tools
- utils: Configuration, logging, and helper functions
"""

__version__ = "1.0.0"
__author__ = "DYNAMO Team"

# Import main components for easy access
from .model import DynamoModel
from .utils import Config, get_config
from .training import run_phase1_training, run_phase2_training, run_phase3_training
from .evaluation import create_evaluator, create_baseline_collection, create_routing_analyzer

__all__ = [
    'DynamoModel',
    'Config',
    'get_config',
    'run_phase1_training',
    'run_phase2_training', 
    'run_phase3_training',
    'create_evaluator',
    'create_baseline_collection',
    'create_routing_analyzer'
]

