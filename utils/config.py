"""
Configuration management for DYNAMO project.
Centralizes all hyperparameters and settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Base model
    base_model_name: str = "roberta-base"
    hidden_size: int = 768
    
    # LoRA adapter configurations
    lora_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "sentiment": {"rank": 16, "alpha": 32, "dropout": 0.1},
        "qa": {"rank": 32, "alpha": 64, "dropout": 0.1},
        "summarization": {"rank": 24, "alpha": 48, "dropout": 0.1},
        "code_generation": {"rank": 20, "alpha": 40, "dropout": 0.1},
        "translation": {"rank": 28, "alpha": 56, "dropout": 0.1}
    })
    
    # Router configuration
    router_hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])
    router_dropout: float = 0.1
    num_tasks: int = 5
    temperature_init: float = 1.0
    temperature_learnable: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    # General training settings
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_length: int = 512
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Phase-specific learning rates
    lora_lr: float = 3e-4
    router_lr: float = 1e-3
    joint_lr: float = 1e-4
    
    # Loss function weights
    load_balance_weight: float = 0.1
    efficiency_weight: float = 0.05
    consistency_weight: float = 0.1
    
    # Router training specifics
    gumbel_temperature: float = 1.0
    temperature_decay: float = 0.999
    min_temperature: float = 0.1
    
    # Curriculum learning
    curriculum_start_ratio: float = 0.8  # Start with 80% clear examples
    curriculum_end_ratio: float = 0.2    # End with 20% clear examples


@dataclass
class DataConfig:
    """Configuration for datasets."""
    # Dataset sizes (for subsampling)
    sst2_size: int = 10000
    squad_size: int = 20000
    xsum_size: int = 15000
    code_gen_size: int = 8000
    translation_size: int = 12000
    
    # Mixed task dataset
    mixed_task_size: int = 5000
    
    # Data paths
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    
    # Preprocessing
    max_input_length: int = 512
    max_target_length: int = 128


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    eval_batch_size: int = 32
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Metrics to compute
    compute_rouge: bool = True
    compute_bleu: bool = True
    compute_accuracy: bool = True
    
    # Analysis settings
    visualize_routing: bool = True
    save_routing_decisions: bool = True


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # General settings
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "dynamo"
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)


def get_config() -> Config:
    """Get default configuration."""
    return Config()


def update_config_from_args(config: Config, args: dict) -> Config:
    """Update configuration from command line arguments."""
    for key, value in args.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.evaluation, key):
            setattr(config.evaluation, key, value)
    return config

