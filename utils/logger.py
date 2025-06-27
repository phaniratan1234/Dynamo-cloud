"""
Logging utility for DYNAMO project.
Provides standardized logging across all modules.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class DynamoLogger:
    """Custom logger for DYNAMO project."""
    
    def __init__(self, name: str, log_dir: str = "./logs", level: int = logging.INFO):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory to save log files
            level: Logging level
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"{self.name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_training_step(self, step: int, loss: float, lr: float, additional_metrics: Optional[dict] = None):
        """Log training step information."""
        message = f"Step {step}: Loss={loss:.4f}, LR={lr:.2e}"
        if additional_metrics:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in additional_metrics.items()])
            message += f", {metrics_str}"
        self.info(message)
    
    def log_evaluation_results(self, phase: str, results: dict):
        """Log evaluation results."""
        self.info(f"=== {phase} Evaluation Results ===")
        for metric, value in results.items():
            if isinstance(value, float):
                self.info(f"{metric}: {value:.4f}")
            else:
                self.info(f"{metric}: {value}")
    
    def log_router_decision(self, input_text: str, routing_probs: list, selected_adapter: str):
        """Log router decision for analysis."""
        probs_str = ", ".join([f"{prob:.3f}" for prob in routing_probs])
        self.debug(f"Router Decision - Input: '{input_text[:50]}...', "
                  f"Probs: [{probs_str}], Selected: {selected_adapter}")
    
    def log_phase_start(self, phase: str):
        """Log the start of a training phase."""
        self.info(f"{'='*50}")
        self.info(f"Starting {phase}")
        self.info(f"{'='*50}")
    
    def log_phase_end(self, phase: str):
        """Log the end of a training phase."""
        self.info(f"{'='*50}")
        self.info(f"Completed {phase}")
        self.info(f"{'='*50}")


# Global logger instances
_loggers = {}


def get_logger(name: str, log_dir: str = "./logs", level: int = logging.INFO) -> DynamoLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
    
    Returns:
        DynamoLogger instance
    """
    if name not in _loggers:
        _loggers[name] = DynamoLogger(name, log_dir, level)
    return _loggers[name]


def setup_logging(log_dir: str = "./logs", level: int = logging.INFO):
    """
    Setup global logging configuration.
    
    Args:
        log_dir: Directory to save log files
        level: Logging level
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "dynamo.log"))
        ]
    )


class TrainingLogger:
    """Specialized logger for training metrics and progress."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.logger = get_logger("training", log_dir)
        self.step_count = 0
        self.epoch_count = 0
    
    def log_step(self, loss: float, lr: float, **kwargs):
        """Log a training step."""
        self.step_count += 1
        self.logger.log_training_step(self.step_count, loss, lr, kwargs)
    
    def log_epoch_start(self, epoch: int):
        """Log epoch start."""
        self.epoch_count = epoch
        self.logger.info(f"Starting Epoch {epoch}")
    
    def log_epoch_end(self, epoch: int, avg_loss: float, **kwargs):
        """Log epoch end."""
        message = f"Epoch {epoch} completed: Avg Loss={avg_loss:.4f}"
        if kwargs:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in kwargs.items()])
            message += f", {metrics_str}"
        self.logger.info(message)
    
    def log_validation(self, results: dict):
        """Log validation results."""
        self.logger.log_evaluation_results("Validation", results)

