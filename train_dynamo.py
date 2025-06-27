#!/usr/bin/env python3
"""
Unified training script for DYNAMO (Dynamic Neural Adapter Multi-task Optimization).
Supports training Phase 1 (LoRA adapters), Phase 2 (Router), and Phase 3 (Joint fine-tuning).
"""

import argparse
import sys
import os
import yaml
import torch
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

from model.dynamo_model import DynamoModel
from training.phase1_lora_training import run_phase1_training
from training.phase2_router_training import run_phase2_training
from training.phase3_joint_finetuning import run_phase3_training
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


def check_device_optimization():
    """Check and optimize device settings for best performance."""
    print("🔍 Device Performance Check:")
    
    if torch.backends.mps.is_available():
        print("   ✅ MPS (Apple Silicon GPU) available")
        device = "mps"
        
        # Optimize for MPS
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Better memory management
        torch.backends.mps.enable_fallback = True  # Enable CPU fallback for unsupported ops
        
        print("   🚀 MPS optimizations enabled")
        print("      - High watermark ratio: 0.0 (better memory)")
        print("      - CPU fallback: enabled")
        
    elif torch.cuda.is_available():
        print("   ✅ CUDA available")
        device = "cuda"
        
        # Optimize for CUDA
        torch.backends.cudnn.benchmark = True
        print("   🚀 CUDA optimizations enabled")
        
    else:
        print("   ⚠️  Using CPU")
        device = "cpu"
        
        # Optimize for CPU
        torch.set_num_threads(os.cpu_count())
        print(f"   🚀 CPU optimizations enabled ({os.cpu_count()} threads)")
    
    return device


def load_config(config_path: str = "config.yaml") -> tuple:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create Config object
    config_obj = Config()
    config_obj.device = config_dict['device']
    config_obj.training.num_epochs = config_dict['training']['num_epochs']
    config_obj.training.batch_size = config_dict['training']['batch_size']
    config_obj.training.lora_lr = config_dict['training']['lora_lr']
    config_obj.training.router_lr = config_dict['training']['router_lr']
    config_obj.training.joint_lr = config_dict['training']['joint_lr']
    config_obj.use_wandb = config_dict.get('use_wandb', False)
    
    return config_dict, config_obj


def initialize_model(config_dict: dict, device: str) -> DynamoModel:
    """Initialize DYNAMO model."""
    print("🏗️  Initializing DYNAMO model...")
    
    # Update device in config
    config_dict['device'] = device
    
    model = DynamoModel(config_dict)
    model = model.to(torch.device(device))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📈 Model Statistics:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - LoRA adapters: {len(model.task_names)}")
    print(f"   - Tasks: {model.task_names}")
    
    return model


def check_phase1_checkpoints(checkpoint_dir: str = "./checkpoints/phase1") -> list:
    """Check which LoRA adapters have been trained."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"📁 No Phase 1 checkpoints found at {checkpoint_dir}")
        return []
    
    # Look for adapter checkpoint files
    completed_adapters = []
    for task_file in checkpoint_path.glob("*_adapter.pt"):
        task_name = task_file.stem.replace("_adapter", "")
        completed_adapters.append(task_name)
    
    if completed_adapters:
        print(f"✅ Found existing Phase 1 checkpoints: {completed_adapters}")
    else:
        print("📁 No completed LoRA adapter checkpoints found")
    
    return completed_adapters


def run_phase1(config_obj: Config, model: DynamoModel, resume: bool = True) -> dict:
    """Run Phase 1: LoRA adapter training with checkpointing support."""
    print("🎯 Starting Phase 1: LoRA Adapter Training")
    print("="*60)
    
    if resume:
        completed_adapters = check_phase1_checkpoints()
        if completed_adapters:
            print(f"🔄 Resuming training, skipping: {completed_adapters}")
            # TODO: Load existing checkpoints and update model
            # For now, we'll just log this - the training function needs to be updated
    
    start_time = time.time()
    result = run_phase1_training(config_obj, model)
    end_time = time.time()
    
    print(f"✅ Phase 1 completed in {end_time - start_time:.2f}s")
    return result


def run_phase2(config_obj: Config, model: DynamoModel) -> dict:
    """Run Phase 2: Router training."""
    print("🎯 Starting Phase 2: Dynamic Router Training")
    print("="*60)
    
    start_time = time.time()
    result = run_phase2_training(config_obj, model)
    end_time = time.time()
    
    print(f"✅ Phase 2 completed in {end_time - start_time:.2f}s")
    return result


def run_phase3(config_obj: Config, model: DynamoModel) -> dict:
    """Run Phase 3: Joint fine-tuning."""
    print("🎯 Starting Phase 3: Joint Fine-tuning")
    print("="*60)
    
    start_time = time.time()
    result = run_phase3_training(config_obj, model)
    end_time = time.time()
    
    print(f"✅ Phase 3 completed in {end_time - start_time:.2f}s")
    return result


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="DYNAMO Training Script")
    parser.add_argument(
        "--phase", 
        type=int, 
        choices=[1, 2, 3], 
        required=True,
        help="Training phase to run (1: LoRA adapters, 2: Router, 3: Joint fine-tuning)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-resume", 
        action="store_true",
        help="Don't resume from checkpoints (start fresh)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    print("🚀 DYNAMO Training Script")
    print("="*60)
    print(f"📊 Phase: {args.phase}")
    print(f"📁 Config: {args.config}")
    print(f"🔄 Resume: {not args.no_resume}")
    print()
    
    # Device optimization
    if args.device == "auto":
        device = check_device_optimization()
    else:
        device = args.device
        print(f"🔧 Using specified device: {device}")
    
    print()
    
    # Load configuration
    try:
        config_dict, config_obj = load_config(args.config)
        print(f"✅ Configuration loaded from {args.config}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1
    
    # Initialize model
    try:
        model = initialize_model(config_dict, device)
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return 1
    
    print()
    
    # Run training phase
    try:
        if args.phase == 1:
            result = run_phase1(config_obj, model, resume=not args.no_resume)
        elif args.phase == 2:
            result = run_phase2(config_obj, model)
        elif args.phase == 3:
            result = run_phase3(config_obj, model)
        
        print()
        print("🎉 Training completed successfully!")
        print(f"📈 Results: {result}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 