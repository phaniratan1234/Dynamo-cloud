#!/usr/bin/env python3
"""
Optimized DYNAMO Training Script with GPU Performance Monitoring
Includes mixed precision, gradient accumulation, and comprehensive GPU optimizations.
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
from utils.checkpoint_utils import validate_checkpoint_transition, print_phase1_status
from gpu_monitor import GPUMonitor, check_gpu_optimization

logger = get_logger(__name__)


class SimpleConfig:
    """Simple configuration class that's pickle-able."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)
    
    def __getstate__(self):
        """Support pickling."""
        return self.__dict__
    
    def __setstate__(self, state):
        """Support unpickling."""
        self.__dict__.update(state)


def setup_gpu_optimizations():
    """Setup comprehensive GPU optimizations."""
    print("🚀 Setting up GPU optimizations...")
    
    if torch.cuda.is_available():
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Memory optimizations
        torch.cuda.empty_cache()
        
        # Set memory fraction if needed (uncomment for memory issues)
        # torch.cuda.set_per_process_memory_fraction(0.9)
        
        print(f"✅ GPU optimizations enabled for {torch.cuda.get_device_name(0)}")
        print(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   - cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"   - Memory cleared: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
        
        return "cuda"
    else:
        print("⚠️  CUDA not available, using CPU")
        # CPU optimizations
        torch.set_num_threads(os.cpu_count())
        return "cpu"

def initialize_optimized_model(config_dict: dict, device: str) -> DynamoModel:
    """Initialize DYNAMO model with optimizations."""
    print("🏗️  Initializing optimized DYNAMO model...")
    
    # Update device in config
    config_dict['device'] = device
    
    # Initialize model
    model = DynamoModel(config_dict)
    model = model.to(torch.device(device))
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📈 Model Statistics:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - LoRA adapters: {len(model.task_names)}")
    print(f"   - Tasks: {model.task_names}")
    
    # Memory check
    if torch.cuda.is_available():
        model_memory = torch.cuda.memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   - GPU memory usage: {model_memory:.1f}GB / {total_memory:.1f}GB ({model_memory/total_memory*100:.1f}%)")
    
    return model

def load_optimized_config(config_path: str) -> tuple:
    """Load and optimize configuration."""
    print(f"📁 Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply GPU-specific optimizations to config
    if torch.cuda.is_available():
        # Ensure GPU optimizations are enabled in config
        training_config = config_dict.get('training', {})
        training_config['use_mixed_precision'] = True
        training_config['dataloader_num_workers'] = min(8, torch.get_num_threads())
        training_config['pin_memory'] = True
        training_config['persistent_workers'] = True
        
        print("✅ GPU-optimized configuration applied:")
        print(f"   - Mixed precision: {training_config['use_mixed_precision']}")
        print(f"   - DataLoader workers: {training_config['dataloader_num_workers']}")
        print(f"   - Pin memory: {training_config['pin_memory']}")
        print(f"   - Gradient accumulation: {training_config.get('gradient_accumulation_steps', 1)}")
    
    # Create a simple Config-like object using the module-level SimpleConfig class
    config_obj = SimpleConfig(config_dict)
    
    return config_dict, config_obj

def run_with_monitoring(phase: int, config_dict: dict, config_obj: Config, model: DynamoModel, resume: bool = True):
    """Run training phase with GPU monitoring."""
    print(f"\n🚀 Starting Phase {phase} with GPU monitoring")
    print("=" * 60)
    
    # Start GPU monitoring
    monitor = GPUMonitor(log_interval=10)  # Log every 10 seconds
    monitor.start_monitoring()
    
    try:
        # Run the appropriate training phase
        if phase == 1:
            results = run_phase1_training(config_obj, model, resume=resume)
        elif phase == 2:
            results = run_phase2_training(config_obj, model, resume=resume)
        elif phase == 3:
            results = run_phase3_training(config_obj, model, resume=resume)
        else:
            raise ValueError(f"Invalid phase: {phase}")
        
        print(f"\n✅ Phase {phase} completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Phase {phase} failed: {e}")
        results = None
        
    finally:
        # Stop monitoring and get summary
        monitor.stop_monitoring()
        time.sleep(2)  # Allow final metrics to be collected
        
        # Generate performance report
        summary = monitor.get_summary()
        print("\n📊 GPU Performance Summary:")
        print("=" * 50)
        
        if "error" not in summary:
            print(f"Monitoring duration: {summary['monitoring_duration_minutes']:.1f} minutes")
            print(f"Peak GPU memory: {summary['gpu_memory']['peak_usage_gb']:.1f}GB ({summary['gpu_memory']['peak_utilization_percent']:.1f}%)")
            print(f"Average GPU utilization: {summary['gpu_utilization']['avg_percent']:.1f}%")
            
            print("\n🔧 Recommendations:")
            for rec in summary['recommendations']:
                print(rec)
            
            # Save monitoring results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            monitor.save_plot(f"gpu_monitoring_phase{phase}_{timestamp}.png")
            monitor.save_metrics(f"gpu_metrics_phase{phase}_{timestamp}.json")
        else:
            print(summary["error"])
    
    return results

def main():
    """Main optimized training function."""
    parser = argparse.ArgumentParser(description="Optimized DYNAMO Training Script")
    parser.add_argument(
        "--phase", 
        type=int, 
        choices=[1, 2, 3], 
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
        "--monitor-only", 
        action="store_true",
        help="Only run GPU monitoring check without training"
    )
    parser.add_argument(
        "--status",
        action="store_true", 
        help="Check training status and phase readiness"
    )
    
    args = parser.parse_args()
    
    # Check if phase is required (not needed for monitor-only or status)
    if not args.monitor_only and not args.status and args.phase is None:
        parser.error("--phase is required when not using --monitor-only or --status")
    
    print("🚀 DYNAMO Optimized Training Script")
    print("=" * 60)
    if args.monitor_only:
        print("🔍 Mode: GPU monitoring only")
    elif args.status:
        print("📊 Mode: Status check")
    else:
        print(f"📊 Phase: {args.phase}")
        print(f"🔄 Resume: {not args.no_resume}")
    print(f"📁 Config: {args.config}")
    print()
    
    # GPU optimization check
    gpu_available = check_gpu_optimization()
    
    if args.monitor_only:
        print("🔍 Running monitoring demo...")
        if gpu_available:
            monitor = GPUMonitor(log_interval=2)
            monitor.start_monitoring()
            time.sleep(30)
            monitor.stop_monitoring()
            summary = monitor.get_summary()
            print("\n📊 Demo Summary:")
            for rec in summary.get('recommendations', []):
                print(rec)
        else:
            print("⚠️  GPU monitoring skipped - no CUDA device available")
            print("ℹ️  On T4 GPU, you would see:")
            print("   - Real-time GPU memory usage")
            print("   - GPU utilization percentages")
            print("   - Performance recommendations")
        return 0
    
    if args.status:
        print("📊 Checking training status...")
        try:
            config_dict, config_obj = load_optimized_config(args.config)
            print_phase1_status(config_obj.checkpoint_dir)
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            return 1
        return 0
    
    if not gpu_available:
        print("⚠️  No GPU available - running in CPU development mode")
        print("ℹ️  For production T4 training, use a GPU-enabled environment")
    
    # Setup optimizations
    device = setup_gpu_optimizations()
    print()
    
    # Load optimized configuration
    try:
        config_dict, config_obj = load_optimized_config(args.config)
        print(f"✅ Optimized configuration loaded")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1
    
    # Initialize optimized model
    try:
        model = initialize_optimized_model(config_dict, device)
        print("✅ Optimized model initialized")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return 1
    
    print()
    
    # Validate phase transition before starting
    print(f"\n🔍 Validating Phase {args.phase} readiness...")
    if not validate_checkpoint_transition(0, args.phase, config_obj.checkpoint_dir):
        print(f"\n❌ Cannot start Phase {args.phase} - prerequisites not met")
        if args.phase == 2:
            print("\n📊 Checking Phase 1 status...")
            print_phase1_status(config_obj.checkpoint_dir)
        return 1
    
    print(f"✅ Phase {args.phase} prerequisites satisfied")
    
    # Run training with monitoring
    try:
        results = run_with_monitoring(
            args.phase, 
            config_dict, 
            config_obj, 
            model, 
            resume=not args.no_resume
        )
        
        if results:
            print("\n🎉 Training completed successfully!")
            return 0
        else:
            print("\n❌ Training failed")
            return 1
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 