"""
Checkpoint utilities for DYNAMO training phases.
Handles checkpoint validation, status checking, and transitions between phases.
"""

import os
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


def check_phase1_adapters_for_phase2(checkpoint_dir: str) -> Tuple[bool, List[str]]:
    """
    Check if Phase 1 adapter checkpoints are available for Phase 2 training.
    Phase 2 only needs the trained adapters, not the complete Phase 1 model.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Tuple of (adapters_ready, missing_adapter_files)
    """
    phase1_dir = os.path.join(checkpoint_dir, "phase1")
    
    # Required adapter files for Phase 2 (router training)
    required_adapters = [
        "sentiment_adapter.pt",
        "qa_adapter.pt",
        "summarization_adapter.pt", 
        "code_generation_adapter.pt",
        "translation_adapter.pt"
    ]
    
    missing_adapters = []
    
    # Check each required adapter file
    for filename in required_adapters:
        filepath = os.path.join(phase1_dir, filename)
        if not os.path.exists(filepath):
            missing_adapters.append(filename)
    
    adapters_ready = len(missing_adapters) == 0
    
    return adapters_ready, missing_adapters


def check_phase1_completion(checkpoint_dir: str) -> Tuple[bool, List[str], Dict[str, any]]:
    """
    Check if Phase 1 training is completed and ready for Phase 2.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Tuple of (is_complete, missing_files, status_info)
    """
    phase1_dir = os.path.join(checkpoint_dir, "phase1")
    
    # Required checkpoint files
    required_files = [
        "sentiment_adapter.pt",
        "qa_adapter.pt",
        "summarization_adapter.pt", 
        "code_generation_adapter.pt",
        "translation_adapter.pt",
        "phase1_model.pt",
        "training_metrics.pt"
    ]
    
    missing_files = []
    existing_files = []
    status_info = {}
    
    # Check each required file
    for filename in required_files:
        filepath = os.path.join(phase1_dir, filename)
        if os.path.exists(filepath):
            existing_files.append(filename)
            
            # Get file info
            try:
                if filename.endswith('_adapter.pt'):
                    checkpoint = torch.load(filepath, map_location='cpu')
                    task_name = filename.replace('_adapter.pt', '')
                    status_info[task_name] = {
                        'epoch': checkpoint.get('epoch', 'unknown'),
                        'loss': checkpoint.get('loss', 'unknown'),
                        'file_size_mb': os.path.getsize(filepath) / (1024*1024)
                    }
                elif filename == 'training_metrics.pt':
                    metrics = torch.load(filepath, map_location='cpu')
                    status_info['training_metrics'] = metrics
                    
            except Exception as e:
                logger.warning(f"Could not read {filename}: {e}")
                
        else:
            missing_files.append(filename)
    
    is_complete = len(missing_files) == 0
    
    return is_complete, missing_files, status_info


def print_phase1_status(checkpoint_dir: str):
    """Print a detailed status report of Phase 1 training."""
    is_complete, missing_files, status_info = check_phase1_completion(checkpoint_dir)
    
    print("\n" + "="*60)
    print("📊 Phase 1 Training Status Report")
    print("="*60)
    
    if is_complete:
        print("✅ Phase 1 COMPLETED - Ready for Phase 2!")
        print("\n📂 Adapter Checkpoints:")
        
        for task_name in ['sentiment', 'qa', 'summarization', 'code_generation', 'translation']:
            if task_name in status_info:
                info = status_info[task_name]
                print(f"  ✅ {task_name:15} | Epoch: {info['epoch']:3} | Loss: {info['loss']:.4f} | Size: {info['file_size_mb']:.1f}MB")
        
        if 'training_metrics' in status_info:
            metrics = status_info['training_metrics']
            print(f"\n📈 Training Summary:")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  - {key}: {value}")
                        
        print("\n🚀 Ready to start Phase 2:")
        print("  python train_optimized.py --phase 2")
        
    else:
        print("⏳ Phase 1 IN PROGRESS or NOT STARTED")
        
        # Get existing files from the check_phase1_completion result
        phase1_dir = os.path.join(checkpoint_dir, "phase1")
        required_files = [
            "sentiment_adapter.pt", "qa_adapter.pt", "summarization_adapter.pt", 
            "code_generation_adapter.pt", "translation_adapter.pt",
            "phase1_model.pt", "training_metrics.pt"
        ]
        existing_files = [f for f in required_files if os.path.exists(os.path.join(phase1_dir, f))]
        
        print(f"\n❌ Missing files ({len(missing_files)}/{len(missing_files) + len(existing_files)}):")
        for filename in missing_files:
            print(f"  - {filename}")
            
        if existing_files:
            print(f"\n✅ Existing files ({len(existing_files)}):")
            for filename in existing_files:
                print(f"  - {filename}")
                
        print("\n⏳ Continue Phase 1 training:")
        print("  python train_optimized.py --phase 1")
        
    print("="*60)


def check_phase2_completion(checkpoint_dir: str) -> Tuple[bool, List[str]]:
    """Check if Phase 2 training is completed."""
    phase2_dir = os.path.join(checkpoint_dir, "phase2")
    
    required_files = [
        "phase2_model.pt",
        "best_router.pt", 
        "training_metrics.pt"
    ]
    
    missing_files = []
    for filename in required_files:
        if not os.path.exists(os.path.join(phase2_dir, filename)):
            missing_files.append(filename)
    
    return len(missing_files) == 0, missing_files


def validate_checkpoint_transition(from_phase: int, to_phase: int, checkpoint_dir: str) -> bool:
    """
    Validate that transition between phases is possible.
    
    Args:
        from_phase: Current phase (1, 2, or 3)
        to_phase: Target phase (1, 2, or 3)
        checkpoint_dir: Checkpoint directory path
        
    Returns:
        True if transition is valid, False otherwise
    """
    if to_phase == 1:
        # Phase 1 can always start (trains from scratch)
        return True
        
    elif to_phase == 2:
        # Phase 2 requires Phase 1 adapters (not the complete Phase 1 model)
        adapters_ready, missing_adapters = check_phase1_adapters_for_phase2(checkpoint_dir)
        if not adapters_ready:
            logger.error(f"❌ Cannot start Phase 2: Missing trained adapters from Phase 1: {missing_adapters}")
            logger.error("Phase 2 router training requires trained adapters from Phase 1.")
            logger.error("Please complete Phase 1 training first by running:")
            logger.error("  python train_optimized.py --phase 1")
            return False
        
        logger.info("✅ Phase 1 adapters found - Phase 2 can proceed with router training")
        return True
        
    elif to_phase == 3:
        # Phase 3 requires Phase 2 completion
        phase1_complete, _, _ = check_phase1_completion(checkpoint_dir)
        phase2_complete, missing2 = check_phase2_completion(checkpoint_dir)
        
        if not phase1_complete:
            logger.error("❌ Cannot start Phase 3: Phase 1 incomplete")
            return False
            
        if not phase2_complete:
            logger.error(f"❌ Cannot start Phase 3: Phase 2 incomplete. Missing: {missing2}")
            return False
            
        return True
        
    else:
        logger.error(f"❌ Invalid phase: {to_phase}")
        return False


def clean_checkpoints(checkpoint_dir: str, phase: Optional[int] = None):
    """
    Clean checkpoint files for fresh training.
    
    Args:
        checkpoint_dir: Checkpoint directory
        phase: Specific phase to clean (None for all)
    """
    if phase is None:
        phases = [1, 2, 3]
    else:
        phases = [phase]
    
    for p in phases:
        phase_dir = os.path.join(checkpoint_dir, f"phase{p}")
        if os.path.exists(phase_dir):
            import shutil
            shutil.rmtree(phase_dir)
            logger.info(f"🧹 Cleaned Phase {p} checkpoints") 