#!/usr/bin/env python3
"""
Test script to verify Phase 2 readiness.
Checks Phase 1 completion and validates adapter checkpoints.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from utils.checkpoint_utils import check_phase1_completion, print_phase1_status
from utils.config import get_config

def main():
    """Main test function."""
    print("🧪 Testing Phase 2 Readiness")
    print("=" * 50)
    
    try:
        # Load config to get checkpoint directory
        config = get_config()
        checkpoint_dir = config.checkpoint_dir
        
        # Check Phase 1 completion
        is_complete, missing_files, status_info = check_phase1_completion(checkpoint_dir)
        
        print_phase1_status(checkpoint_dir)
        
        if is_complete:
            print("\n✅ PHASE 2 READY!")
            print("You can now run: python train_optimized.py --phase 2")
            return 0
        else:
            print(f"\n⏳ Phase 1 still incomplete")
            print(f"Missing {len(missing_files)} files")
            print("Continue Phase 1 training...")
            return 1
            
    except Exception as e:
        print(f"❌ Error checking readiness: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 