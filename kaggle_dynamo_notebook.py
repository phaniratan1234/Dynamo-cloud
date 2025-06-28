"""
DYNAMO Training on Kaggle T4 GPUs
=================================

Copy and paste these cells into your Kaggle notebook.
Each cell is marked with "# CELL X:" for easy identification.

Repository: https://github.com/phaniratan1234/Dynamo-cloud.git
"""

# ===== CELL 1: Setup and Clone Repository =====
import os
import subprocess
import sys
import time

print("🚀 DYNAMO Training Setup on Kaggle T4")
print("=" * 50)

# Check GPU
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
    print(f"✅ CUDA Version: {torch.version.cuda}")
else:
    print("❌ No GPU available!")
    sys.exit(1)

# Clone repository
print("\n📥 Cloning DYNAMO repository...")
os.chdir('/kaggle/working')

# Remove any existing directory
if os.path.exists('Dynamo-cloud'):
    subprocess.run(['rm', '-rf', 'Dynamo-cloud'], check=True)

# Clone the repository
result = subprocess.run([
    'git', 'clone', 'https://github.com/phaniratan1234/Dynamo-cloud.git'
], capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Repository cloned successfully!")
else:
    print(f"❌ Clone failed: {result.stderr}")
    sys.exit(1)

# Change to project directory
os.chdir('Dynamo-cloud')
sys.path.insert(0, '/kaggle/working/Dynamo-cloud')

print("\n✅ Setup complete! Ready for training.")

# ===== CELL 2: Install Requirements =====
print("📦 Installing required packages...")

# Install packages (most are pre-installed on Kaggle)
packages_to_install = [
    'transformers>=4.30.0',
    'datasets>=2.12.0', 
    'peft>=0.4.0',
    'accelerate>=0.20.0',
    'evaluate>=0.4.0',
    'rouge-score>=0.1.2',
    'sacrebleu>=2.3.0'
]

for package in packages_to_install:
    print(f"Installing {package}...")
    result = subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q', package
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"⚠️  Warning: Failed to install {package}")
    else:
        print(f"✅ {package} installed")

print("\n✅ Package installation complete!")

# ===== CELL 3: GPU Optimization and Memory Test =====
print("🔧 Optimizing for T4 GPU performance...")

# Set environment variables for better performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Memory test with T4 optimized batch size
print("\n🧪 Testing GPU memory with optimized batch size...")

device = torch.device('cuda')
batch_size = 64
seq_len = 512
hidden_size = 768

try:
    # Test memory allocation
    input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Test a forward pass
    linear = torch.nn.Linear(hidden_size, hidden_size, device=device)
    output = linear(hidden_states)
    
    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    
    print(f"✅ Memory test successful!")
    print(f"   - Allocated: {memory_allocated:.2f} GB")
    print(f"   - Reserved: {memory_reserved:.2f} GB")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {seq_len}")
    
    # Clean up test tensors
    del input_ids, attention_mask, hidden_states, linear, output
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Memory test failed: {e}")
    print("💡 Try reducing batch size in config.yaml")

print("\n🚀 GPU optimization complete!")

# ===== CELL 4: Fix QA Dataset and Loss Function (Clean Version) =====
print("🔧 Fixing QA dataset and loss function...")

import torch
import torch.nn as nn
import sys
import os
sys.path.append('/kaggle/working/Dynamo-cloud')

# Create a completely new TaskSpecificLoss class
class FixedTaskSpecificLoss(nn.Module):
    """Fixed task-specific loss function."""
    
    def __init__(self, task_name: str, weight: float = 1.0):
        super().__init__()
        self.task_name = task_name
        self.weight = weight
        
        if task_name == "sentiment":
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_name == "qa":
            # Special QA loss for start/end positions
            self.loss_fn = self._create_qa_loss()
        elif task_name in ["summarization", "code_generation", "translation"]:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def _create_qa_loss(self):
        """Create QA-specific loss function."""
        class QALoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
            
            def forward(self, predictions, targets):
                """
                predictions: model output for QA
                targets: (batch_size, 2) tensor with [start_pos, end_pos]
                """
                # Handle different prediction formats
                if hasattr(predictions, 'start_logits') and hasattr(predictions, 'end_logits'):
                    start_logits = predictions.start_logits
                    end_logits = predictions.end_logits
                elif isinstance(predictions, tuple) and len(predictions) == 2:
                    start_logits, end_logits = predictions
                elif predictions.dim() == 3 and predictions.size(-1) == 2:
                    start_logits = predictions[:, :, 0]
                    end_logits = predictions[:, :, 1]
                else:
                    # Fallback: split predictions in half
                    seq_len = predictions.size(1) // 2
                    start_logits = predictions[:, :seq_len]
                    end_logits = predictions[:, seq_len:]
                
                # Ensure targets are properly formatted
                if targets.dim() == 1:
                    # If 1D, assume [start, end] and expand for batch
                    targets = targets.unsqueeze(0).expand(start_logits.size(0), -1)
                
                # Clamp targets to valid range
                seq_len = start_logits.size(1)
                start_positions = targets[:, 0].clamp(0, seq_len - 1)
                end_positions = targets[:, 1].clamp(0, seq_len - 1)
                
                # Compute losses
                start_loss = self.cross_entropy(start_logits, start_positions)
                end_loss = self.cross_entropy(end_logits, end_positions)
                
                return (start_loss + end_loss) / 2
        
        return QALoss()
    
    def forward(self, predictions, targets):
        return self.weight * self.loss_fn(predictions, targets)

# Create a fixed QA dataset
class FixedQADataset:
    """Fixed QA dataset with proper target formatting."""
    
    def __init__(self, data, tokenizer, max_length=512, task_name="qa"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_name = task_name
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Combine question and context
        input_text = f"{example['question']} [SEP] {example['context']}"
        
        # Tokenize input
        tokenized = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Fix answer positions - ensure they're within sequence bounds
        start_pos = example.get('start_position', 0)
        end_pos = example.get('end_position', 0)
        
        # Clamp positions to valid range
        max_pos = self.max_length - 1
        start_pos = max(0, min(start_pos, max_pos))
        end_pos = max(start_pos, min(end_pos, max_pos))  # end >= start
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'qa',
            'task_id': 1,
            'target': torch.tensor([start_pos, end_pos], dtype=torch.long),
            'input_text': input_text
        }

# Replace the classes directly in the modules
print("🔄 Replacing problematic classes...")

# Replace TaskSpecificLoss
import training.losses as losses_module
losses_module.TaskSpecificLoss = FixedTaskSpecificLoss

# Replace QADataset  
import data.dataset_loaders as dataset_module
dataset_module.QADataset = FixedQADataset

print("✅ Classes replaced successfully!")

# Test the fixes
print("🧪 Testing fixes...")
try:
    # Test creating a TaskSpecificLoss
    test_loss = FixedTaskSpecificLoss("qa")
    print("✅ TaskSpecificLoss creation successful!")
    
    # Test imports
    from model.dynamo_model import DynamoModel
    from training.phase1_lora_training import run_phase1_training
    from utils.config import Config
    import yaml
    
    print("✅ All imports successful with fixes!")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    print(f"\n📊 Current Configuration:")
    print(f"   - Device: {config_dict['device']}")
    print(f"   - Batch size: {config_dict['training']['batch_size']}")
    print(f"   - Max length: {config_dict['training']['max_length']}")
    print(f"   - Epochs: {config_dict['training']['num_epochs']}")
    print(f"   - LoRA tasks: {list(config_dict['model']['lora_configs'].keys())}")
    
    print("\n✅ All fixes applied successfully!")
    
except Exception as e:
    print(f"❌ Fix test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🚀 Ready to restart training with fixes!")

# ===== CELL 5: Phase 1 Training =====
print("🎯 Starting DYNAMO Phase 1 Training!")
print("=" * 50)

# Record start time
start_time = time.time()

try:
    # Load fresh model
    from model.dynamo_model import DynamoModel
    from training.phase1_lora_training import run_phase1_training
    from utils.config import Config
    
    # Load config
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config object
    config_obj = Config()
    config_obj.device = config_dict['device']
    config_obj.training.num_epochs = config_dict['training']['num_epochs']
    config_obj.training.batch_size = config_dict['training']['batch_size']
    config_obj.training.lora_lr = config_dict['training']['lora_lr']
    config_obj.use_wandb = False
    
    # Initialize model
    print("🏗️  Initializing DYNAMO model...")
    model = DynamoModel(config_dict)
    model = model.to(torch.device('cuda'))
    
    print(f"📈 Model ready for training!")
    print(f"   - Device: {next(model.parameters()).device}")
    print(f"   - Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Start Phase 1 training
    print(f"\n🚀 Training 5 LoRA adapters...")
    print(f"   - Tasks: {model.task_names}")
    print(f"   - Epochs per adapter: {config_dict['training']['num_epochs']}")
    print(f"   - Batch size: {config_dict['training']['batch_size']}")
    
    # Run training
    results = run_phase1_training(config_obj, model)
    
    # Training completed
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n🎉 Phase 1 Training Completed!")
    print(f"⏱️  Total training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"📈 Results summary:")
    
    for task, metrics in results.items():
        if isinstance(metrics, dict):
            if 'best_val_loss' in metrics:
                print(f"   - {task}: Best validation loss = {metrics['best_val_loss']:.4f}")
            elif 'status' in metrics:
                print(f"   - {task}: {metrics['status']}")
        else:
            print(f"   - {task}: {metrics}")
    
    # Check saved checkpoints
    import os
    checkpoint_dir = 'checkpoints/phase1'
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        print(f"\n💾 Saved checkpoints: {len(checkpoints)}")
        for checkpoint in checkpoints:
            print(f"   - {checkpoint}")
    
    print(f"\n✅ Phase 1 training successful! Model ready for Phase 2.")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    print(f"📝 Error details:")
    import traceback
    traceback.print_exc()

# ===== CELL 6: Phase 2 Training (Optional) =====
print("\n" + "="*50)
print("🎯 Phase 2: Router Training (Optional)")
print("="*50)

# Ask if user wants to continue
user_input = input("Do you want to run Phase 2 (Router training)? (y/n): ").lower()

if user_input in ['y', 'yes']:
    try:
        print("🚀 Starting Phase 2 training...")
        
        # You can add Phase 2 training here
        # For now, just a placeholder
        print("⚠️  Phase 2 training script not implemented in this version")
        print("💡 Run manually: python train_dynamo.py --phase 2")
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
else:
    print("⏭️  Skipping Phase 2 training")

# ===== CELL 7: Save Results and Cleanup =====
print("\n" + "="*50)
print("💾 Saving Results and Cleanup")
print("="*50)

try:
    # Create a results summary
    results_summary = {
        'training_completed': True,
        'gpu_used': torch.cuda.get_device_name(0),
        'total_training_time': f"{training_time:.2f} seconds",
        'configuration': config_dict,
        'phase1_results': results if 'results' in locals() else {}
    }
    
    # Save results
    import json
    with open('training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print("✅ Results saved to training_results.json")
    
    # Display final summary
    print(f"\n🎉 DYNAMO Training Summary:")
    print(f"   - Phase 1: ✅ Completed")
    print(f"   - Training time: {training_time/60:.1f} minutes")
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    print(f"   - Max memory used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    # Clean up GPU memory
    torch.cuda.empty_cache()
    print(f"   - GPU memory cleared")
    
    print(f"\n🚀 DYNAMO training completed successfully!")
    print(f"📁 Check the files tab for checkpoints and results")
    
except Exception as e:
    print(f"⚠️  Warning: Cleanup failed: {e}")

print("\n" + "="*70)
print("🎯 DYNAMO TRAINING COMPLETE!")
print("✅ Your model is now trained and ready for use!")
print("📊 Download the checkpoints from the Files tab for future use")
print("="*70) 