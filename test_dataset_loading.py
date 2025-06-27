#!/usr/bin/env python3
"""
Test script to verify DYNAMO dataset loading functionality.
Tests the actual data loading pipeline that DYNAMO will use for training.
"""

import sys
import os
import torch
from transformers import RobertaTokenizer

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_dataset_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing module imports...")
    
    try:
        from utils.config import get_config
        print("   ✅ Config module imported")
    except Exception as e:
        print(f"   ❌ Config import failed: {e}")
        return False
    
    try:
        from data.dataset_loaders import DatasetLoader
        print("   ✅ DatasetLoader imported")
    except Exception as e:
        print(f"   ❌ DatasetLoader import failed: {e}")
        return False
    
    try:
        from data.mixed_task_dataset import create_mixed_task_dataset
        print("   ✅ Mixed task dataset imported")
    except Exception as e:
        print(f"   ❌ Mixed task dataset import failed: {e}")
        return False
    
    print("   ✅ All imports successful!")
    return True

def test_individual_datasets():
    """Test loading each individual dataset."""
    print("\n🔍 Testing individual dataset loading...")
    
    # Import after path is set
    from utils.config import get_config
    from data.dataset_loaders import DatasetLoader
    
    # Get configuration
    config = get_config()
    loader = DatasetLoader(config.__dict__)
    
    datasets = {
        'sentiment': loader.load_sentiment_data,
        'qa': loader.load_qa_data, 
        'summarization': loader.load_summarization_data,
        'code_generation': loader.load_code_generation_data,
        'translation': loader.load_translation_data
    }
    
    results = {}
    
    for task_name, load_func in datasets.items():
        print(f"\n📊 Testing {task_name.upper()}:")
        
        try:
            # Test loading train split
            train_data = load_func('train')
            print(f"   ✅ Train: {len(train_data):,} examples loaded")
            
            # Test sample data structure
            if train_data:
                sample = train_data[0]
                print(f"   📝 Sample keys: {list(sample.keys())}")
                
                # Show relevant content based on task
                if task_name == 'sentiment':
                    print(f"   📄 Text: {sample['sentence'][:80]}...")
                    print(f"   🏷️  Label: {sample['label']}")
                elif task_name == 'qa':
                    print(f"   ❓ Question: {sample['question'][:60]}...")
                    print(f"   📄 Answer: {sample['answer']}")
                elif task_name == 'summarization':
                    print(f"   📄 Document: {sample['document'][:60]}...")
                    print(f"   📝 Summary: {sample['summary'][:60]}...")
                elif task_name == 'code_generation':
                    print(f"   📄 Description: {sample['description'][:60]}...")
                    print(f"   💻 Code: {sample['code'][:60]}...")
                elif task_name == 'translation':
                    print(f"   🌍 Source: {sample['source'][:60]}...")
                    print(f"   🎯 Target: {sample['target'][:60]}...")
            
            results[task_name] = len(train_data)
            
        except Exception as e:
            print(f"   ❌ Failed to load {task_name}: {e}")
            results[task_name] = 0
    
    return results

def test_dataset_classes():
    """Test DYNAMO dataset classes and tokenization."""
    print("\n🔍 Testing DYNAMO dataset classes...")
    
    from utils.config import get_config
    from data.dataset_loaders import DatasetLoader, SentimentDataset, QADataset
    from transformers import RobertaTokenizer
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    print("   ✅ RoBERTa tokenizer loaded")
    
    # Test sentiment dataset
    try:
        config = get_config()
        loader = DatasetLoader(config.__dict__)
        
        sentiment_data = loader.load_sentiment_data('train')[:100]  # Test with 100 samples
        sentiment_dataset = SentimentDataset(
            sentiment_data, tokenizer, max_length=512, task_name='sentiment'
        )
        
        # Test getting a sample
        sample = sentiment_dataset[0]
        print(f"   ✅ Sentiment dataset: {len(sentiment_dataset)} samples")
        print(f"   📝 Sample shape - input_ids: {sample['input_ids'].shape}")
        print(f"   📝 Sample keys: {list(sample.keys())}")
        
    except Exception as e:
        print(f"   ❌ Sentiment dataset test failed: {e}")
    
    # Test QA dataset
    try:
        qa_data = loader.load_qa_data('train')[:100]  # Test with 100 samples
        qa_dataset = QADataset(
            qa_data, tokenizer, max_length=512, task_name='qa'
        )
        
        sample = qa_dataset[0]
        print(f"   ✅ QA dataset: {len(qa_dataset)} samples")
        print(f"   📝 Sample shape - input_ids: {sample['input_ids'].shape}")
        print(f"   📝 Target shape: {sample['target'].shape}")
        
    except Exception as e:
        print(f"   ❌ QA dataset test failed: {e}")

def test_dataloaders():
    """Test creating PyTorch DataLoaders."""
    print("\n🔍 Testing PyTorch DataLoaders...")
    
    from utils.config import get_config
    from data.dataset_loaders import DatasetLoader
    from torch.utils.data import DataLoader
    
    try:
        config = get_config()
        loader = DatasetLoader(config.__dict__)
        
        # Create datasets
        datasets = loader.create_datasets('train')
        print(f"   ✅ Created {len(datasets)} datasets")
        
        # Create dataloaders
        dataloaders = loader.create_dataloaders(
            datasets, batch_size=4, shuffle=True, num_workers=0
        )
        print(f"   ✅ Created {len(dataloaders)} dataloaders")
        
        # Test each dataloader
        for task_name, dataloader in dataloaders.items():
            try:
                batch = next(iter(dataloader))
                print(f"   📊 {task_name}: batch_size={batch['input_ids'].size(0)}")
                print(f"      Input shape: {batch['input_ids'].shape}")
                print(f"      Attention mask: {batch['attention_mask'].shape}")
                
            except Exception as e:
                print(f"   ❌ {task_name} dataloader failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ DataLoader test failed: {e}")
        return False

def test_mixed_task_dataset():
    """Test mixed task dataset creation."""
    print("\n🔍 Testing mixed task dataset...")
    
    try:
        from utils.config import get_config
        from data.dataset_loaders import DatasetLoader
        from data.mixed_task_dataset import create_mixed_task_dataset
        
        config = get_config()
        loader = DatasetLoader(config.__dict__)
        
        # Create single task datasets (small subset for testing)
        print("   📊 Creating single task datasets...")
        single_task_datasets = {}
        
        for task in ['sentiment', 'qa']:  # Test with 2 tasks for speed
            if task == 'sentiment':
                data = loader.load_sentiment_data('train')[:200]
                from data.dataset_loaders import SentimentDataset
                from transformers import RobertaTokenizer
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                dataset = SentimentDataset(data, tokenizer, task_name=task)
            elif task == 'qa':
                data = loader.load_qa_data('train')[:200]
                from data.dataset_loaders import QADataset
                dataset = QADataset(data, tokenizer, task_name=task)
            
            single_task_datasets[task] = dataset
        
        print(f"   ✅ Created {len(single_task_datasets)} single task datasets")
        
        # Create mixed task dataset
        mixed_dataset = create_mixed_task_dataset(
            single_task_datasets, config.__dict__, num_examples=50
        )
        
        print(f"   ✅ Mixed task dataset: {len(mixed_dataset)} examples")
        
        # Test a sample
        sample = mixed_dataset[0]
        print(f"   📝 Mixed sample keys: {list(sample.keys())}")
        print(f"   🎯 Tasks: {sample['tasks']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Mixed task dataset test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 DYNAMO Dataset Loading Test")
    print("="*60)
    
    # Test imports
    if not test_dataset_imports():
        print("❌ Import tests failed. Cannot proceed.")
        return
    
    # Test individual datasets
    results = test_individual_datasets()
    
    # Test dataset classes
    test_dataset_classes()
    
    # Test dataloaders
    dataloader_success = test_dataloaders()
    
    # Test mixed task dataset
    mixed_task_success = test_mixed_task_dataset()
    
    # Summary
    print("\n" + "="*60)
    print("📋 DATASET LOADING TEST SUMMARY")
    print("="*60)
    
    total_loaded = sum(1 for count in results.values() if count > 0)
    total_examples = sum(results.values())
    
    print(f"✅ Datasets loaded: {total_loaded}/5")
    print(f"📊 Total examples: {total_examples:,}")
    print(f"🔧 DataLoaders: {'✅ Working' if dataloader_success else '❌ Failed'}")
    print(f"🔀 Mixed Tasks: {'✅ Working' if mixed_task_success else '❌ Failed'}")
    
    if total_loaded == 5 and dataloader_success and mixed_task_success:
        print("\n🎉 ALL TESTS PASSED! DYNAMO is ready for training!")
    elif total_loaded >= 3:
        print("\n⚠️  Most tests passed. DYNAMO can proceed with limited functionality.")
    else:
        print("\n❌ Tests failed. Check dataset setup and try again.")

if __name__ == "__main__":
    main() 