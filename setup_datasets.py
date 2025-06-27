#!/usr/bin/env python3
"""
Dataset Setup Script for DYNAMO
Downloads and prepares all required datasets for training.
"""

import os
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer
import json
from tqdm import tqdm
import argparse

def setup_directories():
    """Create necessary data directories."""
    dirs = [
        './data/sst2',
        './data/squad', 
        './data/xsum',
        './data/code_generation',
        './data/translation',
        './data/processed'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Created data directories")

def download_sst2():
    """Download and process SST-2 sentiment dataset."""
    print("\n📊 Setting up SST-2 (Sentiment Analysis)...")
    
    try:
        # Load dataset
        dataset = load_dataset("sst2")
        
        # Save to local directory
        for split in ['train', 'validation', 'test']:
            split_name = 'validation' if split == 'validation' else split
            if split_name in dataset:
                data = []
                for example in tqdm(dataset[split_name], desc=f"Processing {split}"):
                    data.append({
                        'sentence': example['sentence'],
                        'label': example['label'],  # 0=negative, 1=positive
                        'task': 'sentiment'
                    })
                
                with open(f'./data/sst2/{split}.json', 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"   - {split}: {len(data)} examples")
        
        print("✅ SST-2 setup complete")
        return True
        
    except Exception as e:
        print(f"❌ SST-2 setup failed: {e}")
        return False

def download_squad():
    """Download and process SQuAD dataset."""
    print("\n📊 Setting up SQuAD (Question Answering)...")
    
    try:
        # Load dataset
        dataset = load_dataset("squad")
        
        # Save to local directory
        for split in ['train', 'validation']:
            data = []
            for example in tqdm(dataset[split], desc=f"Processing {split}"):
                # Handle answers (SQuAD format)
                if example['answers']['text']:
                    answer = example['answers']['text'][0]
                    start_pos = example['answers']['answer_start'][0]
                    end_pos = start_pos + len(answer)
                else:
                    answer = ""
                    start_pos = 0
                    end_pos = 0
                
                data.append({
                    'question': example['question'],
                    'context': example['context'],
                    'answer': answer,
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'id': example['id'],
                    'task': 'qa'
                })
            
            with open(f'./data/squad/{split}.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"   - {split}: {len(data)} examples")
        
        print("✅ SQuAD setup complete")
        return True
        
    except Exception as e:
        print(f"❌ SQuAD setup failed: {e}")
        return False

def download_xsum():
    """Download and process CNN/DailyMail dataset for summarization."""
    print("\n📊 Setting up CNN/DailyMail (Summarization)...")
    
    try:
        # Load CNN/DailyMail dataset (easier than XSum)
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        
        # Save to local directory  
        for split in ['train', 'validation', 'test']:
            data = []
            # Limit size for faster training
            max_examples = {'train': 20000, 'validation': 2000, 'test': 2000}
            
            split_data = dataset[split]
            if len(split_data) > max_examples[split]:
                split_data = split_data.select(range(max_examples[split]))
            
            for example in tqdm(split_data, desc=f"Processing {split}"):
                data.append({
                    'document': example['article'],
                    'summary': example['highlights'],
                    'id': example['id'],
                    'task': 'summarization'
                })
            
            with open(f'./data/xsum/{split}.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"   - {split}: {len(data)} examples")
        
        print("✅ CNN/DailyMail setup complete")
        return True
        
    except Exception as e:
        print(f"❌ CNN/DailyMail setup failed: {e}")
        # Try alternative summarization dataset
        try:
            print("   🔄 Trying alternative: Multi-News dataset...")
            dataset = load_dataset("multi_news")
            
            for split in ['train', 'validation', 'test']:
                data = []
                max_examples = {'train': 10000, 'validation': 1000, 'test': 1000}
                
                split_data = dataset[split]
                if len(split_data) > max_examples[split]:
                    split_data = split_data.select(range(max_examples[split]))
                
                for example in tqdm(split_data, desc=f"Processing {split}"):
                    data.append({
                        'document': example['document'],
                        'summary': example['summary'],
                        'id': str(len(data)),
                        'task': 'summarization'
                    })
                
                with open(f'./data/xsum/{split}.json', 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"   - {split}: {len(data)} examples")
            
            print("✅ Multi-News setup complete")
            return True
            
        except Exception as e2:
            print(f"❌ Alternative summarization setup failed: {e2}")
            print("   🔄 Using synthetic summarization data instead")
            return False

def download_code_generation():
    """Download and process CodeSearchNet for code generation."""
    print("\n📊 Setting up Code Generation...")
    
    try:
        # Load Python subset of CodeSearchNet
        dataset = load_dataset("code_search_net", "python")
        
        # Save to local directory
        for split in ['train', 'validation', 'test']:
            data = []
            # Limit size for faster training
            max_examples = {'train': 20000, 'validation': 2000, 'test': 2000}
            
            split_data = dataset[split]
            if len(split_data) > max_examples[split]:
                split_data = split_data.select(range(max_examples[split]))
            
            for example in tqdm(split_data, desc=f"Processing {split}"):
                # Filter out examples without docstring
                if example['func_documentation_string'] and example['func_code_string']:
                    data.append({
                        'description': example['func_documentation_string'],
                        'code': example['func_code_string'],
                        'func_name': example['func_name'],
                        'task': 'code_generation'
                    })
            
            with open(f'./data/code_generation/{split}.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"   - {split}: {len(data)} examples")
        
        print("✅ Code Generation setup complete")
        return True
        
    except Exception as e:
        print(f"❌ Code Generation setup failed: {e}")
        return False

def download_translation():
    """Download and process OPUS-100 for translation."""
    print("\n📊 Setting up Translation (OPUS-100 EN-DE)...")
    
    try:
        # Load OPUS-100 English-German dataset (smaller and easier than WMT14)
        dataset = load_dataset("opus100", "en-de")
        
        # Save to local directory
        for split in ['train', 'validation', 'test']:
            data = []
            # Reasonable sizes for training
            max_examples = {'train': 15000, 'validation': 1500, 'test': 1500}
            
            split_data = dataset[split]
            if len(split_data) > max_examples[split]:
                split_data = split_data.select(range(max_examples[split]))
            
            for example in tqdm(split_data, desc=f"Processing {split}"):
                data.append({
                    'source': example['translation']['en'],  # English
                    'target': example['translation']['de'],  # German
                    'language_pair': 'en-de',
                    'task': 'translation'
                })
            
            with open(f'./data/translation/{split}.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"   - {split}: {len(data)} examples")
        
        print("✅ OPUS-100 setup complete")
        return True
        
    except Exception as e:
        print(f"❌ OPUS-100 setup failed: {e}")
        # Try alternative smaller translation dataset
        try:
            print("   🔄 Trying alternative: WMT16 EN-DE (smaller subset)...")
            dataset = load_dataset("wmt16", "de-en")
            
            for split in ['train', 'validation', 'test']:
                if split not in dataset:
                    continue
                    
                data = []
                max_examples = {'train': 10000, 'validation': 1000, 'test': 1000}
                
                split_data = dataset[split]
                if len(split_data) > max_examples[split]:
                    split_data = split_data.select(range(max_examples[split]))
                
                for example in tqdm(split_data, desc=f"Processing {split}"):
                    data.append({
                        'source': example['translation']['en'],
                        'target': example['translation']['de'],
                        'language_pair': 'en-de',
                        'task': 'translation'
                    })
                
                with open(f'./data/translation/{split}.json', 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"   - {split}: {len(data)} examples")
            
            print("✅ WMT16 setup complete")
            return True
            
        except Exception as e2:
            print(f"❌ Alternative translation setup failed: {e2}")
            print("   🔄 Using synthetic translation data instead")
            return False

def create_dataset_summary():
    """Create a summary of available datasets."""
    print("\n📋 Creating dataset summary...")
    
    summary = {
        'datasets': {},
        'total_examples': 0,
        'tasks': []
    }
    
    task_dirs = ['sst2', 'squad', 'xsum', 'code_generation', 'translation']
    task_names = ['sentiment', 'qa', 'summarization', 'code_generation', 'translation']
    
    for task_dir, task_name in zip(task_dirs, task_names):
        task_info = {'splits': {}}
        
        for split in ['train', 'validation', 'test']:
            file_path = f'./data/{task_dir}/{split}.json'
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    task_info['splits'][split] = len(data)
                    summary['total_examples'] += len(data)
                except:
                    task_info['splits'][split] = 0
            else:
                task_info['splits'][split] = 0
        
        summary['datasets'][task_name] = task_info
        summary['tasks'].append(task_name)
    
    # Save summary
    with open('./data/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 DATASET SUMMARY")
    print("="*50)
    
    for task_name, task_info in summary['datasets'].items():
        print(f"\n🎯 {task_name.upper()}:")
        for split, count in task_info['splits'].items():
            if count > 0:
                print(f"   - {split}: {count:,} examples")
    
    print(f"\n📈 TOTAL EXAMPLES: {summary['total_examples']:,}")
    print("✅ Dataset summary saved to ./data/dataset_summary.json")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup DYNAMO datasets")
    parser.add_argument("--force-redownload", action="store_true", 
                       help="Force redownload of existing datasets")
    args = parser.parse_args()
    
    print("🚀 DYNAMO Dataset Setup")
    print("="*50)
    
    # Create directories
    setup_directories()
    
    # Download datasets
    success_count = 0
    total_count = 5
    
    if download_sst2():
        success_count += 1
    
    if download_squad():
        success_count += 1
    
    if download_xsum():
        success_count += 1
    
    if download_code_generation():
        success_count += 1
    
    if download_translation():
        success_count += 1
    
    # Create summary
    create_dataset_summary()
    
    # Final status
    print("\n" + "="*50)
    print(f"🎉 Setup Complete: {success_count}/{total_count} datasets ready")
    
    if success_count >= 2:  # At least sentiment and QA
        print("✅ Minimum datasets available - DYNAMO ready to train!")
    else:
        print("⚠️  Consider using synthetic data fallback")

if __name__ == "__main__":
    main() 