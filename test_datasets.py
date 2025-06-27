#!/usr/bin/env python3
"""
Test script to verify that all datasets are properly set up for DYNAMO.
"""

import json
import os
from pathlib import Path

def test_dataset_files():
    """Test that all dataset files exist and are readable."""
    print("🔍 Testing dataset files...")
    print()
    
    datasets = {
        'sentiment': 'sst2',
        'qa': 'squad', 
        'summarization': 'xsum',
        'code_generation': 'code_generation',
        'translation': 'translation'
    }
    
    results = {}
    
    for task_name, dir_name in datasets.items():
        print(f"📊 {task_name.upper()}:")
        task_results = {}
        
        for split in ['train', 'validation', 'test']:
            file_path = f"./data/{dir_name}/{split}.json"
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    count = len(data)
                    task_results[split] = count
                    print(f"   ✅ {split}: {count:,} examples")
                    
                    # Show sample keys
                    if data:
                        sample_keys = list(data[0].keys())
                        print(f"      Keys: {sample_keys}")
                        
                except Exception as e:
                    task_results[split] = 0
                    print(f"   ❌ {split}: Error reading file - {e}")
            else:
                task_results[split] = 0
                print(f"   ⚠️  {split}: File not found")
        
        results[task_name] = task_results
        print()
    
    return results

def test_sample_data():
    """Test sample data from each dataset."""
    print("🔍 Testing sample data...")
    print()
    
    # Test sentiment data
    try:
        with open('./data/sst2/train.json', 'r') as f:
            sentiment_data = json.load(f)
        
        sample = sentiment_data[0]
        print("📊 SENTIMENT SAMPLE:")
        print(f"   Text: {sample['sentence'][:100]}...")
        print(f"   Label: {sample['label']}")
        print()
    except Exception as e:
        print(f"❌ Sentiment sample test failed: {e}")
    
    # Test QA data
    try:
        with open('./data/squad/train.json', 'r') as f:
            qa_data = json.load(f)
        
        sample = qa_data[0]
        print("📊 QA SAMPLE:")
        print(f"   Question: {sample['question']}")
        print(f"   Context: {sample['context'][:100]}...")
        print(f"   Answer: {sample['answer']}")
        print()
    except Exception as e:
        print(f"❌ QA sample test failed: {e}")
    
    # Test summarization data
    try:
        with open('./data/xsum/train.json', 'r') as f:
            summ_data = json.load(f)
        
        sample = summ_data[0]
        print("📊 SUMMARIZATION SAMPLE:")
        print(f"   Document: {sample['document'][:100]}...")
        print(f"   Summary: {sample['summary']}")
        print()
    except Exception as e:
        print(f"❌ Summarization sample test failed: {e}")
    
    # Test code generation data
    try:
        with open('./data/code_generation/train.json', 'r') as f:
            code_data = json.load(f)
        
        sample = code_data[0]
        print("📊 CODE GENERATION SAMPLE:")
        print(f"   Description: {sample['description'][:100]}...")
        print(f"   Code: {sample['code'][:100]}...")
        print()
    except Exception as e:
        print(f"❌ Code generation sample test failed: {e}")
    
    # Test translation data
    try:
        with open('./data/translation/train.json', 'r') as f:
            trans_data = json.load(f)
        
        sample = trans_data[0]
        print("📊 TRANSLATION SAMPLE:")
        print(f"   Source (EN): {sample['source']}")
        print(f"   Target (DE): {sample['target']}")
        print()
    except Exception as e:
        print(f"❌ Translation sample test failed: {e}")

def main():
    """Main test function."""
    print("🚀 DYNAMO Dataset Verification")
    print("="*50)
    
    # Test dataset files
    results = test_dataset_files()
    
    # Test sample data
    test_sample_data()
    
    # Summary
    print("="*50)
    print("📋 SUMMARY")
    print("="*50)
    
    total_examples = 0
    ready_tasks = 0
    
    for task_name, task_results in results.items():
        train_count = task_results.get('train', 0)
        val_count = task_results.get('validation', 0) 
        test_count = task_results.get('test', 0)
        
        task_total = train_count + val_count + test_count
        total_examples += task_total
        
        if train_count > 0:
            ready_tasks += 1
            status = "✅ READY"
        else:
            status = "❌ NOT READY"
        
        print(f"{task_name.upper():>15}: {task_total:>7,} examples {status}")
    
    print("-" * 50)
    print(f"{'TOTAL':>15}: {total_examples:>7,} examples")
    print(f"{'READY TASKS':>15}: {ready_tasks:>7}/5")
    print()
    
    if ready_tasks == 5:
        print("🎉 All datasets ready! DYNAMO can start training!")
    elif ready_tasks >= 3:
        print("⚠️  Most datasets ready. Can proceed with limited training.")
    else:
        print("❌ Insufficient datasets. Run setup_datasets.py first.")

if __name__ == "__main__":
    main() 