"""
Dataset loaders for DYNAMO training.
Handles loading and preprocessing of SST-2, SQuAD, XSum, code generation, and translation datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Optional, Tuple, Any, Union
import random
import json
import os

try:
    from utils.logger import get_logger
    from utils.helpers import set_seed
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logger import get_logger
    from utils.helpers import set_seed

logger = get_logger(__name__)


class DynamoDataset(Dataset):
    """
    Base dataset class for DYNAMO tasks.
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: RobertaTokenizer,
        max_length: int = 512,
        task_name: str = "unknown"
    ):
        """
        Initialize DYNAMO dataset.
        
        Args:
            data: List of data examples
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
            task_name: Name of the task
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_name = task_name
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example."""
        example = self.data[idx]
        
        # Tokenize input
        tokenized = self.tokenizer(
            example['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare output
        output = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': self.task_name,
            'task_id': example.get('task_id', 0)
        }
        
        # Add task-specific targets
        if 'target' in example:
            output['target'] = example['target']
        
        return output


class SentimentDataset(DynamoDataset):
    """Dataset for sentiment analysis (SST-2)."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        
        # Tokenize input
        tokenized = self.tokenizer(
            example['sentence'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'sentiment',
            'task_id': 0,
            'target': torch.tensor(example['label'], dtype=torch.long),
            'input_text': example['sentence']
        }


class QADataset(DynamoDataset):
    """Dataset for question answering (SQuAD)."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
        
        # Get answer positions (simplified)
        start_pos = example.get('start_position', 0)
        end_pos = example.get('end_position', 0)
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'qa',
            'task_id': 1,
            'target': torch.tensor([start_pos, end_pos], dtype=torch.long),
            'input_text': input_text
        }


class SummarizationDataset(DynamoDataset):
    """Dataset for summarization (XSum)."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        
        # Tokenize input document
        tokenized = self.tokenizer(
            example['document'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target summary (for training)
        target_tokenized = self.tokenizer(
            example['summary'],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'summarization',
            'task_id': 2,
            'target': target_tokenized['input_ids'].squeeze(0),
            'target_attention_mask': target_tokenized['attention_mask'].squeeze(0),
            'input_text': example['document']
        }


class CodeGenerationDataset(DynamoDataset):
    """Dataset for code generation."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        
        # Tokenize input (natural language description)
        tokenized = self.tokenizer(
            example['description'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target code
        target_tokenized = self.tokenizer(
            example['code'],
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'code_generation',
            'task_id': 3,
            'target': target_tokenized['input_ids'].squeeze(0),
            'target_attention_mask': target_tokenized['attention_mask'].squeeze(0),
            'input_text': example['description']
        }


class TranslationDataset(DynamoDataset):
    """Dataset for translation."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        
        # Tokenize source text
        tokenized = self.tokenizer(
            example['source'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target translation
        target_tokenized = self.tokenizer(
            example['target'],
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'translation',
            'task_id': 4,
            'target': target_tokenized['input_ids'].squeeze(0),
            'target_attention_mask': target_tokenized['attention_mask'].squeeze(0),
            'input_text': example['source']
        }


class DatasetLoader:
    """
    Main dataset loader for DYNAMO.
    Handles loading and preprocessing of all task datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get("data", {})
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # Dataset sizes
        # Dataset size limits
        if hasattr(self.data_config, 'get'):  # Dictionary
            self.dataset_sizes = {
                'sentiment': self.data_config.get('sst2_size', 10000),
                'qa': self.data_config.get('squad_size', 20000),
                'summarization': self.data_config.get('xsum_size', 15000),
                'code_generation': self.data_config.get('code_gen_size', 8000),
                'translation': self.data_config.get('translation_size', 12000)
            }
        else:  # Config object
            self.dataset_sizes = {
                'sentiment': getattr(self.data_config, 'sst2_size', 10000),
                'qa': getattr(self.data_config, 'squad_size', 20000),
                'summarization': getattr(self.data_config, 'xsum_size', 15000),
                'code_generation': getattr(self.data_config, 'code_gen_size', 8000),
                'translation': getattr(self.data_config, 'translation_size', 12000)
            }
        
        # Handle both dictionary and config object formats
        if hasattr(self.data_config, 'get'):  # Dictionary
            self.max_length = self.data_config.get('max_input_length', 512)
            self.cache_dir = self.data_config.get('cache_dir', './cache')
        else:  # Config object
            self.max_length = getattr(self.data_config, 'max_input_length', 512)
            self.cache_dir = getattr(self.data_config, 'cache_dir', './cache')
        
        logger.info("Dataset loader initialized")
    
    def load_sentiment_data(self, split: str = 'train') -> List[Dict]:
        """Load sentiment analysis data (SST-2)."""
        # Try to load real dataset first
        data_path = f"./data/sst2/{split}.json"
        if os.path.exists(data_path):
            logger.info(f"Loading real SST-2 {split} data from {data_path}")
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning(f"Failed to load real SST-2 data: {e}")
        
        # Fallback to Hugging Face datasets
        try:
            logger.info(f"Loading SST-2 {split} data from Hugging Face")
            dataset = load_dataset("sst2", split=split)
            data = []
            for example in dataset:
                data.append({
                    'sentence': example['sentence'],
                    'label': example['label'],
                    'task': 'sentiment'
                })
            return data
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
        
        # Final fallback to synthetic data
        logger.info(f"Using synthetic sentiment data for {split}")
        return self._create_synthetic_sentiment_data()
    
    def load_qa_data(self, split: str = 'train') -> List[Dict]:
        """Load question answering data (SQuAD)."""
        # Try to load real dataset first
        data_path = f"./data/squad/{split}.json"
        if os.path.exists(data_path):
            logger.info(f"Loading real SQuAD {split} data from {data_path}")
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning(f"Failed to load real SQuAD data: {e}")
        
        # Fallback to Hugging Face datasets
        try:
            logger.info(f"Loading SQuAD {split} data from Hugging Face")
            dataset = load_dataset("squad", split=split)
            data = []
            for example in dataset:
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
            return data
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
        
        # Final fallback to synthetic data
        logger.info(f"Using synthetic QA data for {split}")
        return self._create_synthetic_qa_data()
    
    def load_summarization_data(self, split: str = 'train') -> List[Dict]:
        """Load summarization data (XSum)."""
        # Try to load real dataset first
        data_path = f"./data/xsum/{split}.json"
        if os.path.exists(data_path):
            logger.info(f"Loading real XSum {split} data from {data_path}")
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning(f"Failed to load real XSum data: {e}")
        
        # Fallback to Hugging Face datasets
        try:
            logger.info(f"Loading XSum {split} data from Hugging Face")
            dataset = load_dataset("xsum", split=split, trust_remote_code=True)
            # Limit size for faster experimentation
            max_examples = {'train': 10000, 'validation': 1000, 'test': 1000}
            if len(dataset) > max_examples.get(split, 1000):
                dataset = dataset.select(range(max_examples.get(split, 1000)))
            
            data = []
            for example in dataset:
                data.append({
                    'document': example['document'],
                    'summary': example['summary'],
                    'id': example['id'],
                    'task': 'summarization'
                })
            return data
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
        
        # Final fallback to synthetic data
        logger.info(f"Using synthetic summarization data for {split}")
        return self._create_synthetic_summarization_data()
    
    def load_code_generation_data(self, split: str = 'train') -> List[Dict]:
        """Load code generation data."""
        # Try to load real dataset first
        data_path = f"./data/code_generation/{split}.json"
        if os.path.exists(data_path):
            logger.info(f"Loading real code generation {split} data from {data_path}")
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning(f"Failed to load real code generation data: {e}")
        
        # Fallback to Hugging Face datasets
        try:
            logger.info(f"Loading CodeSearchNet {split} data from Hugging Face")
            dataset = load_dataset("code_search_net", "python", split=split)
            # Limit size for faster experimentation
            max_examples = {'train': 5000, 'validation': 500, 'test': 500}
            if len(dataset) > max_examples.get(split, 500):
                dataset = dataset.select(range(max_examples.get(split, 500)))
            
            data = []
            for example in dataset:
                # Filter out examples without docstring
                if example['func_documentation_string'] and example['func_code_string']:
                    data.append({
                        'description': example['func_documentation_string'],
                        'code': example['func_code_string'],
                        'func_name': example['func_name'],
                        'task': 'code_generation'
                    })
            return data
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
        
        # Final fallback to synthetic data
        logger.info(f"Using synthetic code generation data for {split}")
        return self._create_synthetic_code_data()
    
    def load_translation_data(self, split: str = 'train') -> List[Dict]:
        """Load translation data."""
        # Try to load real dataset first
        data_path = f"./data/translation/{split}.json"
        if os.path.exists(data_path):
            logger.info(f"Loading real translation {split} data from {data_path}")
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning(f"Failed to load real translation data: {e}")
        
        # Fallback to Hugging Face datasets
        try:
            logger.info(f"Loading WMT14 {split} data from Hugging Face")
            dataset = load_dataset("wmt14", "de-en", split=split)
            # Limit size for faster experimentation
            max_examples = {'train': 5000, 'validation': 500, 'test': 500}
            if len(dataset) > max_examples.get(split, 500):
                dataset = dataset.select(range(max_examples.get(split, 500)))
            
            data = []
            for example in dataset:
                data.append({
                    'source': example['translation']['de'],
                    'target': example['translation']['en'],
                    'language_pair': 'de-en',
                    'task': 'translation'
                })
            return data
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
        
        # Final fallback to synthetic data
        logger.info(f"Using synthetic translation data for {split}")
        return self._create_synthetic_translation_data()
    
    def _create_synthetic_sentiment_data(self) -> List[Dict]:
        """Create synthetic sentiment data for testing."""
        data = []
        positive_examples = [
            "This movie is amazing and wonderful!",
            "I love this product, it's fantastic.",
            "Great service and excellent quality.",
            "Outstanding performance and brilliant acting.",
            "Absolutely perfect and highly recommended."
        ]
        negative_examples = [
            "This movie is terrible and boring.",
            "I hate this product, it's awful.",
            "Poor service and bad quality.",
            "Disappointing performance and bad acting.",
            "Completely useless and not recommended."
        ]
        
        for i in range(self.dataset_sizes['sentiment']):
            if i % 2 == 0:
                sentence = random.choice(positive_examples)
                label = 1
            else:
                sentence = random.choice(negative_examples)
                label = 0
            
            data.append({'sentence': sentence, 'label': label})
        
        return data
    
    def _create_synthetic_qa_data(self) -> List[Dict]:
        """Create synthetic QA data for testing."""
        data = []
        templates = [
            {
                'context': "The capital of France is Paris. Paris is known for the Eiffel Tower.",
                'question': "What is the capital of France?",
                'answer': "Paris",
                'start_position': 23,
                'end_position': 28
            },
            {
                'context': "Python is a programming language. It was created by Guido van Rossum.",
                'question': "Who created Python?",
                'answer': "Guido van Rossum",
                'start_position': 53,
                'end_position': 69
            }
        ]
        
        for i in range(self.dataset_sizes['qa']):
            template = templates[i % len(templates)]
            data.append(template.copy())
        
        return data
    
    def _create_synthetic_summarization_data(self) -> List[Dict]:
        """Create synthetic summarization data for testing."""
        data = []
        examples = [
            {
                'document': "Artificial intelligence is transforming many industries. Machine learning algorithms can process vast amounts of data and identify patterns that humans might miss. This technology is being used in healthcare, finance, and transportation.",
                'summary': "AI and machine learning are transforming industries like healthcare, finance, and transportation."
            },
            {
                'document': "Climate change is one of the most pressing issues of our time. Rising temperatures, melting ice caps, and extreme weather events are affecting ecosystems worldwide. Governments and organizations are working to reduce carbon emissions.",
                'summary': "Climate change causes rising temperatures and extreme weather, prompting efforts to reduce emissions."
            }
        ]
        
        for i in range(self.dataset_sizes['summarization']):
            example = examples[i % len(examples)]
            data.append(example.copy())
        
        return data
    
    def _create_synthetic_code_data(self) -> List[Dict]:
        """Create synthetic code generation data for testing."""
        data = []
        examples = [
            {
                'description': "Function to calculate the factorial of a number",
                'code': "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
            },
            {
                'description': "Function to check if a number is prime",
                'code': "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
            }
        ]
        
        for i in range(self.dataset_sizes['code_generation']):
            example = examples[i % len(examples)]
            data.append(example.copy())
        
        return data
    
    def _create_synthetic_translation_data(self) -> List[Dict]:
        """Create synthetic translation data for testing."""
        data = []
        examples = [
            {'source': "Hallo, wie geht es dir?", 'target': "Hello, how are you?"},
            {'source': "Ich liebe Musik.", 'target': "I love music."},
            {'source': "Das Wetter ist schön heute.", 'target': "The weather is nice today."},
            {'source': "Wo ist die Bibliothek?", 'target': "Where is the library?"}
        ]
        
        for i in range(self.dataset_sizes['translation']):
            example = examples[i % len(examples)]
            data.append(example.copy())
        
        return data
    
    def create_datasets(self, split: str = 'train') -> Dict[str, Dataset]:
        """
        Create all task datasets.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
        
        Returns:
            Dictionary of datasets
        """
        datasets = {}
        
        # Load data for each task
        sentiment_data = self.load_sentiment_data(split)
        qa_data = self.load_qa_data(split)
        summarization_data = self.load_summarization_data(split)
        code_data = self.load_code_generation_data(split)
        translation_data = self.load_translation_data(split)
        
        # Create datasets
        datasets['sentiment'] = SentimentDataset(
            sentiment_data, self.tokenizer, self.max_length, 'sentiment'
        )
        datasets['qa'] = QADataset(
            qa_data, self.tokenizer, self.max_length, 'qa'
        )
        datasets['summarization'] = SummarizationDataset(
            summarization_data, self.tokenizer, self.max_length, 'summarization'
        )
        datasets['code_generation'] = CodeGenerationDataset(
            code_data, self.tokenizer, self.max_length, 'code_generation'
        )
        datasets['translation'] = TranslationDataset(
            translation_data, self.tokenizer, self.max_length, 'translation'
        )
        
        logger.info(f"Created {len(datasets)} task datasets for {split} split")
        return datasets
    
    def create_dataloaders(
        self,
        datasets: Dict[str, Dataset],
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for all datasets.
        
        Args:
            datasets: Dictionary of datasets
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
        
        Returns:
            Dictionary of DataLoaders
        """
        dataloaders = {}
        
        for task_name, dataset in datasets.items():
            dataloaders[task_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        
        return dataloaders

