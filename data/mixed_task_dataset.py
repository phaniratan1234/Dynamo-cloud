"""
Task-specific dataset for DYNAMO router training.
Creates single-task examples with proper task labels for supervised router learning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from typing import Dict, List, Optional, Tuple, Any, Union
import random
import itertools
from collections import defaultdict

try:
    from ..utils.logger import get_logger
    from ..utils.helpers import set_seed
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logger import get_logger
    from utils.helpers import set_seed

logger = get_logger(__name__)


class TaskSpecificExample:
    """
    Represents a single-task example for router training.
    This is the correct approach - teach router to recognize task types.
    """
    
    def __init__(
        self,
        input_text: str,
        task_name: str,
        target: Any,
        task_id: int,
        source_dataset: str = "",
        confidence: float = 1.0
    ):
        """
        Initialize task-specific example.
        
        Args:
            input_text: Input text for the example
            task_name: Name of the task (sentiment, qa, summarization, etc.)
            target: Target output for this task
            task_id: Numeric ID of the task (for router supervision)
            source_dataset: Name of source dataset
            confidence: Confidence in the task label (for curriculum learning)
        """
        self.input_text = input_text
        self.task_name = task_name
        self.target = target
        self.task_id = task_id
        self.source_dataset = source_dataset
        self.confidence = confidence


class RouterTrainingDataset(Dataset):
    """
    Dataset for training the router to recognize task types.
    Contains single-task examples with clear task labels.
    """
    
    def __init__(
        self,
        examples: List[TaskSpecificExample],
        tokenizer: RobertaTokenizer,
        max_length: int = 512,
        task_to_id: Dict[str, int] = None
    ):
        """
        Initialize router training dataset.
        
        Args:
            examples: List of task-specific examples
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
            task_to_id: Mapping from task names to IDs
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if task_to_id is None:
            task_to_id = {
                'sentiment': 0,
                'qa': 1,
                'summarization': 2,
                'code_generation': 3,
                'translation': 4
            }
        self.task_to_id = task_to_id
        self.id_to_task = {v: k for k, v in task_to_id.items()}
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single task-specific example."""
        example = self.examples[idx]
        
        # Tokenize input
        tokenized = self.tokenizer(
            example.input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process target based on task type
        processed_target = self._process_target(example.target, example.task_name)
        
        # Create the data structure
        data = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_label': torch.tensor(example.task_id, dtype=torch.long),  # For router supervision
            'task_name': example.task_name,
            'input_text': example.input_text,
            'confidence': torch.tensor(example.confidence, dtype=torch.float32)
        }
        
        # Add task-specific target
        data[example.task_name] = processed_target
        
        # Add default values for other tasks (for compatibility)
        all_tasks = ['sentiment', 'qa', 'summarization', 'code_generation', 'translation']
        for task_name in all_tasks:
            if task_name not in data:
                data[task_name] = self._get_default_target(task_name)
        
        return data
    
    def _process_target(self, target: Any, task_name: str) -> torch.Tensor:
        """Process target based on task type."""
        if task_name == 'sentiment':
            # Binary classification: 0 or 1
            if isinstance(target, str):
                return torch.tensor(1 if target.lower() in ['positive', '1'] else 0, dtype=torch.long)
            return torch.tensor(int(target), dtype=torch.long)
        
        elif task_name == 'qa':
            # Question answering: [start_pos, end_pos]
            if isinstance(target, (list, tuple)) and len(target) == 2:
                return torch.tensor(target, dtype=torch.long)
            elif isinstance(target, dict) and 'start' in target and 'end' in target:
                return torch.tensor([target['start'], target['end']], dtype=torch.long)
            else:
                return torch.tensor([0, 0], dtype=torch.long)  # Default
        
        elif task_name in ['summarization', 'code_generation', 'translation']:
            # Generation tasks: tokenize string targets
            if isinstance(target, str):
                tokenized_target = self.tokenizer(
                    target,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return tokenized_target['input_ids'].squeeze(0)
            elif isinstance(target, torch.Tensor):
                return target
            else:
                # Default empty sequence
                empty_tokens = torch.zeros(self.max_length, dtype=torch.long)
                empty_tokens[0] = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else 0
                return empty_tokens
        
        else:
            # Unknown task
            return torch.tensor(0, dtype=torch.long)
    
    def _get_default_target(self, task_name: str) -> torch.Tensor:
        """Get default target for tasks not present in this example."""
        if task_name == 'sentiment':
            return torch.tensor(0, dtype=torch.long)  # Default neutral
        elif task_name == 'qa':
            return torch.tensor([0, 0], dtype=torch.long)  # Default positions
        elif task_name in ['summarization', 'code_generation', 'translation']:
            empty_tokens = torch.zeros(self.max_length, dtype=torch.long)
            empty_tokens[0] = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else 0
            return empty_tokens
        else:
            return torch.tensor(0, dtype=torch.long)


class RouterDatasetGenerator:
    """
    Generator for creating router training dataset from single-task datasets.
    """
    
    def __init__(
        self,
        single_task_datasets: Dict[str, Dataset],
        tokenizer: RobertaTokenizer,
        task_to_id: Dict[str, int] = None
    ):
        """
        Initialize router dataset generator.
        
        Args:
            single_task_datasets: Dictionary of single-task datasets
            tokenizer: RoBERTa tokenizer
            task_to_id: Mapping from task names to IDs
        """
        self.single_task_datasets = single_task_datasets
        self.tokenizer = tokenizer
        
        if task_to_id is None:
            task_to_id = {
                'sentiment': 0,
                'qa': 1,
                'summarization': 2,
                'code_generation': 3,
                'translation': 4
            }
        self.task_to_id = task_to_id
        
        logger.info(f"Router dataset generator initialized with {len(single_task_datasets)} task datasets")
    
    def generate_router_dataset(
        self,
        num_examples_per_task: int = 1000,
        balance_tasks: bool = True,
        include_curriculum: bool = True
    ) -> List[TaskSpecificExample]:
        """
        Generate router training dataset from single-task datasets.
        
        Args:
            num_examples_per_task: Number of examples per task
            balance_tasks: Whether to balance examples across tasks
            include_curriculum: Whether to include curriculum learning
        
        Returns:
            List of task-specific examples for router training
        """
        all_examples = []
        
        for task_name, dataset in self.single_task_datasets.items():
            if task_name not in self.task_to_id:
                logger.warning(f"Task {task_name} not in task_to_id mapping, skipping")
                continue
            
            task_id = self.task_to_id[task_name]
            task_examples = []
            
            # Sample examples from the dataset
            num_to_sample = min(num_examples_per_task, len(dataset))
            indices = random.sample(range(len(dataset)), num_to_sample)
            
            for idx in indices:
                try:
                    data_item = dataset[idx]
                    
                    # Extract input text and target
                    input_text, target = self._extract_input_and_target(data_item, task_name)
                    
                    if input_text:  # Only add if we successfully extracted input
                        confidence = 1.0
                        if include_curriculum:
                            confidence = self._compute_curriculum_confidence(input_text, task_name)
                        
                        example = TaskSpecificExample(
                            input_text=input_text,
                            task_name=task_name,
                            target=target,
                            task_id=task_id,
                            source_dataset=f"{task_name}_dataset",
                            confidence=confidence
                        )
                        task_examples.append(example)
                
                except Exception as e:
                    logger.warning(f"Failed to process example {idx} from {task_name}: {e}")
                    continue
            
            logger.info(f"Generated {len(task_examples)} examples for {task_name}")
            all_examples.extend(task_examples)
        
        # Shuffle examples
        random.shuffle(all_examples)
        
        logger.info(f"Generated {len(all_examples)} total examples for router training")
        return all_examples
    
    def _extract_input_and_target(self, data_item: Dict, task_name: str) -> Tuple[str, Any]:
        """Extract input text and target from a dataset item."""
        try:
            # The datasets are already processed and have 'input_text' and 'target' fields
            if 'input_text' in data_item and 'target' in data_item:
                input_text = data_item['input_text']
                target = data_item['target']
                
                # Convert tensor targets to appropriate format
                if isinstance(target, torch.Tensor):
                    if task_name == 'sentiment':
                        # Convert to scalar
                        target = target.item() if target.numel() == 1 else int(target[0])
                    elif task_name == 'qa':
                        # Convert to list for QA
                        if target.numel() == 2:
                            target = target.tolist()
                        else:
                            target = [0, 0]  # Default
                    elif task_name in ['summarization', 'code_generation', 'translation']:
                        # For generation tasks, keep as tensor or convert back to text
                        # We'll keep as tensor since it's already tokenized
                        pass
                
                return input_text, target
            
            # Fallback to original logic for raw data formats
            if task_name == 'sentiment':
                # SST-2 format
                if 'sentence' in data_item:
                    input_text = data_item['sentence']
                elif 'text' in data_item:
                    input_text = data_item['text']
                else:
                    input_text = str(data_item.get('input', ''))
                
                target = data_item.get('label', 0)
                return input_text, target
            
            elif task_name == 'qa':
                # SQuAD format
                if 'question' in data_item and 'context' in data_item:
                    input_text = f"Question: {data_item['question']} Context: {data_item['context']}"
                    answers = data_item.get('answers', {})
                    if 'answer_start' in answers and len(answers['answer_start']) > 0:
                        start = answers['answer_start'][0]
                        text = answers.get('text', [''])[0]
                        end = start + len(text)
                        target = [start, end]
                    else:
                        target = [0, 0]
                    return input_text, target
                else:
                    return "", [0, 0]
            
            elif task_name == 'summarization':
                # XSum format
                if 'document' in data_item:
                    input_text = data_item['document']
                    target = data_item.get('summary', '')
                    return input_text, target
                elif 'article' in data_item:
                    input_text = data_item['article']
                    target = data_item.get('highlights', '')
                    return input_text, target
                else:
                    return "", ""
            
            elif task_name == 'code_generation':
                # Code generation format
                if 'prompt' in data_item:
                    input_text = data_item['prompt']
                    target = data_item.get('canonical_solution', '')
                    return input_text, target
                elif 'description' in data_item:
                    input_text = data_item['description']
                    target = data_item.get('code', '')
                    return input_text, target
                else:
                    return "", ""
            
            elif task_name == 'translation':
                # Translation format
                if 'en' in data_item and 'fr' in data_item:
                    input_text = f"Translate to French: {data_item['en']}"
                    target = data_item['fr']
                    return input_text, target
                elif 'source' in data_item:
                    input_text = f"Translate: {data_item['source']}"
                    target = data_item.get('target', '')
                    return input_text, target
                else:
                    return "", ""
            
            else:
                return "", ""
        
        except Exception as e:
            logger.warning(f"Error extracting input/target for {task_name}: {e}")
            return "", ""
    
    def _compute_curriculum_confidence(self, input_text: str, task_name: str) -> float:
        """Compute confidence for curriculum learning."""
        # Simple heuristics for curriculum learning
        text_length = len(input_text.split())
        
        if task_name == 'sentiment':
            # Shorter texts are easier for sentiment
            if text_length < 10:
                return 1.0
            elif text_length < 20:
                return 0.8
            else:
                return 0.6
        
        elif task_name == 'qa':
            # Medium-length contexts are easier
            if 50 < text_length < 200:
                return 1.0
            else:
                return 0.7
        
        elif task_name in ['summarization', 'code_generation', 'translation']:
            # Shorter inputs are easier for generation
            if text_length < 50:
                return 1.0
            elif text_length < 100:
                return 0.8
            else:
                return 0.6
        
        return 1.0  # Default confidence


def create_router_training_dataset(
    single_task_datasets: Dict[str, Dataset],
    tokenizer: RobertaTokenizer,
    config: Dict[str, Any],
    num_examples_per_task: int = 1000
) -> RouterTrainingDataset:
    """
    Create router training dataset from single-task datasets.
    
    Args:
        single_task_datasets: Dictionary of single-task datasets
        tokenizer: RoBERTa tokenizer
        config: Configuration dictionary
        num_examples_per_task: Number of examples per task
    
    Returns:
        Router training dataset
    """
    generator = RouterDatasetGenerator(single_task_datasets, tokenizer)
    
    examples = generator.generate_router_dataset(
        num_examples_per_task=num_examples_per_task,
        balance_tasks=True,
        include_curriculum=True
    )
    
    dataset = RouterTrainingDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=config.get('max_length', 512)
    )
    
    logger.info(f"Created router training dataset with {len(dataset)} examples")
    return dataset


def router_training_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for router training dataset.
    Handles single-task examples with proper task labels.
    """
    # Get all keys from the first item
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key in ['input_ids', 'attention_mask']:
            # Stack tensor inputs
            collated[key] = torch.stack([item[key] for item in batch])
        
        elif key == 'task_label':
            # Stack task labels for router supervision
            collated[key] = torch.stack([item[key] for item in batch])
        
        elif key == 'confidence':
            # Stack confidence scores
            collated[key] = torch.stack([item[key] for item in batch])
        
        elif key in ['task_name', 'input_text']:
            # Keep as lists for string data
            collated[key] = [item[key] for item in batch]
        
        elif key in ['sentiment', 'qa', 'summarization', 'code_generation', 'translation']:
            # Handle task-specific targets
            values = [item[key] for item in batch]
            
            if key == 'sentiment':
                # Sentiment: stack scalar labels
                collated[key] = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.long) for v in values])
            
            elif key == 'qa':
                # QA: stack [start, end] pairs
                collated[key] = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.long) for v in values])
            
            elif key in ['summarization', 'code_generation', 'translation']:
                # Generation: handle variable sequence lengths
                processed_values = []
                max_length = 512  # Default max length
                
                for v in values:
                    if isinstance(v, torch.Tensor):
                        # Pad or truncate to max_length
                        if v.size(0) > max_length:
                            v = v[:max_length]
                        elif v.size(0) < max_length:
                            # Pad with zeros (pad_token_id should be 1 for RoBERTa, but 0 is fine for targets)
                            padding = torch.zeros(max_length - v.size(0), dtype=torch.long)
                            v = torch.cat([v, padding])
                        processed_values.append(v)
                    else:
                        # Create default tensor
                        default_tensor = torch.zeros(max_length, dtype=torch.long)
                        processed_values.append(default_tensor)
                
                collated[key] = torch.stack(processed_values)
        
        else:
            # Keep other keys as lists
            collated[key] = [item[key] for item in batch]
    
    return collated


def create_router_training_dataloader(
    router_dataset: RouterTrainingDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = None
) -> DataLoader:
    """
    Create DataLoader for router training.
    
    Args:
        router_dataset: Router training dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader for router training
    """
    if num_workers is None:
        num_workers = 0  # Single-threaded for compatibility
    
    return DataLoader(
        router_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=router_training_collate_fn,
        pin_memory=True
    )


# Legacy aliases for backward compatibility
MixedTaskDataset = RouterTrainingDataset
create_mixed_task_dataset = create_router_training_dataset
mixed_task_collate_fn = router_training_collate_fn
create_mixed_task_dataloader = create_router_training_dataloader 