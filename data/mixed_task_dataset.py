"""
Mixed task dataset generator for DYNAMO.
Creates examples that require multiple adapters to handle complex, multi-task inputs.
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


class MixedTaskExample:
    """
    Represents a mixed-task example that requires multiple adapters.
    """
    
    def __init__(
        self,
        input_text: str,
        tasks: List[str],
        expected_outputs: Dict[str, Any],
        instruction: str = "",
        difficulty: str = "medium"
    ):
        """
        Initialize mixed task example.
        
        Args:
            input_text: Input text for the example
            tasks: List of tasks required for this example
            expected_outputs: Expected outputs for each task
            instruction: Human-readable instruction
            difficulty: Difficulty level (easy, medium, hard)
        """
        self.input_text = input_text
        self.tasks = tasks
        self.expected_outputs = expected_outputs
        self.instruction = instruction
        self.difficulty = difficulty
        self.task_weights = {task: 1.0 / len(tasks) for task in tasks}  # Equal weights by default


class MixedTaskDataset(Dataset):
    """
    Dataset containing mixed-task examples for router training.
    """
    
    def __init__(
        self,
        examples: List[MixedTaskExample],
        tokenizer: RobertaTokenizer,
        max_length: int = 512,
        task_to_id: Dict[str, int] = None
    ):
        """
        Initialize mixed task dataset.
        
        Args:
            examples: List of mixed task examples
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
        """Get a single mixed task example."""
        example = self.examples[idx]
        
        # Tokenize input
        tokenized = self.tokenizer(
            example.input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create task labels (multi-hot encoding)
        task_labels = torch.zeros(len(self.task_to_id))
        for task in example.tasks:
            if task in self.task_to_id:
                task_labels[self.task_to_id[task]] = 1.0
        
        # Create task weights
        task_weights = torch.zeros(len(self.task_to_id))
        for task, weight in example.task_weights.items():
            if task in self.task_to_id:
                task_weights[self.task_to_id[task]] = weight
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_labels': task_labels,
            'task_weights': task_weights,
            'tasks': example.tasks,
            'expected_outputs': example.expected_outputs,
            'instruction': example.instruction,
            'difficulty': example.difficulty,
            'input_text': example.input_text
        }


class MixedTaskGenerator:
    """
    Generator for creating mixed-task examples.
    """
    
    def __init__(
        self,
        single_task_datasets: Dict[str, Dataset],
        tokenizer: RobertaTokenizer,
        config: Dict[str, Any] = None
    ):
        """
        Initialize mixed task generator.
        
        Args:
            single_task_datasets: Dictionary of single-task datasets
            tokenizer: RoBERTa tokenizer
            config: Configuration for generation
        """
        self.single_task_datasets = single_task_datasets
        self.tokenizer = tokenizer
        self.config = config or {}
        
        # Task combination strategies
        self.task_combinations = [
            ['sentiment', 'summarization'],  # Analyze sentiment and summarize
            ['qa', 'sentiment'],             # Answer question and analyze sentiment
            ['code_generation', 'summarization'],  # Generate code and summarize
            ['translation', 'sentiment'],    # Translate and analyze sentiment
            ['qa', 'summarization'],         # Answer question and summarize
            ['sentiment', 'code_generation'], # Analyze sentiment and generate code
            ['translation', 'summarization'], # Translate and summarize
        ]
        
        # Instruction templates
        self.instruction_templates = {
            ('sentiment', 'summarization'): [
                "Analyze the sentiment of this text and provide a summary.",
                "Determine if this text is positive or negative, then summarize it.",
                "What's the sentiment of this passage? Also, give me a brief summary."
            ],
            ('qa', 'sentiment'): [
                "Answer the question and tell me the sentiment of the passage.",
                "What's the answer to this question? Also, is the text positive or negative?",
                "Please answer the question and analyze the emotional tone."
            ],
            ('code_generation', 'summarization'): [
                "Write code for this task and explain what it does.",
                "Generate the requested code and provide a summary of its functionality.",
                "Create code to solve this problem and describe how it works."
            ],
            ('translation', 'sentiment'): [
                "Translate this text and tell me its sentiment.",
                "What does this say in English? Also, is it positive or negative?",
                "Please translate and analyze the emotional tone."
            ],
            ('qa', 'summarization'): [
                "Answer the question and summarize the passage.",
                "What's the answer? Also, give me a brief summary of the text.",
                "Please answer the question and provide a summary."
            ]
        }
        
        logger.info("Mixed task generator initialized")
    
    def generate_mixed_examples(
        self,
        num_examples: int = 5000,
        difficulty_distribution: Dict[str, float] = None
    ) -> List[MixedTaskExample]:
        """
        Generate mixed-task examples.
        
        Args:
            num_examples: Number of examples to generate
            difficulty_distribution: Distribution of difficulty levels
        
        Returns:
            List of mixed task examples
        """
        if difficulty_distribution is None:
            difficulty_distribution = {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
        
        examples = []
        
        logger.info(f"Generating {num_examples} mixed-task examples...")
        
        for i in range(num_examples):
            # Select task combination
            task_combo = random.choice(self.task_combinations)
            
            # Select difficulty
            difficulty = random.choices(
                list(difficulty_distribution.keys()),
                weights=list(difficulty_distribution.values())
            )[0]
            
            # Generate example for this combination
            try:
                example = self._generate_single_mixed_example(task_combo, difficulty)
                if example:
                    examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to generate example {i}: {e}")
                continue
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{num_examples} examples")
        
        logger.info(f"Successfully generated {len(examples)} mixed-task examples")
        return examples
    
    def _generate_single_mixed_example(
        self,
        tasks: List[str],
        difficulty: str = "medium"
    ) -> Optional[MixedTaskExample]:
        """
        Generate a single mixed-task example.
        
        Args:
            tasks: List of tasks to combine
            difficulty: Difficulty level
        
        Returns:
            Mixed task example or None if generation fails
        """
        if len(tasks) == 2:
            return self._generate_dual_task_example(tasks, difficulty)
        elif len(tasks) > 2:
            return self._generate_multi_task_example(tasks, difficulty)
        else:
            return None
    
    def _generate_dual_task_example(
        self,
        tasks: List[str],
        difficulty: str
    ) -> Optional[MixedTaskExample]:
        """Generate example combining two tasks."""
        task1, task2 = tasks
        
        # Get sample data from each task
        if task1 not in self.single_task_datasets or task2 not in self.single_task_datasets:
            return None
        
        dataset1 = self.single_task_datasets[task1]
        dataset2 = self.single_task_datasets[task2]
        
        # Sample examples
        idx1 = random.randint(0, len(dataset1) - 1)
        idx2 = random.randint(0, len(dataset2) - 1)
        
        example1 = dataset1[idx1]
        example2 = dataset2[idx2]
        
        # Create mixed example based on task combination
        if (task1, task2) == ('sentiment', 'summarization'):
            return self._create_sentiment_summarization_example(example1, example2, difficulty)
        elif (task1, task2) == ('qa', 'sentiment'):
            return self._create_qa_sentiment_example(example1, example2, difficulty)
        elif (task1, task2) == ('code_generation', 'summarization'):
            return self._create_code_summarization_example(example1, example2, difficulty)
        elif (task1, task2) == ('translation', 'sentiment'):
            return self._create_translation_sentiment_example(example1, example2, difficulty)
        elif (task1, task2) == ('qa', 'summarization'):
            return self._create_qa_summarization_example(example1, example2, difficulty)
        else:
            return self._create_generic_mixed_example(tasks, [example1, example2], difficulty)
    
    def _create_sentiment_summarization_example(
        self,
        sentiment_example: Dict,
        summarization_example: Dict,
        difficulty: str
    ) -> MixedTaskExample:
        """Create sentiment + summarization example."""
        # Use the document from summarization task
        input_text = summarization_example['input_text']
        
        # Create instruction
        templates = self.instruction_templates.get(('sentiment', 'summarization'), [
            "Analyze the sentiment and provide a summary."
        ])
        instruction = random.choice(templates)
        
        # Expected outputs
        expected_outputs = {
            'sentiment': sentiment_example.get('target', 1),  # Default to positive
            'summarization': summarization_example.get('target', "Summary not available")
        }
        
        return MixedTaskExample(
            input_text=input_text,
            tasks=['sentiment', 'summarization'],
            expected_outputs=expected_outputs,
            instruction=instruction,
            difficulty=difficulty
        )
    
    def _create_qa_sentiment_example(
        self,
        qa_example: Dict,
        sentiment_example: Dict,
        difficulty: str
    ) -> MixedTaskExample:
        """Create QA + sentiment example."""
        # Use the QA input (question + context)
        input_text = qa_example['input_text']
        
        templates = self.instruction_templates.get(('qa', 'sentiment'), [
            "Answer the question and analyze the sentiment."
        ])
        instruction = random.choice(templates)
        
        expected_outputs = {
            'qa': qa_example.get('target', [0, 0]),
            'sentiment': sentiment_example.get('target', 1)
        }
        
        return MixedTaskExample(
            input_text=input_text,
            tasks=['qa', 'sentiment'],
            expected_outputs=expected_outputs,
            instruction=instruction,
            difficulty=difficulty
        )
    
    def _create_code_summarization_example(
        self,
        code_example: Dict,
        summarization_example: Dict,
        difficulty: str
    ) -> MixedTaskExample:
        """Create code generation + summarization example."""
        # Use the code description as input
        input_text = code_example['input_text']
        
        templates = self.instruction_templates.get(('code_generation', 'summarization'), [
            "Generate code and explain what it does."
        ])
        instruction = random.choice(templates)
        
        expected_outputs = {
            'code_generation': code_example.get('target', "# Code not available"),
            'summarization': f"This code {code_example['input_text'].lower()}"
        }
        
        return MixedTaskExample(
            input_text=input_text,
            tasks=['code_generation', 'summarization'],
            expected_outputs=expected_outputs,
            instruction=instruction,
            difficulty=difficulty
        )
    
    def _create_translation_sentiment_example(
        self,
        translation_example: Dict,
        sentiment_example: Dict,
        difficulty: str
    ) -> MixedTaskExample:
        """Create translation + sentiment example."""
        # Use the source text from translation
        input_text = translation_example['input_text']
        
        templates = self.instruction_templates.get(('translation', 'sentiment'), [
            "Translate this text and analyze its sentiment."
        ])
        instruction = random.choice(templates)
        
        expected_outputs = {
            'translation': translation_example.get('target', "Translation not available"),
            'sentiment': sentiment_example.get('target', 1)
        }
        
        return MixedTaskExample(
            input_text=input_text,
            tasks=['translation', 'sentiment'],
            expected_outputs=expected_outputs,
            instruction=instruction,
            difficulty=difficulty
        )
    
    def _create_qa_summarization_example(
        self,
        qa_example: Dict,
        summarization_example: Dict,
        difficulty: str
    ) -> MixedTaskExample:
        """Create QA + summarization example."""
        input_text = qa_example['input_text']
        
        templates = self.instruction_templates.get(('qa', 'summarization'), [
            "Answer the question and summarize the passage."
        ])
        instruction = random.choice(templates)
        
        expected_outputs = {
            'qa': qa_example.get('target', [0, 0]),
            'summarization': summarization_example.get('target', "Summary not available")
        }
        
        return MixedTaskExample(
            input_text=input_text,
            tasks=['qa', 'summarization'],
            expected_outputs=expected_outputs,
            instruction=instruction,
            difficulty=difficulty
        )
    
    def _create_generic_mixed_example(
        self,
        tasks: List[str],
        examples: List[Dict],
        difficulty: str
    ) -> MixedTaskExample:
        """Create a generic mixed example."""
        # Use the first example's input
        input_text = examples[0]['input_text']
        
        instruction = f"Perform the following tasks: {', '.join(tasks)}"
        
        expected_outputs = {}
        for i, task in enumerate(tasks):
            if i < len(examples):
                expected_outputs[task] = examples[i].get('target', None)
        
        return MixedTaskExample(
            input_text=input_text,
            tasks=tasks,
            expected_outputs=expected_outputs,
            instruction=instruction,
            difficulty=difficulty
        )
    
    def _generate_multi_task_example(
        self,
        tasks: List[str],
        difficulty: str
    ) -> Optional[MixedTaskExample]:
        """Generate example with more than two tasks."""
        # For now, limit to pairs and extend later
        if len(tasks) > 2:
            # Select two most compatible tasks
            task_pair = tasks[:2]
            return self._generate_dual_task_example(task_pair, difficulty)
        return None
    
    def create_curriculum_examples(
        self,
        num_examples: int = 5000,
        curriculum_ratio: float = 0.8
    ) -> List[MixedTaskExample]:
        """
        Create examples with curriculum learning progression.
        
        Args:
            num_examples: Total number of examples
            curriculum_ratio: Ratio of easy examples at the start
        
        Returns:
            List of examples ordered by difficulty
        """
        # Generate examples with different difficulties
        easy_count = int(num_examples * curriculum_ratio)
        medium_count = int(num_examples * (1 - curriculum_ratio) * 0.7)
        hard_count = num_examples - easy_count - medium_count
        
        examples = []
        
        # Generate easy examples
        easy_examples = self.generate_mixed_examples(
            easy_count,
            {'easy': 1.0, 'medium': 0.0, 'hard': 0.0}
        )
        examples.extend(easy_examples)
        
        # Generate medium examples
        medium_examples = self.generate_mixed_examples(
            medium_count,
            {'easy': 0.0, 'medium': 1.0, 'hard': 0.0}
        )
        examples.extend(medium_examples)
        
        # Generate hard examples
        hard_examples = self.generate_mixed_examples(
            hard_count,
            {'easy': 0.0, 'medium': 0.0, 'hard': 1.0}
        )
        examples.extend(hard_examples)
        
        logger.info(f"Created curriculum with {len(easy_examples)} easy, "
                   f"{len(medium_examples)} medium, {len(hard_examples)} hard examples")
        
        return examples


def create_mixed_task_dataset(
    single_task_datasets: Dict[str, Dataset],
    config: Dict[str, Any],
    num_examples: int = 5000
) -> MixedTaskDataset:
    """
    Create mixed task dataset from single task datasets.
    
    Args:
        single_task_datasets: Dictionary of single task datasets
        config: Configuration dictionary
        num_examples: Number of mixed examples to generate
    
    Returns:
        Mixed task dataset
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Create generator
    generator = MixedTaskGenerator(single_task_datasets, tokenizer, config)
    
    # Generate examples
    examples = generator.generate_mixed_examples(num_examples)
    
    # Create dataset
    # Handle both dictionary and config object formats
    if hasattr(config, 'get'):  # Dictionary
        max_length = config.get('data', {}).get('max_input_length', 512)
    else:  # Config object
        max_length = getattr(config.data, 'max_input_length', 512) if hasattr(config, 'data') else 512
    
    dataset = MixedTaskDataset(
        examples,
        tokenizer,
        max_length=max_length
    )
    
    return dataset


def create_mixed_task_dataloader(
    mixed_dataset: MixedTaskDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for mixed task dataset.
    
    Args:
        mixed_dataset: Mixed task dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
    
    Returns:
        DataLoader for mixed task dataset
    """
    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

