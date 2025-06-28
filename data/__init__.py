"""
Data Package for DYNAMO
Contains dataset loaders and mixed task dataset generators.
"""

from .dataset_loaders import (
    DynamoDataset,
    SentimentDataset,
    QADataset,
    SummarizationDataset,
    CodeGenerationDataset,
    TranslationDataset,
    DatasetLoader
)

from .mixed_task_dataset import (
    TaskSpecificExample,
    RouterTrainingDataset,
    RouterDatasetGenerator,
    create_router_training_dataset,
    create_router_training_dataloader,
    # Legacy aliases for backward compatibility
    MixedTaskDataset,
    create_mixed_task_dataset,
    create_mixed_task_dataloader
)

__all__ = [
    'DynamoDataset',
    'SentimentDataset',
    'QADataset',
    'SummarizationDataset',
    'CodeGenerationDataset',
    'TranslationDataset',
    'DatasetLoader',
    'TaskSpecificExample',
    'RouterTrainingDataset',
    'RouterDatasetGenerator',
    'create_router_training_dataset',
    'create_router_training_dataloader',
    # Legacy aliases
    'MixedTaskDataset',
    'create_mixed_task_dataset',
    'create_mixed_task_dataloader'
]

