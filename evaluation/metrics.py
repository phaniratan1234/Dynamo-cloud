"""
Evaluation metrics for DYNAMO.
Implements task-specific metrics including accuracy, BLEU, ROUGE, and custom metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict
import re

# Import evaluation libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. ROUGE metrics will be disabled.")

try:
    from sacrebleu import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. BLEU metrics will be disabled.")

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some metrics will be disabled.")

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetricCalculator:
    """
    Base class for metric calculation.
    """
    
    def __init__(self, task_name: str):
        """
        Initialize metric calculator.
        
        Args:
            task_name: Name of the task
        """
        self.task_name = task_name
    
    def compute(self, predictions: Any, targets: Any) -> Dict[str, float]:
        """
        Compute metrics for predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Dictionary of computed metrics
        """
        raise NotImplementedError


class SentimentMetrics(MetricCalculator):
    """
    Metrics for sentiment analysis task.
    """
    
    def __init__(self):
        super().__init__("sentiment")
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute sentiment analysis metrics.
        
        Args:
            predictions: Logits [batch_size, num_classes]
            targets: True labels [batch_size]
        
        Returns:
            Dictionary with accuracy, precision, recall, F1
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            pred_labels = torch.argmax(predictions, dim=-1).cpu().numpy()
        else:
            pred_labels = np.array(predictions)
        
        if isinstance(targets, torch.Tensor):
            true_labels = targets.cpu().numpy()
        else:
            true_labels = np.array(targets)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = float(np.mean(pred_labels == true_labels))
        
        if SKLEARN_AVAILABLE:
            # Precision, Recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='weighted', zero_division=0
            )
            
            metrics['precision'] = float(precision)
            metrics['recall'] = float(recall)
            metrics['f1'] = float(f1)
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average=None, zero_division=0
            )
            
            for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
                metrics[f'precision_class_{i}'] = float(p)
                metrics[f'recall_class_{i}'] = float(r)
                metrics[f'f1_class_{i}'] = float(f)
        
        return metrics


class QAMetrics(MetricCalculator):
    """
    Metrics for question answering task.
    """
    
    def __init__(self):
        super().__init__("qa")
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute QA metrics.
        
        Args:
            predictions: Start/end logits [batch_size, seq_len, 2] or positions [batch_size, 2]
            targets: True start/end positions [batch_size, 2]
        
        Returns:
            Dictionary with exact match, F1, start/end accuracy
        """
        if predictions.dim() == 3:
            # Convert logits to positions
            start_logits = predictions[:, :, 0]
            end_logits = predictions[:, :, 1]
            
            start_preds = torch.argmax(start_logits, dim=-1)
            end_preds = torch.argmax(end_logits, dim=-1)
            
            pred_positions = torch.stack([start_preds, end_preds], dim=-1)
        else:
            pred_positions = predictions
        
        # Convert to numpy
        if isinstance(pred_positions, torch.Tensor):
            pred_positions = pred_positions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        metrics = {}
        
        # Exact match (both start and end correct)
        start_correct = pred_positions[:, 0] == targets[:, 0]
        end_correct = pred_positions[:, 1] == targets[:, 1]
        exact_match = start_correct & end_correct
        
        metrics['exact_match'] = float(np.mean(exact_match))
        metrics['start_accuracy'] = float(np.mean(start_correct))
        metrics['end_accuracy'] = float(np.mean(end_correct))
        
        # F1 score based on token overlap
        f1_scores = []
        for i in range(len(pred_positions)):
            pred_start, pred_end = pred_positions[i]
            true_start, true_end = targets[i]
            
            # Calculate overlap
            pred_tokens = set(range(pred_start, pred_end + 1))
            true_tokens = set(range(true_start, true_end + 1))
            
            if len(pred_tokens) == 0 and len(true_tokens) == 0:
                f1 = 1.0
            elif len(pred_tokens) == 0 or len(true_tokens) == 0:
                f1 = 0.0
            else:
                overlap = len(pred_tokens & true_tokens)
                precision = overlap / len(pred_tokens)
                recall = overlap / len(true_tokens)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
            
            f1_scores.append(f1)
        
        metrics['f1'] = float(np.mean(f1_scores))
        
        return metrics


class GenerationMetrics(MetricCalculator):
    """
    Metrics for text generation tasks (summarization, code generation, translation).
    """
    
    def __init__(self, task_name: str):
        super().__init__(task_name)
        
        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
        
        # Initialize BLEU scorer
        if BLEU_AVAILABLE:
            self.bleu_scorer = BLEU()
    
    def compute(
        self, 
        predictions: Union[List[str], torch.Tensor], 
        targets: Union[List[str], torch.Tensor],
        tokenizer=None
    ) -> Dict[str, float]:
        """
        Compute generation metrics.
        
        Args:
            predictions: Generated text or token IDs
            targets: Reference text or token IDs
            tokenizer: Tokenizer for decoding (if inputs are token IDs)
        
        Returns:
            Dictionary with ROUGE, BLEU, and other metrics
        """
        # Convert token IDs to text if necessary
        if isinstance(predictions, torch.Tensor):
            if tokenizer is None:
                raise ValueError("Tokenizer required for decoding token IDs")
            predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        if isinstance(targets, torch.Tensor):
            if tokenizer is None:
                raise ValueError("Tokenizer required for decoding token IDs")
            targets = tokenizer.batch_decode(targets, skip_special_tokens=True)
        
        metrics = {}
        
        # ROUGE metrics
        if ROUGE_AVAILABLE:
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, target in zip(predictions, targets):
                scores = self.rouge_scorer.score(target, pred)
                for rouge_type in rouge_scores:
                    rouge_scores[rouge_type].append(scores[rouge_type].fmeasure)
            
            for rouge_type, scores in rouge_scores.items():
                metrics[f'{rouge_type}_f1'] = float(np.mean(scores))
        
        # BLEU metrics
        if BLEU_AVAILABLE:
            try:
                # sacrebleu expects list of references for each prediction
                refs = [[target] for target in targets]
                bleu_score = self.bleu_scorer.corpus_score(predictions, refs)
                metrics['bleu'] = float(bleu_score.score)
            except Exception as e:
                logger.warning(f"BLEU calculation failed: {e}")
                metrics['bleu'] = 0.0
        
        # Length-based metrics
        pred_lengths = [len(pred.split()) for pred in predictions]
        target_lengths = [len(target.split()) for target in targets]
        
        metrics['avg_pred_length'] = float(np.mean(pred_lengths))
        metrics['avg_target_length'] = float(np.mean(target_lengths))
        metrics['length_ratio'] = float(np.mean(pred_lengths) / np.mean(target_lengths)) if np.mean(target_lengths) > 0 else 0.0
        
        # Exact match
        exact_matches = [pred.strip() == target.strip() for pred, target in zip(predictions, targets)]
        metrics['exact_match'] = float(np.mean(exact_matches))
        
        return metrics


class RoutingMetrics(MetricCalculator):
    """
    Metrics for routing decisions.
    """
    
    def __init__(self):
        super().__init__("routing")
    
    def compute(
        self, 
        routing_probs: torch.Tensor, 
        true_tasks: Optional[torch.Tensor] = None,
        task_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute routing metrics.
        
        Args:
            routing_probs: Routing probabilities [batch_size, num_tasks]
            true_tasks: True task labels [batch_size] (optional)
            task_names: List of task names (optional)
        
        Returns:
            Dictionary with routing metrics
        """
        if isinstance(routing_probs, torch.Tensor):
            routing_probs = routing_probs.cpu().numpy()
        
        metrics = {}
        
        # Routing entropy (measure of uncertainty)
        eps = 1e-8
        entropy = -np.sum(routing_probs * np.log(routing_probs + eps), axis=-1)
        metrics['routing_entropy'] = float(np.mean(entropy))
        metrics['routing_entropy_std'] = float(np.std(entropy))
        
        # Routing sparsity (fraction of low probabilities)
        sparsity_threshold = 0.1
        sparse_probs = (routing_probs < sparsity_threshold).astype(float)
        metrics['routing_sparsity'] = float(np.mean(sparse_probs))
        
        # Max routing probability (confidence)
        max_probs = np.max(routing_probs, axis=-1)
        metrics['max_routing_prob'] = float(np.mean(max_probs))
        metrics['max_routing_prob_std'] = float(np.std(max_probs))
        
        # Load balancing (how evenly distributed across tasks)
        task_usage = np.mean(routing_probs, axis=0)
        ideal_usage = 1.0 / routing_probs.shape[1]
        load_balance_score = 1.0 - np.var(task_usage) / (ideal_usage ** 2)
        metrics['load_balance_score'] = float(max(0.0, load_balance_score))
        
        # Per-task usage
        if task_names is not None:
            for i, task_name in enumerate(task_names):
                metrics[f'{task_name}_usage'] = float(task_usage[i])
        else:
            for i in range(len(task_usage)):
                metrics[f'task_{i}_usage'] = float(task_usage[i])
        
        # Routing accuracy (if true tasks provided)
        if true_tasks is not None:
            if isinstance(true_tasks, torch.Tensor):
                true_tasks = true_tasks.cpu().numpy()
            
            predicted_tasks = np.argmax(routing_probs, axis=-1)
            routing_accuracy = np.mean(predicted_tasks == true_tasks)
            metrics['routing_accuracy'] = float(routing_accuracy)
            
            # Per-task routing accuracy
            if task_names is not None:
                for i, task_name in enumerate(task_names):
                    task_mask = true_tasks == i
                    if np.sum(task_mask) > 0:
                        task_acc = np.mean(predicted_tasks[task_mask] == i)
                        metrics[f'{task_name}_routing_accuracy'] = float(task_acc)
        
        return metrics


class DynamoEvaluator:
    """
    Main evaluator for DYNAMO model and baselines.
    """
    
    def __init__(self, task_names: List[str], tokenizer=None):
        """
        Initialize DYNAMO evaluator.
        
        Args:
            task_names: List of task names
            tokenizer: Tokenizer for decoding (optional)
        """
        self.task_names = task_names
        self.tokenizer = tokenizer
        
        # Initialize task-specific metric calculators
        self.metric_calculators = {}
        
        for task_name in task_names:
            if task_name == "sentiment":
                self.metric_calculators[task_name] = SentimentMetrics()
            elif task_name == "qa":
                self.metric_calculators[task_name] = QAMetrics()
            elif task_name in ["summarization", "code_generation", "translation"]:
                self.metric_calculators[task_name] = GenerationMetrics(task_name)
            else:
                logger.warning(f"No specific metrics for task: {task_name}")
        
        # Routing metrics
        self.routing_metrics = RoutingMetrics()
        
        logger.info(f"DYNAMO evaluator initialized for tasks: {task_names}")
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader
            device: Device to run evaluation on
        
        Returns:
            Dictionary with evaluation results
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.eval()
        model.to(device)
        
        # Collect predictions and targets
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)
        all_routing_probs = []
        all_true_tasks = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    task_labels=batch.get('task_labels'),
                    return_routing_info=True
                )
                
                # Collect task outputs
                if 'task_outputs' in outputs:
                    for task_name, task_output in outputs['task_outputs'].items():
                        all_predictions[task_name].append(task_output.cpu())
                        
                        # Collect targets if available
                        if task_name in batch or 'target' in batch:
                            target = batch.get(task_name, batch.get('target'))
                            if target is not None:
                                all_targets[task_name].append(target.cpu())
                
                # Collect routing information
                if 'routing_probs' in outputs:
                    all_routing_probs.append(outputs['routing_probs'].cpu())
                
                # Collect true task labels
                if 'task_labels' in batch:
                    task_labels = batch['task_labels']
                    if task_labels.dim() > 1:  # Multi-hot encoding
                        task_labels = torch.argmax(task_labels, dim=-1)
                    all_true_tasks.append(task_labels.cpu())
        
        # Compute metrics
        results = {}
        
        # Task-specific metrics
        for task_name in self.task_names:
            if task_name in all_predictions and task_name in all_targets:
                if all_predictions[task_name] and all_targets[task_name]:
                    # Concatenate predictions and targets
                    task_preds = torch.cat(all_predictions[task_name], dim=0)
                    task_targets = torch.cat(all_targets[task_name], dim=0)
                    
                    # Compute metrics
                    if task_name in self.metric_calculators:
                        task_metrics = self.metric_calculators[task_name].compute(
                            task_preds, task_targets
                        )
                        results[task_name] = task_metrics
        
        # Routing metrics
        if all_routing_probs:
            routing_probs = torch.cat(all_routing_probs, dim=0)
            true_tasks = torch.cat(all_true_tasks, dim=0) if all_true_tasks else None
            
            routing_metrics = self.routing_metrics.compute(
                routing_probs, true_tasks, self.task_names
            )
            results['routing'] = routing_metrics
        
        # Overall metrics
        results['overall'] = self._compute_overall_metrics(results)
        
        return results
    
    def _compute_overall_metrics(self, task_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute overall metrics across all tasks."""
        overall = {}
        
        # Average accuracy across tasks
        accuracies = []
        for task_name in self.task_names:
            if task_name in task_results:
                task_metrics = task_results[task_name]
                if 'accuracy' in task_metrics:
                    accuracies.append(task_metrics['accuracy'])
                elif 'exact_match' in task_metrics:
                    accuracies.append(task_metrics['exact_match'])
                elif 'rouge1_f1' in task_metrics:
                    accuracies.append(task_metrics['rouge1_f1'])
        
        if accuracies:
            overall['avg_accuracy'] = float(np.mean(accuracies))
            overall['std_accuracy'] = float(np.std(accuracies))
        
        # F1 scores
        f1_scores = []
        for task_name in self.task_names:
            if task_name in task_results:
                task_metrics = task_results[task_name]
                if 'f1' in task_metrics:
                    f1_scores.append(task_metrics['f1'])
        
        if f1_scores:
            overall['avg_f1'] = float(np.mean(f1_scores))
            overall['std_f1'] = float(np.std(f1_scores))
        
        return overall
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        dataloader: torch.utils.data.DataLoader,
        device: torch.device = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: Dictionary of model name to model
            dataloader: Data loader
            device: Device to run evaluation on
        
        Returns:
            Dictionary with results for each model
        """
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            model_results = self.evaluate_model(model, dataloader, device)
            results[model_name] = model_results
        
        return results


def create_evaluator(task_names: List[str], tokenizer=None) -> DynamoEvaluator:
    """
    Create a DYNAMO evaluator.
    
    Args:
        task_names: List of task names
        tokenizer: Tokenizer for decoding
    
    Returns:
        DYNAMO evaluator
    """
    return DynamoEvaluator(task_names, tokenizer)

