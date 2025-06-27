"""
Router decision analyzer for DYNAMO.
Provides tools for visualizing and interpreting routing decisions and patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, Counter
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import os

from ..model import DynamoModel
from ..utils.logger import get_logger
from ..utils.helpers import move_to_device

logger = get_logger(__name__)


class RoutingAnalyzer:
    """
    Analyzer for DYNAMO routing decisions and patterns.
    """
    
    def __init__(self, model: DynamoModel, task_names: List[str]):
        """
        Initialize routing analyzer.
        
        Args:
            model: DYNAMO model
            task_names: List of task names
        """
        self.model = model
        self.task_names = task_names
        self.task_to_idx = {task: idx for idx, task in enumerate(task_names)}
        self.idx_to_task = {idx: task for task, idx in self.task_to_idx.items()}
        
        # Storage for analysis data
        self.routing_data = []
        self.embedding_data = []
        
        logger.info("Routing analyzer initialized")
    
    def analyze_routing_decisions(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device = None,
        max_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Analyze routing decisions on a dataset.
        
        Args:
            dataloader: Data loader
            device: Device to run analysis on
            max_samples: Maximum number of samples to analyze
        
        Returns:
            Dictionary with routing analysis results
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()
        self.model.to(device)
        
        # Collect routing data
        routing_decisions = []
        input_embeddings = []
        input_texts = []
        true_tasks = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= max_samples:
                    break
                
                batch = move_to_device(batch, device)
                
                # Get backbone embeddings
                backbone_outputs = self.model.backbone(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                cls_embeddings = backbone_outputs.last_hidden_state[:, 0, :]
                
                # Get routing decisions
                routing_output = self.model.router(
                    input_embeddings=cls_embeddings,
                    return_confidence=True,
                    return_entropy=True
                )
                
                # Store data
                routing_decisions.append(routing_output['routing_probs'].cpu())
                input_embeddings.append(cls_embeddings.cpu())
                
                # Get input texts if available
                if 'input_text' in batch:
                    input_texts.extend(batch['input_text'])
                else:
                    # Decode from token IDs
                    decoded_texts = self.model.backbone.decode(batch['input_ids'])
                    input_texts.extend(decoded_texts)
                
                # Get true task labels if available
                if 'task_labels' in batch:
                    task_labels = batch['task_labels']
                    if task_labels.dim() > 1:  # Multi-hot encoding
                        task_labels = torch.argmax(task_labels, dim=-1)
                    true_tasks.append(task_labels.cpu())
                elif 'task_id' in batch:
                    true_tasks.append(batch['task_id'].cpu())
                
                sample_count += batch['input_ids'].size(0)
        
        # Concatenate data
        routing_probs = torch.cat(routing_decisions, dim=0).numpy()
        embeddings = torch.cat(input_embeddings, dim=0).numpy()
        
        if true_tasks:
            true_task_labels = torch.cat(true_tasks, dim=0).numpy()
        else:
            true_task_labels = None
        
        # Perform analysis
        analysis_results = {
            'routing_distribution': self._analyze_routing_distribution(routing_probs),
            'routing_patterns': self._analyze_routing_patterns(routing_probs, true_task_labels),
            'embedding_analysis': self._analyze_embeddings(embeddings, routing_probs),
            'decision_confidence': self._analyze_decision_confidence(routing_probs),
            'task_confusion': self._analyze_task_confusion(routing_probs, true_task_labels),
            'input_analysis': self._analyze_input_patterns(input_texts, routing_probs)
        }
        
        # Store data for visualization
        self.routing_data = {
            'routing_probs': routing_probs,
            'embeddings': embeddings,
            'input_texts': input_texts[:len(routing_probs)],
            'true_tasks': true_task_labels
        }
        
        return analysis_results
    
    def _analyze_routing_distribution(self, routing_probs: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of routing probabilities."""
        results = {}
        
        # Overall statistics
        results['mean_probs'] = routing_probs.mean(axis=0).tolist()
        results['std_probs'] = routing_probs.std(axis=0).tolist()
        results['max_probs'] = routing_probs.max(axis=0).tolist()
        results['min_probs'] = routing_probs.min(axis=0).tolist()
        
        # Per-task usage
        predicted_tasks = np.argmax(routing_probs, axis=1)
        task_counts = Counter(predicted_tasks)
        
        results['task_usage'] = {}
        for task_idx, task_name in self.idx_to_task.items():
            count = task_counts.get(task_idx, 0)
            results['task_usage'][task_name] = {
                'count': count,
                'percentage': count / len(routing_probs) * 100
            }
        
        # Entropy analysis
        eps = 1e-8
        entropy = -np.sum(routing_probs * np.log(routing_probs + eps), axis=1)
        results['entropy'] = {
            'mean': float(entropy.mean()),
            'std': float(entropy.std()),
            'min': float(entropy.min()),
            'max': float(entropy.max())
        }
        
        # Sparsity analysis
        sparsity_threshold = 0.1
        sparse_decisions = (routing_probs < sparsity_threshold).sum(axis=1)
        results['sparsity'] = {
            'mean_sparse_tasks': float(sparse_decisions.mean()),
            'std_sparse_tasks': float(sparse_decisions.std())
        }
        
        return results
    
    def _analyze_routing_patterns(
        self, 
        routing_probs: np.ndarray, 
        true_tasks: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze routing patterns and accuracy."""
        results = {}
        
        predicted_tasks = np.argmax(routing_probs, axis=1)
        
        # Routing accuracy
        if true_tasks is not None:
            accuracy = np.mean(predicted_tasks == true_tasks)
            results['routing_accuracy'] = float(accuracy)
            
            # Per-task accuracy
            results['per_task_accuracy'] = {}
            for task_idx, task_name in self.idx_to_task.items():
                task_mask = true_tasks == task_idx
                if task_mask.sum() > 0:
                    task_accuracy = np.mean(predicted_tasks[task_mask] == task_idx)
                    results['per_task_accuracy'][task_name] = float(task_accuracy)
        
        # Confidence analysis
        max_probs = np.max(routing_probs, axis=1)
        results['confidence'] = {
            'mean': float(max_probs.mean()),
            'std': float(max_probs.std()),
            'high_confidence_threshold': 0.8,
            'high_confidence_percentage': float(np.mean(max_probs > 0.8) * 100)
        }
        
        # Multi-task routing (when multiple tasks have significant probability)
        multi_task_threshold = 0.3
        multi_task_decisions = (routing_probs > multi_task_threshold).sum(axis=1)
        results['multi_task_routing'] = {
            'mean_active_tasks': float(multi_task_decisions.mean()),
            'multi_task_percentage': float(np.mean(multi_task_decisions > 1) * 100)
        }
        
        return results
    
    def _analyze_embeddings(
        self, 
        embeddings: np.ndarray, 
        routing_probs: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze input embeddings and their relationship to routing."""
        results = {}
        
        # Dimensionality reduction
        if embeddings.shape[0] > 50:  # Only if we have enough samples
            # PCA
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)
            
            results['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'embeddings_2d': pca_embeddings.tolist()
            }
            
            # t-SNE (for smaller datasets)
            if embeddings.shape[0] <= 1000:
                tsne = TSNE(n_components=2, random_state=42)
                tsne_embeddings = tsne.fit_transform(embeddings)
                
                results['tsne'] = {
                    'embeddings_2d': tsne_embeddings.tolist()
                }
        
        # Clustering analysis
        predicted_tasks = np.argmax(routing_probs, axis=1)
        
        # K-means clustering on embeddings
        n_clusters = min(len(self.task_names), embeddings.shape[0] // 10)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Analyze cluster-task alignment
            cluster_task_alignment = {}
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if cluster_mask.sum() > 0:
                    cluster_tasks = predicted_tasks[cluster_mask]
                    most_common_task = Counter(cluster_tasks).most_common(1)[0]
                    
                    cluster_task_alignment[f'cluster_{cluster_id}'] = {
                        'most_common_task': self.idx_to_task[most_common_task[0]],
                        'task_purity': most_common_task[1] / cluster_mask.sum(),
                        'size': int(cluster_mask.sum())
                    }
            
            results['clustering'] = {
                'n_clusters': n_clusters,
                'cluster_task_alignment': cluster_task_alignment
            }
        
        return results
    
    def _analyze_decision_confidence(self, routing_probs: np.ndarray) -> Dict[str, Any]:
        """Analyze confidence in routing decisions."""
        results = {}
        
        # Maximum probability (confidence)
        max_probs = np.max(routing_probs, axis=1)
        
        # Confidence distribution
        confidence_bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        confidence_hist, _ = np.histogram(max_probs, bins=confidence_bins)
        
        results['confidence_distribution'] = {
            'bins': confidence_bins,
            'counts': confidence_hist.tolist(),
            'percentages': (confidence_hist / len(max_probs) * 100).tolist()
        }
        
        # Entropy vs confidence correlation
        eps = 1e-8
        entropy = -np.sum(routing_probs * np.log(routing_probs + eps), axis=1)
        
        correlation = np.corrcoef(max_probs, entropy)[0, 1]
        results['confidence_entropy_correlation'] = float(correlation)
        
        # Decision stability (how often the top choice changes)
        # This would require multiple forward passes with dropout, simplified here
        results['decision_stability'] = {
            'note': 'Requires multiple forward passes for proper analysis'
        }
        
        return results
    
    def _analyze_task_confusion(
        self, 
        routing_probs: np.ndarray, 
        true_tasks: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze confusion between different tasks."""
        results = {}
        
        if true_tasks is None:
            results['note'] = 'True task labels not available'
            return results
        
        predicted_tasks = np.argmax(routing_probs, axis=1)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_tasks, predicted_tasks)
        
        results['confusion_matrix'] = cm.tolist()
        results['task_names'] = self.task_names
        
        # Most confused task pairs
        confusion_pairs = []
        for i in range(len(self.task_names)):
            for j in range(len(self.task_names)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_task': self.task_names[i],
                        'predicted_task': self.task_names[j],
                        'count': int(cm[i, j]),
                        'percentage': float(cm[i, j] / cm[i].sum() * 100)
                    })
        
        # Sort by confusion count
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        results['top_confusions'] = confusion_pairs[:10]  # Top 10 confusions
        
        return results
    
    def _analyze_input_patterns(
        self, 
        input_texts: List[str], 
        routing_probs: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze patterns in input text that lead to specific routing decisions."""
        results = {}
        
        predicted_tasks = np.argmax(routing_probs, axis=1)
        
        # Analyze text length vs routing
        text_lengths = [len(text.split()) for text in input_texts]
        
        results['text_length_analysis'] = {}
        for task_idx, task_name in self.idx_to_task.items():
            task_mask = predicted_tasks == task_idx
            if task_mask.sum() > 0:
                task_lengths = np.array(text_lengths)[task_mask]
                results['text_length_analysis'][task_name] = {
                    'mean_length': float(task_lengths.mean()),
                    'std_length': float(task_lengths.std()),
                    'min_length': int(task_lengths.min()),
                    'max_length': int(task_lengths.max())
                }
        
        # Keyword analysis (simplified)
        results['keyword_analysis'] = {}
        for task_idx, task_name in self.idx_to_task.items():
            task_mask = predicted_tasks == task_idx
            if task_mask.sum() > 0:
                task_texts = [input_texts[i] for i in range(len(input_texts)) if task_mask[i]]
                
                # Simple word frequency analysis
                all_words = []
                for text in task_texts:
                    words = text.lower().split()
                    all_words.extend(words)
                
                word_counts = Counter(all_words)
                top_words = word_counts.most_common(10)
                
                results['keyword_analysis'][task_name] = {
                    'top_words': top_words,
                    'total_words': len(all_words),
                    'unique_words': len(word_counts)
                }
        
        return results
    
    def visualize_routing_patterns(
        self, 
        save_dir: str = "./visualizations",
        show_plots: bool = False
    ):
        """
        Create visualizations of routing patterns.
        
        Args:
            save_dir: Directory to save visualizations
            show_plots: Whether to display plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.routing_data:
            logger.warning("No routing data available. Run analyze_routing_decisions first.")
            return
        
        routing_probs = self.routing_data['routing_probs']
        embeddings = self.routing_data['embeddings']
        true_tasks = self.routing_data['true_tasks']
        
        # 1. Routing probability distribution
        plt.figure(figsize=(12, 8))
        
        # Heatmap of routing probabilities
        plt.subplot(2, 3, 1)
        sns.heatmap(routing_probs[:100].T, 
                   xticklabels=False, 
                   yticklabels=self.task_names,
                   cmap='viridis')
        plt.title('Routing Probabilities (First 100 samples)')
        plt.xlabel('Samples')
        plt.ylabel('Tasks')
        
        # Task usage distribution
        plt.subplot(2, 3, 2)
        predicted_tasks = np.argmax(routing_probs, axis=1)
        task_counts = [np.sum(predicted_tasks == i) for i in range(len(self.task_names))]
        
        plt.bar(self.task_names, task_counts)
        plt.title('Task Usage Distribution')
        plt.xlabel('Tasks')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Confidence distribution
        plt.subplot(2, 3, 3)
        max_probs = np.max(routing_probs, axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7)
        plt.title('Routing Confidence Distribution')
        plt.xlabel('Max Probability')
        plt.ylabel('Frequency')
        
        # Entropy distribution
        plt.subplot(2, 3, 4)
        eps = 1e-8
        entropy = -np.sum(routing_probs * np.log(routing_probs + eps), axis=1)
        plt.hist(entropy, bins=20, alpha=0.7)
        plt.title('Routing Entropy Distribution')
        plt.xlabel('Entropy')
        plt.ylabel('Frequency')
        
        # Confusion matrix (if true tasks available)
        if true_tasks is not None:
            plt.subplot(2, 3, 5)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_tasks, predicted_tasks)
            sns.heatmap(cm, annot=True, fmt='d', 
                       xticklabels=self.task_names,
                       yticklabels=self.task_names)
            plt.title('Routing Confusion Matrix')
            plt.xlabel('Predicted Task')
            plt.ylabel('True Task')
        
        # PCA visualization
        if embeddings.shape[0] > 50:
            plt.subplot(2, 3, 6)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)
            
            scatter = plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], 
                                c=predicted_tasks, cmap='tab10', alpha=0.6)
            plt.title('PCA of Input Embeddings')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter, label='Predicted Task')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'routing_analysis.png'), dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"Routing visualizations saved to {save_dir}")
    
    def generate_routing_report(
        self, 
        analysis_results: Dict[str, Any],
        save_path: str = "./routing_report.json"
    ):
        """
        Generate a comprehensive routing analysis report.
        
        Args:
            analysis_results: Results from analyze_routing_decisions
            save_path: Path to save the report
        """
        report = {
            'model_info': {
                'task_names': self.task_names,
                'num_tasks': len(self.task_names),
                'router_parameters': sum(p.numel() for p in self.model.router.parameters()),
                'temperature': self.model.router.get_temperature()
            },
            'analysis_results': analysis_results,
            'summary': self._generate_summary(analysis_results)
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Routing report saved to {save_path}")
        
        return report
    
    def _generate_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate a summary of the routing analysis."""
        summary = {}
        
        # Routing accuracy
        if 'routing_patterns' in analysis_results:
            patterns = analysis_results['routing_patterns']
            if 'routing_accuracy' in patterns:
                accuracy = patterns['routing_accuracy']
                summary['routing_accuracy'] = f"Router achieves {accuracy:.1%} accuracy"
        
        # Task usage balance
        if 'routing_distribution' in analysis_results:
            dist = analysis_results['routing_distribution']
            if 'task_usage' in dist:
                usage_percentages = [info['percentage'] for info in dist['task_usage'].values()]
                balance = 1 - np.std(usage_percentages) / np.mean(usage_percentages)
                summary['task_balance'] = f"Task usage balance score: {balance:.2f}"
        
        # Decision confidence
        if 'decision_confidence' in analysis_results:
            conf = analysis_results['decision_confidence']
            if 'confidence_distribution' in conf:
                high_conf_pct = conf['confidence_distribution']['percentages'][-1]  # Last bin
                summary['decision_confidence'] = f"{high_conf_pct:.1f}% of decisions are high confidence"
        
        return summary


def create_routing_analyzer(model: DynamoModel, task_names: List[str]) -> RoutingAnalyzer:
    """
    Create a routing analyzer.
    
    Args:
        model: DYNAMO model
        task_names: List of task names
    
    Returns:
        Routing analyzer
    """
    return RoutingAnalyzer(model, task_names)

