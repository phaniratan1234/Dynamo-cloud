"""
Example usage script for DYNAMO model.
Demonstrates how to load a trained model and use it for inference.
"""

import torch
from transformers import RobertaTokenizer
from typing import List, Dict, Any

from model import DynamoModel
from utils import Config, get_config, get_logger
from evaluation import create_evaluator, create_routing_analyzer

logger = get_logger(__name__)


class DynamoInference:
    """
    Inference wrapper for DYNAMO model.
    """
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize DYNAMO inference.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path:
            self.config = get_config(config_path)
        else:
            # Try to load config from model directory
            import os
            config_file = os.path.join(model_path, "config.json")
            if os.path.exists(config_file):
                self.config = Config.load(config_file)
            else:
                raise ValueError("Configuration file not found")
        
        # Initialize model
        self.model = DynamoModel(self.config.__dict__)
        
        # Load trained weights
        self.model.load_model(model_path)
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.config.model.base_model_name
        )
        
        # Set device
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        logger.info(f"DYNAMO model loaded from {model_path}")
    
    def predict(
        self, 
        text: str, 
        task_hint: str = None,
        return_routing_info: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction on input text.
        
        Args:
            text: Input text
            task_hint: Optional task hint for oracle routing
            return_routing_info: Whether to return routing information
        
        Returns:
            Dictionary with predictions and optional routing info
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.data.max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prepare task labels if hint provided
        task_labels = None
        if task_hint and task_hint in self.model.task_to_idx:
            task_idx = self.model.task_to_idx[task_hint]
            task_labels = torch.tensor([task_idx], device=self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task_labels=task_labels,
                return_routing_info=True
            )
        
        # Process outputs
        results = {}
        
        # Task predictions
        if 'task_outputs' in outputs:
            for task_name, task_output in outputs['task_outputs'].items():
                results[task_name] = self._process_task_output(
                    task_output, task_name
                )
        
        # Routing information
        if return_routing_info and 'routing_probs' in outputs:
            routing_probs = outputs['routing_probs'][0].cpu().numpy()
            
            results['routing_info'] = {
                'probabilities': {
                    task: float(prob) 
                    for task, prob in zip(self.model.task_names, routing_probs)
                },
                'predicted_task': self.model.task_names[routing_probs.argmax()],
                'confidence': float(routing_probs.max()),
                'entropy': float(-sum(p * torch.log(torch.tensor(p + 1e-8)) 
                                    for p in routing_probs))
            }
        
        return results
    
    def _process_task_output(self, output: torch.Tensor, task_name: str) -> Dict[str, Any]:
        """Process task-specific output."""
        output = output[0].cpu()  # Remove batch dimension
        
        if task_name == "sentiment":
            # Classification output
            probs = torch.softmax(output, dim=-1)
            predicted_class = torch.argmax(probs).item()
            
            return {
                'predicted_class': predicted_class,
                'probabilities': probs.tolist(),
                'label': 'positive' if predicted_class == 1 else 'negative',
                'confidence': float(probs.max())
            }
        
        elif task_name == "qa":
            # Question answering output
            if output.dim() == 2:  # [seq_len, 2]
                start_logits = output[:, 0]
                end_logits = output[:, 1]
                
                start_pos = torch.argmax(start_logits).item()
                end_pos = torch.argmax(end_logits).item()
                
                return {
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'start_confidence': float(torch.softmax(start_logits, dim=-1).max()),
                    'end_confidence': float(torch.softmax(end_logits, dim=-1).max())
                }
            else:
                return {'positions': output.tolist()}
        
        else:
            # Generation tasks (simplified)
            return {
                'logits': output.tolist(),
                'representation': output.mean().item()
            }
    
    def batch_predict(
        self, 
        texts: List[str], 
        task_hints: List[str] = None,
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on a batch of texts.
        
        Args:
            texts: List of input texts
            task_hints: Optional list of task hints
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_hints = task_hints[i:i + batch_size] if task_hints else [None] * len(batch_texts)
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.config.data.max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prepare task labels
            task_labels = None
            if any(hint is not None for hint in batch_hints):
                task_labels = []
                for hint in batch_hints:
                    if hint and hint in self.model.task_to_idx:
                        task_labels.append(self.model.task_to_idx[hint])
                    else:
                        task_labels.append(0)  # Default to first task
                task_labels = torch.tensor(task_labels, device=self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    task_labels=task_labels,
                    return_routing_info=True
                )
            
            # Process batch outputs
            batch_size_actual = inputs['input_ids'].size(0)
            
            for j in range(batch_size_actual):
                result = {}
                
                # Task predictions
                if 'task_outputs' in outputs:
                    for task_name, task_output in outputs['task_outputs'].items():
                        result[task_name] = self._process_task_output(
                            task_output[j:j+1], task_name
                        )
                
                # Routing information
                if 'routing_probs' in outputs:
                    routing_probs = outputs['routing_probs'][j].cpu().numpy()
                    
                    result['routing_info'] = {
                        'probabilities': {
                            task: float(prob) 
                            for task, prob in zip(self.model.task_names, routing_probs)
                        },
                        'predicted_task': self.model.task_names[routing_probs.argmax()],
                        'confidence': float(routing_probs.max())
                    }
                
                results.append(result)
        
        return results


def example_sentiment_analysis():
    """Example: Sentiment analysis."""
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS EXAMPLE")
    print("="*50)
    
    # Initialize model (adjust path to your trained model)
    model_path = "./checkpoints/final_model"
    config_path = "./config.yaml"
    
    try:
        dynamo = DynamoInference(model_path, config_path)
        
        # Test sentences
        sentences = [
            "I love this movie! It's absolutely fantastic.",
            "This is the worst film I've ever seen.",
            "The movie was okay, nothing special.",
            "Amazing cinematography and brilliant acting!"
        ]
        
        print("Analyzing sentiment...")
        for sentence in sentences:
            result = dynamo.predict(
                sentence, 
                task_hint="sentiment",
                return_routing_info=True
            )
            
            sentiment_result = result.get('sentiment', {})
            routing_info = result.get('routing_info', {})
            
            print(f"\nText: {sentence}")
            print(f"Sentiment: {sentiment_result.get('label', 'unknown')}")
            print(f"Confidence: {sentiment_result.get('confidence', 0):.3f}")
            print(f"Router predicted task: {routing_info.get('predicted_task', 'unknown')}")
            print(f"Router confidence: {routing_info.get('confidence', 0):.3f}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a trained model at the specified path.")


def example_question_answering():
    """Example: Question answering."""
    print("\n" + "="*50)
    print("QUESTION ANSWERING EXAMPLE")
    print("="*50)
    
    model_path = "./checkpoints/final_model"
    config_path = "./config.yaml"
    
    try:
        dynamo = DynamoInference(model_path, config_path)
        
        # QA examples
        contexts = [
            "The capital of France is Paris. It is known for the Eiffel Tower.",
            "Python is a programming language created by Guido van Rossum.",
        ]
        
        questions = [
            "What is the capital of France?",
            "Who created Python?",
        ]
        
        print("Answering questions...")
        for context, question in zip(contexts, questions):
            # Combine context and question
            qa_input = f"Question: {question} Context: {context}"
            
            result = dynamo.predict(
                qa_input,
                task_hint="qa",
                return_routing_info=True
            )
            
            qa_result = result.get('qa', {})
            routing_info = result.get('routing_info', {})
            
            print(f"\nQuestion: {question}")
            print(f"Context: {context}")
            print(f"Start position: {qa_result.get('start_position', 'unknown')}")
            print(f"End position: {qa_result.get('end_position', 'unknown')}")
            print(f"Router predicted task: {routing_info.get('predicted_task', 'unknown')}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a trained model at the specified path.")


def example_multi_task_inference():
    """Example: Multi-task inference without task hints."""
    print("\n" + "="*50)
    print("MULTI-TASK INFERENCE EXAMPLE")
    print("="*50)
    
    model_path = "./checkpoints/final_model"
    config_path = "./config.yaml"
    
    try:
        dynamo = DynamoInference(model_path, config_path)
        
        # Mixed inputs
        inputs = [
            "This movie is absolutely terrible!",  # Sentiment
            "What is the capital of Germany?",     # QA
            "Summarize this article about AI.",    # Summarization
            "Write a Python function to sort a list.",  # Code generation
            "Translate 'Hello world' to French."   # Translation
        ]
        
        print("Processing mixed inputs...")
        results = dynamo.batch_predict(inputs, batch_size=2)
        
        for i, (text, result) in enumerate(zip(inputs, results)):
            routing_info = result.get('routing_info', {})
            
            print(f"\nInput {i+1}: {text}")
            print(f"Router prediction: {routing_info.get('predicted_task', 'unknown')}")
            print(f"Confidence: {routing_info.get('confidence', 0):.3f}")
            print(f"Task probabilities:")
            
            for task, prob in routing_info.get('probabilities', {}).items():
                print(f"  {task}: {prob:.3f}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a trained model at the specified path.")


def example_routing_analysis():
    """Example: Analyze routing patterns."""
    print("\n" + "="*50)
    print("ROUTING ANALYSIS EXAMPLE")
    print("="*50)
    
    model_path = "./checkpoints/final_model"
    config_path = "./config.yaml"
    
    try:
        # Load model
        config = get_config(config_path)
        model = DynamoModel(config.__dict__)
        model.load_model(model_path)
        
        # Create analyzer
        analyzer = create_routing_analyzer(model, model.task_names)
        
        # Create some sample data for analysis
        sample_texts = [
            "I love this product!",
            "What is machine learning?",
            "Please summarize this document.",
            "def hello_world():",
            "Translate this to Spanish."
        ] * 20  # Repeat for more samples
        
        # Create a simple dataloader (simplified)
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import RobertaTokenizer
        
        tokenizer = RobertaTokenizer.from_pretrained(config.model.base_model_name)
        
        # Tokenize
        inputs = tokenizer(
            sample_texts,
            return_tensors="pt",
            max_length=config.data.max_length,
            truncation=True,
            padding=True
        )
        
        # Create dataset
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=8)
        
        # Analyze routing
        print("Analyzing routing patterns...")
        analysis_results = analyzer.analyze_routing_decisions(dataloader, max_samples=100)
        
        # Print summary
        if 'routing_distribution' in analysis_results:
            dist = analysis_results['routing_distribution']
            print("\nTask usage distribution:")
            for task, usage in dist.get('task_usage', {}).items():
                print(f"  {task}: {usage['percentage']:.1f}%")
        
        if 'routing_patterns' in analysis_results:
            patterns = analysis_results['routing_patterns']
            if 'confidence' in patterns:
                conf = patterns['confidence']
                print(f"\nRouting confidence: {conf['mean']:.3f} ± {conf['std']:.3f}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        analyzer.visualize_routing_patterns("./analysis_output")
        
        print("Analysis completed! Check ./analysis_output for visualizations.")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a trained model at the specified path.")


def main():
    """Run all examples."""
    print("DYNAMO Model Usage Examples")
    print("="*50)
    
    # Run examples
    example_sentiment_analysis()
    example_question_answering()
    example_multi_task_inference()
    example_routing_analysis()
    
    print("\n" + "="*50)
    print("All examples completed!")
    print("="*50)


if __name__ == "__main__":
    main()

