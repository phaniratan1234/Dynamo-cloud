"""
DYNAMO Model Inference Script
Interactive inference for text queries across multiple tasks.
"""

import torch
import argparse
import json
from typing import Dict, Any, List
from pathlib import Path

def load_trained_model():
    """Load the best trained DYNAMO model."""
    from train_optimized import load_optimized_config
    from model import DynamoModel
    
    # Load configuration
    config_dict, config = load_optimized_config("config.yaml")
    
    # Initialize model
    model = DynamoModel(config_dict)
    
    # Try to load the best checkpoint (Phase 3 > Phase 2 > Phase 1)
    checkpoint_paths = [
        "checkpoints/phase3/best_joint_model.pt",
        "checkpoints/phase2/phase2_model.pt",
        "checkpoints/phase1/phase1_model.pt"
    ]
    
    loaded_checkpoint = None
    for path in checkpoint_paths:
        if Path(path).exists():
            print(f"📂 Loading checkpoint: {path}")
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            loaded_checkpoint = path
            break
    
    if loaded_checkpoint is None:
        print("⚠️  No trained checkpoints found. Using randomly initialized model.")
        print("   Please train the model first using: python train_optimized.py --phase 1")
    else:
        print(f"✅ Loaded trained model from: {loaded_checkpoint}")
    
    model.eval()
    return model, config_dict

def detect_task_from_query(query: str) -> str:
    """
    Automatically detect the most likely task from the query.
    This is a simple heuristic-based approach.
    """
    query_lower = query.lower()
    
    # Question answering indicators
    qa_indicators = ['what', 'how', 'when', 'where', 'why', 'who', '?']
    if any(indicator in query_lower for indicator in qa_indicators):
        return 'qa'
    
    # Sentiment analysis indicators
    sentiment_indicators = ['feel', 'think', 'opinion', 'review', 'good', 'bad', 'great', 'terrible']
    if any(indicator in query_lower for indicator in sentiment_indicators):
        return 'sentiment'
    
    # Code generation indicators
    code_indicators = ['function', 'def', 'class', 'import', 'return', 'print', 'code', 'program']
    if any(indicator in query_lower for indicator in code_indicators):
        return 'code_generation'
    
    # Translation indicators
    translation_indicators = ['translate', 'spanish', 'french', 'german', 'chinese', 'japanese']
    if any(indicator in query_lower for indicator in translation_indicators):
        return 'translation'
    
    # Summarization indicators (default for longer texts)
    if len(query.split()) > 50:
        return 'summarization'
    
    # Default to sentiment for short text
    return 'sentiment'

def prepare_input(query: str, task: str, config_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Prepare input for the model based on task type."""
    from transformers import RobertaTokenizer
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_length = config_dict.get('model', {}).get('max_sequence_length', 512)
    
    if task == 'qa':
        # For QA, we need context. If not provided, use the query as both context and question
        if ' context: ' in query.lower():
            parts = query.split(' context: ', 1)
            question = parts[0].strip()
            context = parts[1].strip()
        else:
            question = query
            context = "Please provide context for this question."
        
        # Encode question and context
        encoding = tokenizer(
            question,
            context,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'task_name': 'qa'
        }
    
    elif task == 'translation':
        # Simple translation format
        encoding = tokenizer(
            f"Translate to English: {query}",
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'task_name': 'translation'
        }
    
    else:
        # For sentiment, summarization, code_generation
        encoding = tokenizer(
            query,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'task_name': task
        }

def decode_output(predictions: torch.Tensor, task: str) -> str:
    """Decode model predictions based on task type."""
    from transformers import RobertaTokenizer
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    if task == 'sentiment':
        # Binary classification: 0 = negative, 1 = positive
        prob = torch.softmax(predictions, dim=-1)
        pred_class = torch.argmax(predictions, dim=-1).item()
        confidence = prob[0, pred_class].item()
        
        sentiment = "Positive" if pred_class == 1 else "Negative"
        return f"{sentiment} (confidence: {confidence:.3f})"
    
    elif task in ['qa', 'summarization', 'code_generation', 'translation']:
        # Text generation tasks
        try:
            # Decode generated tokens
            generated_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
            return generated_text.strip()
        except:
            # Fallback for classification-style outputs
            return f"Generated output (raw logits shape: {predictions.shape})"
    
    else:
        return f"Unknown task output: {predictions.shape}"

def run_inference(model, config_dict: Dict[str, Any], query: str, task: str = None) -> Dict[str, Any]:
    """Run inference on a single query."""
    
    # Auto-detect task if not specified
    if task is None:
        task = detect_task_from_query(query)
        print(f"🔍 Auto-detected task: {task}")
    
    # Prepare input
    inputs = prepare_input(query, task, config_dict)
    
    # Run model inference
    with torch.no_grad():
        # Set model to the specified task mode
        if hasattr(model, 'set_training_phase'):
            model.set_training_phase("inference")
        
        # Forward pass
        outputs = model(inputs)
        
        # Get predictions and routing info
        predictions = outputs.get('predictions', outputs.get('logits'))
        routing_probs = outputs.get('routing_probs')
        
        # Decode output
        decoded_output = decode_output(predictions, task)
        
        # Get routing information
        routing_info = {}
        if routing_probs is not None:
            task_names = ['sentiment', 'qa', 'summarization', 'code_generation', 'translation']
            routing_probs_np = routing_probs[0].cpu().numpy()
            
            routing_info = {
                'predicted_task': task_names[routing_probs_np.argmax()],
                'confidence': float(routing_probs_np.max()),
                'all_scores': {task_names[i]: float(routing_probs_np[i]) for i in range(len(task_names))}
            }
    
    return {
        'query': query,
        'task': task,
        'prediction': decoded_output,
        'routing': routing_info
    }

def interactive_inference():
    """Run interactive inference session."""
    print("🚀 DYNAMO Interactive Inference")
    print("=" * 50)
    
    # Load model
    model, config_dict = load_trained_model()
    
    print("\n💡 Available tasks:")
    print("  - sentiment: Analyze text sentiment")
    print("  - qa: Answer questions (format: 'question context: your context')")
    print("  - summarization: Summarize long text")
    print("  - code_generation: Generate code")
    print("  - translation: Translate text")
    print("  - auto: Auto-detect task (default)")
    
    print("\n📝 Enter your queries (type 'quit' to exit):")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            query = input("\n🔤 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Check if user specified a task
            task = None
            if query.startswith('[') and ']' in query:
                task_end = query.index(']')
                task = query[1:task_end].strip()
                query = query[task_end+1:].strip()
            
            # Run inference
            result = run_inference(model, config_dict, query, task)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"   Task: {result['task']}")
            print(f"   Prediction: {result['prediction']}")
            
            if result['routing']:
                routing = result['routing']
                print(f"   Router prediction: {routing['predicted_task']} ({routing['confidence']:.3f})")
                
                print("   Task scores:")
                for task_name, score in routing['all_scores'].items():
                    print(f"     {task_name}: {score:.3f}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_inference(input_file: str, output_file: str, task: str = None):
    """Run batch inference on a file of queries."""
    print(f"📂 Running batch inference: {input_file} -> {output_file}")
    
    # Load model
    model, config_dict = load_trained_model()
    
    # Read input queries
    with open(input_file, 'r') as f:
        if input_file.endswith('.json'):
            queries = json.load(f)
        else:
            queries = [line.strip() for line in f if line.strip()]
    
    # Run inference on each query
    results = []
    for i, query in enumerate(queries):
        print(f"Processing {i+1}/{len(queries)}: {query[:50]}...")
        result = run_inference(model, config_dict, query, task)
        results.append(result)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved {len(results)} results to {output_file}")

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="DYNAMO Model Inference")
    parser.add_argument(
        "--mode",
        choices=['interactive', 'batch'],
        default='interactive',
        help="Inference mode"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file for batch inference"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file for batch inference"
    )
    parser.add_argument(
        "--task",
        choices=['sentiment', 'qa', 'summarization', 'code_generation', 'translation'],
        help="Force specific task (otherwise auto-detect)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query for quick inference"
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        model, config_dict = load_trained_model()
        result = run_inference(model, config_dict, args.query, args.task)
        
        print("\n📊 Result:")
        print(f"Query: {result['query']}")
        print(f"Task: {result['task']}")
        print(f"Prediction: {result['prediction']}")
        if result['routing']:
            print(f"Router: {result['routing']['predicted_task']} ({result['routing']['confidence']:.3f})")
    
    elif args.mode == 'batch':
        if not args.input or not args.output:
            print("❌ Batch mode requires --input and --output arguments")
            return
        batch_inference(args.input, args.output, args.task)
    
    else:
        # Interactive mode
        interactive_inference()

if __name__ == "__main__":
    main() 