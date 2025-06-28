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
    """Run inference on a single query with smart routing."""
    
    # DYNAMO's key feature: Smart routing determines the task automatically
    # Task parameter is only for comparison/debugging purposes
    
    # Prepare input (we'll let the router decide the task)
    inputs = prepare_input_for_routing(query, config_dict)
    
    # Run model inference with smart routing
    with torch.no_grad():
        # Set model to inference mode
        if hasattr(model, 'set_training_phase'):
            model.set_training_phase("inference")
        
        # Forward pass - let the router decide!
        outputs = model(inputs)
        
        # Get routing decision
        routing_probs = outputs.get('routing_probs')
        if routing_probs is not None:
            task_names = ['sentiment', 'qa', 'summarization', 'code_generation', 'translation']
            routing_probs_np = routing_probs[0].cpu().numpy()
            predicted_task = task_names[routing_probs_np.argmax()]
            confidence = float(routing_probs_np.max())
            
            print(f"🤖 DYNAMO Router Decision: {predicted_task} (confidence: {confidence:.3f})")
        else:
            # Fallback to heuristic detection if routing fails
            predicted_task = detect_task_from_query(query) if task is None else task
            confidence = 0.5
            print(f"🔍 Fallback task detection: {predicted_task}")
        
        # Use router's decision (or provided task for comparison)
        final_task = task if task is not None else predicted_task
        
        # Get predictions for the determined task
        predictions = outputs.get('predictions', outputs.get('logits'))
        decoded_output = decode_output(predictions, final_task)
        
        # Routing information
        routing_info = {}
        if routing_probs is not None:
            routing_info = {
                'predicted_task': predicted_task,
                'confidence': confidence,
                'all_scores': {task_names[i]: float(routing_probs_np[i]) for i in range(len(task_names))},
                'user_override': task is not None
            }
    
    return {
        'query': query,
        'task': final_task,
        'predicted_task': predicted_task,
        'prediction': decoded_output,
        'routing': routing_info
    }

def prepare_input_for_routing(query: str, config_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Prepare input for smart routing (no task specified)."""
    from transformers import RobertaTokenizer
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_length = config_dict.get('model', {}).get('max_sequence_length', 512)
    
    # For smart routing, we encode the raw query and let the router decide
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
        'task_name': None  # Let the router decide!
    }

def interactive_inference():
    """Run interactive inference session with smart routing."""
    print("🚀 DYNAMO Interactive Inference")
    print("=" * 50)
    print("✨ Smart Multi-Task AI - Just type your query!")
    print("🤖 DYNAMO will automatically detect the task and route to the best adapter")
    print()
    
    # Load model
    model, config_dict = load_trained_model()
    
    print("💡 DYNAMO handles these tasks automatically:")
    print("  🎭 Sentiment Analysis - Express opinions, reviews")
    print("  ❓ Question Answering - Ask questions with context")
    print("  📝 Text Summarization - Provide long text to summarize")
    print("  💻 Code Generation - Request programming help")
    print("  🌍 Translation - Text in other languages")
    print()
    print("🎯 Optional: Force a specific task with [task] prefix:")
    print("   Example: [sentiment] This movie is great!")
    print()
    print("📝 Enter your queries (type 'quit' to exit):")
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
            
            # Check if user forced a specific task
            task_override = None
            if query.startswith('[') and ']' in query:
                task_end = query.index(']')
                task_override = query[1:task_end].strip()
                query = query[task_end+1:].strip()
                print(f"🎯 Task override: {task_override}")
            
            # Run inference with smart routing
            result = run_inference(model, config_dict, query, task_override)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"   Prediction: {result['prediction']}")
            
            if result['routing']:
                routing = result['routing']
                
                if routing.get('user_override'):
                    print(f"   📌 Used task: {result['task']} (overridden)")
                    print(f"   🤖 Router would have chosen: {routing['predicted_task']} ({routing['confidence']:.3f})")
                else:
                    print(f"   🤖 Router decision: {routing['predicted_task']} ({routing['confidence']:.3f})")
                
                print("   📊 All task scores:")
                for task_name, score in routing['all_scores'].items():
                    indicator = "👑" if task_name == routing['predicted_task'] else "  "
                    print(f"     {indicator} {task_name}: {score:.3f}")
        
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
    """Main inference function emphasizing smart routing."""
    parser = argparse.ArgumentParser(description="DYNAMO Smart Multi-Task Inference")
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
        "--force-task",
        choices=['sentiment', 'qa', 'summarization', 'code_generation', 'translation'],
        help="Override smart routing with specific task (for testing/comparison)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query for quick inference"
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        print("🚀 DYNAMO Smart Inference")
        print("=" * 40)
        
        model, config_dict = load_trained_model()
        result = run_inference(model, config_dict, args.query, args.force_task)
        
        print(f"\n📝 Query: {result['query']}")
        print(f"🎯 Result: {result['prediction']}")
        
        if result['routing']:
            routing = result['routing']
            if routing.get('user_override'):
                print(f"📌 Forced task: {result['task']}")
                print(f"🤖 Router suggestion: {routing['predicted_task']} ({routing['confidence']:.3f})")
            else:
                print(f"🤖 Smart routing: {routing['predicted_task']} ({routing['confidence']:.3f})")
    
    elif args.mode == 'batch':
        if not args.input or not args.output:
            print("❌ Batch mode requires --input and --output arguments")
            return
        batch_inference(args.input, args.output, args.force_task)
    
    else:
        # Interactive mode
        interactive_inference()

if __name__ == "__main__":
    main() 