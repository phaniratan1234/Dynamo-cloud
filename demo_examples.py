"""
DYNAMO Model Demo Examples
Quick demonstration of model capabilities across all tasks.
"""

from inference import load_trained_model, run_inference

def demo_examples():
    """Run demonstration with example queries showcasing smart routing."""
    
    print("🎮 DYNAMO SMART ROUTING DEMO")
    print("=" * 50)
    print("🤖 Watch DYNAMO automatically detect tasks and route to the best adapters!")
    print("✨ No need to specify tasks - just natural queries\n")
    
    # Load model
    model, config_dict = load_trained_model()
    
    # Example queries that showcase automatic task detection
    examples = [
        {
            "query": "This movie was absolutely fantastic! Great acting and storyline.",
            "expected_task": "sentiment",
            "description": "Movie review → Should route to sentiment analysis"
        },
        {
            "query": "What is the capital of France?",
            "expected_task": "qa", 
            "description": "Direct question → Should route to question answering"
        },
        {
            "query": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Through algorithms, machine learning uses historical data as input to predict new output values. The field has become increasingly important in recent years.",
            "expected_task": "summarization",
            "description": "Long technical text → Should route to summarization"
        },
        {
            "query": "Write a Python function to calculate fibonacci numbers",
            "expected_task": "code_generation",
            "description": "Programming request → Should route to code generation"
        },
        {
            "query": "Bonjour, comment allez-vous aujourd'hui?",
            "expected_task": "translation",
            "description": "French text → Should route to translation"
        }
    ]
    
    results = []
    correct_routes = 0
    
    print("🧪 Testing Smart Routing Capability...")
    print("-" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"\n🔸 Test {i}: {example['description']}")
        print(f"Query: {example['query'][:80]}{'...' if len(example['query']) > 80 else ''}")
        
        # Run inference with NO task specified - pure smart routing!
        result = run_inference(model, config_dict, example['query'], task=None)
        
        # Check if routing was correct
        if result['routing']:
            predicted_task = result['routing']['predicted_task']
            confidence = result['routing']['confidence']
            
            correct = predicted_task == example['expected_task']
            if correct:
                correct_routes += 1
                print(f"✅ Correct! Router chose: {predicted_task} ({confidence:.3f})")
            else:
                print(f"❌ Expected: {example['expected_task']}, Got: {predicted_task} ({confidence:.3f})")
            
            # Show confidence for all tasks
            print("   📊 All routing scores:")
            for task_name, score in result['routing']['all_scores'].items():
                indicator = "👑" if task_name == predicted_task else "  "
                expected = "🎯" if task_name == example['expected_task'] else "  "
                print(f"     {indicator}{expected} {task_name}: {score:.3f}")
        else:
            print("❌ No routing information available")
        
        print(f"   🎯 Prediction: {result['prediction'][:100]}{'...' if len(result['prediction']) > 100 else ''}")
        
        results.append(result)
        print("-" * 30)
    
    # Final Smart Routing Assessment
    print(f"\n🏆 SMART ROUTING PERFORMANCE")
    print("=" * 40)
    print(f"📊 Routing Accuracy: {correct_routes}/{len(examples)} ({correct_routes/len(examples)*100:.1f}%)")
    
    if correct_routes == len(examples):
        print("🎉 PERFECT! DYNAMO correctly identified all tasks!")
        print("✨ Users never need to specify task types")
    elif correct_routes >= len(examples) * 0.8:
        print("🎯 EXCELLENT! DYNAMO shows strong task detection")
        print("✨ Smart routing working effectively")
    elif correct_routes >= len(examples) * 0.6:
        print("👍 GOOD! DYNAMO shows decent task detection")
        print("🔧 Consider Phase 3 training for improvement")
    else:
        print("⚠️  Router needs more training")
        print("🔧 Recommend completing all training phases")
    
    print(f"\n💡 Key Benefits Demonstrated:")
    print(f"  ✅ Zero task specification required")
    print(f"  ✅ Automatic intelligent routing")
    print(f"  ✅ Multi-task capabilities in one model")
    print(f"  ✅ Confidence scores for transparency")
    print(f"  ✅ Handles diverse query types naturally")
    
    return results

if __name__ == "__main__":
    print("🚀 DYNAMO SMART ROUTING DEMONSTRATION")
    print("This demo shows DYNAMO's key advantage: automatic task detection!")
    print("No manual task specification needed - just natural language queries.\n")
    
    demo_examples() 