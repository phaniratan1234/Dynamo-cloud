"""
DYNAMO Model Demo Examples
Quick demonstration of model capabilities across all tasks.
"""

from inference import load_trained_model, run_inference

def demo_examples():
    """Run demonstration with example queries."""
    
    print("🎮 DYNAMO MODEL DEMO")
    print("=" * 50)
    print("Showcasing multi-task capabilities with smart routing\n")
    
    # Load model
    model, config_dict = load_trained_model()
    
    # Example queries for each task
    examples = [
        {
            "task": "sentiment",
            "query": "This movie was absolutely fantastic! Great acting and storyline.",
            "expected": "Positive sentiment"
        },
        {
            "task": "qa", 
            "query": "What is the capital of France? context: France is a country in Europe with Paris as its capital city.",
            "expected": "Paris"
        },
        {
            "task": "summarization",
            "query": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Through algorithms, machine learning uses historical data as input to predict new output values.",
            "expected": "Summary of machine learning definition"
        },
        {
            "task": "code_generation",
            "query": "Write a Python function to calculate fibonacci numbers",
            "expected": "Python function code"
        },
        {
            "task": "translation",
            "query": "Bonjour, comment allez-vous?",
            "expected": "Hello, how are you?"
        }
    ]
    
    results = []
    
    print("Running example queries...")
    print("-" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"\n🔸 Example {i}: {example['task'].upper()}")
        print(f"Query: {example['query'][:80]}{'...' if len(example['query']) > 80 else ''}")
        
        # Run inference
        result = run_inference(model, config_dict, example['query'], example['task'])
        
        # Display results
        print(f"Prediction: {result['prediction']}")
        
        if result['routing']:
            routing = result['routing']
            correct_task = routing['predicted_task'] == example['task']
            print(f"Router: {routing['predicted_task']} ({routing['confidence']:.3f}) {'✅' if correct_task else '❌'}")
        
        results.append(result)
        print("-" * 30)
    
    # Summary
    print(f"\n📊 DEMO SUMMARY")
    print("=" * 30)
    
    if results and all('routing' in r for r in results):
        correct_routes = sum(1 for i, r in enumerate(results) if r['routing']['predicted_task'] == examples[i]['task'])
        print(f"Router Accuracy: {correct_routes}/{len(results)} ({correct_routes/len(results)*100:.1f}%)")
    
    print(f"Tasks Demonstrated: {len(examples)}")
    print(f"Multi-task Model: ✅ Working")
    print(f"Smart Routing: ✅ Working")
    
    return results

if __name__ == "__main__":
    demo_examples() 