#!/usr/bin/env python3
"""
Test script to demonstrate DYNAMO's multi-task routing capabilities.
Shows how different routing modes handle queries with multiple task components.
"""

import torch
import numpy as np

# Simulate DYNAMO routing behavior
def simulate_router_output(query: str, temperature: float = 1.0):
    """Simulate router output for different query types."""
    
    # Task keywords and their typical routing patterns
    task_patterns = {
        'sentiment': ['feel', 'emotion', 'positive', 'negative', 'opinion', 'review'],
        'qa': ['what', 'how', 'why', 'when', 'where', 'which', 'explain', '?'],
        'summarization': ['summarize', 'summary', 'brief', 'overview', 'tl;dr'],
        'code_generation': ['def ', 'function', 'code', 'python', 'program', 'algorithm'],
        'translation': ['translate', 'german', 'french', 'spanish', 'language']
    }
    
    # Calculate base probabilities based on keyword presence
    base_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # Start equal
    query_lower = query.lower()
    
    for i, (task, keywords) in enumerate(task_patterns.items()):
        for keyword in keywords:
            if keyword in query_lower:
                base_probs[i] += 0.3
    
    # Apply temperature and softmax
    logits = np.log(base_probs + 1e-8) / temperature
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    return probs

def demonstrate_routing_modes():
    """Demonstrate different routing modes on multi-task queries."""
    
    test_queries = [
        "Summarize this code: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "What is the sentiment of this summary: The movie was terrible and boring",
        "Translate this to German and explain why: I love programming",
        "Generate Python code that answers: How to sort a list?",
        "Is this code correct and what does it do: print('hello')",
    ]
    
    task_names = ['sentiment', 'qa', 'summarization', 'code_generation', 'translation']
    
    print("🤖 DYNAMO Multi-Task Routing Demonstration")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 60)
        
        # Test different routing modes
        modes = [
            ("Soft Routing (Default)", 1.0, False),
            ("Soft + Low Temp", 0.3, False), 
            ("Hard Routing", 1.0, True)
        ]
        
        for mode_name, temp, hard in modes:
            probs = simulate_router_output(query, temp)
            
            print(f"\n📊 {mode_name}:")
            print("   Task Probabilities:")
            
            for j, (task, prob) in enumerate(zip(task_names, probs)):
                bar = "█" * int(prob * 20)
                print(f"   {task:15} {prob:5.2f} |{bar}")
            
            if hard:
                selected = task_names[np.argmax(probs)]
                print(f"   → HARD: Uses only {selected} adapter")
            else:
                # Show which adapters contribute significantly
                significant = [(task, prob) for task, prob in zip(task_names, probs) if prob > 0.1]
                contributors = [f"{task} ({prob:.1f})" for task, prob in significant]
                print(f"   → SOFT: Combines {', '.join(contributors)}")
        
        print()

def show_multitask_benefits():
    """Show why soft routing is better for multi-task queries."""
    
    print("\n🎯 Why Soft Routing Handles Multi-Task Better:")
    print("=" * 60)
    
    examples = [
        {
            'query': "Summarize this Python code: def sort_list(arr):",
            'hard_routing': "Only summarization adapter → May not understand Python syntax",
            'soft_routing': "Summarization (60%) + Code (35%) → Understands both Python AND summarization"
        },
        {
            'query': "What's the sentiment of this summary: The plot was boring",
            'hard_routing': "Only sentiment adapter → May miss context that it's about a summary",
            'soft_routing': "Sentiment (70%) + Summarization (25%) → Better contextual understanding"
        },
        {
            'query': "Generate code to answer: How to reverse a string?",
            'hard_routing': "Only code generation → May miss the Q&A aspect", 
            'soft_routing': "Code Gen (60%) + QA (30%) → Addresses both question format AND code needs"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Query: '{example['query']}'")
        print(f"   🔸 Hard Routing: {example['hard_routing']}")
        print(f"   ✅ Soft Routing: {example['soft_routing']}")

if __name__ == "__main__":
    demonstrate_routing_modes()
    show_multitask_benefits()
    
    print("\n💡 Recommendations:")
    print("=" * 40)
    print("• Use Soft Routing (default) for complex, multi-faceted queries")
    print("• Use Hard Routing for simple, single-task queries") 
    print("• Lower temperature = more focused, higher = more balanced")
    print("• Phase 2 training teaches optimal routing patterns") 