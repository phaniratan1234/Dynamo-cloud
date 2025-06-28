"""
DYNAMO Performance Analysis & Metrics Report
Comprehensive analysis of model performance across all phases and tasks.
"""

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns
from datetime import datetime

def load_training_metrics():
    """Load training metrics from all phases."""
    metrics = {}
    
    # Phase 1 metrics (adapter training)
    phase1_dir = Path("checkpoints/phase1")
    if phase1_dir.exists():
        adapter_metrics = {}
        for task in ['sentiment', 'qa', 'summarization', 'code_generation', 'translation']:
            adapter_path = phase1_dir / f"{task}_adapter.pt"
            if adapter_path.exists():
                try:
                    checkpoint = torch.load(adapter_path, map_location='cpu')
                    adapter_metrics[task] = {
                        'epoch': checkpoint.get('epoch', 'N/A'),
                        'loss': checkpoint.get('loss', 'N/A'),
                        'file_size_mb': adapter_path.stat().st_size / (1024*1024)
                    }
                except Exception as e:
                    adapter_metrics[task] = {'error': str(e)}
        
        metrics['phase1'] = {
            'adapters': adapter_metrics,
            'status': 'completed' if len(adapter_metrics) == 5 else 'incomplete'
        }
    
    # Phase 2 metrics (router training)
    phase2_dir = Path("checkpoints/phase2")
    if phase2_dir.exists():
        metrics_path = phase2_dir / "training_metrics.pt"
        if metrics_path.exists():
            try:
                phase2_metrics = torch.load(metrics_path, map_location='cpu')
                metrics['phase2'] = phase2_metrics
            except Exception as e:
                metrics['phase2'] = {'error': str(e)}
    
    # Phase 3 metrics (joint training)
    phase3_dir = Path("checkpoints/phase3")
    if phase3_dir.exists():
        metrics_path = phase3_dir / "training_metrics.pt"
        if metrics_path.exists():
            try:
                phase3_metrics = torch.load(metrics_path, map_location='cpu')
                metrics['phase3'] = phase3_metrics
            except Exception as e:
                metrics['phase3'] = {'error': str(e)}
    
    return metrics

def analyze_model_architecture():
    """Analyze the model architecture and parameter counts."""
    try:
        from train_optimized import load_optimized_config
        from model import DynamoModel
        
        config_dict, config = load_optimized_config("config.yaml")
        model = DynamoModel(config_dict)
        
        # Calculate parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Adapter-specific counts
        adapter_params = {}
        for task_name, adapter in model.adapters.adapters.items():
            adapter_params[task_name] = sum(p.numel() for p in adapter.parameters())
        
        # Router parameters
        router_params = sum(p.numel() for p in model.router.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': total_params - trainable_params,
            'router_parameters': router_params,
            'adapter_parameters': adapter_params,
            'total_adapter_parameters': sum(adapter_params.values()),
            'model_size_mb': total_params * 4 / (1024*1024),  # Assuming float32
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_efficiency_metrics():
    """Calculate model efficiency metrics."""
    metrics = {}
    
    # Training efficiency
    phase2_dir = Path("checkpoints/phase2")
    if phase2_dir.exists():
        router_size = (phase2_dir / "best_router.pt").stat().st_size / (1024*1024)
        model_size = (phase2_dir / "phase2_model.pt").stat().st_size / (1024*1024)
        
        metrics['checkpoint_sizes'] = {
            'router_mb': router_size,
            'full_model_mb': model_size,
            'compression_ratio': model_size / router_size if router_size > 0 else 0
        }
    
    # Memory efficiency
    architecture = analyze_model_architecture()
    if 'total_parameters' in architecture:
        metrics['parameter_efficiency'] = {
            'parameters_per_task': architecture['total_adapter_parameters'] / 5,
            'router_to_adapter_ratio': architecture['router_parameters'] / architecture['total_adapter_parameters'],
            'backbone_reuse_factor': 5,  # 5 tasks sharing the same backbone
        }
    
    return metrics

def generate_performance_report():
    """Generate a comprehensive performance report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_architecture': analyze_model_architecture(),
        'training_metrics': load_training_metrics(),
        'efficiency_metrics': calculate_efficiency_metrics(),
    }
    
    return report

def create_visualizations(report: Dict[str, Any]):
    """Create performance visualizations."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DYNAMO Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Parameter Distribution
    ax1 = axes[0, 0]
    arch = report['model_architecture']
    if 'adapter_parameters' in arch:
        tasks = list(arch['adapter_parameters'].keys())
        params = list(arch['adapter_parameters'].values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(tasks)))
        bars = ax1.bar(tasks, [p/1e6 for p in params], color=colors)
        ax1.set_title('Parameters per Task Adapter (Millions)')
        ax1.set_ylabel('Parameters (M)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax1.annotate(f'{param/1e6:.1f}M',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 2. Model Size Breakdown
    ax2 = axes[0, 1]
    if 'total_parameters' in arch:
        sizes = [
            arch['backbone_parameters'],
            arch['total_adapter_parameters'],
            arch['router_parameters']
        ]
        labels = ['Backbone\n(Frozen)', 'Adapters\n(Trainable)', 'Router\n(Trainable)']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Parameter Distribution')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    # 3. Training Progress (if Phase 2 metrics available)
    ax3 = axes[1, 0]
    phase2_metrics = report['training_metrics'].get('phase2', {})
    if 'router_accuracy' in phase2_metrics:
        accuracy = phase2_metrics['router_accuracy']
        epochs = range(1, len(accuracy) + 1)
        
        ax3.plot(epochs, accuracy, 'b-', linewidth=2, marker='o', markersize=4)
        ax3.set_title('Router Training Progress')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Routing Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Add final accuracy annotation
        if accuracy:
            final_acc = accuracy[-1]
            ax3.annotate(f'Final: {final_acc:.3f}',
                        xy=(len(accuracy), final_acc),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    else:
        ax3.text(0.5, 0.5, 'No Phase 2 metrics available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Router Training Progress')
    
    # 4. Efficiency Metrics
    ax4 = axes[1, 1]
    efficiency = report['efficiency_metrics']
    if 'parameter_efficiency' in efficiency:
        eff_metrics = efficiency['parameter_efficiency']
        
        metrics_names = ['Params/Task\n(Millions)', 'Router/Adapter\nRatio', 'Backbone Reuse\nFactor']
        metrics_values = [
            eff_metrics['parameters_per_task'] / 1e6,
            eff_metrics['router_to_adapter_ratio'],
            eff_metrics['backbone_reuse_factor']
        ]
        
        bars = ax4.bar(metrics_names, metrics_values, color=['#ffcc99', '#ff99cc', '#99ffcc'])
        ax4.set_title('Efficiency Metrics')
        ax4.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax4.annotate(f'{value:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dynamo_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_performance_summary(report: Dict[str, Any]):
    """Print a comprehensive performance summary."""
    
    print("🚀 DYNAMO MODEL PERFORMANCE REPORT")
    print("=" * 60)
    print(f"📅 Generated: {report['timestamp']}")
    print()
    
    # Architecture Summary
    arch = report['model_architecture']
    if 'total_parameters' in arch:
        print("🏗️  MODEL ARCHITECTURE")
        print("-" * 30)
        print(f"Total Parameters: {arch['total_parameters']:,}")
        print(f"Trainable Parameters: {arch['trainable_parameters']:,}")
        print(f"Model Size: {arch['model_size_mb']:.1f} MB")
        print(f"Router Parameters: {arch['router_parameters']:,}")
        print(f"Adapter Parameters: {arch['total_adapter_parameters']:,}")
        print()
        
        print("📊 Per-Task Adapter Sizes:")
        for task, params in arch['adapter_parameters'].items():
            print(f"  {task:15}: {params:>10,} params ({params/1e6:.1f}M)")
        print()
    
    # Training Status
    training_metrics = report['training_metrics']
    print("🎯 TRAINING STATUS")
    print("-" * 30)
    
    # Phase 1 Status
    if 'phase1' in training_metrics:
        phase1 = training_metrics['phase1']
        print(f"Phase 1 (Adapters): {phase1['status'].upper()}")
        
        if 'adapters' in phase1:
            completed_adapters = sum(1 for a in phase1['adapters'].values() if 'loss' in a)
            print(f"  Completed adapters: {completed_adapters}/5")
            
            for task, metrics in phase1['adapters'].items():
                if 'loss' in metrics:
                    try:
                        loss_val = float(metrics['loss'])
                        print(f"    ✅ {task}: Epoch {metrics['epoch']}, Loss: {loss_val:.4f}")
                    except (ValueError, TypeError):
                        print(f"    ✅ {task}: Epoch {metrics['epoch']}, Loss: {metrics['loss']}")
                else:
                    print(f"    ❌ {task}: Not trained")
    
    # Phase 2 Status
    if 'phase2' in training_metrics:
        phase2 = training_metrics['phase2']
        if 'router_accuracy' in phase2:
            final_accuracy = phase2['router_accuracy'][-1] if phase2['router_accuracy'] else 0
            print(f"Phase 2 (Router): COMPLETED")
            print(f"  Final routing accuracy: {final_accuracy:.3f}")
            print(f"  Training epochs: {len(phase2['router_accuracy'])}")
        else:
            print("Phase 2 (Router): INCOMPLETE")
    else:
        print("Phase 2 (Router): NOT STARTED")
    
    # Phase 3 Status
    if 'phase3' in training_metrics:
        print("Phase 3 (Joint): COMPLETED")
    else:
        print("Phase 3 (Joint): NOT STARTED")
    
    print()
    
    # Performance Highlights
    print("⭐ PERFORMANCE HIGHLIGHTS")
    print("-" * 30)
    
    if 'phase2' in training_metrics and 'router_accuracy' in training_metrics['phase2']:
        router_acc = training_metrics['phase2']['router_accuracy']
        if router_acc:
            max_acc = max(router_acc)
            improvement = max_acc - router_acc[0] if len(router_acc) > 1 else 0
            print(f"🎯 Router Performance:")
            print(f"  Peak accuracy: {max_acc:.3f}")
            print(f"  Improvement: +{improvement:.3f}")
            print(f"  Learning success: {'✅ Good' if max_acc > 0.2 else '⚠️ Needs improvement'}")
    
    if 'parameter_efficiency' in report['efficiency_metrics']:
        eff = report['efficiency_metrics']['parameter_efficiency']
        print(f"⚡ Efficiency:")
        print(f"  Parameters per task: {eff['parameters_per_task']/1e6:.1f}M")
        print(f"  Router overhead: {eff['router_to_adapter_ratio']:.3f}")
        print(f"  Backbone reuse: {eff['backbone_reuse_factor']}x")
    
    if 'checkpoint_sizes' in report['efficiency_metrics']:
        sizes = report['efficiency_metrics']['checkpoint_sizes']
        print(f"💾 Storage:")
        print(f"  Full model: {sizes['full_model_mb']:.1f} MB")
        print(f"  Router only: {sizes['router_mb']:.1f} MB")
    
    print()
    
    # Key Metrics for Demonstration
    print("📋 KEY METRICS TO SHOW STAKEHOLDERS")
    print("-" * 30)
    
    metrics_to_show = []
    
    if arch and 'total_parameters' in arch:
        metrics_to_show.extend([
            f"✅ Model Scale: {arch['total_parameters']/1e6:.1f}M parameters",
            f"✅ Multi-task: 5 tasks with shared backbone",
            f"✅ Parameter Efficiency: {arch['total_adapter_parameters']/1e6:.1f}M trainable vs {arch['backbone_parameters']/1e6:.1f}M frozen"
        ])
    
    if 'phase2' in training_metrics and 'router_accuracy' in training_metrics['phase2']:
        router_acc = training_metrics['phase2']['router_accuracy']
        if router_acc:
            final_acc = router_acc[-1]
            metrics_to_show.append(f"✅ Smart Routing: {final_acc:.1%} task classification accuracy")
    
    if 'parameter_efficiency' in report['efficiency_metrics']:
        eff = report['efficiency_metrics']['parameter_efficiency']
        metrics_to_show.extend([
            f"✅ Scalability: Only {eff['router_to_adapter_ratio']*100:.1f}% router overhead",
            f"✅ Memory Efficient: {eff['backbone_reuse_factor']}x backbone reuse"
        ])
    
    for metric in metrics_to_show:
        print(f"  {metric}")
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 30)
    
    recommendations = []
    
    if 'phase2' in training_metrics and 'router_accuracy' in training_metrics['phase2']:
        router_acc = training_metrics['phase2']['router_accuracy']
        if router_acc and router_acc[-1] < 0.5:
            recommendations.append("🔧 Consider more router training epochs for better accuracy")
    
    if 'phase3' not in training_metrics:
        recommendations.append("🚀 Run Phase 3 for optimal joint performance")
    
    if not recommendations:
        recommendations.append("✅ Model training appears optimal!")
    
    for rec in recommendations:
        print(f"  {rec}")

def save_metrics_report(report: Dict[str, Any], filename: str = "dynamo_metrics_report.json"):
    """Save the complete metrics report to file."""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📄 Detailed report saved to: {filename}")

def main():
    """Generate complete performance analysis."""
    print("📊 Generating DYNAMO Performance Analysis...")
    print()
    
    # Generate comprehensive report
    report = generate_performance_report()
    
    # Print summary
    print_performance_summary(report)
    
    # Create visualizations
    print("📈 Creating performance visualizations...")
    try:
        create_visualizations(report)
        print("✅ Visualization saved as: dynamo_performance_analysis.png")
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
    
    # Save detailed report
    save_metrics_report(report)
    
    print("\n🎉 Performance analysis complete!")
    print("\n📋 To show model performance to stakeholders:")
    print("  1. Show the performance summary above")
    print("  2. Display dynamo_performance_analysis.png")
    print("  3. Run inference.py for live demonstrations")
    print("  4. Reference dynamo_metrics_report.json for detailed metrics")

if __name__ == "__main__":
    main() 