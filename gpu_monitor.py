#!/usr/bin/env python3
"""
GPU Performance Monitor for DYNAMO Training
Helps diagnose GPU utilization issues and provides optimization recommendations.
"""

import torch
import psutil
import time
import threading
from typing import Dict, List
import matplotlib.pyplot as plt
import json
from datetime import datetime

class GPUMonitor:
    def __init__(self, log_interval: int = 5):
        """
        Initialize GPU monitor.
        
        Args:
            log_interval: Logging interval in seconds
        """
        self.log_interval = log_interval
        self.monitoring = False
        self.metrics = {
            'timestamps': [],
            'gpu_memory_used': [],
            'gpu_memory_total': [],
            'gpu_utilization': [],
            'cpu_percent': [],
            'ram_percent': []
        }
        
    def start_monitoring(self):
        """Start GPU monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 GPU monitoring started...")
        
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("⏹️  GPU monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            if torch.cuda.is_available():
                # GPU metrics
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                
                # Try to get GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except:
                    gpu_util = 0  # Fallback if nvidia-ml-py not available
                
                # CPU and RAM metrics
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                
                # Store metrics
                self.metrics['timestamps'].append(datetime.now())
                self.metrics['gpu_memory_used'].append(memory_used)
                self.metrics['gpu_memory_total'].append(memory_total)
                self.metrics['gpu_utilization'].append(gpu_util)
                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['ram_percent'].append(ram_percent)
                
                # Console logging
                print(f"⚡ GPU: {memory_used:.1f}/{memory_total:.1f}GB ({memory_used/memory_total*100:.1f}%) | "
                      f"Util: {gpu_util}% | CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}%")
            
            time.sleep(self.log_interval)
    
    def get_summary(self) -> Dict:
        """Get monitoring summary."""
        if not self.metrics['timestamps']:
            return {"error": "No monitoring data collected"}
        
        summary = {
            "monitoring_duration_minutes": len(self.metrics['timestamps']) * self.log_interval / 60,
            "gpu_memory": {
                "peak_usage_gb": max(self.metrics['gpu_memory_used']),
                "avg_usage_gb": sum(self.metrics['gpu_memory_used']) / len(self.metrics['gpu_memory_used']),
                "total_gb": self.metrics['gpu_memory_total'][0],
                "peak_utilization_percent": max(self.metrics['gpu_memory_used']) / self.metrics['gpu_memory_total'][0] * 100
            },
            "gpu_utilization": {
                "peak_percent": max(self.metrics['gpu_utilization']),
                "avg_percent": sum(self.metrics['gpu_utilization']) / len(self.metrics['gpu_utilization'])
            },
            "cpu_utilization": {
                "peak_percent": max(self.metrics['cpu_percent']),
                "avg_percent": sum(self.metrics['cpu_percent']) / len(self.metrics['cpu_percent'])
            },
            "recommendations": self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        
        if not self.metrics['gpu_utilization']:
            return ["⚠️  No GPU utilization data available"]
        
        avg_gpu_util = sum(self.metrics['gpu_utilization']) / len(self.metrics['gpu_utilization'])
        avg_memory_util = (sum(self.metrics['gpu_memory_used']) / len(self.metrics['gpu_memory_used'])) / self.metrics['gpu_memory_total'][0] * 100
        
        # GPU utilization recommendations
        if avg_gpu_util < 50:
            recommendations.append("🐌 Low GPU utilization (<50%). Consider:")
            recommendations.append("   - Increasing batch size")
            recommendations.append("   - Using mixed precision training")
            recommendations.append("   - Optimizing data loading (more workers)")
            recommendations.append("   - Reducing CPU preprocessing")
        
        # Memory utilization recommendations  
        if avg_memory_util < 30:
            recommendations.append("💾 Low GPU memory usage (<30%). Consider:")
            recommendations.append("   - Increasing batch size")
            recommendations.append("   - Using larger models")
            recommendations.append("   - Enabling gradient accumulation")
        elif avg_memory_util > 90:
            recommendations.append("⚠️  High GPU memory usage (>90%). Consider:")
            recommendations.append("   - Reducing batch size")
            recommendations.append("   - Using gradient accumulation")
            recommendations.append("   - Enabling mixed precision training")
        
        # Performance recommendations
        if avg_gpu_util > 80 and avg_memory_util > 80:
            recommendations.append("🚀 Good GPU utilization! System is well-optimized")
        
        return recommendations
    
    def save_plot(self, filename: str = "gpu_monitoring.png"):
        """Save monitoring plot."""
        if not self.metrics['timestamps']:
            print("No data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU Performance Monitoring - DYNAMO Training', fontsize=16)
        
        times = [(t - self.metrics['timestamps'][0]).total_seconds() / 60 for t in self.metrics['timestamps']]
        
        # GPU Memory Usage
        ax1.plot(times, self.metrics['gpu_memory_used'], 'b-', label='Used')
        ax1.axhline(y=self.metrics['gpu_memory_total'][0], color='r', linestyle='--', label='Total')
        ax1.set_ylabel('GPU Memory (GB)')
        ax1.set_title('GPU Memory Usage')
        ax1.legend()
        ax1.grid(True)
        
        # GPU Utilization
        ax2.plot(times, self.metrics['gpu_utilization'], 'g-')
        ax2.set_ylabel('GPU Utilization (%)')
        ax2.set_title('GPU Utilization')
        ax2.set_ylim(0, 100)
        ax2.grid(True)
        
        # CPU Usage
        ax3.plot(times, self.metrics['cpu_percent'], 'orange')
        ax3.set_ylabel('CPU Usage (%)')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_title('CPU Usage')
        ax3.set_ylim(0, 100)
        ax3.grid(True)
        
        # RAM Usage
        ax4.plot(times, self.metrics['ram_percent'], 'purple')
        ax4.set_ylabel('RAM Usage (%)')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_title('RAM Usage')
        ax4.set_ylim(0, 100)
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {filename}")
        
    def save_metrics(self, filename: str = "gpu_metrics.json"):
        """Save metrics to JSON file."""
        # Convert timestamps to strings for JSON serialization
        metrics_serializable = {
            'timestamps': [t.isoformat() for t in self.metrics['timestamps']],
            'gpu_memory_used': self.metrics['gpu_memory_used'],
            'gpu_memory_total': self.metrics['gpu_memory_total'],
            'gpu_utilization': self.metrics['gpu_utilization'],
            'cpu_percent': self.metrics['cpu_percent'],
            'ram_percent': self.metrics['ram_percent'],
            'summary': self.get_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"📁 Metrics saved to {filename}")

def check_gpu_optimization():
    """Quick GPU optimization check."""
    print("🔧 GPU Optimization Check")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch Version: {torch.__version__}")
    
    # Check cuDNN
    print(f"🔹 cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"🔹 cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    
    # Memory info
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    
    print(f"💾 GPU Memory: {memory_allocated:.1f}GB used / {memory_reserved:.1f}GB reserved / {memory_total:.1f}GB total")
    
    # Performance recommendations
    print("\n🚀 Optimization Recommendations:")
    print("   - Enable mixed precision training (use autocast)")
    print("   - Set torch.backends.cudnn.benchmark = True")
    print("   - Use pin_memory=True in DataLoader")
    print("   - Optimize num_workers (4-8 for T4)")
    print("   - Use gradient accumulation for larger effective batch sizes")
    
    return True

if __name__ == "__main__":
    # Quick optimization check
    check_gpu_optimization()
    
    # Start monitoring for 30 seconds as demo
    print("\n🔍 Starting 30-second monitoring demo...")
    monitor = GPUMonitor(log_interval=2)
    monitor.start_monitoring()
    time.sleep(30)
    monitor.stop_monitoring()
    
    # Show summary
    summary = monitor.get_summary()
    print("\n📊 Monitoring Summary:")
    print(json.dumps(summary, indent=2))
    
    # Save results
    monitor.save_plot()
    monitor.save_metrics() 