import sys
import os
import logging
import time
import psutil
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Methods.Computational_Methods import PerformanceTracker

class SystemMonitor:
    def __init__(self):
        self.perf_tracker = PerformanceTracker()
        self.metrics_history: Dict[str, List[float]] = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_memory': [],
            'power_draw': []
        }
        self.start_time = time.time()
        
        logging.basicConfig(
            filename='system_monitoring.log',
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )

    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system performance metrics."""
        metrics = {}
        
        # CPU Usage
        metrics['cpu_usage'] = psutil.cpu_percent()
        
        # Memory Usage
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent
        
        # GPU Metrics
        gpu_memory = self.perf_tracker.monitor_memory_nvidia_smi()
        if gpu_memory:
            metrics['gpu_memory'] = (gpu_memory[0] / gpu_memory[1]) * 100
        
        # Power Draw
        power = self.perf_tracker.measure_power_consumption_nvidia_smi()
        if power:
            metrics['power_draw'] = power
            
        return metrics

    def update_history(self, metrics: Dict[str, float]):
        """Update metrics history."""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

    def visualize_metrics(self, save_path: str = 'performance_metrics.png'):
        """Generate visualization of collected metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        time_points = np.linspace(0, (time.time() - self.start_time) / 60, 
                                len(self.metrics_history['cpu_usage']))
        
        # CPU Usage
        ax1.plot(time_points, self.metrics_history['cpu_usage'])
        ax1.set_title('CPU Usage Over Time')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('CPU Usage (%)')
        
        # Memory Usage
        ax2.plot(time_points, self.metrics_history['memory_usage'])
        ax2.set_title('Memory Usage Over Time')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Memory Usage (%)')
        
        # GPU Memory
        if self.metrics_history['gpu_memory']:
            ax3.plot(time_points, self.metrics_history['gpu_memory'])
            ax3.set_title('GPU Memory Usage Over Time')
            ax3.set_xlabel('Time (minutes)')
            ax3.set_ylabel('GPU Memory Usage (%)')
        
        # Power Draw
        if self.metrics_history['power_draw']:
            ax4.plot(time_points, self.metrics_history['power_draw'])
            ax4.set_title('Power Draw Over Time')
            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('Power Draw (W)')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Performance visualization saved to {save_path}")

    def generate_report(self) -> Dict[str, Dict[str, float]]:
        """Generate statistical report of collected metrics."""
        report = {}
        
        for metric, values in self.metrics_history.items():
            if values:  # Only process metrics that have data
                report[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                
        return report

def main():
    monitor = SystemMonitor()
    try:
        while True:
            metrics = monitor.collect_system_metrics()
            monitor.update_history(metrics)
            monitor.visualize_metrics()
            
            # Generate and log report every 5 minutes
            if len(monitor.metrics_history['cpu_usage']) % 30 == 0:  # Assuming 10s intervals
                report = monitor.generate_report()
                logging.info(f"Performance Report: {report}")
                
            time.sleep(10)  # Collect metrics every 10 seconds
            
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user")
        final_report = monitor.generate_report()
        logging.info(f"Final Performance Report: {final_report}")
        monitor.visualize_metrics('final_performance_metrics.png')

if __name__ == "__main__":
    main() 