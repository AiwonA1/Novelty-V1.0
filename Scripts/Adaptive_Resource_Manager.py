import sys
import os
import logging
import time
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Scripts.Performance_Monitor import SystemMonitor
from Methods.Computational_Methods import EnergyOptimizer

@dataclass
class ResourceThresholds:
    cpu_high: float = 80.0
    cpu_low: float = 20.0
    memory_high: float = 85.0
    memory_low: float = 30.0
    gpu_memory_high: float = 90.0
    gpu_memory_low: float = 25.0
    power_high: float = 150.0  # Watts
    power_low: float = 50.0    # Watts

class AdaptiveResourceManager:
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.energy_optimizer = EnergyOptimizer()
        self.thresholds = ResourceThresholds()
        self.resource_history: Dict[str, List[float]] = {
            'cpu': [],
            'memory': [],
            'gpu': [],
            'power': []
        }
        self.optimization_history: List[Dict] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logging.basicConfig(
            filename='resource_management.log',
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def monitor_and_optimize(self, interval: float = 1.0):
        """Continuously monitor and optimize resource usage."""
        try:
            while True:
                metrics = self.system_monitor.collect_system_metrics()
                await self._analyze_and_optimize(metrics)
                await asyncio.sleep(interval)
        except Exception as e:
            self.logger.error(f"Error in resource monitoring: {str(e)}")
            raise

    async def _analyze_and_optimize(self, metrics: Dict[str, float]):
        """Analyze metrics and apply optimizations if needed."""
        optimization_actions = []

        # CPU Optimization
        if metrics['cpu_usage'] > self.thresholds.cpu_high:
            optimization_actions.append(self._optimize_cpu_usage())
        
        # Memory Optimization
        if metrics['memory_usage'] > self.thresholds.memory_high:
            optimization_actions.append(self._optimize_memory_usage())
        
        # GPU Optimization
        if 'gpu_memory' in metrics and metrics['gpu_memory'] > self.thresholds.gpu_memory_high:
            optimization_actions.append(self._optimize_gpu_usage())
        
        # Power Optimization
        if 'power_draw' in metrics and metrics['power_draw'] > self.thresholds.power_high:
            optimization_actions.append(self._optimize_power_usage())

        # Execute optimizations concurrently
        if optimization_actions:
            await asyncio.gather(*optimization_actions)
            self._record_optimization(metrics, optimization_actions)

    async def _optimize_cpu_usage(self) -> Dict[str, any]:
        """Optimize CPU usage through adaptive scheduling."""
        try:
            # Analyze process priorities
            current_processes = await self._get_resource_intensive_processes('cpu')
            
            # Adjust process priorities
            optimizations = {
                'type': 'cpu',
                'actions': []
            }
            
            for proc in current_processes:
                if proc['usage'] > 20.0:  # High CPU process
                    # Reduce priority for non-critical processes
                    if not self._is_critical_process(proc['name']):
                        await self._adjust_process_priority(proc['pid'], -1)
                        optimizations['actions'].append({
                            'process': proc['name'],
                            'action': 'reduce_priority'
                        })
            
            self.logger.info(f"CPU optimization applied: {optimizations}")
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Error in CPU optimization: {str(e)}")
            return {'type': 'cpu', 'error': str(e)}

    async def _optimize_memory_usage(self) -> Dict[str, any]:
        """Optimize memory usage through smart caching and cleanup."""
        try:
            optimizations = {
                'type': 'memory',
                'actions': []
            }
            
            # Analyze memory usage patterns
            usage_patterns = await self._analyze_memory_patterns()
            
            # Clear unnecessary caches
            cleared_caches = await self._clear_system_caches()
            optimizations['actions'].append({
                'action': 'clear_caches',
                'size_freed': cleared_caches
            })
            
            # Suggest memory limits for processes
            for pattern in usage_patterns:
                if pattern['growth_rate'] > 0.1:  # Significant growth
                    limit = self._calculate_optimal_memory_limit(pattern)
                    optimizations['actions'].append({
                        'process': pattern['process'],
                        'action': 'set_memory_limit',
                        'limit': limit
                    })
            
            self.logger.info(f"Memory optimization applied: {optimizations}")
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Error in memory optimization: {str(e)}")
            return {'type': 'memory', 'error': str(e)}

    async def _optimize_gpu_usage(self) -> Dict[str, any]:
        """Optimize GPU usage through workload management."""
        try:
            optimizations = {
                'type': 'gpu',
                'actions': []
            }
            
            # Get current GPU processes
            gpu_processes = await self._get_gpu_processes()
            
            # Analyze and optimize each process
            for proc in gpu_processes:
                if proc['memory_usage'] > 1000:  # Using >1GB VRAM
                    # Check if process can be optimized
                    optimization = await self._optimize_gpu_process(proc)
                    if optimization:
                        optimizations['actions'].append(optimization)
            
            self.logger.info(f"GPU optimization applied: {optimizations}")
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Error in GPU optimization: {str(e)}")
            return {'type': 'gpu', 'error': str(e)}

    async def _optimize_power_usage(self) -> Dict[str, any]:
        """Optimize power usage through frequency and voltage management."""
        try:
            optimizations = {
                'type': 'power',
                'actions': []
            }
            
            # Get current power state
            power_state = await self._get_power_state()
            
            # Apply power optimizations
            if power_state['cpu_frequency'] > 2.0:  # >2GHz
                await self._set_cpu_frequency(min(power_state['cpu_frequency'] * 0.8, 2.0))
                optimizations['actions'].append({
                    'component': 'cpu',
                    'action': 'reduce_frequency'
                })
            
            if power_state['gpu_power_limit'] > 100:  # >100W
                await self._set_gpu_power_limit(min(power_state['gpu_power_limit'] * 0.9, 100))
                optimizations['actions'].append({
                    'component': 'gpu',
                    'action': 'reduce_power_limit'
                })
            
            self.logger.info(f"Power optimization applied: {optimizations}")
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Error in power optimization: {str(e)}")
            return {'type': 'power', 'error': str(e)}

    def _record_optimization(self, metrics: Dict[str, float], optimizations: List[Dict]):
        """Record optimization actions and their effects."""
        record = {
            'timestamp': time.time(),
            'metrics_before': metrics.copy(),
            'optimizations': optimizations,
            'metrics_after': self.system_monitor.collect_system_metrics()
        }
        self.optimization_history.append(record)

    async def generate_optimization_report(self) -> Dict[str, any]:
        """Generate a comprehensive optimization report."""
        report = {
            'total_optimizations': len(self.optimization_history),
            'resource_savings': {
                'cpu': 0.0,
                'memory': 0.0,
                'gpu': 0.0,
                'power': 0.0
            },
            'optimization_effectiveness': {}
        }
        
        # Calculate resource savings
        for record in self.optimization_history:
            for resource in ['cpu_usage', 'memory_usage', 'gpu_memory', 'power_draw']:
                if resource in record['metrics_before'] and resource in record['metrics_after']:
                    saving = record['metrics_before'][resource] - record['metrics_after'][resource]
                    resource_key = resource.split('_')[0]
                    report['resource_savings'][resource_key] += saving
        
        # Calculate optimization effectiveness
        for resource in report['resource_savings']:
            if self.optimization_history:
                report['optimization_effectiveness'][resource] = (
                    report['resource_savings'][resource] / len(self.optimization_history)
                )
        
        return report

async def main():
    try:
        resource_manager = AdaptiveResourceManager()
        
        # Start monitoring and optimization
        monitor_task = asyncio.create_task(
            resource_manager.monitor_and_optimize(interval=2.0)
        )
        
        # Run for a specified duration
        await asyncio.sleep(3600)  # Run for 1 hour
        
        # Generate and save report
        report = await resource_manager.generate_optimization_report()
        
        logging.info("Resource management completed successfully")
        logging.info(f"Optimization Report: {report}")
        
    except Exception as e:
        logging.critical(f"Critical error in resource management: {str(e)}")
        raise
    finally:
        monitor_task.cancel()

if __name__ == "__main__":
    asyncio.run(main()) 