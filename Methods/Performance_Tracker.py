# Performance Tester Module with Optimizer Toggle and Performance Comparison

# This module allows users to run and compare system performance for LLM tasks with and without the unified optimizer.
# It captures key metrics such as CPU usage, memory usage, GPU power, and GPU memory. The optimizer can be toggled 
# on or off, and performance is logged for both modes. The module captures these metrics during task execution, allowing 
# users to analyze system performance and resource efficiency in different configurations. Logs are saved to a file for 
# further analysis.

import time
import psutil
import subprocess
import asyncio

# Global flag to enable or disable optimizer
optimizer_enabled = True  # Default to enabled

def toggle_optimizer(enable: bool):
    """
    Toggles the optimizer on or off based on user input.
    
    Args:
        enable (bool): True to enable optimizer, False to disable.
    """
    global optimizer_enabled
    optimizer_enabled = enable
    if enable:
        print("Optimizer enabled.")
    else:
        print("Optimizer disabled.")

class PerformanceTester:
    def __init__(self):
        self.logs = []
    
    def capture_metrics(self):
        """
        Captures and logs CPU and memory usage, as well as GPU power and memory usage if applicable.
        """
        # Capture CPU and memory usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        
        # Capture GPU metrics if available via nvidia-smi
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw,memory.used", "--format=csv,noheader,nounits"], 
                encoding='utf-8'
            )
            gpu_power, gpu_memory = gpu_info.split(', ')
        except subprocess.CalledProcessError:
            gpu_power, gpu_memory = 'N/A', 'N/A'
        
        # Log metrics
        metrics = {
            "CPU Usage (%)": cpu_usage,
            "Memory Used (MB)": memory_info.used / (1024 * 1024),
            "GPU Power (W)": gpu_power,
            "GPU Memory (MB)": gpu_memory,
        }
        self.logs.append(metrics)
        print(metrics)

    async def run_test(self, mode: str):
        """
        Simulates running LLM work with or without optimization, capturing system performance during execution.
        
        Args:
            mode (str): "with_optimizer" or "without_optimizer" to specify which mode to run in.
        """
        print(f"Running test in {mode} mode...")
        if mode == "with_optimizer":
            toggle_optimizer(True)
        else:
            toggle_optimizer(False)
        
        # Simulate LLM task
        for _ in range(5):  # Simulate some work
            self.capture_metrics()
            await asyncio.sleep(1)

    def compare_performance(self):
        """
        Runs tests with and without optimizer, capturing and comparing performance metrics.
        """
        asyncio.run(self.run_test("with_optimizer"))
        asyncio.run(self.run_test("without_optimizer"))
    
    def save_logs(self, filename="performance_logs.txt"):
        """
        Saves the captured metrics to a file.
        
        Args:
            filename (str): The name of the file to save logs to.
        """
        with open(filename, 'w') as file:
            for log in self.logs:
                file.write(str(log) + '\n')
        print(f"Logs saved to {filename}")

# Example usage
tester = PerformanceTester()
tester.compare_performance()
tester.save_logs()
