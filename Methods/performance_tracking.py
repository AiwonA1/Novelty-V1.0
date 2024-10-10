import subprocess
import torch
import tensorflow as tf
from thop import profile as thop_profile
from tensorflow.python.profiler.model_analyzer import profile as tf_profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from typing import Tuple, Optional, List, Dict
import logging
import asyncio
import time
import psutil

# Configure logging for PerformanceTracker
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[PerformanceTracker] %(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class PerformanceTracker:
    """A class to track FLOPs, memory usage, and power consumption for AI models."""

    @staticmethod
    def track_flops_tensorflow(model: tf.keras.Model, input_data: tf.Tensor) -> int:
        """
        Estimate FLOPs for a TensorFlow model.

        Args:
            model (tf.keras.Model): TensorFlow model.
            input_data (tf.Tensor): Input data for the model.

        Returns:
            int: Total FLOPs.
        """
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            opts = ProfileOptionBuilder.float_operation()
            flops = tf_profile(sess.graph, options=opts)
            logger.info(f'FLOPs: {flops.total_float_ops}')
            return flops.total_float_ops

    @staticmethod
    def track_flops_pytorch(model: torch.nn.Module, input_data: torch.Tensor) -> Tuple[int, int]:
        """
        Estimate FLOPs and parameters for a PyTorch model.

        Args:
            model (torch.nn.Module): PyTorch model.
            input_data (torch.Tensor): Input tensor for the model.

        Returns:
            Tuple[int, int]: Total FLOPs and number of parameters.
        """
        flops, params = thop_profile(model, inputs=(input_data,), verbose=False)
        logger.info(f'FLOPs: {flops}, Parameters: {params}')
        return flops, params

    @staticmethod
    def monitor_memory_nvidia_smi() -> Optional[Tuple[int, int]]:
        """
        Monitor GPU memory usage using nvidia-smi.

        Returns:
            Optional[Tuple[int, int]]: Tuple of GPU memory used and total memory in MB, or None if error.
        """
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
            memory_used, memory_total = map(int, result.strip().split(', '))
            logger.info(f'GPU Memory Used: {memory_used} MB / {memory_total} MB')
            return memory_used, memory_total
        except subprocess.CalledProcessError as e:
            logger.error("Error accessing nvidia-smi:", exc_info=e)
            return None

    @staticmethod
    def monitor_memory_pytorch() -> Tuple[int, int]:
        """
        Monitor GPU memory usage in PyTorch.

        Returns:
            Tuple[int, int]: Allocated and cached memory in bytes.
        """
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        logger.info(f'Allocated: {allocated} bytes, Cached: {cached} bytes')
        return allocated, cached

    @staticmethod
    def monitor_memory_tensorflow() -> Dict[str, int]:
        """
        Monitor GPU memory usage in TensorFlow.

        Returns:
            Dict[str, int]: Memory info for GPU:0.
        """
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        logger.info(f"Memory Info: {memory_info}")
        return memory_info

    @staticmethod
    async def monitor_system_resources(interval: float = 1.0, duration: float = 10.0) -> Dict[str, List[float]]:
        """
        Asynchronously monitor system CPU and memory usage.

        Args:
            interval (float): Time interval between monitoring in seconds.
            duration (float): Total duration for monitoring in seconds.

        Returns:
            Dict[str, List[float]]: Collected CPU and memory usage data.
        """
        logger.info("Starting system resource monitoring.")
        cpu_usage = []
        memory_usage = []
        start_time = time.time()
        while time.time() - start_time < duration:
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory().percent
            cpu_usage.append(cpu)
            memory_usage.append(memory)
            logger.debug(f'CPU Usage: {cpu}%, Memory Usage: {memory}%')
            await asyncio.sleep(interval)
        logger.info("Completed system resource monitoring.")
        return {"cpu_usage": cpu_usage, "memory_usage": memory_usage}

    @staticmethod
    def measure_power_consumption_dcgmi() -> Optional[str]:
        """
        Measure GPU power consumption using NVIDIA DCGM.

        Returns:
            Optional[str]: Power usage in watts as a string, or None if error.
        """
        try:
            result = subprocess.check_output(['dcgmi', 'dmon', '-e', '203'], encoding='utf-8')
            logger.info("Power Consumption (W):")
            logger.info(result)
            return result
        except subprocess.CalledProcessError as e:
            logger.error("Error accessing DCGM:", exc_info=e)
            return None

    @staticmethod
    def measure_power_consumption_nvidia_smi() -> Optional[float]:
        """
        Measure GPU power consumption using nvidia-smi.

        Returns:
            Optional[float]: Power draw in watts, or None if error.
        """
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
            power_draw = float(result.strip())
            logger.info(f'Power Draw: {power_draw} W')
            return power_draw
        except subprocess.CalledProcessError as e:
            logger.error("Error accessing nvidia-smi for power draw:", exc_info=e)
            return None

    @staticmethod
    def aggregate_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Aggregate collected metrics by calculating average values.

        Args:
            metrics (Dict[str, List[float]]): Collected metrics.

        Returns:
            Dict[str, float]: Aggregated average metrics.
        """
        aggregated = {key: sum(values)/len(values) if values else 0.0 for key, values in metrics.items()}
        logger.info(f'Aggregated Metrics: {aggregated}')
        return aggregated

    @staticmethod
    async def comprehensive_performance_report(model: torch.nn.Module, input_data: torch.Tensor, duration: float = 10.0) -> Dict[str, float]:
        """
        Generate a comprehensive performance report including FLOPs, memory, CPU, and power usage.

        Args:
            model (torch.nn.Module): PyTorch model.
            input_data (torch.Tensor): Input tensor for the model.
            duration (float): Duration for system resource monitoring in seconds.

        Returns:
            Dict[str, float]: Aggregated performance metrics.
        """
        logger.info("Generating comprehensive performance report.")
        flops, params = PerformanceTracker.track_flops_pytorch(model, input_data)
        memory = PerformanceTracker.monitor_memory_pytorch()
        power = PerformanceTracker.measure_power_consumption_nvidia_smi()

        system_metrics = await PerformanceTracker.monitor_system_resources(duration=duration)
        aggregated_system_metrics = PerformanceTracker.aggregate_metrics(system_metrics)

        report = {
            "FLOPs": flops,
            "Parameters": params,
            "Memory Allocated (bytes)": memory[0],
            "Memory Cached (bytes)": memory[1],
            "Power Draw (W)": power if power else 0.0,
            "Average CPU Usage (%)": aggregated_system_metrics.get("cpu_usage", 0.0),
            "Average Memory Usage (%)": aggregated_system_metrics.get("memory_usage", 0.0),
        }
        logger.info(f'Comprehensive Performance Report: {report}')
        return report