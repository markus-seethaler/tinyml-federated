"""
Performance measurement utilities for BLE operations.
"""

import time
import asyncio
from statistics import mean, stdev
import numpy as np

class BLETimingHandler:
    def __init__(self, command_handler):
        """
        Initialize a timing handler for benchmarking BLE operations.
        
        Args:
            command_handler: An instance of CommandHandler for device interaction
        """
        self.handler = command_handler
        self.get_weights_times = []
        self.send_weights_times = []
        self.total_weights = command_handler.total_weights
        
    async def measure_get_weights(self, num_trials=5):
        """
        Measure time taken and data rate to receive weights from Arduino.
        
        Args:
            num_trials: Number of trials to perform for statistics
            
        Returns:
            Dictionary with timing statistics if successful, None otherwise
        """
        print(f"\nMeasuring GET_WEIGHTS timing ({num_trials} trials)...")

        # Calculate total data size in bits
        bits_transferred = self.total_weights * 32  # 32 bits per float32
        self.get_weights_times = []  # Reset timing data

        for i in range(num_trials):
            start_time = time.perf_counter()
            weights = await self.handler.get_weights()
            end_time = time.perf_counter()

            if weights is not None:
                duration = end_time - start_time
                data_rate = bits_transferred / (duration * 1000)  # Convert to kbit/s

                self.get_weights_times.append(duration)
                print(f"Trial {i + 1}: {duration:.3f} seconds, {data_rate:.2f} kbit/s")
            else:
                print(f"Trial {i + 1}: Failed")

            await asyncio.sleep(1)  # Wait between trials

        # Compute and display statistics
        stats = self._compute_statistics(self.get_weights_times, bits_transferred, "GET_WEIGHTS")
        self._print_statistics(stats, "GET_WEIGHTS")
        return stats

    async def measure_send_weights(self, num_trials=5):
        """
        Measure time taken and data rate to send weights to Arduino.
        
        Args:
            num_trials: Number of trials to perform for statistics
            
        Returns:
            Dictionary with timing statistics if successful, None otherwise
        """
        print(f"\nMeasuring SET_WEIGHTS timing ({num_trials} trials)...")

        # Create test weights and calculate total data size in bits
        test_weights = np.random.normal(0, 0.5, self.total_weights).astype(np.float32)
        bits_transferred = self.total_weights * 32  # 32 bits per float32
        self.send_weights_times = []  # Reset timing data

        for i in range(num_trials):
            start_time = time.perf_counter()
            success = await self.handler.send_weights(test_weights)
            end_time = time.perf_counter()

            if success:
                duration = end_time - start_time
                data_rate = bits_transferred / (duration * 1000)  # Convert to kbit/s

                self.send_weights_times.append(duration)
                print(f"Trial {i + 1}: {duration:.3f} seconds, {data_rate:.2f} kbit/s")
            else:
                print(f"Trial {i + 1}: Failed")

            await asyncio.sleep(1)  # Wait between trials

        # Compute and display statistics
        stats = self._compute_statistics(self.send_weights_times, bits_transferred, "SET_WEIGHTS")
        self._print_statistics(stats, "SET_WEIGHTS")
        return stats
        
    async def measure_both(self, num_trials=5):
        """
        Measure both GET_WEIGHTS and SET_WEIGHTS performance.
        
        Args:
            num_trials: Number of trials to perform for each operation
            
        Returns:
            Tuple of (get_stats, set_stats) dictionaries
        """
        get_stats = await self.measure_get_weights(num_trials)
        set_stats = await self.measure_send_weights(num_trials)
        return get_stats, set_stats
    
    def _compute_statistics(self, times, bits_transferred, operation_name):
        """
        Compute statistics for timing measurements.
        
        Args:
            times: List of timing measurements
            bits_transferred: Total bits transferred in each operation
            operation_name: Name of the operation for reporting
            
        Returns:
            Dictionary with computed statistics or None if no data
        """
        if not times:
            print(f"\nNo successful {operation_name} operations to analyze")
            return None
            
        avg_time = mean(times)
        avg_data_rate = bits_transferred / (avg_time * 1000)
        
        stats = {
            "operation": operation_name,
            "trials": len(times),
            "avg_time": avg_time,
            "avg_data_rate": avg_data_rate,
            "min_time": min(times),
            "max_time": max(times),
        }
        
        if len(times) > 1:
            stats["std_dev"] = stdev(times)
            
        return stats
    
    def _print_statistics(self, stats, operation_name):
        """
        Print statistics for timing measurements.
        
        Args:
            stats: Dictionary with computed statistics
            operation_name: Name of the operation for reporting
        """
        if not stats:
            return
            
        print(f"\n{operation_name} Statistics:")
        print(f"Average time: {stats['avg_time']:.3f} seconds")
        print(f"Average data rate: {stats['avg_data_rate']:.2f} kbit/s")
        
        if stats.get("std_dev") is not None:
            print(f"Standard deviation: {stats['std_dev']:.3f} seconds")
            
        print(f"Min time: {stats['min_time']:.3f} seconds")
        print(f"Max time: {stats['max_time']:.3f} seconds")
        
    def get_summary(self):
        """
        Get a summary of all timing measurements.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        if self.get_weights_times:
            summary["get_weights"] = {
                "count": len(self.get_weights_times),
                "avg_time": mean(self.get_weights_times),
                "min_time": min(self.get_weights_times),
                "max_time": max(self.get_weights_times)
            }
            
        if self.send_weights_times:
            summary["send_weights"] = {
                "count": len(self.send_weights_times),
                "avg_time": mean(self.send_weights_times),
                "min_time": min(self.send_weights_times),
                "max_time": max(self.send_weights_times)
            }
            
        return summary