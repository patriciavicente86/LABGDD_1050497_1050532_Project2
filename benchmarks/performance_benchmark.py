"""
Comprehensive benchmarking suite for CPU vs GPU and ML vs DL comparisons
"""

import time
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List
import psutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """
    Benchmarking framework for comparing performance across different configurations
    """
    
    def __init__(self, output_path: str = "benchmarks/results.json"):
        self.output_path = output_path
        self.results = {}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def benchmark_function(self, func, name: str, num_runs: int = 3, **kwargs):
        """
        Benchmark a function with multiple runs
        """
        logger.info(f"Benchmarking {name}...")
        
        times = []
        for i in range(num_runs):
            logger.info(f"  Run {i+1}/{num_runs}")
            start_time = time.time()
            
            try:
                result = func(**kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)
                logger.info(f"    Completed in {elapsed:.2f} seconds")
            except Exception as e:
                logger.error(f"    Failed: {str(e)}")
                times.append(None)
        
        # Calculate statistics
        valid_times = [t for t in times if t is not None]
        
        if valid_times:
            self.results[name] = {
                "mean_time": np.mean(valid_times),
                "std_time": np.std(valid_times),
                "min_time": np.min(valid_times),
                "max_time": np.max(valid_times),
                "num_runs": len(valid_times),
                "success_rate": len(valid_times) / num_runs
            }
        else:
            self.results[name] = {
                "error": "All runs failed"
            }
        
        return self.results[name]
    
    def benchmark_ml_training(self, num_runs: int = 3):
        """
        Benchmark Spark MLlib training
        """
        from ml.demand_forecasting import DemandForecaster
        
        def train_ml():
            forecaster = DemandForecaster()
            results = forecaster.run_pipeline()
            return results
        
        return self.benchmark_function(train_ml, "ml_training", num_runs=num_runs)
    
    def benchmark_dl_training_cpu(self, num_runs: int = 3):
        """
        Benchmark LSTM training on CPU
        """
        import tensorflow as tf
        from dl.lstm_forecaster import LSTMDemandForecaster
        
        def train_dl_cpu():
            # Disable GPU
            tf.config.set_visible_devices([], 'GPU')
            
            forecaster = LSTMDemandForecaster(use_gpu=False)
            metrics = forecaster.run_pipeline(zone_id=237, lookback=24)
            return metrics
        
        return self.benchmark_function(train_dl_cpu, "dl_training_cpu", num_runs=num_runs)
    
    def benchmark_dl_training_gpu(self, num_runs: int = 3):
        """
        Benchmark LSTM training on GPU
        """
        import tensorflow as tf
        from dl.lstm_forecaster import LSTMDemandForecaster
        
        def train_dl_gpu():
            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logger.warning("No GPU available, skipping GPU benchmark")
                return None
            
            forecaster = LSTMDemandForecaster(use_gpu=True)
            metrics = forecaster.run_pipeline(zone_id=237, lookback=24)
            return metrics
        
        return self.benchmark_function(train_dl_gpu, "dl_training_gpu", num_runs=num_runs)
    
    def benchmark_data_processing(self, num_runs: int = 3):
        """
        Benchmark Silver layer processing
        """
        from pyspark.sql import SparkSession
        
        def process_data():
            spark = SparkSession.builder \
                .appName("Benchmark_Silver") \
                .getOrCreate()
            
            # Read bronze
            df = spark.read.parquet("data/yellow/2024/yellow_tripdata_2024-01.parquet")
            
            # Apply transformations (simulate silver processing)
            from pyspark.sql import functions as F
            
            df = df.filter(
                (F.col("PULocationID") > 0) &
                (F.col("trip_distance") > 0) &
                (F.col("fare_amount") > 0)
            )
            
            df = df.withColumn(
                "trip_duration_minutes",
                (F.unix_timestamp("tpep_dropoff_datetime") - 
                 F.unix_timestamp("tpep_pickup_datetime")) / 60
            )
            
            # Trigger action
            count = df.count()
            
            return count
        
        return self.benchmark_function(process_data, "silver_processing", num_runs=num_runs)
    
    def benchmark_gold_aggregation(self, num_runs: int = 3):
        """
        Benchmark Gold layer aggregation
        """
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        
        def aggregate_data():
            spark = SparkSession.builder \
                .appName("Benchmark_Gold") \
                .getOrCreate()
            
            # Read silver
            df = spark.read.parquet("lake/silver/service=yellow")
            
            # Aggregate by zone and hour
            agg_df = df.groupBy("PULocationID", "pickup_date", "pickup_hour") \
                .agg(
                    F.count("*").alias("trip_count"),
                    F.avg("trip_distance").alias("avg_distance"),
                    F.avg("fare_amount").alias("avg_fare")
                )
            
            count = agg_df.count()
            
            return count
        
        return self.benchmark_function(aggregate_data, "gold_aggregation", num_runs=num_runs)
    
    def compare_cpu_vs_gpu(self):
        """
        Compare CPU vs GPU for DL training
        """
        logger.info("=== CPU vs GPU Comparison ===")
        
        cpu_result = self.results.get("dl_training_cpu", {})
        gpu_result = self.results.get("dl_training_gpu", {})
        
        if "mean_time" in cpu_result and "mean_time" in gpu_result:
            speedup = cpu_result["mean_time"] / gpu_result["mean_time"]
            
            comparison = {
                "cpu_time": cpu_result["mean_time"],
                "gpu_time": gpu_result["mean_time"],
                "speedup": speedup,
                "conclusion": f"GPU is {speedup:.2f}x faster than CPU"
            }
            
            self.results["cpu_vs_gpu_comparison"] = comparison
            
            logger.info(f"CPU Time: {cpu_result['mean_time']:.2f}s")
            logger.info(f"GPU Time: {gpu_result['mean_time']:.2f}s")
            logger.info(f"Speedup: {speedup:.2f}x")
        else:
            logger.warning("Cannot compare CPU vs GPU - incomplete data")
    
    def compare_ml_vs_dl(self):
        """
        Compare ML vs DL training time
        """
        logger.info("=== ML vs DL Comparison ===")
        
        ml_result = self.results.get("ml_training", {})
        dl_result = self.results.get("dl_training_gpu", {}) or self.results.get("dl_training_cpu", {})
        
        if "mean_time" in ml_result and "mean_time" in dl_result:
            comparison = {
                "ml_time": ml_result["mean_time"],
                "dl_time": dl_result["mean_time"],
                "ratio": dl_result["mean_time"] / ml_result["mean_time"]
            }
            
            self.results["ml_vs_dl_comparison"] = comparison
            
            logger.info(f"ML Time: {ml_result['mean_time']:.2f}s")
            logger.info(f"DL Time: {dl_result['mean_time']:.2f}s")
            logger.info(f"Ratio: {comparison['ratio']:.2f}x")
        else:
            logger.warning("Cannot compare ML vs DL - incomplete data")
    
    def get_system_info(self):
        """
        Get system information
        """
        import tensorflow as tf
        
        self.results["system_info"] = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
            "gpu_devices": [gpu.name for gpu in tf.config.list_physical_devices('GPU')]
        }
    
    def save_results(self):
        """
        Save benchmark results to JSON
        """
        logger.info(f"Saving results to {self.output_path}")
        
        with open(self.output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("Results saved successfully")
    
    def generate_report(self):
        """
        Generate human-readable benchmark report
        """
        report_path = self.output_path.replace('.json', '_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NYC TAXI BIG DATA PIPELINE - PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # System Info
            if "system_info" in self.results:
                f.write("SYSTEM INFORMATION:\n")
                f.write("-" * 40 + "\n")
                for key, value in self.results["system_info"].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Benchmark Results
            f.write("BENCHMARK RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for name, result in self.results.items():
                if name in ["system_info", "cpu_vs_gpu_comparison", "ml_vs_dl_comparison"]:
                    continue
                
                f.write(f"\n{name.upper()}:\n")
                if "mean_time" in result:
                    f.write(f"  Mean Time: {result['mean_time']:.2f} seconds\n")
                    f.write(f"  Std Dev: {result['std_time']:.2f} seconds\n")
                    f.write(f"  Min Time: {result['min_time']:.2f} seconds\n")
                    f.write(f"  Max Time: {result['max_time']:.2f} seconds\n")
                    f.write(f"  Success Rate: {result['success_rate']*100:.1f}%\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
            
            # Comparisons
            if "cpu_vs_gpu_comparison" in self.results:
                f.write("\n" + "=" * 40 + "\n")
                f.write("CPU VS GPU COMPARISON:\n")
                f.write("-" * 40 + "\n")
                comp = self.results["cpu_vs_gpu_comparison"]
                f.write(f"  CPU Time: {comp['cpu_time']:.2f} seconds\n")
                f.write(f"  GPU Time: {comp['gpu_time']:.2f} seconds\n")
                f.write(f"  Speedup: {comp['speedup']:.2f}x\n")
                f.write(f"  Conclusion: {comp['conclusion']}\n")
            
            if "ml_vs_dl_comparison" in self.results:
                f.write("\n" + "=" * 40 + "\n")
                f.write("ML VS DL COMPARISON:\n")
                f.write("-" * 40 + "\n")
                comp = self.results["ml_vs_dl_comparison"]
                f.write(f"  ML Time: {comp['ml_time']:.2f} seconds\n")
                f.write(f"  DL Time: {comp['dl_time']:.2f} seconds\n")
                f.write(f"  Ratio: {comp['ratio']:.2f}x\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def run_all_benchmarks(self):
        """
        Run complete benchmark suite
        """
        logger.info("Starting comprehensive benchmark suite...")
        
        # Get system info
        self.get_system_info()
        
        # Run benchmarks
        try:
            self.benchmark_data_processing(num_runs=2)
        except Exception as e:
            logger.error(f"Silver processing benchmark failed: {str(e)}")
        
        try:
            self.benchmark_gold_aggregation(num_runs=2)
        except Exception as e:
            logger.error(f"Gold aggregation benchmark failed: {str(e)}")
        
        try:
            self.benchmark_ml_training(num_runs=2)
        except Exception as e:
            logger.error(f"ML training benchmark failed: {str(e)}")
        
        try:
            self.benchmark_dl_training_cpu(num_runs=1)
        except Exception as e:
            logger.error(f"DL CPU training benchmark failed: {str(e)}")
        
        try:
            self.benchmark_dl_training_gpu(num_runs=2)
        except Exception as e:
            logger.error(f"DL GPU training benchmark failed: {str(e)}")
        
        # Comparisons
        self.compare_cpu_vs_gpu()
        self.compare_ml_vs_dl()
        
        # Save results
        self.save_results()
        self.generate_report()
        
        logger.info("Benchmark suite completed!")


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
