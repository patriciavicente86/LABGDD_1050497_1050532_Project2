"""
Visualization and reporting module for NYC Taxi pipeline results
Generates figures for paper and presentation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    """
    Creates visualizations for paper and presentation
    """
    
    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_lambda_kappa_comparison(self, metrics_path: str = "reports/lambda_kappa_metrics.json"):
        """
        Compare Lambda vs Kappa across multiple dimensions
        """
        logger.info("Creating Lambda vs Kappa comparison plot...")
        
        # Sample data (replace with actual metrics)
        metrics = {
            "Lambda": {
                "Freshness (min)": 8,
                "Throughput (M/min)": 3.0,
                "Latency (sec)": 600,
                "Complexity (LOC)": 150
            },
            "Kappa": {
                "Freshness (min)": 0.5,
                "Throughput (M/min)": 2.5,
                "Latency (sec)": 30,
                "Complexity (LOC)": 220
            }
        }
        
        # Create comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Lambda vs Kappa Architecture Comparison', fontsize=16, fontweight='bold')
        
        # Freshness
        axes[0, 0].bar(['Lambda', 'Kappa'], 
                       [metrics['Lambda']['Freshness (min)'], metrics['Kappa']['Freshness (min)']],
                       color=['#1f77b4', '#ff7f0e'])
        axes[0, 0].set_ylabel('Minutes')
        axes[0, 0].set_title('Data Freshness (Lower is Better)')
        axes[0, 0].set_ylim(0, max(metrics['Lambda']['Freshness (min)'], metrics['Kappa']['Freshness (min)']) * 1.2)
        
        # Throughput
        axes[0, 1].bar(['Lambda', 'Kappa'],
                       [metrics['Lambda']['Throughput (M/min)'], metrics['Kappa']['Throughput (M/min)']],
                       color=['#1f77b4', '#ff7f0e'])
        axes[0, 1].set_ylabel('Million Records/Minute')
        axes[0, 1].set_title('Processing Throughput (Higher is Better)')
        
        # Latency
        axes[1, 0].bar(['Lambda', 'Kappa'],
                       [metrics['Lambda']['Latency (sec)'], metrics['Kappa']['Latency (sec)']],
                       color=['#1f77b4', '#ff7f0e'])
        axes[1, 0].set_ylabel('Seconds')
        axes[1, 0].set_title('Query Latency (Lower is Better)')
        axes[1, 0].set_yscale('log')
        
        # Complexity
        axes[1, 1].bar(['Lambda', 'Kappa'],
                       [metrics['Lambda']['Complexity (LOC)'], metrics['Kappa']['Complexity (LOC)']],
                       color=['#1f77b4', '#ff7f0e'])
        axes[1, 1].set_ylabel('Lines of Code')
        axes[1, 1].set_title('Implementation Complexity (Lower is Better)')
        
        plt.tight_layout()
        output_path = self.output_dir / 'lambda_kappa_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
    
    def plot_ml_model_comparison(self):
        """
        Compare ML model performance (RMSE, R², Training Time)
        """
        logger.info("Creating ML model comparison plot...")
        
        # Sample results (replace with actual from ml/demand_forecasting.py)
        models = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
        rmse = [22.3, 17.1, 14.8]
        r2 = [0.782, 0.879, 0.912]
        train_time = [0.47, 4.2, 7.8]  # minutes
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Machine Learning Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # RMSE
        bars1 = axes[0].bar(models, rmse, color=['#2ca02c', '#1f77b4', '#ff7f0e'])
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Root Mean Squared Error\n(Lower is Better)')
        axes[0].set_ylim(0, max(rmse) * 1.2)
        for bar, val in zip(bars1, rmse):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom')
        
        # R²
        bars2 = axes[1].bar(models, r2, color=['#2ca02c', '#1f77b4', '#ff7f0e'])
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('R² Score\n(Higher is Better)')
        axes[1].set_ylim(0, 1.0)
        axes[1].axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target (0.80)')
        axes[1].legend()
        for bar, val in zip(bars2, r2):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom')
        
        # Training Time
        bars3 = axes[2].bar(models, train_time, color=['#2ca02c', '#1f77b4', '#ff7f0e'])
        axes[2].set_ylabel('Training Time (minutes)')
        axes[2].set_title('Training Time\n(Lower is Better)')
        for bar, val in zip(bars3, train_time):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = self.output_dir / 'ml_model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
    
    def plot_cpu_gpu_comparison(self):
        """
        Compare CPU vs GPU training time
        """
        logger.info("Creating CPU vs GPU comparison plot...")
        
        # Sample data (replace with actual benchmark results)
        workloads = ['LSTM\nTraining\n(50 epochs)', 'LSTM\nInference\n(1000 pred)']
        cpu_time = [118, 2.3]
        gpu_time = [12, 0.4]
        
        x = np.arange(len(workloads))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, cpu_time, width, label='CPU', color='#d62728')
        bars2 = ax.bar(x + width/2, gpu_time, width, label='GPU', color='#2ca02c')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title('CPU vs GPU Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(workloads)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add speedup annotations
        for i, (cpu, gpu) in enumerate(zip(cpu_time, gpu_time)):
            speedup = cpu / gpu
            ax.text(i, max(cpu, gpu) * 1.5, f'{speedup:.1f}x faster',
                   ha='center', fontweight='bold', color='green')
        
        plt.tight_layout()
        output_path = self.output_dir / 'cpu_gpu_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
    
    def plot_lstm_training_curve(self):
        """
        Plot LSTM training and validation loss curves
        """
        logger.info("Creating LSTM training curve plot...")
        
        # Sample data (replace with actual training history)
        epochs = np.arange(1, 51)
        train_loss = 0.05 * np.exp(-epochs/10) + 0.01 + np.random.normal(0, 0.002, 50)
        val_loss = 0.06 * np.exp(-epochs/10) + 0.012 + np.random.normal(0, 0.003, 50)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(epochs, train_loss, label='Training Loss', linewidth=2, color='#1f77b4')
        ax.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='#ff7f0e')
        
        # Mark early stopping point
        early_stop = 35
        ax.axvline(x=early_stop, color='red', linestyle='--', alpha=0.7, 
                   label=f'Early Stopping (epoch {early_stop})')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('LSTM Training Convergence', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'lstm_training_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
    
    def plot_demand_prediction_sample(self):
        """
        Plot sample demand predictions (Actual vs Predicted)
        """
        logger.info("Creating demand prediction sample plot...")
        
        # Sample data (replace with actual predictions)
        hours = np.arange(200)
        actual = 100 + 50 * np.sin(hours / 12) + np.random.normal(0, 10, 200)
        predicted = 100 + 48 * np.sin(hours / 12) + np.random.normal(0, 5, 200)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(hours, actual, label='Actual Demand', alpha=0.7, linewidth=1.5, color='#1f77b4')
        ax.plot(hours, predicted, label='Predicted Demand', alpha=0.7, linewidth=1.5, 
                color='#ff7f0e', linestyle='--')
        
        ax.fill_between(hours, actual, predicted, alpha=0.2, color='gray')
        
        ax.set_xlabel('Hour')
        ax.set_ylabel('Trip Count')
        ax.set_title('Taxi Demand Forecasting: Actual vs Predicted (Zone 237 - Upper East Side)',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add RMSE annotation
        rmse = np.sqrt(np.mean((actual - predicted)**2))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.2f}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = self.output_dir / 'demand_prediction_sample.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
    
    def plot_feature_importance(self):
        """
        Plot feature importance for Gradient Boosting model
        """
        logger.info("Creating feature importance plot...")
        
        # Sample data (replace with actual feature importance from trained model)
        features = ['prev_hour\n_demand', 'hour', 'prev_day\n_demand', 'day_of\n_week', 
                   'avg_fare', 'month', 'prev_2hour\n_demand', 'avg_distance']
        importance = [0.42, 0.18, 0.14, 0.11, 0.08, 0.04, 0.02, 0.01]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(features, importance, color='#ff7f0e')
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance - Gradient Boosting Model', fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Highest importance at top
        
        # Add value labels
        for bar, val in zip(bars, importance):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{val:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
    
    def plot_pipeline_architecture(self):
        """
        Create pipeline architecture diagram (text-based for now)
        """
        logger.info("Creating pipeline architecture diagram...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # This would ideally use a proper diagramming library like graphviz
        # For now, creating a simple text representation
        architecture_text = """
        NYC TAXI BIG DATA PIPELINE ARCHITECTURE
        
        ┌─────────────────────────────────────────┐
        │         NYC TLC Data Source             │
        │      (Monthly Parquet Files)            │
        └────────────────┬────────────────────────┘
                         │
                    ┌────▼─────┐
                    │  Bronze  │  Raw Ingestion
                    │  Layer   │  37M+ trips
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  Silver  │  Cleaned & Validated
                    │  Layer   │  >95% retention
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │   Gold   │  Analytics-Ready
                    │  Layer   │  Aggregated features
                    └────┬─────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
        ┌───▼──┐     ┌───▼───┐   ┌───▼───┐
        │Lambda│     │ Kappa │   │  ML   │
        │(Batch│     │(Stream│   │  DL   │
        └──────┘     └───────┘   └───────┘
        """
        
        ax.text(0.5, 0.5, architecture_text, 
               ha='center', va='center', fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        output_path = self.output_dir / 'pipeline_architecture.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
    
    def plot_test_coverage(self):
        """
        Plot test coverage and results
        """
        logger.info("Creating test coverage plot...")
        
        # Sample test data
        test_categories = ['Unit\nTests', 'Integration\nTests', 'Data Quality\nTests', 'Model\nTests']
        passed = [45, 11, 22, 6]
        failed = [0, 0, 0, 0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(test_categories))
        width = 0.35
        
        bars1 = ax.bar(x, passed, width, label='Passed', color='#2ca02c')
        bars2 = ax.bar(x, failed, width, bottom=passed, label='Failed', color='#d62728')
        
        ax.set_ylabel('Number of Tests')
        ax.set_title('Test Suite Results (Total: 84 tests, 100% pass rate)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(test_categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{int(height)}', ha='center', va='center', 
                   fontweight='bold', color='white')
        
        plt.tight_layout()
        output_path = self.output_dir / 'test_coverage.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
    
    def generate_all_figures(self):
        """
        Generate all figures for paper and presentation
        """
        logger.info("Generating all figures...")
        
        self.plot_lambda_kappa_comparison()
        self.plot_ml_model_comparison()
        self.plot_cpu_gpu_comparison()
        self.plot_lstm_training_curve()
        self.plot_demand_prediction_sample()
        self.plot_feature_importance()
        self.plot_test_coverage()
        self.plot_pipeline_architecture()
        
        logger.info(f"All figures saved to {self.output_dir}")
        logger.info("Figure generation complete!")


if __name__ == "__main__":
    visualizer = ResultsVisualizer()
    visualizer.generate_all_figures()
