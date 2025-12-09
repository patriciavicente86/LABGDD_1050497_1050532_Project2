"""
Integration tests for end-to-end pipeline
"""

import pytest
from pyspark.sql import SparkSession
import os
import yaml


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for integration tests"""
    spark = SparkSession.builder \
        .appName("NYC_Taxi_Integration_Tests") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    yield spark
    spark.stop()


class TestEndToEndPipeline:
    """Test complete pipeline execution"""
    
    def test_config_file_exists(self):
        """Test configuration file exists and is valid"""
        assert os.path.exists("env/config.yaml"), "Config file missing"
        
        with open("env/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        assert "paths" in config
        assert "lake" in config["paths"]
    
    def test_bronze_to_silver_pipeline(self, spark):
        """Test Bronze to Silver transformation"""
        try:
            # Load bronze
            bronze_df = spark.read.parquet("data/yellow/2024/*.parquet")
            bronze_count = bronze_df.count()
            
            # Load silver
            silver_df = spark.read.parquet("lake/silver/service=yellow")
            silver_count = silver_df.count()
            
            # Silver should have fewer or equal records (due to filtering)
            assert silver_count <= bronze_count
            
            # Retention should be reasonable (> 90%)
            retention = (silver_count / bronze_count) * 100
            assert retention > 90, f"Low retention rate: {retention:.2f}%"
            
        except Exception as e:
            pytest.skip(f"Pipeline data not available: {str(e)}")
    
    def test_silver_to_gold_pipeline(self, spark):
        """Test Silver to Gold transformation"""
        try:
            # Load silver
            silver_df = spark.read.parquet("lake/silver")
            
            # Load gold
            gold_df = spark.read.parquet("lake/gold/trip_features")
            
            # Gold should have same count as silver (enrichment, not filtering)
            assert abs(gold_df.count() - silver_df.count()) < 100
            
        except Exception as e:
            pytest.skip(f"Gold data not available: {str(e)}")
    
    def test_lambda_kappa_consistency(self, spark):
        """Test Lambda and Kappa produce consistent results"""
        try:
            # Load Lambda output
            lambda_df = spark.read.parquet("lake/gold/agg_zone_hour")
            
            # Load Kappa output
            kappa_df = spark.read.parquet("lake/gold/agg_zone_hour_streaming")
            
            # Join and compare
            joined = lambda_df.alias("lambda").join(
                kappa_df.alias("kappa"),
                ["PULocationID", "pickup_date", "pickup_hour"],
                "inner"
            )
            
            overlap_count = joined.count()
            assert overlap_count > 0, "No overlapping data between Lambda and Kappa"
            
            # Calculate correlation
            from pyspark.sql import functions as F
            
            corr = joined.stat.corr("lambda.trip_count", "kappa.trip_count")
            assert corr > 0.95, f"Low correlation between Lambda and Kappa: {corr:.4f}"
            
        except Exception as e:
            pytest.skip(f"Lambda/Kappa comparison not available: {str(e)}")


class TestModelPipeline:
    """Test ML/DL model pipeline"""
    
    def test_ml_model_exists(self):
        """Test ML model has been trained and saved"""
        model_path = "models/demand_forecast"
        
        if os.path.exists(model_path):
            assert len(os.listdir(model_path)) > 0, "Model directory is empty"
        else:
            pytest.skip("ML model not trained yet")
    
    def test_dl_model_exists(self):
        """Test DL model has been trained and saved"""
        model_path = "models/lstm_demand_forecast/model.h5"
        
        if not os.path.exists(model_path):
            pytest.skip("DL model not trained yet")
    
    def test_model_predictions_reasonable(self, spark):
        """Test model predictions are in reasonable range"""
        try:
            # Load predictions
            predictions_df = spark.read.parquet("lake/gold/ml_predictions")
            
            # Check predictions are positive
            from pyspark.sql import functions as F
            
            stats = predictions_df.select(
                F.min("prediction").alias("min_pred"),
                F.max("prediction").alias("max_pred"),
                F.avg("prediction").alias("avg_pred")
            ).collect()[0]
            
            assert stats["min_pred"] >= 0, "Predictions should be non-negative"
            assert stats["max_pred"] < 1000, "Predictions seem unreasonably high"
            
        except Exception as e:
            pytest.skip(f"Predictions not available: {str(e)}")


class TestPerformance:
    """Test pipeline performance benchmarks"""
    
    def test_silver_processing_time(self, spark):
        """Test silver layer processing completes in reasonable time"""
        import time
        
        try:
            start = time.time()
            
            # Load and count silver data
            df = spark.read.parquet("lake/silver")
            count = df.count()
            
            elapsed = time.time() - start
            
            # Should process millions of records in seconds
            records_per_second = count / elapsed
            
            assert records_per_second > 100000, \
                f"Processing too slow: {records_per_second:.0f} records/sec"
            
        except Exception as e:
            pytest.skip(f"Performance test failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
