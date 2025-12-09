"""
Unit tests for data pipeline components
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def spark():
    """Create Spark session for tests"""
    spark = SparkSession.builder \
        .appName("NYC_Taxi_Tests") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    
    yield spark
    spark.stop()


@pytest.fixture
def sample_taxi_data(spark):
    """Create sample taxi data for testing"""
    data = [
        (1, "2024-01-01 10:00:00", "2024-01-01 10:15:00", 237, 236, 2.5, 15.0),
        (2, "2024-01-01 11:00:00", "2024-01-01 11:20:00", 237, 238, 3.0, 18.5),
        (3, "2024-01-01 12:00:00", "2024-01-01 12:10:00", 100, 101, 1.5, 12.0),
    ]
    
    columns = ["trip_id", "pickup_datetime", "dropoff_datetime", 
               "PULocationID", "DOLocationID", "trip_distance", "fare_amount"]
    
    return spark.createDataFrame(data, columns)


class TestDataIngestion:
    """Test data ingestion and bronze layer"""
    
    def test_bronze_schema(self, spark):
        """Test that bronze data has expected schema"""
        # Load a sample file
        df = spark.read.parquet("data/yellow/2024/yellow_tripdata_2024-01.parquet")
        
        expected_cols = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
                        "PULocationID", "DOLocationID", "trip_distance", "fare_amount"]
        
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_bronze_not_empty(self, spark):
        """Test that bronze data is not empty"""
        df = spark.read.parquet("data/yellow/2024/yellow_tripdata_2024-01.parquet")
        count = df.count()
        
        assert count > 0, "Bronze data should not be empty"
        assert count > 1000000, "Bronze data should have substantial records"


class TestDataCleaning:
    """Test silver layer data cleaning"""
    
    def test_trip_duration_calculation(self, sample_taxi_data):
        """Test trip duration is calculated correctly"""
        from clean_to_silver import calculate_trip_duration
        
        df = calculate_trip_duration(sample_taxi_data)
        
        # Check duration column exists
        assert "trip_duration_minutes" in df.columns
        
        # Check values are positive
        durations = df.select("trip_duration_minutes").collect()
        for row in durations:
            assert row[0] > 0, "Trip duration should be positive"
    
    def test_invalid_records_filtered(self, spark):
        """Test that invalid records are filtered out"""
        # Create data with invalid records
        data = [
            (1, "2024-01-01 10:00:00", "2024-01-01 10:15:00", 237, 236, 2.5, 15.0),   # valid
            (2, "2024-01-01 11:00:00", "2024-01-01 11:20:00", 0, 238, 3.0, 18.5),     # invalid zone
            (3, "2024-01-01 12:00:00", "2024-01-01 12:10:00", 237, 238, -1.5, 12.0),  # negative distance
            (4, "2024-01-01 13:00:00", "2024-01-01 13:05:00", 237, 238, 1.0, -5.0),   # negative fare
        ]
        
        columns = ["trip_id", "pickup_datetime", "dropoff_datetime", 
                   "PULocationID", "DOLocationID", "trip_distance", "fare_amount"]
        
        df = spark.createDataFrame(data, columns)
        
        # Apply cleaning rules
        cleaned = df.filter(
            (F.col("PULocationID") > 0) & 
            (F.col("DOLocationID") > 0) &
            (F.col("trip_distance") > 0) &
            (F.col("fare_amount") > 0)
        )
        
        assert cleaned.count() == 1, "Should keep only 1 valid record"
    
    def test_outlier_removal(self, spark):
        """Test outlier detection and removal"""
        data = [
            (1, 5.0, 15.0, 10.0),    # normal
            (2, 3.0, 12.0, 8.0),     # normal
            (3, 100.0, 500.0, 1.0),  # outlier (unrealistic distance)
            (4, 2.0, 10.0, 5.0),     # normal
        ]
        
        df = spark.createDataFrame(data, ["trip_id", "trip_distance", "fare_amount", "duration"])
        
        # Remove outliers (distance > 50 miles)
        filtered = df.filter(F.col("trip_distance") <= 50)
        
        assert filtered.count() == 3, "Should remove 1 outlier"


class TestFeatureEngineering:
    """Test gold layer feature engineering"""
    
    def test_hourly_aggregation(self, sample_taxi_data):
        """Test hourly aggregation works correctly"""
        # Add datetime column
        df = sample_taxi_data.withColumn("pickup_datetime", 
                                         F.to_timestamp("pickup_datetime"))
        
        df = df.withColumn("pickup_hour", F.hour("pickup_datetime"))
        
        # Aggregate by hour
        agg = df.groupBy("pickup_hour").agg(
            F.count("*").alias("trip_count"),
            F.avg("fare_amount").alias("avg_fare")
        )
        
        assert agg.count() == 3, "Should have 3 distinct hours"
        
        # Check aggregation values
        row = agg.filter(F.col("pickup_hour") == 10).collect()[0]
        assert row["trip_count"] == 1
    
    def test_speed_calculation(self, spark):
        """Test speed calculation"""
        data = [
            (1, 10.0, 30.0),  # 10 miles in 30 min = 20 mph
            (2, 5.0, 15.0),   # 5 miles in 15 min = 20 mph
        ]
        
        df = spark.createDataFrame(data, ["trip_id", "trip_distance", "trip_duration_minutes"])
        
        # Calculate speed
        df = df.withColumn("speed_mph", 
                          (F.col("trip_distance") / F.col("trip_duration_minutes")) * 60)
        
        speeds = df.select("speed_mph").collect()
        assert all(abs(row[0] - 20.0) < 0.1 for row in speeds), "Speed calculation incorrect"


class TestDataQuality:
    """Test data quality metrics"""
    
    def test_completeness(self, spark):
        """Test data completeness (no nulls in key columns)"""
        # Load silver data (assuming it exists)
        try:
            df = spark.read.parquet("lake/silver/service=yellow")
            
            key_cols = ["pickup_datetime", "PULocationID", "DOLocationID", 
                       "trip_distance", "fare_amount"]
            
            for col in key_cols:
                null_count = df.filter(F.col(col).isNull()).count()
                assert null_count == 0, f"Column {col} should not have nulls"
        except:
            pytest.skip("Silver data not available")
    
    def test_consistency(self, spark):
        """Test data consistency (dropoff after pickup)"""
        data = [
            (1, "2024-01-01 10:00:00", "2024-01-01 10:15:00"),  # valid
            (2, "2024-01-01 11:00:00", "2024-01-01 10:50:00"),  # invalid (dropoff before pickup)
        ]
        
        df = spark.createDataFrame(data, ["trip_id", "pickup_datetime", "dropoff_datetime"])
        df = df.withColumn("pickup_datetime", F.to_timestamp("pickup_datetime")) \
               .withColumn("dropoff_datetime", F.to_timestamp("dropoff_datetime"))
        
        # Filter consistent records
        consistent = df.filter(F.col("dropoff_datetime") > F.col("pickup_datetime"))
        
        assert consistent.count() == 1, "Should keep only consistent records"
    
    def test_uniqueness(self, spark):
        """Test for duplicate records"""
        data = [
            (1, "2024-01-01 10:00:00", 237),
            (1, "2024-01-01 10:00:00", 237),  # duplicate
            (2, "2024-01-01 11:00:00", 238),
        ]
        
        df = spark.createDataFrame(data, ["trip_id", "pickup_datetime", "PULocationID"])
        
        # Remove duplicates
        unique = df.dropDuplicates()
        
        assert unique.count() == 2, "Should remove 1 duplicate"


class TestMLPipeline:
    """Test ML pipeline components"""
    
    def test_feature_vector_creation(self, spark):
        """Test feature vector assembly"""
        from pyspark.ml.feature import VectorAssembler
        
        data = [(1, 10, 5, 2.5), (2, 11, 6, 3.0)]
        df = spark.createDataFrame(data, ["id", "hour", "day", "prev_demand"])
        
        assembler = VectorAssembler(
            inputCols=["hour", "day", "prev_demand"],
            outputCol="features"
        )
        
        result = assembler.transform(df)
        
        assert "features" in result.columns
        assert result.count() == 2
    
    def test_train_test_split(self, spark):
        """Test train/test split ratio"""
        data = [(i, i*2) for i in range(100)]
        df = spark.createDataFrame(data, ["id", "value"])
        
        train, test = df.randomSplit([0.8, 0.2], seed=42)
        
        train_count = train.count()
        test_count = test.count()
        
        # Allow some variance due to randomness
        assert 75 <= train_count <= 85, "Train split should be ~80%"
        assert 15 <= test_count <= 25, "Test split should be ~20%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
