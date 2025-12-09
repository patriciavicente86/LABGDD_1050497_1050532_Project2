"""
Demand Forecasting with Spark MLlib
Predicts taxi demand per zone and hour using traditional ML algorithms
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import yaml
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandForecaster:
    """
    Forecasts taxi demand using Spark MLlib regression models
    """
    
    def __init__(self, config_path: str = "env/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.spark = SparkSession.builder \
            .appName("NYC_Taxi_ML_Demand_Forecasting") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
    
    def prepare_features(self, df: DataFrame) -> DataFrame:
        """
        Prepare features for ML models
        - Extract temporal features (hour, day_of_week, month)
        - Aggregate by zone and hour
        - Create lag features
        """
        logger.info("Preparing features for demand forecasting")
        
        # Extract temporal features
        df = df.withColumn("hour", F.hour("pickup_datetime")) \
               .withColumn("day_of_week", F.dayofweek("pickup_datetime")) \
               .withColumn("month", F.month("pickup_datetime")) \
               .withColumn("day_of_month", F.dayofmonth("pickup_datetime"))
        
        # Aggregate trips per zone and hour
        agg_df = df.groupBy("PULocationID", "hour", "day_of_week", "month", "day_of_month") \
                   .agg(
                       F.count("*").alias("trip_count"),
                       F.avg("trip_distance").alias("avg_distance"),
                       F.avg("fare_amount").alias("avg_fare"),
                       F.avg("trip_duration_minutes").alias("avg_duration")
                   )
        
        # Create lag features (previous hour demand)
        from pyspark.sql.window import Window
        
        window_spec = Window.partitionBy("PULocationID").orderBy("month", "day_of_month", "hour")
        
        agg_df = agg_df.withColumn("prev_hour_demand", F.lag("trip_count", 1).over(window_spec)) \
                       .withColumn("prev_2hour_demand", F.lag("trip_count", 2).over(window_spec)) \
                       .withColumn("prev_day_demand", F.lag("trip_count", 24).over(window_spec))
        
        # Fill nulls for lag features
        agg_df = agg_df.fillna(0, subset=["prev_hour_demand", "prev_2hour_demand", "prev_day_demand"])
        
        logger.info(f"Features prepared: {agg_df.count()} records")
        return agg_df
    
    def train_models(self, train_df: DataFrame, test_df: DataFrame) -> Dict:
        """
        Train multiple regression models and compare performance
        """
        logger.info("Training ML models")
        
        # Define feature columns
        feature_cols = ["hour", "day_of_week", "month", "day_of_month", 
                       "prev_hour_demand", "prev_2hour_demand", "prev_day_demand",
                       "avg_distance", "avg_fare", "avg_duration"]
        
        # Vector assembler
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
        scaler = StandardScaler(inputCol="features_raw", outputCol="features")
        
        # Models to compare
        models = {
            "linear_regression": LinearRegression(featuresCol="features", labelCol="trip_count"),
            "random_forest": RandomForestRegressor(featuresCol="features", labelCol="trip_count", 
                                                   numTrees=50, maxDepth=10),
            "gradient_boosted": GBTRegressor(featuresCol="features", labelCol="trip_count", 
                                            maxIter=50, maxDepth=5)
        }
        
        results = {}
        evaluator = RegressionEvaluator(labelCol="trip_count", predictionCol="prediction")
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler, model])
            
            # Train
            model_fitted = pipeline.fit(train_df)
            
            # Predict
            predictions = model_fitted.transform(test_df)
            
            # Evaluate
            rmse = evaluator.setMetricName("rmse").evaluate(predictions)
            mae = evaluator.setMetricName("mae").evaluate(predictions)
            r2 = evaluator.setMetricName("r2").evaluate(predictions)
            
            results[name] = {
                "model": model_fitted,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "predictions": predictions
            }
            
            logger.info(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        return results
    
    def save_best_model(self, results: Dict, output_path: str = "models/demand_forecast"):
        """
        Save the best performing model
        """
        # Find best model by R² score
        best_model_name = max(results.items(), key=lambda x: x[1]["r2"])[0]
        best_model = results[best_model_name]["model"]
        
        logger.info(f"Saving best model: {best_model_name}")
        best_model.write().overwrite().save(output_path)
        
        return best_model_name
    
    def run_pipeline(self, gold_path: str = None) -> Dict:
        """
        Complete ML pipeline for demand forecasting
        """
        if gold_path is None:
            gold_path = f"{self.config['paths']['lake']}/gold/trip_features"
        
        logger.info(f"Loading data from {gold_path}")
        df = self.spark.read.parquet(gold_path)
        
        # Prepare features
        features_df = self.prepare_features(df)
        
        # Split train/test (80/20)
        train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=42)
        
        logger.info(f"Train size: {train_df.count()}, Test size: {test_df.count()}")
        
        # Train models
        results = self.train_models(train_df, test_df)
        
        # Save best model
        best_model = self.save_best_model(results)
        
        logger.info("ML pipeline completed successfully")
        return results


if __name__ == "__main__":
    forecaster = DemandForecaster()
    results = forecaster.run_pipeline()
    
    print("\n=== Model Performance Comparison ===")
    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE:  {metrics['mae']:.2f}")
        print(f"  R²:   {metrics['r2']:.4f}")
