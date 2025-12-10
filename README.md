# NYC Taxi ML/DL Pipeline — Lambda, Kappa & Predictive Analytics

> **LABGDD Project 2**: Big Data pipeline with **Machine Learning**, **Deep Learning (GPU)**, and comprehensive **testing framework**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.5%2B-orange)](https://spark.apache.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-ff6f00)](https://tensorflow.org)
[![Tests](https://img.shields.io/badge/Tests-Pytest-green)](https://pytest.org)

---

## 🎯 Project Overview

This project extends **Project 1** (Lambda vs Kappa architecture comparison) with:

- ✅ **Machine Learning** (Spark MLlib): Random Forest, GBT, Linear Regression for demand forecasting
- ✅ **Deep Learning** (TensorFlow + GPU): LSTM time series forecasting  
- ✅ **Testing Framework**: Unit + Integration + Data Quality tests (**30% of grade**)
- ✅ **Performance Benchmarks**: CPU vs GPU, ML vs DL comparisons
- ✅ **Topics Integration**: Parallel Computing, GPU Computing, Cloud Architecture

---

## 📁 Project Structure

```
LABGDD_1050497_1050532_Project2/
├── src/                       # Core pipeline (inherited from Project 1)
│   ├── clean_to_silver.py     # Bronze → Silver transformation
│   ├── features_to_gold.py    # Silver → Gold feature engineering
│   ├── kappa_driver.py        # Streaming pipeline (Kappa)
│   ├── lambda_driver.py       # Batch pipeline (Lambda)
│   ├── compare_lambda_kappa.py# Lambda vs Kappa comparison
│   ├── metrics.py             # Pipeline metrics and reports
│   ├── figures.py             # Visualization generation
│   ├── ingest_bronze.py       # Raw data ingestion
│   └── probe_stream.py        # Streaming data probe
├── ml/                        # Machine Learning module
│   └── demand_forecasting.py  # Spark MLlib models
├── dl/                        # Deep Learning module
│   └── lstm_forecaster.py     # TensorFlow LSTM model
├── tests/                     # Testing framework (30% grade!)
│   ├── conftest.py            # Pytest configuration
│   ├── test_pipeline.py       # Unit tests
│   └── test_integration.py    # Integration tests
├── benchmarks/                # Performance benchmarks
│   └── performance_benchmark.py
├── notebooks/                 # Jupyter notebooks
│   └── NYC_Taxi_Report.ipynb
├── data/                      # Raw data (parquet files)
│   ├── yellow/2024/
│   └── green/2024/
├── lake/                      # Data lake (Bronze/Silver/Gold)
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── docker/                    # Container configuration
│   ├── Dockerfile
│   └── requirements.txt
├── env/                       # Configuration
│   └── config.yaml            # Unified config (paths, ML, DL, tests)
├── make/                      # Makefile modules
│   ├── pipeline.mk            # Pipeline targets
│   ├── ml.mk                  # ML/DL/test targets
│   ├── ingest.mk
│   ├── bench.mk
│   └── validate.mk
├── Makefile                   # Main makefile
├── pytest.ini                 # Pytest configuration
└── README.md
```

---

## 🏗️ Architecture

```
NYC Taxi Data (37M+ trips)
         │
    ┌────▼────┐
    │  Bronze │  Raw data ingestion
    └────┬────┘
         │
    ┌────▼────┐
    │  Silver │  Data cleaning + validation
    └────┬────┘
         │
    ┌────▼────┐
    │   Gold  │  Analytics-ready features
    └────┬────┘
         │
    ┌────┴────────────────┐
    │                     │
┌───▼────┐           ┌────▼────┐
│ Lambda │           │  Kappa  │
│ (Batch)│           │(Stream) │
└───┬────┘           └────┬────┘
    │                     │
    └──────────┬──────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐           ┌────▼────┐
│ Spark  │           │  LSTM   │
│ MLlib  │           │  (GPU)  │
└────────┘           └─────────┘
```

---

## 🔧 Setup

### Prerequisites
- Python 3.10+
- PySpark 3.5+
- TensorFlow 2.15+ (optional, for DL)
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

```bash
cd LABGDD_1050497_1050532_Project2

# Install dependencies
pip install -r docker/requirements.txt
```

### Verify Installation

```bash
# Check Spark
python -c "from pyspark.sql import SparkSession; print('Spark OK')"

# Check TensorFlow (optional)
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## 🚀 Usage

### Quick Start (Data Already Processed)

If you have the `lake/` folder with processed data:

```bash
# Check pipeline metrics
make metrics

# Generate figures
make figures

# Run ML models
make ml

# Run DL model (GPU recommended)
make dl

# Run all tests
make test
```

### Full Pipeline (From Scratch)

```bash
# 1. Ingest raw data to Bronze
make ingest

# 2. Clean and transform to Silver
make silver

# 3. Feature engineering to Gold
make gold

# 4. Train ML models
make ml

# 5. Train DL model
make dl

# 6. Run tests
make test
```

### Lambda vs Kappa Comparison

```bash
# Start Kappa streaming (in separate terminal)
make kappa_start

# Seed data to streaming path
make kappa_seed

# Compare results
make compare
```

---

## 🤖 Machine Learning

### Models Implemented

| Model | Algorithm | Library |
|-------|-----------|---------|
| Linear Regression | OLS | Spark MLlib |
| Random Forest | Ensemble | Spark MLlib |
| Gradient Boosted Trees | Boosting | Spark MLlib |

### Features Used

- **Temporal**: hour, day_of_week, month, day_of_month
- **Lag Features**: prev_hour_demand, prev_2hour_demand, prev_day_demand
- **Trip Metrics**: avg_distance, avg_fare, avg_duration

### Run ML Pipeline

```bash
make ml
```

Or programmatically:

```python
from ml.demand_forecasting import DemandForecaster

forecaster = DemandForecaster()
results = forecaster.run_pipeline()

for name, metrics in results.items():
    print(f"{name}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f}")
```

---

## 🧠 Deep Learning

### LSTM Architecture

```
Input (24 hours) → LSTM(128) → Dropout(0.2) 
                → LSTM(64)  → Dropout(0.2) 
                → Dense(32) → Dense(1) → Output
```

### Configuration (from `env/config.yaml`)

```yaml
dl:
  use_gpu: true
  lookback: 24
  epochs: 50
  batch_size: 32
  lstm_units_1: 128
  lstm_units_2: 64
  dropout_rate: 0.2
  early_stopping_patience: 10
```

### Run DL Pipeline

```bash
make dl
```

Or programmatically:

```python
from dl.lstm_forecaster import LSTMDemandForecaster

forecaster = LSTMDemandForecaster(use_gpu=True)
metrics = forecaster.run_pipeline(zone_id=237, lookback=24)

print(f"LSTM: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f}")
```

---

## ✅ Testing (30% of Grade!)

### Test Categories

| Category | File | Description |
|----------|------|-------------|
| Unit Tests | `test_pipeline.py` | Schema, cleaning, features |
| Integration Tests | `test_integration.py` | End-to-end pipeline |
| Data Quality | `test_pipeline.py` | Completeness, consistency |

### Run Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# With coverage report
make test-coverage
```

### Test Coverage

```bash
pytest --cov=src --cov=ml --cov=dl --cov-report=html
```

---

## ⚙️ Configuration

All configuration is centralized in `env/config.yaml`:

```yaml
# Paths (supports both flat and nested format)
paths:
  lake: "lake"
  bronze: "lake/bronze"
  silver: "lake/silver"
  gold: "lake/gold"
  data: "data"

# Data Quality
data_quality:
  min_trip_duration: 1
  max_trip_duration: 180
  min_trip_distance: 0.1
  max_trip_distance: 100

# Spark
spark:
  master: "local[*]"
  driver_memory: "4g"

# ML Configuration
ml:
  test_split: 0.2
  random_seed: 42

# DL Configuration  
dl:
  use_gpu: true
  epochs: 50
  lookback: 24
```

---

## 📊 Results

### Data Pipeline

| Layer | Records | Retention |
|-------|---------|-----------|
| Bronze | 38.7M | 100% |
| Silver | 37.0M | 95.4% |
| Gold | 37.0M | 100% |

### Model Performance (Expected)

| Model | RMSE | R² | Training Time |
|-------|------|-----|---------------|
| Linear Regression | ~22 | ~0.78 | ~30 sec |
| Random Forest | ~17 | ~0.88 | ~5 min |
| Gradient Boosting | ~15 | ~0.91 | ~8 min |
| LSTM (GPU) | ~13 | ~0.93 | ~2.5 min |

---

## 📚 Topics Coverage

| Course Topic | Implementation |
|--------------|----------------|
| Parallel Computing | ✅ Spark distributed processing |
| GPU Computing | ✅ TensorFlow GPU acceleration |
| Cloud Computing | ✅ Docker containerization |
| Hadoop/Spark | ✅ Lambda + Kappa architectures |
| Machine Learning | ✅ Spark MLlib (RF, GBT, LR) |
| Deep Learning | ✅ LSTM time series forecasting |

---

## 🎓 AI Assistance Disclosure

This project was developed with assistance from:
- GitHub Copilot (code completion and suggestions)
- ChatGPT (architecture planning, documentation)

All code has been reviewed, tested, and validated by the team.

---

## 👥 Authors

**Students**: 1050497, 1050532  
**Course**: LABGDD - Big Data Laboratory  
**Institution**: ISEP - MEI Data Engineering  
**Year**: 2024/2025

---

## 📄 License

Academic project for educational purposes only.

---

## 🔗 Quick Reference

```bash
# Common commands
make metrics          # View data statistics
make figures          # Generate visualizations
make ml               # Train ML models
make dl               # Train DL model
make test             # Run all tests
make test-coverage    # Tests with coverage

# Pipeline commands
make silver           # Bronze → Silver
make gold             # Silver → Gold
make kappa_start      # Start streaming

# Cleanup
make clean            # Remove checkpoints
make clean-models     # Remove trained models
```
