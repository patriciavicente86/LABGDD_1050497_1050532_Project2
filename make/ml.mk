# Makefile targets for ML/DL pipeline

.PHONY: ml dl test test-unit test-integration test-quality test-models benchmarks

# Machine Learning targets
ml: ml-train ml-evaluate

ml-train:
	@echo "=== Training ML models (Spark MLlib) ==="
	python ml/demand_forecasting.py

ml-evaluate:
	@echo "=== Evaluating ML models ==="
	python -c "from ml.demand_forecasting import DemandForecaster; f = DemandForecaster(); print('ML models evaluated')"

# Deep Learning targets
dl: dl-check-gpu dl-train dl-evaluate

dl-check-gpu:
	@echo "=== Checking GPU availability ==="
	python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

dl-train:
	@echo "=== Training DL model (LSTM) ==="
	python dl/lstm_forecaster.py

dl-evaluate:
	@echo "=== Evaluating DL model ==="
	@echo "Model evaluation completed during training"

# Testing targets
test: test-unit test-integration test-quality

test-unit:
	@echo "=== Running unit tests ==="
	pytest tests/test_pipeline.py -v

test-integration:
	@echo "=== Running integration tests ==="
	pytest tests/test_integration.py -v

test-quality:
	@echo "=== Running data quality tests ==="
	pytest tests/test_pipeline.py::TestDataQuality -v

test-models:
	@echo "=== Running model validation tests ==="
	pytest tests/test_integration.py::TestModelPipeline -v

test-coverage:
	@echo "=== Running tests with coverage ==="
	pytest --cov=src --cov=ml --cov=dl --cov-report=html --cov-report=term

# Benchmark targets
benchmarks: benchmark-cpu-gpu benchmark-lambda-kappa

benchmark-cpu-gpu:
	@echo "=== Benchmarking CPU vs GPU ==="
	python benchmarks/benchmark_cpu_gpu.py

benchmark-lambda-kappa:
	@echo "=== Benchmarking Lambda vs Kappa ==="
	python src/compare_lambda_kappa.py

# Complete ML/DL pipeline
ml-pipeline: silver gold ml dl test
	@echo "=== ML/DL pipeline completed ==="

# All targets (complete workflow)
all-ml: clean-models ml-pipeline
	@echo "=== Complete ML/DL workflow finished ==="

# Cleanup
clean-models:
	@echo "=== Cleaning old models ==="
	rm -rf models/demand_forecast models/lstm_demand_forecast

clean-test:
	@echo "=== Cleaning test artifacts ==="
	rm -rf .pytest_cache htmlcov .coverage

clean-all: clean-models clean-test
	@echo "=== All artifacts cleaned ==="
