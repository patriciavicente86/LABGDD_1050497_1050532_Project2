"""
Test configuration and fixtures
"""

import pytest
import os


def pytest_configure(config):
    """Configure pytest"""
    # Set environment variables for tests
    os.environ['PYSPARK_PYTHON'] = 'python'
    os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'


@pytest.fixture(scope="session")
def test_data_path():
    """Return path to test data"""
    return os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture(scope="session")
def config_path():
    """Return path to config file"""
    return os.path.join(os.path.dirname(__file__), '..', 'env', 'config.yaml')
