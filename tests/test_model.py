"""
Test Module
Unit tests for the project
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.trainer import ModelTrainer
from src.models.predictor import Predictor


class TestDataLoader:
    """Test DataLoader class"""

    def test_load_raw_data(self):
        loader = DataLoader(data_path=str(PROJECT_ROOT / "data/raw"))
        df = loader.load_raw_data()
        assert df is not None
        assert df.shape[0] > 0

    def test_data_summary(self):
        loader = DataLoader(data_path=str(PROJECT_ROOT / "data/raw"))
        df = loader.load_raw_data()
        summary = loader.get_data_summary(df)
        assert 'shape' in summary
        assert 'columns' in summary


class TestDataValidator:
    """Test DataValidator class"""

    def test_validate_data(self):
        loader = DataLoader(data_path=str(PROJECT_ROOT / "data/raw"))
        df = loader.load_raw_data()
        validator = DataValidator()
        results = validator.validate_data(df)
        assert results is not None
        assert 'shape' in results


class TestPreprocessing:
    """Test preprocessing pipeline"""

    def test_handle_missing_values(self):
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': ['a', 'b', 'c', None]
        })
        preprocessor = PreprocessingPipeline()
        result = preprocessor.handle_missing_values(df)
        assert result.isnull().sum().sum() == 0

    def test_remove_duplicates(self):
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        preprocessor = PreprocessingPipeline()
        result = preprocessor.remove_duplicates(df)
        assert len(result) == 3


class TestModelTrainer:
    """Test ModelTrainer class"""

    def test_initialize_models(self):
        trainer = ModelTrainer()
        models = trainer.initialize_models()
        assert len(models) > 0
        assert 'Logistic Regression' in models
        assert 'Random Forest' in models

    def test_train_models(self):
        X_train = pd.DataFrame({
            'col1': np.random.rand(50),
            'col2': np.random.rand(50)
        })
        y_train = pd.Series(np.random.randint(0, 2, 50))
        trainer = ModelTrainer()
        models = trainer.train_models(X_train, y_train)
        assert len(models) > 0


class TestPredictor:
    """Test Predictor class"""

    def test_predictor_creation(self):
        predictor = Predictor()
        assert predictor is not None


def test_imports():
    """Test that all modules can be imported"""
    from src import DataLoader, PreprocessingPipeline, ModelTrainer, Predictor
    assert DataLoader is not None
    assert PreprocessingPipeline is not None
    assert ModelTrainer is not None
    assert Predictor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])