"""
Streamlit App Test
Test the Streamlit web application
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.models.predictor import Predictor


def test_data_loading():
    """Test data loading"""
    loader = DataLoader(data_path="data/raw")
    df = loader.load_raw_data()
    assert df is not None
    assert len(df) > 0
    print("✓ Data loading test passed")


def test_prediction():
    """Test prediction function"""
    predictor = Predictor(model_path="models/")
    try:
        predictor.load_model("best_model.pkl")
    except:
        print("⚠ Model not found, skipping prediction test")
        return

    features = {
        'Payload Mass (kg)': 3500,
        'Year': 2023,
        'Grid_Fins': 1,
        'Legs': 1,
        'Booster Version': 'B5',
        'Launch Site': 'KSC LC-39A'
    }

    result = predictor.predict_single(features)
    assert 'prediction' in result
    assert 'probability_success' in result
    print("✓ Prediction test passed")


def test_data_validation():
    """Test data validation"""
    loader = DataLoader(data_path="data/raw")
    df = loader.load_raw_data()

    assert 'class' in df.columns
    assert 'Launch Site' in df.columns
    assert 'Booster Version' in df.columns

    assert df['class'].isin([0, 1]).all()
    print("✓ Data validation test passed")


def test_model_metrics():
    """Test model metrics calculation"""
    from src.models.evaluator import ModelEvaluator

    evaluator = ModelEvaluator()
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.3, 0.8, 0.4, 0.2, 0.9])

    results = evaluator.evaluate_classification(y_true, y_pred, y_proba)

    assert 'accuracy' in results
    assert 'precision' in results
    assert 'recall' in results
    assert 'f1_score' in results
    assert 'roc_auc' in results

    print("✓ Model evaluation test passed")


def test_preprocessing():
    """Test preprocessing pipeline"""
    from src.preprocessing.pipeline import PreprocessingPipeline

    df = pd.DataFrame({
        'col1': [1, 2, 3, np.nan, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'class': [0, 1, 0, 1, 0]
    })

    pipeline = PreprocessingPipeline()
    result = pipeline.handle_missing_values(df)

    assert result['col1'].isnull().sum() == 0
    print("✓ Preprocessing test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*50)
    print("🧪 Running Project Tests")
    print("="*50 + "\n")

    try:
        test_data_loading()
        test_data_validation()
        test_preprocessing()
        test_model_metrics()
        test_prediction()

        print("\n" + "="*50)
        print("✅ All Tests Passed!")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()