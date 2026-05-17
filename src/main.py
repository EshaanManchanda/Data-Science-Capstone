"""
Main Training Script
Trains the model and saves it for deployment
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.data.loader import DataLoader
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    """Main training pipeline"""
    logger.info("Starting model training pipeline...")

    loader = DataLoader(data_path=str(PROJECT_ROOT / "data/raw"))
    df = loader.load_raw_data()
    logger.info(f"Loaded data: {df.shape}")

    preprocessor = PreprocessingPipeline()
    X, y = preprocessor.load_and_preprocess(df)
    logger.info(f"Preprocessed data: X={X.shape}, y={y.shape}")

    X_encoded = preprocessor.encode_categorical(X, fit=True)
    X_scaled = preprocessor.scale_features(X_encoded, fit=True)

    X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    trainer = ModelTrainer()
    trained_models = trainer.train_models(X_train, y_train)
    logger.info("Models trained successfully")

    results = trainer.evaluate_models(X_test, y_test)
    logger.info(f"\n{results.to_string()}")

    best_model, best_name = trainer.get_best_model()
    logger.info(f"Best model: {best_name}")

    os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
    trainer.save_models(str(PROJECT_ROOT / "models"))
    preprocessor.save_preprocessors(str(PROJECT_ROOT / "models"))

    logger.info("Training complete! Models saved.")

    return results


if __name__ == "__main__":
    main()