"""
Predictor Module
Handles model predictions and inference
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import joblib

logger = logging.getLogger(__name__)


class Predictor:
    """Handles model predictions and inference"""

    def __init__(self, model_path: str = "models/"):
        self.model = None
        self.model_path = model_path
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []

    def load_model(self, model_name: str = "best_model.pkl") -> None:
        """Load trained model from disk"""
        try:
            self.model = joblib.load(f"{self.model_path}{model_name}")
            logger.info(f"Loaded model: {model_name}")
        except FileNotFoundError:
            logger.error(f"Model not found: {model_name}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def load_preprocessors(self) -> None:
        """Load preprocessors"""
        try:
            self.scaler = joblib.load(f"{self.model_path}scaler.pkl")
            self.label_encoders = joblib.load(f"{self.model_path}label_encoders.pkl")
            logger.info("Loaded preprocessors")
        except:
            logger.warning("Preprocessors not found, skipping preprocessing")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on input data"""
        if self.model is None:
            logger.error("No model loaded")
            raise ValueError("Model not loaded")

        X_processed = self.preprocess_input(X)
        predictions = self.model.predict(X_processed)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            logger.error("No model loaded")
            raise ValueError("Model not loaded")

        X_processed = self.preprocess_input(X)
        return self.model.predict_proba(X_processed)

    def predict_single(self, features: Dict) -> Dict:
        """Predict for a single sample"""
        X = pd.DataFrame([features])
        prediction = self.predict(X)[0]
        probability = self.predict_proba(X)[0]

        return {
            'prediction': int(prediction),
            'probability_success': float(probability[1]),
            'probability_failure': float(probability[0]),
            'result': 'SUCCESS' if prediction == 1 else 'FAILURE'
        }

    def preprocess_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data"""
        X = X.copy()

        for col in X.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))

        if self.scaler is not None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        return X

    def get_prediction_summary(self, X: pd.DataFrame, y_true: np.ndarray = None) -> Dict:
        """Get comprehensive prediction summary"""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        summary = {
            'total_predictions': len(predictions),
            'predictions_distribution': {
                'success': int((predictions == 1).sum()),
                'failure': int((predictions == 0).sum())
            },
            'average_probability_success': float(probabilities[:, 1].mean()),
            'average_probability_failure': float(probabilities[:, 0].mean())
        }

        if y_true is not None:
            from sklearn.metrics import accuracy_score
            summary['accuracy'] = accuracy_score(y_true, predictions)

        return summary

    def batch_predict(self, data: List[Dict]) -> List[Dict]:
        """Predict for multiple samples"""
        results = []
        for features in data:
            result = self.predict_single(features)
            results.append(result)
        return results

    def explain_prediction(self, X: pd.DataFrame) -> Dict:
        """Explain prediction using SHAP or feature importance"""
        if self.model is None:
            logger.error("No model loaded")
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = dict(zip(X.columns, importance))
            return {
                'feature_importance': feature_importance,
                'top_features': sorted(feature_importance.items(),
                                       key=lambda x: x[1], reverse=True)[:5]
            }

        return {'message': 'Feature importance not available for this model'}

    def save_predictions(self, predictions: np.ndarray, filename: str = "predictions.csv") -> None:
        """Save predictions to CSV"""
        df = pd.DataFrame({
            'prediction': predictions,
            'label': ['Success' if p == 1 else 'Failure' for p in predictions]
        })
        df.to_csv(filename, index=False)
        logger.info(f"Saved predictions to {filename}")