"""
Model Explainability Module
Provides interpretability for the trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Explain model predictions and feature contributions"""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
                try:
                    feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
                except:
                    feature_names = self.feature_names
            else:
                feature_names = self.feature_names

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

            return importance_df
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()

    def get_coefficients(self):
        """Get coefficients for linear models"""
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            coef_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': coef
            }).sort_values('Coefficient', key=abs, ascending=False)
            return coef_df
        return pd.DataFrame()

    def predict_explanation(self, X):
        """Explain a single prediction"""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]

            if hasattr(self.model, 'feature_importances_'):
                importance = self.get_feature_importance()
                top_features = importance.head(5)['Feature'].tolist()

                return {
                    'prediction': 'Success' if proba[1] > 0.5 else 'Failure',
                    'probability_success': proba[1],
                    'probability_failure': proba[0],
                    'top_features': top_features,
                    'explanation': self._generate_explanation(top_features, proba[1])
                }
        return {}

    def _generate_explanation(self, top_features, probability):
        """Generate human-readable explanation"""
        if probability > 0.8:
            confidence = "very confident"
        elif probability > 0.5:
            confidence = "moderately confident"
        else:
            confidence = "not very confident"

        explanation = f"The model is {confidence} that this launch will be successful. "
        explanation += f"Key factors include: {', '.join(top_features[:3])}."

        return explanation

    def plot_feature_importance(self, save_path=None):
        """Plot feature importance"""
        importance = self.get_feature_importance()

        if importance.empty:
            logger.warning("No importance data available")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance['Feature'], importance['Importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        ax.invert_yaxis()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def compare_models(self, models_dict):
        """Compare feature importance across models"""
        comparison = {}

        for name, model in models_dict.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                    try:
                        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                    except:
                        feature_names = self.feature_names
                else:
                    feature_names = self.feature_names

                comparison[name] = dict(zip(feature_names, importance))

        return pd.DataFrame(comparison).fillna(0)


def analyze_prediction(model_path, test_data):
    """Quick analysis function"""
    model = joblib.load(model_path)

    features = pd.DataFrame([test_data])
    explainer = ModelExplainer(model, list(features.columns))

    return explainer.predict_explanation(features)


if __name__ == "__main__":
    print("Model Explainability Module")
    print("Use: from src.models.explainability import ModelExplainer")