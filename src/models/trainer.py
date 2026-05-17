"""
Model Trainer Module
Handles model training, hyperparameter tuning, and model selection
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import time

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.trained_models = {}

    def _default_config(self) -> Dict:
        return {
            'cv_folds': 5,
            'test_size': 0.2,
            'random_state': 42,
            'models': {
                'logistic_regression': True,
                'decision_tree': True,
                'random_forest': True,
                'xgboost': True,
                'gradient_boosting': True
            }
        }

    def initialize_models(self) -> Dict:
        """Initialize all model objects"""
        logger.info("Initializing models...")

        models = {
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.config['random_state']
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                random_state=self.config['random_state']
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.config['random_state']
            ),
            'XGBoost': self._get_xgboost_model(),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.config['random_state']
            )
        }

        return models

    def _get_xgboost_model(self):
        """Get XGBoost model"""
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['random_state'],
                use_label_encoder=False,
                eval_metric='logloss'
            )
        except ImportError:
            logger.warning("XGBoost not available, using Gradient Boosting instead")
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config['random_state']
            )

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train all models"""
        logger.info("Training models...")

        self.models = self.initialize_models()
        trained_models = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            start_time = time.time()

            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                logger.info(f"{name} trained in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")

        self.trained_models = trained_models
        return trained_models

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all trained models"""
        logger.info("Evaluating models...")

        results = []

        for name, model in self.trained_models.items():
            logger.info(f"Evaluating {name}...")

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            cv_scores = cross_val_score(
                model, X_test, y_test,
                cv=self.config['cv_folds'],
                scoring='roc_auc'
            )

            result = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
                'CV-Score': cv_scores.mean(),
                'CV-Std': cv_scores.std()
            }

            results.append(result)

        self.results = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
        logger.info(f"\n{self.results.to_string()}")

        return self.results

    def get_best_model(self) -> Tuple:
        """Get the best performing model"""
        if self.results.empty:
            logger.warning("No results available. Train models first.")
            return None, None

        best_idx = self.results['ROC-AUC'].idxmax()
        best_name = self.results.loc[best_idx, 'Model']
        best_model = self.trained_models[best_name]

        self.best_model = best_model
        self.best_model_name = best_name

        logger.info(f"Best model: {best_name} with ROC-AUC: {self.results.loc[best_idx, 'ROC-AUC']:.4f}")
        return best_model, best_name

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                              model_name: str = 'Random Forest') -> Dict:
        """Perform hyperparameter tuning"""
        logger.info(f"Performing hyperparameter tuning for {model_name}...")

        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=self.config['random_state'])

        elif model_name == 'XGBoost':
            try:
                from xgboost import XGBClassifier
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
                model = XGBClassifier(random_state=self.config['random_state'], use_label_encoder=False, eval_metric='logloss')
            except ImportError:
                logger.warning("XGBoost not available")
                return {}

        else:
            logger.warning(f"No hyperparameter grid defined for {model_name}")
            return {}

        grid_search = RandomizedSearchCV(
            model, param_grid,
            n_iter=20,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.config['random_state']
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }

    def save_models(self, path: str = "models/") -> None:
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)

        for name, model in self.trained_models.items():
            filename = name.lower().replace(' ', '_') + '_model.pkl'
            joblib.dump(model, f"{path}{filename}")
            logger.info(f"Saved {name} to {path}{filename}")

        if self.best_model:
            joblib.dump(self.best_model, f"{path}best_model.pkl")
            logger.info(f"Saved best model to {path}best_model.pkl")

    def load_models(self, path: str = "models/") -> Dict:
        """Load trained models from disk"""
        import os
        models = {}

        for filename in os.listdir(path):
            if filename.endswith('_model.pkl') or filename == 'best_model.pkl':
                name = filename.replace('_model.pkl', '').replace('.pkl', '').replace('_', ' ').title()
                models[name] = joblib.load(f"{path}{filename}")
                logger.info(f"Loaded {name} from {path}{filename}")

        self.trained_models = models
        return models

    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        if model_name is None:
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)

        if model is None:
            logger.warning("No model available")
            return pd.DataFrame()

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return pd.DataFrame({
                'Feature': range(len(importance)),
                'Importance': importance
            }).sort_values('Importance', ascending=False)

        logger.warning(f"Model {model_name} does not support feature importance")
        return pd.DataFrame()

    def get_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series,
                             model_name: str = None) -> np.ndarray:
        """Get confusion matrix for a model"""
        if model_name is None:
            model = self.best_model
        else:
            model = self.trained_models.get(model_name, self.best_model)

        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def get_classification_report(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   model_name: str = None) -> str:
        """Get classification report"""
        if model_name is None:
            model = self.best_model
        else:
            model = self.trained_models.get(model_name, self.best_model)

        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred)