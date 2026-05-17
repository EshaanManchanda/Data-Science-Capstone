"""
SpaceX Unified Machine Learning Pipeline
Complete ML pipeline from all notebooks
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

logger = logging.getLogger(__name__)


class SpaceXModelTrainer:
    """Complete ML pipeline for SpaceX landing prediction"""

    def __init__(self):
        self.models = {}
        self.results = None
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.label_encoders = {}

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'class') -> Tuple:
        """Prepare features for ML"""
        df = df.copy()

        if 'Grid Fins' in df.columns:
            df['Grid_Fins'] = df['Grid Fins'].astype(int)
        if 'Legs' in df.columns:
            df['Legs'] = df['Legs'].astype(int)

        numeric_features = ['Payload Mass (kg)', 'Year']
        if 'Grid_Fins' in df.columns:
            numeric_features.append('Grid_Fins')
        if 'Legs' in df.columns:
            numeric_features.append('Legs')

        categorical_features = []
        if 'Booster Version' in df.columns:
            categorical_features.append('Booster Version')
        if 'Launch Site' in df.columns:
            categorical_features.append('Launch Site')
        if 'Orbit' in df.columns:
            categorical_features.append('Orbit')

        X = df[numeric_features + categorical_features] if categorical_features else df[numeric_features]
        y = df[target_col] if target_col in df.columns else None

        if y is None:
            raise ValueError(f"Target column '{target_col}' not found")

        return X, y, numeric_features, categorical_features

    def create_preprocessor(self, numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        transformers = []

        if numeric_features:
            transformers.append(('num', StandardScaler(), numeric_features))

        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))

        return ColumnTransformer(transformers=transformers)

    def initialize_models(self) -> Dict:
        """Initialize all ML models"""
        return {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series,
                          numeric_features: List[str],
                          categorical_features: List[str],
                          test_size: float = 0.2) -> pd.DataFrame:
        """Train and evaluate all models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        preprocessor = self.create_preprocessor(numeric_features, categorical_features)
        models = self.initialize_models()

        results = []

        for name, model in models.items():
            logger.info(f"Training {name}...")

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')

            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_proba),
                'CV-Score': cv_scores.mean(),
                'CV-Std': cv_scores.std(),
                'Pipeline': pipeline
            })

            self.models[name] = pipeline

        self.results = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
        logger.info(f"\n{self.results[['Model', 'Accuracy', 'ROC-AUC']].to_string()}")

        return self.results

    def get_best_model(self) -> Tuple:
        """Get the best performing model"""
        if self.results is None:
            return None, None

        best_idx = self.results['ROC-AUC'].idxmax()
        self.best_model_name = self.results.loc[best_idx, 'Model']
        self.best_model = self.results.loc[best_idx, 'Pipeline']

        return self.best_model, self.best_model_name

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             model_name: str = 'Random Forest',
                             numeric_features: List[str] = None,
                             categorical_features: List[str] = None) -> Dict:
        """Perform hyperparameter tuning"""
        preprocessor = self.create_preprocessor(
            numeric_features or [],
            categorical_features or []
        )

        if model_name == 'Random Forest':
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42)

        elif model_name == 'Gradient Boosting':
            param_grid = {
                'classifier__n_estimators': [50, 100, 150],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            }
            model = GradientBoostingClassifier(random_state=42)

        else:
            logger.warning(f"No tuning defined for {model_name}")
            return {}

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }

    def get_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
        """Get confusion matrix for best model"""
        if self.best_model is None:
            self.get_best_model()

        y_pred = self.best_model.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def get_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple:
        """Get ROC curve data"""
        if self.best_model is None:
            self.get_best_model()

        y_proba = self.best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        return fpr, tpr, auc_score

    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """Get feature importance from best model"""
        if self.best_model is None:
            self.get_best_model()

        if hasattr(self.best_model.named_steps['classifier'], 'feature_importances_'):
            importance = self.best_model.named_steps['classifier'].feature_importances_

            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importance))]

            return pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

        return pd.DataFrame()

    def save_model(self, path: str = "models/spacex_model.pkl"):
        """Save best model"""
        if self.best_model is None:
            self.get_best_model()

        joblib.dump(self.best_model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str = "models/spacex_model.pkl"):
        """Load saved model"""
        self.best_model = joblib.load(path)
        logger.info(f"Model loaded from {path}")

    def predict(self, features: pd.DataFrame) -> Tuple:
        """Make prediction with best model"""
        if self.best_model is None:
            self.get_best_model()

        # Add missing columns with default values
        features = features.copy()
        required_cols = ['Payload Mass (kg)', 'Year']
        categorical_required = []
        
        # Check what the preprocessor expects
        if hasattr(self.best_model, 'named_steps'):
            preprocessor = self.best_model.named_steps.get('preprocessor')
            if preprocessor:
                # Get feature names from transformers
                for name, transformer, cols in preprocessor.transformers_:
                    if hasattr(transformer, 'get_feature_names_out'):
                        try:
                            out_cols = transformer.get_feature_names_out(cols)
                            for col in out_cols:
                                if col not in features.columns:
                                    if 'Booster' in col or 'Launch' in col or 'Orbit' in col:
                                        features[col] = 'Unknown'
                                    else:
                                        features[col] = 0
                        except:
                            pass

        prediction = self.best_model.predict(features)[0]
        probability = self.best_model.predict_proba(features)[0]

        return prediction, probability


def train_spacex_model(data_path: str, output_path: str = "models/") -> Dict:
    """Main function to train SpaceX model"""
    import os
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(data_path)

    trainer = SpaceXModelTrainer()
    X, y, num_feat, cat_feat = trainer.prepare_features(df)

    results = trainer.train_and_evaluate(X, y, num_feat, cat_feat)

    best_model, best_name = trainer.get_best_model()
    logger.info(f"Best model: {best_name}")

    trainer.save_model(f"{output_path}spacex_model.pkl")

    return {
        'results': results,
        'best_model': best_name,
        'metrics': results[results['Model'] == best_name].to_dict('records')[0]
    }


if __name__ == "__main__":
    results = train_spacex_model("data/raw/spacex_data.csv")
    print(results)