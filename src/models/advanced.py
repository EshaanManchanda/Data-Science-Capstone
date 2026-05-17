"""
Advanced Model Training with Hyperparameter Tuning
Uses GridSearchCV and RandomizedSearchCV for optimization
"""

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedModelTrainer:
    """Advanced model training with hyperparameter tuning"""

    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.results = {}

    def prepare_data(self, df):
        """Prepare data for training"""
        df = df.copy()
        df['Grid_Fins'] = df['Grid Fins'].astype(int)
        df['Legs'] = df['Legs'].astype(int)

        numeric_features = ['Payload Mass (kg)', 'Year', 'Grid_Fins', 'Legs']
        categorical_features = ['Booster Version', 'Launch Site']

        X = df[numeric_features + categorical_features]
        y = df['class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test, numeric_features, categorical_features

    def create_pipeline(self, model, numeric_features, categorical_features):
        """Create preprocessing + model pipeline"""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    def tune_random_forest(self, X_train, y_train, numeric_features, categorical_features):
        """Hyperparameter tuning for Random Forest"""
        logger.info("Tuning Random Forest...")

        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__criterion': ['gini', 'entropy']
        }

        pipeline = self.create_pipeline(
            RandomForestClassifier(random_state=42),
            numeric_features, categorical_features
        )

        grid_search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=30, cv=5,
            scoring='roc_auc', n_jobs=-1, random_state=42
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best RF params: {grid_search.best_params_}")
        logger.info(f"Best RF score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def tune_gradient_boosting(self, X_train, y_train, numeric_features, categorical_features):
        """Hyperparameter tuning for Gradient Boosting"""
        logger.info("Tuning Gradient Boosting...")

        param_grid = {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__min_samples_split': [2, 5],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }

        pipeline = self.create_pipeline(
            GradientBoostingClassifier(random_state=42),
            numeric_features, categorical_features
        )

        grid_search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=20, cv=5,
            scoring='roc_auc', n_jobs=-1, random_state=42
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best GB params: {grid_search.best_params_}")
        logger.info(f"Best GB score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def tune_logistic_regression(self, X_train, y_train, numeric_features, categorical_features):
        """Hyperparameter tuning for Logistic Regression"""
        logger.info("Tuning Logistic Regression...")

        param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__max_iter': [500, 1000, 2000]
        }

        pipeline = self.create_pipeline(
            LogisticRegression(random_state=42),
            numeric_features, categorical_features
        )

        grid_search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=20, cv=5,
            scoring='roc_auc', n_jobs=-1, random_state=42
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best LR params: {grid_search.best_params_}")
        logger.info(f"Best LR score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all tuned models"""
        results = []

        for name, (model, params) in self.results.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_proba),
                'Best_Params': str(params)
            })

        return pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)

    def train_advanced(self, df):
        """Run complete advanced training pipeline"""
        logger.info("Starting advanced model training...")

        X_train, X_test, y_train, y_test, num_feat, cat_feat = self.prepare_data(df)

        rf_model, rf_params = self.tune_random_forest(X_train, y_train, num_feat, cat_feat)
        self.results['Random Forest (Tuned)'] = (rf_model, rf_params)

        gb_model, gb_params = self.tune_gradient_boosting(X_train, y_train, num_feat, cat_feat)
        self.results['Gradient Boosting (Tuned)'] = (gb_model, gb_params)

        lr_model, lr_params = self.tune_logistic_regression(X_train, y_train, num_feat, cat_feat)
        self.results['Logistic Regression (Tuned)'] = (lr_model, lr_params)

        results_df = self.evaluate_all_models(X_test, y_test)
        logger.info(f"\n{results_df.to_string()}")

        best_idx = results_df['ROC-AUC'].idxmax()
        self.best_model = self.results[results_df.loc[best_idx, 'Model']][0]

        return results_df

    def save_best_model(self, path="models/best_model_tuned.pkl"):
        """Save the best model"""
        joblib.dump(self.best_model, path)
        logger.info(f"Best model saved to {path}")


def main():
    """Run advanced training"""
    df = pd.read_csv('data/raw/spacex_data.csv')

    trainer = AdvancedModelTrainer()
    results = trainer.train_advanced(df)

    trainer.save_best_model()

    print("\n" + "="*50)
    print("Advanced Training Complete!")
    print("="*50)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()