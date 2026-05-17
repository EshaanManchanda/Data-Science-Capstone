"""
Model Evaluator Module
Comprehensive model evaluation and metrics
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self):
        self.evaluation_results = {}

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_pred_proba: np.ndarray = None) -> Dict:
        """Evaluate classification model"""
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }

        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba[:, 1]
            results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            results['average_precision'] = average_precision_score(y_true, y_pred_proba)

        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        results['tn'] = int(cm[0, 0])
        results['fp'] = int(cm[0, 1])
        results['fn'] = int(cm[1, 0])
        results['tp'] = int(cm[1, 1])

        self.evaluation_results = results
        return results

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate regression model"""
        results = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }

        self.evaluation_results = results
        return results

    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get detailed classification report"""
        return classification_report(y_true, y_pred)

    def plot_confusion_matrix(self, cm: np.ndarray, labels: list = None,
                              save_path: str = None) -> plt.Figure:
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))

        if labels is None:
            labels = ['Failure', 'Success']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       save_path: str = None) -> plt.Figure:
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    save_path: str = None) -> plt.Figure:
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})', linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_feature_importance(self, feature_names: list, importance: np.ndarray,
                                top_n: int = 10, save_path: str = None) -> plt.Figure:
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_learning_curve(self, train_scores: list, test_scores: list,
                           save_path: str = None) -> plt.Figure:
        """Plot learning curve"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_scores, label='Training Score')
        ax.plot(test_scores, label='Cross-validation Score')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            return "No evaluation results available"

        report = "=" * 50 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"

        for metric, value in self.evaluation_results.items():
            if isinstance(value, (int, float)):
                report += f"{metric.upper()}: {value:.4f}\n"
            else:
                report += f"{metric.upper()}: {value}\n"

        return report

    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models"""
        comparison = pd.DataFrame(results_dict).T
        comparison = comparison.sort_values('roc_auc', ascending=False)
        return comparison

    def calculate_metrics_by_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                       thresholds: list = None) -> pd.DataFrame:
        """Calculate metrics at different thresholds"""
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)

        results = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            results.append({
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            })

        return pd.DataFrame(results)