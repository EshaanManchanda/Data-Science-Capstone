"""
Feature Transformer Module
Custom transformers for data transformation
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """Main feature transformer class"""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select features based on importance or correlation"""

    def __init__(self, features: list = None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features is None:
            return X
        return X[self.features]


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Remove outliers using IQR method"""

    def __init__(self, threshold=1.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.threshold * IQR
            upper = Q3 + self.threshold * IQR
            X = X[(X[col] >= lower) & (X[col] <= upper)]
        return X.values


class CategoryGrouper(BaseEstimator, TransformerMixin):
    """Group rare categories into 'Other'"""

    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.category_map = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            rare_categories = freq[freq < self.threshold].index
            self.category_map[col] = rare_categories
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        for col in X.columns:
            X[col] = X[col].apply(
                lambda x: 'Other' if x in self.category_map.get(col, []) else x
            )
        return X.values


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log transformation to skewed features"""

    def __init__(self, epsilon=1):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        for col in X.columns:
            if X[col].skew() > 1:
                X[col] = np.log1p(X[col] + self.epsilon)
        return X.values


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from date columns"""

    def __init__(self, date_column=None):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        if self.date_column and self.date_column in X.columns:
            X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')
            X[f'{self.date_column}_year'] = X[self.date_column].dt.year
            X[f'{self.date_column}_month'] = X[self.date_column].dt.month
            X[f'{self.date_column}_day'] = X[self.date_column].dt.day
            X[f'{self.date_column}_quarter'] = X[self.date_column].dt.quarter
            X = X.drop(columns=[self.date_column])
        return X.values