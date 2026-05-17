"""
Preprocessing Pipeline Module
Handles data preprocessing, cleaning, and transformation
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Complete preprocessing pipeline for SpaceX data"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.scaler = None
        self.encoder = None
        self.feature_columns = []
        self.target_column = 'class'
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}

    def _default_config(self) -> Dict:
        return {
            'test_size': 0.2,
            'random_state': 42,
            'numeric_strategy': 'median',
            'categorical_strategy': 'mode',
            'scaling_method': 'standard'
        }

    def load_and_preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Main method to load and preprocess data"""
        logger.info("Starting data preprocessing...")

        df_cleaned = self.handle_missing_values(df)
        df_cleaned = self.remove_duplicates(df_cleaned)
        df_cleaned = self.handle_outliers(df_cleaned)
        df_cleaned = self.engineer_features(df_cleaned)

        X = df_cleaned.drop(columns=[self.target_column])
        y = df_cleaned[self.target_column]

        logger.info(f"Preprocessing complete. Shape: X={X.shape}, y={y.shape}")
        return X, y

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")

        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if self.config['numeric_strategy'] == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif self.config['numeric_strategy'] == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)

        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate rows")
            df = df.drop_duplicates()
        return df

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        logger.info("Handling outliers...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col != self.target_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones"""
        logger.info("Engineering features...")

        df = df.copy()

        if 'Booster Version' in df.columns:
            df['Booster_Category'] = df['Booster Version'].apply(self._categorize_booster)

        if 'Launch Site' in df.columns:
            df['Launch_Site_Category'] = df['Launch Site'].apply(self._categorize_site)

        return df

    def _categorize_booster(self, version: str) -> str:
        """Categorize booster versions"""
        if 'B5' in version:
            return 'Block5'
        elif 'FT' in version:
            return 'Falcon9FT'
        elif 'v1.1' in version:
            return 'v1.1'
        else:
            return 'v1.0'

    def _categorize_site(self, site: str) -> str:
        """Categorize launch sites"""
        if 'KSC' in site:
            return 'KSC'
        elif 'VAFB' in site:
            return 'VAFB'
        else:
            return 'CCAFS'

    def encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")

        X = X.copy()
        categorical_cols = X.select_dtypes(include=['object']).columns

        if fit:
            self.label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        else:
            for col in categorical_cols:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))

        return X

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features"""
        logger.info("Scaling features...")

        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=X.columns)

    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = None,
                   random_state: int = None) -> Tuple:
        """Split data into train and test sets"""
        test_size = test_size or self.config['test_size']
        random_state = random_state or self.config['random_state']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        logger.info(f"Data split: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test

    def get_processed_data(self) -> Tuple:
        """Return processed train/test data"""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def save_preprocessors(self, path: str = "models/") -> None:
        """Save preprocessors for later use"""
        joblib.dump(self.scaler, f"{path}scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}label_encoders.pkl")
        logger.info(f"Saved preprocessors to {path}")

    def load_preprocessors(self, path: str = "models/") -> None:
        """Load preprocessors from disk"""
        self.scaler = joblib.load(f"{path}scaler.pkl")
        self.label_encoders = joblib.load(f"{path}label_encoders.pkl")
        logger.info(f"Loaded preprocessors from {path}")