"""
Data Validator Module
Validates data quality and performs data checks
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and checks for issues"""

    def __init__(self):
        self.validation_results = {}
        self.issues = []

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Run all validation checks on dataframe"""
        logger.info("Starting data validation...")

        self.validation_results = {
            'shape': self.check_shape(df),
            'missing_values': self.check_missing_values(df),
            'duplicates': self.check_duplicates(df),
            'data_types': self.check_data_types(df),
            'outliers': self.check_outliers(df),
            'value_ranges': self.check_value_ranges(df),
            'correlations': self.check_correlations(df)
        }

        logger.info(f"Validation complete. Found {len(self.issues)} issues")
        return self.validation_results

    def check_shape(self, df: pd.DataFrame) -> Dict:
        """Check dataframe shape"""
        return {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'is_empty': df.empty
        }

    def check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values"""
        missing = df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()

        if missing_dict:
            self.issues.append(f"Found missing values: {missing_dict}")

        return {
            'total_missing': missing.sum(),
            'columns_with_missing': missing_dict,
            'missing_percentage': (missing.sum() / (df.shape[0] * df.shape[1])) * 100
        }

    def check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows"""
        duplicates = df.duplicated().sum()

        if duplicates > 0:
            self.issues.append(f"Found {duplicates} duplicate rows")

        return {
            'duplicate_count': duplicates,
            'duplicate_percentage': (duplicates / len(df)) * 100
        }

    def check_data_types(self, df: pd.DataFrame) -> Dict:
        """Check data types"""
        return df.dtypes.apply(str).to_dict()

    def check_outliers(self, df: pd.DataFrame, numeric_cols: List[str] = None) -> Dict:
        """Check for outliers using IQR method"""
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = {}
        for col in numeric_cols:
            if df[col].dropna().shape[0] > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

                if outlier_count > 0:
                    outliers[col] = {
                        'count': outlier_count,
                        'percentage': (outlier_count / len(df)) * 100,
                        'bounds': [lower_bound, upper_bound]
                    }

        return outliers

    def check_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Check value ranges for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ranges = {}

        for col in numeric_cols:
            ranges[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median()
            }

        return ranges

    def check_correlations(self, df: pd.DataFrame) -> Dict:
        """Check highly correlated features"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        return {'high_correlations': high_corr}

    def get_validation_report(self) -> str:
        """Generate validation report"""
        report = "=== Data Validation Report ===\n\n"

        for check, results in self.validation_results.items():
            report += f"{check.upper()}:\n"
            for key, value in results.items():
                report += f"  {key}: {value}\n"
            report += "\n"

        if self.issues:
            report += "ISSUES FOUND:\n"
            for issue in self.issues:
                report += f"  - {issue}\n"

        return report