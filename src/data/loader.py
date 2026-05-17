"""
Data Loader Module
Loads and manages SpaceX launch data
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial processing of SpaceX data"""

    def __init__(self, data_path: str = "data/raw/"):
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            file_path = self.data_path / filename
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise

    def load_from_dict(self, data_dict: dict) -> pd.DataFrame:
        """Load data from dictionary"""
        df = pd.DataFrame(data_dict)
        logger.info(f"Loaded from dict: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def save_csv(self, df: pd.DataFrame, filename: str, output_path: str = "data/processed/") -> None:
        """Save dataframe to CSV"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename
        df.to_csv(file_path, index=False)
        logger.info(f"Saved to {file_path}")

    def load_raw_data(self) -> pd.DataFrame:
        """Load and return raw SpaceX data"""
        if self.raw_data is None:
            try:
                self.raw_data = self.load_csv("spacex_data.csv")
            except:
                logger.warning("Using fallback data")
                self.raw_data = self._create_sample_data()
        return self.raw_data

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample SpaceX data for demonstration"""
        data = {
            'Flight Number': list(range(1, 58)),
            'Launch Site': ['CCAFS LC-40']*20 + ['KSC LC-39A']*20 + ['VAFB SLC-4E']*17,
            'Booster Version': ['F9 v1.0']*7 + ['F9 v1.1']*15 + ['F9 FT']*20 + ['B5']*16,
            'Payload Mass (kg)': np.random.randint(0, 5500, 57),
            'Orbit': ['LEO']*20 + ['GTO']*20 + ['SSO']*12 + ['MEO']*5,
            'Grid Fins': np.random.choice([True, False], 57, p=[0.85, 0.15]),
            'Legs': np.random.choice([True, False], 57, p=[0.88, 0.12]),
            'Block': np.random.choice(['B1', 'B2', 'B3', 'B4', 'B5'], 57, p=[0.1, 0.15, 0.25, 0.3, 0.2]),
            'Year': list(range(2010, 2023)) * 5 + list(range(57)),
            'class': np.random.choice([0, 1], 57, p=[0.25, 0.75])
        }
        data['Year'] = [2010 + i % 13 for i in range(57)]
        return pd.DataFrame(data)

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the dataset"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum()
        }