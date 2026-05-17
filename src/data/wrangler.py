"""
SpaceX Data Wrangling Module
Cleans and transforms SpaceX data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class SpaceXDataWrangler:
    """Data wrangling for SpaceX launches"""

    def __init__(self):
        self.df = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV"""
        self.df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.df)} rows from {filepath}")
        return self.df

    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values in dataset"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Handling missing values...")

        for col in self.df.columns:
            missing_pct = self.df[col].isnull().sum() / len(self.df) * 100
            if missing_pct > 0:
                logger.info(f"  {col}: {missing_pct:.1f}% missing")

        self.df = self.df.fillna({
            'Payload Mass (kg)': 0,
            'Orbit': 'Unknown',
            'Launch Site': 'Unknown'
        })

        return self.df

    def create_target_column(self, outcome_column: str = 'Outcome') -> pd.DataFrame:
        """Create binary target column from landing outcomes"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if outcome_column not in self.df.columns:
            logger.warning(f"Column {outcome_column} not found. Using 'class' column directly.")
            return self.df

        bad_outcomes = [
            'Failure', 'Failure (drone ship)', 'Failure (ground pad)',
            'False Ocean', 'False RTLS', 'False ASDS',
            'None ASDS', 'None None', 'Prelaunch failure'
        ]

        self.df['class'] = self.df[outcome_column].apply(
            lambda x: 0 if x in bad_outcomes else 1
        )

        logger.info(f"Created target column. Success rate: {self.df['class'].mean():.2%}")
        return self.df

    def extract_booster_version(self) -> pd.DataFrame:
        """Extract booster version from launch data"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if 'Booster Version' in self.df.columns:
            return self.df

        if 'Version Booster' in self.df.columns:
            self.df['Booster Version'] = self.df['Version Booster']

        return self.df

    def clean_booster_version(self, column: str = 'Booster Version') -> pd.DataFrame:
        """Clean booster version column"""
        if self.df is None or column not in self.df.columns:
            return self.df

        def extract_version(version):
            if pd.isna(version):
                return 'Unknown'
            version = str(version)
            if 'B5' in version or 'Block 5' in version:
                return 'B5'
            elif 'FT' in version or 'Falcon 9' in version:
                return 'F9 FT'
            elif 'v1.1' in version:
                return 'F9 v1.1'
            elif 'v1.0' in version:
                return 'F9 v1.0'
            return 'Unknown'

        self.df['Booster_Category'] = self.df[column].apply(extract_version)
        return self.df

    def add_year_column(self, date_column: str = 'Date') -> pd.DataFrame:
        """Extract year from date column"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if date_column in self.df.columns:
            try:
                self.df['Year'] = pd.to_datetime(self.df[date_column]).dt.year
            except:
                logger.warning(f"Could not parse date column {date_column}")
                if 'Year' not in self.df.columns:
                    self.df['Year'] = 2020
        else:
            if 'Year' not in self.df.columns:
                self.df['Year'] = 2020

        return self.df

    def encode_booleans(self) -> pd.DataFrame:
        """Convert boolean columns to integers"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        bool_cols = self.df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            self.df[col] = self.df[col].astype(int)

        return self.df

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics"""
        if self.df is None:
            return {}

        return {
            'total_launches': len(self.df),
            'success_rate': self.df['class'].mean() if 'class' in self.df.columns else 0,
            'unique_launch_sites': self.df['Launch Site'].nunique() if 'Launch Site' in self.df.columns else 0,
            'date_range': {
                'start': self.df['Date'].min() if 'Date' in self.df.columns else 'N/A',
                'end': self.df['Date'].max() if 'Date' in self.df.columns else 'N/A'
            }
        }

    def process_full_pipeline(self, filepath: str) -> pd.DataFrame:
        """Run full wrangling pipeline"""
        logger.info("Running full data wrangling pipeline...")

        self.load_data(filepath)
        self.handle_missing_values()
        self.extract_booster_version()
        self.clean_booster_version()
        self.add_year_column()
        self.encode_booleans()

        if 'Outcome' in self.df.columns:
            self.create_target_column('Outcome')
        elif 'class' not in self.df.columns:
            self.df['class'] = 1

        logger.info(f"Pipeline complete. Final shape: {self.df.shape}")
        return self.df


def wrangle_spacex_data(input_path: str, output_path: str = "data/processed/spacex_cleaned.csv") -> pd.DataFrame:
    """Main function to wrangle SpaceX data"""
    wrangler = SpaceXDataWrangler()
    df = wrangler.process_full_pipeline(input_path)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")
    return df


if __name__ == "__main__":
    print("Running data wrangling...")
    df = wrangle_spacex_data("data/raw/spacex_data.csv")
    print(f"Cleaned data: {df.shape}")