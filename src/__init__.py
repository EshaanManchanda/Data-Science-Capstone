"""
SpaceX Data Science Project
Complete integration of all Coursera notebooks
"""

__version__ = "2.0.0"
__author__ = "Eshaan Manchanda"

# Data Collection
from src.data.api_client import SpaceXAPIClient, collect_spacex_data

# Data Wrangling
from src.data.wrangler import SpaceXDataWrangler, wrangle_spacex_data

# Data Loading
from src.data.loader import DataLoader

# Data Validation
from src.data.validator import DataValidator

# EDA
from src.visualization.eda_analysis import SpaceXEDA, perform_eda

# SQL Analysis
from src.analysis.sql_analyzer import SpaceXSQLAnalyzer, analyze_with_sql

# Mapping
from src.visualization.mapping import SpaceXMapper, create_launch_map

# ML Models
from src.models.unified_pipeline import SpaceXModelTrainer, train_spacex_model

# Core Models
from src.models.trainer import ModelTrainer
from src.models.predictor import Predictor
from src.models.evaluator import ModelEvaluator

# Preprocessing
from src.preprocessing.pipeline import PreprocessingPipeline

# Visualization
from src.visualization.plots import Plots

__all__ = [
    "SpaceXAPIClient",
    "collect_spacex_data",
    "SpaceXDataWrangler",
    "wrangle_spacex_data",
    "DataLoader",
    "DataValidator",
    "SpaceXEDA",
    "perform_eda",
    "SpaceXSQLAnalyzer",
    "analyze_with_sql",
    "SpaceXMapper",
    "create_launch_map",
    "SpaceXModelTrainer",
    "train_spacex_model",
    "ModelTrainer",
    "Predictor",
    "ModelEvaluator",
    "PreprocessingPipeline",
    "Plots",
    "EDAPlots"
]