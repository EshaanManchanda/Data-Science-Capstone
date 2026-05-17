"""
Visualization Plots Module
Creates professional visualizations for EDA and model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class Plots:
    """Creates various types of plots for data visualization"""

    def __init__(self, theme: str = "seaborn", palette: str = "viridis"):
        self.theme = theme
        self.palette = palette
        sns.set_theme(style=theme)

    def bar_plot(self, df: pd.DataFrame, x: str, y: str, title: str = "",
                 save_path: str = None) -> plt.Figure:
        """Create bar plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind='bar', x=x, y=y, ax=ax, color=self.palette)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def histogram(self, df: pd.DataFrame, column: str, bins: int = 30,
                  title: str = "", save_path: str = None) -> plt.Figure:
        """Create histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[column], bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(title or f"Distribution of {column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def scatter_plot(self, df: pd.DataFrame, x: str, y: str, hue: str = None,
                     title: str = "", save_path: str = None) -> plt.Figure:
        """Create scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=self.palette)
        ax.set_title(title or f"{x} vs {y}", fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def box_plot(self, df: pd.DataFrame, x: str, y: str, hue: str = None,
                 title: str = "", save_path: str = None) -> plt.Figure:
        """Create box plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=self.palette)
        ax.set_title(title or f"Box Plot: {x} vs {y}", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def heatmap(self, df: pd.DataFrame, columns: List[str] = None, annot: bool = True,
                title: str = "", save_path: str = None) -> plt.Figure:
        """Create correlation heatmap"""
        if columns:
            corr = df[columns].corr()
        else:
            corr = df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=annot, cmap='coolwarm', center=0,
                   fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title(title or "Correlation Heatmap", fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def pair_plot(self, df: pd.DataFrame, columns: List[str] = None,
                  hue: str = None, save_path: str = None) -> plt.Figure:
        """Create pair plot"""
        if columns:
            df_subset = df[columns + ([hue] if hue else [])]
        else:
            df_subset = df

        g = sns.pairplot(df_subset, hue=hue, palette=self.palette)
        g.fig.suptitle("Pair Plot", y=1.02, fontsize=14, fontweight='bold')

        if save_path:
            g.savefig(save_path, dpi=300, bbox_inches='tight')

        return g

    def pie_chart(self, df: pd.DataFrame, column: str, title: str = "",
                  save_path: str = None) -> plt.Figure:
        """Create pie chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        counts = df[column].value_counts()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
               colors=sns.color_palette(self.palette, len(counts)))
        ax.set_title(title or f"Distribution of {column}", fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def line_plot(self, df: pd.DataFrame, x: str, y: str, hue: str = None,
                  title: str = "", save_path: str = None) -> plt.Figure:
        """Create line plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=self.palette)
        ax.set_title(title or f"{y} over {x}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def violin_plot(self, df: pd.DataFrame, x: str, y: str, hue: str = None,
                    title: str = "", save_path: str = None) -> plt.Figure:
        """Create violin plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=self.palette)
        ax.set_title(title or f"Violin Plot: {x} vs {y}", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def count_plot(self, df: pd.DataFrame, column: str, hue: str = None,
                   title: str = "", save_path: str = None) -> plt.Figure:
        """Create count plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x=column, hue=hue, ax=ax, palette=self.palette)
        ax.set_title(title or f"Count of {column}", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def save_all_plots(self, df: pd.DataFrame, output_dir: str = "reports/figures/") -> None:
        """Generate and save all common plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in numeric_cols[:5]:
            self.histogram(df, col, save_path=f"{output_dir}hist_{col}.png")

        for col in categorical_cols[:3]:
            self.count_plot(df, col, save_path=f"{output_dir}count_{col}.png")

        if len(numeric_cols) > 1:
            self.heatmap(df, save_path=f"{output_dir}correlation.png")

        logger.info(f"Saved all plots to {output_dir}")