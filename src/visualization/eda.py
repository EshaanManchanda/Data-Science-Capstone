"""
EDA Plots Module
Specialized plots for Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class EDAPlots:
    """Specialized EDA visualization plots"""

    def __init__(self):
        self.figures = []

    def plot_target_distribution(self, df: pd.DataFrame, target: str = 'class',
                                save_path: str = None) -> go.Figure:
        """Plot target variable distribution"""
        labels = {0: 'Failure', 1: 'Success'}
        counts = df[target].value_counts().rename(labels)

        fig = go.Figure(data=[go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=counts.values,
            textposition='auto'
        )])
        fig.update_layout(
            title='Target Variable Distribution (Landing Success)',
            xaxis_title='Outcome',
            yaxis_title='Count',
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_launch_site_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Analyze launches by site"""
        site_stats = df.groupby('Launch Site')['class'].agg(['sum', 'count', 'mean'])
        site_stats.columns = ['Successes', 'Total', 'Success Rate']
        site_stats['Failures'] = site_stats['Total'] - site_stats['Successes']

        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Launches by Site', 'Success Rate by Site'))

        fig.add_trace(go.Bar(
            x=site_stats.index,
            y=site_stats['Successes'],
            name='Successes',
            marker_color='#4ECDC4'
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=site_stats.index,
            y=site_stats['Failures'],
            name='Failures',
            marker_color='#FF6B6B'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=site_stats.index,
            y=site_stats['Success Rate'],
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='#2C3E50', width=3)
        ), row=1, col=2)

        fig.update_layout(height=400, showlegend=True,
                         title_text="Launch Site Analysis")

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_booster_version_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Analyze booster versions"""
        booster_stats = df.groupby('Booster Version')['class'].agg(['sum', 'count', 'mean'])
        booster_stats.columns = ['Successes', 'Total', 'Success Rate']

        fig = go.Figure(data=[go.Bar(
            x=booster_stats.index,
            y=booster_stats['Success Rate'],
            marker_color=px.colors.qualitative.Set2,
            text=[f'{x:.1%}' for x in booster_stats['Success Rate']],
            textposition='auto'
        )])
        fig.update_layout(
            title='Success Rate by Booster Version',
            xaxis_title='Booster Version',
            yaxis_title='Success Rate',
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_payload_mass_distribution(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Plot payload mass distribution by outcome"""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df['class'] == 1]['Payload Mass (kg)'],
            name='Success',
            marker_color='#4ECDC4',
            opacity=0.75
        ))
        fig.add_trace(go.Histogram(
            x=df[df['class'] == 0]['Payload Mass (kg)'],
            name='Failure',
            marker_color='#FF6B6B',
            opacity=0.75
        ))
        fig.update_layout(
            title='Payload Mass Distribution by Outcome',
            xaxis_title='Payload Mass (kg)',
            yaxis_title='Count',
            barmode='overlay',
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_orbit_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Analyze orbits"""
        if 'Orbit' not in df.columns:
            return None

        orbit_stats = df.groupby('Orbit')['class'].agg(['sum', 'count', 'mean'])
        orbit_stats.columns = ['Successes', 'Total', 'Success Rate']
        orbit_stats = orbit_stats.sort_values('Success Rate', ascending=False)

        fig = go.Figure(data=[go.Bar(
            x=orbit_stats.index,
            y=orbit_stats['Success Rate'],
            marker_color=px.colors.qualitative.Bold,
            text=[f'{x:.1%}' for x in orbit_stats['Success Rate']],
            textposition='auto'
        )])
        fig.update_layout(
            title='Success Rate by Orbit Type',
            xaxis_title='Orbit Type',
            yaxis_title='Success Rate',
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_year_trend(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Plot launch trends over years"""
        year_stats = df.groupby('Year')['class'].agg(['sum', 'count', 'mean'])
        year_stats.columns = ['Successes', 'Total', 'Success Rate']

        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Launches per Year', 'Success Rate Trend'))

        fig.add_trace(go.Scatter(
            x=year_stats.index,
            y=year_stats['Total'],
            mode='lines+markers',
            name='Total Launches',
            fill='tozeroy',
            line=dict(color='#3498DB', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=year_stats.index,
            y=year_stats['Success Rate'],
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='#2ECC71', width=3)
        ), row=1, col=2)

        fig.update_layout(height=400, showlegend=True,
                         title_text="Launch Trends Over Time")

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Plot interactive correlation matrix"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            showscale=True
        ))
        fig.update_layout(
            title='Correlation Matrix',
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_feature_importance(self, features: list, importance: list,
                                save_path: str = None) -> go.Figure:
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=True)

        fig = go.Figure(data=[go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color='#9B59B6'
        )])
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_model_comparison(self, model_results: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Plot model comparison"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=model_results['Model'],
                y=model_results[metric]
            ))

        fig.update_layout(
            barmode='group',
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_dashboard(self, df: pd.DataFrame, target: str = 'class') -> go.Figure:
        """Create comprehensive EDA dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Target Distribution',
                'Launch Site Analysis',
                'Booster Version Success',
                'Payload Mass Distribution',
                'Year Trend',
                'Correlation Matrix'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )

        labels = {0: 'Failure', 1: 'Success'}
        counts = df[target].value_counts().rename(labels)
        fig.add_trace(go.Bar(x=counts.index, y=counts.values,
                            marker_color=['#FF6B6B', '#4ECDC4']), row=1, col=1)

        site_stats = df.groupby('Launch Site')['class'].mean()
        fig.add_trace(go.Bar(x=site_stats.index, y=site_stats.values,
                            marker_color='#3498DB'), row=1, col=2)

        booster_stats = df.groupby('Booster Version')['class'].mean().head(10)
        fig.add_trace(go.Bar(x=booster_stats.index, y=booster_stats.values,
                            marker_color='#2ECC71'), row=2, col=1)

        fig.add_trace(go.Histogram(x=df['Payload Mass (kg)'], marker_color='#9B59B6'),
                     row=2, col=2)

        year_stats = df.groupby('Year')['class'].mean()
        fig.add_trace(go.Scatter(x=year_stats.index, y=year_stats.values,
                                mode='lines+markers', line=dict(color='#E74C3C')),
                     row=3, col=1)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                colorscale='RdBu', showscale=False),
                     row=3, col=2)

        fig.update_layout(height=900, width=1200, showlegend=False,
                        title_text="SpaceX Landing Prediction - EDA Dashboard")

        return fig