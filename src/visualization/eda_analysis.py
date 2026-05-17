"""
SpaceX Exploratory Data Analysis Module
Comprehensive EDA functionality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class SpaceXEDA:
    """Comprehensive EDA for SpaceX data"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    def get_data_overview(self) -> Dict:
        """Get overview of the dataset"""
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_summary': self.df[self.numeric_cols].describe().to_dict() if self.numeric_cols else {},
            'categorical_cols': self.categorical_cols
        }

    def analyze_target_distribution(self) -> go.Figure:
        """Analyze target variable distribution"""
        if 'class' not in self.df.columns:
            logger.warning("No 'class' column found")
            return None

        labels = {0: 'Failure', 1: 'Success'}
        counts = self.df['class'].value_counts().rename(labels)

        fig = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            marker_colors=['#FF6B6B', '#4ECDC4'],
            hole=0.4
        )])
        fig.update_layout(title='Landing Outcome Distribution')
        return fig

    def analyze_launch_sites(self) -> go.Figure:
        """Analyze launch site performance"""
        if 'Launch Site' not in self.df.columns or 'class' not in self.df.columns:
            return None

        site_stats = self.df.groupby('Launch Site')['class'].agg(['sum', 'count', 'mean'])
        site_stats.columns = ['Successes', 'Total', 'Success Rate']

        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=['Launches by Site', 'Success Rate by Site'])

        fig.add_trace(go.Bar(
            x=site_stats.index,
            y=site_stats['Total'],
            name='Total',
            marker_color='#3498DB'
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=site_stats.index,
            y=site_stats['Success Rate'],
            name='Success Rate',
            marker_color='#2ECC71'
        ), row=1, col=2)

        fig.update_layout(height=400, showlegend=False)
        return fig

    def analyze_booster_performance(self) -> go.Figure:
        """Analyze booster version performance"""
        if 'Booster Version' not in self.df.columns or 'class' not in self.df.columns:
            return None

        booster_stats = self.df.groupby('Booster Version')['class'].agg(['sum', 'count', 'mean'])
        booster_stats.columns = ['Successes', 'Total', 'Success Rate']
        booster_stats = booster_stats.sort_values('Success Rate', ascending=True)

        fig = px.bar(
            booster_stats,
            x='Success Rate',
            y=booster_stats.index,
            orientation='h',
            title='Success Rate by Booster Version',
            color='Success Rate',
            color_continuous_scale='RdYlGn'
        )
        return fig

    def analyze_payload_mass(self) -> go.Figure:
        """Analyze payload mass distribution"""
        if 'Payload Mass (kg)' not in self.df.columns:
            return None

        fig = go.Figure()
        if 'class' in self.df.columns:
            fig.add_trace(go.Histogram(
                x=self.df[self.df['class'] == 1]['Payload Mass (kg)'],
                name='Success',
                marker_color='#4ECDC4',
                opacity=0.7
            ))
            fig.add_trace(go.Histogram(
                x=self.df[self.df['class'] == 0]['Payload Mass (kg)'],
                name='Failure',
                marker_color='#FF6B6B',
                opacity=0.7
            ))
        else:
            fig.add_trace(go.Histogram(
                x=self.df['Payload Mass (kg)'],
                name='Payload Mass',
                marker_color='#3498DB'
            ))

        fig.update_layout(
            title='Payload Mass Distribution',
            xaxis_title='Payload Mass (kg)',
            yaxis_title='Count',
            barmode='overlay'
        )
        return fig

    def analyze_orbit_types(self) -> go.Figure:
        """Analyze orbit type performance"""
        if 'Orbit' not in self.df.columns or 'class' not in self.df.columns:
            return None

        orbit_stats = self.df.groupby('Orbit')['class'].agg(['sum', 'count', 'mean'])
        orbit_stats.columns = ['Successes', 'Total', 'Success Rate']
        orbit_stats = orbit_stats.sort_values('Success Rate', ascending=False)

        fig = px.bar(
            orbit_stats,
            x=orbit_stats.index,
            y='Success Rate',
            title='Success Rate by Orbit Type',
            color='Success Rate',
            color_continuous_scale='Viridis'
        )
        return fig

    def analyze_year_trends(self) -> go.Figure:
        """Analyze launch trends over years"""
        if 'Year' not in self.df.columns:
            return None

        year_stats = self.df.groupby('Year')['class'].agg(['count', 'mean']) if 'class' in self.df.columns else self.df.groupby('Year').size()
        if 'class' in self.df.columns:
            year_stats.columns = ['Launches', 'Success Rate']
        else:
            year_stats = pd.DataFrame({'Launches': year_stats})

        fig = make_subplots(specs=[[{'secondary_y': True}]])

        fig.add_trace(go.Bar(
            x=year_stats.index,
            y=year_stats['Launches'],
            name='Launches',
            marker_color='#3498DB'
        ), secondary_y=False)

        if 'Success Rate' in year_stats.columns:
            fig.add_trace(go.Scatter(
                x=year_stats.index,
                y=year_stats['Success Rate'],
                name='Success Rate',
                line=dict(color='#2ECC71', width=3)
            ), secondary_y=True)

        fig.update_layout(title='Launch Trends Over Time')
        fig.update_yaxes(title='Total Launches', secondary_y=False)
        if 'Success Rate' in year_stats.columns:
            fig.update_yaxes(title='Success Rate', secondary_y=True)

        return fig

    def plot_correlation_matrix(self) -> go.Figure:
        """Plot correlation matrix"""
        numeric_df = self.df[self.numeric_cols]
        corr = numeric_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}'
        ))
        fig.update_layout(title='Correlation Matrix')
        return fig

    def create_dashboard(self) -> go.Figure:
        """Create comprehensive EDA dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Target Distribution', 'Launch Site Analysis',
                'Booster Performance', 'Payload Distribution',
                'Year Trends', 'Correlation'
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )

        if 'class' in self.df.columns:
            labels = {0: 'Failure', 1: 'Success'}
            counts = self.df['class'].value_counts().rename(labels)
            fig.add_trace(go.Pie(labels=counts.index, values=counts.values,
                                marker_colors=['#FF6B6B', '#4ECDC4']), row=1, col=1)

        if 'Launch Site' in self.df.columns and 'class' in self.df.columns:
            site_stats = self.df.groupby('Launch Site')['class'].mean()
            fig.add_trace(go.Bar(x=site_stats.index, y=site_stats.values,
                                marker_color='#3498DB'), row=1, col=2)

        fig.update_layout(height=900, width=1200, showlegend=False,
                         title_text="SpaceX EDA Dashboard")
        return fig

    def generate_summary_report(self) -> str:
        """Generate text summary report"""
        report = "="*60 + "\n"
        report += "SPACEX DATA ANALYSIS SUMMARY\n"
        report += "="*60 + "\n\n"

        report += f"Dataset Shape: {self.df.shape}\n"
        report += f"Columns: {len(self.df.columns)}\n\n"

        if 'class' in self.df.columns:
            success_rate = self.df['class'].mean() * 100
            report += f"Overall Success Rate: {success_rate:.1f}%\n\n"

        if 'Launch Site' in self.df.columns:
            site_counts = self.df['Launch Site'].value_counts()
            report += "Launch Sites:\n"
            for site, count in site_counts.items():
                report += f"  - {site}: {count} launches\n"
            report += "\n"

        if 'Booster Version' in self.df.columns:
            booster_counts = self.df['Booster Version'].value_counts()
            report += "Booster Versions:\n"
            for booster, count in booster_counts.head(5).items():
                report += f"  - {booster}: {count} launches\n"

        return report


def perform_eda(df: pd.DataFrame, output_dir: str = "reports/figures/") -> Dict:
    """Main function to perform EDA"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    eda = SpaceXEDA(df)

    results = {
        'overview': eda.get_data_overview(),
        'target_dist': eda.analyze_target_distribution(),
        'launch_sites': eda.analyze_launch_sites(),
        'boosters': eda.analyze_booster_performance(),
        'payload': eda.analyze_payload_mass(),
        'orbits': eda.analyze_orbit_types(),
        'trends': eda.analyze_year_trends(),
        'correlation': eda.plot_correlation_matrix(),
        'dashboard': eda.create_dashboard(),
        'summary': eda.generate_summary_report()
    }

    print(results['summary'])
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        results = perform_eda(df)
    else:
        print("Usage: python eda_analysis.py <data_file.csv>")