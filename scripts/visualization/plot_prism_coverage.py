#!/usr/bin/env python3
"""Comprehensive visualization of PRISM atmospheric data for San Diego study beaches.

Creates detailed multi-panel figures showing:
1. Spatial coverage: Beach locations on map with PRISM grid
2. Long-term temporal trends: 9-year time series (2017-2025)
3. Seasonal patterns: Monthly climatology across all beaches
4. Feature distributions: Histograms for all 24 derived features
5. Data completeness: Quality checks and missing data analysis
6. Extreme events: Storm detection and precipitation extremes
7. Cross-beach correlations: Spatial variability analysis

Usage:
    # Generate comprehensive figure set
    python scripts/visualization/plot_prism_coverage.py \
        --atmos-dir data/processed/atmospheric/ \
        --output-dir figures/appendix/

    # Interactive display
    python scripts/visualization/plot_prism_coverage.py --show

    # Generate specific figure type
    python scripts/visualization/plot_prism_coverage.py --figure-type overview
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Beach coordinates (lat, lon) - centroids from MOP transect shapefile
BEACH_COORDS = {
    'blacks': (32.893798, -117.253693),      # MOP 520-567
    'torrey': (32.920009, -117.259033),      # MOP 567-581
    'delmar': (32.949681, -117.265282),      # MOP 595-620
    'solana': (32.988459, -117.274256),      # MOP 637-666
    'sanelijo': (33.026229, -117.287640),    # MOP 683-708
    'encinitas': (33.059110, -117.303186),   # MOP 708-764
}

# Beach display colors
BEACH_COLORS = {
    'blacks': '#e41a1c',
    'torrey': '#377eb8',
    'delmar': '#4daf4a',
    'solana': '#984ea3',
    'sanelijo': '#ff7f00',
    'encinitas': '#a65628',
}

# Core raw variables
RAW_VARIABLES = ['precip_mm', 'temp_min_c', 'temp_max_c', 'temp_mean_c', 'dewpoint_c']

# Derived features for analysis
DERIVED_FEATURES = [
    'precip_7d', 'precip_30d', 'precip_60d', 'precip_90d',
    'api', 'days_since_rain', 'consecutive_dry_days',
    'max_precip_7d', 'max_precip_30d',
    'wet_dry_cycles_30d', 'wet_dry_cycles_90d',
    'vpd', 'vpd_7d_mean',
    'freeze_thaw_cycles_30d', 'freeze_thaw_cycles_season'
]

# Variable display labels
VARIABLE_LABELS = {
    'precip_mm': 'Precipitation (mm/day)',
    'temp_min_c': 'Min Temperature (°C)',
    'temp_max_c': 'Max Temperature (°C)',
    'temp_mean_c': 'Mean Temperature (°C)',
    'dewpoint_c': 'Dewpoint Temperature (°C)',
    'precip_7d': '7-Day Cumulative Precip (mm)',
    'precip_30d': '30-Day Cumulative Precip (mm)',
    'precip_60d': '60-Day Cumulative Precip (mm)',
    'precip_90d': '90-Day Cumulative Precip (mm)',
    'api': 'Antecedent Precipitation Index',
    'days_since_rain': 'Days Since Last Rain',
    'consecutive_dry_days': 'Consecutive Dry Days',
    'max_precip_7d': 'Max 7-Day Precip (mm)',
    'max_precip_30d': 'Max 30-Day Precip (mm)',
    'wet_dry_cycles_30d': 'Wet-Dry Cycles (30d)',
    'wet_dry_cycles_90d': 'Wet-Dry Cycles (90d)',
    'vpd': 'Vapor Pressure Deficit (kPa)',
    'vpd_7d_mean': '7-Day Mean VPD (kPa)',
    'freeze_thaw_cycles_30d': 'Freeze-Thaw Cycles (30d)',
    'freeze_thaw_cycles_season': 'Seasonal Freeze-Thaw Cycles',
}


def load_beach_data(atmos_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load processed atmospheric parquet files for all beaches."""
    data = {}
    for beach in BEACH_COORDS.keys():
        parquet_path = atmos_dir / f'{beach}_atmos.parquet'
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            # Ensure date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            data[beach] = df
            logger.info(f"Loaded {len(df)} records for {beach} ({df['date'].min()} to {df['date'].max()})")
        else:
            logger.warning(f"No data file for {beach}: {parquet_path}")
    return data


def plot_spatial_coverage(
    ax: plt.Axes,
    beach_data: Dict[str, pd.DataFrame],
) -> None:
    """Plot beach locations on a map."""
    ax.set_title('San Diego Study Beach Locations', fontsize=12, fontweight='bold')

    # Plot coastline approximation (simple line)
    coast_lats = [32.85, 32.90, 32.95, 33.00, 33.05, 33.10]
    coast_lons = [-117.25, -117.255, -117.26, -117.27, -117.28, -117.29]
    ax.plot(coast_lons, coast_lats, 'b-', linewidth=2, alpha=0.3, label='Coastline (approx)')

    # Plot each beach location
    for beach, (lat, lon) in BEACH_COORDS.items():
        color = BEACH_COLORS[beach]
        has_data = beach in beach_data

        # Marker style based on data availability
        marker = 'o' if has_data else 'x'
        size = 100 if has_data else 50
        alpha = 1.0 if has_data else 0.5

        ax.scatter(lon, lat, c=color, s=size, marker=marker, alpha=alpha,
                   edgecolors='black', linewidths=1, zorder=5)

        # Label
        ax.annotate(beach.title(), (lon, lat),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    # Add PRISM grid approximation (4km ~ 0.0416667 degrees)
    grid_size = 0.0416667
    lat_min, lat_max = 32.85, 33.12
    lon_min, lon_max = -117.35, -117.20

    # Draw grid lines
    for lat in np.arange(lat_min, lat_max, grid_size):
        ax.axhline(lat, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
    for lon in np.arange(lon_min, lon_max, grid_size):
        ax.axvline(lon, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Add scale bar approximation
    ax.text(-117.29, 32.86, '~4km grid', fontsize=8, alpha=0.6)

    # Legend for data availability
    available = mpatches.Patch(color='green', alpha=0.5, label='Data available')
    missing = mpatches.Patch(color='red', alpha=0.5, label='Data missing')


def plot_temporal_coverage(
    ax: plt.Axes,
    beach_data: Dict[str, pd.DataFrame],
    variable: str = 'precip_mm',
) -> None:
    """Plot long-term time series for a variable across all beaches (monthly aggregated)."""
    ax.set_title(f'{VARIABLE_LABELS.get(variable, variable)} - Long-term Trends (Monthly Mean)',
                 fontsize=12, fontweight='bold')

    for beach, df in beach_data.items():
        if variable in df.columns:
            # Resample to monthly mean for clarity on 9-year timescale
            df_monthly = df.set_index('date').resample('ME')[variable].mean().reset_index()
            color = BEACH_COLORS[beach]
            ax.plot(df_monthly['date'], df_monthly[variable], '-',
                    color=color, label=beach.title(),
                    linewidth=2, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel(VARIABLE_LABELS.get(variable, variable))
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_variable_comparison(
    ax: plt.Axes,
    beach_data: Dict[str, pd.DataFrame],
    beach: str = 'delmar',
) -> None:
    """Plot raw variables for a single beach (monthly aggregated)."""
    if beach not in beach_data:
        ax.text(0.5, 0.5, f'No data for {beach}',
                ha='center', va='center', transform=ax.transAxes)
        return

    df = beach_data[beach]
    # Resample to monthly for clarity
    df_monthly = df.set_index('date').resample('ME').mean().reset_index()

    ax.set_title(f'{beach.title()} - Raw Variables (Monthly Mean)', fontsize=12, fontweight='bold')

    # Precipitation on left y-axis
    ax.bar(df_monthly['date'], df_monthly['precip_mm'],
           color='blue', alpha=0.4, label='Precipitation', width=20)
    ax.set_ylabel('Precipitation (mm/day)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylim(bottom=0)

    # Temperature on right y-axis
    ax2 = ax.twinx()
    ax2.plot(df_monthly['date'], df_monthly['temp_max_c'], 'r-', label='Tmax', linewidth=2)
    ax2.plot(df_monthly['date'], df_monthly['temp_mean_c'], 'g-', label='Tmean', linewidth=2)
    ax2.plot(df_monthly['date'], df_monthly['temp_min_c'], 'b-', label='Tmin', linewidth=2)
    ax2.plot(df_monthly['date'], df_monthly['dewpoint_c'], 'c--', label='Dewpoint', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Temperature (°C)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, ncol=2)

    ax.grid(True, alpha=0.3, axis='x')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_beach_comparison_heatmap(
    ax: plt.Axes,
    beach_data: Dict[str, pd.DataFrame],
    variable: str = 'precip_mm',
) -> None:
    """Plot heatmap of variable across beaches and dates (monthly aggregated)."""
    beaches = list(beach_data.keys())
    if not beaches:
        return

    # Resample all beaches to monthly for visualization
    monthly_data = {}
    for beach, df in beach_data.items():
        if variable in df.columns:
            df_monthly = df.set_index('date').resample('ME')[variable].mean()
            monthly_data[beach] = df_monthly

    if not monthly_data:
        return

    # Combine into matrix
    combined = pd.DataFrame(monthly_data).T

    # Plot heatmap
    cmap = 'YlGnBu' if variable == 'precip_mm' else 'RdYlBu_r'
    im = ax.imshow(combined.values, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_title(f'{VARIABLE_LABELS.get(variable, variable)} - Spatio-Temporal (Monthly)',
                 fontsize=12, fontweight='bold')

    # Labels
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels([b.title() for b in combined.index])

    # Show only yearly labels on x-axis for clarity
    dates = combined.columns
    year_indices = [i for i, d in enumerate(dates) if d.month == 1]
    ax.set_xticks(year_indices)
    ax.set_xticklabels([dates[i].year for i in year_indices], rotation=0)

    ax.set_xlabel('Year')
    ax.set_ylabel('Beach (S → N)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(VARIABLE_LABELS.get(variable, variable))


def plot_coverage_summary(
    ax: plt.Axes,
    beach_data: Dict[str, pd.DataFrame],
) -> None:
    """Plot summary statistics as a table."""
    ax.axis('off')
    ax.set_title('Data Coverage Summary (2017-2025)', fontsize=12, fontweight='bold', pad=20)

    # Build summary table
    rows = []
    for beach in BEACH_COORDS.keys():
        if beach in beach_data:
            df = beach_data[beach]
            lat, lon = BEACH_COORDS[beach]
            n_missing = df[RAW_VARIABLES].isnull().sum().sum()
            completeness = 100 * (1 - n_missing / (len(df) * len(RAW_VARIABLES)))
            rows.append([
                beach.title(),
                f'{lat:.3f}',
                f'{lon:.3f}',
                str(len(df)),
                df['date'].min().strftime('%Y-%m'),
                df['date'].max().strftime('%Y-%m'),
                f"{df['precip_mm'].mean():.1f}",
                f"{df['temp_mean_c'].mean():.1f}",
                f"{completeness:.1f}%",
            ])
        else:
            lat, lon = BEACH_COORDS[beach]
            rows.append([beach.title(), f'{lat:.3f}', f'{lon:.3f}',
                         '0', '-', '-', '-', '-', '0%'])

    columns = ['Beach', 'Lat', 'Lon', 'Days', 'Start', 'End', 'Avg PPT\n(mm)', 'Avg T\n(°C)', 'Complete']

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.11, 0.09, 0.11, 0.08, 0.11, 0.11, 0.11, 0.10, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')


def plot_seasonal_climatology(
    ax: plt.Axes,
    beach_data: Dict[str, pd.DataFrame],
    variable: str = 'precip_mm',
) -> None:
    """Plot monthly climatology (mean annual cycle) for all beaches."""
    ax.set_title(f'{VARIABLE_LABELS.get(variable, variable)} - Seasonal Climatology',
                 fontsize=12, fontweight='bold')

    for beach, df in beach_data.items():
        if variable in df.columns:
            # Group by month and compute mean across all years
            df['month'] = df['date'].dt.month
            monthly_mean = df.groupby('month')[variable].mean()

            color = BEACH_COLORS[beach]
            ax.plot(monthly_mean.index, monthly_mean.values, 'o-',
                    color=color, label=beach.title(),
                    linewidth=2, markersize=6, alpha=0.8)

    # Month labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, rotation=45, ha='right')
    ax.set_xlabel('Month')
    ax.set_ylabel(VARIABLE_LABELS.get(variable, variable))
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_annual_totals(
    ax: plt.Axes,
    beach_data: Dict[str, pd.DataFrame],
) -> None:
    """Plot annual precipitation totals for all beaches."""
    ax.set_title('Annual Precipitation Totals', fontsize=12, fontweight='bold')

    for beach, df in beach_data.items():
        if 'precip_mm' in df.columns:
            # Group by year and sum
            df['year'] = df['date'].dt.year
            annual_sum = df.groupby('year')['precip_mm'].sum()

            color = BEACH_COLORS[beach]
            ax.plot(annual_sum.index, annual_sum.values, 'o-',
                    color=color, label=beach.title(),
                    linewidth=2, markersize=6, alpha=0.8)

    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Precipitation (mm)')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_extreme_events(
    ax: plt.Axes,
    beach_data: Dict[str, pd.DataFrame],
    threshold_mm: float = 25.0,
) -> None:
    """Plot extreme precipitation events (days > threshold)."""
    ax.set_title(f'Extreme Precipitation Events (>{threshold_mm}mm/day)',
                 fontsize=12, fontweight='bold')

    all_extremes = []
    for beach, df in beach_data.items():
        if 'precip_mm' in df.columns:
            extremes = df[df['precip_mm'] > threshold_mm].copy()
            if len(extremes) > 0:
                color = BEACH_COLORS[beach]
                ax.scatter(extremes['date'], extremes['precip_mm'],
                          c=color, label=beach.title(), s=50, alpha=0.7,
                          edgecolors='black', linewidths=0.5)
                all_extremes.append(extremes)

    ax.set_xlabel('Date')
    ax.set_ylabel('Precipitation (mm/day)')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add horizontal line at threshold
    ax.axhline(threshold_mm, color='red', linestyle='--', linewidth=1, alpha=0.5,
               label=f'{threshold_mm}mm threshold')


def plot_feature_distributions(
    fig: plt.Figure,
    beach_data: Dict[str, pd.DataFrame],
    features: List[str],
) -> None:
    """Plot histograms for multiple derived features (6x3 grid)."""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    for idx, feature in enumerate(features):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)

        # Plot histogram for each beach
        for beach, df in beach_data.items():
            if feature in df.columns:
                color = BEACH_COLORS[beach]
                ax.hist(df[feature].dropna(), bins=30, alpha=0.4,
                       color=color, label=beach.title(), density=True)

        ax.set_xlabel(VARIABLE_LABELS.get(feature, feature), fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(feature.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Only show legend on first subplot
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()


def create_overview_figure(
    atmos_dir: Path,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create comprehensive overview visualization (main figure)."""
    # Load data
    beach_data = load_beach_data(atmos_dir)

    if not beach_data:
        logger.error("No beach data found!")
        return None

    # Create figure with subplots (3x3 grid)
    fig = plt.figure(figsize=(18, 14))

    # Layout
    ax1 = fig.add_subplot(3, 3, 1)  # Spatial map
    ax2 = fig.add_subplot(3, 3, 2)  # Precipitation long-term trend
    ax3 = fig.add_subplot(3, 3, 3)  # Temperature long-term trend
    ax4 = fig.add_subplot(3, 3, 4)  # Seasonal climatology (precip)
    ax5 = fig.add_subplot(3, 3, 5)  # Seasonal climatology (temp)
    ax6 = fig.add_subplot(3, 3, 6)  # Annual precipitation totals
    ax7 = fig.add_subplot(3, 3, 7)  # Single beach comparison
    ax8 = fig.add_subplot(3, 3, 8)  # Heatmap
    ax9 = fig.add_subplot(3, 3, 9)  # Summary table

    # Generate plots
    plot_spatial_coverage(ax1, beach_data)
    plot_temporal_coverage(ax2, beach_data, 'precip_mm')
    plot_temporal_coverage(ax3, beach_data, 'temp_mean_c')
    plot_seasonal_climatology(ax4, beach_data, 'precip_mm')
    plot_seasonal_climatology(ax5, beach_data, 'temp_mean_c')
    plot_annual_totals(ax6, beach_data)
    plot_variable_comparison(ax7, beach_data, 'delmar')
    plot_beach_comparison_heatmap(ax8, beach_data, 'precip_mm')
    plot_coverage_summary(ax9, beach_data)

    # Title
    fig.suptitle('PRISM Atmospheric Data - San Diego Study Area (2017-2025)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved overview figure to: {output_path}")

    if show:
        plt.show()

    return fig


def create_feature_distribution_figure(
    atmos_dir: Path,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create feature distribution figure (derived features)."""
    # Load data
    beach_data = load_beach_data(atmos_dir)

    if not beach_data:
        logger.error("No beach data found!")
        return None

    # Create figure for feature distributions
    fig = plt.figure(figsize=(18, 12))

    # Select 15 most important derived features
    selected_features = [
        'precip_7d', 'precip_30d', 'precip_90d',
        'api', 'days_since_rain', 'consecutive_dry_days',
        'max_precip_7d', 'max_precip_30d',
        'wet_dry_cycles_30d', 'wet_dry_cycles_90d',
        'vpd', 'vpd_7d_mean',
        'freeze_thaw_cycles_30d', 'freeze_thaw_cycles_season',
        'rain_day_flag',
    ]

    fig.suptitle('Derived Feature Distributions - All Beaches (2017-2025)',
                 fontsize=16, fontweight='bold', y=0.99)

    plot_feature_distributions(fig, beach_data, selected_features)

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved feature distribution figure to: {output_path}")

    if show:
        plt.show()

    return fig


def create_extreme_events_figure(
    atmos_dir: Path,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create extreme events analysis figure."""
    # Load data
    beach_data = load_beach_data(atmos_dir)

    if not beach_data:
        logger.error("No beach data found!")
        return None

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Layout: 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)  # Extreme events (>25mm)
    ax2 = fig.add_subplot(2, 2, 2)  # Extreme events (>50mm)
    ax3 = fig.add_subplot(2, 2, 3)  # API time series
    ax4 = fig.add_subplot(2, 2, 4)  # VPD time series

    # Generate plots
    plot_extreme_events(ax1, beach_data, threshold_mm=25.0)
    plot_extreme_events(ax2, beach_data, threshold_mm=50.0)
    plot_temporal_coverage(ax3, beach_data, 'api')
    plot_temporal_coverage(ax4, beach_data, 'vpd_7d_mean')

    # Title
    fig.suptitle('Extreme Events and Stress Indicators - San Diego Study Area (2017-2025)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved extreme events figure to: {output_path}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive visualization of PRISM atmospheric data for San Diego beaches'
    )
    parser.add_argument(
        '--atmos-dir',
        type=Path,
        default=Path('data/processed/atmospheric'),
        help='Directory containing processed atmospheric parquet files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('figures/appendix'),
        help='Output directory for figures (creates multiple files)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display figures interactively (does not save)'
    )
    parser.add_argument(
        '--figure-type',
        choices=['all', 'overview', 'features', 'extremes'],
        default='all',
        help='Type of figure(s) to generate'
    )

    args = parser.parse_args()

    # Create output directory if saving
    if not args.show:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")

    # Generate requested figures
    if args.figure_type in ['all', 'overview']:
        logger.info("Generating overview figure...")
        output_path = None if args.show else args.output_dir / 'prism_overview.png'
        create_overview_figure(
            atmos_dir=args.atmos_dir,
            output_path=output_path,
            show=args.show,
        )

    if args.figure_type in ['all', 'features']:
        logger.info("Generating feature distribution figure...")
        output_path = None if args.show else args.output_dir / 'prism_feature_distributions.png'
        create_feature_distribution_figure(
            atmos_dir=args.atmos_dir,
            output_path=output_path,
            show=args.show,
        )

    if args.figure_type in ['all', 'extremes']:
        logger.info("Generating extreme events figure...")
        output_path = None if args.show else args.output_dir / 'prism_extreme_events.png'
        create_extreme_events_figure(
            atmos_dir=args.atmos_dir,
            output_path=output_path,
            show=args.show,
        )

    logger.info("Done!")


if __name__ == '__main__':
    main()
