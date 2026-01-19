#!/usr/bin/env python3
"""Visualize PRISM data coverage for San Diego study beaches.

Creates maps and plots showing:
1. Spatial coverage: Beach locations on a map with PRISM grid
2. Temporal coverage: Time series of downloaded data
3. Variable comparison: Side-by-side comparison across beaches

Usage:
    python scripts/visualization/plot_prism_coverage.py \
        --prism-dir data/raw/prism/ \
        --output figures/prism_coverage.png

    # Interactive display
    python scripts/visualization/plot_prism_coverage.py --show
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

VARIABLES = ['ppt', 'tmin', 'tmax', 'tmean', 'tdmean']
VARIABLE_LABELS = {
    'ppt': 'Precipitation (mm)',
    'tmin': 'Min Temp (°C)',
    'tmax': 'Max Temp (°C)',
    'tmean': 'Mean Temp (°C)',
    'tdmean': 'Dewpoint (°C)',
}


def load_beach_data(prism_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load extracted CSV data for all beaches."""
    data = {}
    for beach in BEACH_COORDS.keys():
        csv_path = prism_dir / f'{beach}_raw.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['date'])
            data[beach] = df
            logger.info(f"Loaded {len(df)} records for {beach}")
        else:
            logger.warning(f"No data file for {beach}: {csv_path}")
    return data


def plot_spatial_coverage(
    ax: plt.Axes,
    prism_dir: Path,
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
    variable: str = 'ppt',
) -> None:
    """Plot time series for a variable across all beaches."""
    ax.set_title(f'{VARIABLE_LABELS.get(variable, variable)} - All Beaches',
                 fontsize=12, fontweight='bold')

    for beach, df in beach_data.items():
        if variable in df.columns:
            color = BEACH_COLORS[beach]
            ax.plot(df['date'], df[variable], 'o-',
                    color=color, label=beach.title(),
                    markersize=4, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel(VARIABLE_LABELS.get(variable, variable))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_variable_comparison(
    ax: plt.Axes,
    beach_data: Dict[str, pd.DataFrame],
    beach: str = 'delmar',
) -> None:
    """Plot all variables for a single beach."""
    if beach not in beach_data:
        ax.text(0.5, 0.5, f'No data for {beach}',
                ha='center', va='center', transform=ax.transAxes)
        return

    df = beach_data[beach]
    ax.set_title(f'{beach.title()} - All Variables', fontsize=12, fontweight='bold')

    # Precipitation on left y-axis
    ax.bar(df['date'], df['ppt'], color='blue', alpha=0.4, label='Precipitation')
    ax.set_ylabel('Precipitation (mm)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    # Temperature on right y-axis
    ax2 = ax.twinx()
    ax2.plot(df['date'], df['tmax'], 'r-', label='Tmax', linewidth=2)
    ax2.plot(df['date'], df['tmean'], 'g-', label='Tmean', linewidth=2)
    ax2.plot(df['date'], df['tmin'], 'b-', label='Tmin', linewidth=2)
    ax2.plot(df['date'], df['tdmean'], 'c--', label='Dewpoint', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Temperature (°C)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)

    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_beach_comparison_heatmap(
    ax: plt.Axes,
    beach_data: Dict[str, pd.DataFrame],
    variable: str = 'ppt',
) -> None:
    """Plot heatmap of variable across beaches and dates."""
    beaches = list(beach_data.keys())
    if not beaches:
        return

    # Get common dates
    dates = beach_data[beaches[0]]['date'].values

    # Build matrix
    matrix = np.zeros((len(beaches), len(dates)))
    for i, beach in enumerate(beaches):
        if variable in beach_data[beach].columns:
            matrix[i, :] = beach_data[beach][variable].values

    # Plot heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='YlGnBu')
    ax.set_title(f'{VARIABLE_LABELS.get(variable, variable)} - Spatial-Temporal',
                 fontsize=12, fontweight='bold')

    # Labels
    ax.set_yticks(range(len(beaches)))
    ax.set_yticklabels([b.title() for b in beaches])

    ax.set_xticks(range(len(dates)))
    date_labels = [pd.Timestamp(d).strftime('%m/%d') for d in dates]
    ax.set_xticklabels(date_labels, rotation=45, ha='right')

    ax.set_xlabel('Date')
    ax.set_ylabel('Beach (S → N)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(VARIABLE_LABELS.get(variable, variable))


def plot_coverage_summary(
    ax: plt.Axes,
    prism_dir: Path,
    beach_data: Dict[str, pd.DataFrame],
) -> None:
    """Plot summary statistics as a table."""
    ax.axis('off')
    ax.set_title('Data Coverage Summary', fontsize=12, fontweight='bold', pad=20)

    # Build summary table
    rows = []
    for beach in BEACH_COORDS.keys():
        if beach in beach_data:
            df = beach_data[beach]
            lat, lon = BEACH_COORDS[beach]
            rows.append([
                beach.title(),
                f'{lat:.3f}',
                f'{lon:.3f}',
                str(len(df)),
                df['date'].min().strftime('%Y-%m-%d'),
                df['date'].max().strftime('%Y-%m-%d'),
                f"{df['ppt'].mean():.1f}",
                f"{df['tmean'].mean():.1f}",
            ])
        else:
            lat, lon = BEACH_COORDS[beach]
            rows.append([beach.title(), f'{lat:.3f}', f'{lon:.3f}',
                         '0', '-', '-', '-', '-'])

    columns = ['Beach', 'Lat', 'Lon', 'Days', 'Start', 'End', 'Avg PPT', 'Avg T']

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.12, 0.10, 0.12, 0.08, 0.14, 0.14, 0.12, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')


def create_coverage_figure(
    prism_dir: Path,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create comprehensive coverage visualization."""
    # Load data
    beach_data = load_beach_data(prism_dir)

    if not beach_data:
        logger.error("No beach data found!")
        return None

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Layout: 2x3 grid
    ax1 = fig.add_subplot(2, 3, 1)  # Spatial map
    ax2 = fig.add_subplot(2, 3, 2)  # Precipitation time series
    ax3 = fig.add_subplot(2, 3, 3)  # Temperature time series
    ax4 = fig.add_subplot(2, 3, 4)  # Single beach all variables
    ax5 = fig.add_subplot(2, 3, 5)  # Heatmap
    ax6 = fig.add_subplot(2, 3, 6)  # Summary table

    # Generate plots
    plot_spatial_coverage(ax1, prism_dir, beach_data)
    plot_temporal_coverage(ax2, beach_data, 'ppt')
    plot_temporal_coverage(ax3, beach_data, 'tmean')
    plot_variable_comparison(ax4, beach_data, 'delmar')
    plot_beach_comparison_heatmap(ax5, beach_data, 'ppt')
    plot_coverage_summary(ax6, prism_dir, beach_data)

    # Title
    fig.suptitle('PRISM Climate Data Coverage - San Diego Study Area',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved figure to: {output_path}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize PRISM data coverage for San Diego beaches'
    )
    parser.add_argument(
        '--prism-dir',
        type=Path,
        default=Path('data/raw/prism'),
        help='Directory containing PRISM data and extracted CSVs'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('figures/prism_coverage.png'),
        help='Output figure path'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display figure interactively'
    )

    args = parser.parse_args()

    create_coverage_figure(
        prism_dir=args.prism_dir,
        output_path=args.output if not args.show else None,
        show=args.show,
    )


if __name__ == '__main__':
    main()
