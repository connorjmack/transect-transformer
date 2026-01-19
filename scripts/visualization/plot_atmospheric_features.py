#!/usr/bin/env python3
"""Visualize atmospheric features for CliffCast.

Creates publication-quality figures showing:
1. Precipitation climatology and extreme events
2. Temperature and VPD patterns
3. Derived features (API, wetting/drying cycles)
4. Spatial variation across beaches
5. Correlation with cliff erosion potential

Usage:
    python scripts/visualization/plot_atmospheric_features.py \
        --atmos-dir data/processed/atmospheric/ \
        --output-dir figures/appendix/
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Beach colors (consistent with other figures)
BEACH_COLORS = {
    'blacks': '#e41a1c',
    'torrey': '#377eb8',
    'delmar': '#4daf4a',
    'solana': '#984ea3',
    'sanelijo': '#ff7f00',
    'encinitas': '#a65628',
}

# Beach order (south to north)
BEACH_ORDER = ['blacks', 'torrey', 'delmar', 'solana', 'sanelijo', 'encinitas']


def load_all_beaches(atmos_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load atmospheric data for all beaches."""
    data = {}
    for beach in BEACH_ORDER:
        path = atmos_dir / f'{beach}_atmos.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df['date'] = pd.to_datetime(df['date'])
            data[beach] = df
            logger.info(f"Loaded {beach}: {len(df)} records")
    return data


def plot_precipitation_climatology(
    data: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Create precipitation climatology figure with monthly averages and extremes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Use Del Mar as representative beach
    df = data['delmar'].copy()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['doy'] = df['date'].dt.dayofyear

    # --- Panel A: Monthly precipitation climatology ---
    ax = axes[0, 0]
    monthly = df.groupby('month')['precip_mm'].agg(['mean', 'std', 'max'])
    months = range(1, 13)
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    bars = ax.bar(months, monthly['mean'], color='steelblue', alpha=0.7, label='Mean')
    ax.errorbar(months, monthly['mean'], yerr=monthly['std'],
                fmt='none', color='black', capsize=3, label='±1 SD')
    ax.scatter(months, monthly['max'], color='red', s=50, zorder=5,
               marker='v', label='Max daily')

    ax.set_xlabel('Month')
    ax.set_ylabel('Daily Precipitation (mm)')
    ax.set_title('A) Monthly Precipitation Climatology', fontweight='bold', loc='left')
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0.5, 12.5)

    # --- Panel B: Annual totals ---
    ax = axes[0, 1]
    annual = df.groupby('year')['precip_mm'].sum()
    colors = ['steelblue' if v < annual.mean() else 'darkblue' for v in annual.values]

    bars = ax.bar(annual.index, annual.values, color=colors, alpha=0.8)
    ax.axhline(annual.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {annual.mean():.0f} mm')

    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Precipitation (mm)')
    ax.set_title('B) Annual Precipitation Totals', fontweight='bold', loc='left')
    ax.legend(loc='upper right')

    # --- Panel C: Extreme events (>25mm days) ---
    ax = axes[1, 0]
    extreme_threshold = 25  # mm
    df['extreme'] = df['precip_mm'] >= extreme_threshold
    extreme_by_year = df.groupby('year')['extreme'].sum()

    ax.bar(extreme_by_year.index, extreme_by_year.values, color='darkred', alpha=0.8)
    ax.axhline(extreme_by_year.mean(), color='black', linestyle='--',
               label=f'Mean: {extreme_by_year.mean():.1f} days/yr')

    ax.set_xlabel('Year')
    ax.set_ylabel(f'Days with Precip ≥{extreme_threshold}mm')
    ax.set_title('C) Extreme Precipitation Events', fontweight='bold', loc='left')
    ax.legend(loc='upper right')

    # --- Panel D: Seasonal distribution of extremes ---
    ax = axes[1, 1]
    extreme_df = df[df['extreme']]
    extreme_by_month = extreme_df.groupby('month').size()

    # Fill missing months with 0
    extreme_by_month = extreme_by_month.reindex(range(1, 13), fill_value=0)

    colors_season = ['#2166ac'] * 2 + ['#67a9cf'] * 3 + ['#d1e5f0'] * 3 + ['#67a9cf'] * 2 + ['#2166ac'] * 2
    ax.bar(months, extreme_by_month.values, color=colors_season, alpha=0.8)

    ax.set_xlabel('Month')
    ax.set_ylabel('Count of Extreme Days')
    ax.set_title('D) Seasonal Distribution of Extremes', fontweight='bold', loc='left')
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_xlim(0.5, 12.5)

    plt.suptitle('San Diego Coastal Precipitation Patterns (2017-2025)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_antecedent_conditions(
    data: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Create figure showing antecedent precipitation index and soil moisture proxy."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df = data['delmar'].copy()

    # --- Panel A: API time series for a wet year ---
    ax = axes[0, 0]
    wet_year = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2023-12-31')].copy()

    ax.fill_between(wet_year['date'], 0, wet_year['precip_mm'],
                    color='steelblue', alpha=0.3, label='Daily Precip')
    ax2 = ax.twinx()
    ax2.plot(wet_year['date'], wet_year['api'], color='darkgreen',
             linewidth=2, label='API')

    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Precipitation (mm)', color='steelblue')
    ax2.set_ylabel('Antecedent Precipitation Index', color='darkgreen')
    ax.set_title('A) Precipitation & API - 2023 (Wet Year)', fontweight='bold', loc='left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # --- Panel B: Cumulative precipitation comparison ---
    ax = axes[0, 1]

    for window, color, label in [(7, '#a6cee3', '7-day'),
                                  (30, '#1f78b4', '30-day'),
                                  (90, '#08306b', '90-day')]:
        col = f'precip_{window}d'
        ax.plot(wet_year['date'], wet_year[col], color=color,
                linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Precipitation (mm)')
    ax.set_title('B) Rolling Precipitation Windows - 2023', fontweight='bold', loc='left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(loc='upper right')

    # --- Panel C: Days since rain distribution ---
    ax = axes[1, 0]

    # Histogram of dry spell lengths
    dry_spells = df['days_since_rain'].values
    bins = np.arange(0, min(100, dry_spells.max()) + 5, 5)

    ax.hist(dry_spells, bins=bins, color='sandybrown', alpha=0.7, edgecolor='black')
    ax.axvline(dry_spells.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {dry_spells.mean():.1f} days')
    ax.axvline(np.median(dry_spells), color='orange', linestyle=':',
               linewidth=2, label=f'Median: {np.median(dry_spells):.1f} days')

    ax.set_xlabel('Days Since Last Rain (>1mm)')
    ax.set_ylabel('Frequency')
    ax.set_title('C) Dry Spell Length Distribution', fontweight='bold', loc='left')
    ax.legend(loc='upper right')

    # --- Panel D: Wet-dry cycles by season ---
    ax = axes[1, 1]

    df['season'] = df['date'].dt.month.map(
        lambda m: 'Winter' if m in [12, 1, 2] else
                  'Spring' if m in [3, 4, 5] else
                  'Summer' if m in [6, 7, 8] else 'Fall'
    )

    seasonal_cycles = df.groupby('season')['wet_dry_cycles_30d'].mean()
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_cycles = seasonal_cycles.reindex(season_order)

    colors = ['#2166ac', '#92c5de', '#f4a582', '#b2182b']
    ax.bar(season_order, seasonal_cycles.values, color=colors, alpha=0.8)

    ax.set_xlabel('Season')
    ax.set_ylabel('Mean Wet-Dry Cycles (30-day window)')
    ax.set_title('D) Seasonal Wetting-Drying Cycles', fontweight='bold', loc='left')

    plt.suptitle('Antecedent Moisture Conditions - Del Mar',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_spatial_variation(
    data: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Create figure showing spatial variation in precipitation across beaches."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: Annual precipitation by beach ---
    ax = axes[0, 0]

    annual_by_beach = {}
    for beach in BEACH_ORDER:
        df = data[beach]
        annual = df.groupby(df['date'].dt.year)['precip_mm'].sum()
        annual_by_beach[beach] = annual.mean()

    beaches = list(annual_by_beach.keys())
    values = list(annual_by_beach.values())
    colors = [BEACH_COLORS[b] for b in beaches]

    ax.barh(beaches, values, color=colors, alpha=0.8)
    ax.set_xlabel('Mean Annual Precipitation (mm)')
    ax.set_title('A) Annual Precipitation by Beach', fontweight='bold', loc='left')
    ax.invert_yaxis()

    # --- Panel B: North-south gradient for a storm event ---
    ax = axes[1, 0]

    # Find a significant storm (high precip day)
    df_delmar = data['delmar']
    storm_date = df_delmar.loc[df_delmar['precip_mm'].idxmax(), 'date']

    storm_precip = []
    for beach in BEACH_ORDER:
        df = data[beach]
        precip = df[df['date'] == storm_date]['precip_mm'].values[0]
        storm_precip.append(precip)

    ax.plot(range(len(BEACH_ORDER)), storm_precip, 'o-', color='steelblue',
            markersize=10, linewidth=2)
    ax.set_xticks(range(len(BEACH_ORDER)))
    ax.set_xticklabels([b.title() for b in BEACH_ORDER], rotation=45, ha='right')
    ax.set_ylabel('Precipitation (mm)')
    ax.set_title(f'B) N-S Gradient: Storm of {storm_date.strftime("%Y-%m-%d")}',
                 fontweight='bold', loc='left')
    ax.set_xlabel('Beach (South → North)')

    # --- Panel C: Correlation matrix of daily precip ---
    ax = axes[0, 1]

    # Build correlation matrix
    precip_df = pd.DataFrame({
        beach: data[beach].set_index('date')['precip_mm']
        for beach in BEACH_ORDER
    })
    corr = precip_df.corr()

    im = ax.imshow(corr, cmap='RdYlBu_r', vmin=0.8, vmax=1.0)
    ax.set_xticks(range(len(BEACH_ORDER)))
    ax.set_yticks(range(len(BEACH_ORDER)))
    ax.set_xticklabels([b.title() for b in BEACH_ORDER], rotation=45, ha='right')
    ax.set_yticklabels([b.title() for b in BEACH_ORDER])

    # Add correlation values
    for i in range(len(BEACH_ORDER)):
        for j in range(len(BEACH_ORDER)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
    ax.set_title('C) Inter-Beach Precipitation Correlation', fontweight='bold', loc='left')

    # --- Panel D: API variation across beaches ---
    ax = axes[1, 1]

    api_data = []
    for beach in BEACH_ORDER:
        df = data[beach]
        api_data.append(df['api'].values)

    bp = ax.boxplot(api_data, labels=[b.title() for b in BEACH_ORDER],
                    patch_artist=True)

    for patch, color in zip(bp['boxes'], [BEACH_COLORS[b] for b in BEACH_ORDER]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Antecedent Precipitation Index')
    ax.set_title('D) API Distribution by Beach', fontweight='bold', loc='left')
    ax.set_xticklabels([b.title() for b in BEACH_ORDER], rotation=45, ha='right')

    plt.suptitle('Spatial Variation in Precipitation Across Study Area',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_erosion_potential(
    data: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Create figure relating atmospheric conditions to erosion potential."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df = data['delmar'].copy()
    df['month'] = df['date'].dt.month

    # --- Panel A: High erosion risk periods ---
    ax = axes[0, 0]

    # Define high risk: high API + recent heavy rain
    df['high_risk'] = (df['api'] > df['api'].quantile(0.75)) & (df['precip_7d'] > 50)

    risk_by_month = df.groupby('month')['high_risk'].mean() * 100

    colors = plt.cm.Reds(risk_by_month.values / risk_by_month.max())
    bars = ax.bar(range(1, 13), risk_by_month.values, color=colors)

    ax.set_xlabel('Month')
    ax.set_ylabel('% Days with High Erosion Potential')
    ax.set_title('A) Seasonal Erosion Risk (High API + Recent Rain)',
                 fontweight='bold', loc='left')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

    # --- Panel B: VPD and drying stress ---
    ax = axes[0, 1]

    vpd_by_month = df.groupby('month')['vpd'].agg(['mean', 'std'])

    ax.bar(range(1, 13), vpd_by_month['mean'], color='orange', alpha=0.7)
    ax.errorbar(range(1, 13), vpd_by_month['mean'], yerr=vpd_by_month['std'],
                fmt='none', color='black', capsize=3)

    ax.set_xlabel('Month')
    ax.set_ylabel('Vapor Pressure Deficit (kPa)')
    ax.set_title('B) Seasonal Drying Stress (VPD)', fontweight='bold', loc='left')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

    # --- Panel C: Wetting-drying cycle frequency ---
    ax = axes[1, 0]

    # Time series of 90-day wet-dry cycles
    ax.plot(df['date'], df['wet_dry_cycles_90d'], color='purple',
            alpha=0.7, linewidth=0.5)

    # Add smoothed trend
    rolling_mean = df.set_index('date')['wet_dry_cycles_90d'].rolling('365D').mean()
    ax.plot(rolling_mean.index, rolling_mean.values, color='darkred',
            linewidth=2, label='1-year rolling mean')

    ax.set_xlabel('Date')
    ax.set_ylabel('Wet-Dry Cycles (90-day window)')
    ax.set_title('C) Weathering Fatigue: Wetting-Drying Cycles',
                 fontweight='bold', loc='left')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # --- Panel D: Feature correlation heatmap ---
    ax = axes[1, 1]

    # Select key features for correlation
    features = ['precip_mm', 'api', 'precip_30d', 'days_since_rain',
                'wet_dry_cycles_30d', 'vpd', 'temp_mean_c']
    feature_labels = ['Daily\nPrecip', 'API', '30-day\nPrecip', 'Days\nSince Rain',
                      'Wet-Dry\nCycles', 'VPD', 'Mean\nTemp']

    corr = df[features].corr()

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(feature_labels, fontsize=8)
    ax.set_yticklabels(feature_labels, fontsize=8)

    # Add correlation values
    for i in range(len(features)):
        for j in range(len(features)):
            color = 'white' if abs(corr.iloc[i, j]) > 0.5 else 'black'
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
    ax.set_title('D) Feature Correlation Matrix', fontweight='bold', loc='left')

    plt.suptitle('Atmospheric Conditions & Cliff Erosion Potential',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_data_cube_structure(
    data: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Create schematic figure showing atmospheric data cube and feature vector structure.

    This figure illustrates:
    1. The 3D data cube structure (beaches × time × features)
    2. The 24 atmospheric features organized by category
    3. The 90-day lookback window alignment to scan dates
    4. How the cube feeds into the model's EnvironmentalEncoder
    """
    fig = plt.figure(figsize=(18, 14))

    # Use gridspec for flexible layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], width_ratios=[1.2, 1, 1],
                          hspace=0.35, wspace=0.3)

    # === Panel A: 3D Data Cube Visualization ===
    ax_cube = fig.add_subplot(gs[0, 0], projection='3d')

    # Cube dimensions
    n_beaches = 6
    n_days = 90  # lookback window
    n_features = 24

    # Draw cube wireframe
    # Front face
    ax_cube.plot([0, n_features], [0, 0], [0, 0], 'k-', linewidth=2)
    ax_cube.plot([0, n_features], [n_days, n_days], [0, 0], 'k-', linewidth=2)
    ax_cube.plot([0, 0], [0, n_days], [0, 0], 'k-', linewidth=2)
    ax_cube.plot([n_features, n_features], [0, n_days], [0, 0], 'k-', linewidth=2)

    # Back face
    ax_cube.plot([0, n_features], [0, 0], [n_beaches, n_beaches], 'k-', linewidth=2)
    ax_cube.plot([0, n_features], [n_days, n_days], [n_beaches, n_beaches], 'k-', linewidth=2)
    ax_cube.plot([0, 0], [0, n_days], [n_beaches, n_beaches], 'k-', linewidth=2)
    ax_cube.plot([n_features, n_features], [0, n_days], [n_beaches, n_beaches], 'k-', linewidth=2)

    # Connecting edges
    ax_cube.plot([0, 0], [0, 0], [0, n_beaches], 'k-', linewidth=2)
    ax_cube.plot([n_features, n_features], [0, 0], [0, n_beaches], 'k-', linewidth=2)
    ax_cube.plot([0, 0], [n_days, n_days], [0, n_beaches], 'k-', linewidth=2)
    ax_cube.plot([n_features, n_features], [n_days, n_days], [0, n_beaches], 'k-', linewidth=2)

    # Fill faces with semi-transparent colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Time-Feature face (front, z=0)
    verts_tf = [[(0, 0, 0), (n_features, 0, 0), (n_features, n_days, 0), (0, n_days, 0)]]
    ax_cube.add_collection3d(Poly3DCollection(verts_tf, alpha=0.3, facecolor='steelblue', edgecolor='k'))

    # Beach-Feature face (bottom, y=0)
    verts_bf = [[(0, 0, 0), (n_features, 0, 0), (n_features, 0, n_beaches), (0, 0, n_beaches)]]
    ax_cube.add_collection3d(Poly3DCollection(verts_bf, alpha=0.3, facecolor='coral', edgecolor='k'))

    # Beach-Time face (left, x=0)
    verts_bt = [[(0, 0, 0), (0, n_days, 0), (0, n_days, n_beaches), (0, 0, n_beaches)]]
    ax_cube.add_collection3d(Poly3DCollection(verts_bt, alpha=0.3, facecolor='lightgreen', edgecolor='k'))

    # Dimension labels with arrows
    ax_cube.set_xlabel('Features (24)', fontsize=11, fontweight='bold', labelpad=10)
    ax_cube.set_ylabel('Time (90 days)', fontsize=11, fontweight='bold', labelpad=10)
    ax_cube.set_zlabel('Beaches (6)', fontsize=11, fontweight='bold', labelpad=10)

    # Add dimension values
    ax_cube.text(n_features/2, -8, -1, f'D = {n_features}', fontsize=10, ha='center')
    ax_cube.text(-5, n_days/2, -1, f'T = {n_days}', fontsize=10, ha='center')
    ax_cube.text(-5, -5, n_beaches/2, f'B = {n_beaches}', fontsize=10, ha='center')

    ax_cube.set_xlim(0, n_features)
    ax_cube.set_ylim(0, n_days)
    ax_cube.set_zlim(0, n_beaches)
    ax_cube.view_init(elev=20, azim=45)
    ax_cube.set_title('A) Atmospheric Data Cube Structure\n(B, T, D) = (6 beaches, 90 days, 24 features)',
                      fontsize=12, fontweight='bold', pad=15)

    # Hide axis ticks for cleaner look
    ax_cube.set_xticks([])
    ax_cube.set_yticks([])
    ax_cube.set_zticks([])

    # === Panel B: Feature Vector Categories ===
    ax_features = fig.add_subplot(gs[0, 1:])
    ax_features.axis('off')

    # Feature categories with colors
    feature_categories = {
        'Raw Measurements\n(from PRISM)': {
            'color': '#e41a1c',
            'features': [
                ('precip_mm', 'Daily precipitation'),
                ('temp_mean_c', 'Mean temperature'),
                ('temp_min_c', 'Min temperature'),
                ('temp_max_c', 'Max temperature'),
                ('dewpoint_c', 'Dewpoint temperature'),
            ]
        },
        'Cumulative\nPrecipitation': {
            'color': '#377eb8',
            'features': [
                ('precip_7d', '7-day rolling sum'),
                ('precip_30d', '30-day rolling sum'),
                ('precip_60d', '60-day rolling sum'),
                ('precip_90d', '90-day rolling sum'),
            ]
        },
        'Antecedent\nConditions': {
            'color': '#4daf4a',
            'features': [
                ('api', 'Antecedent precip index (k=0.9)'),
                ('days_since_rain', 'Days since >1mm rain'),
                ('consecutive_dry_days', 'Dry days before event'),
            ]
        },
        'Intensity\nMetrics': {
            'color': '#984ea3',
            'features': [
                ('rain_day_flag', 'Binary: precip >1mm'),
                ('intensity_class', '0=none/1=light/2=mod/3=heavy'),
                ('max_precip_7d', 'Max daily in 7 days'),
                ('max_precip_30d', 'Max daily in 30 days'),
            ]
        },
        'Wetting-Drying\nCycles': {
            'color': '#ff7f00',
            'features': [
                ('wet_dry_cycles_30d', '30-day cycle count'),
                ('wet_dry_cycles_90d', '90-day cycle count'),
            ]
        },
        'Evaporative\nDemand': {
            'color': '#a65628',
            'features': [
                ('vpd', 'Vapor pressure deficit (kPa)'),
                ('vpd_7d_mean', '7-day mean VPD'),
            ]
        },
        'Freeze-Thaw\n(for transfer)': {
            'color': '#666666',
            'features': [
                ('freeze_flag', 'Binary: Tmin < 0°C'),
                ('marginal_freeze_flag', 'Binary: Tmin < 2°C'),
                ('freeze_thaw_cycles_30d', '30-day F-T cycles'),
                ('freeze_thaw_cycles_season', 'Water year F-T total'),
            ]
        },
    }

    # Calculate positions
    n_cats = len(feature_categories)
    cat_width = 0.13
    cat_spacing = 0.14
    start_x = 0.02

    ax_features.set_title('B) Feature Vector: 24 Atmospheric Features by Category',
                          fontsize=12, fontweight='bold', loc='left', pad=10)

    for i, (cat_name, cat_info) in enumerate(feature_categories.items()):
        x = start_x + i * cat_spacing
        color = cat_info['color']
        features = cat_info['features']

        # Category header box
        rect = mpatches.FancyBboxPatch(
            (x, 0.75), cat_width - 0.01, 0.18,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='black', linewidth=2, alpha=0.8
        )
        ax_features.add_patch(rect)
        ax_features.text(x + cat_width/2 - 0.005, 0.84, cat_name,
                        fontsize=8, fontweight='bold', ha='center', va='center',
                        color='white')

        # Feature list
        y_pos = 0.68
        for feat_name, feat_desc in features:
            ax_features.text(x + 0.005, y_pos, f'• {feat_name}', fontsize=7,
                           fontweight='bold', color=color, va='top')
            ax_features.text(x + 0.005, y_pos - 0.06, f'  {feat_desc}', fontsize=6,
                           color='gray', va='top')
            y_pos -= 0.13

    ax_features.set_xlim(0, 1)
    ax_features.set_ylim(0, 1)

    # === Panel C: Temporal Alignment Diagram ===
    ax_temporal = fig.add_subplot(gs[1, :])
    ax_temporal.axis('off')

    ax_temporal.set_title('C) Temporal Alignment: 90-Day Lookback Window',
                          fontsize=12, fontweight='bold', loc='left', pad=10)

    # Timeline
    ax_temporal.arrow(0.05, 0.5, 0.88, 0, head_width=0.05, head_length=0.02,
                     fc='black', ec='black')
    ax_temporal.text(0.95, 0.5, 'Time', fontsize=10, va='center')

    # Scan date marker
    scan_x = 0.85
    ax_temporal.plot([scan_x], [0.5], 'rv', markersize=15, zorder=5)
    ax_temporal.text(scan_x, 0.35, 'LiDAR\nScan Date', fontsize=9, ha='center', fontweight='bold')

    # 90-day window
    window_start = scan_x - 0.45
    window_rect = mpatches.FancyBboxPatch(
        (window_start, 0.45), 0.45, 0.10,
        boxstyle="round,pad=0.01",
        facecolor='steelblue', edgecolor='darkblue', linewidth=2, alpha=0.3
    )
    ax_temporal.add_patch(window_rect)

    # Window annotation
    ax_temporal.annotate('', xy=(window_start, 0.65), xytext=(scan_x, 0.65),
                        arrowprops=dict(arrowstyle='<->', color='darkblue', lw=2))
    ax_temporal.text((window_start + scan_x) / 2, 0.72, '90 days lookback',
                    fontsize=10, ha='center', fontweight='bold', color='darkblue')

    # Sample days within window
    for i, day_offset in enumerate([0, 30, 60, 89]):
        x = scan_x - (day_offset / 90) * 0.45
        ax_temporal.plot([x], [0.5], 'ko', markersize=6)
        ax_temporal.text(x, 0.58, f't-{day_offset}', fontsize=7, ha='center', rotation=45)

    # Output shape annotation
    ax_temporal.text(0.5, 0.15, 'Output: (90, 24) feature matrix per beach per scan',
                    fontsize=11, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    ax_temporal.set_xlim(0, 1)
    ax_temporal.set_ylim(0, 1)

    # === Panel D: Sample Feature Vector Heatmap ===
    ax_heatmap = fig.add_subplot(gs[2, 0])

    # Get sample data from Del Mar
    df = data['delmar'].copy()

    # Get a 90-day window sample
    sample_end = df['date'].iloc[-1]
    sample_start = sample_end - pd.Timedelta(days=89)
    sample_df = df[(df['date'] >= sample_start) & (df['date'] <= sample_end)].copy()

    # Select feature columns (24 features)
    feature_cols = [
        'precip_mm', 'temp_mean_c', 'temp_min_c', 'temp_max_c', 'dewpoint_c',
        'precip_7d', 'precip_30d', 'precip_60d', 'precip_90d',
        'api', 'days_since_rain', 'consecutive_dry_days',
        'rain_day_flag', 'intensity_class', 'max_precip_7d', 'max_precip_30d',
        'wet_dry_cycles_30d', 'wet_dry_cycles_90d',
        'vpd', 'vpd_7d_mean',
        'freeze_flag', 'marginal_freeze_flag', 'freeze_thaw_cycles_30d', 'freeze_thaw_cycles_season'
    ]

    # Normalize features for visualization
    sample_matrix = sample_df[feature_cols].values
    sample_normalized = (sample_matrix - sample_matrix.min(axis=0)) / (sample_matrix.max(axis=0) - sample_matrix.min(axis=0) + 1e-8)

    im = ax_heatmap.imshow(sample_normalized.T, aspect='auto', cmap='viridis')
    ax_heatmap.set_xlabel('Days (0 = oldest, 89 = scan date)', fontsize=10)
    ax_heatmap.set_ylabel('Feature Index', fontsize=10)
    ax_heatmap.set_title('D) Sample Feature Matrix (90×24)\nDel Mar, normalized',
                         fontsize=11, fontweight='bold', loc='left')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
    cbar.set_label('Normalized Value', fontsize=9)

    # === Panel E: Beach dimension illustration ===
    ax_beaches = fig.add_subplot(gs[2, 1])
    ax_beaches.axis('off')

    ax_beaches.set_title('E) Beach Dimension (B=6)',
                         fontsize=11, fontweight='bold', loc='left')

    beach_names = ['Blacks', 'Torrey', 'Del Mar', 'Solana', 'San Elijo', 'Encinitas']
    beach_colors_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    for i, (name, color) in enumerate(zip(beach_names, beach_colors_list)):
        y = 0.85 - i * 0.13
        rect = mpatches.FancyBboxPatch(
            (0.1, y - 0.04), 0.8, 0.10,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7
        )
        ax_beaches.add_patch(rect)
        ax_beaches.text(0.5, y, f'{i}: {name}', fontsize=10, ha='center', va='center',
                       fontweight='bold', color='white')

    ax_beaches.text(0.5, 0.05, 'South → North ordering', fontsize=9, ha='center',
                   style='italic', color='gray')
    ax_beaches.set_xlim(0, 1)
    ax_beaches.set_ylim(0, 1)

    # === Panel F: Model integration ===
    ax_model = fig.add_subplot(gs[2, 2])
    ax_model.axis('off')

    ax_model.set_title('F) Model Integration',
                       fontsize=11, fontweight='bold', loc='left')

    # Input box
    input_rect = mpatches.FancyBboxPatch(
        (0.1, 0.7), 0.8, 0.20,
        boxstyle="round,pad=0.02",
        facecolor='lightblue', edgecolor='darkblue', linewidth=2
    )
    ax_model.add_patch(input_rect)
    ax_model.text(0.5, 0.8, 'Atmospheric Input\n(B, 90, 24)', fontsize=10,
                 ha='center', va='center', fontweight='bold')

    # Arrow
    ax_model.arrow(0.5, 0.65, 0, -0.15, head_width=0.05, head_length=0.03,
                  fc='black', ec='black')

    # Encoder box
    encoder_rect = mpatches.FancyBboxPatch(
        (0.1, 0.35), 0.8, 0.15,
        boxstyle="round,pad=0.02",
        facecolor='coral', edgecolor='darkred', linewidth=2
    )
    ax_model.add_patch(encoder_rect)
    ax_model.text(0.5, 0.425, 'Environmental\nEncoder', fontsize=10,
                 ha='center', va='center', fontweight='bold')

    # Arrow
    ax_model.arrow(0.5, 0.3, 0, -0.12, head_width=0.05, head_length=0.03,
                  fc='black', ec='black')

    # Output box
    output_rect = mpatches.FancyBboxPatch(
        (0.1, 0.05), 0.8, 0.13,
        boxstyle="round,pad=0.02",
        facecolor='lightgreen', edgecolor='darkgreen', linewidth=2
    )
    ax_model.add_patch(output_rect)
    ax_model.text(0.5, 0.115, 'Embeddings\n(B, 90, d_model)', fontsize=10,
                 ha='center', va='center', fontweight='bold')

    ax_model.set_xlim(0, 1)
    ax_model.set_ylim(0, 1)

    plt.suptitle('Atmospheric Data Cube & Feature Vector Structure',
                 fontsize=16, fontweight='bold', y=0.98)

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_time_series_overview(
    data: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Create a comprehensive time series overview figure."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    df = data['delmar'].copy()

    # --- Panel A: Daily precipitation ---
    ax = axes[0]
    ax.bar(df['date'], df['precip_mm'], color='steelblue', alpha=0.7, width=1)
    ax.set_ylabel('Precip (mm)')
    ax.set_title('Daily Precipitation', fontweight='bold', loc='left')
    ax.set_ylim(0, df['precip_mm'].max() * 1.1)

    # --- Panel B: Temperature range ---
    ax = axes[1]
    ax.fill_between(df['date'], df['temp_min_c'], df['temp_max_c'],
                    color='coral', alpha=0.3, label='Temp Range')
    ax.plot(df['date'], df['temp_mean_c'], color='red', linewidth=0.5,
            alpha=0.7, label='Mean Temp')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Range', fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=8)

    # --- Panel C: Antecedent Precipitation Index ---
    ax = axes[2]
    ax.fill_between(df['date'], 0, df['api'], color='darkgreen', alpha=0.5)
    ax.set_ylabel('API')
    ax.set_title('Antecedent Precipitation Index', fontweight='bold', loc='left')

    # --- Panel D: 90-day cumulative precip ---
    ax = axes[3]
    ax.fill_between(df['date'], 0, df['precip_90d'], color='darkblue', alpha=0.5)
    ax.set_ylabel('90-day Precip (mm)')
    ax.set_title('90-Day Cumulative Precipitation', fontweight='bold', loc='left')
    ax.set_xlabel('Date')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.suptitle('Del Mar Atmospheric Time Series (2017-2025)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate atmospheric feature visualizations'
    )
    parser.add_argument(
        '--atmos-dir',
        type=Path,
        default=Path('data/processed/atmospheric'),
        help='Directory containing processed atmospheric parquet files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('figures/appendix'),
        help='Directory to save output figures'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading atmospheric data...")
    data = load_all_beaches(args.atmos_dir)

    if not data:
        logger.error("No data loaded!")
        return

    # Generate figures
    logger.info("\nGenerating figures...")

    plot_precipitation_climatology(
        data, args.output_dir / 'atmos_precipitation_climatology.png'
    )

    plot_antecedent_conditions(
        data, args.output_dir / 'atmos_antecedent_conditions.png'
    )

    plot_spatial_variation(
        data, args.output_dir / 'atmos_spatial_variation.png'
    )

    plot_erosion_potential(
        data, args.output_dir / 'atmos_erosion_potential.png'
    )

    plot_time_series_overview(
        data, args.output_dir / 'atmos_time_series_overview.png'
    )

    plot_data_cube_structure(
        data, args.output_dir / 'atmos_data_cube_structure.png'
    )

    logger.info(f"\nAll figures saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
