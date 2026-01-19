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

    logger.info(f"\nAll figures saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
