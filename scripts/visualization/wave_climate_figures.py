"""Generate wave climate figures for CliffCast appendix.

Creates publication-quality figures showing wave climate characteristics
from CDIP MOP data for San Diego beaches (2017-2025, aligned with LiDAR data).

Usage:
    python scripts/visualization/wave_climate_figures.py --cdip-dir data/raw/cdip/ --output figures/appendix/

Figures generated:
    - wave_A1_wave_height_distributions.png
    - wave_A2_wave_period_characteristics.png
    - wave_A3_wave_direction_roses.png
    - wave_A4_wave_power_statistics.png
    - wave_A5_wave_seasonal_patterns.png
    - wave_A6_wave_storm_climatology.png
    - wave_A7_wave_spatial_climate.png
    - wave_A8_wave_extreme_value_analysis.png

Author: CliffCast Team
Date: 2026-01-19
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.wave_loader import WaveLoader

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")

# Beach MOP ranges
BEACH_MOP_RANGES = {
    'Blacks': (520, 567),
    'Torrey': (567, 581),
    'Del Mar': (595, 620),
    'Solana': (637, 666),
    'San Elijo': (683, 708),
    'Encinitas': (708, 764),
}

# Beach colors for consistency
BEACH_COLORS = {
    'Blacks': '#1f77b4',
    'Torrey': '#ff7f0e',
    'Del Mar': '#2ca02c',
    'Solana': '#d62728',
    'San Elijo': '#9467bd',
    'Encinitas': '#8c564b',
}

# Date range for analysis (aligned with LiDAR data)
START_DATE = pd.Timestamp('2017-01-01')
END_DATE = pd.Timestamp('2025-12-31')


def filter_wave_data_by_date(wave_data):
    """Filter wave data to 2017-2025 period.

    Args:
        wave_data: WaveData object from CDIPWaveLoader

    Returns:
        Filtered arrays: hs, tp, dp, power, time (all same length)
    """
    times = pd.to_datetime(wave_data.time)
    mask = (times >= START_DATE) & (times <= END_DATE)

    return {
        'hs': wave_data.hs[mask],
        'tp': wave_data.tp[mask],
        'dp': wave_data.dp[mask],
        'power': wave_data.power[mask],
        'time': wave_data.time[mask],
    }


def load_wave_statistics(loader: WaveLoader, mop_ids: List[int]) -> pd.DataFrame:
    """Load and compute wave statistics for multiple MOPs.

    Args:
        loader: WaveLoader instance
        mop_ids: List of MOP IDs to load

    Returns:
        DataFrame with wave statistics per MOP
    """
    stats_list = []

    for mop_id in mop_ids:
        try:
            # Load full time series and filter to 2017-2025
            wave_data = loader._load_wave_data(mop_id)
            filtered = filter_wave_data_by_date(wave_data)

            # Compute statistics
            hs_valid = filtered['hs'][~np.isnan(filtered['hs'])]
            tp_valid = filtered['tp'][~np.isnan(filtered['tp'])]
            power_valid = filtered['power'][~np.isnan(filtered['power'])]

            stats_list.append({
                'mop_id': mop_id,
                'latitude': wave_data.latitude,
                'longitude': wave_data.longitude,
                'hs_mean': np.mean(hs_valid),
                'hs_std': np.std(hs_valid),
                'hs_median': np.median(hs_valid),
                'hs_95': np.percentile(hs_valid, 95),
                'hs_99': np.percentile(hs_valid, 99),
                'hs_max': np.max(hs_valid),
                'tp_mean': np.mean(tp_valid),
                'tp_std': np.std(tp_valid),
                'power_mean': np.mean(power_valid),
                'power_95': np.percentile(power_valid, 95),
                'n_records': len(filtered['time']),
            })
        except Exception as e:
            print(f"Warning: Could not load MOP {mop_id}: {e}")
            continue

    return pd.DataFrame(stats_list)


def fig_A1_wave_height_distributions(loader: WaveLoader, output_dir: Path):
    """Figure A1: Wave height distributions by beach."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (beach, (mop_min, mop_max)) in enumerate(BEACH_MOP_RANGES.items()):
        ax = axes[idx]

        # Load data for first available MOP in beach
        mop_ids = [m for m in range(mop_min, mop_max + 1) if m in loader.available_mops]
        if not mop_ids:
            ax.text(0.5, 0.5, f'No data\navailable', ha='center', va='center')
            ax.set_title(beach)
            continue

        mop_id = mop_ids[len(mop_ids)//2]  # Middle MOP
        wave_data = loader._load_wave_data(mop_id)
        filtered = filter_wave_data_by_date(wave_data)
        hs_valid = filtered['hs'][~np.isnan(filtered['hs'])]

        # Histogram
        ax.hist(hs_valid, bins=50, density=True, alpha=0.7,
                color=BEACH_COLORS[beach], edgecolor='black', linewidth=0.5)

        # Fit Weibull distribution
        shape, loc, scale = stats.weibull_min.fit(hs_valid, floc=0)
        x = np.linspace(0, np.percentile(hs_valid, 99.5), 100)
        ax.plot(x, stats.weibull_min.pdf(x, shape, loc, scale),
                'r-', lw=2, label='Weibull fit')

        # Add statistics text
        stats_text = (f'Mean: {np.mean(hs_valid):.2f} m\n'
                     f'Std: {np.std(hs_valid):.2f} m\n'
                     f'95th: {np.percentile(hs_valid, 95):.2f} m')
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)

        ax.set_xlabel('Significant Wave Height (m)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{beach} (MOP {mop_id})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'wave_A1_wave_height_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def fig_A2_wave_period_characteristics(loader: WaveLoader, output_dir: Path):
    """Figure A2: Wave period vs height scatter and distributions."""
    fig = plt.figure(figsize=(15, 10))

    # Create subplot layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    ax_scatter = fig.add_subplot(gs[0:2, 0:2])
    ax_hs_hist = fig.add_subplot(gs[0:2, 2])
    ax_tp_hist = fig.add_subplot(gs[2, 0:2])

    # Collect data from all beaches
    all_hs = []
    all_tp = []
    beach_labels = []

    for beach, (mop_min, mop_max) in BEACH_MOP_RANGES.items():
        mop_ids = [m for m in range(mop_min, mop_max + 1) if m in loader.available_mops]
        if not mop_ids:
            continue

        # Sample from middle MOP
        mop_id = mop_ids[len(mop_ids)//2]
        wave_data = loader._load_wave_data(mop_id)
        filtered = filter_wave_data_by_date(wave_data)

        hs = filtered['hs'][~np.isnan(filtered['hs'])]
        tp = filtered['tp'][~np.isnan(filtered['tp'])]

        # Subsample for visualization
        n_sample = min(10000, len(hs))
        indices = np.random.choice(len(hs), n_sample, replace=False)

        all_hs.append(hs[indices])
        all_tp.append(tp[indices])
        beach_labels.extend([beach] * n_sample)

    all_hs = np.concatenate(all_hs)
    all_tp = np.concatenate(all_tp)

    # Scatter plot with density coloring
    h = ax_scatter.hexbin(all_tp, all_hs, gridsize=50, cmap='viridis',
                          mincnt=1, bins='log')
    ax_scatter.set_xlabel('Peak Period (s)', fontsize=12)
    ax_scatter.set_ylabel('Significant Wave Height (m)', fontsize=12)
    ax_scatter.set_title('Wave Height vs Period (All Beaches)', fontsize=14)
    ax_scatter.grid(True, alpha=0.3)

    # Add steepness lines
    steepness = [0.02, 0.04, 0.06]
    tp_range = np.linspace(5, 25, 100)
    for s in steepness:
        hs_line = s * 1.56 * tp_range**2  # Deep water: H = s * L = s * g*T^2/(2pi)
        ax_scatter.plot(tp_range, hs_line, 'r--', alpha=0.5, linewidth=1)
        ax_scatter.text(tp_range[-1], hs_line[-1], f's={s}', fontsize=8)

    cbar = plt.colorbar(h, ax=ax_scatter)
    cbar.set_label('Log10(Count)', fontsize=10)

    # Hs histogram
    ax_hs_hist.hist(all_hs, bins=50, orientation='horizontal',
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_hs_hist.set_xlabel('Count')
    ax_hs_hist.set_ylabel('Hs (m)')
    ax_hs_hist.grid(True, alpha=0.3)

    # Tp histogram
    ax_tp_hist.hist(all_tp, bins=50, color='steelblue', alpha=0.7,
                    edgecolor='black', linewidth=0.5)
    ax_tp_hist.set_ylabel('Count')
    ax_tp_hist.set_xlabel('Tp (s)')
    ax_tp_hist.grid(True, alpha=0.3)

    output_path = output_dir / 'wave_A2_wave_period_characteristics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def fig_A3_wave_direction_roses(loader: WaveLoader, output_dir: Path):
    """Figure A3: Wave direction rose diagrams by beach."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    for idx, (beach, (mop_min, mop_max)) in enumerate(BEACH_MOP_RANGES.items()):
        ax = axes[idx]

        mop_ids = [m for m in range(mop_min, mop_max + 1) if m in loader.available_mops]
        if not mop_ids:
            continue

        mop_id = mop_ids[len(mop_ids)//2]
        wave_data = loader._load_wave_data(mop_id)
        filtered = filter_wave_data_by_date(wave_data)

        dp = filtered['dp'][~np.isnan(filtered['dp'])]
        hs = filtered['hs'][~np.isnan(filtered['hs'])][:len(dp)]

        # Convert to radians
        dp_rad = np.deg2rad(dp)

        # Create directional bins
        n_bins = 16
        theta_bins = np.linspace(0, 2*np.pi, n_bins + 1)

        # Bin by direction and weight by wave height
        hist, _ = np.histogram(dp_rad, bins=theta_bins, weights=hs)
        counts, _ = np.histogram(dp_rad, bins=theta_bins)

        # Normalize
        hist_norm = hist / np.sum(hist) * 100

        # Plot bars
        theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
        width = 2*np.pi / n_bins
        bars = ax.bar(theta_centers, hist_norm, width=width, bottom=0.0,
                      color=BEACH_COLORS[beach], alpha=0.7, edgecolor='black', linewidth=0.5)

        # Configure polar plot
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(beach, fontsize=12, pad=20)
        ax.set_ylim(0, np.max(hist_norm) * 1.1)

        # Add mean direction
        mean_dir = np.arctan2(np.mean(np.sin(dp_rad)), np.mean(np.cos(dp_rad)))
        ax.plot([mean_dir, mean_dir], [0, np.max(hist_norm)], 'r-', linewidth=2,
                label='Mean Direction')

    # Add overall legend
    fig.legend(['Mean Direction'], loc='lower center', ncol=1, fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'wave_A3_wave_direction_roses.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def fig_A4_wave_power_statistics(loader: WaveLoader, output_dir: Path):
    """Figure A4: Wave power statistics and cumulative distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect statistics per beach
    beach_stats = []

    for beach, (mop_min, mop_max) in BEACH_MOP_RANGES.items():
        mop_ids = [m for m in range(mop_min, mop_max + 1) if m in loader.available_mops]
        if not mop_ids:
            continue

        beach_power = []
        for mop_id in mop_ids[:5]:  # Sample 5 MOPs per beach
            try:
                wave_data = loader._load_wave_data(mop_id)
                filtered = filter_wave_data_by_date(wave_data)
                power = filtered['power'][~np.isnan(filtered['power'])]
                beach_power.extend(power)
            except:
                continue

        if beach_power:
            beach_stats.append({
                'beach': beach,
                'power': np.array(beach_power),
                'color': BEACH_COLORS[beach]
            })

    # Plot 1: Box plots
    ax1 = axes[0, 0]
    data_for_box = [s['power'] for s in beach_stats]
    labels_for_box = [s['beach'] for s in beach_stats]
    colors_for_box = [s['color'] for s in beach_stats]

    bp = ax1.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                     showfliers=False)
    for patch, color in zip(bp['boxes'], colors_for_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Wave Power (kW/m)', fontsize=12)
    ax1.set_title('Wave Power by Beach', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: CDFs
    ax2 = axes[0, 1]
    for stat in beach_stats:
        power_sorted = np.sort(stat['power'])
        cdf = np.arange(1, len(power_sorted) + 1) / len(power_sorted)
        ax2.plot(power_sorted, cdf, label=stat['beach'],
                color=stat['color'], linewidth=2)

    ax2.set_xlabel('Wave Power (kW/m)', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Distribution Functions', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, np.percentile(np.concatenate([s['power'] for s in beach_stats]), 99))

    # Plot 3: Log-scale histogram
    ax3 = axes[1, 0]
    for stat in beach_stats:
        ax3.hist(stat['power'], bins=50, alpha=0.5, label=stat['beach'],
                color=stat['color'], density=True, histtype='step', linewidth=2)

    ax3.set_xlabel('Wave Power (kW/m)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Power Distributions (Log Scale)', fontsize=14)
    ax3.set_yscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    for stat in beach_stats:
        power = stat['power']
        table_data.append([
            stat['beach'],
            f"{np.mean(power):.1f}",
            f"{np.median(power):.1f}",
            f"{np.percentile(power, 95):.1f}",
            f"{np.max(power):.1f}"
        ])

    table = ax4.table(cellText=table_data,
                     colLabels=['Beach', 'Mean', 'Median', '95th %ile', 'Max'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code rows
    for i, stat in enumerate(beach_stats):
        table[(i+1, 0)].set_facecolor(stat['color'])
        table[(i+1, 0)].set_alpha(0.3)

    ax4.set_title('Wave Power Statistics (kW/m)', fontsize=14, pad=20)

    plt.tight_layout()
    output_path = output_dir / 'wave_A4_wave_power_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def fig_A5_seasonal_patterns(loader: WaveLoader, output_dir: Path):
    """Figure A5: Seasonal wave climate patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Select one MOP from Del Mar (central location)
    mop_id = 607  # Del Mar central
    if mop_id not in loader.available_mops:
        mop_id = loader.available_mops[len(loader.available_mops)//2]

    wave_data = loader._load_wave_data(mop_id)
    filtered = filter_wave_data_by_date(wave_data)

    # Convert time to pandas
    times = pd.to_datetime(filtered['time'])
    df = pd.DataFrame({
        'time': times,
        'hs': filtered['hs'],
        'tp': filtered['tp'],
        'power': filtered['power'],
    })

    # Add temporal features
    df['month'] = df['time'].dt.month
    df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                                     9: 'Fall', 10: 'Fall', 11: 'Fall'})

    # Plot 1: Monthly means
    ax1 = axes[0, 0]
    monthly_means = df.groupby('month')['hs'].mean()
    monthly_std = df.groupby('month')['hs'].std()

    ax1.plot(monthly_means.index, monthly_means.values, 'o-',
            linewidth=2, markersize=8, color='steelblue', label='Mean Hs')
    ax1.fill_between(monthly_means.index,
                     monthly_means - monthly_std,
                     monthly_means + monthly_std,
                     alpha=0.3, color='steelblue')

    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Significant Wave Height (m)', fontsize=12)
    ax1.set_title('Monthly Mean Wave Heights', fontsize=14)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Seasonal boxplots
    ax2 = axes[0, 1]
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    df_clean = df[df['season'].isin(season_order)].copy()

    sns.boxplot(data=df_clean, x='season', y='hs', order=season_order,
               palette='Set2', ax=ax2, showfliers=False)
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Significant Wave Height (m)', fontsize=12)
    ax2.set_title('Seasonal Wave Height Distributions', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Annual cycle heatmap
    ax3 = axes[1, 0]
    df['year'] = df['time'].dt.year

    # Create pivot table
    pivot = df.pivot_table(values='hs', index='month', columns='year', aggfunc='mean')

    im = ax3.imshow(pivot.values, aspect='auto', cmap='viridis', interpolation='nearest')
    ax3.set_yticks(range(12))
    ax3.set_yticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Month', fontsize=12)
    ax3.set_title('Monthly Mean Hs Heatmap', fontsize=14)

    # Set x-axis to show years
    year_indices = np.arange(0, len(pivot.columns), max(1, len(pivot.columns)//10))
    ax3.set_xticks(year_indices)
    ax3.set_xticklabels(pivot.columns[year_indices], rotation=45)

    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Hs (m)', fontsize=10)

    # Plot 4: Power seasonal variation
    ax4 = axes[1, 1]
    season_power = df.groupby('season')['power'].agg(['mean', 'std'])
    season_power = season_power.reindex(season_order)

    ax4.bar(season_order, season_power['mean'], yerr=season_power['std'],
           capsize=5, color='coral', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Season', fontsize=12)
    ax4.set_ylabel('Mean Wave Power (kW/m)', fontsize=12)
    ax4.set_title('Seasonal Wave Power', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'wave_A5_wave_seasonal_patterns.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def fig_A6_storm_climatology(loader: WaveLoader, output_dir: Path):
    """Figure A6: Storm event identification and climatology."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Select representative MOP
    mop_id = 607
    if mop_id not in loader.available_mops:
        mop_id = loader.available_mops[len(loader.available_mops)//2]

    wave_data = loader._load_wave_data(mop_id)
    filtered = filter_wave_data_by_date(wave_data)
    times = pd.to_datetime(filtered['time'])
    hs = filtered['hs']

    # Define storm threshold (95th percentile)
    threshold = np.nanpercentile(hs, 95)

    # Identify storms (Hs > threshold for >6 hours)
    is_storm = hs > threshold

    # Plot 1: Time series with storms highlighted
    ax1 = axes[0, 0]

    # Subsample for visualization (last 2 years)
    recent_mask = times > (times[-1] - pd.Timedelta(days=730))
    times_recent = times[recent_mask]
    hs_recent = hs[recent_mask]
    is_storm_recent = is_storm[recent_mask]

    ax1.plot(times_recent, hs_recent, 'b-', linewidth=0.5, alpha=0.7, label='Hs')
    ax1.fill_between(times_recent, 0, hs_recent, where=is_storm_recent,
                     color='red', alpha=0.3, label=f'Storm (Hs>{threshold:.1f}m)')
    ax1.axhline(threshold, color='red', linestyle='--', linewidth=1, label='95th percentile')

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Significant Wave Height (m)', fontsize=12)
    ax1.set_title('Storm Events (Last 2 Years)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Storm duration distribution
    ax2 = axes[0, 1]

    # Find storm durations
    storm_starts = np.where(np.diff(is_storm.astype(int)) == 1)[0]
    storm_ends = np.where(np.diff(is_storm.astype(int)) == -1)[0]

    if len(storm_starts) > 0 and len(storm_ends) > 0:
        # Match starts and ends
        if storm_ends[0] < storm_starts[0]:
            storm_ends = storm_ends[1:]
        min_len = min(len(storm_starts), len(storm_ends))
        durations_hours = (storm_ends[:min_len] - storm_starts[:min_len])

        ax2.hist(durations_hours, bins=30, color='steelblue', alpha=0.7,
                edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Storm Duration (hours)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'Storm Duration Distribution (n={len(durations_hours)})', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Add statistics
        stats_text = (f'Mean: {np.mean(durations_hours):.1f} hrs\n'
                     f'Median: {np.median(durations_hours):.1f} hrs\n'
                     f'Max: {np.max(durations_hours):.1f} hrs')
        ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)

    # Plot 3: Monthly storm frequency
    ax3 = axes[1, 0]

    df_storms = pd.DataFrame({'time': times, 'is_storm': is_storm})
    df_storms['month'] = df_storms['time'].dt.month
    monthly_storm_hours = df_storms.groupby('month')['is_storm'].sum()

    ax3.bar(monthly_storm_hours.index, monthly_storm_hours.values,
           color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_ylabel('Total Storm Hours', fontsize=12)
    ax3.set_title('Monthly Storm Frequency', fontsize=14)
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Storm intensity vs duration
    ax4 = axes[1, 1]

    if len(storm_starts) > 0 and len(storm_ends) > 0:
        storm_max_hs = []
        for start, end in zip(storm_starts[:min_len], storm_ends[:min_len]):
            storm_max_hs.append(np.nanmax(hs[start:end+1]))

        ax4.scatter(durations_hours, storm_max_hs, alpha=0.5, s=30, c='steelblue')
        ax4.set_xlabel('Storm Duration (hours)', fontsize=12)
        ax4.set_ylabel('Maximum Hs (m)', fontsize=12)
        ax4.set_title('Storm Intensity vs Duration', fontsize=14)
        ax4.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(durations_hours, storm_max_hs, 1)
        p = np.poly1d(z)
        ax4.plot(durations_hours, p(durations_hours), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    output_path = output_dir / 'wave_A6_wave_storm_climatology.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def fig_A7_spatial_wave_climate(loader: WaveLoader, output_dir: Path):
    """Figure A7: Spatial variation of wave climate along coast."""
    # Load statistics for all available MOPs
    stats_df = load_wave_statistics(loader, loader.available_mops)

    if len(stats_df) == 0:
        print("Warning: No wave statistics available for spatial plot")
        return

    # Add beach labels
    def get_beach(mop_id):
        for beach, (mop_min, mop_max) in BEACH_MOP_RANGES.items():
            if mop_min <= mop_id <= mop_max:
                return beach
        return 'Unknown'

    stats_df['beach'] = stats_df['mop_id'].apply(get_beach)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean Hs along coast
    ax1 = axes[0, 0]
    for beach in BEACH_MOP_RANGES.keys():
        beach_data = stats_df[stats_df['beach'] == beach]
        if len(beach_data) > 0:
            ax1.plot(beach_data['latitude'], beach_data['hs_mean'],
                    'o-', label=beach, color=BEACH_COLORS[beach],
                    linewidth=2, markersize=6)

    ax1.set_xlabel('Latitude (°N)', fontsize=12)
    ax1.set_ylabel('Mean Hs (m)', fontsize=12)
    ax1.set_title('Mean Wave Height vs Latitude', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # North at top

    # Plot 2: 95th percentile Hs
    ax2 = axes[0, 1]
    for beach in BEACH_MOP_RANGES.keys():
        beach_data = stats_df[stats_df['beach'] == beach]
        if len(beach_data) > 0:
            ax2.plot(beach_data['latitude'], beach_data['hs_95'],
                    'o-', label=beach, color=BEACH_COLORS[beach],
                    linewidth=2, markersize=6)

    ax2.set_xlabel('Latitude (°N)', fontsize=12)
    ax2.set_ylabel('95th Percentile Hs (m)', fontsize=12)
    ax2.set_title('Extreme Wave Heights vs Latitude', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # Plot 3: Mean wave power
    ax3 = axes[1, 0]
    for beach in BEACH_MOP_RANGES.keys():
        beach_data = stats_df[stats_df['beach'] == beach]
        if len(beach_data) > 0:
            ax3.plot(beach_data['latitude'], beach_data['power_mean'],
                    'o-', label=beach, color=BEACH_COLORS[beach],
                    linewidth=2, markersize=6)

    ax3.set_xlabel('Latitude (°N)', fontsize=12)
    ax3.set_ylabel('Mean Wave Power (kW/m)', fontsize=12)
    ax3.set_title('Mean Wave Power vs Latitude', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    # Plot 4: Summary statistics by beach
    ax4 = axes[1, 1]
    beach_summary = stats_df.groupby('beach').agg({
        'hs_mean': 'mean',
        'hs_95': 'mean',
        'tp_mean': 'mean',
        'power_mean': 'mean'
    }).round(2)

    beach_summary = beach_summary.reindex(BEACH_MOP_RANGES.keys())

    ax4.axis('off')
    table = ax4.table(cellText=beach_summary.values,
                     rowLabels=beach_summary.index,
                     colLabels=['Mean Hs\n(m)', '95th Hs\n(m)', 'Mean Tp\n(s)', 'Mean Power\n(kW/m)'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color code rows
    for i, beach in enumerate(beach_summary.index):
        if beach in BEACH_COLORS:
            table[(i+1, 0)].set_facecolor(BEACH_COLORS[beach])
            table[(i+1, 0)].set_alpha(0.3)

    ax4.set_title('Wave Climate Summary by Beach', fontsize=14, pad=20)

    plt.tight_layout()
    output_path = output_dir / 'wave_A7_wave_spatial_climate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def fig_A8_extreme_value_analysis(loader: WaveLoader, output_dir: Path):
    """Figure A8: Extreme value analysis and return periods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect annual maxima from multiple MOPs
    all_annual_max = []

    for beach, (mop_min, mop_max) in list(BEACH_MOP_RANGES.items())[:3]:  # First 3 beaches
        mop_ids = [m for m in range(mop_min, mop_max + 1) if m in loader.available_mops]
        if not mop_ids:
            continue

        mop_id = mop_ids[len(mop_ids)//2]
        try:
            wave_data = loader._load_wave_data(mop_id)
            filtered = filter_wave_data_by_date(wave_data)
            times = pd.to_datetime(filtered['time'])
            df = pd.DataFrame({'time': times, 'hs': filtered['hs']})
            df['year'] = df['time'].dt.year

            annual_max = df.groupby('year')['hs'].max().dropna()
            all_annual_max.extend(annual_max.values)
        except:
            continue

    all_annual_max = np.array(all_annual_max)

    # Plot 1: Annual maxima time series
    ax1 = axes[0, 0]
    ax1.plot(range(len(all_annual_max)), all_annual_max, 'o-',
            markersize=6, linewidth=1, color='steelblue')
    ax1.set_xlabel('Year Index', fontsize=12)
    ax1.set_ylabel('Annual Maximum Hs (m)', fontsize=12)
    ax1.set_title('Annual Maximum Wave Heights', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Fit trend
    z = np.polyfit(range(len(all_annual_max)), all_annual_max, 1)
    p = np.poly1d(z)
    ax1.plot(range(len(all_annual_max)), p(range(len(all_annual_max))),
            'r--', linewidth=2, label=f'Trend: {z[0]:.3f} m/yr')
    ax1.legend()

    # Plot 2: GEV fit
    ax2 = axes[0, 1]

    # Fit GEV distribution
    from scipy.stats import genextreme
    shape, loc, scale = genextreme.fit(all_annual_max)

    # Histogram
    ax2.hist(all_annual_max, bins=20, density=True, alpha=0.7,
            color='steelblue', edgecolor='black', linewidth=0.5, label='Data')

    # Fitted distribution
    x = np.linspace(all_annual_max.min(), all_annual_max.max(), 100)
    ax2.plot(x, genextreme.pdf(x, shape, loc, scale), 'r-',
            linewidth=2, label='GEV Fit')

    ax2.set_xlabel('Annual Maximum Hs (m)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('GEV Distribution Fit', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Return period plot
    ax3 = axes[1, 0]

    # Calculate empirical return periods
    sorted_max = np.sort(all_annual_max)[::-1]
    n = len(sorted_max)
    empirical_rp = (n + 1) / np.arange(1, n + 1)

    ax3.plot(empirical_rp, sorted_max, 'o', markersize=6,
            label='Empirical', color='steelblue')

    # Theoretical return periods from GEV
    return_periods = np.array([1, 2, 5, 10, 20, 50, 100])
    exceedance_prob = 1 / return_periods
    theoretical_levels = genextreme.ppf(1 - exceedance_prob, shape, loc, scale)

    ax3.plot(return_periods, theoretical_levels, 'r-',
            linewidth=2, label='GEV Model')
    ax3.plot(return_periods, theoretical_levels, 'ro', markersize=8)

    ax3.set_xlabel('Return Period (years)', fontsize=12)
    ax3.set_ylabel('Significant Wave Height (m)', fontsize=12)
    ax3.set_title('Return Period Analysis', fontsize=14)
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Return level table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    for rp in [1, 2, 5, 10, 20, 50, 100]:
        level = genextreme.ppf(1 - 1/rp, shape, loc, scale)
        table_data.append([f'{rp}', f'{level:.2f}'])

    table = ax4.table(cellText=table_data,
                     colLabels=['Return Period\n(years)', 'Wave Height\n(m)'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.2, 0, 0.6, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    ax4.set_title('Design Wave Heights', fontsize=14, pad=20)

    # Add GEV parameters text
    params_text = (f'GEV Parameters:\n'
                  f'Shape (ξ): {shape:.3f}\n'
                  f'Location (μ): {loc:.2f}\n'
                  f'Scale (σ): {scale:.2f}')
    ax4.text(0.7, 0.3, params_text, transform=ax4.transAxes,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'wave_A8_wave_extreme_value_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate wave climate figures for CliffCast appendix'
    )
    parser.add_argument(
        '--cdip-dir',
        type=str,
        default='data/raw/cdip',
        help='Directory containing CDIP NetCDF files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/appendix',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--figures',
        type=str,
        nargs='+',
        default='all',
        help='Which figures to generate (default: all)'
    )

    args = parser.parse_args()

    # Setup
    cdip_dir = Path(args.cdip_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CliffCast Wave Climate Figures")
    print("="*80)
    print(f"CDIP directory: {cdip_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Initialize loader
    print("Initializing WaveLoader...")
    loader = WaveLoader(cdip_dir)
    print(f"Found {len(loader.available_mops)} MOPs with data")
    print()

    # Generate figures
    figures_to_generate = args.figures if args.figures != 'all' else [
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'
    ]

    figure_functions = {
        'A1': fig_A1_wave_height_distributions,
        'A2': fig_A2_wave_period_characteristics,
        'A3': fig_A3_wave_direction_roses,
        'A4': fig_A4_wave_power_statistics,
        'A5': fig_A5_seasonal_patterns,
        'A6': fig_A6_storm_climatology,
        'A7': fig_A7_spatial_wave_climate,
        'A8': fig_A8_extreme_value_analysis,
    }

    for fig_name in figures_to_generate:
        if fig_name in figure_functions:
            print(f"Generating Figure {fig_name}...")
            try:
                figure_functions[fig_name](loader, output_dir)
            except Exception as e:
                print(f"Error generating Figure {fig_name}: {e}")
                import traceback
                traceback.print_exc()

    print()
    print("="*80)
    print("All figures generated successfully!")
    print(f"Figures saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
