"""Quick wave data summary and basic visualizations.

Simpler script for quick wave data exploration.

Usage:
    python scripts/visualization/quick_wave_summary.py --cdip-dir data/raw/cdip/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.wave_loader import WaveLoader

# Beach ranges
BEACH_MOP_RANGES = {
    'Blacks': (520, 567),
    'Torrey': (567, 581),
    'Del Mar': (595, 620),
    'Solana': (637, 666),
    'San Elijo': (683, 708),
    'Encinitas': (708, 764),
}

# Date range for analysis (aligned with LiDAR data)
START_DATE = pd.Timestamp('2017-01-01')
END_DATE = pd.Timestamp('2025-12-31')


def filter_wave_data_by_date(wave_data):
    """Filter wave data to 2017-2025 period."""
    times = pd.to_datetime(wave_data.time)
    mask = (times >= START_DATE) & (times <= END_DATE)

    return {
        'hs': wave_data.hs[mask],
        'tp': wave_data.tp[mask],
        'power': wave_data.power[mask],
        'time': wave_data.time[mask],
    }


def main():
    parser = argparse.ArgumentParser(description='Quick wave data summary')
    parser.add_argument('--cdip-dir', type=str, default='data/raw/cdip',
                       help='CDIP data directory')
    parser.add_argument('--output', type=str, default='figures/appendix',
                       help='Output directory')
    args = parser.parse_args()

    # Setup
    cdip_dir = Path(args.cdip_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Wave Data Summary")
    print("="*60)

    # Initialize loader
    loader = WaveLoader(cdip_dir)
    print(f"\nFound {len(loader.available_mops)} MOPs with data")

    # Get summary statistics
    print("\nGenerating summary statistics...")
    summary = loader.summary()

    # Print by beach
    for beach, (mop_min, mop_max) in BEACH_MOP_RANGES.items():
        beach_mops = [m for m in loader.available_mops
                     if mop_min <= m <= mop_max]
        if beach_mops:
            print(f"\n{beach}: {len(beach_mops)} MOPs ({min(beach_mops)}-{max(beach_mops)})")

            # Sample statistics from first MOP
            mop_id = beach_mops[0]
            if mop_id in summary and 'error' not in summary[mop_id]:
                stats = summary[mop_id]
                print(f"  Sample MOP {mop_id}:")
                print(f"    Records: {stats['n_records']:,}")
                print(f"    Date range: {stats['date_range'][0].date()} to {stats['date_range'][1].date()}")
                print(f"    Mean Hs: {stats['hs_mean']:.2f} m")
                print(f"    Max Hs: {stats['hs_max']:.2f} m")
                print(f"    Data quality: {stats['data_quality']*100:.1f}%")

    # Create simple overview figure
    print("\nGenerating overview figure...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Load sample data from each beach
    beach_data = {}
    for beach, (mop_min, mop_max) in BEACH_MOP_RANGES.items():
        beach_mops = [m for m in loader.available_mops
                     if mop_min <= m <= mop_max]
        if beach_mops:
            mop_id = beach_mops[len(beach_mops)//2]  # Middle MOP
            try:
                wave_data = loader._load_wave_data(mop_id)
                filtered = filter_wave_data_by_date(wave_data)
                beach_data[beach] = {
                    'hs': filtered['hs'][~np.isnan(filtered['hs'])],
                    'tp': filtered['tp'][~np.isnan(filtered['tp'])],
                    'power': filtered['power'][~np.isnan(filtered['power'])],
                }
            except:
                continue

    # Plot 1: Hs distributions
    ax1 = axes[0, 0]
    for beach, data in beach_data.items():
        ax1.hist(data['hs'], bins=50, alpha=0.5, label=beach, density=True)
    ax1.set_xlabel('Significant Wave Height (m)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Wave Height Distributions by Beach')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Hs vs Tp scatter
    ax2 = axes[0, 1]
    for beach, data in beach_data.items():
        # Subsample
        n = min(5000, len(data['hs']))
        idx = np.random.choice(len(data['hs']), n, replace=False)
        ax2.scatter(data['tp'][idx], data['hs'][idx],
                   alpha=0.3, s=1, label=beach)
    ax2.set_xlabel('Peak Period (s)')
    ax2.set_ylabel('Significant Wave Height (m)')
    ax2.set_title('Wave Height vs Period')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box plots of Hs
    ax3 = axes[1, 0]
    ax3.boxplot([data['hs'] for data in beach_data.values()],
                labels=list(beach_data.keys()),
                showfliers=False)
    ax3.set_ylabel('Significant Wave Height (m)')
    ax3.set_title('Wave Height by Beach')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Wave power distributions
    ax4 = axes[1, 1]
    for beach, data in beach_data.items():
        ax4.hist(data['power'], bins=50, alpha=0.5,
                label=beach, density=True, range=(0, 100))
    ax4.set_xlabel('Wave Power (kW/m)')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Wave Power Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'wave_summary_overview_2017-2025.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")
    plt.close()

    print("\n" + "="*60)
    print("Summary complete!")
    print("="*60)


if __name__ == "__main__":
    main()
