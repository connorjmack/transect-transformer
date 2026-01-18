#!/usr/bin/env python3
"""Visualize multiple transects in a grid.

Usage:
    python scripts/visualize_multiple_transects.py \\
        --input results/mops_transects/transects_voxelized.npz \\
        --n 6  # number of transects to show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize multiple transects in a grid"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/mops_transects/transects_voxelized.npz",
        help="Path to extracted transects .npz file"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6,
        help="Number of transects to visualize (default: 6)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["best", "random", "first"],
        default="best",
        help="Which transects to show (default: best = most valid bins)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save figure to this path (default: show interactive)"
    )
    return parser.parse_args()


def plot_transect_profile(ax, bin_features, bin_centers, bin_mask, name):
    """Plot a single transect elevation profile on an axis.

    Args:
        ax: Matplotlib axis
        bin_features: (n_bins, 6) features
        bin_centers: (n_bins,) distances
        bin_mask: (n_bins,) valid mask
        name: Transect name
    """
    # Extract elevation and roughness
    mean_elev = bin_features[:, 0]
    roughness = bin_features[:, 1]

    # Only plot valid bins
    valid_centers = bin_centers[bin_mask]
    valid_elev = mean_elev[bin_mask]
    valid_roughness = roughness[bin_mask]

    if len(valid_centers) > 0:
        # Plot with error bars
        ax.errorbar(
            valid_centers, valid_elev, yerr=valid_roughness,
            fmt='o-', capsize=3, alpha=0.7, markersize=4, linewidth=1.5
        )
        ax.set_ylabel('Elevation (m)', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='red')

    ax.set_title(f'{name} ({bin_mask.sum()}/128 bins)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Distance from origin (m)', fontsize=9)


def main():
    """Main execution."""
    args = parse_args()

    # Load transects
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return

    data = np.load(input_path)

    bin_features = data['bin_features']
    bin_centers = data['bin_centers']
    bin_mask = data['bin_mask']
    names = data['names'] if 'names' in data else [f"T{i}" for i in range(len(bin_features))]

    n_transects = len(bin_features)
    valid_bins = bin_mask.sum(axis=1)

    print(f"Loaded {n_transects} transects")
    print(f"Valid bins per transect: min={valid_bins.min()}, max={valid_bins.max()}, mean={valid_bins.mean():.1f}")

    # Select which transects to show
    n = min(args.n, n_transects)

    if args.mode == "best":
        # Show transects with most valid bins
        indices = np.argsort(valid_bins)[::-1][:n]
        mode_str = "Best (most valid bins)"
    elif args.mode == "random":
        # Random selection
        indices = np.random.choice(n_transects, size=n, replace=False)
        mode_str = "Random selection"
    else:  # first
        indices = np.arange(n)
        mode_str = "First N transects"

    print(f"\nShowing {n} transects ({mode_str})")

    # Create grid of subplots
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Plot each selected transect
    for i, idx in enumerate(indices):
        ax = axes_flat[i]
        plot_transect_profile(
            ax,
            bin_features=bin_features[idx],
            bin_centers=bin_centers[idx],
            bin_mask=bin_mask[idx],
            name=names[idx]
        )

    # Hide unused subplots
    for i in range(len(indices), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle(
        f'Voxelized Transect Profiles - {mode_str}',
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
