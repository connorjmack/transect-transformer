#!/usr/bin/env python3
"""Visualize a sample voxelized transect.

Quick visualization script to inspect extracted transects.

Usage:
    python scripts/visualize_sample_transect.py \\
        --input results/mops_transects/transects_voxelized.npz \\
        --transect 0  # or any transect index
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize a voxelized transect"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/mops_transects/transects_voxelized.npz",
        help="Path to extracted transects .npz file"
    )
    parser.add_argument(
        "--transect",
        type=int,
        default=0,
        help="Index of transect to visualize"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save figure to this path (default: show interactive)"
    )
    return parser.parse_args()


def plot_voxelized_transect(
    bin_features: np.ndarray,
    bin_centers: np.ndarray,
    bin_mask: np.ndarray,
    metadata: np.ndarray,
    name: str,
    output_path: Path = None,
):
    """Plot a voxelized transect profile.

    Args:
        bin_features: (n_bins, 6) voxelized features
        bin_centers: (n_bins,) distances from origin
        bin_mask: (n_bins,) valid bin mask
        metadata: (7,) transect metadata
        name: Transect name
        output_path: Optional path to save figure
    """
    # Extract features
    mean_elev = bin_features[:, 0]
    roughness = bin_features[:, 1]
    height_range = bin_features[:, 2]
    slope = bin_features[:, 3]
    curvature = bin_features[:, 4]
    point_density = bin_features[:, 5]

    # Only use valid bins for plotting
    valid_centers = bin_centers[bin_mask]
    valid_elev = mean_elev[bin_mask]
    valid_roughness = roughness[bin_mask]
    valid_slope = slope[bin_mask]
    valid_density = point_density[bin_mask]

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Elevation profile with error bars (roughness)
    ax = axes[0]
    ax.errorbar(
        valid_centers, valid_elev, yerr=valid_roughness,
        fmt='o-', capsize=3, alpha=0.7, label='Mean elevation ± roughness'
    )
    ax.axhline(metadata[3], color='gray', linestyle='--', alpha=0.5, label='Toe elevation')
    ax.set_ylabel('Elevation (m)', fontsize=11)
    ax.set_title(f'{name} - Voxelized Transect Profile', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Slope
    ax = axes[1]
    ax.plot(valid_centers, valid_slope, 'o-', color='tab:orange', alpha=0.7)
    ax.axhline(metadata[1], color='gray', linestyle='--', alpha=0.5,
               label=f'Mean slope: {metadata[1]:.1f}°')
    ax.set_ylabel('Slope (°)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Point density
    ax = axes[2]
    ax.plot(valid_centers, valid_density, 'o-', color='tab:green', alpha=0.7)
    ax.set_ylabel('Point density\n(pts/m³)', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: Valid bin mask
    ax = axes[3]
    ax.bar(bin_centers, bin_mask, width=np.diff(bin_centers)[0] if len(bin_centers) > 1 else 1.0,
           color='tab:blue', alpha=0.5, edgecolor='none')
    ax.set_ylabel('Valid bins', fontsize=11)
    ax.set_xlabel('Distance from origin (m)', fontsize=11)
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3, axis='x')

    # Add metadata text
    info_text = (
        f"Cliff height: {metadata[0]:.1f}m\n"
        f"Mean slope: {metadata[1]:.1f}°\n"
        f"Max slope: {metadata[2]:.1f}°\n"
        f"Valid bins: {bin_mask.sum()}/{len(bin_mask)}"
    )
    fig.text(0.98, 0.98, info_text, transform=fig.transFigure,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def main():
    """Main execution."""
    args = parse_args()

    # Load transects
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        print("\nPlease run extract_mops_transects.py first to generate transects.")
        return

    data = np.load(input_path)

    bin_features = data['bin_features']
    bin_centers = data['bin_centers']
    bin_mask = data['bin_mask']
    metadata = data['metadata']
    names = data['names'] if 'names' in data else None

    n_transects = len(bin_features)

    if args.transect >= n_transects:
        print(f"Error: Transect index {args.transect} out of range [0, {n_transects-1}]")
        return

    # Get transect name
    if names is not None:
        name = names[args.transect]
    else:
        name = f"Transect {args.transect}"

    print(f"Visualizing {name}")
    print(f"Total transects in file: {n_transects}")

    # Plot
    output_path = Path(args.output) if args.output is not None else None

    plot_voxelized_transect(
        bin_features=bin_features[args.transect],
        bin_centers=bin_centers[args.transect],
        bin_mask=bin_mask[args.transect],
        metadata=metadata[args.transect],
        name=name,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
