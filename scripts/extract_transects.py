#!/usr/bin/env python3
"""Extract transects from LiDAR point clouds using predefined transect lines.

This script processes LAS/LAZ files and extracts point data along transect lines
defined in a shapefile. For each transect, points within a buffer distance are
extracted, projected onto the transect line, and resampled to a fixed number of points.

The output includes all available LAS attributes (elevation, intensity, RGB, classification)
along with computed geometric features (slope, curvature, roughness).

Usage:
    python scripts/extract_transects.py \
        --transects data/mops/transects_10m/transect_lines.shp \
        --las-dir data/testing/ \
        --output data/processed/transects.npz \
        --buffer 1.0 \
        --n-points 128
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.shapefile_transect_extractor import ShapefileTransectExtractor
from src.utils.logging import setup_logger

logger = setup_logger(__name__, level="INFO")


def visualize_transects(transects: Dict, output_path: Path, n_samples: int = 10):
    """Create visualization of extracted transects.

    Args:
        transects: Dictionary from extractor
        output_path: Path to save visualization
        n_samples: Number of sample transects to plot
    """
    import matplotlib.pyplot as plt

    n_transects = len(transects['points'])
    n_plot = min(n_samples, n_transects)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Sample random transects
    sample_idx = np.random.choice(n_transects, n_plot, replace=False)
    sample_idx = np.sort(sample_idx)

    # Plot 1: Elevation profiles
    ax = axes[0, 0]
    for i in sample_idx:
        distances = transects['distances'][i]
        elevations = transects['points'][i, :, 1]  # Feature 1 is elevation
        ax.plot(distances, elevations, alpha=0.7, linewidth=1)
    ax.set_xlabel("Distance from Start (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Elevation Profiles ({n_plot} samples)")
    ax.grid(True, alpha=0.3)

    # Plot 2: Slope profiles
    ax = axes[0, 1]
    for i in sample_idx:
        distances = transects['distances'][i]
        slopes = transects['points'][i, :, 2]  # Feature 2 is slope
        ax.plot(distances, slopes, alpha=0.7, linewidth=1)
    ax.set_xlabel("Distance from Start (m)")
    ax.set_ylabel("Slope (degrees)")
    ax.set_title(f"Slope Profiles ({n_plot} samples)")
    ax.grid(True, alpha=0.3)

    # Plot 3: Intensity profiles
    ax = axes[1, 0]
    for i in sample_idx:
        distances = transects['distances'][i]
        intensity = transects['points'][i, :, 5]  # Feature 5 is intensity
        ax.plot(distances, intensity, alpha=0.7, linewidth=1)
    ax.set_xlabel("Distance from Start (m)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_title(f"Intensity Profiles ({n_plot} samples)")
    ax.grid(True, alpha=0.3)

    # Plot 4: Cliff height distribution
    ax = axes[1, 1]
    cliff_heights = transects['metadata'][:, 0]
    ax.hist(cliff_heights, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Cliff Height (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Cliff Height Distribution (N={n_transects})")
    ax.grid(True, alpha=0.3)

    # Plot 5: Mean slope distribution
    ax = axes[2, 0]
    mean_slopes = transects['metadata'][:, 1]
    ax.hist(mean_slopes, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Mean Slope (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Mean Slope Distribution")
    ax.grid(True, alpha=0.3)

    # Plot 6: Spatial distribution of transects
    ax = axes[2, 1]
    x_coords = transects['metadata'][:, 8]  # longitude/X
    y_coords = transects['metadata'][:, 7]  # latitude/Y
    colors = transects['metadata'][:, 0]    # cliff height for color
    sc = ax.scatter(x_coords, y_coords, c=colors, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Cliff Height (m)')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Transect Locations")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path}")


def main():
    """Main extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract transects from LAS files using predefined transect lines"
    )

    parser.add_argument(
        "--transects",
        type=Path,
        required=True,
        help="Path to shapefile with transect LineStrings",
    )

    parser.add_argument(
        "--las-dir",
        type=Path,
        default=None,
        help="Directory containing LAS/LAZ files",
    )

    parser.add_argument(
        "--las-files",
        type=Path,
        nargs='+',
        default=None,
        help="Specific LAS/LAZ file(s) to process",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NPZ file path",
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=1.0,
        help="Buffer distance around transect line in meters (default: 1.0)",
    )

    parser.add_argument(
        "--n-points",
        type=int,
        default=128,
        help="Number of points per transect (default: 128)",
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=20,
        help="Minimum raw points required for valid transect (default: 20)",
    )

    parser.add_argument(
        "--transect-id-col",
        type=str,
        default="tr_id",
        help="Column name for transect IDs in shapefile (default: tr_id)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of extracted transects",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.las",
        help="Glob pattern for LAS files when using --las-dir (default: *.las)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.transects.exists():
        logger.error(f"Transect shapefile not found: {args.transects}")
        return 1

    # Collect LAS files
    las_files = []
    if args.las_files:
        las_files.extend(args.las_files)
    if args.las_dir:
        las_files.extend(sorted(args.las_dir.glob(args.pattern)))

    if not las_files:
        logger.error("No LAS files specified. Use --las-dir or --las-files")
        return 1

    # Check all files exist
    for f in las_files:
        if not f.exists():
            logger.error(f"LAS file not found: {f}")
            return 1

    logger.info(f"Found {len(las_files)} LAS file(s) to process")

    # Initialize extractor
    extractor = ShapefileTransectExtractor(
        n_points=args.n_points,
        buffer_m=args.buffer,
        min_points=args.min_points,
    )

    # Load transect lines
    transect_gdf = extractor.load_transect_lines(args.transects)

    # Extract transects
    try:
        transects = extractor.extract_from_shapefile_and_las(
            transect_gdf,
            las_files,
            transect_id_col=args.transect_id_col,
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Check results
    n_extracted = len(transects['points'])
    if n_extracted == 0:
        logger.error("No transects extracted! Check that LAS files overlap with transect lines.")
        return 1

    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Total transects extracted: {n_extracted}")
    logger.info(f"  Points per transect: {args.n_points}")
    logger.info(f"  Features per point: {extractor.N_FEATURES}")
    logger.info(f"  Metadata fields: {extractor.N_METADATA}")

    # Print feature/metadata names
    logger.info(f"\n  Feature names: {transects['feature_names']}")
    logger.info(f"  Metadata names: {transects['metadata_names']}")

    # Statistics
    cliff_heights = transects['metadata'][:, 0]
    logger.info(f"\n  Cliff height range: {cliff_heights.min():.1f} - {cliff_heights.max():.1f} m")
    logger.info(f"  Mean cliff height: {cliff_heights.mean():.1f} m")

    mean_slopes = transects['metadata'][:, 1]
    logger.info(f"  Mean slope range: {mean_slopes.min():.1f} - {mean_slopes.max():.1f} degrees")

    # Save
    extractor.save_transects(transects, args.output)

    # Visualize if requested
    if args.visualize:
        viz_path = args.output.parent / f"{args.output.stem}_viz.png"
        try:
            visualize_transects(transects, viz_path)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    logger.info(f"\nExtraction complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
