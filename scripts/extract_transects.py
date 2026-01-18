#!/usr/bin/env python3
"""Extract transects from LiDAR point cloud files.

This script processes LAS/LAZ files and extracts shore-normal transects.

Usage:
    python scripts/extract_transects.py \
        --input data/raw/lidar/cliff_scan.laz \
        --output data/processed/transects.npz \
        --n-points 128 \
        --spacing 10.0 \
        --coastline data/raw/coastline.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.transect_extractor import TransectExtractor
from src.utils.logging import setup_logger

logger = setup_logger(__name__, level="INFO")


def load_coastline(coastline_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load coastline points and normals from file.

    Expected CSV format:
        x,y,normal_x,normal_y
        100.0,200.0,1.0,0.0
        100.0,210.0,1.0,0.0
        ...

    Args:
        coastline_path: Path to coastline CSV file

    Returns:
        Tuple of (coastline_points, coastline_normals)
    """
    data = np.loadtxt(coastline_path, delimiter=',', skiprows=1)

    coastline_points = data[:, :2]  # x, y
    coastline_normals = data[:, 2:4]  # normal_x, normal_y

    # Normalize normals
    norms = np.linalg.norm(coastline_normals, axis=1, keepdims=True)
    coastline_normals = coastline_normals / (norms + 1e-8)

    logger.info(f"Loaded {len(coastline_points)} coastline points")

    return coastline_points, coastline_normals


def main():
    """Main extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract shore-normal transects from LiDAR point clouds"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input LAS/LAZ file path",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NPZ file path for transects",
    )

    parser.add_argument(
        "--coastline",
        type=Path,
        default=None,
        help="Optional CSV file with coastline points and normals (x,y,normal_x,normal_y)",
    )

    parser.add_argument(
        "--n-points",
        type=int,
        default=128,
        help="Number of points per transect (default: 128)",
    )

    parser.add_argument(
        "--spacing",
        type=float,
        default=10.0,
        help="Alongshore spacing between transects in meters (default: 10.0)",
    )

    parser.add_argument(
        "--profile-length",
        type=float,
        default=150.0,
        help="Maximum profile length from toe in meters (default: 150.0)",
    )

    parser.add_argument(
        "--search-radius",
        type=float,
        default=2.0,
        help="Search radius for points near transect line (default: 2.0)",
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=20,
        help="Minimum points required for valid transect (default: 20)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of extracted transects",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Initialize extractor
    logger.info("Initializing TransectExtractor...")
    extractor = TransectExtractor(
        n_points=args.n_points,
        spacing_m=args.spacing,
        profile_length_m=args.profile_length,
        min_points=args.min_points,
        search_radius_m=args.search_radius,
    )

    # Load coastline if provided
    coastline_points = None
    coastline_normals = None

    if args.coastline is not None:
        if args.coastline.exists():
            coastline_points, coastline_normals = load_coastline(args.coastline)
        else:
            logger.warning(f"Coastline file not found: {args.coastline}")
            logger.info("Will attempt automatic coastline detection")

    # Extract transects
    logger.info(f"Extracting transects from {args.input}")
    try:
        transects = extractor.extract_from_file(
            args.input,
            coastline_points=coastline_points,
            coastline_normals=coastline_normals,
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Check if any transects were extracted
    n_transects = len(transects['points'])
    if n_transects == 0:
        logger.error("No transects extracted! Check input data and parameters.")
        return 1

    logger.info(f"Successfully extracted {n_transects} transects")

    # Print summary statistics
    logger.info("\nTransect Summary:")
    logger.info(f"  Number of transects: {n_transects}")
    logger.info(f"  Points per transect: {transects['points'].shape[1]}")
    logger.info(f"  Features per point: {transects['points'].shape[2]}")

    # Metadata statistics
    cliff_heights = transects['metadata'][:, 0]
    logger.info(f"  Cliff height range: {cliff_heights.min():.1f} - {cliff_heights.max():.1f} m")
    logger.info(f"  Mean cliff height: {cliff_heights.mean():.1f} m")

    mean_slopes = transects['metadata'][:, 1]
    logger.info(f"  Mean slope range: {mean_slopes.min():.1f} - {mean_slopes.max():.1f} degrees")

    # Save transects
    logger.info(f"\nSaving transects to {args.output}")
    extractor.save_transects(transects, args.output, format='npz')

    # Visualize if requested
    if args.visualize:
        logger.info("Generating visualizations...")
        try:
            visualize_transects(transects, args.output.parent / "transects_viz.png")
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    logger.info("Extraction complete!")
    return 0


def visualize_transects(transects: dict, output_path: Path):
    """Create visualization of extracted transects.

    Args:
        transects: Dictionary from TransectExtractor
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt

    n_transects = len(transects['points'])

    # Plot first 10 transects
    n_plot = min(10, n_transects)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Transect profiles
    ax = axes[0]
    for i in range(n_plot):
        distances = transects['distances'][i]
        elevations = transects['points'][i, :, 1]  # Feature 1 is elevation
        ax.plot(distances, elevations, alpha=0.5, label=f"Transect {i}")

    ax.set_xlabel("Distance from Toe (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Extracted Transect Profiles (first {n_plot} of {n_transects})")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Plot 2: Feature distributions
    ax = axes[1]
    feature_names = ['Distance', 'Elevation', 'Slope', 'Curvature', 'Roughness']

    # Plot histograms of slope values
    all_slopes = transects['points'][:, :, 2].flatten()
    ax.hist(all_slopes, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Slope (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Slope Values Across All Transects")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
