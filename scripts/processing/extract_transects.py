#!/usr/bin/env python3
"""Extract transects from LiDAR point clouds using predefined transect lines.

This script processes LAS/LAZ files and extracts point data along transect lines
defined in a shapefile. For each transect, points within a buffer distance are
extracted, projected onto the transect line, and resampled to a fixed number of points.

The output is saved in CUBE FORMAT for spatio-temporal modeling:
  - points: (n_transects, T, N, 12) - point features across all epochs
  - distances: (n_transects, T, N) - distance along transect
  - metadata: (n_transects, T, 12) - per-epoch metadata
  - timestamps: (n_transects, T) - scan dates for temporal encoding
  - transect_ids: (n_transects,) - unique transect IDs
  - epoch_names: (T,) - LAS filenames for each epoch

Current per-point features (12):
  distance_m, elevation_m, slope_deg, curvature, roughness,
  intensity, red, green, blue, classification, return_number, num_returns

TODO: Future enhancement - add per-point normal vectors (nx, ny, nz) computed from
      the local point neighborhood. This will enable better characterization of
      cliff face orientation and overhang detection.

Usage:
    python scripts/processing/extract_transects.py \\
        --transects data/mops/transects_10m/transect_lines.shp \\
        --las-dir data/testing/ \\
        --output data/processed/transects_cube.npz \\
        --buffer 1.0 \\
        --n-points 128 \\
        --visualize
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.shapefile_transect_extractor import ShapefileTransectExtractor
from src.utils.logging import setup_logger

logger = setup_logger(__name__, level="INFO")


def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """Parse date from LAS filename.

    Supports formats like:
    - 20171106_00590_00622_NoWaves_DelMar_beach_cliff_ground_cropped.las
    - 2017-11-06_scan.las
    - scan_20171106.las

    Args:
        filename: LAS filename (with or without path)

    Returns:
        datetime object or None if parsing fails
    """
    # Extract just the filename without path
    name = Path(filename).stem

    # Try YYYYMMDD at start of filename
    match = re.match(r'^(\d{8})', name)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d')
        except ValueError:
            pass

    # Try YYYY-MM-DD anywhere in filename
    match = re.search(r'(\d{4}-\d{2}-\d{2})', name)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d')
        except ValueError:
            pass

    # Try YYYYMMDD anywhere in filename
    match = re.search(r'(\d{8})', name)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d')
        except ValueError:
            pass

    logger.warning(f"Could not parse date from filename: {filename}")
    return None


def convert_flat_to_cube(
    flat_transects: Dict[str, np.ndarray],
    n_points: int = 128,
) -> Dict[str, np.ndarray]:
    """Convert flat transect format to cube format for spatio-temporal modeling.

    Args:
        flat_transects: Dictionary with flat arrays from extractor:
            - points: (N_total, n_points, 12)
            - distances: (N_total, n_points)
            - metadata: (N_total, 12)
            - transect_ids: (N_total,)
            - las_sources: (N_total,) list of filenames
        n_points: Number of points per transect

    Returns:
        Dictionary with cube arrays:
            - points: (n_transects, T, n_points, 12)
            - distances: (n_transects, T, n_points)
            - metadata: (n_transects, T, 12)
            - timestamps: (n_transects, T) as ordinal days
            - transect_ids: (n_transects,)
            - epoch_names: (T,) sorted LAS filenames
            - feature_names: list of feature names
            - metadata_names: list of metadata names
    """
    points = flat_transects['points']
    distances = flat_transects['distances']
    metadata = flat_transects['metadata']
    transect_ids = flat_transects['transect_ids']
    las_sources = flat_transects['las_sources']

    # Get unique transect IDs and epochs
    unique_transects = np.unique(transect_ids)
    unique_epochs = np.unique(las_sources)

    # Parse dates and sort epochs chronologically
    epoch_dates = []
    for epoch in unique_epochs:
        date = parse_date_from_filename(epoch)
        if date is None:
            # Use filename hash as fallback for sorting
            date = datetime(1970, 1, 1)
            logger.warning(f"Using fallback date for {epoch}")
        epoch_dates.append((epoch, date))

    # Sort by date
    epoch_dates.sort(key=lambda x: x[1])
    sorted_epochs = [e[0] for e in epoch_dates]
    sorted_dates = [e[1] for e in epoch_dates]

    n_transects = len(unique_transects)
    n_epochs = len(sorted_epochs)
    n_features = points.shape[2]
    n_meta = metadata.shape[1]

    logger.info(f"Converting to cube format:")
    logger.info(f"  Unique transects: {n_transects}")
    logger.info(f"  Unique epochs: {n_epochs}")
    logger.info(f"  Epochs (sorted): {sorted_epochs}")

    # Create epoch index mapping
    epoch_to_idx = {epoch: idx for idx, epoch in enumerate(sorted_epochs)}

    # Create transect index mapping
    transect_to_idx = {tid: idx for idx, tid in enumerate(unique_transects)}

    # Initialize cube arrays with NaN (to detect missing data)
    points_cube = np.full((n_transects, n_epochs, n_points, n_features), np.nan, dtype=np.float32)
    distances_cube = np.full((n_transects, n_epochs, n_points), np.nan, dtype=np.float32)
    metadata_cube = np.full((n_transects, n_epochs, n_meta), np.nan, dtype=np.float32)
    timestamps = np.zeros((n_transects, n_epochs), dtype=np.int64)

    # Fill timestamps with ordinal days (consistent across all transects)
    for t_idx, date in enumerate(sorted_dates):
        timestamps[:, t_idx] = date.toordinal()

    # Fill cube arrays
    for i in range(len(points)):
        tid = transect_ids[i]
        epoch = las_sources[i]

        t_idx = transect_to_idx[tid]
        e_idx = epoch_to_idx[epoch]

        points_cube[t_idx, e_idx] = points[i]
        distances_cube[t_idx, e_idx] = distances[i]
        metadata_cube[t_idx, e_idx] = metadata[i]

    # Check for missing data
    missing_count = np.isnan(points_cube[:, :, 0, 0]).sum()
    total_cells = n_transects * n_epochs
    if missing_count > 0:
        missing_pct = 100 * missing_count / total_cells
        logger.warning(f"Missing data: {missing_count}/{total_cells} ({missing_pct:.1f}%) transect-epoch pairs")
        logger.warning("Consider checking that all LAS files cover the same transects")
    else:
        logger.info(f"Full coverage: all {n_transects} transects present in all {n_epochs} epochs")

    return {
        'points': points_cube,
        'distances': distances_cube,
        'metadata': metadata_cube,
        'timestamps': timestamps,
        'transect_ids': unique_transects,
        'epoch_names': np.array(sorted_epochs, dtype=object),
        'epoch_dates': np.array([d.isoformat() for d in sorted_dates], dtype=object),
        'feature_names': flat_transects['feature_names'],
        'metadata_names': flat_transects['metadata_names'],
    }


def select_representative_transects(cube: Dict, n_samples: int = 4) -> np.ndarray:
    """Select representative transects spanning different cliff heights and locations.

    Selects transects from different quartiles of cliff height to show variety.

    Args:
        cube: Dictionary with cube format arrays
        n_samples: Number of transects to select

    Returns:
        Array of transect indices
    """
    metadata = cube['metadata']
    n_transects, n_epochs, _ = metadata.shape

    # Use latest epoch for selection criteria
    latest = n_epochs - 1
    cliff_heights = metadata[:, latest, 0]

    # Find transects with valid data
    valid_mask = ~np.isnan(cliff_heights)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < n_samples:
        return valid_indices

    # Sort by cliff height and select from different quartiles
    valid_heights = cliff_heights[valid_indices]
    sorted_order = np.argsort(valid_heights)
    sorted_indices = valid_indices[sorted_order]

    # Select evenly spaced samples across the height distribution
    step = len(sorted_indices) // n_samples
    selected = []
    for i in range(n_samples):
        idx = min(i * step + step // 2, len(sorted_indices) - 1)
        selected.append(sorted_indices[idx])

    return np.array(selected)


def visualize_cube_transects(cube: Dict, output_path: Path):
    """Create visualization of extracted transects in cube format.

    Shows temporal evolution of 4 representative transects across all epochs,
    plus summary panels for spatial distribution and data coverage.

    Args:
        cube: Dictionary with cube format arrays
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    import matplotlib.gridspec as gridspec

    points = cube['points']  # (n_transects, T, N, 12)
    distances = cube['distances']  # (n_transects, T, N)
    metadata = cube['metadata']  # (n_transects, T, 12)
    epoch_dates = cube['epoch_dates']
    transect_ids = cube['transect_ids']

    n_transects, n_epochs, n_points, n_features = points.shape

    # Select 4 representative transects
    sample_idx = select_representative_transects(cube, n_samples=4)

    # Create figure with custom layout:
    # Top 2 rows: 2x2 grid of transect temporal evolution
    # Bottom row: spatial map and data coverage
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.25)

    # Color map for epochs - use a perceptually uniform colormap
    cmap = get_cmap('plasma')
    epoch_colors = [cmap(i / max(1, n_epochs - 1)) for i in range(n_epochs)]

    # Create epoch labels (show year only for cleaner display)
    epoch_labels = [d[:4] for d in epoch_dates]

    # Plot 4 representative transects in 2x2 grid
    transect_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for ax_idx, (ax, tidx) in enumerate(zip(transect_axes, sample_idx)):
        tid = transect_ids[tidx]

        # Get cliff height for title
        latest = n_epochs - 1
        cliff_height = metadata[tidx, latest, 0]
        height_str = f"{cliff_height:.1f}m" if not np.isnan(cliff_height) else "N/A"

        # Plot each epoch
        for t in range(n_epochs):
            if not np.isnan(points[tidx, t, 0, 0]):
                dist = distances[tidx, t]
                elev = points[tidx, t, :, 1]  # Feature 1 is elevation
                ax.plot(dist, elev, color=epoch_colors[t], alpha=0.85,
                       linewidth=1.5, label=epoch_labels[t])

        ax.set_xlabel("Distance along transect (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(f"Transect {tid} (height: {height_str})", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add legend only to first plot (top-left) to avoid clutter
        if ax_idx == 0:
            # Create compact legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper right', fontsize=8,
                     title='Year', title_fontsize=8, ncol=2 if n_epochs > 5 else 1)

    # Bottom left: Spatial distribution with selected transects highlighted
    ax_map = fig.add_subplot(gs[2, 0])
    latest = n_epochs - 1
    x_coords = metadata[:, latest, 8]  # longitude/X is index 8
    y_coords = metadata[:, latest, 7]  # latitude/Y is index 7
    valid = ~np.isnan(x_coords)

    # Plot all transects in gray
    ax_map.scatter(x_coords[valid], y_coords[valid], c='lightgray', s=3, alpha=0.5)

    # Highlight selected transects
    colors_highlight = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # Colorblind-friendly
    for i, tidx in enumerate(sample_idx):
        if not np.isnan(x_coords[tidx]):
            ax_map.scatter(x_coords[tidx], y_coords[tidx], c=colors_highlight[i],
                          s=80, marker='o', edgecolors='black', linewidths=1.5,
                          label=f"Transect {transect_ids[tidx]}", zorder=10)

    ax_map.set_xlabel("Easting (m)")
    ax_map.set_ylabel("Northing (m)")
    ax_map.set_title("Transect Locations", fontsize=11, fontweight='bold')
    ax_map.legend(loc='best', fontsize=8)
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.3, linestyle='--')

    # Bottom right: Summary statistics text box
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis('off')

    # Calculate summary statistics
    latest_heights = metadata[:, latest, 0]
    valid_heights = latest_heights[~np.isnan(latest_heights)]
    coverage = ~np.isnan(points[:, :, 0, 0])
    coverage_pct = 100 * coverage.sum() / coverage.size

    # Build summary text
    summary_lines = [
        "EXTRACTION SUMMARY",
        "=" * 40,
        f"",
        f"Cube dimensions:",
        f"  • Transects: {n_transects:,}",
        f"  • Epochs: {n_epochs}",
        f"  • Points/transect: {n_points}",
        f"  • Features/point: {n_features}",
        f"",
        f"Temporal coverage:",
        f"  • First epoch: {epoch_dates[0][:10]}",
        f"  • Last epoch: {epoch_dates[-1][:10]}",
        f"  • Data coverage: {coverage_pct:.1f}%",
        f"",
        f"Cliff heights (latest epoch):",
        f"  • Min: {valid_heights.min():.1f} m",
        f"  • Max: {valid_heights.max():.1f} m",
        f"  • Mean: {valid_heights.mean():.1f} m",
        f"  • Std: {valid_heights.std():.1f} m",
    ]

    summary_text = "\n".join(summary_lines)
    ax_stats.text(0.05, 0.95, summary_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path}")


def save_cube(cube: Dict, output_path: Path) -> None:
    """Save cube format transects to NPZ file.

    Args:
        cube: Dictionary with cube arrays
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert lists to arrays for saving
    save_dict = {}
    for key, value in cube.items():
        if isinstance(value, list):
            save_dict[key] = np.array(value, dtype=object)
        else:
            save_dict[key] = value

    np.savez_compressed(output_path, **save_dict)

    # Log cube dimensions
    n_transects, n_epochs, n_points, n_features = cube['points'].shape
    logger.info(f"Saved cube to {output_path}")
    logger.info(f"  Shape: ({n_transects} transects, {n_epochs} epochs, {n_points} points, {n_features} features)")


def main():
    """Main extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract transects from LAS files using predefined transect lines. "
                    "Output is in CUBE FORMAT (n_transects, T, N, 12) for spatio-temporal modeling."
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
        help="Directory containing LAS/LAZ files (each file = one epoch)",
    )

    parser.add_argument(
        "--las-files",
        type=Path,
        nargs='+',
        default=None,
        help="Specific LAS/LAZ file(s) to process (each file = one epoch)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NPZ file path (cube format)",
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
        help="Generate visualization of extracted transects (cube format)",
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

    logger.info(f"Found {len(las_files)} LAS file(s) to process (each = one temporal epoch)")
    for f in las_files:
        date = parse_date_from_filename(f.name)
        date_str = date.strftime('%Y-%m-%d') if date else 'unknown'
        logger.info(f"  - {f.name} (date: {date_str})")

    # Initialize extractor
    extractor = ShapefileTransectExtractor(
        n_points=args.n_points,
        buffer_m=args.buffer,
        min_points=args.min_points,
    )

    # Load transect lines
    transect_gdf = extractor.load_transect_lines(args.transects)

    # Extract transects (flat format)
    try:
        flat_transects = extractor.extract_from_shapefile_and_las(
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
    n_extracted = len(flat_transects['points'])
    if n_extracted == 0:
        logger.error("No transects extracted! Check that LAS files overlap with transect lines.")
        return 1

    logger.info(f"\n{'='*60}")
    logger.info(f"FLAT EXTRACTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Total transect-epoch pairs extracted: {n_extracted}")
    logger.info(f"  Points per transect: {args.n_points}")
    logger.info(f"  Features per point: {extractor.N_FEATURES}")
    logger.info(f"  Metadata fields: {extractor.N_METADATA}")

    # Convert to cube format
    logger.info(f"\n{'='*60}")
    logger.info(f"CONVERTING TO CUBE FORMAT")
    logger.info(f"{'='*60}")

    cube = convert_flat_to_cube(flat_transects, n_points=args.n_points)

    n_transects, n_epochs, n_points_cube, n_features = cube['points'].shape

    logger.info(f"\n{'='*60}")
    logger.info(f"CUBE FORMAT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Cube shape: ({n_transects}, {n_epochs}, {n_points_cube}, {n_features})")
    logger.info(f"  Unique transects: {n_transects}")
    logger.info(f"  Temporal epochs: {n_epochs}")
    logger.info(f"  Points per transect: {n_points_cube}")
    logger.info(f"  Features per point: {n_features}")

    # Print epoch info
    logger.info(f"\n  Epochs (chronological):")
    for i, (name, date) in enumerate(zip(cube['epoch_names'], cube['epoch_dates'])):
        logger.info(f"    {i}: {date[:10]} - {name}")

    # Print feature/metadata names
    logger.info(f"\n  Feature names: {cube['feature_names']}")
    logger.info(f"  Metadata names: {cube['metadata_names']}")

    # Statistics from latest epoch
    latest = n_epochs - 1
    cliff_heights = cube['metadata'][:, latest, 0]
    valid_heights = cliff_heights[~np.isnan(cliff_heights)]
    if len(valid_heights) > 0:
        logger.info(f"\n  Cliff height range (latest epoch): {valid_heights.min():.1f} - {valid_heights.max():.1f} m")
        logger.info(f"  Mean cliff height (latest epoch): {valid_heights.mean():.1f} m")

    # Save cube
    save_cube(cube, args.output)

    # Visualize if requested
    if args.visualize:
        viz_path = args.output.parent / f"{args.output.stem}_viz.png"
        try:
            visualize_cube_transects(cube, viz_path)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\nExtraction complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
