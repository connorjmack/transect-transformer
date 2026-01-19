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

Cross-Platform Support:
  Automatically converts paths between Mac (/Volumes/group/...) and Linux
  (/project/group/...) formats. The script detects the current OS and converts
  paths from survey CSVs accordingly. Use --target-os to override auto-detection.

TODO: Future enhancement - add per-point normal vectors (nx, ny, nz) computed from
      the local point neighborhood. This will enable better characterization of
      cliff face orientation and overhang detection.

Usage:
    # Basic extraction from LAS directory
    python scripts/processing/extract_transects.py \\
        --transects data/mops/transects_10m/transect_lines.shp \\
        --las-dir data/testing/ \\
        --output data/processed/transects_cube.npz \\
        --buffer 1.0 \\
        --n-points 128 \\
        --visualize

    # From survey CSV (auto-converts paths for current OS)
    python scripts/processing/extract_transects.py \\
        --transects data/mops/transects_10m/transect_lines.shp \\
        --survey-csv data/raw/master_list.csv \\
        --output data/processed/all_transects.npz

    # From survey CSV with LAZ requirement (errors if LAZ files don't exist)
    # Priority: .copc.laz (10-100x faster) > .laz (5-10x faster) > .las (fallback)
    python scripts/processing/extract_transects.py \\
        --transects data/mops/transects_10m/transect_lines.shp \\
        --survey-csv data/raw/master_list.csv \\
        --prefer-laz \\
        --output data/processed/all_transects.npz

    # Force Linux paths (e.g., when CSV has Mac paths but running on Linux)
    python scripts/processing/extract_transects.py \\
        --transects data/mops/transects_10m/transect_lines.shp \\
        --survey-csv data/raw/master_list.csv \\
        --target-os linux \\
        --output data/processed/all_transects.npz
"""

import argparse
import gc
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import laspy
    from laspy import CopcReader
    HAS_COPC = True
except ImportError:
    HAS_COPC = False
    laspy = None
    CopcReader = None

# Add project root to path for imports
# __file__ is scripts/processing/extract_transects.py
# .parent.parent.parent gets us to the project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.shapefile_transect_extractor import ShapefileTransectExtractor
from src.utils.logging import setup_logger

# Import QC module - handle both direct execution and module import
try:
    from qc_cube import run_qc  # Direct script execution
except ImportError:
    from scripts.processing.qc_cube import run_qc  # Module import (e.g., from tests)

logger = setup_logger(__name__, level="INFO")

# Beach name to MOP range mapping (San Diego County)
# These ranges match the transect shapefile definitions
BEACH_MOP_RANGES = {
    'blacks': (520, 567),
    'torrey': (567, 581),
    'delmar': (595, 620),
    'solana': (637, 666),
    'sanelijo': (683, 708),
    'encinitas': (708, 764),
}

# Cross-platform path mapping
# Maps path prefixes between Mac and Linux
PATH_MAPPINGS = {
    'mac_to_linux': {
        '/Volumes/group': '/project/group',
        '/Volumes/': '/project/',
    },
    'linux_to_mac': {
        '/project/group': '/Volumes/group',
        '/project/': '/Volumes/',
    },
}


# =============================================================================
# COPC (Cloud Optimized Point Cloud) Support
# =============================================================================

def is_copc_file(las_path: Path) -> bool:
    """Check if LAS file has COPC spatial index (fast header check).

    COPC files enable direct spatial queries without scanning the entire file,
    providing 10-100x faster loading for narrow spatial queries.

    Args:
        las_path: Path to LAS/LAZ file

    Returns:
        True if file has COPC VLR, False otherwise
    """
    if not HAS_COPC or laspy is None:
        return False

    try:
        with laspy.open(las_path) as f:
            for vlr in f.header.vlrs:
                if vlr.user_id == 'copc':
                    return True
        return False
    except Exception as e:
        logger.debug(f"Could not check COPC status for {las_path}: {e}")
        return False


def get_copc_path(las_path: Path) -> Path:
    """Get the expected COPC path for a LAS file.

    Args:
        las_path: Path to original LAS/LAZ file

    Returns:
        Path where COPC version would be (same dir, .copc.laz extension)
    """
    stem = las_path.stem
    if stem.endswith('.copc'):
        return las_path
    return las_path.with_name(stem + '.copc.laz')


def find_copc_version(las_path: Path) -> Optional[Path]:
    """Check if a COPC version exists for a LAS file.

    Args:
        las_path: Path to original LAS/LAZ file

    Returns:
        Path to COPC version if it exists, None otherwise
    """
    # If already a COPC file, return as-is
    if '.copc.' in las_path.name.lower():
        if las_path.exists():
            return las_path
        return None

    # Check for .copc.laz version
    copc_path = get_copc_path(las_path)
    if copc_path.exists():
        return copc_path

    return None


def substitute_copc_files(las_files: List[Path], verbose: bool = True) -> Tuple[List[Path], int]:
    """Substitute LAS files with COPC versions where available.

    Args:
        las_files: List of LAS/LAZ file paths
        verbose: If True, log substitutions

    Returns:
        Tuple of (updated file list, number of substitutions made)
    """
    result = []
    substituted = 0

    for las_path in las_files:
        copc_path = find_copc_version(las_path)
        if copc_path and copc_path != las_path:
            result.append(copc_path)
            substituted += 1
            if verbose:
                logger.debug(f"Using COPC: {copc_path.name} instead of {las_path.name}")
        else:
            result.append(las_path)

    return result, substituted


def substitute_laz_files(las_files: List[Path], verbose: bool = True, require: bool = False) -> Tuple[List[Path], int, int]:
    """Substitute LAS files with LAZ/COPC.LAZ versions where available.

    Priority order:
    1. .copc.laz (best: compressed + spatial index)
    2. .laz (good: compressed)
    3. .las (fallback: uncompressed)

    Args:
        las_files: List of LAS/LAZ file paths
        verbose: If True, log substitutions
        require: If True, raise error if LAZ version doesn't exist

    Returns:
        Tuple of (updated file list, number of substitutions made, number of COPC files found)
    """
    result = []
    substituted = 0
    copc_count = 0

    for las_path in las_files:
        # Already a COPC.LAZ file - perfect!
        if las_path.name.endswith('.copc.laz'):
            result.append(las_path)
            copc_count += 1
            if verbose:
                logger.debug(f"Using COPC.LAZ: {las_path.name}")
            continue

        # Already a LAZ file (check if COPC)
        if las_path.suffix.lower() == '.laz':
            result.append(las_path)
            # Check if it's COPC
            if is_copc_file(las_path):
                copc_count += 1
                if verbose:
                    logger.debug(f"LAZ file has COPC index: {las_path.name}")
            continue

        # Try substituting .las with .copc.laz first (best option)
        if las_path.suffix.lower() == '.las':
            # Try .copc.laz first
            copc_laz_path = las_path.with_suffix('.copc.laz')
            if copc_laz_path.exists():
                result.append(copc_laz_path)
                substituted += 1
                copc_count += 1
                if verbose:
                    logger.debug(f"Using COPC.LAZ: {copc_laz_path.name} (10-100x faster!)")
                continue

            # Fall back to .laz
            laz_path = las_path.with_suffix('.laz')
            if laz_path.exists():
                result.append(laz_path)
                substituted += 1
                # Check if it's COPC
                if is_copc_file(laz_path):
                    copc_count += 1
                    if verbose:
                        logger.debug(f"Using LAZ with COPC index: {laz_path.name}")
                elif verbose:
                    logger.debug(f"Using LAZ: {laz_path.name} (no COPC index)")
                continue

            # No LAZ version found
            if require:
                raise FileNotFoundError(
                    f"LAZ file required but not found.\n"
                    f"Tried: {copc_laz_path}\n"
                    f"Tried: {laz_path}\n"
                    f"Original LAS path: {las_path}"
                )
            else:
                result.append(las_path)
        else:
            result.append(las_path)

    return result, substituted, copc_count


def get_current_os() -> str:
    """Detect current operating system.

    Returns:
        'mac', 'linux', or 'windows'
    """
    system = platform.system().lower()
    if system == 'darwin':
        return 'mac'
    elif system == 'linux':
        return 'linux'
    elif system == 'windows':
        return 'windows'
    return system


def convert_path_for_os(path: str, target_os: Optional[str] = None) -> str:
    """Convert path between Mac and Linux formats.

    Automatically detects current OS and converts paths accordingly.
    Paths stored as Mac format (/Volumes/group/...) are converted to
    Linux format (/projects/group/...) when running on Linux, and vice versa.

    Args:
        path: Original path string
        target_os: Target OS ('mac' or 'linux'). If None, auto-detects.

    Returns:
        Converted path string
    """
    if target_os is None:
        target_os = get_current_os()

    path_str = str(path)

    # Determine source OS from path
    if path_str.startswith('/Volumes/'):
        source_os = 'mac'
    elif path_str.startswith('/project/'):
        source_os = 'linux'
    else:
        # Unknown format, return as-is
        return path_str

    # No conversion needed if same OS
    if source_os == target_os:
        return path_str

    # Get mapping direction
    if source_os == 'mac' and target_os == 'linux':
        mappings = PATH_MAPPINGS['mac_to_linux']
    elif source_os == 'linux' and target_os == 'mac':
        mappings = PATH_MAPPINGS['linux_to_mac']
    else:
        return path_str

    # Apply mappings (longer prefixes first for specificity)
    for old_prefix, new_prefix in sorted(mappings.items(), key=lambda x: -len(x[0])):
        if path_str.startswith(old_prefix):
            converted = new_prefix + path_str[len(old_prefix):]
            return converted

    return path_str


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

    logger.debug(f"Converting to cube format:")
    logger.debug(f"  Unique transects: {n_transects}")
    logger.debug(f"  Unique epochs: {n_epochs}")
    logger.debug(f"  Epochs (sorted): {sorted_epochs}")

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
        logger.debug(f"Missing data: {missing_count}/{total_cells} ({missing_pct:.1f}%) transect-epoch pairs")
    else:
        logger.debug(f"Full coverage: all {n_transects} transects present in all {n_epochs} epochs")

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
    logger.debug(f"Saved cube to {output_path}")
    logger.debug(f"  Shape: ({n_transects} transects, {n_epochs} epochs, {n_points} points, {n_features} features)")


def print_header():
    """Print clean header with timestamp."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "="*70)
    print("TRANSECT EXTRACTION")
    print("="*70)
    print(f"Started: {now}")
    print()


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

    parser.add_argument(
        "--survey-csv",
        type=Path,
        default=None,
        help="CSV file with survey list (must have 'full_path' column or use --path-col)",
    )

    parser.add_argument(
        "--path-col",
        type=str,
        default="full_path",
        help="Column name for LAS file paths in survey CSV (default: full_path)",
    )

    parser.add_argument(
        "--mop-min",
        type=int,
        default=None,
        help="Filter CSV to surveys covering MOPs >= this value (requires MOP1 column)",
    )

    parser.add_argument(
        "--mop-max",
        type=int,
        default=None,
        help="Filter CSV to surveys covering MOPs <= this value (requires MOP2 column)",
    )

    parser.add_argument(
        "--beach",
        type=str,
        default=None,
        choices=list(BEACH_MOP_RANGES.keys()),
        help=f"Beach name to auto-set MOP range. Options: {', '.join(BEACH_MOP_RANGES.keys())}",
    )

    parser.add_argument(
        "--target-os",
        type=str,
        default=None,
        choices=['mac', 'linux'],
        help="Override OS detection for path conversion (default: auto-detect)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for LAS processing (default: 1 = sequential)",
    )

    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit to first N LAS files (useful for testing before full run)",
    )

    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip automatic QC checks after extraction",
    )

    parser.add_argument(
        "--prefer-copc",
        action="store_true",
        help="Automatically use .copc.laz files when available (10-100x faster loading)",
    )

    parser.add_argument(
        "--prefer-laz",
        action="store_true",
        help="REQUIRE .laz files instead of .las (faster loading, smaller files). Will error if LAZ not found.",
    )

    args = parser.parse_args()

    # Print clean header
    print_header()

    # Apply beach preset if specified
    if args.beach:
        mop_min, mop_max = BEACH_MOP_RANGES[args.beach]
        if args.mop_min is None:
            args.mop_min = mop_min
        if args.mop_max is None:
            args.mop_max = mop_max
        logger.info(f"Beach '{args.beach}' selected: MOP range {args.mop_min}-{args.mop_max}")

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
    if args.survey_csv:
        # Load LAS paths from CSV
        import pandas as pd
        if not args.survey_csv.exists():
            logger.error(f"Survey CSV not found: {args.survey_csv}")
            return 1

        survey_df = pd.read_csv(args.survey_csv)
        logger.info(f"Loaded survey CSV with {len(survey_df)} rows")

        if args.path_col not in survey_df.columns:
            logger.error(f"Path column '{args.path_col}' not found in CSV. Available: {survey_df.columns.tolist()}")
            return 1

        # Apply MOP range filters if specified
        if args.mop_min is not None and 'MOP1' in survey_df.columns:
            original_count = len(survey_df)
            survey_df = survey_df[survey_df['MOP2'] >= args.mop_min]
            logger.info(f"Filtered MOP2 >= {args.mop_min}: {original_count} -> {len(survey_df)} rows")

        if args.mop_max is not None and 'MOP2' in survey_df.columns:
            original_count = len(survey_df)
            survey_df = survey_df[survey_df['MOP1'] <= args.mop_max]
            logger.info(f"Filtered MOP1 <= {args.mop_max}: {original_count} -> {len(survey_df)} rows")

        # Extract paths and convert for current OS
        target_os = args.target_os if args.target_os else get_current_os()
        raw_paths = survey_df[args.path_col].tolist()
        csv_paths = []
        converted_count = 0

        for p in raw_paths:
            converted = convert_path_for_os(p, target_os=target_os)
            if converted != p:
                converted_count += 1
            csv_paths.append(Path(converted))

        las_files.extend(csv_paths)
        logger.info(f"Added {len(csv_paths)} LAS files from CSV (target OS: {target_os})")
        if converted_count > 0:
            logger.info(f"  Converted {converted_count} paths for {target_os}")

    if not las_files:
        logger.error("No LAS files specified. Use --las-dir, --las-files, or --survey-csv")
        return 1

    # Apply limit if specified
    if args.limit is not None:
        original_count = len(las_files)
        las_files = las_files[:args.limit]
        logger.info(f"Limited to first {args.limit} of {original_count} LAS files (--limit)")

    # Substitute LAZ files if --prefer-laz is set
    if args.prefer_laz:
        las_files, n_laz, n_copc = substitute_laz_files(las_files, verbose=False, require=True)
        if n_laz > 0:
            print(f"✓ Using {n_laz} LAZ files (compressed format)")
            if n_copc > 0:
                print(f"✓ {n_copc} files have COPC spatial index (fast loading)")
        else:
            if n_copc > 0:
                print(f"✓ {n_copc} files have COPC spatial index (fast loading)")

    # Substitute COPC files if --prefer-copc is set
    if args.prefer_copc:
        las_files, n_copc = substitute_copc_files(las_files, verbose=True)
        if n_copc > 0:
            logger.info(f"Substituted {n_copc} files with COPC versions (10-100x faster loading)")
        else:
            logger.info("No COPC versions found - using original LAS files")

    # Check all files exist
    for f in las_files:
        if not f.exists():
            logger.error(f"LAS file not found: {f}")
            return 1

    print(f"Files to process: {len(las_files)} (each = one temporal epoch)")
    if len(las_files) <= 10:
        for f in las_files:
            date = parse_date_from_filename(f.name)
            date_str = date.strftime('%Y-%m-%d') if date else 'unknown'
            print(f"  • {f.name} ({date_str})")
    else:
        # Just show first and last
        for f in las_files[:3]:
            date = parse_date_from_filename(f.name)
            date_str = date.strftime('%Y-%m-%d') if date else 'unknown'
            print(f"  • {f.name} ({date_str})")
        print(f"  ... and {len(las_files) - 6} more ...")
        for f in las_files[-3:]:
            date = parse_date_from_filename(f.name)
            date_str = date.strftime('%Y-%m-%d') if date else 'unknown'
            print(f"  • {f.name} ({date_str})")
    print()

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
            n_workers=args.workers,
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

    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Transect-epoch pairs: {n_extracted:,}")
    print(f"Points per transect: {args.n_points}")
    print(f"Features per point: {extractor.N_FEATURES}")
    print()

    # Convert to cube format
    print("Converting to cube format...")

    cube = convert_flat_to_cube(flat_transects, n_points=args.n_points)

    n_transects, n_epochs, n_points_cube, n_features = cube['points'].shape

    print("\n" + "="*70)
    print("CUBE FORMAT")
    print("="*70)
    print(f"Shape: ({n_transects} transects, {n_epochs} epochs, {n_points_cube} points, {n_features} features)")
    print(f"\nEpochs:")
    for i, (name, date) in enumerate(zip(cube['epoch_names'], cube['epoch_dates'])):
        print(f"  {i+1}. {date[:10]}")

    # Statistics from latest epoch
    latest = n_epochs - 1
    cliff_heights = cube['metadata'][:, latest, 0]
    valid_heights = cliff_heights[~np.isnan(cliff_heights)]
    if len(valid_heights) > 0:
        print(f"\nCliff heights (latest epoch):")
        print(f"  Range: {valid_heights.min():.1f} - {valid_heights.max():.1f} m")
        print(f"  Mean: {valid_heights.mean():.1f} ± {valid_heights.std():.1f} m")

    # Save cube
    save_cube(cube, args.output)

    # Run QC checks
    if not args.skip_qc:
        print("\n")
        print("=" * 70)
        print("RUNNING QUALITY CONTROL CHECKS")
        print("=" * 70)

        try:
            qc_report = run_qc(args.output, verbose=False)

            # Display summary
            status_symbol = "✓" if qc_report.passed else "✗"
            status_text = "PASSED" if qc_report.passed else "FAILED"
            if qc_report.passed and qc_report.warnings:
                status_symbol = "⚠"
                status_text = "PASSED WITH WARNINGS"

            print(f"\nQC Status: {status_symbol} {status_text}")
            print(f"   Errors: {len(qc_report.errors)}")
            print(f"   Warnings: {len(qc_report.warnings)}")

            # Show key stats
            print("\nKey Statistics:")
            for key in ['n_transects', 'n_epochs', 'coverage_pct', 'date_range', 'file_size']:
                if key in qc_report.stats:
                    print(f"   {key}: {qc_report.stats[key]}")

            # Show errors
            if qc_report.errors:
                print("\n" + "-" * 50)
                print("ERRORS (must fix):")
                for err in qc_report.errors:
                    print(f"   {err}")

            # Show warnings
            if qc_report.warnings:
                print("\n" + "-" * 50)
                print("WARNINGS (review recommended):")
                for warn in qc_report.warnings[:10]:  # Limit to first 10
                    print(f"   {warn}")
                if len(qc_report.warnings) > 10:
                    print(f"   ... and {len(qc_report.warnings) - 10} more warnings")

            print("\n" + "=" * 70)

            if not qc_report.passed:
                logger.error("QC checks failed! Review errors above.")

        except Exception as e:
            logger.warning(f"QC checks failed to run: {e}")
            import traceback
            traceback.print_exc()

    # Visualize if requested
    if args.visualize:
        viz_path = args.output.parent / f"{args.output.stem}_viz.png"
        try:
            visualize_cube_transects(cube, viz_path)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "="*70)
    print("✓ EXTRACTION COMPLETE")
    print("="*70)
    print(f"Completed: {now}")
    print(f"Output: {args.output}")
    print(f"Size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
