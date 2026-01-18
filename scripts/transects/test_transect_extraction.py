"""Test script for transect extraction methods.

Compares the original interpolation-based TransectExtractor with the
voxelized TransectVoxelizer on real LiDAR data using KML transect lines.

Usage:
    python scripts/test_transect_extraction.py \\
        --lidar data/lidar/my_scan.laz \\
        --kml data/mops/MOPs-SD.kml \\
        --output results/transects/ \\
        --method both  # or 'interpolation' or 'voxelized'
"""

import argparse
import time
from pathlib import Path

import numpy as np

from src.data.kml_parser import KMLParser
from src.data.transect_extractor import TransectExtractor
from src.data.transect_voxelizer import TransectVoxelizer
from src.data.spatial_filter import filter_transects_by_lidar
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test transect extraction methods on real LiDAR data"
    )
    parser.add_argument(
        "--lidar",
        type=str,
        required=True,
        help="Path to LiDAR file (.las or .laz)"
    )
    parser.add_argument(
        "--kml",
        type=str,
        default="data/mops/MOPs-SD.kml",
        help="Path to KML file with transect lines"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/transects/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["interpolation", "voxelized", "both"],
        default="both",
        help="Which extraction method to use"
    )
    parser.add_argument(
        "--utm-zone",
        type=int,
        default=11,
        help="UTM zone for coordinate projection (default: 11 for San Diego)"
    )
    parser.add_argument(
        "--n-transects",
        type=int,
        default=None,
        help="Limit number of transects to process (for testing)"
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=10.0,
        help="Alongshore spacing for auto-detected transects (m)"
    )
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save side-by-side comparison plots"
    )
    parser.add_argument(
        "--filter-overlap",
        action="store_true",
        help="Only extract transects overlapping with LiDAR extent"
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=50.0,
        help="Buffer distance around LiDAR extent (m, default: 50)"
    )
    return parser.parse_args()


def extract_interpolation(
    lidar_path: Path,
    transect_origins: np.ndarray,
    transect_normals: np.ndarray,
    transect_names: list,
) -> dict:
    """Run interpolation-based extraction.

    Args:
        lidar_path: Path to LiDAR file
        transect_origins: (N, 3) transect start points
        transect_normals: (N, 2) shore-normal vectors
        transect_names: List of transect names

    Returns:
        Dictionary with extracted transects and timing info
    """
    logger.info("=" * 80)
    logger.info("INTERPOLATION-BASED EXTRACTION (Original Method)")
    logger.info("=" * 80)

    extractor = TransectExtractor(
        n_points=128,
        spacing_m=10.0,
        profile_length_m=150.0,
        min_points=20,
        search_radius_m=2.0,
    )

    start_time = time.time()

    # Note: Original extractor expects to extract from file directly
    # For compatibility with KML-defined transects, we'd need to modify it
    # For now, use auto-detection
    transects = extractor.extract_from_file(lidar_path)

    elapsed = time.time() - start_time

    logger.info(f"Extracted {len(transects['points'])} transects")
    logger.info(f"Time: {elapsed:.2f}s ({elapsed/len(transects['points']):.3f}s per transect)")
    logger.info(f"Output shape: {transects['points'].shape}")

    return {
        'transects': transects,
        'elapsed': elapsed,
        'n_transects': len(transects['points']),
    }


def extract_voxelized(
    lidar_path: Path,
    transect_origins: np.ndarray,
    transect_normals: np.ndarray,
    transect_names: list,
) -> dict:
    """Run voxelized extraction.

    Args:
        lidar_path: Path to LiDAR file
        transect_origins: (N, 3) transect start points
        transect_normals: (N, 2) shore-normal vectors
        transect_names: List of transect names

    Returns:
        Dictionary with extracted transects and timing info
    """
    logger.info("=" * 80)
    logger.info("VOXELIZED EXTRACTION (1D Binning Method)")
    logger.info("=" * 80)

    voxelizer = TransectVoxelizer(
        bin_size_m=1.0,
        corridor_width_m=2.0,
        max_bins=128,
        min_points_per_bin=3,
        profile_length_m=150.0,
    )

    start_time = time.time()

    transects = voxelizer.extract_from_file(
        lidar_path,
        transect_origins=transect_origins,
        transect_normals=transect_normals,
        transect_names=transect_names,
    )

    elapsed = time.time() - start_time

    logger.info(f"Extracted {len(transects['bin_features'])} transects")
    logger.info(f"Time: {elapsed:.2f}s ({elapsed/len(transects['bin_features']):.3f}s per transect)")
    logger.info(f"Output shape: {transects['bin_features'].shape}")
    logger.info(f"Valid bins: {transects['bin_mask'].sum()}/{transects['bin_mask'].size}")

    return {
        'transects': transects,
        'elapsed': elapsed,
        'n_transects': len(transects['bin_features']),
    }


def compare_results(interp_result: dict, voxel_result: dict, output_dir: Path):
    """Compare results from both methods.

    Args:
        interp_result: Results from interpolation method
        voxel_result: Results from voxelized method
        output_dir: Directory to save comparison
    """
    logger.info("=" * 80)
    logger.info("COMPARISON")
    logger.info("=" * 80)

    # Timing comparison
    logger.info(f"Interpolation: {interp_result['elapsed']:.2f}s for {interp_result['n_transects']} transects")
    logger.info(f"Voxelized:     {voxel_result['elapsed']:.2f}s for {voxel_result['n_transects']} transects")

    speedup = interp_result['elapsed'] / (voxel_result['elapsed'] + 1e-8)
    logger.info(f"Speedup: {speedup:.2f}x")

    # Shape comparison
    interp_transects = interp_result['transects']
    voxel_transects = voxel_result['transects']

    logger.info(f"\nInterpolation output shapes:")
    logger.info(f"  points:    {interp_transects['points'].shape}")
    logger.info(f"  distances: {interp_transects['distances'].shape}")
    logger.info(f"  metadata:  {interp_transects['metadata'].shape}")

    logger.info(f"\nVoxelized output shapes:")
    logger.info(f"  bin_features: {voxel_transects['bin_features'].shape}")
    logger.info(f"  bin_centers:  {voxel_transects['bin_centers'].shape}")
    logger.info(f"  bin_mask:     {voxel_transects['bin_mask'].shape}")
    logger.info(f"  metadata:     {voxel_transects['metadata'].shape}")

    # Summary statistics
    logger.info(f"\nSummary Statistics:")

    # Interpolation
    if len(interp_transects['points']) > 0:
        elevations = interp_transects['points'][:, :, 1]  # Assuming elevation is feature 1
        logger.info(f"Interpolation - Elevation range: [{elevations.min():.2f}, {elevations.max():.2f}]m")

    # Voxelized
    if len(voxel_transects['bin_features']) > 0:
        valid_features = voxel_transects['bin_features'][voxel_transects['bin_mask']]
        if len(valid_features) > 0:
            mean_elevs = valid_features[:, 0]
            logger.info(f"Voxelized     - Elevation range: [{mean_elevs.min():.2f}, {mean_elevs.max():.2f}]m")
            logger.info(f"Voxelized     - Valid bin ratio: {voxel_transects['bin_mask'].mean():.2%}")


def main():
    """Main execution."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("TRANSECT EXTRACTION TEST")
    logger.info("=" * 80)
    logger.info(f"LiDAR file: {args.lidar}")
    logger.info(f"KML file:   {args.kml}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Method:     {args.method}")

    # Step 1: Parse KML to get transect lines
    logger.info("\n" + "=" * 80)
    logger.info("PARSING KML TRANSECT LINES")
    logger.info("=" * 80)

    kml_parser = KMLParser(utm_zone=args.utm_zone, hemisphere='N')
    kml_transects = kml_parser.parse(args.kml)

    logger.info(f"Extracted {len(kml_transects['origins'])} transect lines from KML")
    logger.info(f"Transect length range: [{kml_transects['lengths'].min():.1f}, {kml_transects['lengths'].max():.1f}]m")

    # Filter by spatial overlap if requested
    if args.filter_overlap:
        logger.info(f"\nFiltering transects by LiDAR overlap (buffer: {args.buffer_m}m)...")
        kml_transects = filter_transects_by_lidar(
            kml_transects,
            args.lidar,
            buffer_m=args.buffer_m
        )
        logger.info(f"Kept {len(kml_transects['origins'])} overlapping transects")

    # Limit number of transects if requested
    if args.n_transects is not None:
        n = min(args.n_transects, len(kml_transects['origins']))
        logger.info(f"Limiting to first {n} transects")
        kml_transects = {
            'origins': kml_transects['origins'][:n],
            'endpoints': kml_transects['endpoints'][:n],
            'normals': kml_transects['normals'][:n],
            'names': kml_transects['names'][:n],
            'lengths': kml_transects['lengths'][:n],
        }

    # Save parsed KML transects
    kml_parser.save_transects(kml_transects, output_dir / "kml_transects.npz")

    # Step 2: Extract transects using selected method(s)
    lidar_path = Path(args.lidar)

    results = {}

    if args.method in ["interpolation", "both"]:
        results['interpolation'] = extract_interpolation(
            lidar_path,
            kml_transects['origins'],
            kml_transects['normals'],
            kml_transects['names'],
        )

        # Save results
        save_path = output_dir / "transects_interpolation.npz"
        np.savez_compressed(save_path, **results['interpolation']['transects'])
        logger.info(f"Saved interpolation results to {save_path}")

    if args.method in ["voxelized", "both"]:
        results['voxelized'] = extract_voxelized(
            lidar_path,
            kml_transects['origins'],
            kml_transects['normals'],
            kml_transects['names'],
        )

        # Save results
        save_path = output_dir / "transects_voxelized.npz"
        np.savez_compressed(save_path, **results['voxelized']['transects'])
        logger.info(f"Saved voxelized results to {save_path}")

    # Step 3: Compare results if both methods were run
    if args.method == "both":
        compare_results(
            results['interpolation'],
            results['voxelized'],
            output_dir
        )

    logger.info("\n" + "=" * 80)
    logger.info("DONE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
