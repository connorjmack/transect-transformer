#!/usr/bin/env python3
"""Extract specific transects by name pattern.

Usage:
    # Extract transects matching a pattern
    python scripts/extract_specific_transects.py --pattern "MOP 6"

    # Extract specific transect names
    python scripts/extract_specific_transects.py --names "MOP 600" "MOP 601" "MOP 602"

    # Extract using saved best transect indices
    python scripts/extract_specific_transects.py --use-best 10
"""

import argparse
from pathlib import Path
import numpy as np

from src.data.kml_parser import KMLParser
from src.data.transect_voxelizer import TransectVoxelizer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract specific transects by name or pattern"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Extract transects containing this pattern (e.g., 'MOP 6')"
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Extract specific transect names (space-separated)"
    )
    parser.add_argument(
        "--use-best",
        type=int,
        default=None,
        help="Extract top N best transects from find_best_transects.py"
    )
    parser.add_argument(
        "--kml",
        type=str,
        default="data/mops/MOPs-SD.kml",
        help="Path to KML file"
    )
    parser.add_argument(
        "--lidar",
        type=str,
        default="data/testing/20251105_00589_00639_1447_DelMar_NoWaves_beach_cliff_ground_cropped.las",
        help="Path to LiDAR file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/specific_transects/",
        help="Output directory"
    )
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("EXTRACTING SPECIFIC TRANSECTS")
    logger.info("=" * 80)

    # Parse KML
    logger.info("\n[1/3] Parsing KML...")
    parser = KMLParser(utm_zone=11, hemisphere='N')
    kml_transects = parser.parse(args.kml)

    # Extend MOP transects inland by 100m to reach cliff features
    # MOP lines start on beach and point seaward, but cliffs are inland
    logger.info("\nExtending MOP transects inland by 100m...")
    kml_transects = parser.extend_origins_inland(
        kml_transects,
        extension_m=100.0,
        pattern="MOP"
    )

    all_names = kml_transects['names']
    logger.info(f"Total transects in KML: {len(all_names)}")

    # Select transects based on criteria
    selected_indices = []

    if args.use_best is not None:
        # Use best transects from find_best_transects.py
        best_file = "results/mops_transects/best_transect_indices.npz"
        if not Path(best_file).exists():
            logger.error(f"Best transects file not found: {best_file}")
            logger.error("Run: python scripts/find_best_transects.py first")
            return

        data = np.load(best_file)
        selected_indices = data['indices'][:args.use_best].tolist()
        logger.info(f"\nUsing top {args.use_best} best transects from {best_file}")

    elif args.pattern is not None:
        # Find transects matching pattern
        for i, name in enumerate(all_names):
            if args.pattern.upper() in name.upper():
                selected_indices.append(i)
        logger.info(f"\nFound {len(selected_indices)} transects matching pattern '{args.pattern}'")

    elif args.names is not None:
        # Find exact name matches
        name_set = set([n.upper() for n in args.names])
        for i, name in enumerate(all_names):
            if name.upper() in name_set:
                selected_indices.append(i)
        logger.info(f"\nFound {len(selected_indices)} transects matching specified names")

    else:
        logger.error("Must specify --pattern, --names, or --use-best")
        return

    if len(selected_indices) == 0:
        logger.error("No transects selected!")
        return

    # Filter to selected transects
    selected_indices = np.array(selected_indices)
    filtered_transects = {
        'origins': kml_transects['origins'][selected_indices],
        'endpoints': kml_transects['endpoints'][selected_indices],
        'normals': kml_transects['normals'][selected_indices],
        'names': [all_names[i] for i in selected_indices],
        'lengths': kml_transects['lengths'][selected_indices],
    }

    logger.info(f"\nSelected transects:")
    for name in filtered_transects['names'][:20]:  # Show first 20
        logger.info(f"  - {name}")
    if len(filtered_transects['names']) > 20:
        logger.info(f"  ... and {len(filtered_transects['names']) - 20} more")

    # Save filtered transects
    parser.save_transects(filtered_transects, output_dir / "selected_transects.npz")

    # Extract voxelized transects
    logger.info("\n[2/3] Extracting voxelized transects from LiDAR...")

    voxelizer = TransectVoxelizer(
        bin_size_m=1.0,
        corridor_width_m=5.0,
        max_bins=128,
        min_points_per_bin=3,
        profile_length_m=250.0,
    )

    try:
        transects = voxelizer.extract_from_file(
            args.lidar,
            transect_origins=filtered_transects['origins'],
            transect_normals=filtered_transects['normals'],
            transect_names=filtered_transects['names'],
        )
    except FileNotFoundError:
        logger.error(f"LiDAR file not found: {args.lidar}")
        return
    except Exception as e:
        logger.error(f"Error extracting transects: {e}")
        raise

    # Save results
    logger.info("\n[3/3] Saving results...")

    save_path = output_dir / "transects_voxelized.npz"
    voxelizer.save_transects(transects, save_path)

    # Summary
    valid_bins = transects['bin_mask'].sum(axis=1)
    transects_with_data = (valid_bins > 0).sum()

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Extracted: {len(transects['bin_features'])} transects")
    logger.info(f"Total valid bins: {transects['bin_mask'].sum():,} / {transects['bin_mask'].size:,}")
    logger.info(f"Transects with data: {transects_with_data} / {len(transects['bin_features'])}")

    if transects_with_data > 0:
        logger.info(f"\nTop 10 transects by valid bins:")
        sorted_idx = np.argsort(valid_bins)[::-1]
        for i in sorted_idx[:10]:
            if valid_bins[i] > 0:
                logger.info(f"  {transects['names'][i]:20s}: {valid_bins[i]:3d} / 128 bins")

        logger.info(f"\n✓ Results saved to {output_dir}")
        logger.info(f"\nVisualize with:")
        logger.info(f"  python scripts/visualize_sample_transect.py --input {save_path} --transect 0")
    else:
        logger.warning("\n⚠ No transects have valid data!")
        logger.warning("The selected transects may not overlap with LiDAR coverage.")


if __name__ == "__main__":
    main()
