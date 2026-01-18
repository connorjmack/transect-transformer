#!/usr/bin/env python3
"""Extract transects from shapefile.

Usage:
    # Extract all transects from shapefile
    python scripts/extract_from_shapefile.py

    # Specify custom paths
    python scripts/extract_from_shapefile.py \
        --shapefile data/mops/DelMarTransects595to620at1m/DelMarTransects595to620at1m.shp \
        --lidar data/testing/20251105_00589_00639_1447_DelMar_NoWaves_beach_cliff_ground_cropped.las \
        --output results/shapefile_transects/
"""

import argparse
from pathlib import Path
import numpy as np

from src.data.shapefile_parser import ShapefileParser
from src.data.transect_voxelizer import TransectVoxelizer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract transects from shapefile"
    )
    parser.add_argument(
        "--shapefile",
        type=str,
        default="data/mops/DelMarTransects595to620at1m/DelMarTransects595to620at1m.shp",
        help="Path to shapefile"
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
        default="results/shapefile_transects/",
        help="Output directory"
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=1.0,
        help="Bin size along transect (meters)"
    )
    parser.add_argument(
        "--corridor-width",
        type=float,
        default=5.0,
        help="Width of extraction corridor (meters) - total width, split evenly on both sides"
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=128,
        help="Maximum number of bins per transect"
    )
    parser.add_argument(
        "--profile-length",
        type=float,
        default=500.0,
        help="Search distance in each direction from origin (meters)"
    )
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("EXTRACTING TRANSECTS FROM SHAPEFILE")
    logger.info("=" * 80)
    logger.info(f"Shapefile:       {args.shapefile}")
    logger.info(f"LiDAR file:      {args.lidar}")
    logger.info(f"Output:          {output_dir}")
    logger.info(f"Corridor width:  {args.corridor_width}m (±{args.corridor_width/2.0:.1f}m from centerline)")
    logger.info(f"Bin size:        {args.bin_size}m")
    logger.info(f"Profile length:  ±{args.profile_length}m")

    # Step 1: Parse shapefile
    logger.info("\n[1/3] Parsing shapefile...")

    parser = ShapefileParser(target_crs="EPSG:26911")  # UTM Zone 11N

    try:
        transects = parser.parse(args.shapefile)
    except FileNotFoundError:
        logger.error(f"Shapefile not found: {args.shapefile}")
        return
    except ImportError as e:
        logger.error(str(e))
        logger.error("Install geopandas with: pip install geopandas")
        return
    except Exception as e:
        logger.error(f"Error parsing shapefile: {e}")
        raise

    logger.info(f"\n✓ Parsed {len(transects['origins'])} transects")
    logger.info(f"  First few names: {transects['names'][:5]}")

    # Save parsed transects
    parser.save_transects(transects, output_dir / "shapefile_transects.npz")

    # Step 2: Extract voxelized transects from LiDAR
    logger.info("\n[2/3] Extracting voxelized transects from LiDAR...")

    voxelizer = TransectVoxelizer(
        bin_size_m=args.bin_size,
        corridor_width_m=args.corridor_width,
        max_bins=args.max_bins,
        min_points_per_bin=3,
        profile_length_m=args.profile_length,
    )

    try:
        voxelized = voxelizer.extract_from_file(
            args.lidar,
            transect_origins=transects['origins'],
            transect_normals=transects['normals'],
            transect_names=transects['names'],
        )
    except FileNotFoundError:
        logger.error(f"LiDAR file not found: {args.lidar}")
        return
    except Exception as e:
        logger.error(f"Error extracting transects: {e}")
        raise

    # Step 3: Save results
    logger.info("\n[3/3] Saving results...")

    save_path = output_dir / "transects_voxelized.npz"
    voxelizer.save_transects(voxelized, save_path)

    # Summary
    valid_bins = voxelized['bin_mask'].sum(axis=1)
    transects_with_data = (valid_bins > 0).sum()

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Extracted: {len(voxelized['bin_features'])} transects")
    logger.info(f"Total valid bins: {voxelized['bin_mask'].sum():,} / {voxelized['bin_mask'].size:,}")
    logger.info(f"Transects with data: {transects_with_data} / {len(voxelized['bin_features'])}")

    if transects_with_data > 0:
        logger.info(f"\nTop 10 transects by valid bins:")
        sorted_idx = np.argsort(valid_bins)[::-1]
        for i in sorted_idx[:10]:
            if valid_bins[i] > 0:
                logger.info(f"  {voxelized['names'][i]:20s}: {valid_bins[i]:3d} / {args.max_bins} bins")

        logger.info(f"\n✓ Results saved to {output_dir}")
        logger.info(f"\nVisualize with:")
        logger.info(f"  python scripts/visualize_sample_transect.py --input {save_path} --transect 0")
        logger.info(f"  python scripts/visualize_multiple_transects.py --input {save_path} --n 6")
    else:
        logger.warning("\n⚠ No transects have valid data!")
        logger.warning("The transects may not overlap with LiDAR coverage.")


if __name__ == "__main__":
    main()
