#!/usr/bin/env python3
"""Check which specific transects have LiDAR coverage.

Usage:
    python scripts/check_transect_coverage.py
"""

from pathlib import Path
import numpy as np
import laspy

from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Check transect coverage."""

    logger.info("=" * 80)
    logger.info("TRANSECT COVERAGE CHECK")
    logger.info("=" * 80)

    # Load filtered transects
    filtered_file = "results/mops_transects/kml_transects_filtered.npz"
    data = np.load(filtered_file)

    origins = data['origins']
    endpoints = data['endpoints']
    normals = data['normals']
    names = data['names']

    logger.info(f"\nFiltered transects: {len(origins)}")

    # Load LiDAR extent
    lidar_file = "data/testing/20251105_00589_00639_1447_DelMar_NoWaves_beach_cliff_ground_cropped.las"
    las = laspy.read(lidar_file)

    lidar_x_min, lidar_x_max = float(las.x.min()), float(las.x.max())
    lidar_y_min, lidar_y_max = float(las.y.min()), float(las.y.max())

    logger.info(f"\nLiDAR extent:")
    logger.info(f"  X: [{lidar_x_min:.2f}, {lidar_x_max:.2f}]")
    logger.info(f"  Y: [{lidar_y_min:.2f}, {lidar_y_max:.2f}]")

    # Check each transect
    logger.info("\n" + "=" * 80)
    logger.info("TRANSECT ANALYSIS")
    logger.info("=" * 80)

    # Check how many transects have origins inside LiDAR box
    origins_inside = 0
    endpoints_inside = 0
    fully_inside = 0

    for i in range(len(origins)):
        origin = origins[i]
        endpoint = endpoints[i]

        origin_in = (lidar_x_min <= origin[0] <= lidar_x_max and
                     lidar_y_min <= origin[1] <= lidar_y_max)
        endpoint_in = (lidar_x_min <= endpoint[0] <= lidar_x_max and
                       lidar_y_min <= endpoint[1] <= lidar_y_max)

        if origin_in:
            origins_inside += 1
        if endpoint_in:
            endpoints_inside += 1
        if origin_in and endpoint_in:
            fully_inside += 1

        # Print first 10 for inspection
        if i < 10:
            logger.info(f"{names[i]:15s}: Origin in={origin_in}, Endpoint in={endpoint_in}")
            logger.info(f"                Origin:   X={origin[0]:10.2f}, Y={origin[1]:10.2f}")
            logger.info(f"                Endpoint: X={endpoint[0]:10.2f}, Y={endpoint[1]:10.2f}")
            logger.info(f"                Normal:   dX={normals[i][0]:.4f}, dY={normals[i][1]:.4f}")
            logger.info("")

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Transects with origin inside LiDAR:   {origins_inside}/{len(origins)}")
    logger.info(f"Transects with endpoint inside LiDAR: {endpoints_inside}/{len(origins)}")
    logger.info(f"Transects fully inside LiDAR:         {fully_inside}/{len(origins)}")

    # Find closest transect to LiDAR center
    lidar_center_x = (lidar_x_min + lidar_x_max) / 2
    lidar_center_y = (lidar_y_min + lidar_y_max) / 2

    distances = np.sqrt((origins[:, 0] - lidar_center_x)**2 +
                        (origins[:, 1] - lidar_center_y)**2)
    closest_idx = np.argmin(distances)

    logger.info(f"\nLiDAR center: X={lidar_center_x:.2f}, Y={lidar_center_y:.2f}")
    logger.info(f"Closest transect: {names[closest_idx]} (index {closest_idx})")
    logger.info(f"  Origin: X={origins[closest_idx][0]:.2f}, Y={origins[closest_idx][1]:.2f}")
    logger.info(f"  Distance from center: {distances[closest_idx]:.2f}m")

    # Check voxelized results
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION RESULTS")
    logger.info("=" * 80)

    voxel_file = "results/mops_transects/transects_voxelized.npz"
    voxel_data = np.load(voxel_file)

    bin_mask = voxel_data['bin_mask']
    valid_per_transect = bin_mask.sum(axis=1)

    logger.info(f"\nValid bins per transect:")
    for i in range(min(10, len(valid_per_transect))):
        logger.info(f"  {names[i]:15s}: {valid_per_transect[i]:3d} / 128 bins")

    if valid_per_transect.max() == 0:
        logger.warning("\n⚠ WARNING: ALL TRANSECTS HAVE 0 VALID BINS!")
        logger.warning("")
        logger.warning("Possible issues:")
        logger.warning("1. Transect normals pointing wrong direction (away from cliff)")
        logger.warning("2. Corridor width too narrow (try increasing from 2m to 5m)")
        logger.warning("3. Profile length too short (try increasing from 150m)")
        logger.warning("4. Search radius too small")
        logger.warning("")
        logger.warning("RECOMMENDED FIX:")
        logger.warning("Edit scripts/extract_mops_transects.py and change:")
        logger.warning("  CORRIDOR_WIDTH_M = 5.0  # increased from 2.0")
        logger.warning("  PROFILE_LENGTH_M = 200.0  # increased from 150.0")
    else:
        logger.info(f"\n✓ Some transects have valid bins!")
        logger.info(f"  Max valid bins: {valid_per_transect.max()}")
        logger.info(f"  Transects with >0 bins: {(valid_per_transect > 0).sum()}")


if __name__ == "__main__":
    main()
