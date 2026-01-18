#!/usr/bin/env python3
"""Find which KML transects actually overlap with LiDAR data.

Usage:
    python scripts/find_best_transects.py
"""

from pathlib import Path
import numpy as np
import laspy

from src.data.kml_parser import KMLParser
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Find best overlapping transects."""

    # Paths
    kml_file = "data/mops/MOPs-SD.kml"
    lidar_file = "data/testing/20251105_00589_00639_1447_DelMar_NoWaves_beach_cliff_ground_cropped.las"

    logger.info("=" * 80)
    logger.info("FINDING BEST TRANSECTS")
    logger.info("=" * 80)

    # Load LiDAR extent
    logger.info("\nLoading LiDAR extent...")
    las = laspy.read(lidar_file)
    lidar_x_min, lidar_x_max = float(las.x.min()), float(las.x.max())
    lidar_y_min, lidar_y_max = float(las.y.min()), float(las.y.max())
    lidar_center_x = (lidar_x_min + lidar_x_max) / 2
    lidar_center_y = (lidar_y_min + lidar_y_max) / 2

    logger.info(f"LiDAR extent:")
    logger.info(f"  X: [{lidar_x_min:.2f}, {lidar_x_max:.2f}]")
    logger.info(f"  Y: [{lidar_y_min:.2f}, {lidar_y_max:.2f}]")
    logger.info(f"  Center: ({lidar_center_x:.2f}, {lidar_center_y:.2f})")

    # Parse KML
    logger.info("\nParsing KML transects...")
    parser = KMLParser(utm_zone=11, hemisphere='N')
    kml_transects = parser.parse(kml_file)

    # Extend MOP transects inland by 100m to reach cliff features
    # MOP lines start on beach and point seaward, but cliffs are inland
    logger.info("\nExtending MOP transects inland by 100m...")
    kml_transects = parser.extend_origins_inland(
        kml_transects,
        extension_m=100.0,
        pattern="MOP"
    )

    origins = kml_transects['origins']
    endpoints = kml_transects['endpoints']
    names = kml_transects['names']

    logger.info(f"Total KML transects: {len(origins)}")

    # Find transects that have origins AND endpoints inside the LiDAR box
    logger.info("\nAnalyzing overlap...")

    results = []

    for i in range(len(origins)):
        origin = origins[i]
        endpoint = endpoints[i]
        name = names[i]

        # Check if origin is inside
        origin_in = (lidar_x_min <= origin[0] <= lidar_x_max and
                     lidar_y_min <= origin[1] <= lidar_y_max)

        # Check if endpoint is inside
        endpoint_in = (lidar_x_min <= endpoint[0] <= lidar_x_max and
                       lidar_y_min <= endpoint[1] <= lidar_y_max)

        # Distance from origin to LiDAR center
        dist_to_center = np.sqrt((origin[0] - lidar_center_x)**2 +
                                 (origin[1] - lidar_center_y)**2)

        # Distance from endpoint to LiDAR center
        endpoint_dist = np.sqrt((endpoint[0] - lidar_center_x)**2 +
                               (endpoint[1] - lidar_center_y)**2)

        # Minimum distance (either origin or endpoint)
        min_dist = min(dist_to_center, endpoint_dist)

        # Score: prefer both inside > origin inside > close to center
        if origin_in and endpoint_in:
            score = 1000 - min_dist  # Both inside, prefer closer
        elif origin_in:
            score = 500 - min_dist   # Origin inside
        elif endpoint_in:
            score = 400 - min_dist   # Endpoint inside
        else:
            score = -min_dist        # Neither inside, penalize by distance

        results.append({
            'index': i,
            'name': name,
            'origin_in': origin_in,
            'endpoint_in': endpoint_in,
            'dist_to_center': dist_to_center,
            'min_dist': min_dist,
            'score': score,
            'origin_x': origin[0],
            'origin_y': origin[1],
        })

    # Sort by score (best first)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Show top 20 best transects
    logger.info("\n" + "=" * 80)
    logger.info("TOP 20 BEST TRANSECTS (by overlap score)")
    logger.info("=" * 80)
    logger.info(f"{'Rank':<5} {'Name':<20} {'Origin In':<10} {'End In':<8} {'Dist(m)':<10} {'Score':<10}")
    logger.info("-" * 80)

    for rank, r in enumerate(results[:20], 1):
        logger.info(
            f"{rank:<5} {r['name']:<20} "
            f"{'Yes' if r['origin_in'] else 'No':<10} "
            f"{'Yes' if r['endpoint_in'] else 'No':<8} "
            f"{r['min_dist']:<10.1f} {r['score']:<10.1f}"
        )

    # Show transects that have both origin and endpoint inside
    fully_inside = [r for r in results if r['origin_in'] and r['endpoint_in']]

    logger.info("\n" + "=" * 80)
    logger.info(f"TRANSECTS FULLY INSIDE LiDAR EXTENT: {len(fully_inside)}")
    logger.info("=" * 80)

    if len(fully_inside) > 0:
        for r in fully_inside[:10]:  # Show first 10
            logger.info(f"  {r['name']:<20} at ({r['origin_x']:.1f}, {r['origin_y']:.1f})")
    else:
        logger.info("  None found!")

    # Show transects with at least origin inside
    origin_inside = [r for r in results if r['origin_in']]

    logger.info("\n" + "=" * 80)
    logger.info(f"TRANSECTS WITH ORIGIN INSIDE LiDAR: {len(origin_inside)}")
    logger.info("=" * 80)

    if len(origin_inside) > 0:
        for r in origin_inside[:10]:
            logger.info(f"  {r['name']:<20} at ({r['origin_x']:.1f}, {r['origin_y']:.1f})")

    # Search for MOP transects specifically
    logger.info("\n" + "=" * 80)
    logger.info("SEARCHING FOR 'MOP' TRANSECTS")
    logger.info("=" * 80)

    mop_transects = [r for r in results if 'MOP' in r['name'].upper()]
    logger.info(f"Found {len(mop_transects)} MOP transects")

    if len(mop_transects) > 0:
        logger.info("\nTop 10 MOP transects by overlap:")
        for rank, r in enumerate(mop_transects[:10], 1):
            logger.info(
                f"  {rank}. {r['name']:<20} "
                f"Origin In: {'Yes' if r['origin_in'] else 'No':<4} "
                f"Dist: {r['min_dist']:.1f}m"
            )

        # Check if MOP 600 exists
        mop_600 = [r for r in results if '600' in r['name']]
        if len(mop_600) > 0:
            logger.info(f"\nFound transects with '600' in name:")
            for r in mop_600:
                logger.info(
                    f"  {r['name']:<20} "
                    f"Origin In: {'Yes' if r['origin_in'] else 'No':<4} "
                    f"at ({r['origin_x']:.1f}, {r['origin_y']:.1f}) "
                    f"Dist: {r['min_dist']:.1f}m"
                )

    # Recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)

    if len(fully_inside) > 0:
        logger.info(f"✓ Extract the {len(fully_inside)} transects that are fully inside LiDAR extent")
        logger.info(f"  Top transect: {fully_inside[0]['name']}")
    elif len(origin_inside) > 0:
        logger.info(f"✓ Extract the {len(origin_inside)} transects with origins inside LiDAR extent")
        logger.info(f"  Top transect: {origin_inside[0]['name']}")
    else:
        logger.info("⚠ No transects have origins inside LiDAR extent")
        logger.info(f"  Closest transect: {results[0]['name']} ({results[0]['min_dist']:.1f}m away)")

    # Save indices of best transects
    best_indices = [r['index'] for r in results[:20]]
    best_names = [r['name'] for r in results[:20]]

    output_file = "results/mops_transects/best_transect_indices.npz"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        indices=np.array(best_indices),
        names=np.array(best_names)
    )
    logger.info(f"\n✓ Saved top 20 transect indices to {output_file}")


if __name__ == "__main__":
    main()
