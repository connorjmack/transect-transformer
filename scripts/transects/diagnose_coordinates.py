#!/usr/bin/env python3
"""Diagnose coordinate system mismatch between KML and LiDAR.

Usage:
    python scripts/diagnose_coordinates.py
"""

from pathlib import Path
import numpy as np
import laspy

from src.data.kml_parser import KMLParser
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Diagnose coordinate systems."""

    # Paths
    kml_file = "data/mops/MOPs-SD.kml"
    lidar_file = "data/testing/20251105_00589_00639_1447_DelMar_NoWaves_beach_cliff_ground_cropped.las"

    logger.info("=" * 80)
    logger.info("COORDINATE SYSTEM DIAGNOSTIC")
    logger.info("=" * 80)

    # Parse KML
    logger.info("\n[1/2] Checking KML coordinates...")
    parser = KMLParser(utm_zone=11, hemisphere='N')
    kml_transects = parser.parse(kml_file)

    origins = kml_transects['origins']

    logger.info(f"KML transects: {len(origins)}")
    logger.info(f"KML X range: [{origins[:, 0].min():.2f}, {origins[:, 0].max():.2f}]")
    logger.info(f"KML Y range: [{origins[:, 1].min():.2f}, {origins[:, 1].max():.2f}]")
    logger.info(f"Sample origin (first transect): X={origins[0, 0]:.2f}, Y={origins[0, 1]:.2f}")

    # Check LiDAR
    logger.info("\n[2/2] Checking LiDAR coordinates...")
    las = laspy.read(lidar_file)

    logger.info(f"LiDAR points: {len(las.x):,}")
    logger.info(f"LiDAR X range: [{las.x.min():.2f}, {las.x.max():.2f}]")
    logger.info(f"LiDAR Y range: [{las.y.min():.2f}, {las.y.max():.2f}]")
    logger.info(f"LiDAR Z range: [{las.z.min():.2f}, {las.z.max():.2f}]")

    # Try to get CRS info
    try:
        if hasattr(las.header, 'parse_crs'):
            crs = las.header.parse_crs()
            logger.info(f"LiDAR CRS: {crs}")
        else:
            logger.info("LiDAR CRS: Could not parse from header")
    except Exception as e:
        logger.info(f"LiDAR CRS: Error parsing - {e}")

    # Check for overlap
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSIS")
    logger.info("=" * 80)

    kml_x_min, kml_x_max = origins[:, 0].min(), origins[:, 0].max()
    kml_y_min, kml_y_max = origins[:, 1].min(), origins[:, 1].max()

    lidar_x_min, lidar_x_max = float(las.x.min()), float(las.x.max())
    lidar_y_min, lidar_y_max = float(las.y.min()), float(las.y.max())

    # Check for overlap
    x_overlap = not (kml_x_max < lidar_x_min or kml_x_min > lidar_x_max)
    y_overlap = not (kml_y_max < lidar_y_min or kml_y_min > lidar_y_max)

    if x_overlap and y_overlap:
        logger.info("✓ Bounding boxes OVERLAP - coordinates look compatible")
    else:
        logger.warning("✗ NO OVERLAP - coordinate system mismatch detected!")
        logger.warning("")
        logger.warning("The KML and LiDAR data are in different coordinate systems.")
        logger.warning("")

        # Diagnose the issue
        if abs(kml_x_min) < 1000 and abs(lidar_x_min) > 100000:
            logger.warning("ISSUE: KML appears to be in local/offset coordinates")
            logger.warning("       LiDAR appears to be in UTM or State Plane coordinates")
            logger.warning("")
            logger.warning("SOLUTION: The KML parser may not be transforming correctly.")
            logger.warning("          The LiDAR file likely contains UTM coordinates already,")
            logger.warning("          but the KML was transformed to a different reference.")
        elif abs(kml_x_min) > 100000 and abs(lidar_x_min) > 100000:
            # Both in UTM-like coords but different zones/systems
            logger.warning("ISSUE: Both appear to be in projected coordinates")
            logger.warning("       but in different zones or systems.")
            logger.warning("")
            logger.warning("SOLUTION: Check UTM zone. San Diego is UTM Zone 11N.")
            logger.warning("          The LiDAR may be in California State Plane.")
        else:
            logger.warning("ISSUE: Coordinate system mismatch (unknown type)")
            logger.warning("")
            logger.warning("SOLUTION: Check CRS of LiDAR file with:")
            logger.warning("          pdal info <file.las> or CloudCompare")

        logger.warning("")
        logger.warning("DISTANCES:")
        logger.warning(f"  X separation: {abs(kml_x_min - lidar_x_min):.0f} meters")
        logger.warning(f"  Y separation: {abs(kml_y_min - lidar_y_min):.0f} meters")

    # Sample a few points for visual check
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE COORDINATES")
    logger.info("=" * 80)
    logger.info("First 3 KML transect origins:")
    for i in range(min(3, len(origins))):
        logger.info(f"  {kml_transects['names'][i]:15s}: X={origins[i, 0]:12.2f}, Y={origins[i, 1]:12.2f}")

    logger.info("\nFirst 3 LiDAR points:")
    for i in range(min(3, len(las.x))):
        logger.info(f"  Point {i}: X={las.x[i]:12.2f}, Y={las.y[i]:12.2f}, Z={las.z[i]:8.2f}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
