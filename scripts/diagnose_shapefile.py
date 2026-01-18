#!/usr/bin/env python3
"""Diagnose why shapefile transects aren't extracting data.

This script checks:
1. Shapefile CRS and coordinate ranges
2. LiDAR CRS and coordinate ranges
3. Whether transects actually overlap LiDAR extent
4. Transect normal directions
5. Sample extraction debug info
"""

from pathlib import Path
import numpy as np
import laspy

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("ERROR: geopandas not installed. Install with: pip install geopandas")
    exit(1)

from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Diagnose shapefile and LiDAR compatibility."""

    # Paths
    shp_file = "data/mops/DelMarTransects595to620at1m/DelMarTransects595to620at1m.shp"
    lidar_file = "data/testing/20251105_00589_00639_1447_DelMar_NoWaves_beach_cliff_ground_cropped.las"

    logger.info("=" * 80)
    logger.info("SHAPEFILE & LIDAR DIAGNOSTIC")
    logger.info("=" * 80)

    # ========================================================================
    # 1. Check shapefile
    # ========================================================================
    logger.info("\n[1] SHAPEFILE ANALYSIS")
    logger.info("-" * 80)

    gdf = gpd.read_file(shp_file)

    logger.info(f"File: {shp_file}")
    logger.info(f"CRS: {gdf.crs}")
    logger.info(f"Number of features: {len(gdf)}")
    logger.info(f"Geometry types: {gdf.geom_type.unique()}")

    # Get bounds
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    logger.info(f"\nBounds:")
    logger.info(f"  X: [{bounds[0]:.2f}, {bounds[2]:.2f}]")
    logger.info(f"  Y: [{bounds[1]:.2f}, {bounds[3]:.2f}]")

    # Sample some transects
    logger.info(f"\nFirst 5 transects:")
    for idx in range(min(5, len(gdf))):
        row = gdf.iloc[idx]
        geom = row.geometry
        coords = list(geom.coords)

        # Try to get name
        name = None
        for field in ['name', 'Name', 'NAME', 'id', 'ID', 'FID']:
            if field in row:
                name = row[field]
                break
        if name is None:
            name = f"idx_{idx}"

        start = coords[0]
        end = coords[-1]

        logger.info(f"  {name}:")
        logger.info(f"    Start: ({start[0]:.2f}, {start[1]:.2f})")
        logger.info(f"    End:   ({end[0]:.2f}, {end[1]:.2f})")

        # Compute normal
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        normal_x = dx / length
        normal_y = dy / length
        logger.info(f"    Length: {length:.2f}m")
        logger.info(f"    Normal: ({normal_x:.4f}, {normal_y:.4f})")

    # Check all attribute fields
    logger.info(f"\nAvailable fields: {list(gdf.columns)}")

    # ========================================================================
    # 2. Check LiDAR
    # ========================================================================
    logger.info("\n[2] LIDAR ANALYSIS")
    logger.info("-" * 80)

    las = laspy.read(lidar_file)

    logger.info(f"File: {lidar_file}")
    logger.info(f"Point count: {len(las.points):,}")
    logger.info(f"Point format: {las.point_format}")

    # Check for CRS info
    if hasattr(las, 'header') and hasattr(las.header, 'parse_crs'):
        try:
            las_crs = las.header.parse_crs()
            logger.info(f"CRS: {las_crs}")
        except:
            logger.warning("Could not parse CRS from LAS header")
    else:
        logger.warning("No CRS information in LAS file")

    # Get extent
    lidar_x_min, lidar_x_max = float(las.x.min()), float(las.x.max())
    lidar_y_min, lidar_y_max = float(las.y.min()), float(las.y.max())
    lidar_z_min, lidar_z_max = float(las.z.min()), float(las.z.max())

    logger.info(f"\nExtent:")
    logger.info(f"  X: [{lidar_x_min:.2f}, {lidar_x_max:.2f}]")
    logger.info(f"  Y: [{lidar_y_min:.2f}, {lidar_y_max:.2f}]")
    logger.info(f"  Z: [{lidar_z_min:.2f}, {lidar_z_max:.2f}]")

    lidar_center_x = (lidar_x_min + lidar_x_max) / 2
    lidar_center_y = (lidar_y_min + lidar_y_max) / 2
    logger.info(f"  Center: ({lidar_center_x:.2f}, {lidar_center_y:.2f})")

    # ========================================================================
    # 3. Check overlap
    # ========================================================================
    logger.info("\n[3] OVERLAP ANALYSIS")
    logger.info("-" * 80)

    # Check if shapefile bounds overlap with LiDAR extent
    shp_overlaps_x = not (bounds[2] < lidar_x_min or bounds[0] > lidar_x_max)
    shp_overlaps_y = not (bounds[3] < lidar_y_min or bounds[1] > lidar_y_max)

    logger.info(f"Shapefile X overlaps LiDAR: {shp_overlaps_x}")
    logger.info(f"Shapefile Y overlaps LiDAR: {shp_overlaps_y}")

    if not (shp_overlaps_x and shp_overlaps_y):
        logger.error("\n❌ COORDINATE MISMATCH DETECTED!")
        logger.error("Shapefile and LiDAR extents DO NOT overlap!")
        logger.error("\nPossible issues:")
        logger.error("1. Different coordinate reference systems (CRS)")
        logger.error("2. Different units (meters vs feet)")
        logger.error("3. Shapefile needs reprojection")

        # Check if it's a units issue
        if bounds[0] > 1e6 or lidar_x_min > 1e6:
            logger.error("\n⚠ Coordinates are very large - might be in feet instead of meters")
            logger.error("Or might be in a different UTM zone")
    else:
        logger.info("\n✓ Shapefile and LiDAR extents DO overlap")

        # Check how many transect start/end points are inside LiDAR extent
        starts_inside = 0
        ends_inside = 0

        for idx, row in gdf.iterrows():
            geom = row.geometry
            coords = list(geom.coords)
            start = coords[0]
            end = coords[-1]

            start_in = (lidar_x_min <= start[0] <= lidar_x_max and
                       lidar_y_min <= start[1] <= lidar_y_max)
            end_in = (lidar_x_min <= end[0] <= lidar_x_max and
                     lidar_y_min <= end[1] <= lidar_y_max)

            if start_in:
                starts_inside += 1
            if end_in:
                ends_inside += 1

        logger.info(f"\nTransect points inside LiDAR extent:")
        logger.info(f"  Start points: {starts_inside} / {len(gdf)}")
        logger.info(f"  End points:   {ends_inside} / {len(gdf)}")

        if starts_inside == 0 and ends_inside == 0:
            logger.warning("\n⚠ WARNING: No transect start/end points are inside LiDAR extent!")
            logger.warning("Transects may pass through LiDAR area but not start/end there.")

    # ========================================================================
    # 4. Debug sample extraction
    # ========================================================================
    logger.info("\n[4] SAMPLE EXTRACTION DEBUG")
    logger.info("-" * 80)

    # Get first transect
    row = gdf.iloc[0]
    geom = row.geometry
    coords = list(geom.coords)
    start = np.array([coords[0][0], coords[0][1], 0.0])
    end = np.array([coords[-1][0], coords[-1][1], 0.0])

    # Compute normal
    vector = end - start
    length = np.linalg.norm(vector[:2])
    normal = vector[:2] / length

    logger.info(f"Testing extraction for first transect:")
    logger.info(f"  Origin: ({start[0]:.2f}, {start[1]:.2f})")
    logger.info(f"  Normal: ({normal[0]:.4f}, {normal[1]:.4f})")
    logger.info(f"  Profile length to sample: 250m")

    # Simulate bin locations
    bin_size = 1.0
    n_bins = 128
    corridor_width = 5.0

    logger.info(f"\nSampling parameters:")
    logger.info(f"  Bin size: {bin_size}m")
    logger.info(f"  Number of bins: {n_bins}")
    logger.info(f"  Corridor width: {corridor_width}m")

    # Sample some bin centers
    logger.info(f"\nFirst 5 bin centers along transect:")
    for i in range(5):
        distance = i * bin_size
        bin_center_x = start[0] + distance * normal[0]
        bin_center_y = start[1] + distance * normal[1]

        # Check if bin center is in LiDAR extent
        in_extent = (lidar_x_min <= bin_center_x <= lidar_x_max and
                    lidar_y_min <= bin_center_y <= lidar_y_max)

        logger.info(f"  Bin {i} @ {distance}m: ({bin_center_x:.2f}, {bin_center_y:.2f}) - {'✓ IN' if in_extent else '✗ OUT'}")

    # Check if we're going the right direction
    # Sample at 100m
    far_x = start[0] + 100 * normal[0]
    far_y = start[1] + 100 * normal[1]
    far_in = (lidar_x_min <= far_x <= lidar_x_max and
             lidar_y_min <= far_y <= lidar_y_max)

    logger.info(f"\nPoint 100m along normal: ({far_x:.2f}, {far_y:.2f}) - {'✓ IN' if far_in else '✗ OUT'}")

    # Try going backwards
    back_x = start[0] - 100 * normal[0]
    back_y = start[1] - 100 * normal[1]
    back_in = (lidar_x_min <= back_x <= lidar_x_max and
              lidar_y_min <= back_y <= lidar_y_max)

    logger.info(f"Point 100m OPPOSITE to normal: ({back_x:.2f}, {back_y:.2f}) - {'✓ IN' if back_in else '✗ OUT'}")

    if back_in and not far_in:
        logger.warning("\n⚠ DIRECTION ISSUE DETECTED!")
        logger.warning("Sampling opposite to normal direction finds LiDAR points!")
        logger.warning("The transect normals may be pointing the wrong way.")
        logger.warning("Expected: origin on land, normal points seaward")
        logger.warning("Actual: origin may be on water, normal points landward")
        logger.warning("\nSOLUTION: Reverse the transect normals!")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSIS SUMMARY")
    logger.info("=" * 80)

    if shp_overlaps_x and shp_overlaps_y:
        logger.info("✓ Coordinate systems appear compatible")
        if starts_inside > 0 or ends_inside > 0:
            logger.info("✓ Some transect points are inside LiDAR extent")
            if back_in and not far_in:
                logger.error("✗ TRANSECT NORMALS ARE POINTING THE WRONG DIRECTION")
                logger.error("\nRecommended fix:")
                logger.error("  Add a --reverse-normals flag to the extraction script")
                logger.error("  Or flip the start/end points in the shapefile parser")
            else:
                logger.warning("⚠ Unclear why extraction is failing")
                logger.warning("  Check corridor width and profile length settings")
        else:
            logger.warning("⚠ No transect endpoints inside LiDAR (but bounds overlap)")
            logger.warning("  Transects may only partially intersect LiDAR area")
    else:
        logger.error("✗ COORDINATE SYSTEM MISMATCH")
        logger.error("  Shapefile and LiDAR do not overlap spatially")


if __name__ == "__main__":
    main()
