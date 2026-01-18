"""Spatial filtering utilities for transect extraction.

Filters transects based on spatial overlap with LiDAR extent.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

try:
    import laspy
except ImportError:
    laspy = None

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_lidar_extent(las_path: Union[str, Path]) -> Tuple[float, float, float, float]:
    """Get the spatial extent of a LiDAR file.

    Args:
        las_path: Path to LAS/LAZ file

    Returns:
        Tuple of (min_x, max_x, min_y, max_y)

    Raises:
        ImportError: If laspy is not installed
        FileNotFoundError: If LAS file doesn't exist
    """
    if laspy is None:
        raise ImportError(
            "laspy is required. Install with: pip install laspy[lazrs]"
        )

    las_path = Path(las_path)
    if not las_path.exists():
        raise FileNotFoundError(f"LAS file not found: {las_path}")

    logger.info(f"Reading LiDAR extent from {las_path.name}")

    # Read header only for efficiency
    las = laspy.read(las_path)

    min_x = float(las.x.min())
    max_x = float(las.x.max())
    min_y = float(las.y.min())
    max_y = float(las.y.max())

    logger.info(f"  X range: [{min_x:.2f}, {max_x:.2f}]")
    logger.info(f"  Y range: [{min_y:.2f}, {max_y:.2f}]")
    logger.info(f"  Points: {len(las.x):,}")

    return min_x, max_x, min_y, max_y


def filter_transects_by_extent(
    transects: Dict[str, np.ndarray],
    lidar_extent: Tuple[float, float, float, float],
    buffer_m: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Filter transects to only those overlapping with LiDAR extent.

    Args:
        transects: Dictionary from KMLParser.parse() containing:
            - origins: (N, 3) transect start points
            - endpoints: (N, 3) transect end points
            - normals: (N, 2) shore-normal vectors
            - names: list of names
            - lengths: (N,) transect lengths
        lidar_extent: Tuple of (min_x, max_x, min_y, max_y)
        buffer_m: Buffer distance in meters around LiDAR extent (default: 50)

    Returns:
        Filtered transects dictionary with same structure

    Example:
        >>> extent = get_lidar_extent("scan.laz")
        >>> filtered = filter_transects_by_extent(kml_transects, extent)
        >>> print(f"Kept {len(filtered['origins'])} / {len(kml_transects['origins'])} transects")
    """
    min_x, max_x, min_y, max_y = lidar_extent

    # Add buffer
    min_x -= buffer_m
    max_x += buffer_m
    min_y -= buffer_m
    max_y += buffer_m

    origins = transects['origins']
    endpoints = transects['endpoints']

    # Check if transect line intersects the extent bounding box
    # A line intersects a box if:
    # 1. Either endpoint is inside the box, OR
    # 2. The line crosses any edge of the box

    n_transects = len(origins)
    overlaps = np.zeros(n_transects, dtype=bool)

    for i in range(n_transects):
        origin = origins[i, :2]
        endpoint = endpoints[i, :2]

        # Check if either endpoint is in the box
        origin_in_box = (
            (min_x <= origin[0] <= max_x) and
            (min_y <= origin[1] <= max_y)
        )
        endpoint_in_box = (
            (min_x <= endpoint[0] <= max_x) and
            (min_y <= endpoint[1] <= max_y)
        )

        if origin_in_box or endpoint_in_box:
            overlaps[i] = True
            continue

        # Check if line segment intersects box
        # Use simple bounding box check
        line_min_x = min(origin[0], endpoint[0])
        line_max_x = max(origin[0], endpoint[0])
        line_min_y = min(origin[1], endpoint[1])
        line_max_y = max(origin[1], endpoint[1])

        # Line bounding box overlaps extent bounding box
        if not (line_max_x < min_x or line_min_x > max_x or
                line_max_y < min_y or line_min_y > max_y):
            overlaps[i] = True

    n_overlap = overlaps.sum()
    logger.info(
        f"Found {n_overlap} / {n_transects} transects overlapping with LiDAR "
        f"(buffer: {buffer_m}m)"
    )

    # Filter all arrays
    filtered = {
        'origins': origins[overlaps],
        'endpoints': endpoints[overlaps],
        'normals': transects['normals'][overlaps],
        'lengths': transects['lengths'][overlaps],
        'names': [name for name, keep in zip(transects['names'], overlaps) if keep],
    }

    if n_overlap == 0:
        logger.warning(
            "No transects overlap with LiDAR data! "
            "Check that coordinate systems match."
        )
        logger.warning(f"  LiDAR extent: X=[{lidar_extent[0]:.1f}, {lidar_extent[1]:.1f}], "
                      f"Y=[{lidar_extent[2]:.1f}, {lidar_extent[3]:.1f}]")
        if n_transects > 0:
            logger.warning(f"  Transect origins X range: [{origins[:, 0].min():.1f}, {origins[:, 0].max():.1f}]")
            logger.warning(f"  Transect origins Y range: [{origins[:, 1].min():.1f}, {origins[:, 1].max():.1f}]")

    return filtered


def filter_transects_by_lidar(
    transects: Dict[str, np.ndarray],
    las_path: Union[str, Path],
    buffer_m: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Filter transects by overlap with LiDAR file (convenience function).

    Args:
        transects: Dictionary from KMLParser.parse()
        las_path: Path to LAS/LAZ file
        buffer_m: Buffer distance around LiDAR extent (default: 50m)

    Returns:
        Filtered transects dictionary
    """
    extent = get_lidar_extent(las_path)
    return filter_transects_by_extent(transects, extent, buffer_m)
