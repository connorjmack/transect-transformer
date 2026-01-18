#!/usr/bin/env python3
"""Quick-start script for extracting transects from MOPs KML file.

This is a simplified version of test_transect_extraction.py specifically
for the San Diego MOPs survey lines.

Usage:
    # Edit the LIDAR_FILE path below, then run:
    python scripts/extract_mops_transects.py
"""

from pathlib import Path
import numpy as np

from src.data.kml_parser import KMLParser
from src.data.transect_voxelizer import TransectVoxelizer
from src.data.spatial_filter import filter_transects_by_lidar
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION - Edit these paths for your data
# ============================================================================

# Path to your LiDAR file (.las or .laz)
LIDAR_FILE = "data/testing/20251105_00589_00639_1447_DelMar_NoWaves_beach_cliff_ground_cropped.las"

# Path to KML file (already uploaded)
KML_FILE = "data/mops/MOPs-SD.kml"

# Output directory
OUTPUT_DIR = "results/mops_transects/"

# UTM zone for San Diego
UTM_ZONE = 11
HEMISPHERE = 'N'

# Extraction parameters
BIN_SIZE_M = 1.0          # Size of bins along transect (meters)
CORRIDOR_WIDTH_M = 5.0    # Width of extraction corridor (meters) - INCREASED for better capture
MAX_BINS = 128            # Maximum number of bins per transect
MIN_POINTS_PER_BIN = 3    # Minimum points to form valid bin
PROFILE_LENGTH_M = 250.0  # Maximum profile length from origin (meters) - INCREASED

# For testing, limit to first N transects (set to None for all)
LIMIT_TRANSECTS = None  # or 10 for testing

# Spatial filtering
FILTER_BY_OVERLAP = True  # Only extract transects overlapping with LiDAR
BUFFER_M = 50.0           # Buffer distance around LiDAR extent (meters)

# ============================================================================


def main():
    """Extract transects from MOPs KML and LiDAR data."""

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MOPs TRANSECT EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"LiDAR file: {LIDAR_FILE}")
    logger.info(f"KML file:   {KML_FILE}")
    logger.info(f"Output:     {output_dir}")

    # Step 1: Parse KML transect lines
    logger.info("\n[1/3] Parsing KML transect lines...")

    parser = KMLParser(utm_zone=UTM_ZONE, hemisphere=HEMISPHERE)

    try:
        kml_transects = parser.parse(KML_FILE)
    except FileNotFoundError:
        logger.error(f"KML file not found: {KML_FILE}")
        return
    except Exception as e:
        logger.error(f"Error parsing KML: {e}")
        return

    logger.info(f"  ✓ Extracted {len(kml_transects['origins'])} transect lines")
    logger.info(f"  ✓ Length range: [{kml_transects['lengths'].min():.1f}, "
                f"{kml_transects['lengths'].max():.1f}]m")

    # Filter by spatial overlap with LiDAR
    if FILTER_BY_OVERLAP:
        logger.info(f"\n  → Filtering transects by LiDAR overlap (buffer: {BUFFER_M}m)...")
        try:
            kml_transects = filter_transects_by_lidar(
                kml_transects,
                LIDAR_FILE,
                buffer_m=BUFFER_M
            )
        except FileNotFoundError:
            logger.error(f"LiDAR file not found: {LIDAR_FILE}")
            return
        except Exception as e:
            logger.error(f"Error filtering transects: {e}")
            return

        if len(kml_transects['origins']) == 0:
            logger.error("No transects overlap with LiDAR data!")
            logger.error("Check that KML and LiDAR use compatible coordinate systems.")
            return

    # Limit transects if requested
    if LIMIT_TRANSECTS is not None:
        n = min(LIMIT_TRANSECTS, len(kml_transects['origins']))
        logger.info(f"  → Limiting to first {n} transects for testing")
        kml_transects = {
            'origins': kml_transects['origins'][:n],
            'endpoints': kml_transects['endpoints'][:n],
            'normals': kml_transects['normals'][:n],
            'names': kml_transects['names'][:n],
            'lengths': kml_transects['lengths'][:n],
        }

    # Save KML transects (after filtering)
    kml_save_path = output_dir / "kml_transects_filtered.npz"
    parser.save_transects(kml_transects, kml_save_path)
    logger.info(f"  ✓ Saved {len(kml_transects['origins'])} filtered transects to {kml_save_path}")

    # Step 2: Extract voxelized transects from LiDAR
    logger.info("\n[2/3] Extracting voxelized transects from LiDAR...")
    logger.info(f"  → Processing {len(kml_transects['origins'])} transects...")

    voxelizer = TransectVoxelizer(
        bin_size_m=BIN_SIZE_M,
        corridor_width_m=CORRIDOR_WIDTH_M,
        max_bins=MAX_BINS,
        min_points_per_bin=MIN_POINTS_PER_BIN,
        profile_length_m=PROFILE_LENGTH_M,
    )

    try:
        transects = voxelizer.extract_from_file(
            LIDAR_FILE,
            transect_origins=kml_transects['origins'],
            transect_normals=kml_transects['normals'],
            transect_names=kml_transects['names'],
        )
    except FileNotFoundError:
        logger.error(f"LiDAR file not found: {LIDAR_FILE}")
        logger.error("Please edit LIDAR_FILE in this script to point to your .las/.laz file")
        return
    except Exception as e:
        logger.error(f"Error extracting transects: {e}")
        return

    logger.info(f"  ✓ Extracted {len(transects['bin_features'])} transects")
    logger.info(f"  ✓ Output shape: {transects['bin_features'].shape}")
    logger.info(f"  ✓ Valid bins: {transects['bin_mask'].sum():,} / "
                f"{transects['bin_mask'].size:,} "
                f"({100*transects['bin_mask'].mean():.1f}%)")

    # Step 3: Save results
    logger.info("\n[3/3] Saving results...")

    save_path = output_dir / "transects_voxelized.npz"
    voxelizer.save_transects(transects, save_path)
    logger.info(f"  ✓ Saved to {save_path}")

    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    valid_features = transects['bin_features'][transects['bin_mask']]

    if len(valid_features) > 0:
        mean_elev = valid_features[:, 0]
        roughness = valid_features[:, 1]
        height_range = valid_features[:, 2]
        slope = valid_features[:, 3]
        point_density = valid_features[:, 5]

        logger.info(f"Elevation:      [{mean_elev.min():7.2f}, {mean_elev.max():7.2f}]m "
                    f"(mean: {mean_elev.mean():.2f}m)")
        logger.info(f"Roughness:      [{roughness.min():7.4f}, {roughness.max():7.4f}]m "
                    f"(mean: {roughness.mean():.4f}m)")
        logger.info(f"Height range:   [{height_range.min():7.2f}, {height_range.max():7.2f}]m "
                    f"(mean: {height_range.mean():.2f}m)")
        logger.info(f"Slope:          [{slope.min():7.2f}, {slope.max():7.2f}]° "
                    f"(mean: {slope.mean():.2f}°)")
        logger.info(f"Point density:  [{point_density.min():7.2f}, {point_density.max():7.2f}] pts/m³")

    metadata = transects['metadata']
    if len(metadata) > 0:
        cliff_heights = metadata[:, 0]
        mean_slopes = metadata[:, 1]
        logger.info(f"\nTransect metadata:")
        logger.info(f"Cliff height:   [{cliff_heights.min():7.2f}, {cliff_heights.max():7.2f}]m "
                    f"(mean: {cliff_heights.mean():.2f}m)")
        logger.info(f"Mean slope:     [{mean_slopes.min():7.2f}, {mean_slopes.max():7.2f}]° "
                    f"(mean: {mean_slopes.mean():.2f}°)")

    logger.info("\n" + "=" * 80)
    logger.info("DONE!")
    logger.info("=" * 80)
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info("Next steps:")
    logger.info("  1. Inspect output with: np.load('{}')".format(save_path))
    logger.info("  2. Visualize sample transects")
    logger.info("  3. Use in PyTorch dataset (Phase 1.7)")


if __name__ == "__main__":
    main()
