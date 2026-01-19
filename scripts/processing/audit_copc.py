#!/usr/bin/env python3
"""Audit a COPC file to verify it's usable for transect extraction.

Tests:
1. File exists and is readable
2. Has valid COPC spatial index (VLR check)
3. Spatial queries work correctly
4. Has required attributes for transect extraction
5. Reports file statistics

Usage:
    python scripts/processing/audit_copc.py /path/to/file.copc.laz

    # With custom spatial query test bounds
    python scripts/processing/audit_copc.py /path/to/file.copc.laz --test-bounds 476000 3631000 477000 3632000
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import time

import numpy as np

# Required attributes for transect extraction (from CLAUDE.md)
REQUIRED_ATTRIBUTES = ['x', 'y', 'z', 'intensity']
OPTIONAL_ATTRIBUTES = ['red', 'green', 'blue', 'classification', 'return_number', 'number_of_returns']


def check_file_exists(path: Path) -> Tuple[bool, str]:
    """Check if file exists and is readable."""
    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Not a file: {path}"
    size_mb = path.stat().st_size / (1024 * 1024)
    return True, f"File exists ({size_mb:.1f} MB)"


def check_copc_vlr(path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Check if file has COPC VLR (Variable Length Record)."""
    try:
        import laspy
    except ImportError:
        return False, "laspy not installed", {}

    info = {}
    try:
        with laspy.open(str(path)) as f:
            header = f.header
            info['point_count'] = header.point_count
            info['point_format'] = header.point_format.id
            info['version'] = f"{header.version.major}.{header.version.minor}"
            info['mins'] = (header.mins[0], header.mins[1], header.mins[2])
            info['maxs'] = (header.maxs[0], header.maxs[1], header.maxs[2])
            info['scales'] = header.scales
            info['offsets'] = header.offsets

            # Check for COPC VLR
            has_copc = False
            for vlr in header.vlrs:
                if vlr.user_id == 'copc':
                    has_copc = True
                    break

            if has_copc:
                return True, "Valid COPC file (has 'copc' VLR)", info
            else:
                vlr_ids = [vlr.user_id for vlr in header.vlrs]
                return False, f"No COPC VLR found. VLRs present: {vlr_ids}", info
    except Exception as e:
        return False, f"Error reading header: {e}", info


def check_copc_reader(path: Path) -> Tuple[bool, str]:
    """Check if laspy CopcReader can open the file."""
    try:
        from laspy import CopcReader
    except ImportError:
        return False, "CopcReader not available (need laspy >= 2.5)"

    try:
        reader = CopcReader.open(str(path))
        info = reader.copc_info
        # Get root octree info
        center = info.center
        halfsize = info.halfsize
        spacing = info.spacing
        return True, f"CopcReader works. Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), halfsize: {halfsize:.1f}, spacing: {spacing:.3f}"
    except Exception as e:
        return False, f"CopcReader failed: {e}"


def check_spatial_query(path: Path, bounds: Optional[Tuple[float, float, float, float]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Test spatial query functionality."""
    try:
        from laspy import CopcReader
        from laspy.copc import Bounds
    except ImportError:
        return False, "CopcReader not available", {}

    info = {}
    try:
        reader = CopcReader.open(str(path))

        # Get file bounds from header
        header_mins = reader.header.mins
        header_maxs = reader.header.maxs

        # If no bounds provided, query a small region in the center
        if bounds is None:
            center_x = (header_mins[0] + header_maxs[0]) / 2
            center_y = (header_mins[1] + header_maxs[1]) / 2
            # Query 100m x 100m box around center
            query_size = 50.0
            bounds = (
                center_x - query_size,
                center_y - query_size,
                center_x + query_size,
                center_y + query_size
            )

        x_min, y_min, x_max, y_max = bounds
        info['query_bounds'] = bounds

        # Time the spatial query - use Bounds object with 2D (laspy handles z)
        query_bounds = Bounds(
            mins=np.array([x_min, y_min]),
            maxs=np.array([x_max, y_max])
        )
        start = time.perf_counter()
        points = reader.query(bounds=query_bounds)
        elapsed = time.perf_counter() - start

        info['query_time_sec'] = elapsed
        info['points_returned'] = len(points)

        if len(points) == 0:
            return True, f"Spatial query works but returned 0 points (query: {bounds})", info

        return True, f"Spatial query returned {len(points):,} points in {elapsed:.3f}s", info

    except Exception as e:
        return False, f"Spatial query failed: {e}", info


def check_attributes(path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Check if file has required attributes for transect extraction."""
    try:
        from laspy import CopcReader
        from laspy.copc import Bounds
    except ImportError:
        return False, "CopcReader not available", {}

    info = {}
    try:
        reader = CopcReader.open(str(path))

        # Get file bounds - use center with progressively larger areas
        header_mins = reader.header.mins
        header_maxs = reader.header.maxs
        center_x = (header_mins[0] + header_maxs[0]) / 2
        center_y = (header_mins[1] + header_maxs[1]) / 2

        # Try progressively larger areas until we get points
        sample_sizes = [10, 50, 100, 500]
        points = None

        for size in sample_sizes:
            query_bounds = Bounds(
                mins=np.array([center_x - size, center_y - size]),
                maxs=np.array([center_x + size, center_y + size])
            )
            points = reader.query(bounds=query_bounds)
            if len(points) > 0:
                info['sample_size'] = size
                break

        if points is None or len(points) == 0:
            return False, "No points found in any sample region", info

        # Check which attributes are present
        available = []
        missing_required = []
        missing_optional = []

        for attr in REQUIRED_ATTRIBUTES:
            if hasattr(points, attr):
                available.append(attr)
            else:
                missing_required.append(attr)

        for attr in OPTIONAL_ATTRIBUTES:
            if hasattr(points, attr):
                available.append(attr)
            else:
                missing_optional.append(attr)

        info['available_attributes'] = available
        info['missing_required'] = missing_required
        info['missing_optional'] = missing_optional

        # Also get point format dimension names
        info['point_format_dimensions'] = list(points.point_format.dimension_names)

        if missing_required:
            return False, f"Missing required attributes: {missing_required}", info

        # Check attribute value ranges
        ranges = {}
        for attr in ['x', 'y', 'z', 'intensity']:
            if hasattr(points, attr):
                data = np.asarray(getattr(points, attr))
                ranges[attr] = {
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'dtype': str(data.dtype)
                }

        # Check RGB if present
        if hasattr(points, 'red'):
            red = np.asarray(points.red)
            ranges['red'] = {'min': float(red.min()), 'max': float(red.max())}
            # Check if 8-bit or 16-bit RGB
            info['rgb_scale'] = '16-bit' if red.max() > 255 else '8-bit'

        info['value_ranges'] = ranges

        return True, f"Has all required attributes. Available: {available}", info

    except Exception as e:
        return False, f"Attribute check failed: {e}", info


def check_extractor_loading(path: Path, bounds: Optional[Tuple[float, float, float, float]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Test that the transect extractor's COPC loading works."""
    info = {}
    try:
        import sys
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from src.data.shapefile_transect_extractor import ShapefileTransectExtractor
    except ImportError as e:
        return False, f"Could not import ShapefileTransectExtractor: {e}", info

    try:
        # Create extractor with default settings
        extractor = ShapefileTransectExtractor(
            buffer_m=1.0,
            n_points=128
        )

        # Get file bounds if not provided
        if bounds is None:
            try:
                import laspy
                with laspy.open(str(path)) as f:
                    header_mins = f.header.mins
                    header_maxs = f.header.maxs
                    center_x = (header_mins[0] + header_maxs[0]) / 2
                    center_y = (header_mins[1] + header_maxs[1]) / 2
                    # Use 100m x 100m box around center
                    bounds = (center_x - 50, center_y - 50, center_x + 50, center_y + 50)
            except Exception:
                return False, "Could not determine bounds from file", info

        info['test_bounds'] = bounds

        # Time the load
        start = time.perf_counter()
        result = extractor._load_copc_spatial(path, bounds)
        elapsed = time.perf_counter() - start
        info['load_time'] = elapsed

        if result is None:
            return False, "Extractor _load_copc_spatial returned None", info

        # Check the result format
        info['n_points'] = len(result['xyz'])
        info['features'] = list(result.keys())

        # Verify expected features are present
        expected_features = ['xyz', 'x', 'y', 'z', 'intensity', 'red', 'green', 'blue',
                           'classification', 'return_number', 'num_returns']
        missing = [f for f in expected_features if f not in result]
        if missing:
            return False, f"Missing features in result: {missing}", info

        # Verify no NaN values
        for key in ['xyz', 'intensity', 'red', 'green', 'blue']:
            if key in result and np.any(np.isnan(result[key])):
                return False, f"NaN values found in {key}", info

        return True, f"Extractor loaded {info['n_points']:,} points in {elapsed:.3f}s", info

    except Exception as e:
        return False, f"Extractor loading failed: {e}", info


def run_full_audit(path: Path, test_bounds: Optional[Tuple[float, float, float, float]] = None) -> bool:
    """Run full audit and print results."""
    print(f"\n{'='*70}")
    print(f"COPC FILE AUDIT")
    print(f"{'='*70}")
    print(f"File: {path}")
    print(f"{'='*70}\n")

    all_passed = True

    # Test 1: File exists
    print("1. FILE EXISTS CHECK")
    ok, msg = check_file_exists(path)
    status = "PASS" if ok else "FAIL"
    print(f"   [{status}] {msg}")
    if not ok:
        all_passed = False
        print("\n   Cannot continue - file not found")
        return False
    print()

    # Test 2: COPC VLR check
    print("2. COPC VLR CHECK")
    ok, msg, info = check_copc_vlr(path)
    status = "PASS" if ok else "FAIL"
    print(f"   [{status}] {msg}")
    if info:
        print(f"   Point count: {info.get('point_count', 'N/A'):,}")
        print(f"   Point format: {info.get('point_format', 'N/A')}")
        print(f"   LAS version: {info.get('version', 'N/A')}")
        if 'mins' in info:
            print(f"   X range: {info['mins'][0]:.2f} to {info['maxs'][0]:.2f}")
            print(f"   Y range: {info['mins'][1]:.2f} to {info['maxs'][1]:.2f}")
            print(f"   Z range: {info['mins'][2]:.2f} to {info['maxs'][2]:.2f}")
    if not ok:
        all_passed = False
    print()

    # Test 3: CopcReader works
    print("3. COPC READER CHECK")
    ok, msg = check_copc_reader(path)
    status = "PASS" if ok else "FAIL"
    print(f"   [{status}] {msg}")
    if not ok:
        all_passed = False
        print("\n   Cannot continue - CopcReader failed")
        return False
    print()

    # Test 4: Spatial query
    print("4. SPATIAL QUERY CHECK")
    ok, msg, info = check_spatial_query(path, test_bounds)
    status = "PASS" if ok else "FAIL"
    print(f"   [{status}] {msg}")
    if info:
        if 'query_bounds' in info:
            b = info['query_bounds']
            print(f"   Query bounds: ({b[0]:.1f}, {b[1]:.1f}) to ({b[2]:.1f}, {b[3]:.1f})")
        if 'points_returned' in info and info['points_returned'] > 0:
            print(f"   Query speed: {info['points_returned'] / info['query_time_sec']:.0f} pts/sec")
    if not ok:
        all_passed = False
    print()

    # Test 5: Attributes
    print("5. ATTRIBUTE CHECK")
    ok, msg, info = check_attributes(path)
    status = "PASS" if ok else "FAIL"
    print(f"   [{status}] {msg}")
    if info:
        if 'point_format_dimensions' in info:
            print(f"   Point format dimensions: {info['point_format_dimensions']}")
        if 'missing_optional' in info and info['missing_optional']:
            print(f"   Missing optional: {info['missing_optional']}")
        if 'rgb_scale' in info:
            print(f"   RGB scale: {info['rgb_scale']}")
        if 'value_ranges' in info:
            print(f"   Value ranges:")
            for attr, ranges in info['value_ranges'].items():
                print(f"      {attr}: {ranges['min']:.2f} to {ranges['max']:.2f} ({ranges.get('dtype', '')})")
    if not ok:
        all_passed = False
    print()

    # Test 6: Transect extractor COPC loading
    print("6. TRANSECT EXTRACTOR COPC LOADING")
    ok, msg, load_info = check_extractor_loading(path, test_bounds)
    status = "PASS" if ok else "FAIL"
    print(f"   [{status}] {msg}")
    if load_info:
        if 'n_points' in load_info:
            print(f"   Points loaded: {load_info['n_points']:,}")
        if 'features' in load_info:
            print(f"   Features: {load_info['features']}")
        if 'load_time' in load_info:
            print(f"   Load time: {load_info['load_time']:.3f}s")
    if not ok:
        all_passed = False
    print()

    # Summary
    print(f"{'='*70}")
    if all_passed:
        print("AUDIT RESULT: PASS - File is ready for transect extraction")
    else:
        print("AUDIT RESULT: FAIL - Some checks failed, see above")
    print(f"{'='*70}\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Audit a COPC file for transect extraction compatibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'copc_file',
        type=Path,
        help='Path to COPC file to audit'
    )

    parser.add_argument(
        '--test-bounds',
        type=float,
        nargs=4,
        metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'),
        help='Custom bounds for spatial query test'
    )

    args = parser.parse_args()

    bounds = tuple(args.test_bounds) if args.test_bounds else None

    success = run_full_audit(args.copc_file, bounds)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
