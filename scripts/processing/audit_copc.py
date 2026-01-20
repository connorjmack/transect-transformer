#!/usr/bin/env python3
"""Audit COPC files to verify they're usable for transect extraction.

Tests:
1. File exists and is readable
2. Has valid COPC spatial index (VLR check)
3. Spatial queries work correctly
4. Has required attributes for transect extraction
5. Reports file statistics

Integrity checks (--check-integrity):
- Validates LAS magic bytes ("LASF")
- Compares header point count vs actual readable points
- Checks file size is reasonable (>1MB)
- Detects partially written / corrupted files

Usage:
    # Audit single file
    python scripts/processing/audit_copc.py /path/to/file.copc.laz

    # With custom spatial query test bounds
    python scripts/processing/audit_copc.py /path/to/file.copc.laz --test-bounds 476000 3631000 477000 3632000

    # Batch integrity check all COPC files from master_list.csv
    python scripts/processing/audit_copc.py --survey-csv data/raw/master_list.csv --check-integrity

    # Just list corrupt files (quiet mode)
    python scripts/processing/audit_copc.py --survey-csv data/raw/master_list.csv --check-integrity --quiet
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import time

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# Add project root for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.processing.extract_transects import convert_path_for_os, get_current_os
    HAS_PATH_CONVERTER = True
except ImportError:
    HAS_PATH_CONVERTER = False

# Required attributes for transect extraction (from CLAUDE.md)
REQUIRED_ATTRIBUTES = ['x', 'y', 'z', 'intensity']
OPTIONAL_ATTRIBUTES = ['red', 'green', 'blue', 'classification', 'return_number', 'number_of_returns']

# Minimum file size for valid COPC (1MB) - partial writes are usually much smaller
MIN_VALID_SIZE_BYTES = 1 * 1024 * 1024

# LAS file magic bytes
LAS_MAGIC = b'LASF'


def get_copc_path(las_path: Path) -> Path:
    """Get the COPC output path for a LAS file (mirrors make_copc.py)."""
    stem = las_path.stem
    if stem.endswith('.copc'):
        return las_path
    return las_path.with_name(stem + '.copc.laz')


def check_magic_bytes(path: Path) -> Tuple[bool, str]:
    """Check if file has valid LAS magic bytes ('LASF')."""
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
        if magic == LAS_MAGIC:
            return True, "Valid LAS magic bytes (LASF)"
        else:
            return False, f"Invalid magic bytes: {magic!r} (expected b'LASF')"
    except Exception as e:
        return False, f"Could not read magic bytes: {e}"


def check_file_size(path: Path, min_size: int = MIN_VALID_SIZE_BYTES) -> Tuple[bool, str]:
    """Check if file size is above minimum threshold."""
    try:
        size = path.stat().st_size
        size_mb = size / (1024 * 1024)
        if size >= min_size:
            return True, f"File size OK ({size_mb:.2f} MB)"
        else:
            return False, f"File too small ({size_mb:.2f} MB < {min_size / (1024*1024):.1f} MB minimum)"
    except Exception as e:
        return False, f"Could not check file size: {e}"


def check_point_count_integrity(path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Check if header point count matches actual readable points.

    This detects truncated files where the header was written but point data is incomplete.
    """
    info = {}
    try:
        import laspy
    except ImportError:
        return False, "laspy not installed", info

    try:
        with laspy.open(str(path)) as f:
            header_count = f.header.point_count
            info['header_point_count'] = header_count

            # For very large files, we can't count all points efficiently
            # Instead, try to read a chunk from the end of the file
            if header_count > 10_000_000:
                # For large files, just verify we can read points from various offsets
                try:
                    # Read first chunk
                    chunk_size = min(100_000, header_count)
                    points = f.read_points(chunk_size)
                    if len(points) != chunk_size:
                        return False, f"Could not read expected points from start (got {len(points)}, expected {chunk_size})", info

                    # Seek to near end and read
                    if header_count > chunk_size * 2:
                        # Note: laspy doesn't support seeking, so for large files we do a sampling approach
                        info['validation_method'] = 'start_chunk_only'
                        return True, f"Large file ({header_count:,} pts) - verified start chunk readable", info

                except Exception as e:
                    return False, f"Failed to read points: {e}", info
            else:
                # For smaller files, count all points
                actual_count = 0
                for points in f.chunk_iterator(1_000_000):
                    actual_count += len(points)

                info['actual_point_count'] = actual_count
                info['validation_method'] = 'full_count'

                if actual_count == header_count:
                    return True, f"Point count matches header ({actual_count:,} points)", info
                else:
                    diff = header_count - actual_count
                    pct = (diff / header_count) * 100 if header_count > 0 else 0
                    return False, f"Point count mismatch: header={header_count:,}, actual={actual_count:,} (missing {diff:,}, {pct:.1f}%)", info

    except Exception as e:
        return False, f"Error checking point count: {e}", info


def run_integrity_check(path: Path, verbose: bool = True) -> Tuple[bool, List[str]]:
    """Run quick integrity checks on a COPC file.

    Returns:
        Tuple of (passed: bool, issues: List[str])
    """
    issues = []

    # Check 1: File exists
    if not path.exists():
        return False, ["File not found"]

    # Check 2: File size
    ok, msg = check_file_size(path)
    if not ok:
        issues.append(f"SIZE: {msg}")

    # Check 3: Magic bytes
    ok, msg = check_magic_bytes(path)
    if not ok:
        issues.append(f"MAGIC: {msg}")
        # If magic bytes are wrong, file is definitely corrupt
        return False, issues

    # Check 4: COPC VLR
    ok, msg, _ = check_copc_vlr(path)
    if not ok:
        issues.append(f"VLR: {msg}")

    # Check 5: Point count integrity (skip if VLR already failed)
    if not issues or verbose:
        ok, msg, _ = check_point_count_integrity(path)
        if not ok:
            issues.append(f"POINTS: {msg}")

    passed = len(issues) == 0
    return passed, issues


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


def run_batch_integrity_check(copc_paths: List[Path], quiet: bool = False) -> Tuple[int, int, List[Tuple[Path, List[str]]]]:
    """Run integrity checks on multiple COPC files.

    Args:
        copc_paths: List of COPC file paths to check
        quiet: If True, suppress progress output

    Returns:
        Tuple of (passed_count, failed_count, failed_files_with_issues)
    """
    passed = 0
    failed = 0
    failed_files: List[Tuple[Path, List[str]]] = []

    if HAS_TQDM and not quiet:
        iterator = tqdm(copc_paths, desc="Checking integrity", unit="file")
    else:
        iterator = copc_paths

    for copc_path in iterator:
        ok, issues = run_integrity_check(copc_path, verbose=False)
        if ok:
            passed += 1
        else:
            failed += 1
            failed_files.append((copc_path, issues))
            if not quiet and not HAS_TQDM:
                print(f"  CORRUPT: {copc_path.name}")

    return passed, failed, failed_files


def main():
    parser = argparse.ArgumentParser(
        description='Audit COPC files for transect extraction compatibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input sources
    parser.add_argument(
        'copc_file',
        type=Path,
        nargs='?',
        help='Path to COPC file to audit (for single-file mode)'
    )

    parser.add_argument(
        '--survey-csv',
        type=Path,
        help='Path to master_list.csv - checks COPC versions of all LAS files listed'
    )

    parser.add_argument(
        '--path-col',
        type=str,
        default='full_path',
        help='Column name for LAS paths in CSV (default: full_path)'
    )

    parser.add_argument(
        '--target-os',
        type=str,
        choices=['mac', 'linux'],
        default=None,
        help='Target OS for path conversion (default: auto-detect)'
    )

    parser.add_argument(
        '--check-integrity',
        action='store_true',
        help='Run quick integrity checks only (magic bytes, size, VLR, point count)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode - only output corrupt file paths (useful for piping)'
    )

    parser.add_argument(
        '--test-bounds',
        type=float,
        nargs=4,
        metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'),
        help='Custom bounds for spatial query test (single-file mode only)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Write list of corrupt files to this path (one per line)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.copc_file and not args.survey_csv:
        parser.error("Either copc_file or --survey-csv is required")

    # Single file mode
    if args.copc_file and not args.survey_csv:
        if args.check_integrity:
            # Quick integrity check only
            ok, issues = run_integrity_check(args.copc_file, verbose=True)
            if ok:
                if not args.quiet:
                    print(f"PASS: {args.copc_file}")
                return 0
            else:
                if args.quiet:
                    print(str(args.copc_file))
                else:
                    print(f"FAIL: {args.copc_file}")
                    for issue in issues:
                        print(f"  - {issue}")
                return 1
        else:
            # Full audit
            bounds = tuple(args.test_bounds) if args.test_bounds else None
            success = run_full_audit(args.copc_file, bounds)
            return 0 if success else 1

    # Batch mode with --survey-csv
    if args.survey_csv:
        if not HAS_PANDAS:
            print("Error: pandas required for CSV mode. Install with: pip install pandas")
            return 1

        if not args.survey_csv.exists():
            print(f"Error: CSV file not found: {args.survey_csv}")
            return 1

        df = pd.read_csv(args.survey_csv)
        if not args.quiet:
            print(f"Loaded {len(df)} rows from {args.survey_csv}")

        if args.path_col not in df.columns:
            print(f"Error: Column '{args.path_col}' not found. Available: {df.columns.tolist()}")
            return 1

        # Get unique LAS paths and convert for current OS
        if HAS_PATH_CONVERTER:
            target_os = args.target_os or get_current_os()
        else:
            target_os = 'mac'  # fallback
            if not args.quiet:
                print("Warning: Path converter not available, assuming mac paths")

        raw_paths = df[args.path_col].unique().tolist()

        # Build list of COPC paths to check
        copc_paths: List[Path] = []
        missing_copc = 0

        for p in raw_paths:
            if HAS_PATH_CONVERTER:
                converted = convert_path_for_os(p, target_os=target_os)
            else:
                converted = p
            las_path = Path(converted)

            # Skip files that are already COPC
            if '.copc.' in las_path.name.lower():
                copc_paths.append(las_path)
                continue

            copc_path = get_copc_path(las_path)
            if copc_path.exists():
                copc_paths.append(copc_path)
            else:
                missing_copc += 1

        if not args.quiet:
            print(f"Found {len(copc_paths)} COPC files to check")
            if missing_copc > 0:
                print(f"  ({missing_copc} LAS files have no COPC version yet)")

        if not copc_paths:
            print("No COPC files to check!")
            return 0

        # Run integrity checks
        if not args.quiet:
            print(f"\n{'='*60}")
            print("BATCH INTEGRITY CHECK")
            print(f"{'='*60}\n")

        passed, failed, failed_files = run_batch_integrity_check(copc_paths, quiet=args.quiet)

        # Output results
        if args.quiet:
            # Just output corrupt file paths
            for path, _ in failed_files:
                print(str(path))
        else:
            print(f"\n{'='*60}")
            print("INTEGRITY CHECK RESULTS")
            print(f"{'='*60}")
            print(f"  Passed: {passed}")
            print(f"  Failed: {failed}")

            if failed_files:
                print(f"\nCorrupt/incomplete files ({len(failed_files)}):")
                for path, issues in failed_files:
                    print(f"\n  {path.name}")
                    for issue in issues:
                        print(f"    - {issue}")

        # Write output file if requested
        if args.output and failed_files:
            with open(args.output, 'w') as f:
                for path, issues in failed_files:
                    f.write(f"{path}\n")
            if not args.quiet:
                print(f"\nWrote {len(failed_files)} corrupt file paths to: {args.output}")

        return 1 if failed > 0 else 0

    return 0


if __name__ == '__main__':
    sys.exit(main())
