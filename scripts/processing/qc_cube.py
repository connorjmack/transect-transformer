#!/usr/bin/env python3
"""Quality control diagnostics for NPZ cube files.

Runs comprehensive checks on extracted transect cube files to validate
data integrity, detect anomalies, and report statistics.

Usage:
    # Basic QC report
    python scripts/processing/qc_cube.py data/processed/transects.npz

    # Verbose output with detailed warnings
    python scripts/processing/qc_cube.py data/processed/transects.npz --verbose

    # Save report to file
    python scripts/processing/qc_cube.py data/processed/transects.npz --output qc_report.txt

    # Strict mode (exit with error code if any warnings)
    python scripts/processing/qc_cube.py data/processed/transects.npz --strict
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class QCReport:
    """Accumulates QC results and generates report."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.stats: Dict[str, any] = {}

    def error(self, msg: str):
        self.errors.append(f"ERROR: {msg}")

    def warning(self, msg: str):
        self.warnings.append(f"WARNING: {msg}")

    def add_info(self, msg: str):
        self.info.append(msg)

    def add_stat(self, key: str, value):
        self.stats[key] = value

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    @property
    def passed_strict(self) -> bool:
        return len(self.errors) == 0 and len(self.warnings) == 0

    def generate_report(self, verbose: bool = False) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("TRANSECT CUBE QC REPORT")
        lines.append("=" * 70)
        lines.append(f"File: {self.filepath}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        status = "PASSED" if self.passed else "FAILED"
        if self.passed and self.warnings:
            status = "PASSED WITH WARNINGS"
        lines.append(f"Status: {status}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        lines.append("")

        # Statistics
        if self.stats:
            lines.append("-" * 70)
            lines.append("CUBE STATISTICS")
            lines.append("-" * 70)
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Errors
        if self.errors:
            lines.append("-" * 70)
            lines.append("ERRORS (must fix)")
            lines.append("-" * 70)
            for err in self.errors:
                lines.append(f"  {err}")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append("-" * 70)
            lines.append("WARNINGS (review recommended)")
            lines.append("-" * 70)
            for warn in self.warnings:
                lines.append(f"  {warn}")
            lines.append("")

        # Info (verbose only)
        if verbose and self.info:
            lines.append("-" * 70)
            lines.append("DETAILED INFO")
            lines.append("-" * 70)
            for info in self.info:
                lines.append(f"  {info}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def load_cube(filepath: Path) -> Dict[str, np.ndarray]:
    """Load NPZ cube file."""
    data = np.load(filepath, allow_pickle=True)
    result = {}
    for key in data.keys():
        arr = data[key]
        # Handle object arrays (lists stored as arrays)
        if arr.dtype == object and arr.ndim == 0:
            result[key] = arr.item()
        elif arr.dtype == object:
            result[key] = arr.tolist() if arr.ndim == 1 else arr
        else:
            result[key] = arr
    return result


def is_unified_cube(cube: Dict) -> bool:
    """Check if this is a unified cube (has coverage_mask)."""
    return 'coverage_mask' in cube


def check_required_keys(cube: Dict, report: QCReport) -> bool:
    """Check that all required keys are present."""
    required_keys = ['points', 'distances', 'metadata', 'transect_ids']
    optional_keys = ['timestamps', 'epoch_names', 'epoch_dates', 'feature_names', 'metadata_names']
    unified_keys = ['coverage_mask', 'beach_slices', 'mop_ids', 'epoch_files', 'epoch_mop_ranges']

    missing_required = [k for k in required_keys if k not in cube]
    if missing_required:
        report.error(f"Missing required keys: {missing_required}")
        return False

    present_optional = [k for k in optional_keys if k in cube]
    missing_optional = [k for k in optional_keys if k not in cube]
    present_unified = [k for k in unified_keys if k in cube]

    # Detect unified mode
    if is_unified_cube(cube):
        report.add_info("Cube format: UNIFIED (coverage_mask present)")
        report.add_stat("cube_format", "unified")
    else:
        report.add_info("Cube format: per-beach (standard)")
        report.add_stat("cube_format", "per-beach")

    report.add_info(f"Required keys present: {required_keys}")
    report.add_info(f"Optional keys present: {present_optional}")
    if present_unified:
        report.add_info(f"Unified mode keys present: {present_unified}")
    if missing_optional:
        report.add_info(f"Optional keys missing: {missing_optional}")

    return True


def check_shapes(cube: Dict, report: QCReport) -> Optional[Tuple[int, int, int, int]]:
    """Validate array shapes and consistency."""
    points = cube['points']
    distances = cube['distances']
    metadata = cube['metadata']
    transect_ids = cube['transect_ids']

    # Check points shape (should be 4D: n_transects, T, N, features)
    if points.ndim != 4:
        report.error(f"points should be 4D (n_transects, T, N, features), got {points.ndim}D")
        return None

    n_transects, n_epochs, n_points, n_features = points.shape
    report.add_stat("n_transects", n_transects)
    report.add_stat("n_epochs", n_epochs)
    report.add_stat("n_points", n_points)
    report.add_stat("n_features", n_features)

    # Check distances shape
    expected_dist_shape = (n_transects, n_epochs, n_points)
    if distances.shape != expected_dist_shape:
        report.error(f"distances shape {distances.shape} != expected {expected_dist_shape}")

    # Check metadata shape
    expected_meta_shape = (n_transects, n_epochs, 12)  # 12 metadata fields
    if metadata.shape[:2] != expected_meta_shape[:2]:
        report.error(f"metadata shape {metadata.shape} doesn't match transects/epochs")
    if metadata.shape[2] != 12:
        report.warning(f"metadata has {metadata.shape[2]} fields, expected 12")

    # Check transect_ids
    if isinstance(transect_ids, np.ndarray):
        n_ids = len(transect_ids)
    else:
        n_ids = len(transect_ids)

    if n_ids != n_transects:
        report.error(f"transect_ids length {n_ids} != n_transects {n_transects}")

    # Check timestamps if present
    if 'timestamps' in cube:
        ts = cube['timestamps']
        if is_unified_cube(cube):
            # Unified mode: timestamps is 1D (n_epochs,) - same for all transects
            expected_ts_shape = (n_epochs,)
            if ts.shape != expected_ts_shape:
                report.error(f"timestamps shape {ts.shape} != expected {expected_ts_shape} (unified mode)")
        else:
            # Per-beach mode: timestamps is 2D (n_transects, n_epochs)
            expected_ts_shape = (n_transects, n_epochs)
            if ts.shape != expected_ts_shape:
                report.error(f"timestamps shape {ts.shape} != expected {expected_ts_shape}")

    # Check epoch_names if present
    if 'epoch_names' in cube:
        epoch_names = cube['epoch_names']
        if isinstance(epoch_names, np.ndarray):
            n_epoch_names = len(epoch_names)
        else:
            n_epoch_names = len(epoch_names)
        if n_epoch_names != n_epochs:
            report.error(f"epoch_names length {n_epoch_names} != n_epochs {n_epochs}")

    return n_transects, n_epochs, n_points, n_features


def check_data_types(cube: Dict, report: QCReport):
    """Check data types are appropriate."""
    points = cube['points']
    distances = cube['distances']
    metadata = cube['metadata']

    if not np.issubdtype(points.dtype, np.floating):
        report.warning(f"points dtype is {points.dtype}, expected float")

    if not np.issubdtype(distances.dtype, np.floating):
        report.warning(f"distances dtype is {distances.dtype}, expected float")

    if not np.issubdtype(metadata.dtype, np.floating):
        report.warning(f"metadata dtype is {metadata.dtype}, expected float")


def check_missing_data(cube: Dict, report: QCReport, dims: Tuple[int, int, int, int]):
    """Check for NaN/missing values and coverage."""
    n_transects, n_epochs, n_points, n_features = dims
    points = cube['points']
    unified = is_unified_cube(cube)

    # Check overall coverage using first feature of first point
    valid_mask = ~np.isnan(points[:, :, 0, 0])
    n_valid = valid_mask.sum()
    total_cells = n_transects * n_epochs
    coverage_pct = 100 * n_valid / total_cells

    report.add_stat("coverage_pct", f"{coverage_pct:.1f}%")
    report.add_stat("valid_transect_epochs", f"{n_valid}/{total_cells}")

    if coverage_pct < 100:
        missing_count = total_cells - n_valid
        report.add_info(f"Missing {missing_count} transect-epoch pairs ({100-coverage_pct:.1f}%)")

        # Check if any transects are completely missing
        transects_with_data = valid_mask.any(axis=1).sum()
        if transects_with_data < n_transects:
            empty_transects = n_transects - transects_with_data
            if unified:
                # Expected in unified mode - just info, not warning
                report.add_info(f"{empty_transects} transects have no data in any epoch (expected for partial-coverage surveys)")
            else:
                report.warning(f"{empty_transects} transects have no data in any epoch")

        # Check if any epochs are completely missing
        epochs_with_data = valid_mask.any(axis=0).sum()
        if epochs_with_data < n_epochs:
            empty_epochs = n_epochs - epochs_with_data
            report.warning(f"{empty_epochs} epochs have no data for any transect")

    # Coverage thresholds differ for unified vs per-beach mode
    if unified:
        # Unified mode: low coverage is expected with partial-coverage surveys
        if coverage_pct < 10:
            report.warning(f"Very low coverage: only {coverage_pct:.1f}% of transect-epochs have data")
        elif coverage_pct < 50:
            report.add_info(f"Low coverage: {coverage_pct:.1f}% (expected for unified mode with partial surveys)")
    else:
        # Per-beach mode: expect higher coverage
        if coverage_pct < 50:
            report.warning(f"Low coverage: only {coverage_pct:.1f}% of transect-epochs have data")
        elif coverage_pct < 80:
            report.add_info(f"Moderate coverage: {coverage_pct:.1f}%")

    # Check for partial NaN within valid transect-epochs
    for t in range(min(n_transects, 100)):  # Sample first 100
        for e in range(n_epochs):
            if valid_mask[t, e]:
                pt_data = points[t, e]
                nan_count = np.isnan(pt_data).sum()
                if nan_count > 0:
                    total_vals = pt_data.size
                    report.warning(f"Transect {t}, epoch {e} has {nan_count}/{total_vals} NaN values within valid data")
                    break
        else:
            continue
        break


def check_value_ranges(cube: Dict, report: QCReport, dims: Tuple[int, int, int, int]):
    """Check that values are in reasonable ranges."""
    n_transects, n_epochs, n_points, n_features = dims
    points = cube['points']
    distances = cube['distances']
    metadata = cube['metadata']

    # Get valid data mask
    valid_mask = ~np.isnan(points[:, :, 0, 0])

    # Feature indices (from ShapefileTransectExtractor)
    FEATURE_NAMES = [
        'distance_m', 'elevation_m', 'slope_deg', 'curvature', 'roughness',
        'intensity', 'red', 'green', 'blue', 'classification', 'return_number', 'num_returns'
    ]

    # Expected ranges for each feature
    expected_ranges = {
        'distance_m': (0, 500),        # 0-500m transect length
        'elevation_m': (-50, 200),     # Below sea level to cliff top
        'slope_deg': (-90, 90),        # Vertical down to vertical up
        'curvature': (-10, 10),        # Curvature range
        'roughness': (0, 10),          # Roughness (std of residuals)
        'intensity': (0, 1),           # Normalized 0-1
        'red': (0, 1),                 # Normalized 0-1
        'green': (0, 1),               # Normalized 0-1
        'blue': (0, 1),                # Normalized 0-1
        'classification': (0, 255),    # LAS classification codes
        'return_number': (0, 15),      # Return number
        'num_returns': (0, 15),        # Number of returns
    }

    for f_idx, f_name in enumerate(FEATURE_NAMES):
        if f_idx >= n_features:
            break

        # Get all valid values for this feature
        feature_data = points[:, :, :, f_idx][valid_mask]
        if len(feature_data) == 0:
            continue

        # Remove NaN
        feature_data = feature_data[~np.isnan(feature_data)]
        if len(feature_data) == 0:
            continue

        f_min, f_max = feature_data.min(), feature_data.max()
        f_mean = feature_data.mean()
        f_std = feature_data.std()

        report.add_info(f"{f_name}: min={f_min:.3f}, max={f_max:.3f}, mean={f_mean:.3f}, std={f_std:.3f}")

        # Check against expected range
        if f_name in expected_ranges:
            exp_min, exp_max = expected_ranges[f_name]
            if f_min < exp_min - 0.1:  # Allow small tolerance
                report.warning(f"{f_name} min ({f_min:.3f}) below expected ({exp_min})")
            if f_max > exp_max + 0.1:
                report.warning(f"{f_name} max ({f_max:.3f}) above expected ({exp_max})")

    # Check distances are monotonically increasing
    n_checked = 0
    n_violations = 0
    for t in range(n_transects):
        for e in range(n_epochs):
            if valid_mask[t, e]:
                dist = distances[t, e]
                if not np.all(np.diff(dist) >= -0.001):  # Allow tiny numerical errors
                    n_violations += 1
                n_checked += 1

    if n_violations > 0:
        report.warning(f"{n_violations}/{n_checked} transect-epochs have non-monotonic distances")
    else:
        report.add_info(f"All {n_checked} transect-epoch distances are monotonically increasing")

    # Check metadata ranges
    METADATA_NAMES = [
        'cliff_height_m', 'mean_slope_deg', 'max_slope_deg', 'toe_elevation_m',
        'top_elevation_m', 'orientation_deg', 'transect_length_m', 'latitude',
        'longitude', 'transect_id', 'mean_intensity', 'dominant_class'
    ]

    meta_expected = {
        'cliff_height_m': (0, 100),
        'mean_slope_deg': (0, 90),
        'max_slope_deg': (0, 90),
        'orientation_deg': (0, 360),
        'transect_length_m': (1, 500),
        'mean_intensity': (0, 1),
    }

    valid_meta = metadata[valid_mask]
    for m_idx, m_name in enumerate(METADATA_NAMES):
        if m_idx >= metadata.shape[2]:
            break

        meta_vals = valid_meta[:, m_idx]
        meta_vals = meta_vals[~np.isnan(meta_vals)]
        if len(meta_vals) == 0:
            continue

        m_min, m_max = meta_vals.min(), meta_vals.max()

        if m_name in meta_expected:
            exp_min, exp_max = meta_expected[m_name]
            if m_min < exp_min - 0.1:
                report.warning(f"metadata {m_name} min ({m_min:.3f}) below expected ({exp_min})")
            if m_max > exp_max + 0.1:
                report.warning(f"metadata {m_name} max ({m_max:.3f}) above expected ({exp_max})")


def check_temporal_consistency(cube: Dict, report: QCReport):
    """Check temporal ordering and consistency."""
    if 'epoch_dates' not in cube:
        report.add_info("No epoch_dates found, skipping temporal checks")
        return

    epoch_dates = cube['epoch_dates']
    if isinstance(epoch_dates, np.ndarray):
        epoch_dates = epoch_dates.tolist()

    # Parse dates
    parsed_dates = []
    for d in epoch_dates:
        try:
            if isinstance(d, str):
                # Try ISO format
                parsed = datetime.fromisoformat(d.replace('Z', '+00:00'))
                parsed_dates.append(parsed)
            else:
                parsed_dates.append(None)
        except:
            parsed_dates.append(None)

    valid_dates = [d for d in parsed_dates if d is not None]

    if len(valid_dates) != len(epoch_dates):
        report.warning(f"Could not parse {len(epoch_dates) - len(valid_dates)} epoch dates")

    if len(valid_dates) >= 2:
        # Check chronological order
        is_sorted = all(valid_dates[i] <= valid_dates[i+1] for i in range(len(valid_dates)-1))
        if not is_sorted:
            report.error("Epochs are not in chronological order")
        else:
            report.add_info("Epochs are in chronological order")

        # Report date range
        date_range = f"{valid_dates[0].strftime('%Y-%m-%d')} to {valid_dates[-1].strftime('%Y-%m-%d')}"
        report.add_stat("date_range", date_range)

        # Check for duplicate dates
        date_strs = [d.strftime('%Y-%m-%d') for d in valid_dates]
        if len(date_strs) != len(set(date_strs)):
            report.warning("Duplicate dates found in epochs")

    if 'timestamps' in cube:
        timestamps = cube['timestamps']
        # Only check for uniform timestamps in per-beach mode (2D timestamps)
        # In unified mode, timestamps is already 1D so this check doesn't apply
        if not is_unified_cube(cube) and timestamps.ndim == 2 and timestamps.shape[0] > 1:
            if not np.allclose(timestamps[0], timestamps[1:].mean(axis=0)):
                report.warning("Timestamps vary across transects (expected uniform)")


def check_transect_ids(cube: Dict, report: QCReport):
    """Check transect ID consistency."""
    transect_ids = cube['transect_ids']
    if isinstance(transect_ids, np.ndarray):
        transect_ids = transect_ids.tolist()

    n_ids = len(transect_ids)
    n_unique = len(set(str(t) for t in transect_ids))

    if n_unique != n_ids:
        report.error(f"Duplicate transect IDs: {n_ids} total, {n_unique} unique")
    else:
        report.add_info(f"All {n_ids} transect IDs are unique")

    # Check if IDs look like MOP format
    mop_count = 0
    for tid in transect_ids:
        tid_str = str(tid)
        if 'MOP' in tid_str.upper() or tid_str.isdigit():
            mop_count += 1

    if mop_count == n_ids:
        report.add_info("All transect IDs appear to be MOP format")
    elif mop_count > 0:
        report.add_info(f"{mop_count}/{n_ids} transect IDs appear to be MOP format")

    # Show sample IDs
    sample_ids = transect_ids[:5] if len(transect_ids) > 5 else transect_ids
    report.add_info(f"Sample transect IDs: {sample_ids}")


def check_unified_specific(cube: Dict, report: QCReport, dims: Tuple[int, int, int, int]):
    """Check unified-mode specific arrays."""
    if not is_unified_cube(cube):
        return

    n_transects, n_epochs, _, _ = dims

    # Check coverage_mask shape and consistency
    if 'coverage_mask' in cube:
        coverage_mask = cube['coverage_mask']
        expected_shape = (n_transects, n_epochs)
        if coverage_mask.shape != expected_shape:
            report.error(f"coverage_mask shape {coverage_mask.shape} != expected {expected_shape}")
        else:
            report.add_info(f"coverage_mask shape OK: {coverage_mask.shape}")

        # Check that coverage_mask matches actual data presence
        points = cube['points']
        actual_valid = ~np.isnan(points[:, :, 0, 0])
        if not np.array_equal(coverage_mask, actual_valid):
            mismatch = (coverage_mask != actual_valid).sum()
            report.warning(f"coverage_mask has {mismatch} mismatches with actual data presence")

    # Check beach_slices
    if 'beach_slices' in cube:
        beach_slices = cube['beach_slices']
        if isinstance(beach_slices, np.ndarray):
            beach_slices = beach_slices.item()

        expected_beaches = ['blacks', 'torrey', 'delmar', 'solana', 'sanelijo', 'encinitas']
        present_beaches = list(beach_slices.keys())

        missing_beaches = [b for b in expected_beaches if b not in beach_slices]
        if missing_beaches:
            report.warning(f"Missing beach slices: {missing_beaches}")

        # Check slices are non-overlapping and contiguous
        prev_end = 0
        for beach in expected_beaches:
            if beach in beach_slices:
                start, end = beach_slices[beach]
                if start < prev_end:
                    report.error(f"Beach {beach} slice [{start}, {end}) overlaps with previous (ends at {prev_end})")
                if start > prev_end:
                    report.add_info(f"Gap between beaches: indices {prev_end}-{start}")
                n_beach_transects = end - start
                report.add_info(f"Beach {beach}: indices {start}-{end} ({n_beach_transects} transects)")
                prev_end = end

    # Check mop_ids
    if 'mop_ids' in cube:
        mop_ids = cube['mop_ids']
        if len(mop_ids) != n_transects:
            report.error(f"mop_ids length {len(mop_ids)} != n_transects {n_transects}")
        else:
            valid_mops = mop_ids[mop_ids > 0]
            if len(valid_mops) > 0:
                report.add_info(f"MOP ID range: {valid_mops.min()} - {valid_mops.max()}")

    # Check epoch_mop_ranges
    if 'epoch_mop_ranges' in cube:
        epoch_mop_ranges = cube['epoch_mop_ranges']
        if epoch_mop_ranges.shape[0] != n_epochs:
            report.error(f"epoch_mop_ranges has {epoch_mop_ranges.shape[0]} rows, expected {n_epochs}")
        else:
            report.add_info(f"epoch_mop_ranges shape OK: {epoch_mop_ranges.shape}")


def check_spatial_distribution(cube: Dict, report: QCReport, dims: Tuple[int, int, int, int]):
    """Check spatial distribution of transects."""
    n_transects, n_epochs, _, _ = dims
    metadata = cube['metadata']

    # Use latest epoch for spatial check
    latest = n_epochs - 1

    # Metadata indices: latitude=7, longitude=8
    lats = metadata[:, latest, 7]
    lons = metadata[:, latest, 8]

    valid_lats = lats[~np.isnan(lats)]
    valid_lons = lons[~np.isnan(lons)]

    if len(valid_lats) > 0:
        lat_range = f"{valid_lats.min():.6f} to {valid_lats.max():.6f}"
        lon_range = f"{valid_lons.min():.6f} to {valid_lons.max():.6f}"
        report.add_stat("latitude_range", lat_range)
        report.add_stat("longitude_range", lon_range)

        # Check if coordinates look like projected (large values) vs geographic (small values)
        if abs(valid_lats.mean()) > 1000:
            report.add_info("Coordinates appear to be projected (UTM or similar)")
        else:
            report.add_info("Coordinates appear to be geographic (lat/lon)")


def check_file_size(filepath: Path, report: QCReport):
    """Report file size and compression ratio."""
    size_bytes = filepath.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    report.add_stat("file_size", f"{size_mb:.1f} MB")

    if size_mb > 1000:
        report.warning(f"Large file size ({size_mb:.0f} MB) may cause memory issues")


def run_qc(filepath: Path, verbose: bool = False) -> QCReport:
    """Run all QC checks on a cube file."""
    report = QCReport(filepath)

    # Check file exists
    if not filepath.exists():
        report.error(f"File not found: {filepath}")
        return report

    # Check file size
    check_file_size(filepath, report)

    # Load file
    try:
        cube = load_cube(filepath)
    except Exception as e:
        report.error(f"Failed to load file: {e}")
        return report

    # Run checks
    if not check_required_keys(cube, report):
        return report

    dims = check_shapes(cube, report)
    if dims is None:
        return report

    check_data_types(cube, report)
    check_missing_data(cube, report, dims)
    check_value_ranges(cube, report, dims)
    check_temporal_consistency(cube, report)
    check_transect_ids(cube, report)
    check_unified_specific(cube, report, dims)
    check_spatial_distribution(cube, report, dims)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run quality control diagnostics on NPZ cube files."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input NPZ cube file to check",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed info messages",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save report to file",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if any warnings (not just errors)",
    )

    args = parser.parse_args()

    # Run QC
    report = run_qc(args.input, verbose=args.verbose)

    # Generate and print report
    report_text = report.generate_report(verbose=args.verbose)
    print(report_text)

    # Save if requested
    if args.output:
        args.output.write_text(report_text)
        print(f"\nReport saved to {args.output}")

    # Exit code
    if args.strict:
        return 0 if report.passed_strict else 1
    else:
        return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
