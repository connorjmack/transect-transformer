#!/usr/bin/env python3
"""Convert LAS/LAZ files to COPC format for fast spatial queries.

Reads LAS paths from master_list.csv (or other sources) and creates COPC
versions alongside the original files. Original files are NEVER modified
or deleted.

For each input file, creates a .copc.laz file in the same directory:
    input.laz  ->  input.copc.laz
    input.las  ->  input.copc.laz

COPC (Cloud Optimized Point Cloud) enables direct spatial queries without
scanning the entire file, providing 10-100x faster loading for narrow
spatial queries like transect extraction.

Usage:
    # Dry run - show what would be converted (safe, no changes made)
    python scripts/processing/make_copc.py \
        --survey-csv data/raw/master_list.csv \
        --dry-run

    # Convert all files (4 parallel workers)
    python scripts/processing/make_copc.py \
        --survey-csv data/raw/master_list.csv \
        --workers 4

    # Skip files that already have COPC versions
    python scripts/processing/make_copc.py \
        --survey-csv data/raw/master_list.csv \
        --workers 4 \
        --skip-existing

    # Convert specific range (useful for batch processing)
    python scripts/processing/make_copc.py \
        --survey-csv data/raw/master_list.csv \
        --start 0 --end 100 --workers 4

    # Convert a single file
    python scripts/processing/make_copc.py \
        --input-file /path/to/scan.laz

Requirements:
    - PDAL must be installed: conda install -c conda-forge pdal
    - Or: pip install pdal python-pdal
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

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

from scripts.processing.extract_transects import convert_path_for_os, get_current_os


def get_copc_path(las_path: Path) -> Path:
    """Get the COPC output path for a LAS file.

    Places .copc.laz file alongside original file.
    Original file is never modified.

    Args:
        las_path: Path to original LAS/LAZ file

    Returns:
        Path for COPC output file

    Examples:
        >>> get_copc_path(Path("scan.las"))
        PosixPath('scan.copc.laz')
        >>> get_copc_path(Path("scan.laz"))
        PosixPath('scan.copc.laz')
    """
    # Remove existing extension and add .copc.laz
    stem = las_path.stem
    # Handle case where file is already .copc.laz
    if stem.endswith('.copc'):
        return las_path
    return las_path.with_name(stem + '.copc.laz')


def check_pdal_available() -> Tuple[bool, str]:
    """Check if PDAL is available and return version info.

    Returns:
        Tuple of (is_available, version_or_error_message)
    """
    try:
        result = subprocess.run(
            ['pdal', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Extract version from output
            version_line = result.stdout.strip().split('\n')[0]
            return True, version_line
        return False, f"PDAL returned error: {result.stderr}"
    except FileNotFoundError:
        return False, "PDAL not found. Install with: conda install -c conda-forge pdal"
    except subprocess.TimeoutExpired:
        return False, "PDAL version check timed out"
    except Exception as e:
        return False, f"Error checking PDAL: {e}"


def convert_to_copc(las_path: Path, copc_path: Path, verbose: bool = False, force: bool = False) -> Tuple[bool, str]:
    """Convert a single LAS/LAZ file to COPC format using PDAL.

    The original file is NEVER modified. A new .copc.laz file is created.

    Args:
        las_path: Path to input LAS/LAZ file
        copc_path: Path for output COPC file
        verbose: If True, show PDAL output
        force: If True, overwrite existing COPC file

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not las_path.exists():
        return False, f"Input file not found: {las_path}"

    if copc_path.exists():
        if force:
            # Remove existing COPC file to replace it
            copc_path.unlink()
        else:
            return False, f"Output already exists: {copc_path}"

    try:
        # Build PDAL pipeline command
        # Using 'pdal translate' with writers.copc.forward=all preserves all attributes
        cmd = [
            'pdal', 'translate',
            str(las_path),
            str(copc_path),
            '--writers.copc.forward=all'
        ]

        # Run conversion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per file
        )

        if result.returncode != 0:
            # Clean up partial output if it exists
            if copc_path.exists():
                copc_path.unlink()

            # Collect error info
            error_msg = result.stderr.strip()
            if not error_msg:
                error_msg = result.stdout.strip()
            if not error_msg:
                error_msg = f"PDAL failed with exit code {result.returncode}"

            return False, f"PDAL error: {error_msg}"

        # Verify output exists and has reasonable size
        if not copc_path.exists():
            return False, "Output file not created"

        input_size = las_path.stat().st_size
        output_size = copc_path.stat().st_size

        # COPC files should be similar size (within 50% typically)
        if output_size < input_size * 0.1:
            copc_path.unlink()
            return False, f"Output suspiciously small ({output_size} vs {input_size} bytes)"

        size_mb = output_size / (1024 * 1024)
        return True, f"Created {copc_path.name} ({size_mb:.1f} MB)"

    except subprocess.TimeoutExpired:
        # Clean up partial output
        if copc_path.exists():
            copc_path.unlink()
        return False, "Conversion timed out (>10 min)"
    except FileNotFoundError:
        return False, "PDAL not found. Install with: conda install -c conda-forge pdal"
    except Exception as e:
        # Clean up partial output
        if copc_path.exists():
            copc_path.unlink()
        return False, f"Error: {str(e)}"


def convert_worker(args: Tuple[Path, Path, bool, bool]) -> Tuple[Path, bool, str]:
    """Worker function for parallel conversion.

    Args:
        args: Tuple of (las_path, copc_path, verbose, force)

    Returns:
        Tuple of (las_path, success, message)
    """
    las_path, copc_path, verbose, force = args
    success, message = convert_to_copc(las_path, copc_path, verbose, force)
    return las_path, success, message


def main():
    parser = argparse.ArgumentParser(
        description='Convert LAS/LAZ files to COPC format for fast spatial queries. '
                    'Original files are NEVER modified.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input sources (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--survey-csv',
        type=Path,
        help='Path to master_list.csv with LAS file paths'
    )
    input_group.add_argument(
        '--input-file',
        type=Path,
        help='Single LAS/LAZ file to convert'
    )
    input_group.add_argument(
        '--input-dir',
        type=Path,
        help='Directory containing LAS/LAZ files to convert'
    )

    parser.add_argument(
        '--path-col',
        type=str,
        default='full_path',
        help='Column name for LAS paths in CSV (default: full_path)'
    )

    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1). Use -j 4 for faster conversion.'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be converted without actually converting (safe preview)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already have COPC versions'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing COPC files (mutually exclusive with --skip-existing)'
    )

    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start index in file list (for batch processing)'
    )

    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End index in file list (for batch processing)'
    )

    parser.add_argument(
        '-n', '--limit',
        type=int,
        default=None,
        help='Limit to first N files (useful for testing, e.g. -n 4)'
    )

    parser.add_argument(
        '--target-os',
        type=str,
        choices=['mac', 'linux'],
        default=None,
        help='Target OS for path conversion (default: auto-detect)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.la[sz]',
        help='Glob pattern when using --input-dir (default: *.la[sz])'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )

    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.skip_existing and args.force:
        print("Error: --skip-existing and --force are mutually exclusive")
        return 1

    # Check PDAL availability first
    pdal_ok, pdal_msg = check_pdal_available()
    if not pdal_ok and not args.dry_run:
        print(f"Error: {pdal_msg}")
        return 1
    elif pdal_ok:
        print(f"PDAL: {pdal_msg}")

    # Collect LAS files to process
    las_paths: List[Path] = []

    if args.input_file:
        # Single file mode
        las_paths = [args.input_file]

    elif args.input_dir:
        # Directory mode
        if not args.input_dir.exists():
            print(f"Error: Directory not found: {args.input_dir}")
            return 1
        las_paths = sorted(args.input_dir.glob(args.pattern))
        print(f"Found {len(las_paths)} files matching '{args.pattern}' in {args.input_dir}")

    elif args.survey_csv:
        # CSV mode
        if not HAS_PANDAS:
            print("Error: pandas required for CSV mode. Install with: pip install pandas")
            return 1

        if not args.survey_csv.exists():
            print(f"Error: CSV file not found: {args.survey_csv}")
            return 1

        df = pd.read_csv(args.survey_csv)
        print(f"Loaded {len(df)} rows from {args.survey_csv}")

        if args.path_col not in df.columns:
            print(f"Error: Column '{args.path_col}' not found. Available: {df.columns.tolist()}")
            return 1

        # Get unique LAS paths and convert for current OS
        target_os = args.target_os or get_current_os()
        raw_paths = df[args.path_col].unique().tolist()

        for p in raw_paths:
            converted = convert_path_for_os(p, target_os=target_os)
            las_paths.append(Path(converted))

        print(f"Found {len(las_paths)} unique LAS files (target OS: {target_os})")

    if not las_paths:
        print("No files to process!")
        return 0

    # Apply range filter
    original_count = len(las_paths)
    if args.end is not None:
        las_paths = las_paths[args.start:args.end]
    else:
        las_paths = las_paths[args.start:]

    # Apply limit (-n) if specified
    if args.limit is not None:
        las_paths = las_paths[:args.limit]
        print(f"Limited to first {args.limit} files (-n/--limit)")

    if len(las_paths) != original_count:
        print(f"Processing {len(las_paths)} of {original_count} files")

    # Build conversion list
    conversions: List[Tuple[Path, Path]] = []
    skipped_existing = 0
    skipped_missing = 0
    skipped_already_copc = 0

    for las_path in las_paths:
        # Skip files that are already COPC
        if '.copc.' in las_path.name.lower():
            skipped_already_copc += 1
            continue

        copc_path = get_copc_path(las_path)

        if args.skip_existing and copc_path.exists():
            skipped_existing += 1
            if args.verbose:
                print(f"  Skipping (exists): {copc_path.name}")
            continue

        if not las_path.exists():
            skipped_missing += 1
            print(f"  Warning: File not found: {las_path}")
            continue

        conversions.append((las_path, copc_path))

    # Print summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Files to convert: {len(conversions)}")
    print(f"  Skipped (COPC exists): {skipped_existing}")
    print(f"  Skipped (already COPC): {skipped_already_copc}")
    print(f"  Skipped (file missing): {skipped_missing}")

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - No files will be modified")
        print(f"{'='*60}")
        print("\nFiles that would be converted:")
        for i, (las_path, copc_path) in enumerate(conversions[:20]):
            size_mb = las_path.stat().st_size / (1024 * 1024) if las_path.exists() else 0
            print(f"  {i+1}. {las_path.name} ({size_mb:.1f} MB)")
            print(f"      -> {copc_path.name}")
        if len(conversions) > 20:
            print(f"  ... and {len(conversions) - 20} more files")

        # Estimate time
        if len(conversions) > 0:
            avg_time_sec = 30  # rough estimate
            total_sec = len(conversions) * avg_time_sec / max(args.workers, 1)
            hours = total_sec / 3600
            print(f"\nEstimated time: ~{hours:.1f} hours with {args.workers} worker(s)")

        return 0

    if not conversions:
        print("\nNothing to convert!")
        return 0

    # Run conversions
    print(f"\n{'='*60}")
    print(f"CONVERTING {len(conversions)} FILES")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}\n")

    success_count = 0
    fail_count = 0
    failed_files: List[Tuple[Path, str]] = []

    if args.workers == 1:
        # Sequential processing
        if HAS_TQDM:
            iterator = tqdm(conversions, desc="Converting", unit="file")
        else:
            iterator = conversions

        for las_path, copc_path in iterator:
            success, message = convert_to_copc(las_path, copc_path, args.verbose, args.force)
            if success:
                success_count += 1
                if args.verbose:
                    print(f"  OK: {message}")
            else:
                fail_count += 1
                failed_files.append((las_path, message))
                print(f"  FAILED: {las_path.name} - {message}")
    else:
        # Parallel processing
        work_items = [(las, copc, args.verbose, args.force) for las, copc in conversions]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(convert_worker, item): item[0]
                for item in work_items
            }

            if HAS_TQDM:
                iterator = tqdm(as_completed(futures), total=len(conversions), desc="Converting", unit="file")
            else:
                iterator = as_completed(futures)

            for future in iterator:
                las_path, success, message = future.result()
                if success:
                    success_count += 1
                    if args.verbose:
                        print(f"  OK: {message}")
                else:
                    fail_count += 1
                    failed_files.append((las_path, message))
                    print(f"  FAILED: {las_path.name} - {message}")

    # Final summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Succeeded: {success_count}")
    print(f"  Failed: {fail_count}")

    if failed_files:
        print(f"\nFailed files:")
        for las_path, error in failed_files[:10]:
            print(f"  - {las_path.name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more failures")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
