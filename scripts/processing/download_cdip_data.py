"""Batch download CDIP wave data for CliffCast.

This script downloads wave data from CDIP's THREDDS server for all San Diego
MOP transects and caches them locally as NetCDF files.

Usage:
    # Download all San Diego MOPs
    python download_cdip_data.py --output data/raw/cdip/

    # Download specific beach
    python download_cdip_data.py --output data/raw/cdip/ --beach delmar

    # Download custom MOP range
    python download_cdip_data.py --output data/raw/cdip/ --mop-min 595 --mop-max 620

    # Verify existing downloads
    python download_cdip_data.py --output data/raw/cdip/ --verify-only

Author: CliffCast Team
Date: 2026-01-18
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.cdip_wave_loader import CDIPWaveLoader

# Beach MOP ranges (canonical from CLAUDE.md)
BEACH_MOP_RANGES = {
    'blacks': (520, 567),
    'torrey': (567, 581),
    'delmar': (595, 620),
    'solana': (637, 666),
    'sanelijo': (683, 708),
    'encinitas': (708, 764),
}

# Default San Diego range
DEFAULT_MOP_MIN = 520
DEFAULT_MOP_MAX = 764

logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[Path] = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def get_mop_range(
    beach: Optional[str] = None,
    mop_min: Optional[int] = None,
    mop_max: Optional[int] = None,
) -> Tuple[int, int]:
    """Determine MOP range from arguments.

    Args:
        beach: Beach name (blacks, torrey, delmar, solana, sanelijo, encinitas)
        mop_min: Custom minimum MOP
        mop_max: Custom maximum MOP

    Returns:
        Tuple of (mop_min, mop_max)
    """
    if beach:
        if beach not in BEACH_MOP_RANGES:
            raise ValueError(
                f"Unknown beach: {beach}. Options: {', '.join(BEACH_MOP_RANGES.keys())}"
            )
        return BEACH_MOP_RANGES[beach]

    if mop_min is not None and mop_max is not None:
        if mop_min > mop_max:
            raise ValueError(f"mop_min ({mop_min}) must be <= mop_max ({mop_max})")
        return mop_min, mop_max

    # Default: all San Diego
    return DEFAULT_MOP_MIN, DEFAULT_MOP_MAX


def try_multiple_site_labels(
    loader: CDIPWaveLoader,
    mop_id: int,
    start_date: str,
    end_date: str,
) -> Tuple[bool, Optional[str]]:
    """Try multiple site label formats for a MOP.

    Args:
        loader: CDIPWaveLoader instance
        mop_id: MOP transect ID
        start_date: Start date string
        end_date: End date string

    Returns:
        Tuple of (success: bool, site_label: Optional[str])
    """
    # Try multiple formats
    formats = [
        f"D{mop_id:04d}",  # D0582
        f"D{mop_id:03d}",  # D582
        f"D{mop_id}",      # D582
    ]

    for site_label in formats:
        try:
            logger.debug(f"Trying site label: {site_label}")
            wave_data = loader.load_mop(
                mop_id=mop_id,
                start_date=start_date,
                end_date=end_date,
                site_label_override=site_label,
            )
            logger.info(f"✓ MOP {mop_id} downloaded successfully as {site_label}")
            return True, site_label
        except Exception as e:
            logger.debug(f"Failed with {site_label}: {e}")
            continue

    return False, None


def download_single_mop(
    mop_id: int,
    output_dir: Path,
    start_date: str,
    end_date: str,
    skip_existing: bool = True,
) -> Tuple[int, bool, Optional[str]]:
    """Download CDIP data for a single MOP.

    Args:
        mop_id: MOP transect ID
        output_dir: Output directory for NetCDF files
        start_date: Start date string
        end_date: End date string
        skip_existing: Skip if file already exists

    Returns:
        Tuple of (mop_id, success: bool, error_message: Optional[str])
    """
    # Check if file already exists
    possible_files = [
        output_dir / f"D{mop_id:04d}_hindcast.nc",
        output_dir / f"D{mop_id:03d}_hindcast.nc",
        output_dir / f"D{mop_id}_hindcast.nc",
    ]

    if skip_existing:
        for file_path in possible_files:
            if file_path.exists():
                logger.info(f"⊳ MOP {mop_id} already exists, skipping")
                return mop_id, True, None

    # Initialize loader with cache directory
    loader = CDIPWaveLoader(cache_dir=output_dir)

    # Try to download with multiple site label formats
    success, site_label = try_multiple_site_labels(
        loader, mop_id, start_date, end_date
    )

    if success:
        return mop_id, True, None
    else:
        error_msg = f"Failed to download MOP {mop_id} with all site label formats"
        logger.warning(f"✗ {error_msg}")
        return mop_id, False, error_msg


def verify_file(file_path: Path) -> Tuple[bool, str]:
    """Verify that a NetCDF file is valid.

    Args:
        file_path: Path to NetCDF file

    Returns:
        Tuple of (valid: bool, message: str)
    """
    try:
        import xarray as xr

        # Try to open and read basic metadata
        ds = xr.open_dataset(file_path, decode_times=False)

        # Check required variables
        required_vars = ['waveTime', 'waveHs', 'waveTp', 'waveDp']
        missing_vars = [v for v in required_vars if v not in ds]

        if missing_vars:
            ds.close()
            return False, f"Missing variables: {missing_vars}"

        # Check that data is not empty
        n_records = len(ds['waveTime'])
        if n_records == 0:
            ds.close()
            return False, "No data records"

        ds.close()
        return True, f"Valid ({n_records} records)"

    except Exception as e:
        return False, f"Error: {e}"


def verify_downloads(output_dir: Path, mop_range: Tuple[int, int]) -> dict:
    """Verify existing downloads.

    Args:
        output_dir: Directory containing NetCDF files
        mop_range: Tuple of (mop_min, mop_max)

    Returns:
        Dictionary with verification results
    """
    mop_min, mop_max = mop_range
    results = {
        'total': mop_max - mop_min + 1,
        'found': 0,
        'valid': 0,
        'invalid': 0,
        'missing': [],
        'invalid_files': [],
    }

    logger.info(f"\nVerifying downloads for MOP range {mop_min}-{mop_max}...")

    for mop_id in range(mop_min, mop_max + 1):
        # Check all possible file formats
        possible_files = [
            output_dir / f"D{mop_id:04d}_hindcast.nc",
            output_dir / f"D{mop_id:03d}_hindcast.nc",
            output_dir / f"D{mop_id}_hindcast.nc",
        ]

        found = None
        for file_path in possible_files:
            if file_path.exists():
                found = file_path
                break

        if found:
            results['found'] += 1
            valid, message = verify_file(found)

            if valid:
                results['valid'] += 1
                logger.info(f"✓ MOP {mop_id}: {message}")
            else:
                results['invalid'] += 1
                results['invalid_files'].append((mop_id, found, message))
                logger.warning(f"✗ MOP {mop_id}: {message}")
        else:
            results['missing'].append(mop_id)
            logger.info(f"⊳ MOP {mop_id}: Missing")

    return results


def batch_download(
    output_dir: Path,
    mop_range: Tuple[int, int],
    start_date: str,
    end_date: str,
    max_workers: int = 4,
    skip_existing: bool = True,
) -> dict:
    """Batch download CDIP data for multiple MOPs.

    Args:
        output_dir: Output directory
        mop_range: Tuple of (mop_min, mop_max)
        start_date: Start date string
        end_date: End date string
        max_workers: Number of parallel workers
        skip_existing: Skip existing files

    Returns:
        Dictionary with download statistics
    """
    mop_min, mop_max = mop_range
    mop_ids = list(range(mop_min, mop_max + 1))

    results = {
        'total': len(mop_ids),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'errors': [],
    }

    logger.info(f"\nDownloading CDIP data for {len(mop_ids)} MOPs ({mop_min}-{mop_max})")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Parallel workers: {max_workers}")
    logger.info(f"Skip existing: {skip_existing}\n")

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                download_single_mop,
                mop_id,
                output_dir,
                start_date,
                end_date,
                skip_existing,
            ): mop_id
            for mop_id in mop_ids
        }

        # Process completed downloads
        try:
            from tqdm import tqdm
            progress = tqdm(total=len(futures), desc="Downloading", unit="MOP")
            has_tqdm = True
        except ImportError:
            logger.warning("tqdm not available, progress bar disabled")
            has_tqdm = False
            progress = None

        for future in as_completed(futures):
            mop_id, success, error_msg = future.result()

            if success:
                if skip_existing and error_msg is None:
                    # File existed, counted as success but note it was skipped
                    results['success'] += 1
                else:
                    results['success'] += 1
            else:
                results['failed'] += 1
                if error_msg:
                    results['errors'].append((mop_id, error_msg))

            if has_tqdm:
                progress.update(1)

        if has_tqdm:
            progress.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch download CDIP wave data for CliffCast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all San Diego MOPs
  python download_cdip_data.py --output data/raw/cdip/

  # Download specific beach
  python download_cdip_data.py --output data/raw/cdip/ --beach delmar

  # Download custom range
  python download_cdip_data.py --output data/raw/cdip/ --mop-min 595 --mop-max 620

  # Verify existing downloads
  python download_cdip_data.py --output data/raw/cdip/ --verify-only

Beach options: blacks, torrey, delmar, solana, sanelijo, encinitas
        """,
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for NetCDF files',
    )
    parser.add_argument(
        '--beach',
        type=str,
        choices=list(BEACH_MOP_RANGES.keys()),
        help='Download specific beach',
    )
    parser.add_argument(
        '--mop-min',
        type=int,
        help='Minimum MOP ID (default: 520 for all San Diego)',
    )
    parser.add_argument(
        '--mop-max',
        type=int,
        help='Maximum MOP ID (default: 764 for all San Diego)',
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2017-01-01',
        help='Start date (default: 2017-01-01)',
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-12-31',
        help='End date (default: 2025-12-31)',
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Number of parallel download workers (default: 4)',
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-download existing files (default: skip existing)',
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing downloads, do not download',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / f"download_log_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt"
    setup_logging(log_file)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("="*80)
    logger.info("CDIP Wave Data Download Script")
    logger.info("="*80)

    # Determine MOP range
    try:
        mop_range = get_mop_range(args.beach, args.mop_min, args.mop_max)
    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    mop_min, mop_max = mop_range
    logger.info(f"MOP range: {mop_min}-{mop_max} ({mop_max - mop_min + 1} MOPs)")

    if args.beach:
        logger.info(f"Beach: {args.beach}")

    # Verify-only mode
    if args.verify_only:
        results = verify_downloads(output_dir, mop_range)

        logger.info("\n" + "="*80)
        logger.info("VERIFICATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total MOPs in range: {results['total']}")
        logger.info(f"Files found: {results['found']}")
        logger.info(f"Valid files: {results['valid']}")
        logger.info(f"Invalid files: {results['invalid']}")
        logger.info(f"Missing files: {len(results['missing'])}")

        if results['missing']:
            logger.info(f"\nMissing MOPs: {results['missing']}")

        if results['invalid_files']:
            logger.info("\nInvalid files:")
            for mop_id, file_path, message in results['invalid_files']:
                logger.info(f"  MOP {mop_id} ({file_path.name}): {message}")

        coverage_pct = 100 * results['valid'] / results['total']
        logger.info(f"\nCoverage: {coverage_pct:.1f}%")

        sys.exit(0)

    # Download mode
    skip_existing = not args.no_skip_existing

    results = batch_download(
        output_dir,
        mop_range,
        args.start_date,
        args.end_date,
        args.max_workers,
        skip_existing,
    )

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*80)
    logger.info(f"Total MOPs: {results['total']}")
    logger.info(f"Successful: {results['success']}")
    logger.info(f"Failed: {results['failed']}")

    if results['errors']:
        logger.info(f"\nFailed downloads ({len(results['errors'])} MOPs):")
        for mop_id, error_msg in results['errors']:
            logger.info(f"  MOP {mop_id}: {error_msg}")

    success_rate = 100 * results['success'] / results['total']
    logger.info(f"\nSuccess rate: {success_rate:.1f}%")
    logger.info(f"Log file: {log_file}")

    if results['failed'] > 0:
        logger.warning(
            f"\n{results['failed']} MOPs failed to download. "
            "These may not be available on CDIP THREDDS server."
        )
        logger.warning("Check the log file for details.")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
