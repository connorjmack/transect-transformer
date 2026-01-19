#!/usr/bin/env python3
"""Prepare atmospheric features for CliffCast model training.

This script processes raw PRISM climate data into the 24 atmospheric features
used by the CliffCast model. Features are computed per-beach and saved as
parquet files for efficient loading during training.

Supports two input formats:
1. PRISM Bulk Export (daily_4km/): Multiple yearly CSVs from bulk.php
2. Legacy format: Individual {beach}_raw.csv files

Usage:
    # Process from PRISM bulk export (recommended)
    python scripts/processing/prepare_atmos_features.py \
        --bulk-dir data/raw/prism/daily_4km/ \
        --output-dir data/processed/atmospheric/

    # Process from legacy individual CSV files
    python scripts/processing/prepare_atmos_features.py \
        --raw-dir data/raw/prism/ \
        --output-dir data/processed/atmospheric/

    # Process specific beaches
    python scripts/processing/prepare_atmos_features.py \
        --bulk-dir data/raw/prism/daily_4km/ \
        --output-dir data/processed/atmospheric/ \
        --beaches delmar solana

Output:
    Creates one parquet file per beach with all 24 atmospheric features:
    - data/processed/atmospheric/blacks_atmos.parquet
    - data/processed/atmospheric/torrey_atmos.parquet
    - data/processed/atmospheric/delmar_atmos.parquet
    - data/processed/atmospheric/solana_atmos.parquet
    - data/processed/atmospheric/sanelijo_atmos.parquet
    - data/processed/atmospheric/encinitas_atmos.parquet
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.atmos_features import AtmosFeatureComputer, ATMOS_FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Beach definitions
BEACH_COORDS = {
    'blacks': (32.885, -117.255),
    'torrey': (32.920, -117.260),
    'delmar': (32.960, -117.265),
    'solana': (32.990, -117.270),
    'sanelijo': (33.025, -117.280),
    'encinitas': (33.055, -117.285),
}

# Column mapping from PRISM variable names (legacy format)
PRISM_COLUMNS = {
    'ppt': 'precip_mm',
    'tmin': 'temp_min_c',
    'tmax': 'temp_max_c',
    'tmean': 'temp_mean_c',
    'tdmean': 'dewpoint_c',
}

# Column mapping from PRISM bulk export format
PRISM_BULK_COLUMNS = {
    'ppt (mm)': 'precip_mm',
    'tmin (degrees C)': 'temp_min_c',
    'tmax (degrees C)': 'temp_max_c',
    'tmean (degrees C)': 'temp_mean_c',
    'tdmean (degrees C)': 'dewpoint_c',
    'vpdmin (hPa)': 'vpdmin_hpa',
    'vpdmax (hPa)': 'vpdmax_hpa',
    'Date': 'date',
    'Name': 'beach',
}


def load_prism_bulk_data(bulk_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load PRISM data from bulk export CSVs.

    The bulk export format has:
    - 10 header lines to skip
    - All beaches in one file, differentiated by 'Name' column
    - Multiple yearly files to merge

    Args:
        bulk_dir: Directory containing PRISM bulk CSV files

    Returns:
        Dictionary mapping beach names to DataFrames
    """
    csv_files = sorted(bulk_dir.glob('PRISM_*.csv'))

    if not csv_files:
        logger.warning(f"No PRISM CSV files found in {bulk_dir}")
        return {}

    logger.info(f"Found {len(csv_files)} PRISM bulk CSV files")

    # Load and concatenate all files
    dfs = []
    for csv_path in csv_files:
        logger.info(f"  Loading {csv_path.name}...")
        df = pd.read_csv(csv_path, skiprows=10)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Total records: {len(combined)}")

    # Rename columns
    combined = combined.rename(columns=PRISM_BULK_COLUMNS)

    # Parse dates
    combined['date'] = pd.to_datetime(combined['date'])

    # Split by beach
    beach_data = {}
    for beach in combined['beach'].unique():
        beach_df = combined[combined['beach'] == beach].copy()
        beach_df = beach_df.drop(columns=['beach', 'Longitude', 'Latitude', 'Elevation (m)'])
        beach_df = beach_df.sort_values('date').reset_index(drop=True)
        beach_data[beach] = beach_df
        logger.info(f"  {beach}: {len(beach_df)} records")

    return beach_data


def load_raw_beach_data(
    raw_dir: Path,
    beach: str,
) -> Optional[pd.DataFrame]:
    """Load raw PRISM data for a beach.

    Looks for CSV file in format: {beach}_raw.csv

    Args:
        raw_dir: Directory containing raw CSV files
        beach: Beach name

    Returns:
        DataFrame with raw PRISM data or None if not found
    """
    csv_path = raw_dir / f"{beach}_raw.csv"

    if not csv_path.exists():
        logger.warning(f"Raw data file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path, parse_dates=['date'])

    # Rename columns if using PRISM variable names
    df = df.rename(columns=PRISM_COLUMNS)

    logger.info(f"Loaded {len(df)} records for {beach} from {csv_path}")

    return df


def create_synthetic_test_data(
    start_date: datetime,
    end_date: datetime,
    beach: str,
) -> pd.DataFrame:
    """Create synthetic test data for development/testing.

    Generates realistic-looking climate data for San Diego:
    - Mediterranean climate: wet winters, dry summers
    - Mild temperatures year-round
    - Occasional coastal fog

    Args:
        start_date: Start date
        end_date: End date
        beach: Beach name (for seed variation)

    Returns:
        DataFrame with synthetic climate data
    """
    # Use beach name for reproducible randomness
    seed = hash(beach) % (2**32)
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start_date, end_date, freq='D')
    n = len(dates)

    # Day of year for seasonal patterns
    doy = dates.dayofyear

    # Temperature (Mediterranean climate)
    # Mean around 18°C, seasonal variation of ±5°C
    temp_mean = 18 + 5 * np.sin(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 2, n)
    temp_range = 8 + 2 * rng.random(n)
    temp_min = temp_mean - temp_range / 2
    temp_max = temp_mean + temp_range / 2

    # Dewpoint (typically 10-15°C in San Diego)
    dewpoint = temp_min - 2 - 3 * rng.random(n)

    # Precipitation (rainy season Nov-Mar)
    # Use seasonal probability
    rain_prob = 0.3 * (1 + np.cos(2 * np.pi * (doy - 15) / 365)) / 2
    rain_occurs = rng.random(n) < rain_prob

    # Rain amount when it occurs (exponential distribution)
    precip = np.zeros(n)
    precip[rain_occurs] = rng.exponential(5, rain_occurs.sum())
    precip = np.clip(precip, 0, 100)  # Cap at 100mm

    df = pd.DataFrame({
        'date': dates,
        'precip_mm': precip,
        'temp_mean_c': temp_mean,
        'temp_min_c': temp_min,
        'temp_max_c': temp_max,
        'dewpoint_c': dewpoint,
    })

    return df


def process_beach(
    raw_dir: Path,
    output_dir: Path,
    beach: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    use_synthetic: bool = False,
) -> Optional[Path]:
    """Process raw data for a single beach.

    Args:
        raw_dir: Directory containing raw PRISM CSVs
        output_dir: Directory to save processed parquet files
        beach: Beach name
        start_date: Optional filter start date
        end_date: Optional filter end date
        use_synthetic: Generate synthetic data for testing

    Returns:
        Path to output parquet file, or None if failed
    """
    logger.info(f"Processing {beach}...")

    # Load raw data
    if use_synthetic:
        start = start_date or datetime(2017, 1, 1)
        end = end_date or datetime(2024, 12, 31)
        raw_df = create_synthetic_test_data(start, end, beach)
        logger.info(f"  Generated {len(raw_df)} days of synthetic data")
    else:
        raw_df = load_raw_beach_data(raw_dir, beach)

    if raw_df is None:
        return None

    # Filter by date range if specified
    if start_date:
        raw_df = raw_df[raw_df['date'] >= start_date]
    if end_date:
        raw_df = raw_df[raw_df['date'] <= end_date]

    if len(raw_df) == 0:
        logger.warning(f"  No data in specified date range for {beach}")
        return None

    # Compute features
    computer = AtmosFeatureComputer()
    features_df = computer.compute_all_features(raw_df)

    # Validate features
    missing_features = set(ATMOS_FEATURE_NAMES) - set(features_df.columns)
    if missing_features:
        logger.warning(f"  Missing features: {missing_features}")

    # Save to parquet
    output_path = output_dir / f"{beach}_atmos.parquet"
    features_df.to_parquet(output_path, index=False)

    # Log summary statistics
    logger.info(f"  Date range: {features_df['date'].min()} to {features_df['date'].max()}")
    logger.info(f"  Records: {len(features_df)}")
    logger.info(f"  Features: {len([c for c in features_df.columns if c != 'date'])}")

    # Check for NaN values
    nan_counts = features_df.isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    if len(nan_features) > 0:
        logger.info(f"  Features with NaN values:")
        for feat, count in nan_features.items():
            if feat != 'date':
                pct = 100 * count / len(features_df)
                logger.info(f"    {feat}: {count} ({pct:.1f}%)")

    logger.info(f"  Saved: {output_path}")

    return output_path


def validate_output(output_dir: Path, beaches: List[str]) -> Dict[str, dict]:
    """Validate processed output files.

    Args:
        output_dir: Directory containing processed parquet files
        beaches: List of beach names to validate

    Returns:
        Dictionary with validation results per beach
    """
    results = {}

    for beach in beaches:
        parquet_path = output_dir / f"{beach}_atmos.parquet"

        if not parquet_path.exists():
            results[beach] = {'status': 'missing'}
            continue

        df = pd.read_parquet(parquet_path)

        results[beach] = {
            'status': 'ok',
            'n_records': len(df),
            'date_range': (df['date'].min(), df['date'].max()),
            'n_features': len([c for c in df.columns if c != 'date']),
            'nan_pct': 100 * df.isna().mean().mean(),
        }

        # Check for expected features
        missing = set(ATMOS_FEATURE_NAMES) - set(df.columns)
        if missing:
            results[beach]['missing_features'] = list(missing)
            results[beach]['status'] = 'incomplete'

    return results


def process_bulk_data(
    bulk_dir: Path,
    output_dir: Path,
    beaches: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[str]:
    """Process PRISM bulk export data for all beaches.

    Args:
        bulk_dir: Directory containing PRISM bulk CSV files
        output_dir: Directory to save processed parquet files
        beaches: Optional list of beaches to process (default: all)
        start_date: Optional filter start date
        end_date: Optional filter end date

    Returns:
        List of successfully processed beach names
    """
    # Load all bulk data
    beach_data = load_prism_bulk_data(bulk_dir)

    if not beach_data:
        logger.error("No data loaded from bulk files")
        return []

    # Filter beaches if specified
    if beaches:
        beach_data = {k: v for k, v in beach_data.items() if k in beaches}

    processed = []
    computer = AtmosFeatureComputer()

    for beach, raw_df in beach_data.items():
        logger.info(f"\nProcessing {beach}...")

        # Filter by date range if specified
        if start_date:
            raw_df = raw_df[raw_df['date'] >= start_date]
        if end_date:
            raw_df = raw_df[raw_df['date'] <= end_date]

        if len(raw_df) == 0:
            logger.warning(f"  No data in specified date range for {beach}")
            continue

        logger.info(f"  Records: {len(raw_df)}")
        logger.info(f"  Date range: {raw_df['date'].min().date()} to {raw_df['date'].max().date()}")

        # Compute features
        features_df = computer.compute_all_features(raw_df)

        # Validate features
        missing_features = set(ATMOS_FEATURE_NAMES) - set(features_df.columns)
        if missing_features:
            logger.warning(f"  Missing features: {missing_features}")

        # Save to parquet
        output_path = output_dir / f"{beach}_atmos.parquet"
        features_df.to_parquet(output_path, index=False)

        # Log summary
        logger.info(f"  Output features: {len([c for c in features_df.columns if c != 'date'])}")

        # Check for NaN values in non-lookback period
        # (first 90 days will have NaN for rolling features)
        valid_df = features_df[features_df['date'] >= features_df['date'].min() + pd.Timedelta(days=90)]
        if len(valid_df) > 0:
            nan_pct = 100 * valid_df.drop(columns=['date']).isna().mean().mean()
            logger.info(f"  NaN % (after 90-day warmup): {nan_pct:.2f}%")

        logger.info(f"  Saved: {output_path}")
        processed.append(beach)

    return processed


def main():
    parser = argparse.ArgumentParser(
        description='Prepare atmospheric features for CliffCast',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--bulk-dir',
        type=Path,
        help='Directory containing PRISM bulk export CSVs (from bulk.php)'
    )
    input_group.add_argument(
        '--raw-dir',
        type=Path,
        help='Directory containing legacy individual beach CSV files'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed/atmospheric'),
        help='Directory to save processed parquet files'
    )
    parser.add_argument(
        '--beaches',
        nargs='+',
        default=None,
        help='Beaches to process (default: all found in data)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Filter start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='Filter end date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic test data instead of using real data'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing output files'
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine beaches to validate
    beaches_to_validate = args.beaches or list(BEACH_COORDS.keys())

    if args.validate_only:
        # Validation only
        logger.info("Validating existing output files...")
        results = validate_output(args.output_dir, beaches_to_validate)

        for beach, info in results.items():
            if info['status'] == 'missing':
                logger.warning(f"{beach}: MISSING")
            elif info['status'] == 'incomplete':
                logger.warning(f"{beach}: INCOMPLETE - missing {info.get('missing_features')}")
            else:
                logger.info(
                    f"{beach}: OK - {info['n_records']} records, "
                    f"{info['n_features']} features, "
                    f"{info['nan_pct']:.1f}% NaN"
                )
        return

    # Process data
    if args.bulk_dir:
        # Use bulk format
        logger.info(f"Processing PRISM bulk data from {args.bulk_dir}")
        logger.info(f"Output directory: {args.output_dir}")

        processed = process_bulk_data(
            bulk_dir=args.bulk_dir,
            output_dir=args.output_dir,
            beaches=args.beaches,
            start_date=start_date,
            end_date=end_date,
        )
        failed = [b for b in beaches_to_validate if b not in processed]

    elif args.raw_dir or args.synthetic:
        # Use legacy format or synthetic data
        raw_dir = args.raw_dir or Path('data/raw/prism')
        logger.info(f"Processing {'synthetic' if args.synthetic else 'legacy'} data...")
        logger.info(f"Output directory: {args.output_dir}")

        processed = []
        failed = []

        for beach in beaches_to_validate:
            result = process_beach(
                raw_dir=raw_dir,
                output_dir=args.output_dir,
                beach=beach,
                start_date=start_date,
                end_date=end_date,
                use_synthetic=args.synthetic,
            )

            if result:
                processed.append(beach)
            else:
                failed.append(beach)
    else:
        parser.error("Must specify either --bulk-dir, --raw-dir, or --synthetic")

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Processing complete!")
    logger.info(f"  Processed: {len(processed)} beaches")
    if failed:
        logger.warning(f"  Failed: {len(failed)} beaches: {failed}")

    # Validate output
    logger.info("\nValidating output...")
    results = validate_output(args.output_dir, processed)

    all_ok = all(r.get('status') == 'ok' for r in results.values())
    if all_ok:
        logger.info("All outputs validated successfully!")
    else:
        logger.warning("Some outputs have issues - check logs above")


if __name__ == '__main__':
    main()
