#!/usr/bin/env python3
"""Download PRISM daily climate data for San Diego beaches.

PRISM (Parameter-elevation Regressions on Independent Slopes Model) provides
high-quality gridded climate data at 4km resolution. This script downloads
precipitation and temperature variables needed for the CliffCast atmospheric
feature pipeline.

Usage:
    # Download for specific date range
    python scripts/processing/download_prism_data.py \
        --start-date 2017-01-01 \
        --end-date 2024-12-31 \
        --output data/raw/prism/

    # Download based on survey dates (auto-detect range from master_list.csv)
    python scripts/processing/download_prism_data.py \
        --survey-csv data/raw/master_list.csv \
        --lookback-days 90 \
        --output data/raw/prism/

    # Download specific variables only
    python scripts/processing/download_prism_data.py \
        --start-date 2017-01-01 \
        --end-date 2024-12-31 \
        --variables ppt tmin tmax \
        --output data/raw/prism/

Variables downloaded:
    - ppt: Daily precipitation (mm)
    - tmin: Daily minimum temperature (°C)
    - tmax: Daily maximum temperature (°C)
    - tmean: Daily mean temperature (°C)
    - tdmean: Daily dewpoint temperature (°C)
"""

import argparse
import logging
import os
import struct
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Beach coordinates for San Diego study area
BEACH_COORDS = {
    'blacks': (32.885, -117.255),
    'torrey': (32.920, -117.260),
    'delmar': (32.960, -117.265),
    'solana': (32.990, -117.270),
    'sanelijo': (33.025, -117.280),
    'encinitas': (33.055, -117.285),
}


class PRISMDownloader:
    """Download PRISM daily climate data.

    PRISM provides daily gridded climate data at 4km resolution. Data is
    distributed as BIL (Band Interleaved by Line) format files within
    ZIP archives, accessible via anonymous HTTPS/FTP.

    Attributes:
        output_dir: Directory to save downloaded files
        variables: List of PRISM variables to download
    """

    # PRISM data directory URL (anonymous access)
    BASE_URL = "https://data.prism.oregonstate.edu/time_series/us/an/4km"

    # URL pattern for daily data files
    # Format: {BASE_URL}/{var}/daily/{year}/prism_{var}_us_25m_{YYYYMMDD}.zip
    DAILY_URL = "{base}/{var}/daily/{year}/prism_{var}_us_25m_{date}.zip"

    # Available variables
    VARIABLES = ['ppt', 'tmin', 'tmax', 'tmean', 'tdmean', 'vpdmin', 'vpdmax']

    def __init__(
        self,
        output_dir: Path,
        variables: Optional[List[str]] = None,
    ):
        """Initialize PRISM downloader.

        Args:
            output_dir: Directory to save downloaded files
            variables: List of variables to download (default: all)
        """
        self.output_dir = Path(output_dir)
        self.variables = variables or self.VARIABLES

        # Validate variables
        for var in self.variables:
            if var not in self.VARIABLES:
                raise ValueError(
                    f"Unknown variable '{var}'. "
                    f"Valid options: {self.VARIABLES}"
                )

        # Create output directories
        for var in self.variables:
            (self.output_dir / var).mkdir(parents=True, exist_ok=True)

    def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        skip_existing: bool = True,
        retry_count: int = 3,
        retry_delay: float = 5.0,
    ) -> Dict[str, List[Path]]:
        """Download PRISM data for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            skip_existing: Skip files that already exist
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            Dictionary mapping variable names to lists of downloaded file paths
        """
        downloaded = {var: [] for var in self.variables}

        # Generate date list
        current = start_date
        dates = []
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        total_downloads = len(dates) * len(self.variables)
        completed = 0

        logger.info(
            f"Downloading PRISM data for {len(dates)} days, "
            f"{len(self.variables)} variables ({total_downloads} files total)"
        )

        for date in dates:
            for var in self.variables:
                completed += 1

                # Check if file exists
                output_path = self._get_output_path(var, date)
                if skip_existing and output_path.exists():
                    downloaded[var].append(output_path)
                    continue

                # Download with retries
                success = False
                for attempt in range(retry_count):
                    try:
                        self._download_single(var, date, output_path)
                        downloaded[var].append(output_path)
                        success = True
                        break
                    except (URLError, HTTPError) as e:
                        logger.warning(
                            f"Attempt {attempt + 1}/{retry_count} failed for "
                            f"{var} {date.strftime('%Y-%m-%d')}: {e}"
                        )
                        if attempt < retry_count - 1:
                            time.sleep(retry_delay)

                if not success:
                    logger.error(
                        f"Failed to download {var} for {date.strftime('%Y-%m-%d')} "
                        f"after {retry_count} attempts"
                    )

                # Progress update
                if completed % 100 == 0 or completed == total_downloads:
                    logger.info(f"Progress: {completed}/{total_downloads} files")

        return downloaded

    def _get_output_path(self, var: str, date: datetime) -> Path:
        """Get output file path for a variable and date."""
        filename = f"PRISM_{var}_{date.strftime('%Y%m%d')}.tif"
        return self.output_dir / var / filename

    def _download_single(
        self,
        var: str,
        date: datetime,
        output_path: Path,
    ) -> None:
        """Download a single PRISM file.

        Downloads ZIP file from PRISM data server and extracts the BIL file.
        Uses anonymous HTTPS access (no authentication required).
        """
        import zipfile
        import tempfile
        from urllib.request import urlretrieve

        date_str = date.strftime('%Y%m%d')
        year = date.year

        # Build download URL
        url = self.DAILY_URL.format(
            base=self.BASE_URL,
            var=var,
            year=year,
            date=date_str
        )

        logger.debug(f"Downloading: {url}")

        # Download to temporary file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            urlretrieve(url, tmp_path)

            # Extract GeoTIFF file from ZIP
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                # Find the .tif file in the archive
                tif_files = [f for f in zf.namelist() if f.endswith('.tif')]
                if not tif_files:
                    raise ValueError(f"No .tif file found in {url}")

                # Extract to output directory
                tif_name = tif_files[0]
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Change output path to .tif extension
                output_tif = output_path.with_suffix('.tif')

                # Extract and rename to our standard naming
                with zf.open(tif_name) as src:
                    with open(output_tif, 'wb') as dst:
                        dst.write(src.read())

            logger.debug(f"Extracted: {output_tif}")

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def extract_grid_value(
        self,
        tif_path: Path,
        lat: float,
        lon: float,
    ) -> float:
        """Extract value at a coordinate from PRISM GeoTIFF file.

        Args:
            tif_path: Path to GeoTIFF file
            lat: Latitude (decimal degrees)
            lon: Longitude (decimal degrees)

        Returns:
            Value at the specified coordinate, or NaN if no data
        """
        try:
            import rasterio
            from rasterio.transform import rowcol
        except ImportError:
            raise ImportError(
                "rasterio is required for reading GeoTIFF files. "
                "Install with: pip install rasterio"
            )

        with rasterio.open(tif_path) as src:
            # Convert lat/lon to row/col using the raster's transform
            try:
                row, col = rowcol(src.transform, lon, lat)
            except Exception:
                logger.warning(f"Coordinate ({lat}, {lon}) outside raster bounds")
                return np.nan

            # Validate indices
            if not (0 <= row < src.height and 0 <= col < src.width):
                logger.warning(f"Coordinate ({lat}, {lon}) outside raster bounds")
                return np.nan

            # Read the value
            value = src.read(1)[row, col]

            # Check for nodata
            if src.nodata is not None and value == src.nodata:
                return np.nan

            return float(value)

    def extract_beach_values(
        self,
        date: datetime,
        beaches: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Extract values for all beaches on a given date.

        Args:
            date: Date to extract values for
            beaches: List of beach names (default: all)

        Returns:
            Nested dict: {beach: {variable: value}}
        """
        beaches = beaches or list(BEACH_COORDS.keys())
        results = {beach: {} for beach in beaches}

        for var in self.variables:
            bil_path = self._get_output_path(var, date)

            if not bil_path.exists():
                logger.warning(f"Missing file: {bil_path}")
                for beach in beaches:
                    results[beach][var] = np.nan
                continue

            for beach in beaches:
                lat, lon = BEACH_COORDS[beach]
                results[beach][var] = self.extract_grid_value(bil_path, lat, lon)

        return results

    def extract_timeseries(
        self,
        beach: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Extract time series for a single beach.

        Args:
            beach: Beach name
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with columns: date, ppt, tmin, tmax, tmean, tdmean
        """
        if beach not in BEACH_COORDS:
            raise ValueError(f"Unknown beach '{beach}'. Valid: {list(BEACH_COORDS.keys())}")

        lat, lon = BEACH_COORDS[beach]

        records = []
        current = start_date
        while current <= end_date:
            record = {'date': current}

            for var in self.variables:
                bil_path = self._get_output_path(var, current)
                if bil_path.exists():
                    record[var] = self.extract_grid_value(bil_path, lat, lon)
                else:
                    record[var] = np.nan

            records.append(record)
            current += timedelta(days=1)

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])

        return df


def get_survey_date_range(
    survey_csv: Path,
    lookback_days: int = 90,
) -> Tuple[datetime, datetime]:
    """Get date range needed based on survey dates.

    Args:
        survey_csv: Path to survey CSV (master_list.csv)
        lookback_days: Days of lookback needed before earliest survey

    Returns:
        Tuple of (start_date, end_date)
    """
    df = pd.read_csv(survey_csv)

    # Parse dates (format: YYYYMMDD)
    dates = pd.to_datetime(df['date'], format='%Y%m%d')

    start_date = dates.min() - timedelta(days=lookback_days)
    end_date = dates.max()

    logger.info(
        f"Survey date range: {dates.min().date()} to {dates.max().date()}"
    )
    logger.info(
        f"With {lookback_days}-day lookback: {start_date.date()} to {end_date.date()}"
    )

    return start_date.to_pydatetime(), end_date.to_pydatetime()


def main():
    parser = argparse.ArgumentParser(
        description='Download PRISM daily climate data for CliffCast',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Date range options (mutually exclusive with survey-csv)
    date_group = parser.add_argument_group('Date Range')
    date_group.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    date_group.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )

    # Survey-based date range
    survey_group = parser.add_argument_group('Survey-Based Range')
    survey_group.add_argument(
        '--survey-csv',
        type=Path,
        help='Path to survey CSV to auto-detect date range'
    )
    survey_group.add_argument(
        '--lookback-days',
        type=int,
        default=90,
        help='Days of lookback before earliest survey (default: 90)'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/raw/prism'),
        help='Output directory (default: data/raw/prism)'
    )
    parser.add_argument(
        '--variables',
        nargs='+',
        choices=PRISMDownloader.VARIABLES,
        default=PRISMDownloader.VARIABLES,
        help='Variables to download (default: all)'
    )

    # Download options
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip files that already exist (default: True)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download of existing files'
    )
    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='Skip download, only extract values from existing files'
    )

    args = parser.parse_args()

    # Determine date range
    if args.survey_csv:
        start_date, end_date = get_survey_date_range(
            args.survey_csv,
            args.lookback_days
        )
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        parser.error(
            "Must specify either --survey-csv or both --start-date and --end-date"
        )

    # Initialize downloader
    downloader = PRISMDownloader(
        output_dir=args.output,
        variables=args.variables,
    )

    if args.extract_only:
        # Extract values for all beaches
        logger.info("Extracting values from existing files...")
        for beach in BEACH_COORDS:
            logger.info(f"Extracting time series for {beach}...")
            df = downloader.extract_timeseries(beach, start_date, end_date)

            # Save to CSV
            output_csv = args.output / f"{beach}_raw.csv"
            df.to_csv(output_csv, index=False)
            logger.info(f"Saved: {output_csv}")

            # Show summary
            logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"  Records: {len(df)}")
            for var in args.variables:
                valid = df[var].notna().sum()
                logger.info(f"  {var}: {valid}/{len(df)} valid values")
    else:
        # Download data
        logger.info(
            f"Downloading PRISM data from {start_date.date()} to {end_date.date()}"
        )

        try:
            downloaded = downloader.download_date_range(
                start_date,
                end_date,
                skip_existing=not args.force,
            )

            # Summary
            for var, files in downloaded.items():
                logger.info(f"Downloaded {len(files)} files for {var}")

        except NotImplementedError as e:
            logger.error(str(e))
            logger.info("\nAlternative: Use Google Earth Engine to download PRISM data.")
            logger.info("See: https://developers.google.com/earth-engine/datasets/catalog/OREGONSTATE_PRISM_AN81d")
            sys.exit(1)


if __name__ == '__main__':
    main()
