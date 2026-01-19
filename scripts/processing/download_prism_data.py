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
        --survey-csv data/survey_lists/master_list.csv \
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
    distributed as BIL (Band Interleaved by Line) format files with
    accompanying header files.

    Attributes:
        output_dir: Directory to save downloaded files
        variables: List of PRISM variables to download
    """

    # PRISM base URL for daily data
    BASE_URL = "https://prism.oregonstate.edu/fetchData.php"

    # Alternative direct download URL pattern
    # Format: PRISM_{var}_stable_4kmD2_{YYYYMMDD}_bil.zip
    DIRECT_URL = "https://prism.oregonstate.edu/downloads/grid/{var}/daily/{year}/PRISM_{var}_stable_4kmD2_{date}_bil.zip"

    # Available variables
    VARIABLES = ['ppt', 'tmin', 'tmax', 'tmean', 'tdmean']

    # PRISM grid parameters (CONUS)
    GRID_PARAMS = {
        'nrows': 621,
        'ncols': 1405,
        'xllcorner': -125.0208333,
        'yllcorner': 24.0625000,
        'cellsize': 0.0416667,  # ~4km
        'nodata': -9999,
    }

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
        filename = f"PRISM_{var}_stable_4kmD2_{date.strftime('%Y%m%d')}_bil.bil"
        return self.output_dir / var / filename

    def _download_single(
        self,
        var: str,
        date: datetime,
        output_path: Path,
    ) -> None:
        """Download a single PRISM file.

        Note: PRISM requires authentication for bulk downloads. For research
        use, you may need to register at https://prism.oregonstate.edu/ and
        use their official download tools.

        This implementation provides a template that can be adapted based on
        your access method (direct download, API, or FTP).
        """
        # PRISM data is available through their web interface
        # For bulk downloads, consider using:
        # 1. PRISM's official download tools
        # 2. Google Earth Engine (has PRISM dataset)
        # 3. Climate Engine (https://climateengine.com)

        # Placeholder for actual download logic
        # The exact method depends on your PRISM access arrangement

        date_str = date.strftime('%Y%m%d')
        year = date.year

        # Example URL pattern (may need adjustment based on PRISM's current API)
        url = self.DIRECT_URL.format(var=var, year=year, date=date_str)

        logger.debug(f"Downloading: {url}")

        # For now, create a placeholder that indicates download is needed
        # In production, replace with actual download logic
        raise NotImplementedError(
            f"PRISM download requires authentication. "
            f"Please download manually from {url} or use Google Earth Engine. "
            f"See: https://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf"
        )

    def extract_grid_value(
        self,
        bil_path: Path,
        lat: float,
        lon: float,
    ) -> float:
        """Extract value at a coordinate from PRISM BIL grid file.

        Args:
            bil_path: Path to BIL file
            lat: Latitude (decimal degrees)
            lon: Longitude (decimal degrees)

        Returns:
            Value at the specified coordinate, or NaN if no data
        """
        # Calculate grid indices from coordinates
        row, col = self._coord_to_index(lat, lon)

        # Validate indices
        if not (0 <= row < self.GRID_PARAMS['nrows'] and
                0 <= col < self.GRID_PARAMS['ncols']):
            logger.warning(
                f"Coordinate ({lat}, {lon}) outside PRISM grid bounds"
            )
            return np.nan

        # Read value from BIL file
        # BIL format: Band Interleaved by Line, 32-bit float
        with open(bil_path, 'rb') as f:
            # Seek to the correct position
            # BIL stores data row by row, 4 bytes per float32 value
            offset = (row * self.GRID_PARAMS['ncols'] + col) * 4
            f.seek(offset)

            # Read single float32 value
            value = struct.unpack('f', f.read(4))[0]

        # Check for no-data value
        if value == self.GRID_PARAMS['nodata'] or value < -999:
            return np.nan

        return value

    def _coord_to_index(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to PRISM grid row/col indices.

        Args:
            lat: Latitude (decimal degrees)
            lon: Longitude (decimal degrees)

        Returns:
            Tuple of (row, col) indices
        """
        params = self.GRID_PARAMS

        # PRISM grid starts from lower-left corner
        # Row 0 is at the top (north), so we need to invert
        col = int((lon - params['xllcorner']) / params['cellsize'])
        row = int((params['yllcorner'] + params['nrows'] * params['cellsize'] - lat)
                  / params['cellsize'])

        return row, col

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
