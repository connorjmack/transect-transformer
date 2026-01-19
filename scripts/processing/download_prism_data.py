#!/usr/bin/env python3
"""Download PRISM daily climate data for San Diego beaches.

PRISM (Parameter-elevation Regressions on Independent Slopes Model) provides
high-quality gridded climate data at 4km resolution. This script downloads
precipitation and temperature variables needed for the CliffCast atmospheric
feature pipeline.

**Storage mode**: Point-only extraction. Downloads each grid, extracts values
at beach coordinates, then deletes the grid. Only CSV time series are stored.

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

    # Keep full TIF grids (for statewide expansion)
    python scripts/processing/download_prism_data.py \
        --start-date 2017-01-01 \
        --end-date 2024-12-31 \
        --keep-grids \
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
import re
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


# Beach coordinates for San Diego study area (centroids from MOP transect shapefile)
# Calculated from MOP transect midpoints for each beach's MOP range
BEACH_COORDS = {
    'blacks': (32.893798, -117.253693),      # MOP 520-567
    'torrey': (32.920009, -117.259033),      # MOP 567-581
    'delmar': (32.949681, -117.265282),      # MOP 595-620
    'solana': (32.988459, -117.274256),      # MOP 637-666
    'sanelijo': (33.026229, -117.287640),    # MOP 683-708
    'encinitas': (33.059110, -117.303186),   # MOP 708-764
}

# Beach MOP ranges (for dynamic coordinate calculation)
BEACH_MOP_RANGES = {
    'blacks': (520, 567),
    'torrey': (567, 581),
    'delmar': (595, 620),
    'solana': (637, 666),
    'sanelijo': (683, 708),
    'encinitas': (708, 764),
}


class PRISMDownloader:
    """Download PRISM daily climate data with point-only extraction.

    PRISM provides daily gridded climate data at 4km resolution. Data is
    distributed as GeoTIFF files within ZIP archives, accessible via
    anonymous HTTPS.

    By default, this downloads each grid, extracts values at beach coordinates,
    then deletes the grid to minimize storage. Use keep_grids=True to retain
    full CONUS grids for statewide expansion.

    Attributes:
        output_dir: Directory to save downloaded/extracted files
        variables: List of PRISM variables to download
        keep_grids: Whether to retain full TIF grids after extraction
        beach_coords: Dictionary of beach coordinates {name: (lat, lon)}
    """

    # PRISM data directory URL (anonymous access)
    BASE_URL = "https://data.prism.oregonstate.edu/time_series/us/an/4km"

    # URL pattern for daily data files
    # Format: {BASE_URL}/{var}/daily/{year}/prism_{var}_us_25m_{YYYYMMDD}.zip
    DAILY_URL = "{base}/{var}/daily/{year}/prism_{var}_us_25m_{date}.zip"

    # Available variables (core set for CliffCast)
    VARIABLES = ['ppt', 'tmin', 'tmax', 'tmean', 'tdmean']

    def __init__(
        self,
        output_dir: Path,
        variables: Optional[List[str]] = None,
        keep_grids: bool = False,
        beach_coords: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """Initialize PRISM downloader.

        Args:
            output_dir: Directory to save downloaded files
            variables: List of variables to download (default: core 5)
            keep_grids: Whether to keep full TIF grids (default: False)
            beach_coords: Beach coordinates {name: (lat, lon)} (default: SD beaches)
        """
        self.output_dir = Path(output_dir)
        self.variables = variables or self.VARIABLES
        self.keep_grids = keep_grids
        self.beach_coords = beach_coords or BEACH_COORDS

        # Validate variables
        all_vars = ['ppt', 'tmin', 'tmax', 'tmean', 'tdmean', 'vpdmin', 'vpdmax']
        for var in self.variables:
            if var not in all_vars:
                raise ValueError(
                    f"Unknown variable '{var}'. "
                    f"Valid options: {all_vars}"
                )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create grid directories only if keeping grids
        if self.keep_grids:
            for var in self.variables:
                (self.output_dir / var).mkdir(parents=True, exist_ok=True)

    def download_and_extract(
        self,
        start_date: datetime,
        end_date: datetime,
        retry_count: int = 3,
        retry_delay: float = 5.0,
    ) -> Dict[str, pd.DataFrame]:
        """Download PRISM data and extract point values for all beaches.

        This is the main entry point. For each date:
        1. Downloads all variable grids
        2. Extracts values at each beach coordinate
        3. Deletes grids (unless keep_grids=True)
        4. Accumulates results in DataFrames

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            Dictionary mapping beach names to DataFrames with columns:
            [date, ppt, tmin, tmax, tmean, tdmean]
        """
        import tempfile
        import rasterio
        from rasterio.transform import rowcol

        # Initialize result accumulators
        beach_data = {beach: [] for beach in self.beach_coords}

        # Generate date list
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        total_days = len(dates)
        logger.info(
            f"Downloading PRISM data for {total_days} days, "
            f"{len(self.variables)} variables, {len(self.beach_coords)} beaches"
        )
        if not self.keep_grids:
            logger.info("Point-only mode: grids will be deleted after extraction")

        for day_idx, date in enumerate(dates):
            # Dictionary to hold this day's values for each beach
            day_values = {beach: {'date': date} for beach in self.beach_coords}

            # Download and extract each variable
            for var in self.variables:
                tif_path = None
                try:
                    # Download to temp file or permanent location
                    if self.keep_grids:
                        tif_path = self._get_output_path(var, date)
                        if not tif_path.exists():
                            self._download_single(var, date, tif_path, retry_count, retry_delay)
                    else:
                        # Download to temp file
                        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                            tif_path = Path(tmp.name)
                        self._download_single(var, date, tif_path, retry_count, retry_delay)

                    # Extract values for all beaches
                    with rasterio.open(tif_path) as src:
                        data = src.read(1)
                        for beach, (lat, lon) in self.beach_coords.items():
                            try:
                                row, col = rowcol(src.transform, lon, lat)
                                if 0 <= row < src.height and 0 <= col < src.width:
                                    value = float(data[row, col])
                                    if src.nodata is not None and value == src.nodata:
                                        value = np.nan
                                else:
                                    value = np.nan
                            except Exception:
                                value = np.nan
                            day_values[beach][var] = value

                except Exception as e:
                    logger.warning(f"Failed to get {var} for {date.strftime('%Y-%m-%d')}: {e}")
                    for beach in self.beach_coords:
                        day_values[beach][var] = np.nan

                finally:
                    # Delete temp file if not keeping grids
                    if not self.keep_grids and tif_path and tif_path.exists():
                        try:
                            tif_path.unlink()
                        except Exception:
                            pass

            # Append day's values to each beach's list
            for beach in self.beach_coords:
                beach_data[beach].append(day_values[beach])

            # Progress update
            if (day_idx + 1) % 10 == 0 or day_idx == total_days - 1:
                logger.info(f"Progress: {day_idx + 1}/{total_days} days")

        # Convert to DataFrames
        result = {}
        for beach, records in beach_data.items():
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            # Ensure column order
            cols = ['date'] + [v for v in self.variables if v in df.columns]
            result[beach] = df[cols]

        return result

    def save_beach_csvs(
        self,
        beach_data: Dict[str, pd.DataFrame],
        suffix: str = '_raw',
    ) -> List[Path]:
        """Save beach DataFrames to CSV files.

        Args:
            beach_data: Dictionary from download_and_extract()
            suffix: Suffix for CSV filenames (default: '_raw')

        Returns:
            List of saved file paths
        """
        saved_paths = []
        for beach, df in beach_data.items():
            csv_path = self.output_dir / f"{beach}{suffix}.csv"
            df.to_csv(csv_path, index=False)
            saved_paths.append(csv_path)
            logger.info(f"Saved: {csv_path} ({len(df)} records)")
        return saved_paths

    def _get_output_path(self, var: str, date: datetime) -> Path:
        """Get output file path for a variable and date."""
        filename = f"PRISM_{var}_{date.strftime('%Y%m%d')}.tif"
        return self.output_dir / var / filename

    def _download_single(
        self,
        var: str,
        date: datetime,
        output_path: Path,
        retry_count: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        """Download a single PRISM file with retries.

        Downloads ZIP file from PRISM data server and extracts the GeoTIFF.
        Uses anonymous HTTPS access (no authentication required).

        Args:
            var: Variable name (ppt, tmin, etc.)
            date: Date to download
            output_path: Where to save the extracted TIF
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
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

        last_error = None
        for attempt in range(retry_count):
            tmp_path = None
            try:
                # Download to temporary file
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                    tmp_path = tmp.name

                urlretrieve(url, tmp_path)

                # Extract GeoTIFF file from ZIP
                with zipfile.ZipFile(tmp_path, 'r') as zf:
                    # Find the .tif file in the archive
                    tif_files = [f for f in zf.namelist() if f.endswith('.tif')]
                    if not tif_files:
                        raise ValueError(f"No .tif file found in {url}")

                    # Extract to output path
                    tif_name = tif_files[0]
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    with zf.open(tif_name) as src:
                        with open(output_path, 'wb') as dst:
                            dst.write(src.read())

                logger.debug(f"Downloaded: {var} {date_str}")
                return  # Success

            except (URLError, HTTPError, Exception) as e:
                last_error = e
                if attempt < retry_count - 1:
                    logger.debug(f"Retry {attempt + 1}/{retry_count} for {var} {date_str}: {e}")
                    time.sleep(retry_delay)

            finally:
                # Clean up temp file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        # All retries failed
        raise last_error or Exception(f"Failed to download {var} for {date_str}")

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
        help='Variables to download (default: ppt tmin tmax tmean tdmean)'
    )

    # Storage options
    parser.add_argument(
        '--keep-grids',
        action='store_true',
        help='Keep full CONUS TIF grids (for statewide expansion). '
             'Default: point-only extraction, no TIF storage.'
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
        keep_grids=args.keep_grids,
    )

    # Download and extract
    logger.info(
        f"Downloading PRISM data from {start_date.date()} to {end_date.date()}"
    )
    logger.info(f"Variables: {', '.join(args.variables)}")
    logger.info(f"Storage mode: {'keep grids' if args.keep_grids else 'point-only'}")

    try:
        beach_data = downloader.download_and_extract(start_date, end_date)

        # Save CSVs
        saved_paths = downloader.save_beach_csvs(beach_data)

        # Summary
        logger.info("\n=== Download Summary ===")
        for beach, df in beach_data.items():
            valid_pct = (df['ppt'].notna().sum() / len(df)) * 100
            logger.info(
                f"{beach:12s}: {len(df)} days, "
                f"{valid_pct:.0f}% valid, "
                f"precip range: {df['ppt'].min():.1f}-{df['ppt'].max():.1f} mm"
            )

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
