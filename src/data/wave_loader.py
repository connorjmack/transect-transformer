"""Wave data loading for CliffCast model training.

This module provides the WaveLoader class which loads CDIP wave data aligned to
LiDAR scan dates. Wave data is loaded from pre-downloaded NetCDF files and
returned as numpy arrays ready for the model.

Usage:
    from src.data.wave_loader import WaveLoader

    loader = WaveLoader('data/raw/cdip/')

    # Get wave features for a single scan
    features, doy = loader.get_wave_for_scan(mop_id=582, scan_date='2023-12-15')

    # Get features for a batch
    features, doy = loader.get_batch_wave(mop_ids, scan_dates)
"""

import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .cdip_wave_loader import CDIPWaveLoader, WaveData

logger = logging.getLogger(__name__)


class WaveLoader:
    """Load wave data aligned to scan dates.

    This class manages loading pre-downloaded CDIP wave data from NetCDF files
    and aligning them to LiDAR scan dates for model training.

    Follows the exact pattern from AtmosphericLoader for consistency.

    Attributes:
        cdip_dir: Directory containing CDIP NetCDF files
        lookback_days: Number of days to look back from scan date
        resample_hours: Resampling interval in hours
        n_features: Number of wave features (4)
    """

    def __init__(
        self,
        cdip_dir: Union[str, Path],
        lookback_days: int = 90,
        resample_hours: int = 6,
        cache_size: int = 50,
    ):
        """Initialize wave loader.

        Args:
            cdip_dir: Directory containing D{mop}_hindcast.nc files
            lookback_days: Days to look back from scan date (default: 90)
            resample_hours: Resample interval in hours (default: 6)
            cache_size: Number of WaveData objects to keep in memory
        """
        self.cdip_dir = Path(cdip_dir)
        self.lookback_days = lookback_days
        self.resample_hours = resample_hours
        self.n_features = 4  # hs, tp, dp, power

        # Validate directory exists
        if not self.cdip_dir.exists():
            raise FileNotFoundError(f"CDIP data directory not found: {self.cdip_dir}")

        # Cache loaded WaveData objects
        self._cache: OrderedDict[int, WaveData] = OrderedDict()
        self._cache_size = cache_size

        # Initialize CDIP loader for local files
        self._cdip_loader = CDIPWaveLoader(local_dir=self.cdip_dir)

        # Track available MOPs
        self._available_mops = self._scan_available_mops()

        if not self._available_mops:
            logger.warning(f"No CDIP data files found in {self.cdip_dir}")

    def _scan_available_mops(self) -> List[int]:
        """Scan directory for available MOP data files.

        Returns:
            List of MOP IDs with downloaded NetCDF files
        """
        available = []

        # Scan for NetCDF files matching expected patterns
        for file_path in self.cdip_dir.glob("D*_hindcast.nc"):
            try:
                # Extract MOP ID from filename
                # Handles: D0582_hindcast.nc, D582_hindcast.nc, etc.
                stem = file_path.stem  # D0582_hindcast or D582_hindcast
                mop_str = stem.split('_')[0][1:]  # Remove 'D' prefix
                mop_id = int(mop_str)
                available.append(mop_id)
            except (ValueError, IndexError):
                logger.debug(f"Could not parse MOP ID from {file_path.name}")
                continue

        return sorted(available)

    def _get_site_label_from_mop(self, mop_id: int) -> Optional[str]:
        """Find the site label for a MOP by checking which file exists.

        Args:
            mop_id: MOP transect ID

        Returns:
            Site label if file found, None otherwise
        """
        # Try multiple formats
        formats = [
            f"D{mop_id:04d}",  # D0582
            f"D{mop_id:03d}",  # D582
            f"D{mop_id}",      # D582
        ]

        for site_label in formats:
            file_path = self.cdip_dir / f"{site_label}_hindcast.nc"
            if file_path.exists():
                return site_label

        return None

    def _load_wave_data(self, mop_id: int) -> WaveData:
        """Load and cache wave data for a MOP.

        Args:
            mop_id: MOP transect ID

        Returns:
            WaveData object

        Raises:
            FileNotFoundError: If wave data file not found
        """
        # Check cache first
        if mop_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(mop_id)
            return self._cache[mop_id]

        # Find site label
        site_label = self._get_site_label_from_mop(mop_id)
        if site_label is None:
            raise FileNotFoundError(
                f"No CDIP data found for MOP {mop_id} in {self.cdip_dir}"
            )

        # Load wave data (reads entire file from disk)
        logger.debug(f"Loading wave data for MOP {mop_id} (site {site_label})")
        wave_data = self._cdip_loader.load_mop(
            mop_id=mop_id,
            site_label_override=site_label,
        )

        # Manage cache size (LRU eviction)
        if len(self._cache) >= self._cache_size:
            # Remove oldest (first) item
            oldest_mop = next(iter(self._cache))
            del self._cache[oldest_mop]
            logger.debug(f"Evicted MOP {oldest_mop} from cache (LRU)")

        # Add to cache
        self._cache[mop_id] = wave_data

        return wave_data

    def get_wave_for_scan(
        self,
        mop_id: int,
        scan_date: Union[datetime, pd.Timestamp, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get wave features for a single scan.

        Extracts the wave time series for [scan_date - lookback_days, scan_date].

        Args:
            mop_id: MOP transect ID
            scan_date: Scan date (datetime, Timestamp, or ordinal day)

        Returns:
            Tuple of:
                features: (T_w, 4) array of [hs, tp, dp, power]
                day_of_year: (T_w,) array for seasonality encoding
        """
        # Convert ordinal to datetime if needed
        if isinstance(scan_date, int):
            scan_date = datetime.fromordinal(scan_date)
        elif isinstance(scan_date, pd.Timestamp):
            scan_date = scan_date.to_pydatetime()

        # Load wave data for this MOP
        try:
            wave_data = self._load_wave_data(mop_id)
        except FileNotFoundError as e:
            logger.warning(
                f"Wave data not available for MOP {mop_id}, returning zeros. Error: {e}"
            )
            # Return zeros as graceful degradation
            T_w = int((self.lookback_days * 24) / self.resample_hours)
            features = np.zeros((T_w, self.n_features), dtype=np.float32)

            # Generate day-of-year for time window
            start_date = scan_date - timedelta(days=self.lookback_days - 1)
            dates = pd.date_range(start_date, scan_date, freq=f'{self.resample_hours}h')
            doy = dates.dayofyear.values[:T_w].astype(np.int32)

            return features, doy

        # Use to_tensor method from WaveData
        try:
            features, doy = wave_data.to_tensor(
                history_days=self.lookback_days,
                reference_date=scan_date,
                resample_hours=self.resample_hours,
            )
        except ValueError as e:
            logger.warning(
                f"Wave data extraction failed for MOP {mop_id} at {scan_date}: {e}. "
                "Returning zeros."
            )
            # Return zeros as fallback
            T_w = int((self.lookback_days * 24) / self.resample_hours)
            features = np.zeros((T_w, self.n_features), dtype=np.float32)

            start_date = scan_date - timedelta(days=self.lookback_days - 1)
            dates = pd.date_range(start_date, scan_date, freq=f'{self.resample_hours}h')
            doy = dates.dayofyear.values[:T_w].astype(np.int32)

            return features, doy

        return features, doy

    def get_wave_for_mop(
        self,
        mop_id: int,
        scan_date: Union[datetime, pd.Timestamp, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get wave features for a MOP transect.

        Alias for get_wave_for_scan for consistency with AtmosphericLoader API.

        Args:
            mop_id: MOP transect ID
            scan_date: Scan date

        Returns:
            Tuple of (features, day_of_year) arrays
        """
        return self.get_wave_for_scan(mop_id, scan_date)

    def get_batch_wave(
        self,
        mop_ids: List[int],
        scan_dates: List[Union[datetime, pd.Timestamp, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get wave features for a batch of samples.

        Args:
            mop_ids: List of MOP IDs (one per sample)
            scan_dates: List of scan dates (one per sample)

        Returns:
            Tuple of:
                features: (batch_size, T_w, 4) array
                day_of_year: (batch_size, T_w) array
        """
        if len(mop_ids) != len(scan_dates):
            raise ValueError(
                f"Length mismatch: {len(mop_ids)} MOPs, {len(scan_dates)} dates"
            )

        batch_size = len(mop_ids)
        T_w = int((self.lookback_days * 24) / self.resample_hours)

        features = np.zeros((batch_size, T_w, self.n_features), dtype=np.float32)
        day_of_year = np.zeros((batch_size, T_w), dtype=np.int32)

        for i, (mop_id, date) in enumerate(zip(mop_ids, scan_dates)):
            features[i], day_of_year[i] = self.get_wave_for_scan(mop_id, date)

        return features, day_of_year

    def get_date_range(self, mop_id: int) -> Tuple[datetime, datetime]:
        """Get available date range for a MOP.

        Args:
            mop_id: MOP transect ID

        Returns:
            Tuple of (start_date, end_date)
        """
        wave_data = self._load_wave_data(mop_id)
        return wave_data.time[0].astype('datetime64[s]').item(), wave_data.time[-1].astype('datetime64[s]').item()

    def validate_coverage(
        self,
        mop_ids: List[int],
        scan_dates: List[Union[datetime, pd.Timestamp, int]],
    ) -> Dict[str, any]:
        """Validate data coverage for a list of samples.

        Args:
            mop_ids: List of MOP IDs to check
            scan_dates: List of scan dates to check

        Returns:
            Dictionary with coverage statistics
        """
        if len(mop_ids) != len(scan_dates):
            raise ValueError(
                f"Length mismatch: {len(mop_ids)} MOPs, {len(scan_dates)} dates"
            )

        covered = 0
        partial = 0
        missing = 0
        total = len(mop_ids)

        for mop_id, scan_date in zip(mop_ids, scan_dates):
            # Convert to datetime
            if isinstance(scan_date, int):
                scan_date = datetime.fromordinal(scan_date)
            elif isinstance(scan_date, pd.Timestamp):
                scan_date = scan_date.to_pydatetime()

            # Check if MOP has data
            if mop_id not in self._available_mops:
                missing += 1
                continue

            try:
                # Get date range for this MOP
                data_start, data_end = self.get_date_range(mop_id)
                start_date = scan_date - timedelta(days=self.lookback_days - 1)

                # Check if window is fully covered
                if start_date >= data_start and scan_date <= data_end:
                    covered += 1
                elif start_date < data_end and scan_date > data_start:
                    # Partial overlap
                    partial += 1
                else:
                    missing += 1

            except Exception as e:
                logger.debug(f"Coverage check failed for MOP {mop_id}: {e}")
                missing += 1

        return {
            'total': total,
            'covered': covered,
            'partial': partial,
            'missing': missing,
            'coverage_pct': 100 * covered / total if total > 0 else 0,
        }

    @property
    def available_mops(self) -> List[int]:
        """List of MOPs with available wave data."""
        return self._available_mops

    def summary(self) -> Dict[int, dict]:
        """Get summary of all available MOP data.

        Returns:
            Dictionary with summary per MOP
        """
        result = {}

        for mop_id in self._available_mops:
            try:
                wave_data = self._load_wave_data(mop_id)

                result[mop_id] = {
                    'n_records': len(wave_data.time),
                    'date_range': (
                        wave_data.time[0].astype('datetime64[s]').item(),
                        wave_data.time[-1].astype('datetime64[s]').item(),
                    ),
                    'latitude': wave_data.latitude,
                    'longitude': wave_data.longitude,
                    'water_depth': wave_data.water_depth,
                    'hs_mean': float(np.nanmean(wave_data.hs)),
                    'hs_max': float(np.nanmax(wave_data.hs)),
                    'data_quality': float(np.mean(~np.isnan(wave_data.hs))),
                }
            except Exception as e:
                logger.warning(f"Failed to load summary for MOP {mop_id}: {e}")
                result[mop_id] = {'error': str(e)}

        return result


class WaveDataset:
    """Integration helper for PyTorch Dataset.

    Provides methods to integrate wave loading with the main
    CliffCast dataset class.

    Mirrors the AtmosphericDataset API exactly.
    """

    def __init__(
        self,
        wave_loader: WaveLoader,
        transect_ids: np.ndarray,
        timestamps: np.ndarray,
    ):
        """Initialize dataset integration.

        Args:
            wave_loader: WaveLoader instance
            transect_ids: Array of MOP IDs from cube (n_transects,)
            timestamps: Array of ordinal timestamps from cube (n_transects, n_epochs)
        """
        self.loader = wave_loader
        self.transect_ids = transect_ids
        self.timestamps = timestamps

        # MOP IDs are the same as transect IDs in this project
        self.mops = transect_ids

    def get_wave_for_sample(
        self,
        transect_idx: int,
        epoch_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get wave features for a single sample.

        Args:
            transect_idx: Index into transect_ids
            epoch_idx: Index into timestamps

        Returns:
            Tuple of (features, day_of_year) arrays
        """
        mop_id = int(self.mops[transect_idx])
        scan_date = int(self.timestamps[transect_idx, epoch_idx])

        return self.loader.get_wave_for_scan(mop_id, scan_date)

    def get_wave_for_indices(
        self,
        indices: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get wave features for a batch of (transect_idx, epoch_idx) pairs.

        Args:
            indices: List of (transect_idx, epoch_idx) tuples

        Returns:
            Tuple of:
                features: (batch_size, T_w, 4) array
                day_of_year: (batch_size, T_w) array
        """
        mop_ids = [int(self.mops[ti]) for ti, ei in indices]
        dates = [int(self.timestamps[ti, ei]) for ti, ei in indices]

        return self.loader.get_batch_wave(mop_ids, dates)
