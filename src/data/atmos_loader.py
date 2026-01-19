"""Atmospheric data loading for CliffCast model training.

This module provides the AtmosphericLoader class which loads pre-computed
atmospheric features aligned to LiDAR scan dates. Features are loaded from
per-beach parquet files and returned as numpy arrays ready for the model.

Usage:
    from src.data.atmos_loader import AtmosphericLoader

    loader = AtmosphericLoader('data/processed/atmospheric/')

    # Get features for a single scan
    features, doy = loader.get_atmos_for_scan('delmar', scan_date)

    # Get features for a batch
    features, doy = loader.get_batch_atmos(beaches, scan_dates)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .atmos_features import ATMOS_FEATURE_NAMES

logger = logging.getLogger(__name__)


# MOP to beach mapping (from extract_transects.py)
BEACH_MOP_RANGES = {
    'blacks': (520, 567),
    'torrey': (567, 581),
    'delmar': (595, 620),
    'solana': (637, 666),
    'sanelijo': (683, 708),
    'encinitas': (708, 764),
}


def get_beach_for_mop(mop_id: int) -> str:
    """Map MOP ID to beach name.

    Args:
        mop_id: MOP transect ID

    Returns:
        Beach name

    Raises:
        ValueError: If MOP ID not in any known beach range
    """
    for beach, (mop_min, mop_max) in BEACH_MOP_RANGES.items():
        if mop_min <= mop_id <= mop_max:
            return beach
    raise ValueError(f"MOP {mop_id} not in any known beach range")


class AtmosphericLoader:
    """Load atmospheric data aligned to scan dates.

    This class manages loading pre-computed atmospheric features from parquet
    files and aligning them to LiDAR scan dates for model training.

    Attributes:
        atmos_dir: Directory containing per-beach parquet files
        lookback_days: Number of days to look back from scan date
        feature_names: List of feature names to load
    """

    def __init__(
        self,
        atmos_dir: Union[str, Path],
        lookback_days: int = 90,
        feature_names: Optional[List[str]] = None,
        cache_size: int = 10,
    ):
        """Initialize atmospheric loader.

        Args:
            atmos_dir: Directory containing {beach}_atmos.parquet files
            lookback_days: Days to look back from scan date (default: 90)
            feature_names: Features to load (default: all 25)
            cache_size: Number of beach DataFrames to keep in memory
        """
        self.atmos_dir = Path(atmos_dir)
        self.lookback_days = lookback_days
        self.feature_names = feature_names or ATMOS_FEATURE_NAMES
        self.n_features = len(self.feature_names)

        # Validate directory exists
        if not self.atmos_dir.exists():
            raise FileNotFoundError(f"Atmospheric data directory not found: {self.atmos_dir}")

        # Cache loaded DataFrames
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_size = cache_size

        # Track available beaches
        self._available_beaches = self._scan_available_beaches()

        if not self._available_beaches:
            logger.warning(f"No atmospheric data files found in {self.atmos_dir}")

    def _scan_available_beaches(self) -> List[str]:
        """Scan directory for available beach data files."""
        available = []
        for beach in BEACH_MOP_RANGES.keys():
            parquet_path = self.atmos_dir / f"{beach}_atmos.parquet"
            if parquet_path.exists():
                available.append(beach)
        return available

    def _load_beach_data(self, beach: str) -> pd.DataFrame:
        """Load and cache beach data.

        Args:
            beach: Beach name

        Returns:
            DataFrame with atmospheric features

        Raises:
            FileNotFoundError: If beach data file not found
        """
        if beach in self._cache:
            return self._cache[beach]

        parquet_path = self.atmos_dir / f"{beach}_atmos.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"No atmospheric data for {beach}: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        # Manage cache size
        if len(self._cache) >= self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[beach] = df

        return df

    def get_atmos_for_scan(
        self,
        beach: str,
        scan_date: Union[datetime, pd.Timestamp, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get atmospheric features for a single scan.

        Extracts the atmospheric time series for [scan_date - lookback_days, scan_date].

        Args:
            beach: Beach name
            scan_date: Scan date (datetime, Timestamp, or ordinal day)

        Returns:
            Tuple of:
                features: (lookback_days, n_features) array
                day_of_year: (lookback_days,) array for seasonality encoding
        """
        # Convert ordinal to datetime if needed
        if isinstance(scan_date, int):
            scan_date = datetime.fromordinal(scan_date)
        elif isinstance(scan_date, pd.Timestamp):
            scan_date = scan_date.to_pydatetime()

        # Load beach data
        df = self._load_beach_data(beach)

        # Calculate date range
        end_date = scan_date
        start_date = scan_date - timedelta(days=self.lookback_days - 1)

        # Extract window
        mask = (df.index >= start_date) & (df.index <= end_date)
        window = df.loc[mask]

        # Handle missing data
        if len(window) < self.lookback_days:
            # Create full date range and reindex with interpolation
            full_range = pd.date_range(start_date, end_date, freq='D')
            window = window.reindex(full_range)

            # Interpolate missing values (linear for gaps <= 3 days)
            window = window.interpolate(method='linear', limit=3)

            # Fill remaining with column means (climatological fill)
            window = window.fillna(window.mean())

            # If still NaN (completely missing), fill with zeros
            window = window.fillna(0)

        # Select features
        available_features = [f for f in self.feature_names if f in window.columns]
        features = window[available_features].values[-self.lookback_days:]

        # Pad if we have fewer features than expected
        if len(available_features) < self.n_features:
            padded = np.zeros((self.lookback_days, self.n_features))
            padded[:, :len(available_features)] = features
            features = padded

        # Compute day of year
        dates = pd.date_range(start_date, end_date, freq='D')
        day_of_year = dates.dayofyear.values[-self.lookback_days:]

        return features.astype(np.float32), day_of_year.astype(np.int32)

    def get_atmos_for_mop(
        self,
        mop_id: int,
        scan_date: Union[datetime, pd.Timestamp, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get atmospheric features for a MOP transect.

        Convenience method that maps MOP ID to beach and calls get_atmos_for_scan.

        Args:
            mop_id: MOP transect ID
            scan_date: Scan date

        Returns:
            Tuple of (features, day_of_year) arrays
        """
        beach = get_beach_for_mop(mop_id)
        return self.get_atmos_for_scan(beach, scan_date)

    def get_batch_atmos(
        self,
        beaches: List[str],
        scan_dates: List[Union[datetime, pd.Timestamp, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get atmospheric features for a batch of samples.

        Args:
            beaches: List of beach names (one per sample)
            scan_dates: List of scan dates (one per sample)

        Returns:
            Tuple of:
                features: (batch_size, lookback_days, n_features) array
                day_of_year: (batch_size, lookback_days) array
        """
        if len(beaches) != len(scan_dates):
            raise ValueError(
                f"Length mismatch: {len(beaches)} beaches, {len(scan_dates)} dates"
            )

        batch_size = len(beaches)
        features = np.zeros((batch_size, self.lookback_days, self.n_features), dtype=np.float32)
        day_of_year = np.zeros((batch_size, self.lookback_days), dtype=np.int32)

        for i, (beach, date) in enumerate(zip(beaches, scan_dates)):
            features[i], day_of_year[i] = self.get_atmos_for_scan(beach, date)

        return features, day_of_year

    def get_batch_atmos_for_mops(
        self,
        mop_ids: List[int],
        scan_dates: List[Union[datetime, pd.Timestamp, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get atmospheric features for a batch of MOP transects.

        Args:
            mop_ids: List of MOP IDs
            scan_dates: List of scan dates

        Returns:
            Tuple of (features, day_of_year) arrays
        """
        beaches = [get_beach_for_mop(mop) for mop in mop_ids]
        return self.get_batch_atmos(beaches, scan_dates)

    def get_date_range(self, beach: str) -> Tuple[datetime, datetime]:
        """Get available date range for a beach.

        Args:
            beach: Beach name

        Returns:
            Tuple of (start_date, end_date)
        """
        df = self._load_beach_data(beach)
        return df.index.min().to_pydatetime(), df.index.max().to_pydatetime()

    def validate_coverage(
        self,
        beach: str,
        scan_dates: List[Union[datetime, pd.Timestamp]],
    ) -> Dict[str, any]:
        """Validate data coverage for a list of scan dates.

        Args:
            beach: Beach name
            scan_dates: List of scan dates to check

        Returns:
            Dictionary with coverage statistics
        """
        df = self._load_beach_data(beach)
        data_start = df.index.min()
        data_end = df.index.max()

        covered = 0
        partial = 0
        missing = 0

        for scan_date in scan_dates:
            if isinstance(scan_date, int):
                scan_date = datetime.fromordinal(scan_date)

            start_date = scan_date - timedelta(days=self.lookback_days - 1)

            if start_date >= data_start and scan_date <= data_end:
                # Check actual coverage in window
                mask = (df.index >= start_date) & (df.index <= scan_date)
                n_present = mask.sum()

                if n_present >= self.lookback_days:
                    covered += 1
                elif n_present > 0:
                    partial += 1
                else:
                    missing += 1
            else:
                missing += 1

        total = len(scan_dates)

        return {
            'total': total,
            'covered': covered,
            'partial': partial,
            'missing': missing,
            'coverage_pct': 100 * covered / total if total > 0 else 0,
            'data_range': (data_start, data_end),
        }

    @property
    def available_beaches(self) -> List[str]:
        """List of beaches with available atmospheric data."""
        return self._available_beaches

    def summary(self) -> Dict[str, dict]:
        """Get summary of all available beach data.

        Returns:
            Dictionary with summary per beach
        """
        result = {}

        for beach in self._available_beaches:
            df = self._load_beach_data(beach)

            result[beach] = {
                'n_records': len(df),
                'date_range': (df.index.min(), df.index.max()),
                'n_features': len([c for c in df.columns if c in self.feature_names]),
                'nan_pct': 100 * df[self.feature_names].isna().mean().mean(),
            }

        return result


class AtmosphericDataset:
    """Integration helper for PyTorch Dataset.

    Provides methods to integrate atmospheric loading with the main
    CliffCast dataset class.
    """

    def __init__(
        self,
        atmos_loader: AtmosphericLoader,
        transect_ids: np.ndarray,
        timestamps: np.ndarray,
    ):
        """Initialize dataset integration.

        Args:
            atmos_loader: AtmosphericLoader instance
            transect_ids: Array of MOP IDs from cube
            timestamps: Array of ordinal timestamps from cube (n_transects, n_epochs)
        """
        self.loader = atmos_loader
        self.transect_ids = transect_ids
        self.timestamps = timestamps

        # Pre-compute beach mapping
        self.beaches = [get_beach_for_mop(int(tid)) for tid in transect_ids]

    def get_atmos_for_sample(
        self,
        transect_idx: int,
        epoch_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get atmospheric features for a single sample.

        Args:
            transect_idx: Index into transect_ids
            epoch_idx: Index into timestamps

        Returns:
            Tuple of (features, day_of_year) arrays
        """
        beach = self.beaches[transect_idx]
        scan_date = int(self.timestamps[transect_idx, epoch_idx])

        return self.loader.get_atmos_for_scan(beach, scan_date)

    def get_atmos_for_indices(
        self,
        indices: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get atmospheric features for a batch of (transect_idx, epoch_idx) pairs.

        Args:
            indices: List of (transect_idx, epoch_idx) tuples

        Returns:
            Tuple of:
                features: (batch_size, lookback_days, n_features) array
                day_of_year: (batch_size, lookback_days) array
        """
        beaches = [self.beaches[ti] for ti, ei in indices]
        dates = [int(self.timestamps[ti, ei]) for ti, ei in indices]

        return self.loader.get_batch_atmos(beaches, dates)
