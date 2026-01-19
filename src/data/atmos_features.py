"""Atmospheric feature computation for CliffCast.

This module computes derived atmospheric features from raw PRISM climate data.
Features are designed to capture processes relevant to coastal cliff erosion:
- Precipitation patterns and antecedent moisture
- Wetting/drying cycles (fatigue weathering)
- Evaporative demand (VPD)
- Freeze-thaw cycles (mechanical weathering)

Usage:
    from src.data.atmos_features import AtmosFeatureComputer

    computer = AtmosFeatureComputer()
    features_df = computer.compute_all_features(raw_df)
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Feature names for reference
ATMOS_FEATURE_NAMES = [
    # Raw measurements (5)
    'precip_mm',
    'temp_mean_c',
    'temp_min_c',
    'temp_max_c',
    'dewpoint_c',
    # Cumulative precipitation (4)
    'precip_7d',
    'precip_30d',
    'precip_60d',
    'precip_90d',
    # Antecedent conditions (3)
    'api',
    'days_since_rain',
    'consecutive_dry_days',
    # Intensity (4)
    'rain_day_flag',
    'intensity_class',
    'max_precip_7d',
    'max_precip_30d',
    # Wetting/drying cycles (2)
    'wet_dry_cycles_30d',
    'wet_dry_cycles_90d',
    # Evaporative demand (2)
    'vpd',
    'vpd_7d_mean',
    # Freeze-thaw (4)
    'freeze_flag',
    'marginal_freeze_flag',
    'freeze_thaw_cycles_30d',
    'freeze_thaw_cycles_season',
]

# Column name mapping from PRISM variables
PRISM_TO_FEATURE = {
    'ppt': 'precip_mm',
    'tmean': 'temp_mean_c',
    'tmin': 'temp_min_c',
    'tmax': 'temp_max_c',
    'tdmean': 'dewpoint_c',
}


class AtmosFeatureComputer:
    """Compute derived atmospheric features from raw PRISM data.

    This class transforms raw daily climate data into a comprehensive set of
    25 features relevant to coastal cliff erosion processes.

    Attributes:
        rain_threshold_mm: Precipitation threshold for rain day (default: 1.0 mm)
        api_decay: Decay factor for antecedent precipitation index (default: 0.9)
        water_year_start_month: Month when water year begins (default: 10 = October)
    """

    # Intensity class thresholds (mm/day)
    INTENSITY_THRESHOLDS = {
        'none': 0.0,      # 0
        'light': 1.0,     # 1
        'moderate': 10.0, # 2
        'heavy': 25.0,    # 3
    }

    def __init__(
        self,
        rain_threshold_mm: float = 1.0,
        api_decay: float = 0.9,
        water_year_start_month: int = 10,
    ):
        """Initialize feature computer.

        Args:
            rain_threshold_mm: Threshold for defining a rain day
            api_decay: Decay factor k for antecedent precipitation index
            water_year_start_month: Month when water year starts (1-12)
        """
        self.rain_threshold_mm = rain_threshold_mm
        self.api_decay = api_decay
        self.water_year_start_month = water_year_start_month

    def compute_all_features(
        self,
        raw_df: pd.DataFrame,
        date_col: str = 'date',
    ) -> pd.DataFrame:
        """Compute all 25 atmospheric features from raw daily values.

        Args:
            raw_df: DataFrame with columns:
                - date: datetime
                - ppt or precip_mm: daily precipitation (mm)
                - tmin or temp_min_c: minimum temperature (°C)
                - tmax or temp_max_c: maximum temperature (°C)
                - tmean or temp_mean_c: mean temperature (°C)
                - tdmean or dewpoint_c: dewpoint temperature (°C)
            date_col: Name of date column

        Returns:
            DataFrame with all 25 feature columns plus date
        """
        # Standardize column names
        df = self._standardize_columns(raw_df.copy(), date_col)

        # Ensure sorted by date
        df = df.sort_values('date').reset_index(drop=True)

        # Compute feature groups
        df = self._compute_cumulative_precip(df)
        df = self._compute_antecedent_conditions(df)
        df = self._compute_intensity_features(df)
        df = self._compute_wet_dry_cycles(df)
        df = self._compute_evaporative_demand(df)
        df = self._compute_freeze_thaw(df)

        # Select and order final columns
        output_cols = ['date'] + ATMOS_FEATURE_NAMES
        available_cols = [c for c in output_cols if c in df.columns]

        return df[available_cols]

    def _standardize_columns(
        self,
        df: pd.DataFrame,
        date_col: str,
    ) -> pd.DataFrame:
        """Standardize column names from PRISM to feature names."""
        # Rename date column
        if date_col != 'date':
            df = df.rename(columns={date_col: 'date'})

        # Rename PRISM variables to feature names
        df = df.rename(columns=PRISM_TO_FEATURE)

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        return df

    def _compute_cumulative_precip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling cumulative precipitation features."""
        precip = df['precip_mm'].fillna(0)

        # Rolling sums for different windows
        df['precip_7d'] = precip.rolling(window=7, min_periods=1).sum()
        df['precip_30d'] = precip.rolling(window=30, min_periods=1).sum()
        df['precip_60d'] = precip.rolling(window=60, min_periods=1).sum()
        df['precip_90d'] = precip.rolling(window=90, min_periods=1).sum()

        return df

    def _compute_antecedent_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute antecedent moisture and drying duration features."""
        precip = df['precip_mm'].fillna(0).values

        # Antecedent Precipitation Index (API)
        df['api'] = self.compute_api(precip)

        # Days since last rain
        df['days_since_rain'] = self._days_since_rain(precip)

        # Consecutive dry days before current (for event analysis)
        df['consecutive_dry_days'] = self._consecutive_dry_days(precip)

        return df

    def compute_api(
        self,
        precip: np.ndarray,
        k: Optional[float] = None,
    ) -> np.ndarray:
        """Compute Antecedent Precipitation Index.

        API is an exponentially weighted sum of past precipitation that
        approximates soil moisture memory:
            API_t = precip_t + k * API_{t-1}

        Args:
            precip: Daily precipitation array
            k: Decay factor (default: self.api_decay)

        Returns:
            Array of API values
        """
        k = k or self.api_decay
        n = len(precip)
        api = np.zeros(n)

        for i in range(n):
            if i == 0:
                api[i] = precip[i]
            else:
                api[i] = precip[i] + k * api[i - 1]

        return api

    def _days_since_rain(self, precip: np.ndarray) -> np.ndarray:
        """Compute days since last significant rain."""
        n = len(precip)
        days_since = np.zeros(n)

        count = 0
        for i in range(n):
            if precip[i] >= self.rain_threshold_mm:
                count = 0
            else:
                count += 1
            days_since[i] = count

        return days_since

    def _consecutive_dry_days(self, precip: np.ndarray) -> np.ndarray:
        """Compute consecutive dry days before each day.

        This counts how many dry days preceded the current day,
        useful for understanding pre-event dryness.
        """
        n = len(precip)
        consec = np.zeros(n)

        for i in range(1, n):
            if precip[i - 1] < self.rain_threshold_mm:
                consec[i] = consec[i - 1] + 1
            else:
                consec[i] = 0

        return consec

    def _compute_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute precipitation intensity features."""
        precip = df['precip_mm'].fillna(0)

        # Binary rain day flag
        df['rain_day_flag'] = (precip >= self.rain_threshold_mm).astype(float)

        # Intensity classification
        df['intensity_class'] = precip.apply(self._classify_intensity)

        # Maximum precipitation in rolling windows
        df['max_precip_7d'] = precip.rolling(window=7, min_periods=1).max()
        df['max_precip_30d'] = precip.rolling(window=30, min_periods=1).max()

        return df

    def _classify_intensity(self, precip_mm: float) -> int:
        """Classify precipitation intensity.

        Returns:
            0: none (< 1mm)
            1: light (1-10mm)
            2: moderate (10-25mm)
            3: heavy (>= 25mm)
        """
        if precip_mm < self.INTENSITY_THRESHOLDS['light']:
            return 0
        elif precip_mm < self.INTENSITY_THRESHOLDS['moderate']:
            return 1
        elif precip_mm < self.INTENSITY_THRESHOLDS['heavy']:
            return 2
        else:
            return 3

    def _compute_wet_dry_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute wetting/drying cycle counts.

        These cycles represent fatigue weathering events where repeated
        wetting and drying causes material degradation.
        """
        precip = df['precip_mm'].fillna(0).values

        # Create wet/dry state array (1 = wet, 0 = dry)
        wet_state = (precip >= self.rain_threshold_mm).astype(int)

        # Count transitions in rolling windows
        df['wet_dry_cycles_30d'] = self._count_transitions(wet_state, window=30)
        df['wet_dry_cycles_90d'] = self._count_transitions(wet_state, window=90)

        return df

    def _count_transitions(
        self,
        state: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Count wet→dry transitions in a rolling window.

        A transition is counted when state goes from 1 (wet) to 0 (dry).
        """
        n = len(state)
        transitions = np.zeros(n)

        # Compute state changes (wet to dry = 1→0 = -1)
        changes = np.diff(state)
        wet_to_dry = (changes == -1).astype(int)

        # Pad to match original length
        wet_to_dry = np.concatenate([[0], wet_to_dry])

        # Rolling sum of transitions
        for i in range(n):
            start = max(0, i - window + 1)
            transitions[i] = wet_to_dry[start:i + 1].sum()

        return transitions

    def _compute_evaporative_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute vapor pressure deficit (VPD) features.

        VPD represents the drying potential of the atmosphere.
        Higher VPD means more evaporative stress on materials.
        """
        # Compute daily VPD
        if 'temp_mean_c' in df.columns and 'dewpoint_c' in df.columns:
            df['vpd'] = df.apply(
                lambda row: self.compute_vpd(
                    row['temp_mean_c'],
                    row['dewpoint_c']
                ),
                axis=1
            )
        else:
            df['vpd'] = np.nan

        # 7-day mean VPD
        df['vpd_7d_mean'] = df['vpd'].rolling(window=7, min_periods=1).mean()

        return df

    def compute_vpd(
        self,
        temp_c: float,
        dewpoint_c: float,
    ) -> float:
        """Compute Vapor Pressure Deficit.

        VPD = e_s(T) - e_a(T_d)

        Where:
            e_s(T) = saturation vapor pressure at temperature T
            e_a(T_d) = actual vapor pressure at dewpoint T_d

        Both computed using Tetens equation.

        Args:
            temp_c: Air temperature (°C)
            dewpoint_c: Dewpoint temperature (°C)

        Returns:
            VPD in kPa
        """
        if pd.isna(temp_c) or pd.isna(dewpoint_c):
            return np.nan

        # Tetens equation for saturation vapor pressure (kPa)
        e_sat = self._saturation_vapor_pressure(temp_c)
        e_actual = self._saturation_vapor_pressure(dewpoint_c)

        vpd = e_sat - e_actual

        # VPD should be non-negative
        return max(0.0, vpd)

    def _saturation_vapor_pressure(self, temp_c: float) -> float:
        """Compute saturation vapor pressure using Tetens equation.

        Args:
            temp_c: Temperature in Celsius

        Returns:
            Saturation vapor pressure in kPa
        """
        # Tetens equation
        # e_s = 0.6108 * exp(17.27 * T / (T + 237.3))
        return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))

    def _compute_freeze_thaw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute freeze-thaw cycle features.

        Freeze-thaw cycles cause mechanical weathering through ice expansion
        in pore spaces. While limited in San Diego, these features support
        model transfer to colder climates.
        """
        if 'temp_min_c' not in df.columns:
            df['freeze_flag'] = 0.0
            df['marginal_freeze_flag'] = 0.0
            df['freeze_thaw_cycles_30d'] = 0.0
            df['freeze_thaw_cycles_season'] = 0.0
            return df

        temp_min = df['temp_min_c'].fillna(999)  # Default to no freeze if missing

        # Freeze flags
        df['freeze_flag'] = (temp_min < 0.0).astype(float)
        df['marginal_freeze_flag'] = (temp_min < 2.0).astype(float)

        # Freeze-thaw cycles (freeze followed by thaw)
        df['freeze_thaw_cycles_30d'] = self._count_freeze_thaw_cycles(
            temp_min.values, window=30
        )

        # Seasonal cumulative (water year: Oct-Sep)
        df['freeze_thaw_cycles_season'] = self._seasonal_freeze_thaw(
            df['date'].values,
            temp_min.values,
        )

        return df

    def _count_freeze_thaw_cycles(
        self,
        temp_min: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Count freeze-thaw cycles in rolling window.

        A cycle is counted when temp_min goes from < 0 to >= 0.
        """
        n = len(temp_min)
        cycles = np.zeros(n)

        # Create freeze state (1 = frozen, 0 = not frozen)
        frozen = (temp_min < 0).astype(int)

        # Detect thaw events (frozen yesterday, not frozen today)
        thaw_events = np.zeros(n)
        for i in range(1, n):
            if frozen[i - 1] == 1 and frozen[i] == 0:
                thaw_events[i] = 1

        # Rolling sum
        for i in range(n):
            start = max(0, i - window + 1)
            cycles[i] = thaw_events[start:i + 1].sum()

        return cycles

    def _seasonal_freeze_thaw(
        self,
        dates: np.ndarray,
        temp_min: np.ndarray,
    ) -> np.ndarray:
        """Compute cumulative freeze-thaw cycles for water year.

        Water year runs Oct 1 to Sep 30. Counts reset at start of each water year.
        """
        n = len(dates)
        seasonal = np.zeros(n)

        # Convert dates to pandas for easier manipulation
        dates = pd.to_datetime(dates)

        # Create freeze state
        frozen = (temp_min < 0).astype(int)

        # Track cycles per water year
        current_year = None
        cumulative = 0

        for i in range(n):
            # Determine water year
            year = dates[i].year
            month = dates[i].month
            water_year = year if month >= self.water_year_start_month else year - 1

            # Reset at start of new water year
            if water_year != current_year:
                current_year = water_year
                cumulative = 0

            # Check for thaw event
            if i > 0 and frozen[i - 1] == 1 and frozen[i] == 0:
                cumulative += 1

            seasonal[i] = cumulative

        return seasonal


def compute_features_for_beach(
    raw_csv: str,
    output_parquet: str,
    date_col: str = 'date',
) -> pd.DataFrame:
    """Convenience function to compute features for a single beach.

    Args:
        raw_csv: Path to raw PRISM CSV for beach
        output_parquet: Path to save processed features
        date_col: Name of date column

    Returns:
        DataFrame with computed features
    """
    # Load raw data
    raw_df = pd.read_csv(raw_csv, parse_dates=[date_col])

    # Compute features
    computer = AtmosFeatureComputer()
    features_df = computer.compute_all_features(raw_df, date_col)

    # Save to parquet
    features_df.to_parquet(output_parquet, index=False)

    return features_df
