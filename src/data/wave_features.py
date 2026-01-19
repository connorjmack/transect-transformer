#!/usr/bin/env python3
"""
Wave Metrics Calculator for CliffCast

Computes derived wave features from CDIP MOP NetCDF files for cliff erosion prediction.
Generates 16 physically-motivated features from raw wave data.

Features computed:
  Raw (4):
    - hs_m: Significant wave height
    - tp_s: Peak period  
    - dp_deg: Peak direction
    - power_kw: Wave power flux

  Integrated (3):
    - cumulative_energy_mj: Total wave energy over window
    - cumulative_power_kwh: Total wave power over window
    - mean_power_kw: Average power over window

  Extreme (3):
    - max_hs_m: Maximum wave height
    - hs_p90: 90th percentile wave height
    - hs_p99: 99th percentile wave height

  Storm/Event (5):
    - storm_hours: Hours with Hs > threshold
    - storm_count: Number of discrete storm events
    - max_storm_duration_hr: Longest storm duration
    - time_since_storm_hr: Hours since last storm at end of window
    - mean_storm_duration_hr: Average storm duration

  Temporal (2):
    - rolling_max_7d_m: Maximum Hs in any 7-day window
    - hs_trend_slope: Linear trend in Hs over window (m/day)

  Physical (computed per-timestep if cliff orientation provided):
    - shore_normal_hs_m: Shore-normal wave height component
    - runup_2pct_m: Estimated 2% exceedance runup (Stockdon 2006)

Usage:
    # Compute metrics for a single MOP and date range
    python wave_metrics_calculator.py \
        --input data/raw/cdip/D0582_hindcast.nc \
        --output data/processed/wave_metrics/D0582_metrics.parquet \
        --start-date 2023-09-15 \
        --end-date 2023-12-15

    # Batch process all MOPs for training dataset
    python wave_metrics_calculator.py \
        --input-dir data/raw/cdip/ \
        --output-dir data/processed/wave_metrics/ \
        --scan-dates data/processed/scan_dates.csv

    # Compute time series features for model input
    python wave_metrics_calculator.py \
        --input data/raw/cdip/D0582_hindcast.nc \
        --scan-date 2023-12-15 \
        --lookback-days 90 \
        --output-format tensor

Author: CliffCast Team
"""

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import netCDF4
except ImportError:
    netCDF4 = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Physical constants
RHO_WATER = 1025.0  # kg/m³ seawater density
G = 9.81  # m/s² gravitational acceleration


@dataclass
class WaveMetricsConfig:
    """Configuration for wave metrics computation."""
    
    # Lookback window
    lookback_days: int = 90
    
    # Resampling
    resample_hours: int = 6  # Resample to 6-hourly for model input
    
    # Storm detection thresholds
    storm_hs_threshold_m: float = 2.0  # Hs threshold for "storm" conditions
    storm_gap_hours: int = 12  # Minimum gap between distinct storm events
    
    # Rolling window sizes (in days)
    rolling_windows_days: List[int] = field(default_factory=lambda: [7, 30])
    
    # Runup calculation parameters (Stockdon 2006)
    beach_slope: float = 0.1  # Default beach slope (tan β)
    
    # Fill value handling
    fill_value: float = -999.99
    
    # Output feature names (in order)
    feature_names: List[str] = field(default_factory=lambda: [
        'hs_m', 'tp_s', 'dp_deg', 'power_kw',
        'shore_normal_hs_m', 'runup_2pct_m',
        'cumulative_energy_mj', 'cumulative_power_kwh', 'mean_power_kw',
        'max_hs_m', 'hs_p90', 'hs_p99',
        'storm_hours', 'storm_count', 'max_storm_duration_hr',
        'time_since_storm_hr', 'mean_storm_duration_hr',
        'rolling_max_7d_m', 'hs_trend_slope'
    ])


class WaveMetricsCalculator:
    """
    Compute derived wave metrics from CDIP MOP data.
    
    This class loads CDIP NetCDF files and computes physically-motivated
    features for cliff erosion prediction.
    """
    
    def __init__(self, config: Optional[WaveMetricsConfig] = None):
        """
        Initialize the calculator.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or WaveMetricsConfig()
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Check that required packages are available."""
        if xr is None and netCDF4 is None:
            raise ImportError(
                "Either xarray or netCDF4 is required. "
                "Install with: pip install xarray netCDF4"
            )
    
    def load_cdip_data(
        self,
        filepath: Union[str, Path],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load CDIP MOP data from NetCDF file.
        
        Args:
            filepath: Path to CDIP NetCDF file (e.g., D0582_hindcast.nc)
            start_date: Start of time window (inclusive)
            end_date: End of time window (inclusive)
            
        Returns:
            DataFrame with columns: time, hs, tp, dp, ta
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CDIP file not found: {filepath}")
        
        logger.info(f"Loading CDIP data from {filepath}")
        
        # Try xarray first, fall back to netCDF4
        if xr is not None:
            ds = xr.open_dataset(filepath)
            df = self._xarray_to_dataframe(ds, start_date, end_date)
            ds.close()
        else:
            df = self._netcdf4_to_dataframe(filepath, start_date, end_date)
        
        # Handle fill values
        for col in ['hs', 'tp', 'dp', 'ta']:
            if col in df.columns:
                df[col] = df[col].replace(self.config.fill_value, np.nan)
        
        # Basic validation
        if df.empty:
            logger.warning(f"No data found in {filepath} for specified date range")
            return df
        
        logger.info(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    
    def _xarray_to_dataframe(
        self,
        ds: 'xr.Dataset',
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Convert xarray Dataset to pandas DataFrame."""
        # CDIP files typically have 'waveTime' as the time dimension
        time_var = 'waveTime' if 'waveTime' in ds.dims else 'time'
        
        # Extract variables - CDIP uses waveHs, waveTp, waveDp, waveTa
        var_mapping = {
            'waveHs': 'hs',
            'waveTp': 'tp', 
            'waveDp': 'dp',
            'waveTa': 'ta',
            'Hs': 'hs',
            'Tp': 'tp',
            'Dp': 'dp',
            'Ta': 'ta',
            'hs': 'hs',
            'tp': 'tp',
            'dp': 'dp',
            'ta': 'ta'
        }
        
        data = {}
        for nc_var, df_var in var_mapping.items():
            if nc_var in ds.data_vars and df_var not in data:
                data[df_var] = ds[nc_var].values
        
        # Get time values
        if time_var in ds.coords:
            time_values = pd.to_datetime(ds[time_var].values)
        elif time_var in ds.data_vars:
            time_values = pd.to_datetime(ds[time_var].values)
        else:
            raise ValueError(f"Could not find time variable in dataset. Available: {list(ds.dims)}")
        
        df = pd.DataFrame(data, index=time_values)
        df.index.name = 'time'
        
        # Filter by date range
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        return df.sort_index()
    
    def _netcdf4_to_dataframe(
        self,
        filepath: Path,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Load data using netCDF4 directly."""
        nc = netCDF4.Dataset(filepath, 'r')
        
        try:
            # Find time variable
            time_var = None
            for var_name in ['waveTime', 'time']:
                if var_name in nc.variables:
                    time_var = var_name
                    break
            
            if time_var is None:
                raise ValueError("Could not find time variable")
            
            # Parse time
            time_data = nc.variables[time_var][:]
            time_units = nc.variables[time_var].units
            time_values = netCDF4.num2date(time_data, time_units)
            time_values = pd.to_datetime([t.isoformat() for t in time_values])
            
            # Extract wave variables
            var_mapping = {
                'waveHs': 'hs', 'waveTp': 'tp', 'waveDp': 'dp', 'waveTa': 'ta',
                'Hs': 'hs', 'Tp': 'tp', 'Dp': 'dp', 'Ta': 'ta'
            }
            
            data = {}
            for nc_var, df_var in var_mapping.items():
                if nc_var in nc.variables and df_var not in data:
                    data[df_var] = nc.variables[nc_var][:]
            
            df = pd.DataFrame(data, index=time_values)
            df.index.name = 'time'
            
        finally:
            nc.close()
        
        # Filter by date range
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        return df.sort_index()
    
    def compute_wave_power(self, hs: np.ndarray, tp: np.ndarray) -> np.ndarray:
        """
        Compute wave power flux (kW/m).
        
        Formula: P = (ρ * g² / 64π) * Hs² * Tp
        
        Args:
            hs: Significant wave height (m)
            tp: Peak period (s)
            
        Returns:
            Wave power in kW/m
        """
        # P = (ρg²/64π) * Hs² * Tp  [W/m]
        power_wm = (RHO_WATER * G**2 / (64 * np.pi)) * hs**2 * tp
        return power_wm / 1000  # Convert to kW/m
    
    def compute_wave_energy(self, hs: np.ndarray) -> np.ndarray:
        """
        Compute wave energy density (J/m²).
        
        Formula: E = (1/16) * ρ * g * Hs²
        
        Args:
            hs: Significant wave height (m)
            
        Returns:
            Wave energy density in J/m²
        """
        return (1/16) * RHO_WATER * G * hs**2
    
    def compute_runup_stockdon(
        self,
        hs: np.ndarray,
        tp: np.ndarray,
        beach_slope: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute 2% exceedance runup using Stockdon et al. (2006).
        
        For dissipative beaches (ξ < 0.3):
            R2% = 0.043 * sqrt(Hs * L0)
        
        For intermediate/reflective beaches:
            R2% = 1.1 * (0.35 * βf * sqrt(Hs * L0) + 
                   0.5 * sqrt(Hs * L0 * (0.563 * βf² + 0.004)))
        
        Args:
            hs: Significant wave height (m)
            tp: Peak period (s)
            beach_slope: Beach slope tan(β). Uses config default if None.
            
        Returns:
            2% exceedance runup (m)
        """
        if beach_slope is None:
            beach_slope = self.config.beach_slope
        
        # Deep water wavelength: L0 = g * T² / (2π)
        L0 = G * tp**2 / (2 * np.pi)
        
        # Iribarren number: ξ = tan(β) / sqrt(Hs/L0)
        with np.errstate(divide='ignore', invalid='ignore'):
            iribarren = beach_slope / np.sqrt(hs / L0)
        
        # Compute runup based on beach type
        runup = np.zeros_like(hs)
        
        # Dissipative (ξ < 0.3)
        dissipative = iribarren < 0.3
        runup[dissipative] = 0.043 * np.sqrt(hs[dissipative] * L0[dissipative])
        
        # Intermediate/reflective
        other = ~dissipative & ~np.isnan(iribarren)
        sqrt_hs_l0 = np.sqrt(hs[other] * L0[other])
        setup = 0.35 * beach_slope * sqrt_hs_l0
        swash = 0.5 * np.sqrt(
            hs[other] * L0[other] * (0.563 * beach_slope**2 + 0.004)
        )
        runup[other] = 1.1 * (setup + swash)
        
        return runup
    
    def compute_shore_normal_component(
        self,
        hs: np.ndarray,
        dp: np.ndarray,
        cliff_orientation_deg: float
    ) -> np.ndarray:
        """
        Compute shore-normal wave height component.
        
        Args:
            hs: Significant wave height (m)
            dp: Peak wave direction (degrees from N, 0=N, 90=E)
            cliff_orientation_deg: Cliff face orientation (degrees from N)
            
        Returns:
            Shore-normal component of wave height (m)
        """
        # Convert to radians
        dp_rad = np.deg2rad(dp)
        cliff_rad = np.deg2rad(cliff_orientation_deg)
        
        # Shore-normal is perpendicular to cliff face
        # Waves approaching perpendicular to cliff have max impact
        angle_diff = dp_rad - cliff_rad
        
        # Use absolute cosine - waves from either side can impact
        return hs * np.abs(np.cos(angle_diff))
    
    def detect_storms(
        self,
        hs: np.ndarray,
        time_index: pd.DatetimeIndex
    ) -> Dict[str, Union[int, float, List]]:
        """
        Detect storm events and compute storm statistics.
        
        Args:
            hs: Significant wave height time series (m)
            time_index: Datetime index for the time series
            
        Returns:
            Dictionary with storm metrics
        """
        threshold = self.config.storm_hs_threshold_m
        gap_hours = self.config.storm_gap_hours
        
        # Identify storm hours
        is_storm = hs >= threshold
        storm_hours = np.sum(is_storm)
        
        if storm_hours == 0:
            return {
                'storm_hours': 0,
                'storm_count': 0,
                'max_storm_duration_hr': 0,
                'mean_storm_duration_hr': 0,
                'time_since_storm_hr': len(hs),  # Assume hourly data
                'storm_events': []
            }
        
        # Find storm events (contiguous periods with potential gaps)
        storm_events = []
        in_storm = False
        storm_start = None
        last_storm_end = None
        
        for i, (t, storm) in enumerate(zip(time_index, is_storm)):
            if storm and not in_storm:
                # Start of new storm or continuation after gap
                if last_storm_end is not None:
                    gap = (t - last_storm_end).total_seconds() / 3600
                    if gap <= gap_hours:
                        # Continue previous storm
                        in_storm = True
                        continue
                
                storm_start = t
                in_storm = True
                
            elif not storm and in_storm:
                # Potential end of storm
                last_storm_end = time_index[i-1]
                in_storm = False
                
                # Check if storm actually ended (gap > threshold)
                remaining = is_storm[i:]
                if len(remaining) > 0:
                    next_storm_idx = np.where(remaining)[0]
                    if len(next_storm_idx) > 0:
                        next_storm_time = time_index[i + next_storm_idx[0]]
                        gap = (next_storm_time - last_storm_end).total_seconds() / 3600
                        if gap > gap_hours:
                            # Storm ended
                            storm_events.append({
                                'start': storm_start,
                                'end': last_storm_end,
                                'duration_hr': (last_storm_end - storm_start).total_seconds() / 3600
                            })
                    else:
                        # No more storms
                        storm_events.append({
                            'start': storm_start,
                            'end': last_storm_end,
                            'duration_hr': (last_storm_end - storm_start).total_seconds() / 3600
                        })
        
        # Handle case where storm extends to end of series
        if in_storm:
            storm_events.append({
                'start': storm_start,
                'end': time_index[-1],
                'duration_hr': (time_index[-1] - storm_start).total_seconds() / 3600
            })
        
        # Compute statistics
        if len(storm_events) > 0:
            durations = [e['duration_hr'] for e in storm_events]
            max_duration = max(durations)
            mean_duration = np.mean(durations)
            
            # Time since last storm
            last_storm_end_time = storm_events[-1]['end']
            time_since = (time_index[-1] - last_storm_end_time).total_seconds() / 3600
        else:
            max_duration = 0
            mean_duration = 0
            time_since = len(hs)  # Full window length
        
        return {
            'storm_hours': int(storm_hours),
            'storm_count': len(storm_events),
            'max_storm_duration_hr': float(max_duration),
            'mean_storm_duration_hr': float(mean_duration),
            'time_since_storm_hr': float(max(0, time_since)),
            'storm_events': storm_events
        }
    
    def compute_rolling_max(
        self,
        hs: pd.Series,
        window_days: int
    ) -> float:
        """
        Compute maximum of rolling max over specified window.
        
        Args:
            hs: Wave height time series (hourly)
            window_days: Rolling window size in days
            
        Returns:
            Maximum value of rolling max
        """
        window_hours = window_days * 24
        rolling_max = hs.rolling(window=window_hours, min_periods=1).max()
        return float(rolling_max.max())
    
    def compute_trend(self, values: np.ndarray, time_hours: np.ndarray) -> float:
        """
        Compute linear trend using least squares.
        
        Args:
            values: Data values
            time_hours: Time in hours from start
            
        Returns:
            Slope in units per day
        """
        # Remove NaN values
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return 0.0
        
        x = time_hours[mask]
        y = values[mask]
        
        # Linear regression: y = mx + b
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x**2)
        
        denom = n * sum_x2 - sum_x**2
        if abs(denom) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denom
        
        # Convert from per-hour to per-day
        return float(slope * 24)
    
    def compute_circular_mean(self, angles_deg: np.ndarray) -> float:
        """
        Compute circular mean of angles.
        
        Args:
            angles_deg: Angles in degrees
            
        Returns:
            Mean angle in degrees [0, 360)
        """
        angles_rad = np.deg2rad(angles_deg)
        sin_mean = np.nanmean(np.sin(angles_rad))
        cos_mean = np.nanmean(np.cos(angles_rad))
        mean_rad = np.arctan2(sin_mean, cos_mean)
        mean_deg = np.rad2deg(mean_rad)
        return float(mean_deg % 360)
    
    def compute_all_metrics(
        self,
        df: pd.DataFrame,
        cliff_orientation_deg: Optional[float] = None,
        beach_slope: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute all wave metrics from a DataFrame.
        
        Args:
            df: DataFrame with columns hs, tp, dp (and optionally ta)
            cliff_orientation_deg: Cliff face orientation for shore-normal calc
            beach_slope: Beach slope for runup calculation
            
        Returns:
            Dictionary of computed metrics
        """
        if df.empty:
            logger.warning("Empty DataFrame, returning NaN metrics")
            return {name: np.nan for name in self.config.feature_names}
        
        hs = df['hs'].values
        tp = df['tp'].values
        dp = df['dp'].values
        
        # Handle missing values
        valid_mask = ~(np.isnan(hs) | np.isnan(tp))
        
        metrics = {}
        
        # ===== Raw features (mean over window for summary) =====
        metrics['hs_m'] = float(np.nanmean(hs))
        metrics['tp_s'] = float(np.nanmean(tp))
        metrics['dp_deg'] = self.compute_circular_mean(dp)
        
        # Wave power
        power = self.compute_wave_power(hs, tp)
        metrics['power_kw'] = float(np.nanmean(power))
        
        # ===== Shore-normal component =====
        if cliff_orientation_deg is not None:
            shore_normal = self.compute_shore_normal_component(
                hs, dp, cliff_orientation_deg
            )
            metrics['shore_normal_hs_m'] = float(np.nanmean(shore_normal))
        else:
            metrics['shore_normal_hs_m'] = np.nan
        
        # ===== Runup =====
        runup = self.compute_runup_stockdon(hs, tp, beach_slope)
        metrics['runup_2pct_m'] = float(np.nanmean(runup[valid_mask]))
        
        # ===== Integrated metrics =====
        # Assume hourly data, convert to proper units
        hours = len(df)
        
        # Energy: integrate over time (J/m² * hours)
        energy = self.compute_wave_energy(hs)
        # Convert to MJ: J/m² * 3600 s/hr * hours / 1e6
        metrics['cumulative_energy_mj'] = float(
            np.nansum(energy) * 3600 / 1e6
        )
        
        # Power: integrate over time (kW/m * hours = kWh/m)
        metrics['cumulative_power_kwh'] = float(np.nansum(power))
        metrics['mean_power_kw'] = float(np.nanmean(power))
        
        # ===== Extreme metrics =====
        metrics['max_hs_m'] = float(np.nanmax(hs))
        metrics['hs_p90'] = float(np.nanpercentile(hs[~np.isnan(hs)], 90))
        metrics['hs_p99'] = float(np.nanpercentile(hs[~np.isnan(hs)], 99))
        
        # ===== Storm metrics =====
        storm_stats = self.detect_storms(hs, df.index)
        metrics['storm_hours'] = storm_stats['storm_hours']
        metrics['storm_count'] = storm_stats['storm_count']
        metrics['max_storm_duration_hr'] = storm_stats['max_storm_duration_hr']
        metrics['time_since_storm_hr'] = storm_stats['time_since_storm_hr']
        metrics['mean_storm_duration_hr'] = storm_stats['mean_storm_duration_hr']
        
        # ===== Rolling/temporal metrics =====
        hs_series = pd.Series(hs, index=df.index)
        metrics['rolling_max_7d_m'] = self.compute_rolling_max(hs_series, 7)
        
        # Trend
        time_hours = (df.index - df.index[0]).total_seconds().values / 3600
        metrics['hs_trend_slope'] = self.compute_trend(hs, time_hours)
        
        return metrics
    
    def compute_timeseries_features(
        self,
        df: pd.DataFrame,
        resample_hours: Optional[int] = None,
        cliff_orientation_deg: Optional[float] = None,
        beach_slope: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time series features for model input.
        
        Returns resampled features suitable for the transformer encoder.
        
        Args:
            df: DataFrame with wave data
            resample_hours: Resampling interval (default from config)
            cliff_orientation_deg: Cliff orientation for shore-normal calc
            beach_slope: Beach slope for runup calculation
            
        Returns:
            Tuple of:
                - features: (T, n_features) array
                - day_of_year: (T,) array
        """
        if resample_hours is None:
            resample_hours = self.config.resample_hours
        
        # Resample to specified interval
        resample_rule = f'{resample_hours}h'
        
        # Resample with appropriate aggregations
        resampled = pd.DataFrame()
        
        # For hs, tp - use mean
        resampled['hs'] = df['hs'].resample(resample_rule).mean()
        resampled['tp'] = df['tp'].resample(resample_rule).mean()
        
        # For direction - use circular mean
        def circular_mean_resample(x):
            if len(x) == 0 or x.isna().all():
                return np.nan
            return self.compute_circular_mean(x.dropna().values)
        
        resampled['dp'] = df['dp'].resample(resample_rule).apply(circular_mean_resample)
        
        # Drop any rows with all NaN
        resampled = resampled.dropna(how='all')
        
        # Compute derived features
        hs = resampled['hs'].values
        tp = resampled['tp'].values
        dp = resampled['dp'].values
        
        # Wave power
        power = self.compute_wave_power(hs, tp)
        
        # Basic features for model
        features = np.column_stack([hs, tp, dp, power])
        
        # Add shore-normal if orientation provided
        if cliff_orientation_deg is not None:
            shore_normal = self.compute_shore_normal_component(
                hs, dp, cliff_orientation_deg
            )
            features = np.column_stack([features, shore_normal])
        
        # Add runup
        runup = self.compute_runup_stockdon(hs, tp, beach_slope)
        features = np.column_stack([features, runup])
        
        # Day of year for seasonality encoding
        day_of_year = resampled.index.dayofyear.values
        
        return features.astype(np.float32), day_of_year.astype(np.int32)
    
    def process_file(
        self,
        filepath: Union[str, Path],
        scan_date: datetime,
        lookback_days: Optional[int] = None,
        cliff_orientation_deg: Optional[float] = None,
        beach_slope: Optional[float] = None,
        return_timeseries: bool = False
    ) -> Union[Dict[str, float], Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
        """
        Process a single CDIP file for a given scan date.
        
        Args:
            filepath: Path to CDIP NetCDF file
            scan_date: Date of LiDAR scan (end of lookback window)
            lookback_days: Days of history to include
            cliff_orientation_deg: Cliff face orientation
            beach_slope: Beach slope for runup
            return_timeseries: If True, also return time series features
            
        Returns:
            If return_timeseries=False: Dictionary of summary metrics
            If return_timeseries=True: Tuple of (features, day_of_year, metrics)
        """
        if lookback_days is None:
            lookback_days = self.config.lookback_days
        
        # Compute date range
        end_date = pd.Timestamp(scan_date)
        start_date = end_date - timedelta(days=lookback_days)
        
        # Load data
        df = self.load_cdip_data(filepath, start_date, end_date)
        
        if df.empty:
            logger.warning(f"No data for {filepath} in range {start_date} to {end_date}")
            metrics = {name: np.nan for name in self.config.feature_names}
            if return_timeseries:
                n_timesteps = lookback_days * 24 // self.config.resample_hours
                empty_features = np.full((n_timesteps, 6), np.nan, dtype=np.float32)
                empty_doy = np.zeros(n_timesteps, dtype=np.int32)
                return empty_features, empty_doy, metrics
            return metrics
        
        # Compute summary metrics
        metrics = self.compute_all_metrics(
            df, cliff_orientation_deg, beach_slope
        )
        
        if return_timeseries:
            features, day_of_year = self.compute_timeseries_features(
                df, cliff_orientation_deg=cliff_orientation_deg,
                beach_slope=beach_slope
            )
            return features, day_of_year, metrics
        
        return metrics


def process_batch(
    input_dir: Path,
    output_dir: Path,
    scan_dates_file: Optional[Path] = None,
    config: Optional[WaveMetricsConfig] = None
):
    """
    Batch process all CDIP files in a directory.
    
    Args:
        input_dir: Directory containing CDIP NetCDF files
        output_dir: Output directory for processed metrics
        scan_dates_file: CSV with mop_id, scan_date, cliff_orientation columns
        config: Processing configuration
    """
    calculator = WaveMetricsCalculator(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CDIP files
    nc_files = list(input_dir.glob('D*_hindcast.nc'))
    logger.info(f"Found {len(nc_files)} CDIP files")
    
    if scan_dates_file is not None:
        # Process for specific scan dates
        scan_df = pd.read_csv(scan_dates_file)
        scan_df['scan_date'] = pd.to_datetime(scan_df['scan_date'])
        
        all_metrics = []
        
        for _, row in scan_df.iterrows():
            mop_id = row['mop_id']
            scan_date = row['scan_date']
            orientation = row.get('cliff_orientation', None)
            
            # Find matching file
            nc_file = input_dir / f'D{mop_id:04d}_hindcast.nc'
            if not nc_file.exists():
                nc_file = input_dir / f'D{mop_id}_hindcast.nc'
            
            if not nc_file.exists():
                logger.warning(f"No file found for MOP {mop_id}")
                continue
            
            metrics = calculator.process_file(
                nc_file, scan_date, cliff_orientation_deg=orientation
            )
            metrics['mop_id'] = mop_id
            metrics['scan_date'] = scan_date
            all_metrics.append(metrics)
        
        # Save as parquet
        result_df = pd.DataFrame(all_metrics)
        output_path = output_dir / 'wave_metrics.parquet'
        result_df.to_parquet(output_path)
        logger.info(f"Saved metrics to {output_path}")
        
    else:
        # Process each file for full time range
        for nc_file in nc_files:
            mop_id = nc_file.stem.replace('_hindcast', '').replace('D', '')
            
            df = calculator.load_cdip_data(nc_file)
            if df.empty:
                continue
            
            metrics = calculator.compute_all_metrics(df)
            metrics['mop_id'] = mop_id
            
            output_path = output_dir / f'{nc_file.stem}_metrics.parquet'
            pd.DataFrame([metrics]).to_parquet(output_path)
            logger.info(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute wave metrics from CDIP MOP data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=Path,
        help='Single CDIP NetCDF file to process'
    )
    input_group.add_argument(
        '--input-dir',
        type=Path,
        help='Directory of CDIP NetCDF files for batch processing'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file path (for single file processing)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory (for batch processing)'
    )
    parser.add_argument(
        '--output-format',
        choices=['parquet', 'csv', 'json', 'tensor'],
        default='parquet',
        help='Output format (default: parquet)'
    )
    
    # Date options
    parser.add_argument(
        '--scan-date',
        type=str,
        help='Scan date for lookback window (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for time range (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for time range (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=90,
        help='Days of history for lookback window (default: 90)'
    )
    parser.add_argument(
        '--scan-dates',
        type=Path,
        help='CSV file with scan dates for batch processing'
    )
    
    # Physical parameters
    parser.add_argument(
        '--cliff-orientation',
        type=float,
        help='Cliff face orientation in degrees from N (for shore-normal calc)'
    )
    parser.add_argument(
        '--beach-slope',
        type=float,
        default=0.1,
        help='Beach slope tan(β) for runup calculation (default: 0.1)'
    )
    parser.add_argument(
        '--storm-threshold',
        type=float,
        default=2.0,
        help='Hs threshold for storm detection in meters (default: 2.0)'
    )
    
    # Processing options
    parser.add_argument(
        '--resample-hours',
        type=int,
        default=6,
        help='Resampling interval in hours (default: 6)'
    )
    parser.add_argument(
        '--include-timeseries',
        action='store_true',
        help='Also output resampled time series features'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = WaveMetricsConfig(
        lookback_days=args.lookback_days,
        resample_hours=args.resample_hours,
        storm_hs_threshold_m=args.storm_threshold,
        beach_slope=args.beach_slope
    )
    
    calculator = WaveMetricsCalculator(config)
    
    if args.input_dir:
        # Batch processing
        if not args.output_dir:
            args.output_dir = args.input_dir / 'metrics'
        
        process_batch(
            args.input_dir,
            args.output_dir,
            args.scan_dates,
            config
        )
        
    else:
        # Single file processing
        if args.scan_date:
            scan_date = datetime.strptime(args.scan_date, '%Y-%m-%d')
        elif args.end_date:
            scan_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        else:
            # Use end of file
            df = calculator.load_cdip_data(args.input)
            scan_date = df.index.max()
        
        if args.include_timeseries:
            features, day_of_year, metrics = calculator.process_file(
                args.input,
                scan_date,
                cliff_orientation_deg=args.cliff_orientation,
                beach_slope=args.beach_slope,
                return_timeseries=True
            )
        else:
            metrics = calculator.process_file(
                args.input,
                scan_date,
                cliff_orientation_deg=args.cliff_orientation,
                beach_slope=args.beach_slope
            )
        
        # Output
        if args.output:
            output_path = args.output
        else:
            output_path = args.input.with_suffix(f'.metrics.{args.output_format}')
        
        if args.output_format == 'parquet':
            pd.DataFrame([metrics]).to_parquet(output_path)
        elif args.output_format == 'csv':
            pd.DataFrame([metrics]).to_csv(output_path, index=False)
        elif args.output_format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        elif args.output_format == 'tensor':
            # Save as NPZ for PyTorch
            if args.include_timeseries:
                np.savez(
                    output_path,
                    features=features,
                    day_of_year=day_of_year,
                    metrics=np.array(list(metrics.values())),
                    metric_names=list(metrics.keys())
                )
            else:
                np.savez(
                    output_path,
                    metrics=np.array(list(metrics.values())),
                    metric_names=list(metrics.keys())
                )
        
        logger.info(f"Saved metrics to {output_path}")
        
        # Print summary
        print("\n=== Wave Metrics Summary ===")
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.3f}")
            else:
                print(f"  {name}: {value}")


if __name__ == '__main__':
    main()
