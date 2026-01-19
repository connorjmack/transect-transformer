"""
CDIP MOP Wave Data Loader for CliffCast

Fetches nearshore wave predictions from CDIP's MOP (Monitoring and Prediction) system
via THREDDS/OPeNDAP. Data is available at 100m alongshore spacing at 10m water depth.

CDIP MOP Documentation: https://cdip.ucsd.edu/MOP_v1.1/
THREDDS Server: https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/model/MOP_alongshore/

Author: Connor (CliffCast project)
Date: 2026-01-18
"""

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Union, List
import logging
from dataclasses import dataclass
import warnings

# Constants
RHO_SEAWATER = 1025  # kg/m³
G = 9.81  # m/s²

logger = logging.getLogger(__name__)


@dataclass
class WaveData:
    """Container for wave data output."""
    time: np.ndarray           # datetime64 array
    hs: np.ndarray             # Significant wave height (m)
    tp: np.ndarray             # Peak period (s)
    dp: np.ndarray             # Peak direction (deg from N)
    ta: np.ndarray             # Average period (s)
    power: np.ndarray          # Wave power (kW/m)
    sxy: Optional[np.ndarray]  # Alongshore radiation stress (m²)
    sxx: Optional[np.ndarray]  # Onshore radiation stress (m²)
    latitude: float
    longitude: float
    water_depth: float
    mop_id: int
    
    def to_tensor(self, history_days: int = 90, 
                  reference_date: Optional[datetime] = None,
                  resample_hours: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to CliffCast tensor format.
        
        Args:
            history_days: Number of days of history to include
            reference_date: End date for the time window (default: last available)
            resample_hours: Resample interval in hours (default: 6)
            
        Returns:
            features: (T_w, 4) array of [hs, tp, dp, power]
            doy: (T_w,) array of day-of-year for temporal encoding
        """
        import pandas as pd
        
        # Create time index
        times = pd.to_datetime(self.time)
        
        # Determine reference date
        if reference_date is None:
            ref_dt = times[-1]
        else:
            ref_dt = pd.Timestamp(reference_date)
            
        # Filter to history window
        start_dt = ref_dt - pd.Timedelta(days=history_days)
        mask = (times >= start_dt) & (times <= ref_dt)
        
        if not mask.any():
            raise ValueError(f"No data available in window {start_dt} to {ref_dt}")
        
        # Create DataFrame for resampling
        df = pd.DataFrame({
            'hs': self.hs[mask],
            'tp': self.tp[mask],
            'dp': self.dp[mask],
            'power': self.power[mask],
        }, index=times[mask])
        
        # Handle direction resampling with circular mean
        df['dp_sin'] = np.sin(np.radians(df['dp']))
        df['dp_cos'] = np.cos(np.radians(df['dp']))
        
        # Resample
        resample_rule = f'{resample_hours}h'
        df_resampled = df[['hs', 'tp', 'power', 'dp_sin', 'dp_cos']].resample(resample_rule).mean()
        
        # Compute circular mean direction
        df_resampled['dp'] = np.degrees(np.arctan2(
            df_resampled['dp_sin'], 
            df_resampled['dp_cos']
        )) % 360
        
        # Drop helper columns and NaN rows
        df_resampled = df_resampled[['hs', 'tp', 'dp', 'power']].dropna()
        
        # Convert to arrays
        features = df_resampled.values.astype(np.float32)
        doy = df_resampled.index.dayofyear.values.astype(np.int32)
        
        return features, doy


class CDIPWaveLoader:
    """
    Load wave data from CDIP MOP THREDDS server or local files.
    
    CDIP provides nearshore wave predictions at 100m alongshore spacing
    at 10m water depth for the California coast. Data is hourly from 2000-present.
    
    Data access options:
        1. THREDDS/OPeNDAP (default): Direct remote access
        2. Local NetCDF files: Pre-downloaded from THREDDS
        3. Local cache: Auto-caches THREDDS downloads
    
    San Diego MOP ID mapping:
        Your transect MOP IDs (520-764) map to CDIP site labels:
        - MOP 520 → D0520 (or similar county prefix)
        - MOP 582 → D0582 (Black's Beach area)
        - MOP 637 → D0637 (Solana Beach)
        
    THREDDS URLs follow the pattern:
        https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore/{site}_hindcast.nc
        
    To find available sites, browse:
        https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/model/MOP_alongshore/catalog.html
    
    Example usage:
        loader = CDIPWaveLoader()
        
        # Load wave data for MOP 582 (Black's Beach area)
        waves = loader.load_mop(
            mop_id=582,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        # Convert to CliffCast tensor format
        features, doy = waves.to_tensor(
            history_days=90,
            reference_date='2023-12-15',
            resample_hours=6
        )
        print(f"Shape: {features.shape}")  # (360, 4) for 90 days @ 6hr
        
    Example with local files:
        loader = CDIPWaveLoader(local_dir='data/cdip/')
        waves = loader.load_mop(mop_id=582)  # Loads from data/cdip/D0582_hindcast.nc
    """
    
    # THREDDS base URL for MOP alongshore data
    THREDDS_BASE = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore"
    
    # County codes for MOP site naming (south to north)
    COUNTY_CODES = {
        'san_diego': 'D',
        'orange': 'OC',
        'los_angeles': 'L',
        'ventura': 'VE',
        'santa_barbara': 'B',
        'san_luis_obispo': 'SL',
        'monterey': 'MO',
        'santa_cruz': 'SC',
        'san_mateo': 'SM',
        'san_francisco': 'SF',
        'marin': 'MA',
        'sonoma': 'SN',
        'mendocino': 'M',
        'humboldt': 'HU',
        'del_norte': 'DN',
    }
    
    # San Diego MOP range (approximate)
    SD_MOP_RANGE = (1, 1000)  # MOPs in San Diego county
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        local_dir: Optional[Path] = None,
        timeout: int = 60,
    ):
        """
        Initialize the CDIP wave loader.
        
        Args:
            cache_dir: Directory for caching downloaded data (optional)
            local_dir: Directory containing pre-downloaded NetCDF files (optional)
                       Files should be named like D0582_hindcast.nc
            timeout: Request timeout in seconds
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.local_dir = Path(local_dir) if local_dir else None
        self.timeout = timeout
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_site_label(self, mop_id: int, county: str = 'san_diego') -> str:
        """
        Convert MOP ID to CDIP site label.
        
        San Diego MOPs are labeled as D0001, D0002, ..., D1000
        """
        prefix = self.COUNTY_CODES.get(county, 'D')
        return f"{prefix}{mop_id:04d}"
    
    def _get_thredds_url(self, site_label: str, dataset: str = 'hindcast') -> str:
        """
        Construct THREDDS OPeNDAP URL for a site.
        
        Args:
            site_label: CDIP site label (e.g., 'D0582')
            dataset: 'hindcast' or 'nowcast'
            
        Returns:
            OPeNDAP URL for the dataset
        """
        return f"{self.THREDDS_BASE}/{site_label}_{dataset}.nc"
    
    def _compute_wave_power(self, hs: np.ndarray, tp: np.ndarray) -> np.ndarray:
        """
        Compute wave power flux (kW/m) from Hs and Tp.
        
        P = (ρ * g² / 64π) * Hs² * Tp
        
        In deep water, this simplifies to approximately:
        P ≈ 0.49 * Hs² * Tp (kW/m)
        """
        # Full formula
        coeff = (RHO_SEAWATER * G**2) / (64 * np.pi)
        power_w_per_m = coeff * hs**2 * tp
        power_kw_per_m = power_w_per_m / 1000
        
        return power_kw_per_m
    
    def _handle_fill_values(self, data: np.ndarray, fill_value: float = -999.99) -> np.ndarray:
        """Replace fill values with NaN."""
        data = data.astype(np.float64)
        data[data <= fill_value + 1] = np.nan
        return data
    
    def load_mop(
        self,
        mop_id: int,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        county: str = 'san_diego',
        dataset: str = 'hindcast',
        variables: Optional[List[str]] = None,
        site_label_override: Optional[str] = None,
    ) -> WaveData:
        """
        Load wave data for a specific MOP transect.
        
        Data source priority:
            1. Local directory (if configured and file exists)
            2. Cache directory (if configured and file exists)
            3. THREDDS server (remote fetch)
        
        Args:
            mop_id: MOP transect ID (e.g., 582 for Black's Beach)
            start_date: Start of time range (default: all available)
            end_date: End of time range (default: all available)
            county: County name (default: 'san_diego')
            dataset: 'hindcast' or 'nowcast' (default: 'hindcast')
            variables: List of variables to load (default: all)
            site_label_override: Override auto-generated site label (e.g., 'D0582')
            
        Returns:
            WaveData object with time series and metadata
        """
        site_label = site_label_override or self._get_site_label(mop_id, county)
        filename = f"{site_label}_{dataset}.nc"
        
        logger.info(f"Loading wave data for site {site_label}")
        
        ds = None
        
        # Try local directory first
        if self.local_dir:
            local_path = self.local_dir / filename
            if local_path.exists():
                logger.info(f"Loading from local file: {local_path}")
                ds = xr.open_dataset(local_path, decode_times=False)
        
        # Try cache directory
        if ds is None and self.cache_dir:
            cache_path = self.cache_dir / filename
            if cache_path.exists():
                logger.info(f"Loading from cache: {cache_path}")
                ds = xr.open_dataset(cache_path, decode_times=False)
        
        # Fetch from THREDDS
        if ds is None:
            url = self._get_thredds_url(site_label, dataset)
            logger.info(f"Fetching from THREDDS: {url}")
            
            try:
                ds = self._fetch_from_thredds(url)
                
                # Cache the full dataset
                if self.cache_dir:
                    cache_path = self.cache_dir / filename
                    logger.info(f"Caching to: {cache_path}")
                    ds.to_netcdf(cache_path)
                    
            except Exception as e:
                # Provide helpful error message
                raise ConnectionError(
                    f"Failed to load wave data for MOP {mop_id} (site {site_label}).\n"
                    f"URL: {url}\n"
                    f"Error: {e}\n\n"
                    f"Possible solutions:\n"
                    f"  1. Check if site label is correct (browse THREDDS catalog)\n"
                    f"  2. Pre-download files to local_dir: {self.local_dir or 'not configured'}\n"
                    f"  3. Check network connectivity to thredds.cdip.ucsd.edu\n"
                    f"  4. Try site_label_override='D0582' format\n"
                ) from e
        
        # Parse time range
        if start_date:
            start_dt = np.datetime64(start_date)
        else:
            start_dt = None
            
        if end_date:
            end_dt = np.datetime64(end_date)
        else:
            end_dt = None
        
        # Convert time from seconds since 1970 to datetime64
        time_seconds = ds['waveTime'].values
        time = (np.datetime64('1970-01-01') + 
                (time_seconds * 1e9).astype('timedelta64[ns]'))
        
        # Filter time range
        if start_dt is not None or end_dt is not None:
            mask = np.ones(len(time), dtype=bool)
            if start_dt is not None:
                mask &= time >= start_dt
            if end_dt is not None:
                mask &= time <= end_dt
            time_idx = np.where(mask)[0]
        else:
            time_idx = slice(None)
        
        # Extract variables
        hs = self._handle_fill_values(ds['waveHs'].values[time_idx])
        tp = self._handle_fill_values(ds['waveTp'].values[time_idx])
        dp = self._handle_fill_values(ds['waveDp'].values[time_idx])
        ta = self._handle_fill_values(ds['waveTa'].values[time_idx])
        
        # Optional radiation stress variables
        sxy = None
        sxx = None
        if 'waveSxy' in ds:
            sxy = self._handle_fill_values(ds['waveSxy'].values[time_idx])
        if 'waveSxx' in ds:
            sxx = self._handle_fill_values(ds['waveSxx'].values[time_idx])
        
        # Compute wave power
        power = self._compute_wave_power(hs, tp)
        
        # Extract metadata
        latitude = float(ds['metaLatitude'].values)
        longitude = float(ds['metaLongitude'].values)
        water_depth = float(ds['metaWaterDepth'].values)
        
        ds.close()
        
        return WaveData(
            time=time[time_idx] if isinstance(time_idx, np.ndarray) else time,
            hs=hs,
            tp=tp,
            dp=dp,
            ta=ta,
            power=power,
            sxy=sxy,
            sxx=sxx,
            latitude=latitude,
            longitude=longitude,
            water_depth=water_depth,
            mop_id=mop_id,
        )
    
    def _fetch_from_thredds(self, url: str) -> xr.Dataset:
        """Fetch dataset from THREDDS server."""
        try:
            # Use decode_times=False to handle the custom time format
            ds = xr.open_dataset(
                url,
                decode_times=False,
                engine='netcdf4',
            )
            return ds
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise
    
    def load_multiple_mops(
        self,
        mop_ids: List[int],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        **kwargs,
    ) -> dict[int, WaveData]:
        """
        Load wave data for multiple MOP transects.
        
        Args:
            mop_ids: List of MOP IDs to load
            start_date: Start of time range
            end_date: End of time range
            **kwargs: Additional arguments passed to load_mop
            
        Returns:
            Dictionary mapping MOP ID to WaveData
        """
        results = {}
        for mop_id in mop_ids:
            try:
                results[mop_id] = self.load_mop(
                    mop_id, start_date, end_date, **kwargs
                )
            except Exception as e:
                logger.warning(f"Failed to load MOP {mop_id}: {e}")
        return results
    
    def get_cliffcast_tensors(
        self,
        mop_id: int,
        reference_date: Union[str, datetime],
        history_days: int = 90,
        resample_hours: int = 6,
        county: str = 'san_diego',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method to get wave data in CliffCast format directly.
        
        Args:
            mop_id: MOP transect ID
            reference_date: Reference date (end of history window)
            history_days: Days of history to include (default: 90)
            resample_hours: Resample interval (default: 6)
            county: County name
            
        Returns:
            features: (T_w, 4) array of [hs, tp, dp, power]
            doy: (T_w,) array of day-of-year
        """
        # Load data with buffer for resampling edge effects
        ref_dt = datetime.fromisoformat(str(reference_date).split('T')[0])
        start_dt = ref_dt - timedelta(days=history_days + 7)
        
        wave_data = self.load_mop(
            mop_id=mop_id,
            start_date=start_dt,
            end_date=ref_dt,
            county=county,
        )
        
        return wave_data.to_tensor(
            history_days=history_days,
            reference_date=ref_dt,
            resample_hours=resample_hours,
        )


def compute_derived_features(wave_data: WaveData) -> dict:
    """
    Compute derived wave features for epoch-level analysis.
    
    These features summarize wave conditions between LiDAR epochs
    and can be used as additional inputs or for validation.
    
    Args:
        wave_data: WaveData object with time series
        
    Returns:
        Dictionary of derived features
    """
    hs = wave_data.hs
    tp = wave_data.tp
    dp = wave_data.dp
    power = wave_data.power
    
    # Handle NaN values
    hs_valid = hs[~np.isnan(hs)]
    tp_valid = tp[~np.isnan(tp)]
    power_valid = power[~np.isnan(power)]
    
    # Direction circular statistics
    dp_rad = np.radians(dp[~np.isnan(dp)])
    
    # Compute derived features
    features = {
        # Wave height statistics
        'hs_mean': np.nanmean(hs),
        'hs_median': np.nanmedian(hs),
        'hs_std': np.nanstd(hs),
        'hs_max': np.nanmax(hs),
        'hs_90q': np.nanpercentile(hs_valid, 90) if len(hs_valid) > 0 else np.nan,
        'hs_95q': np.nanpercentile(hs_valid, 95) if len(hs_valid) > 0 else np.nan,
        
        # Period statistics
        'tp_mean': np.nanmean(tp),
        'tp_max': np.nanmax(tp),
        
        # Wave power statistics
        'power_mean': np.nanmean(power),
        'power_max': np.nanmax(power),
        'power_cumulative': np.nansum(power),  # Cumulative energy proxy
        
        # Storm metrics (Hs > threshold)
        'storm_hours_3m': np.sum(hs > 3.0),  # Hours with Hs > 3m
        'storm_hours_4m': np.sum(hs > 4.0),  # Hours with Hs > 4m
        'high_energy_fraction': np.mean(hs > 2.0),  # Fraction of time Hs > 2m
        
        # Direction statistics (circular)
        'dp_mean': np.degrees(np.arctan2(
            np.mean(np.sin(dp_rad)),
            np.mean(np.cos(dp_rad))
        )) % 360 if len(dp_rad) > 0 else np.nan,
        'dp_spread': np.degrees(np.std(dp_rad)) if len(dp_rad) > 0 else np.nan,
        
        # Radiation stress (if available)
        'sxy_net': np.nansum(wave_data.sxy) if wave_data.sxy is not None else np.nan,
        'sxx_net': np.nansum(wave_data.sxx) if wave_data.sxx is not None else np.nan,
        
        # Data quality
        'n_valid_hours': np.sum(~np.isnan(hs)),
        'data_coverage': np.mean(~np.isnan(hs)),
    }
    
    return features


# ============================================================================
# Helper functions for discovering site labels
# ============================================================================

def discover_mop_site_labels(mop_range: Tuple[int, int] = (500, 800)) -> List[str]:
    """
    Generate possible site labels to try for a range of MOP IDs.
    
    CDIP site labels vary by region. For San Diego:
        - Some use D0XXX (4-digit with leading zeros)
        - Some use DXXX (3-digit)
        
    This function returns candidate labels to try.
    """
    candidates = []
    for mop_id in range(mop_range[0], mop_range[1] + 1):
        candidates.extend([
            f"D{mop_id:04d}",  # D0582
            f"D{mop_id:03d}",  # D582
            f"D{mop_id}",      # D582
        ])
    return candidates


def print_download_instructions():
    """Print instructions for downloading CDIP data."""
    instructions = """
================================================================================
CDIP Wave Data Download Instructions
================================================================================

The CDIP THREDDS server provides wave data that can be downloaded for offline use.

OPTION 1: Manual Download via Browser
--------------------------------------
1. Browse: https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/model/MOP_alongshore/catalog.html
2. Find your site (e.g., D0582 for San Diego MOP 582)
3. Click on the file (e.g., D0582_hindcast.nc)
4. Choose "NetCDF" from the download options
5. Save to your local data directory

OPTION 2: Command Line Download (wget/curl)
-------------------------------------------
# Download a single site
wget -O D0582_hindcast.nc \\
    "https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/model/MOP_alongshore/D0582_hindcast.nc"

# Download multiple sites (bash loop)
for MOP in $(seq 520 764); do
    SITE=$(printf "D%04d" $MOP)
    wget -O ${SITE}_hindcast.nc \\
        "https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/model/MOP_alongshore/${SITE}_hindcast.nc"
done

OPTION 3: Python Download Script
--------------------------------
from cdip_wave_loader import CDIPWaveLoader
loader = CDIPWaveLoader(cache_dir='data/cache/cdip')
# Data will be automatically cached on first access
waves = loader.load_mop(mop_id=582)

OPTION 4: OPeNDAP Subset (for large files)
------------------------------------------
Use OPeNDAP to download only the time range you need:

import xarray as xr
url = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore/D0582_hindcast.nc"
ds = xr.open_dataset(url, decode_times=False)

# Select only 2020-2024 data
ds_subset = ds.sel(waveTime=slice(start_idx, end_idx))
ds_subset.to_netcdf('D0582_2020-2024.nc')

FILE NAMING CONVENTION
----------------------
San Diego MOPs: D0520, D0521, ..., D0764
Orange County:  OC0001, OC0002, ...
Los Angeles:    L0001, L0002, ...
Ventura:        VE001, VE002, ...

================================================================================
"""
    print(instructions)


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CDIP Wave Data Loader")
    parser.add_argument('--mop', type=int, default=582, help='MOP ID to load')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2023-12-31', help='End date')
    parser.add_argument('--cache-dir', type=str, default=None, help='Cache directory')
    parser.add_argument('--local-dir', type=str, default=None, help='Local data directory')
    parser.add_argument('--site-label', type=str, default=None, help='Override site label')
    parser.add_argument('--download-help', action='store_true', help='Show download instructions')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.download_help:
        print_download_instructions()
        exit(0)
    
    # Initialize loader
    loader = CDIPWaveLoader(
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
    )
    
    print(f"\n{'='*60}")
    print(f"Loading CDIP MOP {args.mop} wave data")
    print(f"Time range: {args.start} to {args.end}")
    print(f"{'='*60}\n")
    
    try:
        # Load wave data
        waves = loader.load_mop(
            mop_id=args.mop,
            start_date=args.start,
            end_date=args.end,
            site_label_override=args.site_label,
        )
        
        print(f"Location: ({waves.latitude:.4f}, {waves.longitude:.4f})")
        print(f"Water depth: {waves.water_depth:.1f} m")
        print(f"Time range: {waves.time[0]} to {waves.time[-1]}")
        print(f"Number of records: {len(waves.time)}")
        print()
        
        # Print wave statistics
        print("Wave Statistics:")
        print(f"  Hs: mean={np.nanmean(waves.hs):.2f}m, max={np.nanmax(waves.hs):.2f}m")
        print(f"  Tp: mean={np.nanmean(waves.tp):.1f}s, max={np.nanmax(waves.tp):.1f}s")
        print(f"  Power: mean={np.nanmean(waves.power):.1f} kW/m")
        print()
        
        # Test tensor conversion
        print("Testing CliffCast tensor conversion...")
        features, doy = waves.to_tensor(
            history_days=90,
            reference_date=args.end,
            resample_hours=6,
        )
        
        print(f"  Features shape: {features.shape}")
        print(f"  DOY shape: {doy.shape}")
        print(f"  Expected shape for 90 days @ 6hr: (360, 4)")
        print()
        
        # Compute derived features
        print("Derived features:")
        derived = compute_derived_features(waves)
        for key, value in list(derived.items())[:10]:  # First 10
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"\nERROR: {e}\n")
        print("Try: python cdip_wave_loader.py --download-help")
        import traceback
        traceback.print_exc()
