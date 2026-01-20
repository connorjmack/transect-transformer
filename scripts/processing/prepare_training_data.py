#!/usr/bin/env python3
"""
Prepare training data for CliffCast model.

This script aligns transect cubes with event data and environmental features,
generating the final training tensors.

Usage:
    python scripts/processing/prepare_training_data.py \
        --cube data/processed/100_transects_with_cliffs.npz \
        --events-dir data/raw/events \
        --wave-dir data/raw/cdip \
        --atmos-dir data/processed/atmospheric \
        --output data/processed/training_data.npz

Author: CliffCast Project
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

BEACH_ORIGIN_MOP = {
    'blacks': 520,
    'torrey': 567,
    'delmar': 595,
    'solana': 637,
    'sanelijo': 683,
    'encinitas': 708,
}

BEACH_END_MOP = {
    'blacks': 567,
    'torrey': 581,
    'delmar': 620,
    'solana': 666,
    'sanelijo': 708,
    'encinitas': 764,
}

EVENT_FILENAME_PATTERNS = {
    'blacks': ['blacks_events_sig.csv', 'Blacks_events_sig.csv'],
    'torrey': ['torrey_events_sig.csv', 'Torrey_events_sig.csv'],
    'delmar': ['delmar_events_sig.csv', 'DelMar_events_sig.csv'],
    'solana': ['solana_events_sig.csv', 'Solana_events_sig.csv'],
    'sanelijo': ['sanelijo_events_sig.csv', 'SanElijo_events_sig.csv'],
    'encinitas': ['encinitas_events_sig.csv', 'Encinitas_events_sig.csv'],
}

# Model parameters
WAVE_LOOKBACK_DAYS = 90
WAVE_INTERVAL_HOURS = 6
WAVE_TIMESTEPS = (WAVE_LOOKBACK_DAYS * 24) // WAVE_INTERVAL_HOURS  # 360
WAVE_FEATURES = 4

ATMOS_LOOKBACK_DAYS = 90
ATMOS_TIMESTEPS = 90
ATMOS_FEATURES = 24

MIN_CONTEXT_EPOCHS = 3
MAX_CONTEXT_EPOCHS = 10


# =============================================================================
# Coordinate Conversion
# =============================================================================

def alongshore_to_transect_idx(alongshore_m: float, beach: str, cube: dict, transect_spacing_m: float = 10.0) -> int:
    """Convert local alongshore coordinate to transect index in cube."""
    beach_slices = cube['beach_slices'].item() if isinstance(cube['beach_slices'], np.ndarray) else cube['beach_slices']
    beach_start, beach_end = beach_slices[beach]
    n_transects = beach_end - beach_start
    transect_offset = int(round(alongshore_m / transect_spacing_m))
    transect_offset = max(0, min(n_transects - 1, transect_offset))
    return beach_start + transect_offset


def transect_idx_to_beach(transect_idx: int, cube: dict) -> str:
    """Get beach name for a transect index."""
    beach_slices = cube['beach_slices'].item() if isinstance(cube['beach_slices'], np.ndarray) else cube['beach_slices']
    for beach, (start, end) in beach_slices.items():
        if start <= transect_idx < end:
            return beach
    raise ValueError(f"Transect index {transect_idx} not in any beach range")


# =============================================================================
# Event Loading
# =============================================================================

def volume_to_class(volume: float) -> int:
    """Convert volume to event class index."""
    if volume < 10:
        return 0  # stable
    elif volume < 50:
        return 1  # minor
    elif volume < 200:
        return 2  # major
    else:
        return 3  # failure


def load_events(beach: str, events_dir: Path) -> Optional[pd.DataFrame]:
    """Load event CSV for a specific beach."""
    patterns = EVENT_FILENAME_PATTERNS.get(beach.lower(), [f'{beach}_events_sig.csv'])
    for pattern in patterns:
        path = events_dir / pattern
        if path.exists():
            df = pd.read_csv(path)
            df['mid_date'] = pd.to_datetime(df['mid_date'])
            df['start_date'] = pd.to_datetime(df['start_date'])
            df['end_date'] = pd.to_datetime(df['end_date'])
            df['event_class'] = df['volume'].apply(volume_to_class)
            df['beach'] = beach.lower()
            return df
    return None


def load_all_events(events_dir: Path) -> pd.DataFrame:
    """Load and concatenate events from all beaches."""
    beaches = ['blacks', 'torrey', 'delmar', 'solana', 'sanelijo', 'encinitas']
    dfs = []
    for beach in beaches:
        df = load_events(beach, events_dir)
        if df is not None:
            logger.info(f"  {beach}: {len(df)} events")
            dfs.append(df)
        else:
            logger.info(f"  {beach}: no events found")
    if not dfs:
        raise ValueError('No event files found')
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total: {len(combined)} events")
    return combined


# =============================================================================
# Event Alignment
# =============================================================================

def find_bracketing_epochs(event_start: datetime, event_end: datetime, epoch_dates: np.ndarray) -> Tuple[int, int]:
    """Find the epoch indices that bracket an event (before and after)."""
    epoch_dts = pd.to_datetime(epoch_dates)
    before_mask = epoch_dts <= event_start
    after_mask = epoch_dts >= event_end

    if not before_mask.any() or not after_mask.any():
        return -1, -1

    epoch_before = np.where(before_mask)[0][-1]
    epoch_after = np.where(after_mask)[0][0]

    return epoch_before, epoch_after


def align_events_to_cube(events_df: pd.DataFrame, cube: dict) -> pd.DataFrame:
    """Align events to transect indices and epoch pairs."""
    epoch_dates = cube['epoch_dates']
    coverage_mask = cube['coverage_mask']

    aligned = []
    for _, event in events_df.iterrows():
        beach = event['beach']
        transect_idx = alongshore_to_transect_idx(event['alongshore_centroid_m'], beach, cube)

        epoch_before, epoch_after = find_bracketing_epochs(event['start_date'], event['end_date'], epoch_dates)

        if epoch_before < 0 or epoch_after < 0:
            continue
        if epoch_before >= epoch_after:
            continue
        if not coverage_mask[transect_idx, epoch_before] or not coverage_mask[transect_idx, epoch_after]:
            continue

        aligned.append({
            'transect_idx': transect_idx,
            'epoch_before': epoch_before,
            'epoch_after': epoch_after,
            'volume': event['volume'],
            'vol_unc': event.get('vol_unc', 0),
            'elevation': event.get('elevation', 15),
            'width': event.get('width', 10),
            'height': event.get('height', 5),
            'event_class': event['event_class'],
            'beach': beach,
        })

    if not aligned:
        return pd.DataFrame()

    return pd.DataFrame(aligned)


def aggregate_events_by_sample(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple events per (transect, epoch_pair) into single labels."""
    if aligned_df.empty:
        return aligned_df

    grouped = aligned_df.groupby(['transect_idx', 'epoch_before', 'epoch_after']).agg({
        'volume': 'sum',
        'vol_unc': lambda x: np.sqrt((x**2).sum()),
        'elevation': 'mean',
        'width': 'sum',
        'height': 'max',
        'beach': 'first',
    }).reset_index()

    grouped['event_class'] = grouped['volume'].apply(volume_to_class)
    grouped['n_events'] = aligned_df.groupby(['transect_idx', 'epoch_before', 'epoch_after']).size().values

    return grouped


# =============================================================================
# Label Computation
# =============================================================================

def compute_risk_index(total_volume: float, cliff_height: float) -> float:
    """Compute risk index from event volume and cliff height."""
    log_vol = np.log1p(total_volume)
    height_factor = 1 + 0.05 * (cliff_height - 15)
    height_factor = np.clip(height_factor, 0.5, 2.0)
    score = log_vol * height_factor
    risk = 1 / (1 + np.exp(-0.5 * (score - 4)))
    return float(np.clip(risk, 0, 1))


def compute_collapse_labels(volume: float, horizons: List[int] = [30, 90, 180, 365]) -> np.ndarray:
    """Compute binary collapse labels for multiple time horizons."""
    # Simplified: if event occurred, mark all horizons as 1
    # In production, this would use actual collapse dates
    if volume >= 10:
        return np.ones(len(horizons), dtype=np.float32)
    return np.zeros(len(horizons), dtype=np.float32)


# =============================================================================
# Environmental Data Caching
# =============================================================================

class WaveDataCache:
    """Cache for wave data to avoid repeated file loading."""

    def __init__(self, wave_dir: Path):
        self.wave_dir = wave_dir
        self._cache: Dict[int, 'WaveData'] = {}  # mop_id -> WaveData
        self._loader = None

    def _init_loader(self):
        if self._loader is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
            from data.cdip_wave_loader import CDIPWaveLoader
            self._loader = CDIPWaveLoader(local_dir=self.wave_dir)

    def get_wave_tensor(self, mop_id: int, target_date: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Get wave features for a sample (90-day lookback before target)."""
        try:
            self._init_loader()

            # Load full wave data for this MOP if not cached
            if mop_id not in self._cache:
                wave_data = self._loader.load_mop(mop_id=mop_id)
                self._cache[mop_id] = wave_data

            # Convert to tensor for this specific date
            features, doy = self._cache[mop_id].to_tensor(history_days=WAVE_LOOKBACK_DAYS, reference_date=target_date, resample_hours=WAVE_INTERVAL_HOURS)
            return features, doy
        except Exception as e:
            # Fall back to zero tensor on error
            return np.zeros((WAVE_TIMESTEPS, WAVE_FEATURES), dtype=np.float32), np.ones(WAVE_TIMESTEPS, dtype=np.int32) * 180


class AtmosDataCache:
    """Cache for atmospheric data to avoid repeated file loading."""

    def __init__(self, atmos_dir: Path):
        self.atmos_dir = atmos_dir
        self._loader = None

    def _init_loader(self):
        if self._loader is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
            from data.atmos_loader import AtmosphericLoader
            self._loader = AtmosphericLoader(atmos_dir=self.atmos_dir, lookback_days=ATMOS_LOOKBACK_DAYS)

    def get_atmos_tensor(self, beach: str, target_date: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Get atmospheric features for a sample (90-day lookback before target)."""
        try:
            self._init_loader()
            features, doy = self._loader.get_atmos_for_scan(beach, target_date)

            # Pad to ATMOS_FEATURES if needed
            if features.shape[1] < ATMOS_FEATURES:
                padded = np.zeros((ATMOS_TIMESTEPS, ATMOS_FEATURES), dtype=np.float32)
                padded[:, :features.shape[1]] = features
                features = padded

            return features, doy
        except Exception as e:
            return np.zeros((ATMOS_TIMESTEPS, ATMOS_FEATURES), dtype=np.float32), np.ones(ATMOS_TIMESTEPS, dtype=np.int32) * 180


# =============================================================================
# Sample Generation
# =============================================================================

def generate_training_samples(cube: dict, aggregated_events: pd.DataFrame, wave_dir: Path, atmos_dir: Path, min_context: int = MIN_CONTEXT_EPOCHS) -> dict:
    """Generate all training samples from cube and aligned events."""
    points = cube['points']
    metadata = cube['metadata']
    distances = cube['distances']
    coverage_mask = cube['coverage_mask']
    epoch_dates = cube['epoch_dates']
    mop_ids = cube['mop_ids']

    n_transects, n_epochs, n_points, n_features = points.shape

    # Create event lookup: (transect_idx, epoch_after) -> event_row
    event_lookup = {}
    for _, row in aggregated_events.iterrows():
        key = (int(row['transect_idx']), int(row['epoch_after']))
        if key not in event_lookup:
            event_lookup[key] = row
        else:
            # Aggregate if multiple (shouldn't happen after aggregation)
            event_lookup[key]['volume'] += row['volume']

    # Initialize cached data loaders
    wave_cache = WaveDataCache(wave_dir)
    atmos_cache = AtmosDataCache(atmos_dir)

    # Storage
    samples = {
        'point_features': [],
        'metadata': [],
        'distances': [],
        'context_mask': [],
        'wave_features': [],
        'wave_doy': [],
        'atmos_features': [],
        'atmos_doy': [],
        'total_volume': [],
        'event_class': [],
        'risk_index': [],
        'collapse_labels': [],
        'label_source': [],  # 0=derived, 1=observed
        'confidence': [],
        'transect_idx': [],
        'target_epoch': [],
        'beach': [],
        'mop_id': [],
    }

    logger.info(f"Generating samples from {n_transects} transects...")

    for t_idx in tqdm(range(n_transects), desc="Transects"):
        # Find valid epochs for this transect
        valid_epochs = np.where(coverage_mask[t_idx])[0]

        if len(valid_epochs) < min_context + 1:
            continue

        beach = transect_idx_to_beach(t_idx, cube)
        mop_id = int(mop_ids[t_idx])

        # Sliding window: use valid_epochs[:-1] as context, valid_epochs[1:] as targets
        for i in range(len(valid_epochs) - 1):
            # Context epochs (up to MAX_CONTEXT_EPOCHS before target)
            context_end = i + 1
            context_start = max(0, context_end - MAX_CONTEXT_EPOCHS)
            context_epochs = valid_epochs[context_start:context_end]

            if len(context_epochs) < min_context:
                continue

            target_epoch_idx = valid_epochs[i + 1]
            target_date_str = epoch_dates[target_epoch_idx]
            target_date = pd.to_datetime(target_date_str).to_pydatetime()

            # Get context data
            n_ctx = len(context_epochs)
            ctx_points = np.zeros((MAX_CONTEXT_EPOCHS, n_points, n_features), dtype=np.float32)
            ctx_meta = np.zeros((MAX_CONTEXT_EPOCHS, metadata.shape[2]), dtype=np.float32)
            ctx_dist = np.zeros((MAX_CONTEXT_EPOCHS, n_points), dtype=np.float32)
            ctx_mask = np.zeros(MAX_CONTEXT_EPOCHS, dtype=bool)

            # Fill from end (most recent first is at the end)
            for j, ep in enumerate(context_epochs):
                idx = MAX_CONTEXT_EPOCHS - n_ctx + j
                ctx_points[idx] = points[t_idx, ep]
                ctx_meta[idx] = metadata[t_idx, ep]
                ctx_dist[idx] = distances[t_idx, ep]
                ctx_mask[idx] = True

            # Get labels
            event_key = (t_idx, target_epoch_idx)
            if event_key in event_lookup:
                # Observed event
                event = event_lookup[event_key]
                volume = float(event['volume'])
                event_class = int(event['event_class'])
                label_source = 1
                confidence = 1.0
            else:
                # Derive from transect changes (simplified: assume stable)
                volume = 0.0
                event_class = 0
                label_source = 0
                confidence = 0.5

            # Compute derived labels
            cliff_height = float(ctx_meta[ctx_mask][-1, 0]) if ctx_mask.any() else 15.0
            risk = compute_risk_index(volume, cliff_height)
            collapse = compute_collapse_labels(volume)

            # Load environmental data (using cached loaders)
            wave_feat, wave_doy = wave_cache.get_wave_tensor(mop_id, target_date)
            atmos_feat, atmos_doy = atmos_cache.get_atmos_tensor(beach, target_date)

            # Store sample
            samples['point_features'].append(ctx_points)
            samples['metadata'].append(ctx_meta)
            samples['distances'].append(ctx_dist)
            samples['context_mask'].append(ctx_mask)
            samples['wave_features'].append(wave_feat)
            samples['wave_doy'].append(wave_doy)
            samples['atmos_features'].append(atmos_feat)
            samples['atmos_doy'].append(atmos_doy)
            samples['total_volume'].append(volume)
            samples['event_class'].append(event_class)
            samples['risk_index'].append(risk)
            samples['collapse_labels'].append(collapse)
            samples['label_source'].append(label_source)
            samples['confidence'].append(confidence)
            samples['transect_idx'].append(t_idx)
            samples['target_epoch'].append(target_epoch_idx)
            samples['beach'].append(beach)
            samples['mop_id'].append(mop_id)

    # Convert to numpy arrays
    for key in samples:
        if key in ['beach']:
            samples[key] = np.array(samples[key], dtype=object)
        elif key in ['event_class', 'label_source', 'transect_idx', 'target_epoch', 'mop_id']:
            samples[key] = np.array(samples[key], dtype=np.int32)
        else:
            samples[key] = np.array(samples[key], dtype=np.float32)

    return samples


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare CliffCast training data")
    parser.add_argument('--cube', type=str, required=True, help='Path to transect cube NPZ file')
    parser.add_argument('--events-dir', type=str, default='data/raw/events', help='Directory containing event CSVs')
    parser.add_argument('--wave-dir', type=str, default='data/raw/cdip', help='Directory containing CDIP wave NetCDF files')
    parser.add_argument('--atmos-dir', type=str, default='data/processed/atmospheric', help='Directory containing atmospheric parquet files')
    parser.add_argument('--output', type=str, default='data/processed/training_data.npz', help='Output path for training data')
    parser.add_argument('--min-context', type=int, default=MIN_CONTEXT_EPOCHS, help=f'Minimum context epochs (default: {MIN_CONTEXT_EPOCHS})')
    args = parser.parse_args()

    cube_path = Path(args.cube)
    events_dir = Path(args.events_dir)
    wave_dir = Path(args.wave_dir)
    atmos_dir = Path(args.atmos_dir)
    output_path = Path(args.output)

    # Load cube
    logger.info(f"Loading cube from {cube_path}")
    cube = dict(np.load(cube_path, allow_pickle=True))
    logger.info(f"  Shape: {cube['points'].shape}")
    logger.info(f"  Coverage: {cube['coverage_mask'].sum()} / {cube['coverage_mask'].size} ({100*cube['coverage_mask'].mean():.1f}%)")

    # Load events
    logger.info(f"\nLoading events from {events_dir}")
    events_df = load_all_events(events_dir)

    # Align events to cube
    logger.info("\nAligning events to cube...")
    aligned = align_events_to_cube(events_df, cube)
    logger.info(f"  Aligned {len(aligned)} event instances")

    # Aggregate by sample
    aggregated = aggregate_events_by_sample(aligned)
    logger.info(f"  Aggregated to {len(aggregated)} unique (transect, epoch_pair) samples")

    # Generate training samples
    logger.info("\nGenerating training samples...")
    samples = generate_training_samples(cube, aggregated, wave_dir, atmos_dir, min_context=args.min_context)

    n_samples = len(samples['total_volume'])
    n_observed = (samples['label_source'] == 1).sum()
    n_derived = (samples['label_source'] == 0).sum()

    logger.info(f"\nGenerated {n_samples} training samples:")
    logger.info(f"  Observed events: {n_observed} ({100*n_observed/n_samples:.1f}%)")
    logger.info(f"  Derived (stable): {n_derived} ({100*n_derived/n_samples:.1f}%)")

    # Class distribution
    for cls in range(4):
        count = (samples['event_class'] == cls).sum()
        logger.info(f"  Class {cls}: {count} ({100*count/n_samples:.1f}%)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nSaving to {output_path}")
    np.savez_compressed(output_path, **samples)

    # Report shapes
    logger.info("\nOutput shapes:")
    for key, arr in samples.items():
        logger.info(f"  {key}: {arr.shape} ({arr.dtype})")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
