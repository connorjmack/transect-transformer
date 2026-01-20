"""Event loading and coordinate conversion utilities.

Loads significant erosion events from CSV files and converts local
alongshore coordinates to MOP IDs for prioritized labeling.

Coordinate System:
- Event CSVs use 1m local transect spacing (alongshore_centroid_m, etc.)
- Our NPZ data uses 10m transect spacing (10 transects per MOP)
- Transect IDs: "MOP 595", "MOP 595_001", ..., "MOP 595_009", "MOP 596", ...
- Each MOP covers 100m of local coordinates (10 transects * 10m spacing)
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# MOP ranges for each beach (from CLAUDE.md)
# These are the canonical ranges used for filtering and subsetting
BEACH_MOP_RANGES = {
    'blacks': (520, 567),
    'torrey': (567, 581),
    'delmar': (595, 620),
    'solana': (637, 666),
    'sanelijo': (683, 708),
    'encinitas': (708, 764),
}

# Map CSV filename prefixes to beach names
CSV_BEACH_MAP = {
    'DelMar': 'delmar',
    'Torrey': 'torrey',
    'Solana': 'solana',
    'SanElijo': 'sanelijo',
    'Encinitas': 'encinitas',
    'Blacks': 'blacks',
}

# Default events directory
DEFAULT_EVENTS_DIR = "data/raw/events"


def parse_transect_id(tid: str) -> float:
    """
    Parse a transect ID string to a numeric MOP value.

    Args:
        tid: Transect ID like "MOP 595" or "MOP 595_006"

    Returns:
        Numeric MOP value (e.g., 595.0 or 595.6)

    Examples:
        "MOP 595" -> 595.0
        "MOP 595_001" -> 595.1
        "MOP 595_006" -> 595.6
        "MOP 608_006" -> 608.6
    """
    if isinstance(tid, (int, float)):
        return float(tid)

    # Parse 'MOP 595' or 'MOP 595_001' format
    parts = str(tid).replace('MOP ', '').split('_')
    base_mop = int(parts[0])

    if len(parts) > 1:
        # Sub-MOP: 'MOP 595_001' -> 595.1, 'MOP 595_006' -> 595.6
        # There are 10 sub-transects (0-9), so divide by 10
        sub_idx = int(parts[1])
        return base_mop + sub_idx / 10.0
    else:
        return float(base_mop)


def local_to_mop(local_coord_m: float, beach: str) -> float:
    """
    Convert local alongshore coordinate (1m spacing) to MOP value.

    Local coordinates start at 0 in the south and increase northward.
    Each MOP spans 100m (100 local transect lines at 1m spacing).

    Our transects are at 10m spacing, so:
    - local_coord / 100 = MOP offset from beach start
    - (local_coord % 100) / 10 = sub-transect index within MOP

    Args:
        local_coord_m: Alongshore position in meters (1m transect system)
        beach: Beach name (lowercase)

    Returns:
        MOP value (e.g., 608.6 for local=1360m in Del Mar)

    Examples (Del Mar, MOP 595-620):
        local=0 -> MOP 595.0
        local=60 -> MOP 595.6
        local=100 -> MOP 596.0
        local=1360 -> MOP 608.6
    """
    if beach not in BEACH_MOP_RANGES:
        raise ValueError(f"Unknown beach: {beach}. Valid: {list(BEACH_MOP_RANGES.keys())}")

    mop_start, _ = BEACH_MOP_RANGES[beach]

    # Convert to MOP value (100m per MOP)
    mop_offset = local_coord_m / 100.0
    return mop_start + mop_offset


def mop_to_transect_id(mop_value: float) -> str:
    """
    Convert a MOP value to a transect ID string.

    Args:
        mop_value: Numeric MOP value (e.g., 608.6)

    Returns:
        Transect ID string (e.g., "MOP 608_006")

    Examples:
        608.0 -> "MOP 608"
        608.6 -> "MOP 608_006"
        608.63 -> "MOP 608_006" (rounded to nearest 10m transect)
    """
    base_mop = int(mop_value)
    sub_idx = round((mop_value - base_mop) * 10)

    if sub_idx == 0:
        return f"MOP {base_mop}"
    elif sub_idx >= 10:
        # Overflow to next MOP
        return f"MOP {base_mop + 1}"
    else:
        return f"MOP {base_mop}_{sub_idx:03d}"


def mop_to_transect_idx(
    mop_value: float,
    transect_ids: list,
    tolerance: float = 0.15
) -> Optional[int]:
    """
    Find the transect index closest to a MOP value.

    Args:
        mop_value: Target MOP value (e.g., 608.63)
        transect_ids: List of transect IDs (e.g., ['MOP 595', 'MOP 595_001', ...])
        tolerance: Maximum MOP difference to consider a match (0.1 = 10m)

    Returns:
        Index into transect_ids, or None if no close match
    """
    # Convert all transect IDs to numeric MOP values
    mop_values = []
    for tid in transect_ids:
        try:
            mop_values.append(parse_transect_id(tid))
        except (ValueError, AttributeError):
            mop_values.append(np.nan)

    mop_array = np.array(mop_values)
    distances = np.abs(mop_array - mop_value)

    min_idx = np.nanargmin(distances)
    if distances[min_idx] <= tolerance:
        return int(min_idx)
    return None


def load_events_csv(csv_path: str) -> pd.DataFrame:
    """Load an events CSV file."""
    df = pd.read_csv(csv_path)

    # Parse dates
    for col in ['mid_date', 'start_date', 'end_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df


def load_all_sig_events(events_dir: str = DEFAULT_EVENTS_DIR) -> pd.DataFrame:
    """
    Load all significant event CSVs and combine into a single DataFrame.

    Args:
        events_dir: Directory containing event CSV files

    Returns:
        Combined DataFrame with 'beach' column and MOP columns added,
        sorted by volume descending (largest events first)
    """
    events_dir = Path(events_dir)
    all_events = []

    for csv_file in events_dir.glob("*_sig.csv"):
        # Extract beach name from filename
        prefix = csv_file.stem.replace('_events_sig', '')
        beach = CSV_BEACH_MAP.get(prefix, prefix.lower())

        df = load_events_csv(str(csv_file))
        df['beach'] = beach
        df['source_file'] = csv_file.name

        all_events.append(df)

    if not all_events:
        return pd.DataFrame()

    combined = pd.concat(all_events, ignore_index=True)

    # Add MOP columns
    combined['mop_centroid'] = combined.apply(
        lambda row: local_to_mop(row['alongshore_centroid_m'], row['beach']),
        axis=1
    )
    combined['mop_start'] = combined.apply(
        lambda row: local_to_mop(row['alongshore_start_m'], row['beach']),
        axis=1
    )
    combined['mop_end'] = combined.apply(
        lambda row: local_to_mop(row['alongshore_end_m'], row['beach']),
        axis=1
    )

    # Sort by volume descending (largest events first)
    combined = combined.sort_values('volume', ascending=False).reset_index(drop=True)

    return combined


def find_matching_epoch_pair(
    event: pd.Series,
    epoch_dates: list[str],
) -> Optional[int]:
    """
    Find the epoch pair that matches an event's date range.

    An event matches a pair if:
    1. event.start_date >= epoch[i] and event.end_date <= epoch[i+1], OR
    2. The dates are within 7 days of each other

    Args:
        event: Event row with start_date and end_date
        epoch_dates: List of epoch date strings (ISO format)

    Returns:
        Pair index (i.e., the earlier epoch index), or None if no match
    """
    event_start = event['start_date']
    event_end = event['end_date']

    # Parse epoch dates
    parsed_dates = []
    for d in epoch_dates:
        try:
            if isinstance(d, str):
                parsed_dates.append(pd.to_datetime(d[:10]))
            else:
                parsed_dates.append(pd.to_datetime(d))
        except Exception:
            parsed_dates.append(None)

    # Find epoch pair that contains the event
    for i in range(len(parsed_dates) - 1):
        if parsed_dates[i] is None or parsed_dates[i + 1] is None:
            continue

        epoch1 = parsed_dates[i]
        epoch2 = parsed_dates[i + 1]

        # Check if event falls within this epoch pair
        if epoch1.date() <= event_start.date() and epoch2.date() >= event_end.date():
            return i

        # Also check if dates are very close (within 7 days)
        if (abs((epoch1.date() - event_start.date()).days) <= 7 and
                abs((epoch2.date() - event_end.date()).days) <= 7):
            return i

    return None


def build_event_queue(
    events_df: pd.DataFrame,
    data: dict[str, Any],
    max_events: int = 2000,
) -> list[dict]:
    """
    Build a prioritized queue of events to label.

    Args:
        events_df: DataFrame with significant events (sorted by volume desc)
        data: Loaded transect NPZ data
        max_events: Maximum number of events to include

    Returns:
        List of event dicts with transect_idx, pair_idx, and event info
    """
    from apps.transect_labeler.utils.data_loader import get_transect_ids, get_epoch_dates

    transect_ids = get_transect_ids(data)
    epoch_dates = get_epoch_dates(data)

    queue = []
    seen = set()  # Track (transect_idx, pair_idx) to avoid duplicates

    for _, event in events_df.iterrows():
        if len(queue) >= max_events:
            break

        # Find matching transect
        mop_centroid = event['mop_centroid']
        transect_idx = mop_to_transect_idx(mop_centroid, transect_ids)

        if transect_idx is None:
            continue

        # Find matching epoch pair
        pair_idx = find_matching_epoch_pair(event, epoch_dates)

        if pair_idx is None:
            continue

        # Skip duplicates (keep first = largest volume)
        key = (transect_idx, pair_idx)
        if key in seen:
            continue
        seen.add(key)

        queue.append({
            'transect_idx': transect_idx,
            'pair_idx': pair_idx,
            'transect_id': transect_ids[transect_idx],
            'event': {
                'volume': float(event['volume']),
                'elevation': float(event['elevation']),
                'beach': event['beach'],
                'start_date': str(event['start_date'].date()),
                'end_date': str(event['end_date'].date()),
                'mop_centroid': float(mop_centroid),
                'mop_start': float(event['mop_start']),
                'mop_end': float(event['mop_end']),
                'width': float(event['width']),
                'height': float(event['height']),
            }
        })

    return queue


def get_events_for_transect_pair(
    events_df: pd.DataFrame,
    transect_idx: int,
    pair_idx: int,
    data: dict[str, Any],
    mop_tolerance: float = 0.15,
) -> list[dict]:
    """
    Get all events that match a specific transect-pair.

    Args:
        events_df: DataFrame with significant events
        transect_idx: Transect index
        pair_idx: Epoch pair index
        data: Loaded transect NPZ data
        mop_tolerance: MOP matching tolerance (0.1 = 10m)

    Returns:
        List of matching event dicts, sorted by volume descending
    """
    from apps.transect_labeler.utils.data_loader import get_transect_ids, get_epoch_dates

    transect_ids = get_transect_ids(data)
    epoch_dates = get_epoch_dates(data)

    if transect_idx >= len(transect_ids):
        return []

    # Get MOP for this transect
    tid = transect_ids[transect_idx]
    try:
        transect_mop = parse_transect_id(tid)
    except Exception:
        return []

    # Filter events by MOP (event spans this transect's MOP)
    matching = events_df[
        (events_df['mop_start'] - mop_tolerance <= transect_mop) &
        (events_df['mop_end'] + mop_tolerance >= transect_mop)
    ]

    # Further filter by epoch pair dates
    if pair_idx >= len(epoch_dates) - 1:
        return []

    epoch1_date = pd.to_datetime(epoch_dates[pair_idx][:10])
    epoch2_date = pd.to_datetime(epoch_dates[pair_idx + 1][:10])

    result = []
    for _, event in matching.iterrows():
        event_start = event['start_date']
        event_end = event['end_date']

        # Check if event overlaps with epoch pair
        if event_start.date() >= epoch1_date.date() and event_end.date() <= epoch2_date.date():
            result.append({
                'volume': float(event['volume']),
                'elevation': float(event['elevation']),
                'beach': event['beach'],
                'start_date': str(event['start_date'].date()),
                'end_date': str(event['end_date'].date()),
                'mop_centroid': float(event['mop_centroid']),
                'width': float(event['width']),
                'height': float(event['height']),
            })

    # Sort by volume descending
    result.sort(key=lambda x: x['volume'], reverse=True)
    return result
