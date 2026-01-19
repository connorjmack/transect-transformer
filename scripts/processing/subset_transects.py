#!/usr/bin/env python3
"""Subset transects from an existing NPZ cube file by MOP range.

This script filters an extracted cube NPZ file to include only transects
within a specified MOP range. Useful for breaking up a large extraction
into beach-specific files.

Usage:
    # Subset by MOP range
    python scripts/processing/subset_transects.py \
        --input data/processed/all_transects.npz \
        --output data/processed/delmar.npz \
        --mop-min 595 --mop-max 620

    # Subset by beach name
    python scripts/processing/subset_transects.py \
        --input data/processed/all_transects.npz \
        --output data/processed/delmar.npz \
        --beach delmar

    # List available transects without subsetting
    python scripts/processing/subset_transects.py \
        --input data/processed/all_transects.npz \
        --list
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logger

logger = setup_logger(__name__, level="INFO")

# Beach name to MOP range mapping (San Diego County)
BEACH_MOP_RANGES = {
    'blacks': (520, 567),
    'torrey': (567, 581),
    'delmar': (595, 620),
    'solana': (637, 666),
    'sanelijo': (683, 708),
    'encinitas': (708, 764),
}


def parse_mop_number(transect_id: str) -> Optional[int]:
    """Extract MOP number from transect ID string.

    Handles formats like:
    - "MOP 595"
    - "MOP 600_01"
    - "595"
    - 595 (int)

    Args:
        transect_id: Transect ID string or number

    Returns:
        Integer MOP number or None if parsing fails
    """
    if isinstance(transect_id, (int, float)):
        return int(transect_id)

    tid_str = str(transect_id)

    # Try "MOP XXX" or "MOP XXX_YY" format
    match = re.search(r'MOP\s*(\d+)', tid_str, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try just a number
    match = re.match(r'^(\d+)', tid_str)
    if match:
        return int(match.group(1))

    return None


def load_cube(input_path: Path) -> Dict[str, np.ndarray]:
    """Load cube NPZ file.

    Args:
        input_path: Path to NPZ file

    Returns:
        Dictionary with cube arrays
    """
    logger.info(f"Loading cube from {input_path}")
    data = np.load(input_path, allow_pickle=True)

    result = {}
    for key in data.keys():
        arr = data[key]
        if arr.dtype == object:
            result[key] = arr.tolist() if arr.ndim == 1 else arr
        else:
            result[key] = arr

    return result


def list_transects(cube: Dict) -> None:
    """Print summary of transects in cube.

    Args:
        cube: Cube data dictionary
    """
    transect_ids = cube['transect_ids']
    if isinstance(transect_ids, np.ndarray):
        transect_ids = transect_ids.tolist()

    n_transects = len(transect_ids)
    n_epochs = cube['points'].shape[1]

    print(f"\nCube contains {n_transects} transects × {n_epochs} epochs")
    print(f"Shape: {cube['points'].shape}")

    # Parse MOP numbers and find ranges
    mop_numbers = []
    for tid in transect_ids:
        mop = parse_mop_number(tid)
        if mop is not None:
            mop_numbers.append(mop)

    if mop_numbers:
        mop_numbers.sort()
        print(f"\nMOP range: {min(mop_numbers)} - {max(mop_numbers)}")

        # Show beach breakdown
        print("\nTransects per beach:")
        for beach, (mop_min, mop_max) in BEACH_MOP_RANGES.items():
            count = sum(1 for m in mop_numbers if mop_min <= m <= mop_max)
            if count > 0:
                print(f"  {beach:12s}: {count:4d} transects (MOP {mop_min}-{mop_max})")

    # Show first/last few transect IDs
    print(f"\nFirst 5 transect IDs: {transect_ids[:5]}")
    print(f"Last 5 transect IDs: {transect_ids[-5:]}")

    if 'epoch_dates' in cube:
        epoch_dates = cube['epoch_dates']
        if isinstance(epoch_dates, np.ndarray):
            epoch_dates = epoch_dates.tolist()
        print(f"\nEpochs: {epoch_dates}")


def subset_cube(
    cube: Dict,
    mop_min: int,
    mop_max: int
) -> Dict[str, np.ndarray]:
    """Subset cube to transects within MOP range.

    Args:
        cube: Full cube data dictionary
        mop_min: Minimum MOP number (inclusive)
        mop_max: Maximum MOP number (inclusive)

    Returns:
        Subset cube dictionary
    """
    transect_ids = cube['transect_ids']
    if isinstance(transect_ids, np.ndarray):
        transect_ids = transect_ids.tolist()

    # Find indices of transects in range
    keep_indices = []
    keep_ids = []

    for i, tid in enumerate(transect_ids):
        mop = parse_mop_number(tid)
        if mop is not None and mop_min <= mop <= mop_max:
            keep_indices.append(i)
            keep_ids.append(tid)

    if not keep_indices:
        logger.warning(f"No transects found in MOP range {mop_min}-{mop_max}")
        return None

    keep_indices = np.array(keep_indices)

    logger.info(f"Found {len(keep_indices)} transects in MOP range {mop_min}-{mop_max}")

    # Subset all arrays
    subset = {
        'points': cube['points'][keep_indices],
        'distances': cube['distances'][keep_indices],
        'metadata': cube['metadata'][keep_indices],
        'transect_ids': np.array(keep_ids, dtype=object),
    }

    # Copy timestamps if present (subset by transect)
    if 'timestamps' in cube:
        subset['timestamps'] = cube['timestamps'][keep_indices]

    # Copy arrays that don't need subsetting
    for key in ['epoch_names', 'epoch_dates', 'feature_names', 'metadata_names']:
        if key in cube:
            value = cube[key]
            if isinstance(value, list):
                subset[key] = np.array(value, dtype=object)
            else:
                subset[key] = value

    return subset


def save_cube(cube: Dict, output_path: Path) -> None:
    """Save cube to NPZ file.

    Args:
        cube: Cube data dictionary
        output_path: Output path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert lists to arrays for saving
    save_dict = {}
    for key, value in cube.items():
        if isinstance(value, list):
            save_dict[key] = np.array(value, dtype=object)
        else:
            save_dict[key] = value

    np.savez_compressed(output_path, **save_dict)

    n_transects, n_epochs = cube['points'].shape[:2]
    logger.info(f"Saved {n_transects} transects × {n_epochs} epochs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Subset transects from NPZ cube by MOP range."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input NPZ cube file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ file (required unless --list)",
    )

    parser.add_argument(
        "--mop-min",
        type=int,
        default=None,
        help="Minimum MOP number (inclusive)",
    )

    parser.add_argument(
        "--mop-max",
        type=int,
        default=None,
        help="Maximum MOP number (inclusive)",
    )

    parser.add_argument(
        "--beach",
        type=str,
        default=None,
        choices=list(BEACH_MOP_RANGES.keys()),
        help=f"Beach name to auto-set MOP range. Options: {', '.join(BEACH_MOP_RANGES.keys())}",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List transects in input file without subsetting",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Load cube
    cube = load_cube(args.input)

    # List mode
    if args.list:
        list_transects(cube)
        return 0

    # Apply beach preset if specified
    if args.beach:
        mop_min, mop_max = BEACH_MOP_RANGES[args.beach]
        if args.mop_min is None:
            args.mop_min = mop_min
        if args.mop_max is None:
            args.mop_max = mop_max
        logger.info(f"Beach '{args.beach}' selected: MOP range {args.mop_min}-{args.mop_max}")

    # Validate subsetting parameters
    if args.mop_min is None or args.mop_max is None:
        logger.error("Must specify --mop-min and --mop-max, or --beach")
        return 1

    if args.output is None:
        logger.error("Must specify --output for subsetting")
        return 1

    # Subset
    subset = subset_cube(cube, args.mop_min, args.mop_max)

    if subset is None:
        return 1

    # Save
    save_cube(subset, args.output)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
