"""Shared helper functions for transect viewer components."""

from typing import Any, List, Optional, Union
import numpy as np

from apps.transect_viewer import config


def safe_date_label(dates: Optional[List[str]], idx: int, fallback_prefix: str = "Epoch") -> str:
    """
    Safely get date label with bounds and type checking.

    Args:
        dates: List of date strings (e.g., ['2018-01-01', '2019-01-01'])
        idx: Index to access
        fallback_prefix: Prefix for fallback label if date not available

    Returns:
        Date string (first 10 chars) or fallback label like "Epoch 0"
    """
    if dates and idx < len(dates) and dates[idx]:
        d = dates[idx]
        if isinstance(d, str) and len(d) >= 10:
            return d[:10]
        elif isinstance(d, str):
            return d
    return f"{fallback_prefix} {idx}"


def safe_epoch_option(dates: Optional[List[str]], idx: int) -> str:
    """
    Create a safe epoch option string for selectbox display.

    Args:
        dates: List of date strings
        idx: Epoch index

    Returns:
        String like "0: 2018-01-01" or "Epoch 0"
    """
    # Check if we have a valid date for this index
    if dates and idx < len(dates) and dates[idx]:
        d = dates[idx]
        if isinstance(d, str) and len(d) >= 10:
            return f"{idx}: {d[:10]}"
        elif isinstance(d, str) and d:
            return f"{idx}: {d}"
    return f"Epoch {idx}"


def safe_metadata_value(
    metadata: Union[np.ndarray, List],
    idx: int,
    decimals: int = 2,
    suffix: str = ""
) -> str:
    """
    Safely access metadata value with bounds checking and NaN handling.

    Args:
        metadata: Metadata array or list
        idx: Index to access
        decimals: Number of decimal places for formatting
        suffix: Suffix to append (e.g., " m" for meters)

    Returns:
        Formatted value string or "N/A"
    """
    n_meta = len(metadata) if hasattr(metadata, '__len__') else 0
    if idx < n_meta:
        return config.format_value(metadata[idx], decimals, suffix)
    return "N/A"


def safe_metadata_format(
    metadata: Union[np.ndarray, List],
    idx: int,
    fmt: str = ".2f"
) -> str:
    """
    Safely format metadata value with custom format string.

    Args:
        metadata: Metadata array or list
        idx: Index to access
        fmt: Format string (e.g., ".2f", ".1f")

    Returns:
        Formatted value string or "N/A"
    """
    n_meta = len(metadata) if hasattr(metadata, '__len__') else 0
    if idx < n_meta:
        val = metadata[idx]
        if not np.isnan(val):
            return f"{val:{fmt}}"
    return "N/A"
