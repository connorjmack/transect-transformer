"""Date parsing utilities for LAS source filenames."""

import re
from collections import Counter
from datetime import datetime
from typing import Optional


def parse_date_from_las_source(las_source: str) -> Optional[datetime]:
    """
    Extract date from LAS filename.

    Expected format: YYYYMMDD_... (e.g., '20251105_00589_00639_1447_DelMar_...')

    Args:
        las_source: LAS filename string

    Returns:
        datetime object or None if parsing fails
    """
    if not las_source:
        return None

    # Try YYYYMMDD prefix pattern
    match = re.match(r'^(\d{8})_', las_source)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d')
        except ValueError:
            pass

    # Try YYYY-MM-DD pattern anywhere in string
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', las_source)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    # Try YYYYMMDD pattern anywhere in string
    match = re.search(r'(\d{4})(\d{2})(\d{2})', las_source)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    return None


def infer_epoch_date(las_sources: list[str]) -> Optional[datetime]:
    """
    Infer the epoch date from a list of LAS source filenames.

    Uses the most common date found across all sources.

    Args:
        las_sources: List of LAS filenames

    Returns:
        Most common datetime or None if no dates found
    """
    if not las_sources:
        return None

    dates = []
    for source in las_sources:
        date = parse_date_from_las_source(source)
        if date:
            dates.append(date)

    if not dates:
        return None

    # Return most common date
    date_counts = Counter(dates)
    return date_counts.most_common(1)[0][0]


def format_date_for_display(date: Optional[datetime]) -> str:
    """Format date for display in UI."""
    if date is None:
        return "Unknown"
    return date.strftime('%Y-%m-%d')


def get_date_range(epochs: dict[str, dict]) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Get the date range across multiple epochs.

    Args:
        epochs: Dictionary mapping date strings to epoch data

    Returns:
        Tuple of (earliest_date, latest_date)
    """
    dates = []
    for date_str in epochs.keys():
        try:
            dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
        except ValueError:
            pass

    if not dates:
        return None, None

    return min(dates), max(dates)
