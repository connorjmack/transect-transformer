"""Forcing timeseries view for wave, rain, and cliff profiles."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from apps.transect_viewer import config
from apps.transect_viewer.utils.data_loader import (
    get_all_transect_ids,
    get_cube_dimensions,
    get_epoch_dates,
    get_transect_by_id,
    is_cube_format,
)
from src.data.atmos_loader import AtmosphericLoader, get_beach_for_mop
from src.data.cdip_wave_loader import CDIPWaveLoader


@st.cache_resource
def _get_wave_loader(data_dir: str) -> CDIPWaveLoader:
    """Create a cached CDIP loader for wave data."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Wave data directory not found: {path}")
    return CDIPWaveLoader(local_dir=path)


@st.cache_resource
def _get_atmos_loader(data_dir: str) -> AtmosphericLoader:
    """Create a cached atmospheric loader for precipitation features."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Atmospheric data directory not found: {path}")
    return AtmosphericLoader(path)


@st.cache_data(show_spinner=False)
def _load_wave_dataframe(
    mop_id: int,
    data_dir: str,
) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load full wave time series for a MOP and resample to daily averages."""
    try:
        loader = _get_wave_loader(data_dir)
    except FileNotFoundError:
        return None, None

    try:
        wave = loader.load_mop(mop_id=mop_id)
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None

    df = pd.DataFrame({
        'time': pd.to_datetime(wave.time),
        'hs': wave.hs,
        'tp': wave.tp,
        'dp': wave.dp,
        'power': wave.power,
    }).dropna()

    # Crop to recent years for clarity
    cutoff = pd.Timestamp('2017-01-01')
    df = df[df['time'] >= cutoff]

    # Downsample to daily means for cleaner visualization
    df = (
        df.set_index('time')
        .resample('D')
        .mean()
        .reset_index()
    )

    meta = {
        'latitude': wave.latitude,
        'longitude': wave.longitude,
        'water_depth': wave.water_depth,
    }
    return df, meta


@st.cache_data(show_spinner=False)
def _load_atmos_dataframe(
    beach: str,
    data_dir: str,
) -> Optional[pd.DataFrame]:
    """Load full atmospheric time series for a beach."""
    try:
        loader = _get_atmos_loader(data_dir)
    except FileNotFoundError:
        return None
    except Exception:
        return None

    try:
        df = loader._load_beach_data(beach)  # pylint: disable=protected-access
    except FileNotFoundError:
        return None
    except Exception:
        return None

    return df.reset_index().rename(columns={'index': 'date'})


def render_forcing_timeseries():
    """Render combined forcing and cliff profile timelines."""
    if st.session_state.data is None:
        st.warning("No data loaded")
        return

    data = st.session_state.data

    if not is_cube_format(data):
        st.warning("Forcing view requires cube format data with epoch dates.")
        return

    dims = get_cube_dimensions(data)
    epoch_dates = get_epoch_dates(data)
    transect_ids = get_all_transect_ids(data)
    feature_names = data.get('feature_names', [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    st.header("Forcing Timeseries")
    st.caption("Wave and rain timelines with LiDAR scan dates, plus a cliff profile slider.")

    # Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        transect_id = st.selectbox(
            "Transect",
            transect_ids,
            index=transect_ids.index(st.session_state.selected_transect_id)
            if st.session_state.selected_transect_id in transect_ids
            else 0,
            key="forcing_transect_select",
        )
        st.session_state.selected_transect_id = transect_id

    mop_id = _get_mop_id_from_data(data, transect_id)
    beach = _safe_beach_lookup(mop_id)

    with col2:
        st.markdown("**Data Sources**")
        st.write(f"MOP: `{mop_id if mop_id is not None else 'unknown'}`")
        st.write(f"Beach: `{beach if beach else 'n/a'}`")
        st.write(f"Epochs: {dims['n_epochs']}")

    wave_df, wave_meta = _load_wave_dataframe(
        mop_id=mop_id,
        data_dir=config.WAVE_DATA_DIR,
    ) if mop_id is not None else (None, None)

    atmos_df = _load_atmos_dataframe(
        beach=beach,
        data_dir=config.ATMOS_DATA_DIR,
    ) if beach else None

    _render_availability(wave_df, atmos_df, dims)

    selected_epoch_idx, selected_epoch_ts, selected_label = _select_epoch(epoch_dates)

    # Cliff profiles near the slider
    _render_cliff_slider(
        data,
        transect_id,
        feature_names,
        epoch_dates,
        selected_epoch_idx,
        selected_label,
    )

    st.markdown("---")
    _render_wave_section(wave_df, epoch_dates, wave_meta, selected_epoch_ts)
    _render_rain_section(atmos_df, epoch_dates, beach, selected_epoch_ts)


def _render_availability(
    wave_df: Optional[pd.DataFrame],
    atmos_df: Optional[pd.DataFrame],
    dims: dict,
):
    """Show quick availability metrics for forcing data."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Wave samples",
            f"{len(wave_df):,}" if wave_df is not None else "0",
            help=f"Pulled from {config.WAVE_DATA_DIR}",
        )
    with col2:
        st.metric(
            "Rain days",
            f"{len(atmos_df):,}" if atmos_df is not None else "0",
            help=f"Pulled from {config.ATMOS_DATA_DIR}",
        )
    with col3:
        st.metric("Cliff epochs", dims['n_epochs'])


def _select_epoch(epoch_dates: List[str]) -> Tuple[int, Optional[pd.Timestamp], str]:
    """Select epoch via slider and return index, timestamp, and label."""
    if not epoch_dates:
        return 0, None, "Epoch 0"

    options = list(range(len(epoch_dates)))
    labels = [
        f"{idx}: {epoch_dates[idx][:10]}" if epoch_dates else f"Epoch {idx}"
        for idx in options
    ]
    default_epoch = st.session_state.get('forcing_epoch_idx', options[-1])
    if default_epoch not in options:
        default_epoch = options[-1]

    selected_epoch = st.select_slider(
        "LiDAR epoch",
        options=options,
        value=default_epoch,
        format_func=lambda idx: labels[idx],
        key="forcing_epoch_slider",
    )
    st.session_state.forcing_epoch_idx = selected_epoch

    ts = None
    try:
        ts = pd.to_datetime(epoch_dates[selected_epoch])
    except Exception:
        ts = None

    return selected_epoch, ts, labels[selected_epoch]


def _render_wave_section(
    df: Optional[pd.DataFrame],
    epoch_dates: List[str],
    meta: Optional[dict],
    selected_date: Optional[pd.Timestamp],
):
    """Plot wave forcing timelines."""
    st.subheader("Wave parameters")
    if df is None or df.empty:
        st.info(f"No wave data found in {config.WAVE_DATA_DIR} for this transect.")
        return

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "Significant wave height (m)",
            "Peak period (s)",
            "Peak direction (deg)",
            "Wave power (kW/m)",
        ],
    )

    fig.add_trace(
        go.Scatter(x=df['time'], y=df['hs'], mode='lines', name='Hs', line=dict(color='#1f77b4')),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['tp'], mode='lines', name='Tp', line=dict(color='#ff7f0e')),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['dp'], mode='lines', name='Dp', line=dict(color='#2ca02c')),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['power'], mode='lines', name='Power', line=dict(color='#d62728')),
        row=4, col=1,
    )

    _add_epoch_lines(fig, epoch_dates)
    _add_selected_line(fig, selected_date)

    fig.update_layout(
        height=800,
        showlegend=False,
    )
    fig.update_yaxes(title_text="m", row=1, col=1)
    fig.update_yaxes(title_text="s", row=2, col=1)
    fig.update_yaxes(title_text="deg", row=3, col=1)
    fig.update_yaxes(title_text="kW/m", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

    if meta:
        st.caption(f"Wave node depth {meta['water_depth']:.1f} m at ({meta['latitude']:.3f}, {meta['longitude']:.3f})")


def _render_rain_section(
    df: Optional[pd.DataFrame],
    epoch_dates: List[str],
    beach: Optional[str],
    selected_date: Optional[pd.Timestamp],
):
    """Plot precipitation-related timelines."""
    st.subheader("Rain parameters")
    if df is None or df.empty:
        missing = f" for {beach}" if beach else ""
        st.info(f"No atmospheric data found in {config.ATMOS_DATA_DIR}{missing}.")
        return

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Daily precipitation (mm)",
            "7 / 30 day accumulation (mm)",
            "90 day accumulation + API",
        ],
    )

    fig.add_trace(
        go.Bar(x=df['date'], y=df['precip_mm'], name='Daily precip', marker_color='#1f77b4'),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['precip_7d'], name='Precip 7d', line=dict(color='#ff7f0e')),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['precip_30d'], name='Precip 30d', line=dict(color='#2ca02c')),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['precip_90d'], name='Precip 90d', line=dict(color='#9467bd')),
        row=3, col=1,
    )
    if 'api' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['api'], name='API', line=dict(color='#e377c2')),
            row=3, col=1,
        )

    _add_epoch_lines(fig, epoch_dates)
    _add_selected_line(fig, selected_date)

    fig.update_layout(
        height=700,
        showlegend=False,
    )
    fig.update_yaxes(title_text="mm", row=1, col=1)
    fig.update_yaxes(title_text="mm", row=2, col=1)
    fig.update_yaxes(title_text="mm", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_cliff_slider(
    data: dict,
    transect_id: int,
    feature_names: List[str],
    epoch_dates: List[str],
    selected_epoch: int,
    selected_label: str,
):
    """Render cliff profile slider for elevation over time with toe/top markers."""
    st.subheader("Cliff profiles over time")

    # Compute fixed x-axis range across all epochs
    x_range = None
    try:
        all_epochs_transect = get_transect_by_id(data, transect_id)
        distances_all = all_epochs_transect['distances']
        if isinstance(distances_all, np.ndarray):
            valid = distances_all[~np.isnan(distances_all)]
            if valid.size > 0:
                x_min, x_max = float(valid.min()), float(valid.max())
                x_range = (x_min, x_max)
    except Exception:
        x_range = None

    elevation_idx = _get_feature_index(feature_names, 'elevation_m')
    if elevation_idx is None:
        st.error("elevation_m feature not found in dataset.")
        return

    try:
        transect = get_transect_by_id(data, transect_id, epoch_idx=selected_epoch)
    except ValueError as exc:
        st.error(str(exc))
        return

    distances = transect['distances']
    values = transect['points'][:, elevation_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distances,
            y=values,
            mode='lines',
            line=dict(color='#1f77b4', width=3),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)',
            name='Elevation',
        )
    )

    # Add cliff toe and top markers if available
    toe_dist, top_dist, toe_elev, top_elev = _get_cliff_positions_for_plot(
        data, transect_id, selected_epoch, distances, values
    )

    if toe_dist is not None and toe_elev is not None:
        fig.add_trace(
            go.Scatter(
                x=[toe_dist],
                y=[toe_elev],
                mode='markers',
                marker=dict(color='#e74c3c', size=14, symbol='triangle-up'),
                name='Cliff Toe',
                hovertemplate='Toe: %{x:.1f}m, %{y:.1f}m<extra></extra>',
            )
        )

    if top_dist is not None and top_elev is not None:
        fig.add_trace(
            go.Scatter(
                x=[top_dist],
                y=[top_elev],
                mode='markers',
                marker=dict(color='#27ae60', size=14, symbol='triangle-down'),
                name='Cliff Top',
                hovertemplate='Top: %{x:.1f}m, %{y:.1f}m<extra></extra>',
            )
        )

    fig.update_layout(
        height=450,
        title=f"Elevation profile - {selected_label}",
        xaxis_title="Distance from toe (m)",
        yaxis_title="Elevation (m)",
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    if x_range:
        fig.update_xaxes(range=x_range)
    st.plotly_chart(fig, use_container_width=True)

    # Show toe/top info below the plot
    if toe_dist is not None or top_dist is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            if toe_dist is not None and toe_elev is not None:
                st.metric("Cliff Toe", f"{toe_dist:.1f}m @ {toe_elev:.1f}m elev")
            else:
                st.metric("Cliff Toe", "Not detected")
        with col2:
            if top_dist is not None and top_elev is not None:
                st.metric("Cliff Top", f"{top_dist:.1f}m @ {top_elev:.1f}m elev")
            else:
                st.metric("Cliff Top", "Not detected")
        with col3:
            if toe_elev is not None and top_elev is not None:
                cliff_height = top_elev - toe_elev
                st.metric("Cliff Height", f"{cliff_height:.1f}m")
            else:
                st.metric("Cliff Height", "N/A")


def _parse_mop_id(transect_id) -> Optional[int]:
    """Extract MOP id from transect identifier.

    Handles formats like:
    - 520 (int)
    - 'MOP 520' (string with MOP prefix)
    - 'MOP 520_001' (string with MOP prefix and sub-index)
    """
    if transect_id is None:
        return None
    if isinstance(transect_id, (int, np.integer)):
        return int(transect_id)
    if isinstance(transect_id, str):
        # Remove 'MOP ' prefix if present
        cleaned = transect_id.replace('MOP ', '').replace('mop ', '')
        # Take only the part before underscore (e.g., '520' from '520_001')
        base_id = cleaned.split('_')[0]
        # Extract digits from the base ID
        digits = ''.join(ch for ch in base_id if ch.isdigit())
        if digits:
            try:
                return int(digits)
            except ValueError:
                return None
    return None


def _get_mop_id_from_data(data: dict, transect_id) -> Optional[int]:
    """Get MOP ID from cube data's mop_ids field if available."""
    # If mop_ids field exists, use it directly
    if 'mop_ids' in data:
        transect_ids = data.get('transect_ids', [])
        if isinstance(transect_ids, np.ndarray):
            transect_ids = transect_ids.tolist()

        mop_ids = data['mop_ids']

        try:
            idx = transect_ids.index(transect_id)
            return int(mop_ids[idx])
        except (ValueError, IndexError):
            pass

    # Fallback to parsing from transect_id string
    return _parse_mop_id(transect_id)


def _safe_beach_lookup(mop_id: Optional[int]) -> Optional[str]:
    """Resolve beach name from MOP id, ignoring errors."""
    if mop_id is None:
        return None
    try:
        return get_beach_for_mop(mop_id)
    except Exception:
        return None


def _add_epoch_lines(fig: go.Figure, epoch_dates: List[str]):
    """Add vertical dashed lines for LiDAR scan dates."""
    scan_dates = _parse_epoch_timestamps(epoch_dates)
    for dt in scan_dates:
        fig.add_vline(
            x=dt,
            line_width=1,
            line_dash="dash",
            line_color="#555",
            opacity=0.6,
        )


def _add_selected_line(fig: go.Figure, selected_date: Optional[pd.Timestamp]):
    """Add highlighted line for currently selected epoch."""
    if selected_date is None:
        return
    fig.add_vline(
        x=selected_date,
        line_width=2,
        line_dash="solid",
        line_color="#e74c3c",
        opacity=0.9,
    )


def _parse_epoch_timestamps(epoch_dates: List[str]) -> List[pd.Timestamp]:
    """Convert epoch date strings to pandas Timestamps."""
    parsed = []
    for date in epoch_dates or []:
        try:
            parsed.append(pd.to_datetime(date))
        except Exception:
            continue
    return parsed


def _get_feature_index(feature_names: List[str], target: str) -> Optional[int]:
    """Find feature index by name."""
    if not feature_names:
        return None
    try:
        return feature_names.index(target)
    except ValueError:
        return None


def _get_cliff_positions_for_plot(
    data: dict,
    transect_id,
    epoch_idx: int,
    distances: np.ndarray,
    elevations: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Get cliff toe and top positions for plotting.

    Returns:
        Tuple of (toe_distance, top_distance, toe_elevation, top_elevation)
        Any value may be None if not available.
    """
    toe_dist = None
    top_dist = None
    toe_elev = None
    top_elev = None

    # Get transect index
    transect_ids = data.get('transect_ids', [])
    if isinstance(transect_ids, np.ndarray):
        transect_ids = transect_ids.tolist()

    try:
        transect_idx = transect_ids.index(transect_id)
    except (ValueError, TypeError):
        return toe_dist, top_dist, toe_elev, top_elev

    # Try toe_relative_m / top_relative_m format (processed data)
    if 'toe_relative_m' in data and 'top_relative_m' in data:
        toe_arr = data['toe_relative_m']
        top_arr = data['top_relative_m']

        # Handle 2D (transects x epochs) or 1D (transects only) arrays
        if toe_arr.ndim == 2:
            toe_dist = float(toe_arr[transect_idx, epoch_idx])
            top_dist = float(top_arr[transect_idx, epoch_idx])
        else:
            toe_dist = float(toe_arr[transect_idx])
            top_dist = float(top_arr[transect_idx])

        # Check for NaN
        if np.isnan(toe_dist):
            toe_dist = None
        if np.isnan(top_dist):
            top_dist = None

    # Try toe_distances / top_distances format (raw extraction data)
    elif 'toe_distances' in data and 'top_distances' in data:
        toe_arr = data['toe_distances']
        top_arr = data['top_distances']
        has_cliff = data.get('has_cliff')

        if has_cliff is not None:
            if has_cliff.ndim == 2:
                if not has_cliff[transect_idx, epoch_idx]:
                    return toe_dist, top_dist, toe_elev, top_elev
            elif not has_cliff[transect_idx]:
                return toe_dist, top_dist, toe_elev, top_elev

        if toe_arr.ndim == 2:
            toe_dist = float(toe_arr[transect_idx, epoch_idx])
            top_dist = float(top_arr[transect_idx, epoch_idx])
        else:
            toe_dist = float(toe_arr[transect_idx])
            top_dist = float(top_arr[transect_idx])

    # Interpolate elevations at toe/top distances
    if toe_dist is not None and not np.isnan(toe_dist):
        toe_elev = _interpolate_elevation(distances, elevations, toe_dist)

    if top_dist is not None and not np.isnan(top_dist):
        top_elev = _interpolate_elevation(distances, elevations, top_dist)

    return toe_dist, top_dist, toe_elev, top_elev


def _interpolate_elevation(
    distances: np.ndarray,
    elevations: np.ndarray,
    target_dist: float,
) -> Optional[float]:
    """Interpolate elevation at a given distance."""
    # Remove NaN values
    valid_mask = ~np.isnan(distances) & ~np.isnan(elevations)
    if not valid_mask.any():
        return None

    valid_dist = distances[valid_mask]
    valid_elev = elevations[valid_mask]

    # Check if target is in range
    if target_dist < valid_dist.min() or target_dist > valid_dist.max():
        # Use nearest point if out of range
        idx = np.argmin(np.abs(valid_dist - target_dist))
        return float(valid_elev[idx])

    # Linear interpolation
    return float(np.interp(target_dist, valid_dist, valid_elev))
