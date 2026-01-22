"""Temporal slider view for single transect time-series analysis."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from apps.transect_viewer import config
from apps.transect_viewer.utils.data_loader import (
    get_transect_by_id,
    get_all_transect_ids,
    get_cube_dimensions,
    get_epoch_dates,
    is_cube_format,
    has_cliff_data,
    get_cliff_positions_by_id,
)
from apps.transect_viewer.utils.helpers import safe_date_label, safe_metadata_value


def _get_valid_epochs_for_transect(data: dict, transect_id: int) -> list:
    """
    Get list of epoch indices that have valid data for a given transect.

    Returns:
        List of epoch indices where data is present (not NaN)
    """
    transect_ids = data['transect_ids']
    if isinstance(transect_ids, list):
        transect_ids = np.array(transect_ids)

    idx = np.where(transect_ids == transect_id)[0]
    if len(idx) == 0:
        return []
    idx = idx[0]

    points = data['points']
    # Check for NaN at first point of each epoch for this transect
    valid_epochs = []
    for epoch_idx in range(points.shape[1]):
        if not np.isnan(points[idx, epoch_idx, 0, 0]):
            valid_epochs.append(epoch_idx)

    return valid_epochs


def render_temporal_slider():
    """Render the temporal slider view for single transect analysis."""
    if st.session_state.data is None:
        st.warning("No data loaded")
        return

    data = st.session_state.data
    is_cube = is_cube_format(data)
    dims = get_cube_dimensions(data)

    st.header("Temporal Slider")

    if not is_cube:
        st.warning("Temporal slider requires cube format data.")
        st.info("Load a cube format NPZ file with multiple temporal epochs.")
        return

    if dims['n_epochs'] < 2:
        st.warning("Temporal slider requires at least 2 epochs.")
        return

    epoch_dates = get_epoch_dates(data)
    transect_ids = get_all_transect_ids(data)

    # Controls row
    col1, col2 = st.columns([2, 1])

    with col1:
        transect_id = st.selectbox(
            "Select Transect",
            transect_ids,
            index=transect_ids.index(st.session_state.selected_transect_id)
            if st.session_state.selected_transect_id in transect_ids else 0,
            key="temporal_slider_transect",
        )
        st.session_state.selected_transect_id = transect_id

    feature_names = data.get('feature_names', [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    with col2:
        selected_feature = st.selectbox(
            "Feature",
            feature_names,
            index=feature_names.index('elevation_m') if 'elevation_m' in feature_names else 0,
            key="temporal_slider_feature",
        )

    st.markdown("---")

    # Get valid epochs for this transect (epochs with data)
    valid_epochs = _get_valid_epochs_for_transect(data, transect_id)

    if len(valid_epochs) == 0:
        st.warning(f"No data available for transect {transect_id}")
        return

    if len(valid_epochs) == 1:
        st.info(f"Only one epoch with data for transect {transect_id}")
        slider_position = 0
        actual_epoch_idx = valid_epochs[0]
    else:
        # Create labels only for valid epochs with safe date slicing
        valid_epoch_labels = [safe_date_label(epoch_dates, i) for i in valid_epochs]

        # Get previous slider position, clamped to valid range
        prev_position = st.session_state.get('slider_epoch_idx', 0)
        initial_position = min(prev_position, len(valid_epochs) - 1)

        slider_position = st.slider(
            "Scrub through time",
            min_value=0,
            max_value=len(valid_epochs) - 1,
            value=initial_position,
            format=f"Epoch %d",
            key="epoch_slider",
        )
        st.session_state.slider_epoch_idx = slider_position
        actual_epoch_idx = valid_epochs[slider_position]

    # Epoch labels for display with safe date slicing
    epoch_labels = [safe_date_label(epoch_dates, i) for i in range(dims['n_epochs'])]

    # Show current epoch info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Epoch", epoch_labels[actual_epoch_idx])
    with col2:
        st.metric("Epoch Index", f"{slider_position + 1} of {len(valid_epochs)} available")
    with col3:
        if slider_position > 0:
            first_valid_epoch = valid_epochs[0]
            days_diff = _compute_days_between(epoch_dates, first_valid_epoch, actual_epoch_idx)
            if days_diff:
                st.metric("Days from First", days_diff)

    st.markdown("---")

    # Get transect data for current epoch
    try:
        transect = get_transect_by_id(data, transect_id, epoch_idx=actual_epoch_idx)
    except ValueError as e:
        st.error(str(e))
        return

    # Compute fixed y-axis range from first valid epoch for consistent comparison
    y_range = _get_fixed_y_range_valid(data, transect_id, selected_feature, feature_names, valid_epochs)

    # Get cliff positions if available
    cliff_pos = None
    if has_cliff_data(data):
        cliff_pos = get_cliff_positions_by_id(data, transect_id, actual_epoch_idx)

    # Main profile plot
    _render_profile_plot(transect, selected_feature, feature_names, epoch_labels[actual_epoch_idx], y_range, cliff_pos)

    # Show context: small multiples of valid epochs only
    with st.expander("Show All Available Epochs Overview", expanded=False):
        _render_epoch_thumbnails_valid(data, transect_id, selected_feature, feature_names, actual_epoch_idx, epoch_labels, valid_epochs)

    st.markdown("---")

    # Metadata at current epoch
    _render_epoch_metadata(transect, epoch_labels[actual_epoch_idx])


def _get_fixed_y_range(data: dict, transect_id: int, feature_name: str, feature_names: list) -> tuple:
    """
    Compute y-axis range from first epoch for consistent comparison.

    Returns:
        Tuple of (y_min, y_max) with padding
    """
    try:
        # Get first epoch data
        first_transect = get_transect_by_id(data, transect_id, epoch_idx=0)
        feature_idx = feature_names.index(feature_name)
        values = first_transect['points'][:, feature_idx]
        valid = values[~np.isnan(values)]

        if len(valid) == 0:
            return None

        y_min, y_max = float(valid.min()), float(valid.max())
        padding = (y_max - y_min) * 0.1

        return (y_min - padding, y_max + padding)
    except Exception:
        return None


def _get_fixed_y_range_valid(data: dict, transect_id: int, feature_name: str, feature_names: list, valid_epochs: list) -> tuple:
    """
    Compute y-axis range from all valid epochs for consistent comparison.

    Returns:
        Tuple of (y_min, y_max) with padding
    """
    try:
        feature_idx = feature_names.index(feature_name)
        all_values = []

        for epoch_idx in valid_epochs:
            transect = get_transect_by_id(data, transect_id, epoch_idx=epoch_idx)
            values = transect['points'][:, feature_idx]
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                all_values.extend(valid.tolist())

        if len(all_values) == 0:
            return None

        y_min, y_max = min(all_values), max(all_values)
        padding = (y_max - y_min) * 0.1

        return (y_min - padding, y_max + padding)
    except Exception:
        return None


def _compute_days_between(epoch_dates: list, idx1: int, idx2: int) -> str:
    """Compute days between two epochs."""
    if not epoch_dates:
        return None
    # Bounds check
    if idx1 >= len(epoch_dates) or idx2 >= len(epoch_dates):
        return None
    try:
        from datetime import datetime
        date1_str = epoch_dates[idx1]
        date2_str = epoch_dates[idx2]
        # Safe string slicing
        date1_part = date1_str[:10] if isinstance(date1_str, str) and len(date1_str) >= 10 else None
        date2_part = date2_str[:10] if isinstance(date2_str, str) and len(date2_str) >= 10 else None
        if date1_part is None or date2_part is None:
            return None
        d1 = datetime.fromisoformat(date1_part)
        d2 = datetime.fromisoformat(date2_part)
        return str((d2 - d1).days)
    except Exception:
        return None


def _render_profile_plot(
    transect: dict,
    feature_name: str,
    feature_names: list,
    epoch_label: str,
    y_range: tuple = None,
    cliff_pos: dict = None
):
    """Render the main profile plot for current epoch."""
    points = transect['points']
    distances = transect['distances']

    feature_idx = feature_names.index(feature_name)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=distances,
        y=points[:, feature_idx],
        mode='lines',
        name=feature_name,
        line=dict(
            color=config.FEATURE_COLORS.get(feature_name, '#1f77b4'),
            width=3,
        ),
        fill='tozeroy',
        fillcolor=f"rgba(31, 119, 180, 0.2)",
    ))

    # Add cliff toe/top markers if available and showing elevation
    if cliff_pos is not None and feature_name == 'elevation_m':
        toe_dist = cliff_pos['toe_distance']
        top_dist = cliff_pos['top_distance']
        toe_idx = cliff_pos.get('toe_idx')
        top_idx = cliff_pos.get('top_idx')
        n_points = points.shape[0] if points.ndim >= 1 else 0

        # Get elevations at toe/top with bounds checking
        if (toe_idx is not None and top_idx is not None and
            0 <= toe_idx < n_points and 0 <= top_idx < n_points):
            toe_elev = points[toe_idx, feature_idx]
            top_elev = points[top_idx, feature_idx]

            # Add toe marker
            fig.add_trace(go.Scatter(
                x=[toe_dist],
                y=[toe_elev],
                mode='markers+text',
                name='Cliff Toe',
                marker=dict(color='#e74c3c', size=12, symbol='triangle-up'),
                text=['Toe'],
                textposition='bottom center',
                textfont=dict(color='#e74c3c', size=10),
            ))

            # Add top marker
            fig.add_trace(go.Scatter(
                x=[top_dist],
                y=[top_elev],
                mode='markers+text',
                name='Cliff Top',
                marker=dict(color='#27ae60', size=12, symbol='triangle-down'),
                text=['Top'],
                textposition='top center',
                textfont=dict(color='#27ae60', size=10),
            ))

            # Add vertical lines at toe/top
            fig.add_vline(x=toe_dist, line_dash="dash", line_color="#e74c3c", opacity=0.5)
            fig.add_vline(x=top_dist, line_dash="dash", line_color="#27ae60", opacity=0.5)

    unit = config.FEATURE_UNITS.get(feature_name, '')
    fig.update_layout(
        title=f"{feature_name} Profile - {epoch_label}",
        xaxis_title="Distance from Toe (m)",
        yaxis_title=f"{feature_name} ({unit})" if unit else feature_name,
        height=450,
        showlegend=cliff_pos is not None and feature_name == 'elevation_m',
    )

    # Fix y-axis range for consistent comparison across epochs
    if y_range:
        fig.update_yaxes(range=list(y_range))

    st.plotly_chart(fig, use_container_width=True)


def _render_epoch_thumbnails(
    data: dict,
    transect_id: int,
    feature_name: str,
    feature_names: list,
    current_epoch: int,
    epoch_labels: list
):
    """Render small multiples showing all epochs."""
    dims = get_cube_dimensions(data)
    n_epochs = dims['n_epochs']
    feature_idx = feature_names.index(feature_name)

    # Determine grid layout
    n_cols = min(4, n_epochs)
    n_rows = (n_epochs + n_cols - 1) // n_cols

    # Calculate safe spacing (must be less than 1/(rows-1) for rows > 1)
    if n_rows > 1:
        max_v_spacing = 1.0 / (n_rows - 1) - 0.001  # Small buffer for float precision
        vertical_spacing = min(0.12, max_v_spacing)
    else:
        vertical_spacing = 0.12

    if n_cols > 1:
        max_h_spacing = 1.0 / (n_cols - 1) - 0.001
        horizontal_spacing = min(0.05, max_h_spacing)
    else:
        horizontal_spacing = 0.05

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=epoch_labels,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    # Get global y range for consistency
    all_values = []
    for t in range(n_epochs):
        try:
            transect = get_transect_by_id(data, transect_id, epoch_idx=t)
            values = transect['points'][:, feature_idx]
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                all_values.extend(valid.tolist())
        except Exception:
            pass

    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_padding = (y_max - y_min) * 0.1
        y_range = [y_min - y_padding, y_max + y_padding]
    else:
        y_range = None

    for t in range(n_epochs):
        row = t // n_cols + 1
        col = t % n_cols + 1

        try:
            transect = get_transect_by_id(data, transect_id, epoch_idx=t)
            distances = transect['distances']
            values = transect['points'][:, feature_idx]

            # Highlight current epoch
            line_color = '#e74c3c' if t == current_epoch else '#1f77b4'
            line_width = 2 if t == current_epoch else 1

            fig.add_trace(
                go.Scatter(
                    x=distances,
                    y=values,
                    mode='lines',
                    line=dict(color=line_color, width=line_width),
                    showlegend=False,
                ),
                row=row, col=col
            )

            if y_range:
                fig.update_yaxes(range=y_range, row=row, col=col)

        except Exception:
            pass

    fig.update_layout(
        height=200 * n_rows,
        title_text=f"All Epochs - {feature_name} (current epoch highlighted in red)",
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_epoch_thumbnails_valid(
    data: dict,
    transect_id: int,
    feature_name: str,
    feature_names: list,
    current_epoch: int,
    epoch_labels: list,
    valid_epochs: list
):
    """Render small multiples showing only valid epochs for this transect."""
    feature_idx = feature_names.index(feature_name)
    n_valid = len(valid_epochs)

    if n_valid == 0:
        st.info("No valid epochs to display")
        return

    # Determine grid layout
    n_cols = min(4, n_valid)
    n_rows = (n_valid + n_cols - 1) // n_cols

    # Calculate safe spacing (must be less than 1/(rows-1) for rows > 1)
    if n_rows > 1:
        max_v_spacing = 1.0 / (n_rows - 1) - 0.001
        vertical_spacing = min(0.12, max_v_spacing)
    else:
        vertical_spacing = 0.12

    if n_cols > 1:
        max_h_spacing = 1.0 / (n_cols - 1) - 0.001
        horizontal_spacing = min(0.05, max_h_spacing)
    else:
        horizontal_spacing = 0.05

    # Create labels for valid epochs only with bounds checking
    valid_labels = [epoch_labels[e] if e < len(epoch_labels) else f"Epoch {e}" for e in valid_epochs]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=valid_labels,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    # Get global y range from valid epochs for consistency
    all_values = []
    for epoch_idx in valid_epochs:
        try:
            transect = get_transect_by_id(data, transect_id, epoch_idx=epoch_idx)
            values = transect['points'][:, feature_idx]
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                all_values.extend(valid.tolist())
        except Exception:
            pass

    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_padding = (y_max - y_min) * 0.1
        y_range = [y_min - y_padding, y_max + y_padding]
    else:
        y_range = None

    for i, epoch_idx in enumerate(valid_epochs):
        row = i // n_cols + 1
        col = i % n_cols + 1

        try:
            transect = get_transect_by_id(data, transect_id, epoch_idx=epoch_idx)
            distances = transect['distances']
            values = transect['points'][:, feature_idx]

            # Highlight current epoch
            line_color = '#e74c3c' if epoch_idx == current_epoch else '#1f77b4'
            line_width = 2 if epoch_idx == current_epoch else 1

            fig.add_trace(
                go.Scatter(
                    x=distances,
                    y=values,
                    mode='lines',
                    line=dict(color=line_color, width=line_width),
                    showlegend=False,
                ),
                row=row, col=col
            )

            if y_range:
                fig.update_yaxes(range=y_range, row=row, col=col)

        except Exception:
            pass

    fig.update_layout(
        height=200 * n_rows,
        title_text=f"Available Epochs - {feature_name} (current epoch highlighted in red)",
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_epoch_metadata(transect: dict, epoch_label: str):
    """Render metadata summary for current epoch."""
    st.subheader(f"Transect Metadata - {epoch_label}")

    metadata = transect['metadata']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cliff Height", safe_metadata_value(metadata, 0, 2, " m"))
    with col2:
        st.metric("Mean Slope", safe_metadata_value(metadata, 1, 1, " deg"))
    with col3:
        st.metric("Max Slope", safe_metadata_value(metadata, 2, 1, " deg"))
    with col4:
        st.metric("Length", safe_metadata_value(metadata, 6, 1, " m"))
