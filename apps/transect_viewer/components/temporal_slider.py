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
)


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

    # Epoch slider
    epoch_labels = [
        f"{epoch_dates[i][:10]}" if epoch_dates else f"Epoch {i}"
        for i in range(dims['n_epochs'])
    ]

    slider_epoch = st.slider(
        "Scrub through time",
        min_value=0,
        max_value=dims['n_epochs'] - 1,
        value=st.session_state.get('slider_epoch_idx', 0),
        format=f"Epoch %d",
        key="epoch_slider",
    )
    st.session_state.slider_epoch_idx = slider_epoch

    # Show current epoch info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Epoch", epoch_labels[slider_epoch])
    with col2:
        st.metric("Epoch Index", f"{slider_epoch + 1} of {dims['n_epochs']}")
    with col3:
        if slider_epoch > 0:
            days_diff = _compute_days_between(epoch_dates, 0, slider_epoch)
            if days_diff:
                st.metric("Days from Start", days_diff)

    st.markdown("---")

    # Get transect data for current epoch
    try:
        transect = get_transect_by_id(data, transect_id, epoch_idx=slider_epoch)
    except ValueError as e:
        st.error(str(e))
        return

    # Main profile plot
    _render_profile_plot(transect, selected_feature, feature_names, epoch_labels[slider_epoch])

    # Show context: small multiples of all epochs
    with st.expander("Show All Epochs Overview", expanded=False):
        _render_epoch_thumbnails(data, transect_id, selected_feature, feature_names, slider_epoch, epoch_labels)

    st.markdown("---")

    # Metadata at current epoch
    _render_epoch_metadata(transect, epoch_labels[slider_epoch])


def _compute_days_between(epoch_dates: list, idx1: int, idx2: int) -> str:
    """Compute days between two epochs."""
    if not epoch_dates:
        return None
    try:
        from datetime import datetime
        d1 = datetime.fromisoformat(epoch_dates[idx1][:10])
        d2 = datetime.fromisoformat(epoch_dates[idx2][:10])
        return str((d2 - d1).days)
    except Exception:
        return None


def _render_profile_plot(transect: dict, feature_name: str, feature_names: list, epoch_label: str):
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

    unit = config.FEATURE_UNITS.get(feature_name, '')
    fig.update_layout(
        title=f"{feature_name} Profile - {epoch_label}",
        xaxis_title="Distance from Toe (m)",
        yaxis_title=f"{feature_name} ({unit})" if unit else feature_name,
        height=450,
        showlegend=False,
    )

    # Add range for consistency across epochs (optional)
    # This helps see changes more clearly
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

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=epoch_labels,
        vertical_spacing=0.12,
        horizontal_spacing=0.05,
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


def _render_epoch_metadata(transect: dict, epoch_label: str):
    """Render metadata summary for current epoch."""
    st.subheader(f"Transect Metadata - {epoch_label}")

    metadata = transect['metadata']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cliff Height", f"{metadata[0]:.2f} m")
    with col2:
        st.metric("Mean Slope", f"{metadata[1]:.1f} deg")
    with col3:
        st.metric("Max Slope", f"{metadata[2]:.1f} deg")
    with col4:
        st.metric("Length", f"{metadata[6]:.1f} m")
