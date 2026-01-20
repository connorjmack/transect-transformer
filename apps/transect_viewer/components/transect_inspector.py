"""Single transect inspector component."""

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


def render_inspector():
    """Render the single transect inspector view."""
    if st.session_state.data is None:
        st.warning("No data loaded")
        return

    data = st.session_state.data
    transect_id = st.session_state.selected_transect_id
    epoch_idx = st.session_state.get('selected_epoch_idx', -1)

    dims = get_cube_dimensions(data)
    is_cube = is_cube_format(data)
    epoch_dates = get_epoch_dates(data)

    if transect_id is None:
        transect_ids = get_all_transect_ids(data)
        if transect_ids:
            transect_id = transect_ids[0]
            st.session_state.selected_transect_id = transect_id
        else:
            st.error("No transects found in data")
            return

    # Header with navigation
    epoch_label = ""
    if is_cube and epoch_dates:
        epoch_label = f" - {epoch_dates[epoch_idx][:10]}"
    elif is_cube:
        epoch_label = f" - Epoch {epoch_idx}"

    st.header(f"Transect Inspector: ID {transect_id}{epoch_label}")

    # Navigation buttons
    transect_ids = get_all_transect_ids(data)
    current_idx = transect_ids.index(transect_id) if transect_id in transect_ids else 0

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous", disabled=current_idx == 0):
            st.session_state.selected_transect_id = transect_ids[current_idx - 1]
            st.rerun()
    with col2:
        st.write(f"Transect {current_idx + 1} of {len(transect_ids)}")
    with col3:
        if st.button("Next", disabled=current_idx >= len(transect_ids) - 1):
            st.session_state.selected_transect_id = transect_ids[current_idx + 1]
            st.rerun()

    # Get transect data for specific epoch
    try:
        transect = get_transect_by_id(data, transect_id, epoch_idx=epoch_idx)
    except ValueError as e:
        st.error(str(e))
        return

    # Metadata summary
    _render_metadata_summary(transect, is_cube, epoch_dates, epoch_idx)

    st.markdown("---")

    # Get cliff positions if available
    cliff_pos = None
    if has_cliff_data(data):
        cliff_pos = get_cliff_positions_by_id(data, transect_id, epoch_idx)

    # Feature plots
    _render_feature_plots(transect, cliff_pos)

    st.markdown("---")

    # RGB visualization
    _render_rgb_visualization(transect)


def _render_metadata_summary(transect: dict, is_cube: bool, epoch_dates: list, epoch_idx: int):
    """Render metadata summary card."""
    st.subheader("Transect Metadata")

    metadata = transect['metadata']
    metadata_names = transect.get('metadata_names', [])

    if isinstance(metadata_names, np.ndarray):
        metadata_names = metadata_names.tolist()

    if not metadata_names:
        metadata_names = [f'meta_{i}' for i in range(len(metadata))]

    # Create columns for key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cliff Height", f"{metadata[0]:.2f} m")
        st.metric("Mean Slope", f"{metadata[1]:.1f}°")

    with col2:
        st.metric("Max Slope", f"{metadata[2]:.1f}°")
        st.metric("Toe Elevation", f"{metadata[3]:.2f} m")

    with col3:
        st.metric("Top Elevation", f"{metadata[4]:.2f} m")
        st.metric("Orientation", f"{metadata[5]:.1f}°")

    with col4:
        st.metric("Length", f"{metadata[6]:.1f} m")
        if is_cube and epoch_dates:
            st.metric("Epoch Date", epoch_dates[epoch_idx][:10])
        else:
            st.metric("Transect ID", transect.get('transect_id', 'N/A'))


def _render_feature_plots(transect: dict, cliff_pos: dict = None):
    """Render plots for all features."""
    st.subheader("Feature Profiles")

    points = transect['points']
    distances = transect['distances']
    feature_names = transect.get('feature_names', [])

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    if not feature_names:
        feature_names = [f'feature_{i}' for i in range(points.shape[1])]

    # Primary feature plot (large)
    selected_feature = st.session_state.selected_feature
    if selected_feature in feature_names:
        feature_idx = feature_names.index(selected_feature)
    else:
        feature_idx = 1  # Default to elevation

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=distances,
        y=points[:, feature_idx],
        mode='lines',
        name=feature_names[feature_idx],
        line=dict(
            color=config.FEATURE_COLORS.get(feature_names[feature_idx], '#1f77b4'),
            width=2,
        ),
    ))

    # Add cliff toe/top markers if available and showing elevation
    if cliff_pos is not None and selected_feature == 'elevation_m':
        toe_dist = cliff_pos['toe_distance']
        top_dist = cliff_pos['top_distance']
        toe_idx = cliff_pos.get('toe_idx')
        top_idx = cliff_pos.get('top_idx')

        # Get elevations at toe/top
        if toe_idx is not None and top_idx is not None:
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

    unit = config.FEATURE_UNITS.get(feature_names[feature_idx], '')
    fig.update_layout(
        title=f"{feature_names[feature_idx]} Profile",
        xaxis_title="Distance (m)",
        yaxis_title=f"{feature_names[feature_idx]} ({unit})" if unit else feature_names[feature_idx],
        height=config.PLOT_HEIGHT,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show cliff info if available
    if cliff_pos is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cliff Toe", f"{cliff_pos['toe_distance']:.1f} m")
        with col2:
            st.metric("Cliff Top", f"{cliff_pos['top_distance']:.1f} m")
        with col3:
            cliff_width = cliff_pos['top_distance'] - cliff_pos['toe_distance']
            st.metric("Cliff Width", f"{cliff_width:.1f} m")

    # All features grid
    with st.expander("Show All Features", expanded=False):
        _render_all_features_grid(transect)


def _render_all_features_grid(transect: dict):
    """Render grid of all feature plots."""
    points = transect['points']
    distances = transect['distances']
    feature_names = transect.get('feature_names', [])

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    if not feature_names:
        feature_names = [f'feature_{i}' for i in range(points.shape[1])]

    # Create subplot grid
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=feature_names,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for i, name in enumerate(feature_names):
        row = i // 3 + 1
        col = i % 3 + 1

        fig.add_trace(
            go.Scatter(
                x=distances,
                y=points[:, i],
                mode='lines',
                name=name,
                line=dict(color=config.FEATURE_COLORS.get(name, '#1f77b4')),
                showlegend=False,
            ),
            row=row, col=col
        )

    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)


def _render_rgb_visualization(transect: dict):
    """Render RGB color visualization along transect."""
    st.subheader("RGB Color Profile")

    points = transect['points']
    distances = transect['distances']
    feature_names = transect.get('feature_names', [])

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    # Find RGB indices
    try:
        r_idx = feature_names.index('red')
        g_idx = feature_names.index('green')
        b_idx = feature_names.index('blue')
    except ValueError:
        st.info("RGB features not found in data")
        return

    # Get RGB values
    r = points[:, r_idx]
    g = points[:, g_idx]
    b = points[:, b_idx]

    # Create color bar visualization
    fig = go.Figure()

    # Plot each channel
    fig.add_trace(go.Scatter(
        x=distances, y=r,
        mode='lines', name='Red',
        line=dict(color='red', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=distances, y=g,
        mode='lines', name='Green',
        line=dict(color='green', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=distances, y=b,
        mode='lines', name='Blue',
        line=dict(color='blue', width=2),
    ))

    fig.update_layout(
        title="RGB Channel Values Along Transect",
        xaxis_title="Distance (m)",
        yaxis_title="Normalized Value (0-1)",
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Color bar image
    st.markdown("**Color Bar:**")

    # Create a simple color representation
    n_points = len(r)
    colors = []
    for i in range(n_points):
        ri = int(np.clip(r[i] * 255, 0, 255)) if not np.isnan(r[i]) else 128
        gi = int(np.clip(g[i] * 255, 0, 255)) if not np.isnan(g[i]) else 128
        bi = int(np.clip(b[i] * 255, 0, 255)) if not np.isnan(b[i]) else 128
        colors.append(f'rgb({ri},{gi},{bi})')

    # Create heatmap-style color bar
    fig = go.Figure(data=go.Heatmap(
        z=[[i for i in range(n_points)]],
        colorscale=[[i / (n_points - 1), colors[i]] for i in range(n_points)],
        showscale=False,
    ))

    fig.update_layout(
        height=100,
        xaxis_title="Point Index",
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
