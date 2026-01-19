"""Single transect inspector component."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from apps.transect_viewer import config
from apps.transect_viewer.utils.data_loader import get_transect_by_id, get_all_transect_ids


def render_inspector():
    """Render the single transect inspector view."""
    if st.session_state.data is None:
        st.warning("No data loaded")
        return

    data = st.session_state.data
    transect_id = st.session_state.selected_transect_id

    if transect_id is None:
        transect_ids = get_all_transect_ids(data)
        if transect_ids:
            transect_id = transect_ids[0]
            st.session_state.selected_transect_id = transect_id
        else:
            st.error("No transects found in data")
            return

    # Header with navigation
    st.header(f"Transect Inspector: ID {transect_id}")

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

    # Get transect data
    try:
        transect = get_transect_by_id(data, transect_id)
    except ValueError as e:
        st.error(str(e))
        return

    # Metadata summary
    _render_metadata_summary(transect)

    st.markdown("---")

    # Feature plots
    _render_feature_plots(transect)

    st.markdown("---")

    # RGB visualization
    _render_rgb_visualization(transect)


def _render_metadata_summary(transect: dict):
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
        st.metric("LAS Source", transect.get('las_source', 'N/A')[:20] + "...")


def _render_feature_plots(transect: dict):
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

    unit = config.FEATURE_UNITS.get(feature_names[feature_idx], '')
    fig.update_layout(
        title=f"{feature_names[feature_idx]} Profile",
        xaxis_title="Distance (m)",
        yaxis_title=f"{feature_names[feature_idx]} ({unit})" if unit else feature_names[feature_idx],
        height=config.PLOT_HEIGHT,
    )

    st.plotly_chart(fig, use_container_width=True)

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
        ri = int(np.clip(r[i] * 255, 0, 255))
        gi = int(np.clip(g[i] * 255, 0, 255))
        bi = int(np.clip(b[i] * 255, 0, 255))
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
