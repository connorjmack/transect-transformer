"""Transect evolution view for temporal comparison (cube format)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from apps.transect_viewer import config
from apps.transect_viewer.utils.data_loader import (
    get_transect_by_id,
    get_transect_temporal_slice,
    compute_temporal_change,
    get_cube_dimensions,
    get_epoch_dates,
    is_cube_format,
    get_all_transect_ids,
)
from apps.transect_viewer.utils.validators import compute_temporal_statistics


def render_evolution():
    """Render the transect evolution view for cube format data."""
    if st.session_state.data is None:
        st.warning("No data loaded")
        return

    data = st.session_state.data
    is_cube = is_cube_format(data)
    dims = get_cube_dimensions(data)

    st.header("Transect Evolution")

    if not is_cube:
        st.warning("Temporal evolution requires cube format data.")
        st.info("""
        **Cube format data required:**
        Your data appears to be in flat format (single epoch).
        To view temporal evolution, load a cube format NPZ file
        generated from multiple LiDAR scans.
        """)
        return

    if dims['n_epochs'] < 2:
        st.warning("Temporal evolution requires at least 2 epochs.")
        return

    epoch_dates = get_epoch_dates(data)
    transect_id = st.session_state.selected_transect_id

    # Info about temporal coverage
    st.success(f"Cube data loaded: {dims['n_transects']} transects × {dims['n_epochs']} epochs")
    if epoch_dates:
        st.info(f"Date range: {epoch_dates[0][:10]} to {epoch_dates[-1][:10]}")

    # Transect selector (moved from sidebar for better UX)
    transect_ids = get_all_transect_ids(data)
    col1, col2 = st.columns([2, 1])

    with col1:
        transect_id = st.selectbox(
            "Select Transect",
            transect_ids,
            index=transect_ids.index(transect_id) if transect_id in transect_ids else 0,
            key="evolution_transect_selector",
        )
        st.session_state.selected_transect_id = transect_id

    # Feature selector
    feature_names = data.get('feature_names', [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    with col2:
        selected_feature = st.selectbox(
            "Feature to Analyze",
            feature_names,
            index=feature_names.index('elevation_m') if 'elevation_m' in feature_names else 0,
            key="evolution_feature_selector",
        )

    st.markdown("---")

    # Render temporal comparison
    _render_temporal_comparison(data, transect_id, selected_feature, feature_names, epoch_dates)

    st.markdown("---")

    # Render change detection
    _render_change_detection(data, transect_id, selected_feature, feature_names, epoch_dates)

    st.markdown("---")

    # Render temporal heatmap
    _render_temporal_heatmap(data, transect_id, selected_feature, feature_names, epoch_dates)


def _render_temporal_comparison(
    data: dict,
    transect_id: int,
    feature_name: str,
    feature_names: list,
    epoch_dates: list
):
    """Render overlaid profiles from all epochs."""
    st.subheader("Profile Comparison Across All Epochs")

    try:
        distances, values, dates = get_transect_temporal_slice(data, transect_id, feature_name)
    except ValueError as e:
        st.error(str(e))
        return

    n_epochs = values.shape[0]

    fig = go.Figure()

    for t in range(n_epochs):
        epoch_label = dates[t][:10] if dates else f"Epoch {t}"
        color = config.EPOCH_COLORS[t % len(config.EPOCH_COLORS)]

        # Skip if all NaN for this epoch
        if np.all(np.isnan(values[t])):
            continue

        fig.add_trace(go.Scatter(
            x=distances[t],
            y=values[t],
            mode='lines',
            name=epoch_label,
            line=dict(color=color, width=2),
        ))

    unit = config.FEATURE_UNITS.get(feature_name, '')
    fig.update_layout(
        title=f"{feature_name} Profile Evolution - Transect {transect_id}",
        xaxis_title="Distance (m)",
        yaxis_title=f"{feature_name} ({unit})" if unit else feature_name,
        height=config.PLOT_HEIGHT,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_change_detection(
    data: dict,
    transect_id: int,
    feature_name: str,
    feature_names: list,
    epoch_dates: list
):
    """Render difference between selected epochs."""
    st.subheader("Change Detection")

    dims = get_cube_dimensions(data)

    # Epoch selection for comparison
    col1, col2 = st.columns(2)

    epoch_options = [
        f"{i}: {epoch_dates[i][:10]}" if epoch_dates else f"Epoch {i}"
        for i in range(dims['n_epochs'])
    ]

    with col1:
        epoch1_idx = st.selectbox(
            "First Epoch (baseline)",
            range(dims['n_epochs']),
            index=0,
            format_func=lambda x: epoch_options[x],
            key="change_epoch1",
        )

    with col2:
        epoch2_idx = st.selectbox(
            "Second Epoch (comparison)",
            range(dims['n_epochs']),
            index=dims['n_epochs'] - 1,
            format_func=lambda x: epoch_options[x],
            key="change_epoch2",
        )

    # Compute change
    try:
        change = compute_temporal_change(
            data, transect_id, feature_name,
            epoch1_idx=epoch1_idx, epoch2_idx=epoch2_idx
        )
    except Exception as e:
        st.error(f"Error computing change: {e}")
        return

    # Plot difference
    fig = go.Figure()

    # Fill positive/negative areas differently
    difference = change['difference']
    distances = change['distances']

    # Positive changes (gain)
    fig.add_trace(go.Scatter(
        x=distances,
        y=np.where(difference >= 0, difference, 0),
        mode='lines',
        fill='tozeroy',
        name='Increase',
        line=dict(color='#27ae60', width=1),
        fillcolor='rgba(39, 174, 96, 0.3)',
    ))

    # Negative changes (loss)
    fig.add_trace(go.Scatter(
        x=distances,
        y=np.where(difference < 0, difference, 0),
        mode='lines',
        fill='tozeroy',
        name='Decrease',
        line=dict(color='#e74c3c', width=1),
        fillcolor='rgba(231, 76, 60, 0.3)',
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    unit = config.FEATURE_UNITS.get(feature_name, '')
    fig.update_layout(
        title=f"Change in {feature_name}: {change['epoch2_date'][:10]} - {change['epoch1_date'][:10]}",
        xaxis_title="Distance (m)",
        yaxis_title=f"Δ {feature_name} ({unit})" if unit else f"Δ {feature_name}",
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean Change", f"{change['mean_change']:.3f}")

    with col2:
        st.metric("Max Increase", f"{change['max_change']:.3f}")

    with col3:
        st.metric("Max Decrease", f"{change['min_change']:.3f}")

    with col4:
        st.metric("Std Dev", f"{change['std_change']:.3f}")


def _render_temporal_heatmap(
    data: dict,
    transect_id: int,
    feature_name: str,
    feature_names: list,
    epoch_dates: list
):
    """Render heatmap showing feature values over time and distance."""
    st.subheader("Temporal Heatmap")

    try:
        distances, values, dates = get_transect_temporal_slice(data, transect_id, feature_name)
    except ValueError as e:
        st.error(str(e))
        return

    n_epochs = values.shape[0]
    n_points = values.shape[1]

    # Create a COMMON distance grid across all epochs
    # Find the common distance range (intersection of all valid data)
    all_min_dist = []
    all_max_dist = []
    for t in range(n_epochs):
        valid_mask = ~np.isnan(values[t])
        if valid_mask.any():
            valid_distances = distances[t][valid_mask]
            all_min_dist.append(valid_distances.min())
            all_max_dist.append(valid_distances.max())

    if not all_min_dist:
        st.warning("No valid data for heatmap")
        return

    # Use the INTERSECTION of distance ranges for consistent comparison
    common_min = max(all_min_dist)
    common_max = min(all_max_dist)

    if common_max <= common_min:
        st.warning("No overlapping distance range across epochs")
        return

    # Create common distance grid
    common_distances = np.linspace(common_min, common_max, n_points)

    # Interpolate each epoch onto the common grid
    interpolated_values = np.full((n_epochs, n_points), np.nan)
    for t in range(n_epochs):
        valid_mask = ~np.isnan(values[t])
        if valid_mask.sum() > 1:
            # Interpolate onto common grid
            interpolated_values[t] = np.interp(
                common_distances,
                distances[t][valid_mask],
                values[t][valid_mask],
                left=np.nan,
                right=np.nan
            )

    # Epoch labels
    epoch_labels = [d[:10] if dates else f"E{i}" for i, d in enumerate(dates)] if dates else [f"E{i}" for i in range(n_epochs)]

    # Create heatmap with actual distance values on x-axis
    fig = go.Figure(data=go.Heatmap(
        z=interpolated_values,
        x=common_distances,
        y=epoch_labels,
        colorscale='RdBu_r' if feature_name == 'elevation_m' else 'Viridis',
        colorbar=dict(title=feature_name),
    ))

    fig.update_layout(
        title=f"{feature_name} Evolution Heatmap - Transect {transect_id}",
        xaxis_title="Distance from Toe (m)",
        yaxis_title="Epoch",
        height=300 + n_epochs * 20,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Temporal statistics
    with st.expander("Show Temporal Statistics", expanded=False):
        try:
            stats_df = compute_temporal_statistics(data, feature_name)
            st.dataframe(
                stats_df.style.format({
                    'min': '{:.3f}',
                    'max': '{:.3f}',
                    'mean': '{:.3f}',
                    'std': '{:.3f}',
                }),
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Could not compute temporal statistics: {e}")
