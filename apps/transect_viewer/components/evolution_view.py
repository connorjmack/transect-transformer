"""Transect evolution view for temporal comparison."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from apps.transect_viewer import config
from apps.transect_viewer.utils.data_loader import get_common_transect_ids, get_transect_by_id


def render_evolution():
    """Render the transect evolution view."""
    st.header("Transect Evolution")

    epochs = st.session_state.epochs

    if len(epochs) < 2:
        st.warning("Load at least 2 epochs to compare temporal evolution.")
        st.info("""
        **How to load multiple epochs:**
        1. Use the sidebar's "Add epoch" file uploader
        2. Each NPZ file should be from a different scan date
        3. Transects are matched by ID across epochs
        """)

        # Show single epoch if available
        if len(epochs) == 1:
            st.subheader("Current Epoch")
            date_key = list(epochs.keys())[0]
            data = epochs[date_key]
            st.write(f"- Date: {date_key}")
            st.write(f"- Transects: {data['points'].shape[0]}")

        return

    # Find common transects
    common_ids = get_common_transect_ids(epochs)

    if not common_ids:
        st.error("No common transect IDs found across epochs")
        return

    st.success(f"Found {len(common_ids)} transects present in all {len(epochs)} epochs")

    # Transect selector
    selected_id = st.selectbox(
        "Select Transect ID",
        common_ids,
        index=0,
    )

    # Feature selector
    first_epoch = list(epochs.values())[0]
    feature_names = first_epoch.get('feature_names', [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    selected_feature = st.selectbox(
        "Select Feature",
        feature_names,
        index=feature_names.index('elevation_m') if 'elevation_m' in feature_names else 0,
    )

    # Render comparison plot
    _render_temporal_comparison(epochs, selected_id, selected_feature, feature_names)

    st.markdown("---")

    # Render difference plot
    _render_difference_plot(epochs, selected_id, selected_feature, feature_names)


def _render_temporal_comparison(
    epochs: dict,
    transect_id: int,
    feature_name: str,
    feature_names: list
):
    """Render overlaid profiles from different epochs."""
    st.subheader("Profile Comparison Across Epochs")

    feature_idx = feature_names.index(feature_name)

    fig = go.Figure()

    # Sort epochs by date
    sorted_dates = sorted(epochs.keys())

    for i, date_key in enumerate(sorted_dates):
        data = epochs[date_key]
        transect = get_transect_by_id(data, transect_id)

        color = config.EPOCH_COLORS[i % len(config.EPOCH_COLORS)]

        fig.add_trace(go.Scatter(
            x=transect['distances'],
            y=transect['points'][:, feature_idx],
            mode='lines',
            name=date_key,
            line=dict(color=color, width=2),
        ))

    unit = config.FEATURE_UNITS.get(feature_name, '')
    fig.update_layout(
        title=f"{feature_name} Profile - Transect {transect_id}",
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


def _render_difference_plot(
    epochs: dict,
    transect_id: int,
    feature_name: str,
    feature_names: list
):
    """Render difference between first and last epoch."""
    st.subheader("Change Detection")

    if len(epochs) < 2:
        return

    feature_idx = feature_names.index(feature_name)
    sorted_dates = sorted(epochs.keys())

    # Get first and last epoch
    first_date = sorted_dates[0]
    last_date = sorted_dates[-1]

    first_transect = get_transect_by_id(epochs[first_date], transect_id)
    last_transect = get_transect_by_id(epochs[last_date], transect_id)

    # Compute difference
    first_values = first_transect['points'][:, feature_idx]
    last_values = last_transect['points'][:, feature_idx]
    distances = first_transect['distances']

    difference = last_values - first_values

    # Create plot
    fig = go.Figure()

    # Fill positive/negative differently
    fig.add_trace(go.Scatter(
        x=distances,
        y=difference,
        mode='lines',
        fill='tozeroy',
        name='Change',
        line=dict(color='#3498db', width=2),
        fillcolor='rgba(52, 152, 219, 0.3)',
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    unit = config.FEATURE_UNITS.get(feature_name, '')
    fig.update_layout(
        title=f"Change in {feature_name}: {last_date} - {first_date}",
        xaxis_title="Distance (m)",
        yaxis_title=f"Δ {feature_name} ({unit})" if unit else f"Δ {feature_name}",
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean Change", f"{np.nanmean(difference):.3f}")

    with col2:
        st.metric("Max Increase", f"{np.nanmax(difference):.3f}")

    with col3:
        st.metric("Max Decrease", f"{np.nanmin(difference):.3f}")

    with col4:
        st.metric("Std Dev", f"{np.nanstd(difference):.3f}")
