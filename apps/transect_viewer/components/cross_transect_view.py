"""Cross-transect view for spatial analysis."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.transect_viewer import config
from apps.transect_viewer.utils.data_loader import get_all_transect_ids, get_transect_by_id


def render_cross_transect():
    """Render the cross-transect spatial view."""
    if st.session_state.data is None:
        st.warning("No data loaded")
        return

    data = st.session_state.data

    st.header("Cross-Transect View")

    # Map section
    _render_location_map(data)

    st.markdown("---")

    # Multi-transect comparison
    _render_multi_transect_comparison(data)


def _render_location_map(data: dict):
    """Render map of transect locations."""
    st.subheader("Transect Locations")

    metadata = data['metadata']
    metadata_names = data.get('metadata_names', [])
    transect_ids = get_all_transect_ids(data)

    if isinstance(metadata_names, np.ndarray):
        metadata_names = metadata_names.tolist()

    # Get coordinate indices
    try:
        lat_idx = metadata_names.index('latitude')
        lon_idx = metadata_names.index('longitude')
    except ValueError:
        st.warning("Coordinate data not found in metadata")
        return

    # Get color field from sidebar selection or default
    color_field = st.session_state.get('map_color_field', 'cliff_height_m')
    if color_field in metadata_names:
        color_idx = metadata_names.index(color_field)
    else:
        color_idx = 0  # cliff_height_m

    # Create dataframe for plotting
    df = pd.DataFrame({
        'transect_id': transect_ids,
        'x': metadata[:, lon_idx],
        'y': metadata[:, lat_idx],
        'color_value': metadata[:, color_idx],
        'cliff_height': metadata[:, 0],
        'mean_slope': metadata[:, 1],
    })

    # Check if coordinates are UTM (large values)
    is_utm = df['x'].max() > 180 or df['y'].max() > 90

    if is_utm:
        st.info("Coordinates appear to be in UTM. Displaying as scatter plot.")

        # Simple scatter plot for UTM coordinates
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='color_value',
            hover_data=['transect_id', 'cliff_height', 'mean_slope'],
            color_continuous_scale='Viridis',
            labels={
                'x': 'Easting (m)',
                'y': 'Northing (m)',
                'color_value': color_field,
            },
        )

        fig.update_layout(
            height=500,
            title=f"Transect Locations (colored by {color_field})",
        )

        # Make points clickable
        fig.update_traces(
            marker=dict(size=10),
            selector=dict(mode='markers'),
        )

    else:
        # Use mapbox for geographic coordinates
        fig = px.scatter_mapbox(
            df,
            lat='y',
            lon='x',
            color='color_value',
            hover_data=['transect_id', 'cliff_height', 'mean_slope'],
            color_continuous_scale='Viridis',
            zoom=10,
            mapbox_style='open-street-map',
        )

        fig.update_layout(
            height=500,
            title=f"Transect Locations (colored by {color_field})",
        )

    st.plotly_chart(fig, use_container_width=True)

    # Transect selection
    st.markdown("**Select transects to compare:**")

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_ids = st.multiselect(
            "Transect IDs",
            transect_ids,
            default=st.session_state.selected_transects[:5] if st.session_state.selected_transects else transect_ids[:3],
            max_selections=10,
        )
        st.session_state.selected_transects = selected_ids

    with col2:
        if st.button("Select All"):
            st.session_state.selected_transects = transect_ids[:10]
            st.rerun()


def _render_multi_transect_comparison(data: dict):
    """Render comparison of multiple selected transects."""
    st.subheader("Multi-Transect Comparison")

    selected_ids = st.session_state.selected_transects

    if not selected_ids:
        st.info("Select transects from the map above to compare")
        return

    # Feature selector
    feature_names = data.get('feature_names', [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    selected_feature = st.selectbox(
        "Compare Feature",
        feature_names,
        index=feature_names.index('elevation_m') if 'elevation_m' in feature_names else 0,
        key="cross_transect_feature",
    )

    feature_idx = feature_names.index(selected_feature)

    # Create comparison plot
    fig = go.Figure()

    for i, tid in enumerate(selected_ids):
        transect = get_transect_by_id(data, tid)
        color = config.EPOCH_COLORS[i % len(config.EPOCH_COLORS)]

        fig.add_trace(go.Scatter(
            x=transect['distances'],
            y=transect['points'][:, feature_idx],
            mode='lines',
            name=f"T-{tid}",
            line=dict(color=color, width=2),
        ))

    unit = config.FEATURE_UNITS.get(selected_feature, '')
    fig.update_layout(
        title=f"{selected_feature} Comparison - {len(selected_ids)} Transects",
        xaxis_title="Distance (m)",
        yaxis_title=f"{selected_feature} ({unit})" if unit else selected_feature,
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

    # Summary table
    st.subheader("Selected Transects Summary")

    summary_data = []
    for tid in selected_ids:
        transect = get_transect_by_id(data, tid)
        metadata = transect['metadata']

        summary_data.append({
            'ID': tid,
            'Cliff Height (m)': f"{metadata[0]:.2f}",
            'Mean Slope (°)': f"{metadata[1]:.1f}",
            'Max Slope (°)': f"{metadata[2]:.1f}",
            'Length (m)': f"{metadata[6]:.1f}",
            'Orientation (°)': f"{metadata[5]:.1f}",
        })

    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
