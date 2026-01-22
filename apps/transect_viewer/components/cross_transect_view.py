"""Cross-transect view for spatial analysis (cube format)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.transect_viewer import config
from apps.transect_viewer.utils.data_loader import (
    get_all_transect_ids,
    get_transect_by_id,
    get_cube_dimensions,
    get_epoch_dates,
    is_cube_format,
    get_epoch_slice,
    has_cliff_data,
    get_cliff_positions_by_id,
)


def render_cross_transect():
    """Render the cross-transect spatial view."""
    if st.session_state.data is None:
        st.warning("No data loaded")
        return

    data = st.session_state.data
    is_cube = is_cube_format(data)
    dims = get_cube_dimensions(data)
    epoch_dates = get_epoch_dates(data)
    epoch_idx = st.session_state.get('selected_epoch_idx', dims['n_epochs'] - 1)

    st.header("Cross-Transect View")

    # Show which epoch is being displayed
    if is_cube and epoch_dates and epoch_idx < len(epoch_dates):
        date_str = epoch_dates[epoch_idx]
        date_label = date_str[:10] if isinstance(date_str, str) and len(date_str) >= 10 else str(date_str)
        st.info(f"Showing data for epoch: {date_label}")
    elif is_cube:
        st.info(f"Showing data for epoch {epoch_idx}")

    # Map section
    _render_location_map(data, epoch_idx, is_cube)

    st.markdown("---")

    # Multi-transect comparison
    _render_multi_transect_comparison(data, epoch_idx, is_cube)


def _render_location_map(data: dict, epoch_idx: int, is_cube: bool):
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

    # Get metadata for specific epoch (cube format) or directly (flat format)
    if is_cube:
        # Use specific epoch metadata
        epoch_metadata = metadata[:, epoch_idx, :]
    else:
        epoch_metadata = metadata

    # Create dataframe for plotting with bounds checking
    n_cols = epoch_metadata.shape[1] if epoch_metadata.ndim >= 2 else 0
    df_data = {
        'transect_id': transect_ids,
        'x': epoch_metadata[:, lon_idx] if lon_idx < n_cols else np.zeros(len(transect_ids)),
        'y': epoch_metadata[:, lat_idx] if lat_idx < n_cols else np.zeros(len(transect_ids)),
        'color_value': epoch_metadata[:, color_idx] if color_idx < n_cols else np.zeros(len(transect_ids)),
        'cliff_height': epoch_metadata[:, 0] if 0 < n_cols else np.zeros(len(transect_ids)),
        'mean_slope': epoch_metadata[:, 1] if 1 < n_cols else np.zeros(len(transect_ids)),
    }
    df = pd.DataFrame(df_data)

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
        # Safely compute default selections
        if st.session_state.selected_transects:
            default_selections = [t for t in st.session_state.selected_transects[:5] if t in transect_ids]
        else:
            default_selections = transect_ids[:min(3, len(transect_ids))]

        selected_ids = st.multiselect(
            "Transect IDs",
            transect_ids,
            default=default_selections,
            max_selections=10,
        )
        st.session_state.selected_transects = selected_ids

    with col2:
        if st.button("Select All"):
            st.session_state.selected_transects = transect_ids[:10]
            st.rerun()


def _render_multi_transect_comparison(data: dict, epoch_idx: int, is_cube: bool):
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

    # Check if cliff data is available
    show_cliff_markers = has_cliff_data(data) and selected_feature == 'elevation_m'

    for i, tid in enumerate(selected_ids):
        # Get transect for specific epoch
        transect = get_transect_by_id(data, tid, epoch_idx=epoch_idx)
        color = config.EPOCH_COLORS[i % len(config.EPOCH_COLORS)]

        fig.add_trace(go.Scatter(
            x=transect['distances'],
            y=transect['points'][:, feature_idx],
            mode='lines',
            name=f"T-{tid}",
            line=dict(color=color, width=2),
        ))

        # Add cliff markers if showing elevation
        if show_cliff_markers:
            cliff_pos = get_cliff_positions_by_id(data, tid, epoch_idx)
            if cliff_pos is not None:
                toe_idx = cliff_pos.get('toe_idx')
                top_idx = cliff_pos.get('top_idx')
                points = transect['points']
                n_points = points.shape[0] if points.ndim >= 1 else 0
                # Validate indices are within bounds
                if (toe_idx is not None and top_idx is not None and
                    0 <= toe_idx < n_points and 0 <= top_idx < n_points and
                    feature_idx < points.shape[1] if points.ndim >= 2 else False):
                    # Toe marker
                    fig.add_trace(go.Scatter(
                        x=[cliff_pos['toe_distance']],
                        y=[points[toe_idx, feature_idx]],
                        mode='markers',
                        name=f'Toe T-{tid}',
                        marker=dict(color=color, size=10, symbol='triangle-up', line=dict(color='white', width=1)),
                        showlegend=False,
                    ))
                    # Top marker
                    fig.add_trace(go.Scatter(
                        x=[cliff_pos['top_distance']],
                        y=[points[top_idx, feature_idx]],
                        mode='markers',
                        name=f'Top T-{tid}',
                        marker=dict(color=color, size=10, symbol='triangle-down', line=dict(color='white', width=1)),
                        showlegend=False,
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

    metadata_names = data.get('metadata_names', [])
    if isinstance(metadata_names, np.ndarray):
        metadata_names = metadata_names.tolist()

    summary_data = []
    for tid in selected_ids:
        transect = get_transect_by_id(data, tid, epoch_idx=epoch_idx)
        metadata = transect['metadata']
        n_meta = len(metadata) if hasattr(metadata, '__len__') else 0

        # Safe access to metadata fields with bounds checking
        def safe_meta(idx, fmt=".2f"):
            if idx < n_meta and not np.isnan(metadata[idx]):
                return f"{metadata[idx]:{fmt}}"
            return "N/A"

        summary_data.append({
            'ID': tid,
            'Cliff Height (m)': safe_meta(0, ".2f"),
            'Mean Slope (deg)': safe_meta(1, ".1f"),
            'Max Slope (deg)': safe_meta(2, ".1f"),
            'Length (m)': safe_meta(6, ".1f"),
            'Orientation (deg)': safe_meta(5, ".1f"),
        })

    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
