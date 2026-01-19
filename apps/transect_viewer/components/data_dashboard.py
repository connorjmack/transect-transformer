"""Data dashboard component showing overview and statistics."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from apps.transect_viewer import config
from apps.transect_viewer.utils.validators import (
    validate_dataset,
    compute_statistics,
    compute_metadata_statistics,
)
from apps.transect_viewer.utils.date_parser import infer_epoch_date, format_date_for_display


def render_dashboard():
    """Render the data dashboard view."""
    if st.session_state.data is None:
        st.warning("No data loaded")
        return

    data = st.session_state.data

    # Header
    st.header("Data Dashboard")

    # Overview metrics
    _render_overview_metrics(data)

    st.markdown("---")

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Feature Distributions",
        "Metadata Distributions",
        "Data Quality",
        "Statistics Table",
    ])

    with tab1:
        _render_feature_distributions(data)

    with tab2:
        _render_metadata_distributions(data)

    with tab3:
        _render_data_quality(data)

    with tab4:
        _render_statistics_tables(data)


def _render_overview_metrics(data: dict):
    """Render overview metrics cards."""
    points = data['points']
    metadata = data['metadata']

    # Get epoch date
    epoch_date = infer_epoch_date(data.get('las_sources', []))

    # Create columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Transects", points.shape[0])

    with col2:
        st.metric("Points/Transect", points.shape[1])

    with col3:
        st.metric("Features", points.shape[2])

    with col4:
        st.metric("Epoch Date", format_date_for_display(epoch_date))

    with col5:
        # Calculate average cliff height
        cliff_heights = metadata[:, 0]  # cliff_height_m
        avg_height = np.nanmean(cliff_heights)
        st.metric("Avg Cliff Height", f"{avg_height:.1f} m")


def _render_feature_distributions(data: dict):
    """Render histograms for all features."""
    st.subheader("Feature Distributions")

    points = data['points']
    feature_names = data.get('feature_names', [])

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    if not feature_names:
        feature_names = [f'feature_{i}' for i in range(points.shape[2])]

    # Create subplot grid (4 rows x 3 cols for 12 features)
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=feature_names,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for i, name in enumerate(feature_names):
        row = i // 3 + 1
        col = i % 3 + 1

        # Get all values for this feature
        values = points[:, :, i].flatten()
        values = values[~np.isnan(values)]

        # Get color
        color = config.FEATURE_COLORS.get(name, '#1f77b4')

        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=config.HISTOGRAM_BINS,
                marker_color=color,
                opacity=0.7,
                name=name,
                showlegend=False,
            ),
            row=row, col=col
        )

        # Add unit to x-axis label
        unit = config.FEATURE_UNITS.get(name, '')
        if unit:
            fig.update_xaxes(title_text=unit, row=row, col=col)

    fig.update_layout(
        height=800,
        title_text="Distribution of All Features (across all transects and points)",
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_metadata_distributions(data: dict):
    """Render histograms for metadata fields."""
    st.subheader("Metadata Distributions")

    metadata = data['metadata']
    metadata_names = data.get('metadata_names', [])

    if isinstance(metadata_names, np.ndarray):
        metadata_names = metadata_names.tolist()

    if not metadata_names:
        metadata_names = [f'meta_{i}' for i in range(metadata.shape[1])]

    # Create subplot grid (4 rows x 3 cols for 12 metadata fields)
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=metadata_names,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for i, name in enumerate(metadata_names):
        row = i // 3 + 1
        col = i % 3 + 1

        # Get all values for this metadata field
        values = metadata[:, i]
        values = values[~np.isnan(values)]

        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=config.HISTOGRAM_BINS,
                marker_color='#2ecc71',
                opacity=0.7,
                name=name,
                showlegend=False,
            ),
            row=row, col=col
        )

        # Add unit to x-axis label
        unit = config.METADATA_UNITS.get(name, '')
        if unit:
            fig.update_xaxes(title_text=unit, row=row, col=col)

    fig.update_layout(
        height=800,
        title_text="Distribution of Metadata Fields (one value per transect)",
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_data_quality(data: dict):
    """Render data quality checks and warnings."""
    st.subheader("Data Quality Report")

    # Run validation
    report = validate_dataset(data)

    # Status indicator
    if report['is_valid']:
        st.success("Data validation passed")
    else:
        st.error("Data validation issues found")

    # Warnings
    if report['warnings']:
        st.subheader("Warnings")
        for warning in report['warnings']:
            st.warning(warning)
    else:
        st.info("No warnings")

    # NaN values
    st.subheader("NaN Values by Feature")
    nan_counts = report['nan_counts']
    if nan_counts:
        total_nan = sum(nan_counts.values())
        if total_nan == 0:
            st.success("No NaN values found in any feature")
        else:
            nan_df = pd.DataFrame([
                {'Feature': k, 'NaN Count': v}
                for k, v in nan_counts.items()
                if v > 0
            ])
            st.dataframe(nan_df, use_container_width=True)
    else:
        st.info("No feature names available for NaN check")

    # Feature range issues
    if report['feature_issues']:
        st.subheader("Feature Range Issues")
        issues_df = pd.DataFrame(report['feature_issues'])
        st.dataframe(issues_df, use_container_width=True)
    else:
        st.success("All features within expected ranges")

    # Metadata range issues
    if report['metadata_issues']:
        st.subheader("Metadata Range Issues")
        issues_df = pd.DataFrame(report['metadata_issues'])
        st.dataframe(issues_df, use_container_width=True)
    else:
        st.success("All metadata within expected ranges")

    # Shape information
    st.subheader("Data Shape")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- Points array: {data['points'].shape}")
        st.write(f"- Distances array: {data['distances'].shape}")
        st.write(f"- Metadata array: {data['metadata'].shape}")
    with col2:
        st.write(f"- Transect IDs: {len(data['transect_ids'])}")
        st.write(f"- LAS sources: {len(data['las_sources'])}")
        st.write(f"- Feature names: {len(data.get('feature_names', []))}")


def _render_statistics_tables(data: dict):
    """Render statistics tables for features and metadata."""
    st.subheader("Feature Statistics")

    # Feature statistics
    feature_stats = compute_statistics(data)
    st.dataframe(
        feature_stats.style.format({
            'min': '{:.4f}',
            'max': '{:.4f}',
            'mean': '{:.4f}',
            'std': '{:.4f}',
            'median': '{:.4f}',
        }),
        use_container_width=True,
    )

    st.subheader("Metadata Statistics")

    # Metadata statistics
    metadata_stats = compute_metadata_statistics(data)
    st.dataframe(
        metadata_stats.style.format({
            'min': '{:.4f}',
            'max': '{:.4f}',
            'mean': '{:.4f}',
            'std': '{:.4f}',
            'median': '{:.4f}',
        }),
        use_container_width=True,
    )
