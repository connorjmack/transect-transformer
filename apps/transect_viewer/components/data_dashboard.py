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
from apps.transect_viewer.utils.data_loader import (
    get_cube_dimensions,
    get_epoch_dates,
    is_cube_format,
    check_data_coverage,
)


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Feature Distributions",
        "Metadata Distributions",
        "Data Quality",
        "Statistics Table",
        "Temporal Coverage",
    ])

    with tab1:
        _render_feature_distributions(data)

    with tab2:
        _render_metadata_distributions(data)

    with tab3:
        _render_data_quality(data)

    with tab4:
        _render_statistics_tables(data)

    with tab5:
        _render_temporal_coverage(data)


def _render_overview_metrics(data: dict):
    """Render overview metrics cards."""
    dims = get_cube_dimensions(data)
    is_cube = is_cube_format(data)
    epoch_dates = get_epoch_dates(data)

    # Create columns for metrics
    if is_cube:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
    else:
        col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Transects", dims['n_transects'])

    with col2:
        st.metric("Points/Transect", dims['n_points'])

    with col3:
        st.metric("Features", dims['n_features'])

    if is_cube:
        with col4:
            st.metric("Temporal Epochs", dims['n_epochs'])

        with col5:
            if epoch_dates:
                date_range = f"{epoch_dates[0][:10]} to {epoch_dates[-1][:10]}"
            else:
                date_range = "N/A"
            st.metric("Date Range", date_range)

        with col6:
            # Calculate average cliff height at latest epoch
            metadata = data['metadata']
            cliff_heights = metadata[:, -1, 0]  # Latest epoch, cliff_height_m
            avg_height = np.nanmean(cliff_heights)
            st.metric("Avg Cliff Height", f"{avg_height:.1f} m")
    else:
        with col4:
            st.metric("Format", "Flat (1 epoch)")

        with col5:
            metadata = data['metadata']
            cliff_heights = metadata[:, 0]  # cliff_height_m
            avg_height = np.nanmean(cliff_heights)
            st.metric("Avg Cliff Height", f"{avg_height:.1f} m")


def _render_feature_distributions(data: dict):
    """Render histograms for all features."""
    st.subheader("Feature Distributions")

    points = data['points']
    feature_names = data.get('feature_names', [])
    is_cube = is_cube_format(data)
    dims = get_cube_dimensions(data)

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    if not feature_names:
        feature_names = [f'feature_{i}' for i in range(dims['n_features'])]

    # Epoch selector for cube format
    epoch_idx = st.session_state.get('selected_epoch_idx', dims['n_epochs'] - 1)
    if is_cube and dims['n_epochs'] > 1:
        epoch_dates = get_epoch_dates(data)
        epoch_options = [
            f"{i}: {epoch_dates[i][:10]}" if epoch_dates else f"Epoch {i}"
            for i in range(dims['n_epochs'])
        ]
        epoch_options.append("All epochs combined")

        selected = st.selectbox(
            "Show distributions for:",
            range(dims['n_epochs'] + 1),
            index=dims['n_epochs'],  # Default to "All epochs"
            format_func=lambda x: epoch_options[x] if x < dims['n_epochs'] else "All epochs combined",
            key="feature_dist_epoch",
        )
        use_all_epochs = selected == dims['n_epochs']
        if not use_all_epochs:
            epoch_idx = selected
    else:
        use_all_epochs = True

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

        # Get values based on format and epoch selection
        if is_cube:
            if use_all_epochs:
                values = points[:, :, :, i].flatten()
            else:
                values = points[:, epoch_idx, :, i].flatten()
        else:
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

    title_suffix = "(all epochs)" if use_all_epochs else f"(epoch {epoch_idx})"
    fig.update_layout(
        height=800,
        title_text=f"Distribution of All Features {title_suffix}",
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_metadata_distributions(data: dict):
    """Render histograms for metadata fields."""
    st.subheader("Metadata Distributions")

    metadata = data['metadata']
    metadata_names = data.get('metadata_names', [])
    is_cube = is_cube_format(data)
    dims = get_cube_dimensions(data)

    if isinstance(metadata_names, np.ndarray):
        metadata_names = metadata_names.tolist()

    if not metadata_names:
        n_meta = metadata.shape[-1]
        metadata_names = [f'meta_{i}' for i in range(n_meta)]

    # Epoch selector for cube format
    epoch_idx = st.session_state.get('selected_epoch_idx', dims['n_epochs'] - 1)
    if is_cube and dims['n_epochs'] > 1:
        epoch_dates = get_epoch_dates(data)
        epoch_options = [
            f"{i}: {epoch_dates[i][:10]}" if epoch_dates else f"Epoch {i}"
            for i in range(dims['n_epochs'])
        ]

        epoch_idx = st.selectbox(
            "Show metadata for epoch:",
            range(dims['n_epochs']),
            index=dims['n_epochs'] - 1,
            format_func=lambda x: epoch_options[x],
            key="meta_dist_epoch",
        )

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

        # Get values based on format
        if is_cube:
            values = metadata[:, epoch_idx, i]
        else:
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

    epoch_label = f"epoch {epoch_idx}" if is_cube else "single epoch"
    fig.update_layout(
        height=800,
        title_text=f"Distribution of Metadata Fields ({epoch_label})",
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

    # Format info
    st.info(f"Data format: {'Cube (4D)' if report['is_cube_format'] else 'Flat (3D)'}")

    # Coverage for cube format
    if report['is_cube_format'] and 'coverage_pct' in report:
        st.metric("Data Coverage", f"{report['coverage_pct']:.1f}%")

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
        epoch_names = data.get('epoch_names', [])
        if isinstance(epoch_names, np.ndarray):
            epoch_names = epoch_names.tolist()
        st.write(f"- Epoch names: {len(epoch_names)}")
        st.write(f"- Feature names: {len(data.get('feature_names', []))}")


def _render_statistics_tables(data: dict):
    """Render statistics tables for features and metadata."""
    dims = get_cube_dimensions(data)
    is_cube = is_cube_format(data)

    # Epoch selector
    epoch_idx = -1  # Default to latest
    if is_cube and dims['n_epochs'] > 1:
        epoch_dates = get_epoch_dates(data)
        epoch_options = [
            f"{i}: {epoch_dates[i][:10]}" if epoch_dates else f"Epoch {i}"
            for i in range(dims['n_epochs'])
        ]

        epoch_idx = st.selectbox(
            "Compute statistics for epoch:",
            range(dims['n_epochs']),
            index=dims['n_epochs'] - 1,
            format_func=lambda x: epoch_options[x],
            key="stats_epoch",
        )

    st.subheader("Feature Statistics")

    # Feature statistics
    feature_stats = compute_statistics(data, epoch_idx=epoch_idx)
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
    metadata_stats = compute_metadata_statistics(data, epoch_idx=epoch_idx)
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


def _render_temporal_coverage(data: dict):
    """Render temporal coverage visualization for cube format."""
    st.subheader("Temporal Coverage")

    is_cube = is_cube_format(data)

    if not is_cube:
        st.info("Temporal coverage analysis requires cube format data (multiple epochs)")
        return

    # Get coverage stats
    coverage = check_data_coverage(data)
    epoch_dates = get_epoch_dates(data)
    dims = get_cube_dimensions(data)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transect-Epoch Cells", coverage['total_cells'])
    with col2:
        st.metric("Present", coverage['present_cells'])
    with col3:
        st.metric("Coverage", f"{coverage['coverage_pct']:.1f}%")

    if coverage['full_coverage']:
        st.success("All transects have data for all epochs")
    else:
        st.warning(f"Missing {coverage['missing_cells']} transect-epoch pairs")

    # Coverage heatmap
    st.subheader("Coverage Matrix")
    st.caption("Green = data present, Red = missing")

    coverage_matrix = coverage['coverage_matrix']

    # Create heatmap
    epoch_labels = [d[:10] if epoch_dates else f"E{i}" for i, d in enumerate(epoch_dates)] if epoch_dates else [f"E{i}" for i in range(dims['n_epochs'])]
    transect_ids = data['transect_ids']
    if isinstance(transect_ids, np.ndarray):
        transect_ids = transect_ids.tolist()

    # Sample if too many transects for visualization
    max_display = 100
    if len(transect_ids) > max_display:
        st.info(f"Showing first {max_display} of {len(transect_ids)} transects")
        display_matrix = coverage_matrix[:max_display]
        display_ids = transect_ids[:max_display]
    else:
        display_matrix = coverage_matrix
        display_ids = transect_ids

    fig = go.Figure(data=go.Heatmap(
        z=display_matrix.astype(int),
        x=epoch_labels,
        y=[f"T-{tid}" for tid in display_ids],
        colorscale=[[0, 'red'], [1, 'green']],
        showscale=False,
    ))

    fig.update_layout(
        title="Data Presence by Transect and Epoch",
        xaxis_title="Epoch",
        yaxis_title="Transect ID",
        height=min(600, 100 + len(display_ids) * 5),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Per-epoch coverage bar chart
    st.subheader("Coverage per Epoch")

    coverage_per_epoch = coverage['coverage_per_epoch']
    total_transects = dims['n_transects']

    fig = go.Figure(data=go.Bar(
        x=epoch_labels,
        y=coverage_per_epoch,
        marker_color='#3498db',
    ))

    fig.add_hline(y=total_transects, line_dash="dash", line_color="gray",
                  annotation_text=f"Total: {total_transects}")

    fig.update_layout(
        title="Number of Transects with Data per Epoch",
        xaxis_title="Epoch",
        yaxis_title="Transects with Data",
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)
