"""Progress dashboard showing labeling statistics."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from apps.transect_labeler import config
from apps.transect_labeler.utils.data_loader import get_transect_ids


def render_progress_dashboard():
    """Render labeling progress statistics and summary."""
    if st.session_state.labels is None:
        return

    labels = st.session_state.labels
    n_transects, n_pairs = labels.shape

    st.subheader("Labeling Progress")

    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)

    total = labels.size
    labeled = int(np.sum(labels != config.UNLABELED_VALUE))
    unlabeled = total - labeled
    pct_complete = 100 * labeled / total if total > 0 else 0

    with col1:
        st.metric("Total Pairs", f"{total:,}")
    with col2:
        st.metric("Labeled", f"{labeled:,}")
    with col3:
        st.metric("Remaining", f"{unlabeled:,}")
    with col4:
        st.metric("Progress", f"{pct_complete:.1f}%")

    # Progress bar
    st.progress(pct_complete / 100)

    # Class distribution (excluding unlabeled)
    if labeled > 0:
        st.markdown("**Class Distribution:**")
        _render_class_distribution(labels)

    # Per-transect progress heatmap (optional, in expander)
    with st.expander("Show Per-Transect Progress", expanded=False):
        _render_transect_heatmap(labels)


def _render_class_distribution(labels: np.ndarray):
    """Render bar chart of class distribution."""
    class_counts = []
    for class_id in range(len(config.CLASS_NAMES)):
        count = int(np.sum(labels == class_id))
        class_counts.append(count)

    fig = go.Figure(data=[
        go.Bar(
            x=config.CLASS_NAMES,
            y=class_counts,
            marker_color=config.CLASS_COLORS,
            text=class_counts,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Labels by Class",
        xaxis_title="Class",
        yaxis_title="Count",
        height=250,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_transect_heatmap(labels: np.ndarray):
    """Render heatmap showing labeling progress per transect."""
    data = st.session_state.data
    transect_ids = get_transect_ids(data)

    # Create binary matrix: 1 = labeled, 0 = unlabeled
    labeled_matrix = (labels != config.UNLABELED_VALUE).astype(int)

    # Limit display if too many transects
    max_transects_display = 100
    if labels.shape[0] > max_transects_display:
        st.info(f"Showing first {max_transects_display} of {labels.shape[0]} transects")
        labeled_matrix = labeled_matrix[:max_transects_display]
        transect_ids = transect_ids[:max_transects_display]

    fig = go.Figure(data=go.Heatmap(
        z=labeled_matrix,
        x=[f"Pair {i}" for i in range(labels.shape[1])],
        y=[str(tid) for tid in transect_ids],
        colorscale=[[0, config.UNLABELED_COLOR], [1, '#27ae60']],
        showscale=False,
        hovertemplate='Transect: %{y}<br>Pair: %{x}<br>Status: %{z}<extra></extra>',
    ))

    fig.update_layout(
        title="Labeling Coverage (green = labeled)",
        xaxis_title="Epoch Pair",
        yaxis_title="Transect ID",
        height=max(200, min(labels.shape[0], max_transects_display) * 15),
    )

    st.plotly_chart(fig, use_container_width=True)
