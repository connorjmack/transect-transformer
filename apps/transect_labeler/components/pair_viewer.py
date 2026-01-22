"""Side-by-side epoch pair viewer component."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from apps.transect_labeler import config
from apps.transect_labeler.utils.data_loader import (
    get_transect_pair,
    compute_pair_change,
    get_transect_ids,
    has_cliff_data,
    get_cliff_positions,
)


def render_pair_viewer():
    """Render side-by-side comparison of two consecutive epochs."""
    if st.session_state.data is None:
        return

    data = st.session_state.data
    transect_ids = get_transect_ids(data)
    transect_idx = st.session_state.current_transect_idx
    pair_idx = st.session_state.current_pair_idx

    transect_id = transect_ids[transect_idx] if transect_ids else transect_idx

    # Get pair data
    try:
        pair_data = get_transect_pair(data, transect_idx, pair_idx)
    except Exception as e:
        st.error(f"Error loading transect pair: {e}")
        return

    # Get cliff positions if available
    cliff_pos_epoch1 = None
    cliff_pos_epoch2 = None
    if has_cliff_data(data):
        cliff_pos_epoch1 = get_cliff_positions(data, transect_idx, pair_idx)
        cliff_pos_epoch2 = get_cliff_positions(data, transect_idx, pair_idx + 1)

    # Get feature index for plots (compute once)
    feature_name = st.session_state.selected_feature
    feature_names = pair_data.get('feature_names', [])
    if feature_name in feature_names:
        feature_idx = feature_names.index(feature_name)
    else:
        feature_idx = 1  # Default to elevation

    # Compute change data once (used by both difference plot and summary)
    change_data = compute_pair_change(pair_data, feature_idx)

    # Header with current position info
    st.subheader(f"Transect {transect_id}: {pair_data['epoch1_date']} -> {pair_data['epoch2_date']}")

    # Create comparison plot with cliff markers
    fig = _create_comparison_plot(pair_data, cliff_pos_epoch1, cliff_pos_epoch2, feature_idx)
    st.plotly_chart(fig, use_container_width=True, key=f"comparison_plot_{transect_idx}_{pair_idx}")

    # Difference plot below
    fig_diff = _create_difference_plot(pair_data, cliff_pos_epoch1, cliff_pos_epoch2, change_data)
    st.plotly_chart(fig_diff, use_container_width=True, key=f"diff_plot_{transect_idx}_{pair_idx}")

    # Summary statistics
    _render_change_summary(change_data)


def _create_comparison_plot(pair_data: dict, cliff_pos_epoch1: dict | None, cliff_pos_epoch2: dict | None, feature_idx: int) -> go.Figure:
    """Create overlaid profile plot comparing two epochs with cliff markers."""
    feature_name = st.session_state.selected_feature

    fig = go.Figure()

    # Epoch 1 (earlier)
    fig.add_trace(go.Scatter(
        x=pair_data['epoch1']['distances'],
        y=pair_data['epoch1']['points'][:, feature_idx],
        mode='lines',
        name=f'{pair_data["epoch1_date"]} (earlier)',
        line=dict(color=config.EPOCH_1_COLOR, width=config.PROFILE_LINE_WIDTH),
    ))

    # Epoch 2 (later)
    fig.add_trace(go.Scatter(
        x=pair_data['epoch2']['distances'],
        y=pair_data['epoch2']['points'][:, feature_idx],
        mode='lines',
        name=f'{pair_data["epoch2_date"]} (later)',
        line=dict(color=config.EPOCH_2_COLOR, width=config.PROFILE_LINE_WIDTH),
    ))

    # Add cliff markers for epoch 1
    if cliff_pos_epoch1 is not None:
        toe_idx = cliff_pos_epoch1.get('toe_idx')
        top_idx = cliff_pos_epoch1.get('top_idx')
        if toe_idx is not None and top_idx is not None:
            points1 = pair_data['epoch1']['points']
            # Toe marker (epoch 1)
            fig.add_trace(go.Scatter(
                x=[cliff_pos_epoch1['toe_distance']],
                y=[points1[toe_idx, feature_idx]],
                mode='markers',
                name=f'Toe ({pair_data["epoch1_date"]})',
                marker=dict(symbol='triangle-up', size=12, color=config.EPOCH_1_COLOR, line=dict(width=2, color='white')),
                showlegend=True,
            ))
            # Top marker (epoch 1)
            fig.add_trace(go.Scatter(
                x=[cliff_pos_epoch1['top_distance']],
                y=[points1[top_idx, feature_idx]],
                mode='markers',
                name=f'Top ({pair_data["epoch1_date"]})',
                marker=dict(symbol='triangle-down', size=12, color=config.EPOCH_1_COLOR, line=dict(width=2, color='white')),
                showlegend=True,
            ))

    # Add cliff markers for epoch 2
    if cliff_pos_epoch2 is not None:
        toe_idx = cliff_pos_epoch2.get('toe_idx')
        top_idx = cliff_pos_epoch2.get('top_idx')
        if toe_idx is not None and top_idx is not None:
            points2 = pair_data['epoch2']['points']
            # Toe marker (epoch 2)
            fig.add_trace(go.Scatter(
                x=[cliff_pos_epoch2['toe_distance']],
                y=[points2[toe_idx, feature_idx]],
                mode='markers',
                name=f'Toe ({pair_data["epoch2_date"]})',
                marker=dict(symbol='triangle-up', size=12, color=config.EPOCH_2_COLOR, line=dict(width=2, color='white')),
                showlegend=True,
            ))
            # Top marker (epoch 2)
            fig.add_trace(go.Scatter(
                x=[cliff_pos_epoch2['top_distance']],
                y=[points2[top_idx, feature_idx]],
                mode='markers',
                name=f'Top ({pair_data["epoch2_date"]})',
                marker=dict(symbol='triangle-down', size=12, color=config.EPOCH_2_COLOR, line=dict(width=2, color='white')),
                showlegend=True,
            ))

    unit = config.FEATURE_UNITS.get(feature_name, '')
    y_label = f"{feature_name} ({unit})" if unit else feature_name

    fig.update_layout(
        title=f"{feature_name} Profile Comparison",
        xaxis_title="Distance (m)",
        yaxis_title=y_label,
        height=config.PLOT_HEIGHT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
    )

    return fig


def _create_difference_plot(pair_data: dict, cliff_pos_epoch1: dict | None, cliff_pos_epoch2: dict | None, change_data: dict) -> go.Figure:
    """Create difference plot (epoch2 - epoch1) with cliff position indicators."""
    feature_name = st.session_state.selected_feature

    distances = change_data['distances']
    difference = change_data['difference']

    fig = go.Figure()

    # Fill positive (gain) and negative (loss)
    positive_diff = np.where(difference >= 0, difference, 0)
    negative_diff = np.where(difference < 0, difference, 0)

    fig.add_trace(go.Scatter(
        x=distances,
        y=positive_diff,
        mode='lines',
        fill='tozeroy',
        name='Gain (accretion)',
        line=dict(color='#27ae60', width=1),
        fillcolor='rgba(39, 174, 96, 0.3)',
    ))

    fig.add_trace(go.Scatter(
        x=distances,
        y=negative_diff,
        mode='lines',
        fill='tozeroy',
        name='Loss (erosion)',
        line=dict(color='#e74c3c', width=1),
        fillcolor='rgba(231, 76, 60, 0.3)',
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    # Add vertical lines for cliff positions
    if cliff_pos_epoch1 is not None:
        # Toe position (epoch 1) - dashed blue
        fig.add_vline(
            x=cliff_pos_epoch1['toe_distance'],
            line_dash="dash",
            line_color=config.EPOCH_1_COLOR,
            annotation_text="Toe (E1)",
            annotation_position="top left",
            opacity=0.7,
        )
        # Top position (epoch 1) - dashed blue
        fig.add_vline(
            x=cliff_pos_epoch1['top_distance'],
            line_dash="dash",
            line_color=config.EPOCH_1_COLOR,
            annotation_text="Top (E1)",
            annotation_position="top right",
            opacity=0.7,
        )

    if cliff_pos_epoch2 is not None:
        # Toe position (epoch 2) - solid orange
        fig.add_vline(
            x=cliff_pos_epoch2['toe_distance'],
            line_dash="solid",
            line_color=config.EPOCH_2_COLOR,
            annotation_text="Toe (E2)",
            annotation_position="bottom left",
            opacity=0.7,
        )
        # Top position (epoch 2) - solid orange
        fig.add_vline(
            x=cliff_pos_epoch2['top_distance'],
            line_dash="solid",
            line_color=config.EPOCH_2_COLOR,
            annotation_text="Top (E2)",
            annotation_position="bottom right",
            opacity=0.7,
        )

    unit = config.FEATURE_UNITS.get(feature_name, '')
    y_label = f"Delta {feature_name} ({unit})" if unit else f"Delta {feature_name}"

    fig.update_layout(
        title=f"Change: {pair_data['epoch2_date']} - {pair_data['epoch1_date']}",
        xaxis_title="Distance (m)",
        yaxis_title=y_label,
        height=250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
    )

    return fig


def _render_change_summary(change_data: dict):
    """Render summary statistics for the change."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Change", f"{change_data['mean_change']:.3f}")
    with col2:
        st.metric("Max Gain", f"{change_data['max_gain']:.3f}")
    with col3:
        st.metric("Max Loss", f"{change_data['max_loss']:.3f}")
    with col4:
        st.metric("Std Dev", f"{change_data['std_change']:.3f}")
