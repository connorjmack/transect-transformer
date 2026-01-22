"""Classification buttons and label display component."""

import streamlit as st
import numpy as np

from apps.transect_labeler import config
from apps.transect_labeler.utils.label_manager import save_labels


def render_classification_panel():
    """Render the classification buttons and current label info."""
    if st.session_state.labels is None:
        st.warning("No labels file loaded. Create or load labels first.")
        return

    transect_idx = st.session_state.current_transect_idx
    pair_idx = st.session_state.current_pair_idx

    # Current label
    current_label = st.session_state.labels[transect_idx, pair_idx]

    st.subheader("Classification")

    # Show current label status
    if current_label == config.UNLABELED_VALUE:
        st.info("This pair is **unlabeled**")
    else:
        class_info = config.EROSION_CLASSES[current_label]
        st.success(f"Current label: **{class_info['name']}**")

    st.markdown("---")

    # Classification buttons (one per class)
    st.markdown("**Select classification:**")

    for class_id, class_info in config.EROSION_CLASSES.items():
        is_selected = (current_label == class_id)
        button_type = "primary" if is_selected else "secondary"

        if st.button(
            f"{class_info['name']} ({class_info['shortcut']})",
            key=f"class_btn_{class_id}",
            type=button_type,
            use_container_width=True,
        ):
            _set_label(class_id)

    st.markdown("---")

    # Clear label button
    if st.button(
        "Clear Label",
        key="clear_label_btn",
        use_container_width=True,
        disabled=current_label == config.UNLABELED_VALUE
    ):
        _set_label(config.UNLABELED_VALUE)

    st.markdown("---")

    # Quick actions
    st.markdown("**Quick Actions:**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Label & Next",
            key="label_next_btn",
            use_container_width=True,
            disabled=current_label == config.UNLABELED_VALUE
        ):
            _advance_to_next()
    with col2:
        if st.button("Skip", key="skip_btn", use_container_width=True):
            _advance_to_next()


def _set_label(class_id: int):
    """Set label for current transect-pair and mark dirty."""
    transect_idx = st.session_state.current_transect_idx
    pair_idx = st.session_state.current_pair_idx

    old_value = st.session_state.labels[transect_idx, pair_idx]
    st.session_state.labels[transect_idx, pair_idx] = class_id
    st.session_state.labels_dirty = True

    # Track labeling count (only if actually labeling, not clearing)
    if class_id != config.UNLABELED_VALUE and old_value == config.UNLABELED_VALUE:
        st.session_state.pairs_labeled_this_session += 1

        # Auto-save check
        if st.session_state.pairs_labeled_this_session % config.AUTO_SAVE_INTERVAL == 0:
            _trigger_auto_save()

    st.rerun()


def _advance_to_next():
    """Advance to next pair (or next transect if at end)."""
    data = st.session_state.data
    n_pairs = data['points'].shape[1] - 1
    n_transects = data['points'].shape[0]

    if st.session_state.current_pair_idx < n_pairs - 1:
        st.session_state.current_pair_idx += 1
    elif st.session_state.current_transect_idx < n_transects - 1:
        st.session_state.current_transect_idx += 1
        st.session_state.current_pair_idx = 0
    # else: at end, stay in place

    st.rerun()


def _trigger_auto_save():
    """Trigger auto-save of labels."""
    if st.session_state.labels_path:
        try:
            save_labels(
                st.session_state.labels_path,
                st.session_state.labels,
                st.session_state.labels_metadata,
                st.session_state.labeler_name,
            )
            st.session_state.labels_dirty = False
            st.toast(f"Auto-saved labels ({st.session_state.pairs_labeled_this_session} labeled this session)")
        except Exception as e:
            st.warning(f"Auto-save failed: {e}")
