"""Sidebar component for file loading, labeler info, and navigation."""

from pathlib import Path

import streamlit as st
import numpy as np

from apps.transect_labeler import config
from apps.transect_labeler.utils.data_loader import (
    load_npz,
    load_npz_from_upload,
    get_cube_dimensions,
    get_epoch_dates,
    get_transect_ids,
)
from apps.transect_labeler.utils.label_manager import (
    create_empty_labels,
    load_labels,
    save_labels,
    get_default_labels_path,
)
from apps.transect_labeler.utils.event_loader import (
    load_all_sig_events,
    build_event_queue,
    get_events_for_transect_pair,
)


def render_sidebar():
    """Render the sidebar with file loading and navigation controls."""
    st.sidebar.title("Transect Labeler")

    # Section 1: Data file loading
    _render_data_loading_section()

    # Labeler name (collapsed in expander, defaults to Connor)
    with st.sidebar.expander("Settings", expanded=False):
        labeler_name = st.text_input(
            "Labeler name",
            value=st.session_state.labeler_name or "Connor",
            help="Stored in labels file for tracking",
        )
        st.session_state.labeler_name = labeler_name

    if st.session_state.data is not None:
        st.sidebar.markdown("---")

        # Section 3: Labels file management
        _render_labels_section()

        st.sidebar.markdown("---")

        # Section 4: Events loading
        _render_events_section()

        st.sidebar.markdown("---")

        # Section 5: Navigation
        _render_navigation_section()

        st.sidebar.markdown("---")

        # Section 6: View options
        _render_view_options()


def _render_data_loading_section():
    """Render data file upload/path input."""
    st.sidebar.header("Data File")

    uploaded_file = st.sidebar.file_uploader(
        "Upload NPZ file",
        type=["npz"],
        help="Upload a transect NPZ cube file",
    )

    st.sidebar.markdown("**Or enter file path:**")
    file_path = st.sidebar.text_input(
        "NPZ file path",
        value=st.session_state.get('data_path_input', config.DEFAULT_DATA_PATH),
        label_visibility="collapsed",
        key="data_path_input_field",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Load File", use_container_width=True):
            _load_data_from_path(file_path)
    with col2:
        if st.button("Clear", use_container_width=True):
            _clear_all()

    if uploaded_file is not None:
        if st.session_state.current_file != uploaded_file.name:
            _load_data_from_upload(uploaded_file)

    # Show loaded data info
    if st.session_state.data is not None:
        dims = get_cube_dimensions(st.session_state.data)
        st.sidebar.success(f"Loaded: {st.session_state.current_file}")
        st.sidebar.text(f"  {dims['n_transects']} transects")
        st.sidebar.text(f"  {dims['n_epochs']} epochs")
        st.sidebar.text(f"  {dims['n_epochs'] - 1} pairs to label")


def _render_labels_section():
    """Render labels file management."""
    st.sidebar.header("Labels File")

    data_path = st.session_state.data_path
    if data_path:
        default_path = get_default_labels_path(data_path)
    else:
        default_path = Path("labels.npz")

    # Show current labels status
    if st.session_state.labels is not None:
        n_labeled = int(np.sum(st.session_state.labels != config.UNLABELED_VALUE))
        n_total = st.session_state.labels.size
        pct = 100 * n_labeled / n_total if n_total > 0 else 0
        st.sidebar.metric("Progress", f"{n_labeled}/{n_total} ({pct:.1f}%)")

        if st.session_state.labels_dirty:
            st.sidebar.warning("Unsaved changes!")

    # Labels path input
    current_labels_path = st.session_state.labels_path or str(default_path)
    labels_path = st.sidebar.text_input(
        "Labels file path",
        value=current_labels_path,
        help="Path for loading/saving labels",
        key="labels_path_input",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Load Labels", use_container_width=True, disabled=st.session_state.data is None):
            _load_labels(labels_path)
    with col2:
        if st.button("New Labels", use_container_width=True, disabled=st.session_state.data is None):
            _create_new_labels(labels_path)

    # Save button (always visible if labels exist)
    if st.session_state.labels is not None:
        if st.sidebar.button(
            "Save Labels",
            type="primary" if st.session_state.labels_dirty else "secondary",
            use_container_width=True,
        ):
            _save_labels(labels_path)


def _render_events_section():
    """Render events status and queue navigation."""
    st.sidebar.header("Event Queue")

    event_queue = st.session_state.get('event_queue', [])
    events_df = st.session_state.get('events_df')

    if events_df is None or events_df.empty:
        st.sidebar.text("No events loaded")
        st.sidebar.text(f"Looking in: {config.DEFAULT_EVENTS_DIR}")
        return

    if not event_queue:
        n_events = len(events_df)
        st.sidebar.text(f"{n_events} events loaded")
        st.sidebar.warning("No events match loaded data")
        return

    # Queue statistics
    n_queue = len(event_queue)
    n_labeled = 0
    if st.session_state.labels is not None:
        for item in event_queue:
            if st.session_state.labels[item['transect_idx'], item['pair_idx']] != config.UNLABELED_VALUE:
                n_labeled += 1

    # Find current position in queue
    current_transect = st.session_state.current_transect_idx
    current_pair = st.session_state.current_pair_idx
    current_queue_idx = None
    for i, item in enumerate(event_queue):
        if item['transect_idx'] == current_transect and item['pair_idx'] == current_pair:
            current_queue_idx = i
            break

    # Status display
    st.sidebar.metric("Events Labeled", f"{n_labeled} / {n_queue}")
    if current_queue_idx is not None:
        st.sidebar.text(f"Current: Event #{current_queue_idx + 1}")
        # Show current event info
        current_event = event_queue[current_queue_idx]['event']
        st.sidebar.text(f"Volume: {current_event['volume']:.1f} m³")
        st.sidebar.text(f"Beach: {current_event['beach'].title()}")
    else:
        st.sidebar.text("(Not on an event)")

    # Navigation buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("< Prev Event", use_container_width=True, key="prev_event_sidebar"):
            _go_to_prev_event_in_queue()
    with col2:
        if st.button("Next Event >", use_container_width=True, key="next_event_sidebar"):
            _go_to_next_event_in_queue()

    if st.sidebar.button("Next Unlabeled Event", use_container_width=True, key="next_unlabeled_event_sidebar"):
        _go_to_next_unlabeled_event_in_queue()


def _go_to_prev_event_in_queue():
    """Navigate to previous event in queue."""
    event_queue = st.session_state.get('event_queue', [])
    if not event_queue:
        return

    current_transect = st.session_state.current_transect_idx
    current_pair = st.session_state.current_pair_idx

    # Find current position
    current_idx = None
    for i, item in enumerate(event_queue):
        if item['transect_idx'] == current_transect and item['pair_idx'] == current_pair:
            current_idx = i
            break

    if current_idx is None:
        # Not in queue, go to first event
        target = event_queue[0]
    elif current_idx > 0:
        target = event_queue[current_idx - 1]
    else:
        return  # Already at first

    st.session_state.current_transect_idx = target['transect_idx']
    st.session_state.current_pair_idx = target['pair_idx']
    st.rerun()


def _go_to_next_event_in_queue():
    """Navigate to next event in queue."""
    event_queue = st.session_state.get('event_queue', [])
    if not event_queue:
        return

    current_transect = st.session_state.current_transect_idx
    current_pair = st.session_state.current_pair_idx

    # Find current position
    current_idx = None
    for i, item in enumerate(event_queue):
        if item['transect_idx'] == current_transect and item['pair_idx'] == current_pair:
            current_idx = i
            break

    if current_idx is None:
        target = event_queue[0]
    elif current_idx < len(event_queue) - 1:
        target = event_queue[current_idx + 1]
    else:
        return  # Already at last

    st.session_state.current_transect_idx = target['transect_idx']
    st.session_state.current_pair_idx = target['pair_idx']
    st.rerun()


def _go_to_next_unlabeled_event_in_queue():
    """Navigate to next unlabeled event in queue."""
    event_queue = st.session_state.get('event_queue', [])
    labels = st.session_state.labels

    if not event_queue or labels is None:
        return

    current_transect = st.session_state.current_transect_idx
    current_pair = st.session_state.current_pair_idx

    # Find current position in queue
    current_idx = 0
    for i, item in enumerate(event_queue):
        if item['transect_idx'] == current_transect and item['pair_idx'] == current_pair:
            current_idx = i + 1  # Start from next
            break

    # Search from current position to end
    for i in range(current_idx, len(event_queue)):
        item = event_queue[i]
        if labels[item['transect_idx'], item['pair_idx']] == config.UNLABELED_VALUE:
            st.session_state.current_transect_idx = item['transect_idx']
            st.session_state.current_pair_idx = item['pair_idx']
            st.rerun()
            return

    # Wrap around to beginning
    for i in range(0, current_idx):
        item = event_queue[i]
        if labels[item['transect_idx'], item['pair_idx']] == config.UNLABELED_VALUE:
            st.session_state.current_transect_idx = item['transect_idx']
            st.session_state.current_pair_idx = item['pair_idx']
            st.rerun()
            return

    st.sidebar.info("All events labeled!")


def _render_navigation_section():
    """Render navigation controls."""
    st.sidebar.header("Navigation")

    data = st.session_state.data
    dims = get_cube_dimensions(data)
    transect_ids = get_transect_ids(data)
    n_pairs = dims['n_epochs'] - 1

    # Transect selector
    current_transect_id = transect_ids[st.session_state.current_transect_idx]
    selected_id = st.sidebar.selectbox(
        "Transect ID",
        transect_ids,
        index=st.session_state.current_transect_idx,
        key="transect_selector",
    )
    if selected_id != current_transect_id:
        st.session_state.current_transect_idx = transect_ids.index(selected_id)
        st.session_state.current_pair_idx = 0
        st.rerun()

    # Epoch pair selector
    epoch_dates = get_epoch_dates(data)
    pair_options = []
    for i in range(n_pairs):
        d1 = epoch_dates[i][:10] if epoch_dates else f"E{i}"
        d2 = epoch_dates[i + 1][:10] if epoch_dates else f"E{i + 1}"
        label_status = ""
        if st.session_state.labels is not None:
            label_val = st.session_state.labels[st.session_state.current_transect_idx, i]
            if label_val != config.UNLABELED_VALUE:
                label_status = f" [{config.CLASS_NAMES[label_val]}]"
            else:
                label_status = " [unlabeled]"
        pair_options.append(f"{d1} -> {d2}{label_status}")

    selected_pair = st.sidebar.selectbox(
        "Epoch Pair",
        range(n_pairs),
        index=st.session_state.current_pair_idx,
        format_func=lambda x: pair_options[x],
        key="pair_selector",
    )
    if selected_pair != st.session_state.current_pair_idx:
        st.session_state.current_pair_idx = selected_pair
        st.rerun()

    # Navigation buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("< Prev Pair", use_container_width=True,
                     disabled=st.session_state.current_pair_idx == 0):
            st.session_state.current_pair_idx -= 1
            st.rerun()
    with col2:
        if st.button("Next Pair >", use_container_width=True,
                     disabled=st.session_state.current_pair_idx >= n_pairs - 1):
            st.session_state.current_pair_idx += 1
            st.rerun()

    # Skip to next unlabeled
    if st.sidebar.button("Next Unlabeled", use_container_width=True,
                         disabled=st.session_state.labels is None):
        _jump_to_next_unlabeled()


def _render_view_options():
    """Render view customization options."""
    st.sidebar.header("View Options")

    # Feature selector
    data = st.session_state.data
    feature_names = data.get('feature_names', [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    if feature_names:
        default_idx = 0
        if st.session_state.selected_feature in feature_names:
            default_idx = feature_names.index(st.session_state.selected_feature)

        selected_feature = st.sidebar.selectbox(
            "Feature to display",
            feature_names,
            index=default_idx,
            key="feature_selector",
        )
        st.session_state.selected_feature = selected_feature

    # Show only unlabeled toggle
    st.session_state.show_only_unlabeled = st.sidebar.checkbox(
        "Show only unlabeled pairs",
        value=st.session_state.show_only_unlabeled,
        key="show_unlabeled_checkbox",
    )


# Helper functions


def _load_data_from_path(file_path: str) -> None:
    """Load data from file path and auto-load events."""
    try:
        path = Path(file_path)
        if not path.exists():
            st.sidebar.error(f"File not found: {file_path}")
            return

        data = load_npz(str(path))
        st.session_state.data = data
        st.session_state.data_path = str(path)
        st.session_state.current_file = path.name

        # Reset navigation
        st.session_state.current_transect_idx = 0
        st.session_state.current_pair_idx = 0

        # Clear labels when loading new data
        st.session_state.labels = None
        st.session_state.labels_metadata = {}
        st.session_state.labels_path = None
        st.session_state.labels_dirty = False

        # Auto-load events and build queue
        _auto_load_events_and_queue(data)

        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")


def _load_data_from_upload(uploaded_file) -> None:
    """Load data from uploaded file and auto-load events."""
    try:
        data = load_npz_from_upload(uploaded_file)
        st.session_state.data = data
        st.session_state.data_path = None  # No file path for uploads
        st.session_state.current_file = uploaded_file.name

        # Reset navigation
        st.session_state.current_transect_idx = 0
        st.session_state.current_pair_idx = 0

        # Clear labels
        st.session_state.labels = None
        st.session_state.labels_metadata = {}
        st.session_state.labels_path = None
        st.session_state.labels_dirty = False

        # Auto-load events and build queue
        _auto_load_events_and_queue(data)

        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")


def _auto_load_events_and_queue(data: dict) -> None:
    """Auto-load events from default directory and build event queue."""
    try:
        events_dir = Path(config.DEFAULT_EVENTS_DIR)
        if not events_dir.exists():
            st.session_state.events_df = None
            st.session_state.event_queue = []
            return

        # Load events
        events_df = load_all_sig_events(str(events_dir))
        if events_df.empty:
            st.session_state.events_df = None
            st.session_state.event_queue = []
            return

        st.session_state.events_df = events_df
        st.session_state.events_dir = str(events_dir)

        # Build event queue
        queue = build_event_queue(events_df, data, max_events=2000)
        st.session_state.event_queue = queue

        # Jump to first event if queue is not empty
        if queue:
            st.session_state.current_transect_idx = queue[0]['transect_idx']
            st.session_state.current_pair_idx = queue[0]['pair_idx']

    except Exception as e:
        # Silently fail - events are optional
        st.session_state.events_df = None
        st.session_state.event_queue = []


def _clear_all() -> None:
    """Clear all loaded data and labels."""
    st.session_state.data = None
    st.session_state.data_path = None
    st.session_state.current_file = None
    st.session_state.labels = None
    st.session_state.labels_metadata = {}
    st.session_state.labels_path = None
    st.session_state.labels_dirty = False
    st.session_state.current_transect_idx = 0
    st.session_state.current_pair_idx = 0
    st.rerun()


def _load_labels(labels_path: str) -> None:
    """Load existing labels file."""
    try:
        path = Path(labels_path)
        if not path.exists():
            st.sidebar.error(f"Labels file not found: {labels_path}")
            return

        labels, metadata, warnings = load_labels(
            str(path),
            data=st.session_state.data,
            data_path=st.session_state.data_path,
        )

        st.session_state.labels = labels
        st.session_state.labels_metadata = metadata
        st.session_state.labels_path = str(path)
        st.session_state.labels_dirty = False

        for warning in warnings:
            st.sidebar.warning(warning)

        st.sidebar.success(f"Loaded labels from {path.name}")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error loading labels: {e}")


def _create_new_labels(labels_path: str) -> None:
    """Create new empty labels file."""
    try:
        if st.session_state.data is None:
            st.sidebar.error("Load data first")
            return

        if st.session_state.data_path is None:
            st.sidebar.error("Cannot create labels for uploaded files (need file path for hash)")
            return

        labels, metadata = create_empty_labels(
            st.session_state.data,
            st.session_state.data_path,
            st.session_state.labeler_name,
        )

        st.session_state.labels = labels
        st.session_state.labels_metadata = metadata
        st.session_state.labels_path = labels_path
        st.session_state.labels_dirty = True

        st.sidebar.success("Created new labels file")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error creating labels: {e}")


def _save_labels(labels_path: str) -> None:
    """Save labels to file."""
    try:
        save_labels(
            labels_path,
            st.session_state.labels,
            st.session_state.labels_metadata,
            st.session_state.labeler_name,
        )
        st.session_state.labels_path = labels_path
        st.session_state.labels_dirty = False
        st.sidebar.success(f"Saved labels to {Path(labels_path).name}")
    except Exception as e:
        st.sidebar.error(f"Error saving labels: {e}")


def _jump_to_next_unlabeled() -> None:
    """Jump to the next unlabeled pair."""
    if st.session_state.labels is None:
        return

    labels = st.session_state.labels
    n_transects, n_pairs = labels.shape

    # Start from current position
    start_transect = st.session_state.current_transect_idx
    start_pair = st.session_state.current_pair_idx + 1  # Start after current

    # Search from current position to end
    for t in range(start_transect, n_transects):
        pair_start = start_pair if t == start_transect else 0
        for p in range(pair_start, n_pairs):
            if labels[t, p] == config.UNLABELED_VALUE:
                st.session_state.current_transect_idx = t
                st.session_state.current_pair_idx = p
                st.rerun()
                return

    # Wrap around to beginning
    for t in range(0, start_transect + 1):
        pair_end = start_pair if t == start_transect else n_pairs
        for p in range(0, pair_end):
            if labels[t, p] == config.UNLABELED_VALUE:
                st.session_state.current_transect_idx = t
                st.session_state.current_pair_idx = p
                st.rerun()
                return

    st.sidebar.info("All pairs are labeled!")
