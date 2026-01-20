"""
CliffCast Transect Labeler

Streamlit application for labeling erosion classifications on transect epoch pairs.
Supports incremental labeling with parallel NPZ file persistence.

Usage:
    streamlit run apps/transect_labeler/app.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from apps.transect_labeler import config
from apps.transect_labeler.components.sidebar import render_sidebar
from apps.transect_labeler.components.pair_viewer import render_pair_viewer
from apps.transect_labeler.components.classification_panel import render_classification_panel
from apps.transect_labeler.components.progress_dashboard import render_progress_dashboard
from apps.transect_labeler.components.event_panel import render_event_panel


def init_session_state():
    """Initialize Streamlit session state variables."""
    # Data state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_path' not in st.session_state:
        st.session_state.data_path = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

    # Labels state
    if 'labels' not in st.session_state:
        st.session_state.labels = None
    if 'labels_metadata' not in st.session_state:
        st.session_state.labels_metadata = {}
    if 'labels_path' not in st.session_state:
        st.session_state.labels_path = None
    if 'labels_dirty' not in st.session_state:
        st.session_state.labels_dirty = False

    # Navigation state
    if 'current_transect_idx' not in st.session_state:
        st.session_state.current_transect_idx = 0
    if 'current_pair_idx' not in st.session_state:
        st.session_state.current_pair_idx = 0

    # Labeler info
    if 'labeler_name' not in st.session_state:
        st.session_state.labeler_name = "Connor"

    # View preferences
    if 'show_only_unlabeled' not in st.session_state:
        st.session_state.show_only_unlabeled = False
    if 'selected_feature' not in st.session_state:
        st.session_state.selected_feature = config.DEFAULT_FEATURE

    # Session tracking
    if 'session_start' not in st.session_state:
        st.session_state.session_start = datetime.now().isoformat()
    if 'pairs_labeled_this_session' not in st.session_state:
        st.session_state.pairs_labeled_this_session = 0

    # Event data state
    if 'events_df' not in st.session_state:
        st.session_state.events_df = None
    if 'events_dir' not in st.session_state:
        st.session_state.events_dir = config.DEFAULT_EVENTS_DIR
    if 'event_queue' not in st.session_state:
        st.session_state.event_queue = []
    if 'current_events' not in st.session_state:
        st.session_state.current_events = []


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon=config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Render sidebar (file loading, labeler info, navigation)
    render_sidebar()

    # Main content area
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")

    # Check if data is loaded
    if st.session_state.data is None:
        _render_welcome_screen()
        return

    # Update current events for this transect-pair
    _update_current_events()

    # Main labeling interface (three columns: viewer, classification, events)
    col_viewer, col_controls, col_events = st.columns([3, 1, 1])

    with col_viewer:
        render_pair_viewer()

    with col_controls:
        render_classification_panel()

    with col_events:
        render_event_panel()

    st.markdown("---")

    # Progress dashboard at bottom
    render_progress_dashboard()


def _update_current_events():
    """Update current events for the selected transect-pair."""
    if st.session_state.events_df is None or st.session_state.events_df.empty:
        st.session_state.current_events = []
        return

    try:
        from apps.transect_labeler.utils.event_loader import get_events_for_transect_pair

        events = get_events_for_transect_pair(
            st.session_state.events_df,
            st.session_state.current_transect_idx,
            st.session_state.current_pair_idx,
            st.session_state.data,
        )
        st.session_state.current_events = events
    except Exception:
        st.session_state.current_events = []


def _render_welcome_screen():
    """Render instructions when no data is loaded."""
    st.info("Please load a transect NPZ file using the sidebar.")
    st.markdown("""
    ### Getting Started

    1. **Enter your name**: For tracking labeling sessions
    2. **Load data**: Upload or enter path to a transect NPZ file
    3. **Load events** (recommended): Load significant erosion events from CSV files
    4. **Build event queue**: Prioritize labeling based on known events
    5. **Load existing labels** (optional): Continue from previous session
    6. **Start labeling**: View epoch pairs and classify erosion type

    ### Event-Based Workflow

    This app prioritizes labeling based on significant erosion events detected by M3C2:

    1. Load events from `data/raw/events/*_sig.csv` files
    2. Events are matched to transect-pairs by:
       - Converting alongshore coordinates to MOP IDs
       - Matching event dates to survey epochs
    3. Navigate through events sorted by volume (largest first)

    ### Classification Classes

    | Class | ID | Description |
    |-------|-----|-------------|
    | Stable | 0 | No significant change between epochs |
    | Beach erosion | 1 | Erosion at beach level (below cliff toe) |
    | Toe erosion | 2 | Erosion at cliff toe (notching, undercutting) |
    | Small rockfall | 3 | Minor cliff face material loss |
    | Large failure | 4 | Major cliff collapse event |

    ### Dominance Hierarchy

    When multiple processes are visible, classify by the **dominant** (most severe) process:

    **Large failure > Small rockfall > Toe erosion > Beach erosion > Stable**

    ### Keyboard Shortcuts

    Use the first letter of each class as a mental shortcut:
    - **S**table
    - **B**each erosion
    - **T**oe erosion
    - **R**ockfall (small)
    - **L**arge failure
    """)


if __name__ == "__main__":
    main()
