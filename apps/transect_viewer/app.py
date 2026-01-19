"""
CliffCast Transect Viewer

Streamlit application for visual inspection of transect NPZ files.
Supports data validation and temporal analysis of coastal cliff transects.

Usage:
    streamlit run apps/transect_viewer/app.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from apps.transect_viewer import config
from apps.transect_viewer.components.sidebar import render_sidebar
from apps.transect_viewer.components.data_dashboard import render_dashboard
from apps.transect_viewer.components.transect_inspector import render_inspector
from apps.transect_viewer.components.temporal_slider import render_temporal_slider
from apps.transect_viewer.components.evolution_view import render_evolution
from apps.transect_viewer.components.cross_transect_view import render_cross_transect


def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'selected_transect_id' not in st.session_state:
        st.session_state.selected_transect_id = None

    if 'selected_epoch_idx' not in st.session_state:
        st.session_state.selected_epoch_idx = 0

    if 'slider_epoch_idx' not in st.session_state:
        st.session_state.slider_epoch_idx = 0

    if 'selected_feature' not in st.session_state:
        st.session_state.selected_feature = 'elevation_m'

    if 'selected_transects' not in st.session_state:
        st.session_state.selected_transects = []

    if 'current_file' not in st.session_state:
        st.session_state.current_file = None


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

    # Render sidebar (file loading and navigation)
    view_mode = render_sidebar()

    # Main content area
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")

    # Check if data is loaded
    if st.session_state.data is None:
        st.info("Please load a transect NPZ file using the sidebar.")
        st.markdown("""
        ### Getting Started

        1. **Upload a file** using the file uploader in the sidebar, or
        2. **Enter a path** to an existing NPZ file

        The viewer supports **cube format** NPZ files with spatio-temporal data:
        - **Data Dashboard**: Overview statistics, quality checks, and temporal coverage
        - **Single Transect Inspector**: Detailed view of individual transects at any epoch
        - **Transect Evolution**: Analyze temporal changes across all epochs (built-in)
        - **Cross-Transect View**: Spatial analysis across multiple transects

        Cube format shape: `(n_transects, T, N, 12)` where T = temporal epochs
        """)
        return

    # Render selected view
    if view_mode == "Data Dashboard":
        render_dashboard()
    elif view_mode == "Single Transect Inspector":
        render_inspector()
    elif view_mode == "Temporal Slider":
        render_temporal_slider()
    elif view_mode == "Transect Evolution":
        render_evolution()
    elif view_mode == "Cross-Transect View":
        render_cross_transect()
    else:
        st.error(f"Unknown view mode: {view_mode}")


if __name__ == "__main__":
    main()
