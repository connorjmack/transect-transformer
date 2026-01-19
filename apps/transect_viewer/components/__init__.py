"""UI components for transect viewer."""

from apps.transect_viewer.components.sidebar import render_sidebar
from apps.transect_viewer.components.data_dashboard import render_dashboard
from apps.transect_viewer.components.transect_inspector import render_inspector
from apps.transect_viewer.components.temporal_slider import render_temporal_slider
from apps.transect_viewer.components.evolution_view import render_evolution
from apps.transect_viewer.components.cross_transect_view import render_cross_transect

__all__ = [
    "render_sidebar",
    "render_dashboard",
    "render_inspector",
    "render_temporal_slider",
    "render_evolution",
    "render_cross_transect",
]
