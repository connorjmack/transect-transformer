"""Configuration for transect viewer app."""

# App settings
APP_TITLE = "CliffCast Transect Viewer"
APP_ICON = ":ocean:"
DEFAULT_DATA_PATH = "data/processed/transects.npz"

# Supported data types (future extensibility)
SUPPORTED_DATA_TYPES = {
    'transects': True,
    'wave': False,      # Future: wave forcing time series
    'precipitation': False,  # Future: precip time series
}

# View mode options
VIEW_MODES = [
    "Data Dashboard",
    "Single Transect Inspector",
    "Temporal Slider",
    "Transect Evolution",
    "Cross-Transect View",
]

# Color schemes for features
FEATURE_COLORS = {
    'elevation_m': '#1f77b4',     # Blue
    'slope_deg': '#ff7f0e',       # Orange
    'curvature': '#2ca02c',       # Green
    'roughness': '#d62728',       # Red
    'intensity': '#9467bd',       # Purple
    'red': '#e74c3c',             # Red
    'green': '#27ae60',           # Green
    'blue': '#3498db',            # Blue
    'classification': '#8c564b',  # Brown
    'return_number': '#e377c2',   # Pink
    'num_returns': '#7f7f7f',     # Gray
    'distance_m': '#bcbd22',      # Yellow-green
}

# Color scheme for epochs (temporal comparison)
EPOCH_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
]

# Plot settings
PLOT_HEIGHT = 400
PLOT_WIDTH = 800
HISTOGRAM_BINS = 50

# Feature units for display
FEATURE_UNITS = {
    'distance_m': 'm',
    'elevation_m': 'm',
    'slope_deg': 'deg',
    'curvature': '1/m',
    'roughness': 'm',
    'intensity': '',
    'red': '',
    'green': '',
    'blue': '',
    'classification': '',
    'return_number': '',
    'num_returns': '',
}

# Metadata units for display
METADATA_UNITS = {
    'cliff_height_m': 'm',
    'mean_slope_deg': 'deg',
    'max_slope_deg': 'deg',
    'toe_elevation_m': 'm',
    'top_elevation_m': 'm',
    'orientation_deg': 'deg',
    'transect_length_m': 'm',
    'latitude': '',
    'longitude': '',
    'transect_id': '',
    'mean_intensity': '',
    'dominant_class': '',
}
