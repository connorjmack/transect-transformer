"""Configuration for transect labeler app."""

# App settings
APP_TITLE = "CliffCast Transect Labeler"
APP_ICON = ":label:"
DEFAULT_DATA_PATH = "data/processed/transects.npz"

# Classification classes
EROSION_CLASSES = {
    0: {'name': 'Stable', 'color': '#27ae60', 'shortcut': 's'},
    1: {'name': 'Beach erosion', 'color': '#3498db', 'shortcut': 'b'},
    2: {'name': 'Toe erosion', 'color': '#f39c12', 'shortcut': 't'},
    3: {'name': 'Small rockfall', 'color': '#e67e22', 'shortcut': 'r'},
    4: {'name': 'Large failure', 'color': '#e74c3c', 'shortcut': 'l'},
}

CLASS_NAMES = [c['name'] for c in EROSION_CLASSES.values()]
CLASS_COLORS = [c['color'] for c in EROSION_CLASSES.values()]

UNLABELED_VALUE = -1
UNLABELED_COLOR = '#95a5a6'

# Plot settings
PLOT_HEIGHT = 400
PROFILE_LINE_WIDTH = 2

# Epoch pair colors for side-by-side view
EPOCH_1_COLOR = '#1f77b4'  # Blue (earlier)
EPOCH_2_COLOR = '#ff7f0e'  # Orange (later)

# Default feature to display
DEFAULT_FEATURE = 'elevation_m'

# Feature colors (match transect_viewer)
FEATURE_COLORS = {
    'elevation_m': '#1f77b4',
    'slope_deg': '#ff7f0e',
    'curvature': '#2ca02c',
    'roughness': '#d62728',
}

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

# Labels file suffix
LABELS_SUFFIX = '_labels'

# Auto-save interval (pairs labeled)
AUTO_SAVE_INTERVAL = 10

# Event data settings
DEFAULT_EVENTS_DIR = "data/raw/events"

# MOP ranges for each beach (canonical)
BEACH_MOP_RANGES = {
    'blacks': (520, 567),
    'torrey': (567, 581),
    'delmar': (595, 620),
    'solana': (637, 666),
    'sanelijo': (683, 708),
    'encinitas': (708, 764),
}

# Event display colors
EVENT_MARKER_COLOR = '#9b59b6'  # Purple
EVENT_HIGHLIGHT_COLOR = 'rgba(155, 89, 182, 0.2)'

# Cliff marker colors
CLIFF_TOE_COLOR = '#e74c3c'  # Red
CLIFF_TOP_COLOR = '#9b59b6'  # Purple
