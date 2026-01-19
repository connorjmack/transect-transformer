"""
Publication-quality figure showing CliffCast model input tensor shapes and features.

Creates a descriptive diagram illustrating the three input modalities:
- Transect data (spatio-temporal cliff geometry)
- Wave data (nearshore wave conditions)
- Atmospheric data (precipitation and temperature)

Output: figures/appendix/input_tensors.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Publication settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

# Color scheme
COLORS = {
    'transect': '#2E86AB',      # Blue for transect (geometry)
    'wave': '#A23B72',           # Purple for wave (forcing)
    'atmospheric': '#F18F01',    # Orange for atmospheric (forcing)
    'metadata': '#6A994E',       # Green for metadata
    'text': '#2D3142',           # Dark gray for text
    'light_bg': '#F8F9FA',       # Light background
    'border': '#495057',         # Border color
}


def draw_tensor_box(ax, x, y, width, height, color, label, shape_text,
                    features=None, alpha=0.3, border_width=2, n_columns=1):
    """Draw a styled tensor box with labels and multi-column feature list."""
    # Main box
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01",
        facecolor=color,
        edgecolor=COLORS['border'],
        alpha=alpha,
        linewidth=border_width,
        zorder=2
    )
    ax.add_patch(box)

    # Label at top
    ax.text(
        x + width/2, y + height - 0.02,
        label,
        fontsize=11,
        fontweight='bold',
        ha='center',
        va='top',
        color=COLORS['text'],
        zorder=3
    )

    # Shape annotation
    ax.text(
        x + width/2, y + height - 0.06,
        shape_text,
        fontsize=9,
        ha='center',
        va='top',
        color=COLORS['text'],
        family='monospace',
        zorder=3
    )

    # Feature list in columns
    if features:
        feature_y_start = y + height - 0.11
        col_width = (width - 0.02) / n_columns

        # Distribute features across columns
        features_per_col = int(np.ceil(len(features) / n_columns))

        for col_idx in range(n_columns):
            col_x = x + 0.01 + col_idx * col_width
            feature_y = feature_y_start

            start_idx = col_idx * features_per_col
            end_idx = min(start_idx + features_per_col, len(features))

            for feature in features[start_idx:end_idx]:
                ax.text(
                    col_x, feature_y,
                    feature,
                    fontsize=7,
                    ha='left',
                    va='top',
                    color=COLORS['text'],
                    zorder=3
                )
                feature_y -= 0.027


def draw_dimension_annotation(ax, x, y, width, label, offset=0.03):
    """Draw dimension annotation with arrows."""
    # Horizontal line with arrows
    ax.annotate(
        '', xy=(x + width, y - offset), xytext=(x, y - offset),
        arrowprops=dict(
            arrowstyle='<->',
            lw=1.5,
            color=COLORS['border']
        ),
        zorder=4
    )
    # Label
    ax.text(
        x + width/2, y - offset - 0.02,
        label,
        fontsize=9,
        ha='center',
        va='top',
        color=COLORS['text'],
        style='italic',
        zorder=4
    )


def create_input_tensor_diagram():
    """Create comprehensive input tensor diagram for CliffCast."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    fig.suptitle(
        'CliffCast Model Input Tensor Architecture',
        fontsize=16,
        fontweight='bold',
        y=0.98,
        color=COLORS['text']
    )

    # ========== TRANSECT DATA (Top) ==========
    transect_x, transect_y = 0.05, 0.55
    transect_w, transect_h = 0.40, 0.38

    transect_features = [
        "• distance_m: From cliff toe",
        "• elevation_m: Height MSL",
        "• slope_deg: Local slope",
        "• curvature: Profile curvature",
        "• roughness: Surface texture",
        "• intensity: LiDAR return",
        "• red, green, blue: Color",
        "• classification: Point type",
        "• return_number: Echo #",
        "• num_returns: Total echoes",
    ]

    draw_tensor_box(
        ax, transect_x, transect_y, transect_w, transect_h,
        COLORS['transect'],
        "Transect Data (Spatio-Temporal Geometry)",
        "(B, T, N, 12)  where T≈10 epochs, N=128 points",
        transect_features,
        alpha=0.25,
        n_columns=2
    )

    # Dimension annotations
    draw_dimension_annotation(
        ax, transect_x, transect_y, transect_w/3,
        "B: Batch size", offset=0.03
    )
    draw_dimension_annotation(
        ax, transect_x + transect_w/3 + 0.01, transect_y, transect_w/3,
        "T: LiDAR epochs (~10)", offset=0.03
    )
    draw_dimension_annotation(
        ax, transect_x + 2*transect_w/3 + 0.02, transect_y, transect_w/3 - 0.02,
        "N: Points (128)", offset=0.03
    )

    # Metadata box (smaller, to the right of transect)
    metadata_x = transect_x + transect_w + 0.03
    metadata_y = transect_y
    metadata_w = 0.45
    metadata_h = 0.38

    metadata_features = [
        "• cliff_height_m: Total height",
        "• mean_slope_deg: Avg slope",
        "• max_slope_deg: Max slope",
        "• toe_elevation_m: Base elev",
        "• top_elevation_m: Top elev",
        "• orientation_deg: Azimuth",
        "• transect_length_m: Length",
        "• latitude: Lat coord",
        "• longitude: Lon coord",
        "• transect_id: MOP ID",
        "• mean_intensity: Avg LiDAR",
        "• dominant_class: Main type",
    ]

    draw_tensor_box(
        ax, metadata_x, metadata_y, metadata_w, metadata_h,
        COLORS['metadata'],
        "Metadata (Broadcasted to all points)",
        "(B, T, 12)",
        metadata_features,
        alpha=0.25,
        n_columns=2
    )

    # ========== WAVE DATA (Bottom Left) ==========
    wave_x, wave_y = 0.05, 0.08
    wave_w, wave_h = 0.40, 0.42

    wave_features_basic = [
        "Basic (n=4):",
        "• hs: Wave height (m)",
        "• tp: Peak period (s)",
        "• dp: Direction (°N)",
        "• power: Flux (kW/m)",
        "",
        "Derived (n=6):",
        "• shore_normal: Impact",
        "• runup_2pct: Runup (m)",
        "",
        "Config:",
        "• 90d lookback",
        "• 6hr sampling",
        "• T_w = 360 steps",
    ]

    draw_tensor_box(
        ax, wave_x, wave_y, wave_w, wave_h,
        COLORS['wave'],
        "Wave Data (Nearshore Forcing)",
        "(B, T_w, n_features)  where T_w=360, n_features∈{4,6}",
        wave_features_basic,
        alpha=0.25,
        n_columns=2
    )

    # ========== ATMOSPHERIC DATA (Bottom Right) ==========
    atmos_x = wave_x + wave_w + 0.08
    atmos_y = wave_y
    atmos_w = 0.40
    atmos_h = 0.42

    atmos_features = [
        "Precip:",
        "• precip_mm",
        "• cumulative",
        "• intensity",
        "• api_7d, 30d",
        "• max_1d, 3d",
        "",
        "Temp:",
        "• temp_mean",
        "• temp_min",
        "• temp_max",
        "• freeze_thaw",
        "",
        "Cycles:",
        "• wet_days_7d",
        "• dry_days_7d",
        "• wet_dry_cycles",
        "",
        "Derived:",
        "• vpd",
        "• gdd",
        "• eto",
        "• (24 total)",
    ]

    draw_tensor_box(
        ax, atmos_x, atmos_y, atmos_w, atmos_h,
        COLORS['atmospheric'],
        "Atmospheric Data (Climate Forcing)",
        "(B, T_a, 24)  where T_a=90 days",
        atmos_features,
        alpha=0.25,
        n_columns=3
    )

    # ========== BATCH SIZE ANNOTATION ==========
    # Add batch size explanation
    ax.text(
        0.5, 0.02,
        "B = Batch size (typically 32)\n" +
        "All inputs aligned by transect ID and scan date",
        fontsize=9,
        ha='center',
        va='bottom',
        color=COLORS['text'],
        style='italic',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor=COLORS['light_bg'],
            edgecolor=COLORS['border'],
            linewidth=1
        )
    )

    # ========== DATA SOURCE ANNOTATIONS ==========
    # Add data sources
    sources_text = (
        "Data Sources:\n"
        "• Transects: LiDAR (2017-2025, annual)\n"
        "• Waves: CDIP MOP (100m, hourly)\n"
        "• Atmospheric: PRISM (daily, 4km)"
    )
    ax.text(
        0.98, 0.51,
        sources_text,
        fontsize=7.5,
        ha='right',
        va='top',
        color=COLORS['text'],
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor=COLORS['light_bg'],
            edgecolor=COLORS['border'],
            linewidth=1
        )
    )

    # ========== TEMPORAL ALIGNMENT NOTE ==========
    alignment_text = (
        "Temporal Alignment:\n"
        "• Transect T: Multi-epoch\n"
        "• Wave T_w: 90d @ 6hr\n"
        "• Atmos T_a: 90d @ daily"
    )
    ax.text(
        0.02, 0.51,
        alignment_text,
        fontsize=7.5,
        ha='left',
        va='top',
        color=COLORS['text'],
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor=COLORS['light_bg'],
            edgecolor=COLORS['border'],
            linewidth=1
        )
    )

    plt.tight_layout()
    return fig


def main():
    """Generate and save input tensor diagram."""
    # Create figure
    fig = create_input_tensor_diagram()

    # Ensure output directory exists
    output_dir = Path(__file__).parents[2] / 'figures' / 'appendix'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save figure
    output_path = output_dir / 'input_tensors.png'
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    print(f"✓ Saved input tensor diagram to: {output_path}")

    # Also save as PDF for publication
    output_path_pdf = output_dir / 'input_tensors.pdf'
    fig.savefig(
        output_path_pdf,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    print(f"✓ Saved PDF version to: {output_path_pdf}")

    plt.close()


if __name__ == '__main__':
    main()
