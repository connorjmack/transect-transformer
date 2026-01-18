#!/usr/bin/env python
"""Create a publication-quality study site map from a shapefile."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import CRS
from shapely import affinity
from shapely.geometry import box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-quality map for a study site shapefile. "
            "Adds satellite imagery, a legend, north arrow, and scale bar."
        )
    )
    parser.add_argument(
        "shapefile",
        type=Path,
        help="Path to the shapefile (.shp) describing the study site.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/study_site_map.png"),
        help="Where to write the figure (png, pdf, etc.).",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.05,
        help="Fractional padding around the geometry bounds for plotting.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for the saved figure."
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title placed above the map.",
    )
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="Skip adding satellite imagery (requires contextily if enabled).",
    )
    return parser.parse_args()


def utm_crs_for_geometries(gdf: gpd.GeoDataFrame) -> CRS:
    """Choose a suitable projected CRS (UTM) based on the data centroid."""
    centroid_lonlat = gdf.to_crs(4326).geometry.unary_union.centroid
    lon, lat = centroid_lonlat.x, centroid_lonlat.y
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def choose_projected_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Return a projected CRS; convert geographic data to UTM."""
    if gdf.crs is None:
        raise ValueError("Input shapefile lacks a defined CRS.")
    crs = CRS(gdf.crs)
    if crs.is_geographic:
        return utm_crs_for_geometries(gdf)
    return crs


def padded_bounds(
    gdf: gpd.GeoDataFrame, buffer_fraction: float
) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    width, height = maxx - minx, maxy - miny
    xpad = width * buffer_fraction if width > 0 else 1.0
    ypad = height * buffer_fraction if height > 0 else 1.0
    return minx - xpad, maxx + xpad, miny - ypad, maxy + ypad


def rotate_bounds(
    bounds: Tuple[float, float, float, float], angle_deg: float, origin: Tuple[float, float]
) -> Tuple[float, float, float, float]:
    """Rotate rectangular bounds around an origin and return new bounds."""
    minx, maxx, miny, maxy = bounds
    rotated = affinity.rotate(box(minx, miny, maxx, maxy), angle_deg, origin=origin)
    r_minx, r_miny, r_maxx, r_maxy = rotated.bounds
    return r_minx, r_maxx, r_miny, r_maxy


def rotate_geometries(gdf: gpd.GeoDataFrame, angle_deg: float, origin: Tuple[float, float]) -> gpd.GeoDataFrame:
    """Rotate all geometries around an origin by a fixed angle (degrees)."""
    rotated = gdf.copy()
    rotated["geometry"] = rotated.geometry.apply(lambda geom: affinity.rotate(geom, angle_deg, origin=origin))
    return rotated


def add_north_arrow(ax: plt.Axes, xy=(0.88, 0.12)) -> None:
    """Draw a horizontal north arrow pointing left (north = left, south = right)."""
    ax.annotate(
        "",
        xy=(xy[0] - 0.14, xy[1]),
        xytext=xy,
        xycoords="axes fraction",
        arrowprops=dict(facecolor="black", width=4, headwidth=12),
    )
    ax.text(
        xy[0] - 0.16,
        xy[1],
        "N",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.2"),
    )
    ax.text(
        xy[0] + 0.02,
        xy[1],
        "S",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.15"),
    )


def nice_scale_length(map_width_m: float) -> float:
    """Pick a rounded scale length (in meters) based on map width."""
    target = map_width_m / 5
    candidates = [1, 2, 5]
    exponent = 0
    while target >= 10:
        target /= 10
        exponent += 1
    while target < 1:
        target *= 10
        exponent -= 1
    scaled_candidates = [c * (10**exponent) for c in candidates]
    return min(scaled_candidates, key=lambda c: abs(c - map_width_m / 5))


def add_scale_bar(ax: plt.Axes, location=(0.08, 0.04)) -> None:
    """Add a scale bar using axis limits (assumes meters)."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    map_width = x_max - x_min
    bar_length = nice_scale_length(map_width)
    bar_length = max(bar_length, 1.0)
    bar_height = (y_max - y_min) * 0.006

    x0 = x_min + map_width * location[0]
    y0 = y_min + (y_max - y_min) * location[1]

    ax.add_patch(
        Rectangle(
            (x0, y0),
            bar_length,
            bar_height,
            facecolor="black",
            edgecolor="black",
            linewidth=0.8,
        )
    )
    if bar_length >= 1000:
        label = f"{bar_length/1000:.0f} km"
    elif bar_length >= 10:
        label = f"{bar_length:.0f} m"
    else:
        label = f"{bar_length:.1f} m"
    ax.text(
        x0 + bar_length / 2,
        y0 + bar_height * 1.8,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2),
    )


def add_satellite_basemap(
    ax: plt.Axes,
    bounds: Tuple[float, float, float, float],
    rotation_center: Tuple[float, float],
    rotation_angle: float,
) -> bool:
    """Fetch and place a satellite basemap rotated to match the plot."""
    try:
        import contextily as ctx
    except ImportError:
        warnings.warn("contextily not installed; skipping satellite basemap.", stacklevel=2)
        return False

    minx, maxx, miny, maxy = bounds
    # contextily expects bounds in Web Mercator if ll=False.
    img, _ = ctx.bounds2img(
        minx, miny, maxx, maxy, source=ctx.providers.Esri.WorldImagery, ll=False
    )

    k = int(rotation_angle / 90) % 4
    if k:
        img = np.rot90(img, k=k)
        extent = rotate_bounds(bounds, rotation_angle, origin=rotation_center)
    else:
        extent = (minx, maxx, miny, maxy)

    ax.imshow(img, extent=(extent[0], extent[1], extent[2], extent[3]), origin="upper")
    return True


ROTATION_DEG = 90  # counter-clockwise; north -> left, south -> right


def plot_study_site(
    shapefile: Path,
    output: Path,
    buffer_fraction: float,
    dpi: int,
    title: str | None,
    use_basemap: bool,
) -> None:
    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        raise ValueError(f"No features found in {shapefile}")

    # Web Mercator keeps satellite tiles simple; when basemap is disabled fall back to a local projected CRS.
    target_crs = CRS.from_epsg(3857) if use_basemap else choose_projected_crs(gdf)
    gdf_proj = gdf.to_crs(target_crs)

    minx, maxx, miny, maxy = padded_bounds(gdf_proj, buffer_fraction)
    rotation_center = ((minx + maxx) / 2, (miny + maxy) / 2)

    gdf_rot = rotate_geometries(gdf_proj, ROTATION_DEG, origin=rotation_center)
    rotated_bounds = rotate_bounds((minx, maxx, miny, maxy), ROTATION_DEG, origin=rotation_center)

    fig, ax = plt.subplots(figsize=(12, 7))

    if use_basemap:
        add_satellite_basemap(ax, (minx, maxx, miny, maxy), rotation_center, ROTATION_DEG)

    geom_types = set(gdf_rot.geom_type.str.lower())
    if any("polygon" in g for g in geom_types):
        gdf_rot.plot(ax=ax, facecolor="#6aa2c9", edgecolor="#1f3c5b", linewidth=0.9, alpha=0.45)
        legend_label = "Study site extent"
        legend_handle = Patch(facecolor="#6aa2c9", edgecolor="#1f3c5b", label=legend_label)
    elif any("line" in g for g in geom_types):
        gdf_rot.plot(ax=ax, color="#1f3c5b", linewidth=1.6)
        legend_label = "Study site alignment"
        legend_handle = Line2D([0], [0], color="#1f3c5b", linewidth=1.6, label=legend_label)
    else:
        gdf_rot.plot(ax=ax, color="#1f3c5b", markersize=30, alpha=0.9)
        legend_label = "Study site points"
        legend_handle = Line2D(
            [0], [0], color="#1f3c5b", marker="o", linestyle="", markersize=8, label=legend_label
        )

    ax.set_xlim(rotated_bounds[0], rotated_bounds[1])
    ax.set_ylim(rotated_bounds[2], rotated_bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=12, pad=10)

    ax.legend(handles=[legend_handle], loc="upper right", frameon=True, framealpha=0.9)

    add_north_arrow(ax)
    add_scale_bar(ax)
    ax.set_aspect("equal")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_study_site(
        shapefile=args.shapefile,
        output=args.output,
        buffer_fraction=args.buffer,
        dpi=args.dpi,
        title=args.title,
        use_basemap=not args.no_basemap,
    )
    print(f"Saved map to {args.output}")


if __name__ == "__main__":
    main()
