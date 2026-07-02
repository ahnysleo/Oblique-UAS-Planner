"""
Oblique UAS Highway Mapping Experiment Planning Simulator
Version 3.1 prototype

Run:
    streamlit run app.py

Purpose:
    Load a KML/KMZ centerline, generate simplified highway/corridor/ROW geometry,
    flight lines, true ground-projected camera footprints, GCP/checkpoint layouts, constraint checks,
    and export KMZ/CSV/JSON planning products for Google Earth review.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Basic camera database. Move this to camera_models.py later if desired.
# -----------------------------------------------------------------------------
CAMERA_MODELS: Dict[str, Dict[str, Dict[str, float]]] = {
    "Freefly Astro": {
        "Sony ILX-LR1 24mm": {
            "focal_length_mm": 24.0,
            "sensor_width_mm": 35.7,
            "sensor_height_mm": 23.8,
            "image_width_px": 9504,
            "image_height_px": 6336,
            "hfov_deg": 73.7,
            "vfov_deg": 53.1,
            "resolution_mp": 61.0,
        }
    },
    "Skydio X10": {
        "V100-L Wide": {
            "focal_length_mm": 0.0,
            "sensor_width_mm": 0.0,
            "sensor_height_mm": 0.0,
            "image_width_px": 8192,
            "image_height_px": 6144,
            "hfov_deg": 93.0,
            "vfov_deg": 70.0,
            "resolution_mp": 50.0,
        },
        "V100-L Narrow": {
            "focal_length_mm": 0.0,
            "sensor_width_mm": 0.0,
            "sensor_height_mm": 0.0,
            "image_width_px": 9248,
            "image_height_px": 6944,
            "hfov_deg": 50.0,
            "vfov_deg": 38.0,
            "resolution_mp": 64.0,
        },
    },
}

FT_TO_M = 0.3048
M_TO_FT = 3.280839895
EARTH_RADIUS_M = 6378137.0


@dataclass
class Scenario:
    project_name: str = "Cal Expo"
    scenario_name: str = "Scenario_001"
    description: str = ""
    output_folder: str = "outputs"
    units: str = "feet"

    roadway_width_ft: float = 150.0
    row_width_ft: float = 250.0
    corridor_widths_ft: Tuple[float, ...] = (50.0, 100.0, 150.0)
    shoulder_width_ft: float = 10.0
    median_width_ft: float = 20.0
    station_interval_ft: float = 100.0

    platform: str = "Freefly Astro"
    camera: str = "Sony ILX-LR1 24mm"
    focal_length_mm: float = 24.0
    sensor_width_mm: float = 35.7
    sensor_height_mm: float = 23.8
    image_width_px: int = 9504
    image_height_px: int = 6336
    hfov_deg: float = 73.7
    vfov_deg: float = 53.1
    oblique_look_angle_deg: float = 45.0

    flight_mode: str = "Oblique + Nadir"
    flight_side: str = "Both"
    altitude_ft: float = 200.0
    ground_height_for_projection_ft: float = 0.0
    offset_from_road_edge_ft: float = 100.0
    lines_per_side: int = 1
    cross_flight: bool = True
    cross_flight_type: str = "Single"
    cross_flight_angle_deg: float = 90.0
    forward_overlap_pct: float = 80.0
    side_overlap_pct: float = 70.0

    survey_mode: str = "Cal Expo Validation Mode"
    gcp_spacing_ft: float = 300.0
    checkpoint_spacing_ft: float = 300.0
    gcp_offset_from_road_edge_ft: float = 25.0
    checkpoint_offset_from_road_edge_ft: float = 25.0
    safety_offset_ft: float = 10.0
    target_size_ft: float = 2.0
    placement_side: str = "Both"
    include_centerline_checkpoints: bool = True
    include_edge_checkpoints: bool = True
    include_near_far_zone_checkpoints: bool = False
    include_outside_checkpoints: bool = True
    checkpoint_distribution: str = "Balanced: Centerline + Road Edges + Outside"
    minimum_checkpoint_count: int = 50

    show_footprints: bool = True
    show_image_centers: bool = True
    show_viewing_direction: bool = True
    show_camera_orientation_3d: bool = True
    coverage_target: str = "Roadway"
    minimum_required_coverage_pct: float = 100.0


# -----------------------------------------------------------------------------
# Coordinate utilities: simple local tangent approximation for planning scale.
# -----------------------------------------------------------------------------
def lonlat_to_xy(lon: float, lat: float, lon0: float, lat0: float) -> Tuple[float, float]:
    lat0_rad = math.radians(lat0)
    x_m = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(lat0_rad)
    y_m = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x_m * M_TO_FT, y_m * M_TO_FT


def xy_to_lonlat(x_ft: float, y_ft: float, lon0: float, lat0: float) -> Tuple[float, float]:
    x_m = x_ft * FT_TO_M
    y_m = y_ft * FT_TO_M
    lat = lat0 + math.degrees(y_m / EARTH_RADIUS_M)
    lon = lon0 + math.degrees(x_m / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    return lon, lat


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def polyline_length(points: List[Tuple[float, float]]) -> float:
    return sum(distance(points[i], points[i + 1]) for i in range(len(points) - 1))


def cumulative_lengths(points: List[Tuple[float, float]]) -> List[float]:
    out = [0.0]
    for i in range(len(points) - 1):
        out.append(out[-1] + distance(points[i], points[i + 1]))
    return out


def interpolate_polyline(points: List[Tuple[float, float]], d: float) -> Tuple[float, float, float]:
    """Return x, y, heading radians at distance d along line."""
    if len(points) < 2:
        raise ValueError("Need at least two points.")
    cum = cumulative_lengths(points)
    total = cum[-1]
    d = max(0.0, min(d, total))
    for i in range(len(points) - 1):
        if cum[i] <= d <= cum[i + 1] or i == len(points) - 2:
            seg_len = max(cum[i + 1] - cum[i], 1e-9)
            t = (d - cum[i]) / seg_len
            x = points[i][0] + t * (points[i + 1][0] - points[i][0])
            y = points[i][1] + t * (points[i + 1][1] - points[i][1])
            heading = math.atan2(points[i + 1][1] - points[i][1], points[i + 1][0] - points[i][0])
            return x, y, heading
    heading = math.atan2(points[-1][1] - points[-2][1], points[-1][0] - points[-2][0])
    return points[-1][0], points[-1][1], heading


def offset_polyline(points: List[Tuple[float, float]], offset_ft: float) -> List[Tuple[float, float]]:
    """Approximate offset by shifting each vertex using averaged adjacent normals."""
    if len(points) < 2:
        return points
    shifted = []
    for i, p in enumerate(points):
        normals = []
        if i > 0:
            dx, dy = p[0] - points[i - 1][0], p[1] - points[i - 1][1]
            L = math.hypot(dx, dy) or 1.0
            normals.append((-dy / L, dx / L))
        if i < len(points) - 1:
            dx, dy = points[i + 1][0] - p[0], points[i + 1][1] - p[1]
            L = math.hypot(dx, dy) or 1.0
            normals.append((-dy / L, dx / L))
        nx = sum(n[0] for n in normals)
        ny = sum(n[1] for n in normals)
        nL = math.hypot(nx, ny) or 1.0
        shifted.append((p[0] + offset_ft * nx / nL, p[1] + offset_ft * ny / nL))
    return shifted


def corridor_polygon(points: List[Tuple[float, float]], width_ft: float) -> List[Tuple[float, float]]:
    left = offset_polyline(points, width_ft / 2.0)
    right = offset_polyline(points, -width_ft / 2.0)
    return left + right[::-1] + [left[0]]


def points_at_interval(points: List[Tuple[float, float]], interval_ft: float) -> List[Tuple[float, float, float, float]]:
    total = polyline_length(points)
    interval_ft = max(interval_ft, 1.0)
    ds = [0.0]
    d = interval_ft
    while d < total:
        ds.append(d)
        d += interval_ft
    if total not in ds:
        ds.append(total)
    return [(d, *interpolate_polyline(points, d)) for d in ds]


def rectangle(center: Tuple[float, float], heading: float, length_ft: float, width_ft: float) -> List[Tuple[float, float]]:
    """Plan-view rectangle used for nadir footprints."""
    cx, cy = center
    ux, uy = math.cos(heading), math.sin(heading)
    vx, vy = -math.sin(heading), math.cos(heading)
    hl, hw = length_ft / 2.0, width_ft / 2.0
    pts = [
        (cx + ux * hl + vx * hw, cy + uy * hl + vy * hw),
        (cx - ux * hl + vx * hw, cy - uy * hl + vy * hw),
        (cx - ux * hl - vx * hw, cy - uy * hl - vy * hw),
        (cx + ux * hl - vx * hw, cy + uy * hl - vy * hw),
    ]
    return pts + [pts[0]]


def polygon_extent_along_axis(points: List[Tuple[float, float]], axis_heading: float) -> float:
    """Return projected polygon extent along a ground-plane axis."""
    ux, uy = math.cos(axis_heading), math.sin(axis_heading)
    vals = [x * ux + y * uy for x, y in points[:-1] if points]
    return max(vals) - min(vals) if vals else 0.0


def projected_oblique_footprint(
    station: Tuple[float, float],
    flight_heading: float,
    side: str,
    s: "Scenario",
) -> List[Tuple[float, float]]:
    """Project a side-looking oblique camera frame to the ground plane.

    Version 2.2 convention:
    - The uploaded centerline start-to-end direction defines flight_heading.
    - Nadir and oblique footprints use the same image-axis convention.
    - Image height / VFOV is aligned with the flight direction.
    - Image width / HFOV is aligned across the flight direction.
    - Left/right oblique cameras look laterally toward the roadway.

    This fixes the previous apparent 90-degree rotation where the oblique
    footprint used HFOV along the flight direction while the nadir footprint
    used HFOV across the flight direction.
    """
    h = max(1.0, s.altitude_ft - s.ground_height_for_projection_ft)
    beta = math.radians(max(0.0, min(85.0, s.oblique_look_angle_deg)))
    hfov2 = math.radians(max(0.1, min(170.0, s.hfov_deg)) / 2.0)
    vfov2 = math.radians(max(0.1, min(170.0, s.vfov_deg)) / 2.0)

    # If look angle is zero, oblique reduces to the nadir convention.
    if abs(beta) < 1e-9:
        width = 2.0 * h * math.tan(hfov2)
        length = 2.0 * h * math.tan(vfov2)
        return rectangle(station, flight_heading, length, width)

    # Flight axis: same as the nadir footprint length direction.
    ax, ay = math.cos(flight_heading), math.sin(flight_heading)

    # Across-flight normal. Left side flight line looks right/down toward road;
    # right side flight line looks left/down toward road.
    nx_left, ny_left = -math.sin(flight_heading), math.cos(flight_heading)
    if side == "Left":
        look_x, look_y = -nx_left, -ny_left
    elif side == "Right":
        look_x, look_y = nx_left, ny_left
    else:
        look_x, look_y = nx_left, ny_left

    sx, sy = station

    # Across-image angular offsets use HFOV. These determine near/far distance
    # across the road. Along-image half-width uses VFOV and is aligned with the
    # flight direction, matching the nadir footprint axis convention.
    def point(across_angle: float, along_angle: float) -> Tuple[float, float]:
        theta = max(math.radians(-89.0), min(math.radians(89.0), beta + across_angle))
        across_dist = h * math.tan(theta)
        along_dist = (h / max(math.cos(theta), 1e-6)) * math.tan(along_angle)
        return (
            sx + across_dist * look_x + along_dist * ax,
            sy + across_dist * look_y + along_dist * ay,
        )

    # Non-crossing polygon order: far-forward -> near-forward -> near-backward -> far-backward.
    # "Forward/backward" are along the flight direction; "near/far" are across the road.
    corners = [
        point(+hfov2, +vfov2),
        point(-hfov2, +vfov2),
        point(-hfov2, -vfov2),
        point(+hfov2, -vfov2),
    ]
    return corners + [corners[0]]


def ground_projected_footprint(
    station: Tuple[float, float],
    flight_heading: float,
    image_type: str,
    side: str,
    s: "Scenario",
) -> List[Tuple[float, float]]:
    """Create a nadir rectangle or true oblique trapezoid on the projection plane."""
    h = max(1.0, s.altitude_ft - s.ground_height_for_projection_ft)
    if image_type == "nadir":
        width = 2.0 * h * math.tan(math.radians(s.hfov_deg) / 2.0)
        length = 2.0 * h * math.tan(math.radians(s.vfov_deg) / 2.0)
        return rectangle(station, flight_heading, length, width)

    return projected_oblique_footprint(station, flight_heading, side, s)


def camera_center_3d(station: Tuple[float, float], s: "Scenario") -> Tuple[float, float, float]:
    """Camera center as (x, y, z_ft), where z is relative to local ground."""
    return (station[0], station[1], s.altitude_ft)


def footprint_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    usable = points[:-1] if len(points) > 1 and points[0] == points[-1] else points
    if not usable:
        return (0.0, 0.0)
    return (sum(p[0] for p in usable) / len(usable), sum(p[1] for p in usable) / len(usable))


def camera_orientation_3d(footprint: Dict, s: "Scenario") -> Dict[str, object]:
    """Create 3D camera center, principal viewing ray, and corner rays.

    The rays are exported to KMZ with altitudeMode=relativeToGround so Google
    Earth displays them as pyramid/frustum-like camera orientation graphics.
    Ground endpoints use the same projection height used for footprint polygons.
    """
    cx, cy = footprint_centroid(footprint["points"])
    camera_xyz = camera_center_3d((footprint["center_x"], footprint["center_y"]), s)
    ground_z = s.ground_height_for_projection_ft
    corner_rays = [
        [camera_xyz, (x, y, ground_z)]
        for x, y in footprint["points"][:-1]
    ]
    center_ray = [camera_xyz, (cx, cy, ground_z)]
    return {
        "image_id": footprint["image_id"],
        "line": footprint["line"],
        "type": footprint["type"],
        "camera_center": camera_xyz,
        "center_ray": center_ray,
        "corner_rays": corner_rays,
    }


# -----------------------------------------------------------------------------
# KML/KMZ parsing and writing.
# -----------------------------------------------------------------------------
def read_centerline_from_upload(uploaded_file) -> Tuple[List[Tuple[float, float]], str]:
    if uploaded_file is None:
        # Built-in fallback near Cal Expo, Sacramento. Replace with Path_Line.kmz for actual use.
        fallback = [
            (-121.42425, 38.59135),
            (-121.42330, 38.59192),
            (-121.42218, 38.59235),
            (-121.42095, 38.59260),
        ]
        return fallback, "Built-in sample centerline near Cal Expo"

    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    kml_text = None
    if name.endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("KMZ file does not contain a KML file.")
            kml_text = zf.read(kml_names[0]).decode("utf-8", errors="ignore")
    elif name.endswith(".kml"):
        kml_text = raw.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Please upload a .kml or .kmz file.")

    coords_blocks = re.findall(r"<LineString[\s\S]*?<coordinates>([\s\S]*?)</coordinates>", kml_text)
    if not coords_blocks:
        raise ValueError("No LineString coordinates found in the uploaded KML/KMZ.")
    coords_text = coords_blocks[0]
    lonlat = []
    for token in coords_text.split():
        parts = token.split(",")
        if len(parts) >= 2:
            lonlat.append((float(parts[0]), float(parts[1])))
    if len(lonlat) < 2:
        raise ValueError("Centerline must contain at least two vertices.")
    return lonlat, uploaded_file.name


def kml_color(hex_rgb: str, alpha: str = "cc") -> str:
    """KML color format is AABBGGRR."""
    h = hex_rgb.strip("#")
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    return f"{alpha}{bb}{gg}{rr}"


def lonlat_coords(points: Iterable[Tuple[float, float]], lon0: float, lat0: float) -> str:
    return " ".join(f"{lon:.8f},{lat:.8f},0" for lon, lat in (xy_to_lonlat(x, y, lon0, lat0) for x, y in points))


def lonlatalt_coords(points: Iterable[Tuple[float, float, float]], lon0: float, lat0: float) -> str:
    """KML coordinate string for 3D relative-to-ground points. z is in feet."""
    out = []
    for x, y, z_ft in points:
        lon, lat = xy_to_lonlat(x, y, lon0, lat0)
        out.append(f"{lon:.8f},{lat:.8f},{z_ft * FT_TO_M:.3f}")
    return " ".join(out)


def add_style(doc: ET.Element, style_id: str, line_rgb: str, poly_rgb: Optional[str] = None, width: int = 2) -> None:
    style = ET.SubElement(doc, "Style", id=style_id)
    line = ET.SubElement(style, "LineStyle")
    ET.SubElement(line, "color").text = kml_color(line_rgb, "ff")
    ET.SubElement(line, "width").text = str(width)
    if poly_rgb:
        poly = ET.SubElement(style, "PolyStyle")
        ET.SubElement(poly, "color").text = kml_color(poly_rgb, "55")


def placemark_line(folder: ET.Element, name: str, pts: List[Tuple[float, float]], lon0: float, lat0: float, style: str) -> None:
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = name
    ET.SubElement(pm, "styleUrl").text = f"#{style}"
    ls = ET.SubElement(pm, "LineString")
    ET.SubElement(ls, "tessellate").text = "1"
    ET.SubElement(ls, "coordinates").text = lonlat_coords(pts, lon0, lat0)


def placemark_polygon(folder: ET.Element, name: str, pts: List[Tuple[float, float]], lon0: float, lat0: float, style: str) -> None:
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = name
    ET.SubElement(pm, "styleUrl").text = f"#{style}"
    poly = ET.SubElement(pm, "Polygon")
    ET.SubElement(poly, "tessellate").text = "1"
    outer = ET.SubElement(poly, "outerBoundaryIs")
    ring = ET.SubElement(outer, "LinearRing")
    ET.SubElement(ring, "coordinates").text = lonlat_coords(pts, lon0, lat0)


def placemark_point(folder: ET.Element, name: str, pt: Tuple[float, float], lon0: float, lat0: float, style: str, desc: str = "") -> None:
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = name
    if desc:
        ET.SubElement(pm, "description").text = desc
    ET.SubElement(pm, "styleUrl").text = f"#{style}"
    p = ET.SubElement(pm, "Point")
    ET.SubElement(p, "coordinates").text = lonlat_coords([pt], lon0, lat0)


def placemark_point_3d(folder: ET.Element, name: str, pt: Tuple[float, float, float], lon0: float, lat0: float, style: str, desc: str = "") -> None:
    """Add a camera center point with altitude relative to the Google Earth ground."""
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = name
    if desc:
        ET.SubElement(pm, "description").text = desc
    ET.SubElement(pm, "styleUrl").text = f"#{style}"
    p = ET.SubElement(pm, "Point")
    ET.SubElement(p, "altitudeMode").text = "relativeToGround"
    ET.SubElement(p, "coordinates").text = lonlatalt_coords([pt], lon0, lat0)


def placemark_line_3d(folder: ET.Element, name: str, pts: List[Tuple[float, float, float]], lon0: float, lat0: float, style: str, desc: str = "") -> None:
    """Add a 3D camera ray, viewing axis, or frustum edge. z is in feet."""
    pm = ET.SubElement(folder, "Placemark")
    ET.SubElement(pm, "name").text = name
    if desc:
        ET.SubElement(pm, "description").text = desc
    ET.SubElement(pm, "styleUrl").text = f"#{style}"
    ls = ET.SubElement(pm, "LineString")
    ET.SubElement(ls, "tessellate").text = "0"
    ET.SubElement(ls, "altitudeMode").text = "relativeToGround"
    ET.SubElement(ls, "coordinates").text = lonlatalt_coords(pts, lon0, lat0)


# -----------------------------------------------------------------------------
# Planning engines.
# -----------------------------------------------------------------------------
def camera_footprint(s: Scenario) -> Dict[str, float]:
    """Compute footprint metrics on the selected horizontal projection plane.

    Nadir is a rectangle. Oblique uses four corner rays projected to the
    ground-height plane and therefore yields a trapezoid.
    """
    h = max(1.0, s.altitude_ft - s.ground_height_for_projection_ft)
    hfov = math.radians(s.hfov_deg)
    vfov = math.radians(s.vfov_deg)
    nadir_w = 2.0 * h * math.tan(hfov / 2.0)
    nadir_l = 2.0 * h * math.tan(vfov / 2.0)

    beta = math.radians(max(0.1, min(85.0, s.oblique_look_angle_deg)))
    # Use a station at origin looking along +Y, with cross-frame axis +X, only
    # to derive footprint dimensions. Actual KMZ footprints are generated per
    # image center with line-specific headings.
    ob_poly = projected_oblique_footprint((0.0, 0.0), 0.0, "Right", s)
    oblique_width_ft = polygon_extent_along_axis(ob_poly, 0.0)
    oblique_length_ft = polygon_extent_along_axis(ob_poly, math.pi / 2.0)

    # Near/far distances from camera nadir point on the projection plane.
    near_angle = max(math.radians(0.1), beta - vfov / 2.0)
    far_angle = min(math.radians(89.0), beta + vfov / 2.0)
    near_distance_ft = h * math.tan(near_angle)
    far_distance_ft = h * math.tan(far_angle)
    near_width_ft = 2.0 * (h / max(math.cos(near_angle), 1e-6)) * math.tan(hfov / 2.0)
    far_width_ft = 2.0 * (h / max(math.cos(far_angle), 1e-6)) * math.tan(hfov / 2.0)

    gsd_cm = (s.sensor_width_mm / max(s.image_width_px, 1)) * (h * 304.8) / max(s.focal_length_mm, 1.0) / 10.0

    return {
        "projection_height_ft": h,
        "ground_height_for_projection_ft": s.ground_height_for_projection_ft,
        "nadir_width_ft": nadir_w,
        "nadir_length_ft": nadir_l,
        "oblique_width_ft": oblique_width_ft,
        "oblique_length_ft": oblique_length_ft,
        "oblique_near_distance_ft": near_distance_ft,
        "oblique_far_distance_ft": far_distance_ft,
        "oblique_near_width_ft": near_width_ft,
        "oblique_far_width_ft": far_width_ft,
        "oblique_center_range_ft": h * math.tan(beta),
        "approx_nadir_gsd_cm": gsd_cm,
    }


def build_geometry(center_xy: List[Tuple[float, float]], s: Scenario) -> Dict:
    return {
        "centerline": center_xy,
        "roadway_polygon": corridor_polygon(center_xy, s.roadway_width_ft),
        "row_polygon": corridor_polygon(center_xy, s.row_width_ft),
        "corridor_polygons": {w: corridor_polygon(center_xy, w) for w in s.corridor_widths_ft},
        "stations": points_at_interval(center_xy, s.station_interval_ft),
        "length_ft": polyline_length(center_xy),
    }


def build_flight_plan(center_xy: List[Tuple[float, float]], s: Scenario, fp: Dict[str, float]) -> Dict:
    """Build main flight lines, one midpoint cross flight, image centers, and footprints.

    Version 1.4 behavior:
    - No footprint display interval is used.
    - A footprint is created for every planned image center.
    - Cross flight is placed once near the midpoint of the corridor.
    - Cross flight is planned as a single nadir validation pass, perpendicular to the main corridor.
    - Cross flight uses the same forward-overlap spacing rule, but its footprints are nadir rectangles.
    - Parallel-line spacing is computed from side overlap when more than one line per side is selected.
    """
    lines = []
    sides = ["Left", "Right"] if s.flight_side == "Both" else [s.flight_side]
    base_offset = s.roadway_width_ft / 2.0 + s.offset_from_road_edge_ft

    side_line_spacing_ft = None
    if s.lines_per_side > 1:
        if "Oblique" in s.flight_mode:
            side_line_spacing_ft = fp["oblique_width_ft"] * (1.0 - s.side_overlap_pct / 100.0)
        else:
            side_line_spacing_ft = fp["nadir_width_ft"] * (1.0 - s.side_overlap_pct / 100.0)
        side_line_spacing_ft = max(10.0, side_line_spacing_ft)

    if "Oblique" in s.flight_mode:
        for side in sides:
            sign = 1.0 if side == "Left" else -1.0
            for i in range(max(1, s.lines_per_side)):
                spacing = side_line_spacing_ft if side_line_spacing_ft is not None else 0.0
                off = sign * (base_offset + i * spacing)
                lines.append({
                    "name": f"Oblique {side} Line {i+1}",
                    "type": "oblique",
                    "side": side,
                    "points": offset_polyline(center_xy, off),
                })

    if "Nadir" in s.flight_mode:
        lines.append({
            "name": "Nadir Reference Line",
            "type": "nadir",
            "side": "Center",
            "points": center_xy,
        })

    cross_lines = []
    if s.cross_flight and s.cross_flight_type != "None":
        mid_d = polyline_length(center_xy) / 2.0
        x, y, heading = interpolate_polyline(center_xy, mid_d)
        cross_heading = heading + math.radians(s.cross_flight_angle_deg)
        half_len = max(s.row_width_ft / 2.0 + s.offset_from_road_edge_ft + 100.0, s.row_width_ft)
        ux, uy = math.cos(cross_heading), math.sin(cross_heading)
        cross_lines.append({
            "name": "Cross Flight 1",
            "type": "cross",
            "side": "Cross",
            "points": [(x - ux * half_len, y - uy * half_len), (x + ux * half_len, y + uy * half_len)],
        })
        if s.cross_flight_type in ("Double", "Grid"):
            cross_lines.append({
                "name": "Cross Flight 1 Reverse",
                "type": "cross",
                "side": "Cross",
                "points": [(x + ux * half_len, y + uy * half_len), (x - ux * half_len, y - uy * half_len)],
            })

    all_lines = lines + cross_lines
    image_centers = []
    footprints = []
    image_id = 1

    for line in all_lines:
        if line["type"] == "oblique":
            length, width = fp["oblique_length_ft"], fp["oblique_width_ft"]
            footprint_type = "oblique"
        else:
            # Nadir reference and cross flight both fly nadir.
            length, width = fp["nadir_length_ft"], fp["nadir_width_ft"]
            footprint_type = "nadir"

        image_spacing = max(20.0, length * (1.0 - s.forward_overlap_pct / 100.0))

        for _d, x, y, heading in points_at_interval(line["points"], image_spacing):
            image_centers.append({
                "id": image_id,
                "line": line["name"],
                "x": x,
                "y": y,
                "heading": heading,
                "type": line["type"],
            })
            footprints.append({
                "image_id": image_id,
                "line": line["name"],
                "type": footprint_type,
                "center_x": x,
                "center_y": y,
                "heading": heading,
                "points": ground_projected_footprint((x, y), heading, footprint_type, line.get("side", "Center"), s),
            })
            image_id += 1

    camera_orientations = [camera_orientation_3d(foot, s) for foot in footprints]

    return {
        "lines": lines,
        "cross_lines": cross_lines,
        "image_centers": image_centers,
        "footprints": footprints,
        "camera_orientations": camera_orientations,
        "image_spacing_ft": max(20.0, fp["oblique_length_ft"] * (1.0 - s.forward_overlap_pct / 100.0)),
        "side_line_spacing_ft": side_line_spacing_ft,
    }


def build_targets(center_xy: List[Tuple[float, float]], s: Scenario) -> Dict:
    """Generate GCPs and checkpoints.

    Version 2.0 checkpoint strategy:
    - GCPs remain outside the roadway.
    - The default checkpoint distribution is balanced among three practical
      accuracy-assessment zones: Roadway Center, Roadway Edge Zone, and Outside Roadway.
    - The default checkpoint spacing is 300 ft, but if the total number of
      checkpoints is below minimum_checkpoint_count, the app densifies the
      station interval automatically while keeping the same zone balance.
    - For both-side placement, roadway-edge and outside-roadway points alternate
      left/right by station. This keeps the counts balanced by zone while still
      distributing points across both sides of the corridor.

    Roadway center and edge-zone checkpoints can be validated using mobile
    mapping, road marking extraction, or controlled test-site access. Outside-roadway
    checkpoints are safer conventional GNSS/total station targets.
    """
    gcps: List[Dict] = []
    checkpoints: List[Dict] = []
    sides = ["Left", "Right"] if s.placement_side == "Both" else [s.placement_side]

    def add_checkpoint(x: float, y: float, side: str, station_ft: float, zone: str, source: str) -> None:
        checkpoints.append({
            "id": f"CP_{len(checkpoints)+1:03d}",
            "x": x,
            "y": y,
            "side": side,
            "station_ft": station_ft,
            "zone": zone,
            "source": source,
        })

    for d, x, y, heading in points_at_interval(center_xy, s.gcp_spacing_ft):
        nx, ny = -math.sin(heading), math.cos(heading)
        for side in sides:
            sign = 1.0 if side == "Left" else -1.0
            off = sign * (s.roadway_width_ft / 2.0 + s.gcp_offset_from_road_edge_ft + s.safety_offset_ft)
            gcps.append({
                "id": f"GCP_{len(gcps)+1:03d}",
                "x": x + nx * off,
                "y": y + ny * off,
                "side": side,
                "station_ft": d,
                "zone": "Outside Roadway",
                "source": "Survey Control",
            })

    def alternating_side(station_index: int) -> str:
        if s.placement_side != "Both":
            return s.placement_side
        return "Left" if station_index % 2 == 0 else "Right"

    def add_edge_checkpoint(x: float, y: float, nx: float, ny: float, d: float, station_index: int) -> None:
        side = alternating_side(station_index)
        sign = 1.0 if side == "Left" else -1.0
        inside_edge_off = sign * (s.roadway_width_ft / 2.0 - s.checkpoint_offset_from_road_edge_ft)
        add_checkpoint(
            x + nx * inside_edge_off,
            y + ny * inside_edge_off,
            side,
            d,
            "Roadway Edge Zone",
            "MMS or Controlled Access",
        )

    def add_outside_checkpoint(x: float, y: float, nx: float, ny: float, d: float, station_index: int) -> None:
        side = alternating_side(station_index + 1)
        sign = 1.0 if side == "Left" else -1.0
        outside_off = sign * (s.roadway_width_ft / 2.0 + s.checkpoint_offset_from_road_edge_ft + s.safety_offset_ft)
        add_checkpoint(
            x + nx * outside_off,
            y + ny * outside_off,
            side,
            d,
            "Outside Roadway",
            "GNSS/Total Station",
        )

    def add_near_far_checkpoint(x: float, y: float, nx: float, ny: float, d: float) -> None:
        for side, frac in [("Left Near/Far", 0.25), ("Right Near/Far", -0.25)]:
            off = frac * s.roadway_width_ft
            add_checkpoint(
                x + nx * off,
                y + ny * off,
                side,
                d,
                "Roadway Near/Far Zone",
                "MMS or Controlled Access",
            )

    def add_checkpoint_set(interval_ft: float) -> None:
        checkpoints.clear()
        distribution = s.checkpoint_distribution
        balanced = distribution.startswith("Balanced")

        for station_index, (d, x, y, heading) in enumerate(points_at_interval(center_xy, interval_ft)):
            nx, ny = -math.sin(heading), math.cos(heading)

            if balanced:
                # One point per zone per station: centerline, one road edge, and one outside-roadway point.
                # Edge and outside sides alternate along the alignment for left/right balance.
                add_checkpoint(x, y, "Center", d, "Roadway Center", "MMS or Controlled Access")
                add_edge_checkpoint(x, y, nx, ny, d, station_index)
                add_outside_checkpoint(x, y, nx, ny, d, station_index)
                continue

            if distribution == "Centerline only":
                add_checkpoint(x, y, "Center", d, "Roadway Center", "MMS or Controlled Access")
                continue

            if distribution == "Road edges only":
                add_edge_checkpoint(x, y, nx, ny, d, station_index)
                continue

            if distribution == "Outside roadway only":
                add_outside_checkpoint(x, y, nx, ny, d, station_index)
                continue

            # Custom mode uses the checkboxes below.
            if s.include_centerline_checkpoints:
                add_checkpoint(x, y, "Center", d, "Roadway Center", "MMS or Controlled Access")

            if s.include_edge_checkpoints:
                add_edge_checkpoint(x, y, nx, ny, d, station_index)

            if s.include_near_far_zone_checkpoints:
                add_near_far_checkpoint(x, y, nx, ny, d)

            if s.include_outside_checkpoints:
                add_outside_checkpoint(x, y, nx, ny, d, station_index)

    add_checkpoint_set(s.checkpoint_spacing_ft)

    # Ensure at least the requested number of checkpoints when geometrically possible.
    # This is done by decreasing the interval, while keeping 300 ft as the default/user value.
    effective_interval = s.checkpoint_spacing_ft
    attempts = 0
    while len(checkpoints) < s.minimum_checkpoint_count and effective_interval > 25.0 and attempts < 12:
        effective_interval = max(25.0, effective_interval * 0.75)
        add_checkpoint_set(effective_interval)
        attempts += 1

    return {"gcps": gcps, "checkpoints": checkpoints, "effective_checkpoint_spacing_ft": effective_interval}


def run_checks(s: Scenario, geometry: Dict, flight: Dict, targets: Dict, fp: Dict[str, float]) -> List[Dict[str, str]]:
    checks = []

    def add(name: str, status: str, message: str) -> None:
        checks.append({"Constraint": name, "Status": status, "Message": message})

    if geometry["length_ft"] < 300:
        add("Path length sufficient", "Warning", "Centerline is short. Consider using a longer test path for overlap and station planning.")
    else:
        add("Path length sufficient", "Pass", "Centerline length is sufficient for initial planning.")

    if flight["lines"] or flight["cross_lines"]:
        add("Flight line generated", "Pass", f"{len(flight['lines']) + len(flight['cross_lines'])} flight line(s) generated.")
    else:
        add("Flight line generated", "Error", "No flight lines were generated.")

    coverage_half = fp["nadir_width_ft"] / 2.0
    if "Oblique" in s.flight_mode:
        coverage_half = max(coverage_half, fp["oblique_width_ft"] / 2.0)
    needed_half = s.roadway_width_ft / 2.0
    if coverage_half >= needed_half * s.minimum_required_coverage_pct / 100.0:
        add("Roadway coverage", "Pass", "Estimated footprint width covers the simulated roadway width.")
    else:
        add("Roadway coverage", "Warning", "Estimated footprint width may not cover the full roadway. Increase altitude or add opposite-side/cross flights.")

    far_edge_need = s.roadway_width_ft + s.offset_from_road_edge_ft
    if "Oblique" in s.flight_mode and fp["oblique_width_ft"] >= far_edge_need:
        add("Far-side coverage", "Pass", "Simplified oblique footprint reaches the far roadway edge from at least one side.")
    elif "Oblique" in s.flight_mode:
        add("Far-side coverage", "Warning", "Far roadway edge may not be covered from a single oblique side.")
    else:
        add("Far-side coverage", "Info", "Nadir-only scenario; far-side oblique coverage is not applicable.")

    gcp_reach = s.offset_from_road_edge_ft + s.gcp_offset_from_road_edge_ft + s.safety_offset_ft
    if "Oblique" in s.flight_mode and fp["oblique_width_ft"] / 2.0 >= gcp_reach:
        add("GCP visibility", "Pass", "Current simplified footprint can likely see outside-roadway GCPs.")
    else:
        add("GCP visibility", "Warning", "GCP visibility may be insufficient. Increase altitude, reduce offset, or add nadir/cross coverage.")

    if s.forward_overlap_pct >= 75:
        add("Forward overlap", "Pass", "Forward overlap is suitable for initial SfM/photogrammetry planning.")
    else:
        add("Forward overlap", "Warning", "Forward overlap is below 75%; consider 80% or higher for corridor mapping.")

    if s.side_overlap_pct >= 60:
        add("Side overlap", "Pass", "Side overlap is suitable for initial planning.")
    else:
        add("Side overlap", "Warning", "Side overlap is below 60%; consider increasing side overlap or adding more lines.")

    if targets["gcps"]:
        add("GCP outside roadway", "Pass", f"{len(targets['gcps'])} GCPs generated outside the simulated roadway.")
    else:
        add("GCP outside roadway", "Error", "No GCPs generated.")

    center_cp_count = sum(1 for cp in targets["checkpoints"] if cp.get("zone") == "Roadway Center")
    edge_cp_count = sum(1 for cp in targets["checkpoints"] if cp.get("zone") == "Roadway Edge Zone")
    outside_cp_count = sum(1 for cp in targets["checkpoints"] if cp.get("zone") == "Outside Roadway")
    roadway_cp_count = center_cp_count + edge_cp_count
    if roadway_cp_count > 0 and outside_cp_count > 0:
        add("Mixed checkpoint layout", "Pass", f"{center_cp_count} centerline, {edge_cp_count} road-edge, and {outside_cp_count} outside-roadway checkpoints generated.")
    elif roadway_cp_count > 0:
        add("Mixed checkpoint layout", "Warning", "Roadway checkpoints generated, but no outside-roadway checkpoints were generated.")
    elif outside_cp_count > 0:
        add("Mixed checkpoint layout", "Warning", "Outside-roadway checkpoints generated, but no roadway checkpoints were generated.")
    else:
        add("Mixed checkpoint layout", "Error", "No checkpoints were generated.")

    if len(targets["checkpoints"]) >= s.minimum_checkpoint_count:
        add("Minimum checkpoint count", "Pass", f"{len(targets['checkpoints'])} checkpoints generated; minimum target is {s.minimum_checkpoint_count}.")
    else:
        add("Minimum checkpoint count", "Warning", f"Only {len(targets['checkpoints'])} checkpoints generated; target is {s.minimum_checkpoint_count}.")

    if "Nadir" in s.flight_mode:
        add("Nadir reference included", "Info", "Nadir reference line is included for baseline comparison.")
    else:
        add("Nadir reference included", "Info", "Nadir reference is not included. Add nadir flight for baseline comparison.")

    return checks


def scenario_status(checks: List[Dict[str, str]]) -> str:
    if any(c["Status"] == "Error" for c in checks):
        return "Error"
    if any(c["Status"] == "Warning" for c in checks):
        return "Warning"
    return "Ready"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_") or "Scenario"


def targets_to_df(targets: List[Dict], lon0: float, lat0: float) -> pd.DataFrame:
    rows = []
    for t in targets:
        lon, lat = xy_to_lonlat(t["x"], t["y"], lon0, lat0)
        row = dict(t)
        row.update({"longitude": lon, "latitude": lat})
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_summary_dict(s: Scenario, geometry: Dict, flight: Dict, targets: Dict, checks: List[Dict], fp: Dict[str, float]) -> Dict:
    return {
        "scenario_name": s.scenario_name,
        "project_name": s.project_name,
        "platform": s.platform,
        "camera": s.camera,
        "altitude_ft": s.altitude_ft,
        "ground_height_for_projection_ft": s.ground_height_for_projection_ft,
        "effective_projection_height_ft": round(fp.get("projection_height_ft", s.altitude_ft), 2),
        "offset_from_road_edge_ft": s.offset_from_road_edge_ft,
        "flight_mode": s.flight_mode,
        "cross_flight": s.cross_flight,
        "forward_overlap_pct": s.forward_overlap_pct,
        "side_overlap_pct": s.side_overlap_pct,
        "path_length_ft": round(geometry["length_ft"], 2),
        "gcp_count": len(targets["gcps"]),
        "checkpoint_count": len(targets["checkpoints"]),
        "centerline_checkpoint_count": sum(1 for cp in targets["checkpoints"] if cp.get("zone") == "Roadway Center"),
        "road_edge_checkpoint_count": sum(1 for cp in targets["checkpoints"] if cp.get("zone") == "Roadway Edge Zone"),
        "outside_roadway_checkpoint_count": sum(1 for cp in targets["checkpoints"] if cp.get("zone") == "Outside Roadway"),
        "checkpoint_distribution": s.checkpoint_distribution,
        "effective_checkpoint_spacing_ft": round(targets.get("effective_checkpoint_spacing_ft", s.checkpoint_spacing_ft), 2),
        "estimated_image_count": len(flight["image_centers"]),
        "estimated_footprint_count": len(flight["footprints"]),
        "estimated_camera_orientation_count": len(flight.get("camera_orientations", [])),
        "side_line_spacing_ft": None if flight.get("side_line_spacing_ft") is None else round(flight["side_line_spacing_ft"], 2),
        "nadir_width_ft": round(fp["nadir_width_ft"], 2),
        "nadir_length_ft": round(fp["nadir_length_ft"], 2),
        "oblique_width_ft": round(fp["oblique_width_ft"], 2),
        "oblique_length_ft": round(fp["oblique_length_ft"], 2),
        "approx_nadir_gsd_cm": round(fp["approx_nadir_gsd_cm"], 3),
        "status": scenario_status(checks),
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def build_kmz_bytes(s: Scenario, center_lonlat: List[Tuple[float, float]], geometry: Dict, flight: Dict, targets: Dict, checks: List[Dict], fp: Dict[str, float]) -> bytes:
    lon0, lat0 = center_lonlat[0]
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = s.scenario_name

    add_style(doc, "centerline", "ff0000", width=3)
    add_style(doc, "roadway", "666666", "999999", width=2)
    add_style(doc, "corridor", "00aa00", "00ff00", width=1)
    add_style(doc, "row", "0000ff", "0000ff", width=2)
    add_style(doc, "flight", "ff9900", width=3)
    add_style(doc, "nadir", "00ffff", width=3)
    add_style(doc, "cross", "ff00ff", width=2)
    add_style(doc, "footprint", "9900ff", "9900ff", width=1)
    add_style(doc, "camera_center", "00ff00", width=2)
    add_style(doc, "camera_ray", "00ff00", width=1)
    add_style(doc, "view_axis", "0000ff", width=2)
    add_style(doc, "gcp", "00ff00", width=2)
    add_style(doc, "checkpoint", "ff0000", width=2)

    folders = {}
    for folder_name in [
        "01 Centerline", "02 Roadway Polygon", "03 Corridor Polygons", "04 ROW Polygon", "05 Flight Lines",
        "06 Nadir Flight Lines", "07 Cross Flight Lines", "08 Camera Footprints", "09 Image Centers",
        "10 GCPs", "11 Checkpoints", "12 Camera Centers and Orientation", "13 Scenario Summary",
    ]:
        f = ET.SubElement(doc, "Folder")
        ET.SubElement(f, "name").text = folder_name
        folders[folder_name] = f

    placemark_line(folders["01 Centerline"], "Centerline", geometry["centerline"], lon0, lat0, "centerline")
    placemark_polygon(folders["02 Roadway Polygon"], f"Roadway {s.roadway_width_ft:g} ft", geometry["roadway_polygon"], lon0, lat0, "roadway")
    for w, poly in geometry["corridor_polygons"].items():
        placemark_polygon(folders["03 Corridor Polygons"], f"Corridor {w:g} ft", poly, lon0, lat0, "corridor")
    placemark_polygon(folders["04 ROW Polygon"], f"ROW {s.row_width_ft:g} ft", geometry["row_polygon"], lon0, lat0, "row")

    for line in flight["lines"]:
        folder = folders["06 Nadir Flight Lines"] if line["type"] == "nadir" else folders["05 Flight Lines"]
        style = "nadir" if line["type"] == "nadir" else "flight"
        placemark_line(folder, line["name"], line["points"], lon0, lat0, style)
    for line in flight["cross_lines"]:
        placemark_line(folders["07 Cross Flight Lines"], line["name"], line["points"], lon0, lat0, "cross")
    for i, foot in enumerate(flight["footprints"], start=1):
        placemark_polygon(folders["08 Camera Footprints"], f"Footprint {foot.get('image_id', i):04d} - {foot['line']}", foot["points"], lon0, lat0, "footprint")
    if s.show_image_centers:
        for im in flight["image_centers"]:
            placemark_point(folders["09 Image Centers"], f"IMG_{im['id']:04d}", (im["x"], im["y"]), lon0, lat0, "flight", im["line"])

    if s.show_camera_orientation_3d:
        cam_folder = folders["12 Camera Centers and Orientation"]
        for cam in flight.get("camera_orientations", []):
            img_name = f"IMG_{cam['image_id']:04d}"
            placemark_point_3d(
                cam_folder,
                f"Camera Center {img_name}",
                cam["camera_center"],
                lon0, lat0, "camera_center",
                f"{cam['line']} | {cam['type']} | altitude {s.altitude_ft:g} ft AGL",
            )
            placemark_line_3d(cam_folder, f"Viewing Axis {img_name}", cam["center_ray"], lon0, lat0, "view_axis", cam["line"])
            for j, ray in enumerate(cam["corner_rays"], start=1):
                placemark_line_3d(cam_folder, f"Corner Ray {img_name}-{j}", ray, lon0, lat0, "camera_ray", cam["line"])

    for g in targets["gcps"]:
        placemark_point(folders["10 GCPs"], g["id"], (g["x"], g["y"]), lon0, lat0, "gcp", g["zone"])
    for cp in targets["checkpoints"]:
        placemark_point(folders["11 Checkpoints"], cp["id"], (cp["x"], cp["y"]), lon0, lat0, "checkpoint", cp["zone"])

    summary = scenario_summary_dict(s, geometry, flight, targets, checks, fp)
    pm = ET.SubElement(folders["13 Scenario Summary"], "Placemark")
    ET.SubElement(pm, "name").text = "Scenario Summary"
    ET.SubElement(pm, "description").text = "\n".join(f"{k}: {v}" for k, v in summary.items())
    point = ET.SubElement(pm, "Point")
    ET.SubElement(point, "coordinates").text = f"{center_lonlat[0][0]:.8f},{center_lonlat[0][1]:.8f},0"

    kml_bytes = ET.tostring(kml, encoding="utf-8", xml_declaration=True)
    kmz_io = io.BytesIO()
    with zipfile.ZipFile(kmz_io, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_bytes)
    return kmz_io.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")



# -----------------------------------------------------------------------------
# Pre-flight accuracy assessment and PDF reporting.
# -----------------------------------------------------------------------------
def point_in_polygon(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test in local XY coordinates."""
    x, y = pt
    inside = False
    pts = poly[:-1] if len(poly) > 1 and poly[0] == poly[-1] else poly
    n = len(pts)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def distance_to_nearest(point: Tuple[float, float], targets: List[Dict]) -> float:
    if not targets:
        return float("inf")
    return min(math.hypot(point[0] - t["x"], point[1] - t["y"]) for t in targets)


def classify_line_direction(line_name: str, image_type: str) -> str:
    name = line_name.lower()
    if image_type == "nadir" and "cross" in name:
        return "Cross Nadir"
    if image_type == "nadir":
        return "Nadir"
    if "left" in name:
        return "Left Oblique"
    if "right" in name:
        return "Right Oblique"
    return "Oblique"


def preflight_accuracy_assessment(s: Scenario, geometry: Dict, flight: Dict, targets: Dict, fp: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Estimate pre-flight geometry quality and planning-level accuracy.

    This is not measured RMSE. It is a scenario screening model based on:
    GSD, footprint coverage count, viewing-direction diversity, base-to-height
    ratio, distance to GCPs, and oblique look geometry.
    """
    base_gsd_cm = fp.get("approx_nadir_gsd_cm", 2.0)
    h = max(1.0, fp.get("projection_height_ft", s.altitude_ft))
    rows = []

    for cp in targets.get("checkpoints", []):
        cp_xy = (cp["x"], cp["y"])
        covering = []
        camera_positions = []
        look_factors = []
        directions = set()

        for foot in flight.get("footprints", []):
            if point_in_polygon(cp_xy, foot["points"]):
                covering.append(foot)
                cam_xy = (foot["center_x"], foot["center_y"])
                camera_positions.append(cam_xy)
                horizontal_range = math.hypot(cp_xy[0] - cam_xy[0], cp_xy[1] - cam_xy[1])
                look_angle = math.atan2(horizontal_range, h)
                look_factors.append(min(4.0, 1.0 / max(math.cos(look_angle), 0.25)))
                directions.add(classify_line_direction(foot["line"], foot["type"]))

        image_count = len(covering)
        direction_count = len(directions)
        if look_factors:
            mean_look_factor = sum(look_factors) / len(look_factors)
        else:
            mean_look_factor = 3.0

        if len(camera_positions) >= 2:
            max_baseline = 0.0
            for i in range(len(camera_positions)):
                for j in range(i + 1, len(camera_positions)):
                    max_baseline = max(max_baseline, distance(camera_positions[i], camera_positions[j]))
            bh_ratio = max_baseline / h
        else:
            bh_ratio = 0.0

        local_gsd_cm = base_gsd_cm * mean_look_factor
        coverage_factor = 1.0 if image_count >= 5 else (1.3 if image_count >= 3 else 2.0)
        direction_factor = 1.0 if direction_count >= 2 else 1.35
        gcp_dist = distance_to_nearest(cp_xy, targets.get("gcps", []))
        gcp_factor = 1.0 + min(gcp_dist / 1000.0, 0.75)
        bh_factor = 1.0 if bh_ratio >= 0.30 else 0.30 / max(bh_ratio, 0.05)

        predicted_rmseh_cm = 1.5 * local_gsd_cm * coverage_factor * direction_factor * gcp_factor
        predicted_rmsev_cm = 2.5 * local_gsd_cm * coverage_factor * direction_factor * gcp_factor * bh_factor

        coverage_score = min(100.0, image_count / 6.0 * 100.0)
        direction_score = min(100.0, direction_count / 3.0 * 100.0)
        bh_score = min(100.0, bh_ratio / 0.35 * 100.0)
        gsd_score = max(0.0, 100.0 - max(0.0, local_gsd_cm / max(base_gsd_cm, 0.01) - 1.0) * 35.0)
        gcp_score = max(0.0, 100.0 - min(gcp_dist / 10.0, 100.0))
        quality_score = 0.25 * coverage_score + 0.20 * direction_score + 0.20 * bh_score + 0.20 * gsd_score + 0.15 * gcp_score

        warnings = []
        if image_count < 3:
            warnings.append("low image count")
        if direction_count < 2:
            warnings.append("single viewing direction")
        if bh_ratio < 0.15:
            warnings.append("weak B/H")
        if local_gsd_cm > base_gsd_cm * 1.75:
            warnings.append("large oblique GSD")
        if gcp_dist > 500:
            warnings.append("far from GCP")

        if quality_score >= 80:
            quality_class = "Good"
        elif quality_score >= 60:
            quality_class = "Moderate"
        else:
            quality_class = "Weak"

        lon, lat = xy_to_lonlat(cp["x"], cp["y"], 0, 0)  # placeholder overwritten below only if needed externally
        rows.append({
            "checkpoint_id": cp.get("id", ""),
            "zone": cp.get("zone", ""),
            "source": cp.get("source", ""),
            "station_ft": round(cp.get("station_ft", 0.0), 2),
            "image_count": image_count,
            "view_direction_count": direction_count,
            "view_directions": ", ".join(sorted(directions)) if directions else "None",
            "local_gsd_cm": round(local_gsd_cm, 3),
            "bh_ratio": round(bh_ratio, 3),
            "nearest_gcp_ft": round(gcp_dist, 1) if math.isfinite(gcp_dist) else None,
            "predicted_rmseh_cm": round(predicted_rmseh_cm, 2),
            "predicted_rmsev_cm": round(predicted_rmsev_cm, 2),
            "coverage_score": round(coverage_score, 1),
            "quality_score": round(quality_score, 1),
            "quality_class": quality_class,
            "warnings": "; ".join(warnings) if warnings else "",
            "x": cp["x"],
            "y": cp["y"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "overall_score": 0,
            "mean_rmseh_cm": None,
            "mean_rmsev_cm": None,
            "weak_count": 0,
            "moderate_count": 0,
            "good_count": 0,
            "minimum_image_count": 0,
            "average_image_count": 0,
            "status": "No checkpoints",
        }
        return df, summary

    overall_score = float(df["quality_score"].mean())
    weak_count = int((df["quality_class"] == "Weak").sum())
    moderate_count = int((df["quality_class"] == "Moderate").sum())
    good_count = int((df["quality_class"] == "Good").sum())
    status = "Good" if overall_score >= 80 and weak_count == 0 else ("Moderate" if overall_score >= 60 else "Weak")
    summary = {
        "overall_score": round(overall_score, 1),
        "mean_rmseh_cm": round(float(df["predicted_rmseh_cm"].mean()), 2),
        "mean_rmsev_cm": round(float(df["predicted_rmsev_cm"].mean()), 2),
        "max_rmseh_cm": round(float(df["predicted_rmseh_cm"].max()), 2),
        "max_rmsev_cm": round(float(df["predicted_rmsev_cm"].max()), 2),
        "weak_count": weak_count,
        "moderate_count": moderate_count,
        "good_count": good_count,
        "minimum_image_count": int(df["image_count"].min()),
        "average_image_count": round(float(df["image_count"].mean()), 1),
        "status": status,
    }
    return df, summary


def make_accuracy_map_png(s: Scenario, geometry: Dict, flight: Dict, targets: Dict, acc_df: pd.DataFrame, metric: str = "quality_score") -> bytes:
    """Create a simple plan-view map PNG without background imagery."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 6))

    def plot_poly(poly, label=None, linewidth=1.0):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        ax.plot(xs, ys, linewidth=linewidth, label=label)

    plot_poly(geometry["row_polygon"], "ROW", 1.0)
    plot_poly(geometry["roadway_polygon"], "Roadway", 1.5)
    xs = [p[0] for p in geometry["centerline"]]
    ys = [p[1] for p in geometry["centerline"]]
    ax.plot(xs, ys, linewidth=2.0, label="Centerline")

    for line in flight.get("lines", []):
        xs = [p[0] for p in line["points"]]
        ys = [p[1] for p in line["points"]]
        ax.plot(xs, ys, linewidth=1.0)
    for line in flight.get("cross_lines", []):
        xs = [p[0] for p in line["points"]]
        ys = [p[1] for p in line["points"]]
        ax.plot(xs, ys, linewidth=1.0)

    if not acc_df.empty and metric in acc_df.columns:
        x = acc_df["x"].to_numpy()
        y = acc_df["y"].to_numpy()
        val = acc_df[metric].to_numpy()
        sc = ax.scatter(x, y, c=val, s=38, edgecolors="black", linewidths=0.3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.75)
        cb.set_label(metric)

    gcp_x = [g["x"] for g in targets.get("gcps", [])]
    gcp_y = [g["y"] for g in targets.get("gcps", [])]
    if gcp_x:
        ax.scatter(gcp_x, gcp_y, marker="^", s=45, label="GCP")

    ax.set_title(f"Pre-flight Accuracy Map - {s.scenario_name}")
    ax.set_xlabel("Local X (ft)")
    ax.set_ylabel("Local Y (ft)")
    ax.axis("equal")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    return buf.getvalue()


def make_coverage_histogram_png(acc_df: pd.DataFrame) -> bytes:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    if not acc_df.empty:
        ax.hist(acc_df["image_count"], bins=range(0, int(acc_df["image_count"].max()) + 3))
    ax.set_title("Checkpoint Image Coverage Count")
    ax.set_xlabel("Number of Images Covering Checkpoint")
    ax.set_ylabel("Checkpoint Count")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    return buf.getvalue()




def make_metric_histogram_png(acc_df: pd.DataFrame, metric: str, title: str, xlabel: str) -> bytes:
    """Create a histogram for a checkpoint-level accuracy metric."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    if not acc_df.empty and metric in acc_df.columns:
        ax.hist(acc_df[metric].dropna(), bins=12)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Checkpoint Count")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    return buf.getvalue()


def make_zone_summary_table(acc_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize predicted accuracy by checkpoint zone."""
    if acc_df.empty:
        return pd.DataFrame()
    return (
        acc_df.groupby("zone", dropna=False)
        .agg(
            checkpoints=("checkpoint_id", "count"),
            mean_image_count=("image_count", "mean"),
            mean_bh_ratio=("bh_ratio", "mean"),
            mean_gsd_cm=("local_gsd_cm", "mean"),
            mean_rmseh_cm=("predicted_rmseh_cm", "mean"),
            mean_rmsev_cm=("predicted_rmsev_cm", "mean"),
            mean_score=("quality_score", "mean"),
        )
        .reset_index()
        .round(2)
    )


def make_recommendations(s: Scenario, acc_summary: Dict[str, object], acc_df: pd.DataFrame, checks: List[Dict]) -> List[Dict[str, str]]:
    """Generate simple automatic recommendations for scenario refinement."""
    recs: List[Dict[str, str]] = []
    weak_count = int(acc_summary.get("weak_count", 0) or 0)
    min_images = int(acc_summary.get("minimum_image_count", 0) or 0)
    mean_rmsev = acc_summary.get("mean_rmsev_cm")
    mean_rmseh = acc_summary.get("mean_rmseh_cm")
    mean_bh = float(acc_df["bh_ratio"].mean()) if not acc_df.empty and "bh_ratio" in acc_df else 0.0
    mean_img = float(acc_df["image_count"].mean()) if not acc_df.empty and "image_count" in acc_df else 0.0

    def add(priority: str, action: str, reason: str, expected_effect: str) -> None:
        recs.append({"priority": priority, "action": action, "reason": reason, "expected_effect": expected_effect})

    if min_images < 3:
        add("High", "Increase forward overlap or extend flight-line limits", "At least one checkpoint is covered by fewer than three images.", "Improves image redundancy and reduces weak-coverage warnings.")
    if weak_count > 0:
        add("High", "Review weak-zone checkpoints on the accuracy map", f"{weak_count} checkpoint(s) are classified as weak.", "Targets additional flight lines, GCPs, or checkpoints where they are most needed.")
    if mean_bh < 0.20:
        add("High", "Add opposite-side oblique coverage or increase baseline diversity", f"Mean B/H is low ({mean_bh:.2f}).", "Improves vertical geometry and predicted RMSEV.")
    if s.flight_side != "Both" and "Oblique" in s.flight_mode:
        add("Medium", "Add both-side oblique flights", "Single-side oblique geometry can produce weak intersection geometry across the road.", "Improves viewing-direction diversity and far-side accuracy.")
    if mean_img < 5:
        add("Medium", "Consider 85% forward overlap", f"Average checkpoint image count is {mean_img:.1f}.", "Improves redundancy, tie-point reliability, and coverage robustness.")
    if any(c.get("Status") == "Warning" and "GCP" in c.get("Constraint", "") for c in checks):
        add("Medium", "Reduce offset or increase altitude to improve GCP visibility", "Constraint checker reports possible GCP visibility limitations.", "Improves control strength and reduces extrapolation risk.")
    if isinstance(mean_rmsev, (int, float)) and isinstance(mean_rmseh, (int, float)) and mean_rmsev > 2.0 * mean_rmseh:
        add("Medium", "Strengthen vertical geometry", "Predicted vertical RMSE is much larger than horizontal RMSE.", "Improves Z reliability through better B/H and viewing diversity.")
    if not recs:
        add("Low", "Proceed with current scenario and validate in the field", "No major pre-flight warnings were detected.", "Maintains efficient acquisition while still requiring independent checkpoint validation.")
    return recs


def make_scenario_comparison_df(saved_scenarios: List[Dict[str, object]], current_summary: Dict[str, object]) -> pd.DataFrame:
    """Build a comparison table from saved scenarios plus the current scenario."""
    rows = []
    for item in saved_scenarios or []:
        rows.append(dict(item))
    rows.append(dict(current_summary))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    preferred = [
        "scenario_name", "altitude_ft", "offset_from_road_edge_ft", "oblique_look_angle_deg",
        "flight_mode", "flight_side", "forward_overlap_pct", "side_overlap_pct",
        "estimated_image_count", "gcp_count", "checkpoint_count",
        "mean_rmseh_cm", "mean_rmsev_cm", "overall_score", "preflight_status", "status",
    ]
    cols = [c for c in preferred if c in df.columns]
    return df[cols].drop_duplicates(subset=["scenario_name"], keep="last") if "scenario_name" in df.columns else df[cols]


def make_scenario_comparison_png(comparison_df: pd.DataFrame) -> bytes:
    """Create a bar chart comparing overall score and predicted RMSE across scenarios."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if not comparison_df.empty and "scenario_name" in comparison_df.columns:
        labels = comparison_df["scenario_name"].astype(str).tolist()
        x = list(range(len(labels)))
        if "overall_score" in comparison_df.columns:
            ax.bar(x, comparison_df["overall_score"].fillna(0).astype(float), label="Overall Score")
            ax.set_ylabel("Overall Score")
        if "mean_rmsev_cm" in comparison_df.columns:
            ax2 = ax.twinx()
            ax2.plot(x, comparison_df["mean_rmsev_cm"].fillna(0).astype(float), marker="o", label="Mean RMSEV")
            ax2.set_ylabel("Mean RMSEV (cm)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title("Scenario Comparison")
        ax.grid(True, axis="y", linewidth=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    return buf.getvalue()

def build_preflight_pdf_report(
    s: Scenario,
    centerline_source: str,
    geometry: Dict,
    flight: Dict,
    targets: Dict,
    checks: List[Dict],
    fp: Dict[str, float],
    acc_df: pd.DataFrame,
    acc_summary: Dict[str, object],
    saved_scenarios: Optional[List[Dict[str, object]]] = None,
) -> bytes:
    """Build a 17-part PDF report for pre-flight QA/QC and scenario comparison."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=7, leading=8))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["Normal"], fontSize=6, leading=7))
    story = []

    def heading(n: int, title: str) -> None:
        story.append(Paragraph(f"Part {n}. {title}", styles["Heading2"]))

    def paragraph(txt: str) -> None:
        story.append(Paragraph(txt, styles["BodyText"]))
        story.append(Spacer(1, 0.06 * inch))

    def simple_table(rows, widths=None, font_size=8):
        if widths is None:
            widths = [2.2 * inch, 4.5 * inch]
        table = Table(rows, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.12 * inch))

    current_summary = scenario_summary_dict(s, geometry, flight, targets, checks, fp)
    current_summary.update({
        "overall_score": acc_summary.get("overall_score"),
        "mean_rmseh_cm": acc_summary.get("mean_rmseh_cm"),
        "mean_rmsev_cm": acc_summary.get("mean_rmsev_cm"),
        "preflight_status": acc_summary.get("status"),
        "oblique_look_angle_deg": s.oblique_look_angle_deg,
        "flight_side": s.flight_side,
    })
    comparison_df = make_scenario_comparison_df(saved_scenarios or [], current_summary)
    zone_summary = make_zone_summary_table(acc_df)
    recs = make_recommendations(s, acc_summary, acc_df, checks)

    story.append(Paragraph("Oblique UAS Mission Planning and Pre-flight Accuracy Assessment Report", styles["Title"]))
    story.append(Paragraph(f"Scenario: {s.scenario_name}", styles["Heading2"]))
    story.append(Paragraph(f"Generated UTC: {datetime.utcnow().isoformat(timespec='seconds')}Z", styles["Normal"]))
    paragraph("This report is an automatic pre-flight QA/QC document. Predicted RMSE values are planning-level estimates based on camera geometry, footprint coverage, viewing diversity, B/H ratio, and GCP proximity. They are not final product accuracy results.")

    # 1. Executive Dashboard
    heading(1, "Executive Dashboard")
    exec_rows = [
        ["Metric", "Value"],
        ["Overall scenario score", str(acc_summary.get("overall_score"))],
        ["Overall rating", str(acc_summary.get("status"))],
        ["Predicted horizontal RMSE", f"{acc_summary.get('mean_rmseh_cm')} cm mean / {acc_summary.get('max_rmseh_cm')} cm max"],
        ["Predicted vertical RMSE", f"{acc_summary.get('mean_rmsev_cm')} cm mean / {acc_summary.get('max_rmsev_cm')} cm max"],
        ["Weak geometry checkpoints", str(acc_summary.get("weak_count"))],
        ["Image coverage", f"minimum {acc_summary.get('minimum_image_count')} images; average {acc_summary.get('average_image_count')} images"],
        ["Primary recommendation", recs[0]["action"] if recs else "Proceed with field validation"],
    ]
    simple_table(exec_rows)

    # 2. Project Metadata
    heading(2, "Project Metadata and Input Data")
    meta_rows = [
        ["Input", "Value"],
        ["Project", s.project_name],
        ["Scenario", s.scenario_name],
        ["Description", s.description or ""],
        ["Centerline source", centerline_source],
        ["Coordinate basis", "Local tangent plane derived from first centerline vertex"],
        ["Units", s.units],
        ["Generated date", datetime.utcnow().isoformat(timespec="seconds") + "Z"],
    ]
    simple_table(meta_rows)

    # 3. Flight Planning Summary
    heading(3, "Flight Planning Summary")
    flight_distance = sum(polyline_length(line["points"]) for line in flight.get("lines", []) + flight.get("cross_lines", []))
    flight_rows = [
        ["Parameter", "Value"],
        ["Flight mode", s.flight_mode],
        ["Flight side", s.flight_side],
        ["Main flight lines", str(len(flight.get("lines", [])))],
        ["Cross flight lines", str(len(flight.get("cross_lines", [])))],
        ["Image centers", str(len(flight.get("image_centers", [])))],
        ["Approx. flight-line distance", f"{flight_distance:.1f} ft"],
        ["Altitude AGL", f"{s.altitude_ft:g} ft"],
        ["Offset from road edge", f"{s.offset_from_road_edge_ft:g} ft"],
        ["Forward / side overlap", f"{s.forward_overlap_pct:g}% / {s.side_overlap_pct:g}%"],
        ["Side line spacing", "N/A" if flight.get("side_line_spacing_ft") is None else f"{flight['side_line_spacing_ft']:.1f} ft"],
    ]
    simple_table(flight_rows)

    # 4. Camera Geometry
    heading(4, "Camera Geometry")
    camera_rows = [
        ["Camera parameter", "Value"],
        ["Platform / Camera", f"{s.platform} / {s.camera}"],
        ["Focal length", f"{s.focal_length_mm:g} mm"],
        ["Sensor size", f"{s.sensor_width_mm:g} mm x {s.sensor_height_mm:g} mm"],
        ["Image size", f"{s.image_width_px} x {s.image_height_px} px"],
        ["HFOV / VFOV", f"{s.hfov_deg:g} deg / {s.vfov_deg:g} deg"],
        ["Oblique look angle", f"{s.oblique_look_angle_deg:g} deg"],
        ["Effective projection height", f"{fp.get('projection_height_ft', 0):.1f} ft"],
        ["Approx. nadir GSD", f"{fp.get('approx_nadir_gsd_cm', 0):.3f} cm"],
    ]
    simple_table(camera_rows)

    # 5. Footprint Analysis
    heading(5, "Footprint Analysis")
    footprint_rows = [
        ["Footprint metric", "Value"],
        ["Nadir width / length", f"{fp.get('nadir_width_ft', 0):.1f} ft / {fp.get('nadir_length_ft', 0):.1f} ft"],
        ["Oblique width / length", f"{fp.get('oblique_width_ft', 0):.1f} ft / {fp.get('oblique_length_ft', 0):.1f} ft"],
        ["Oblique near / far distance", f"{fp.get('oblique_near_distance_ft', 0):.1f} ft / {fp.get('oblique_far_distance_ft', 0):.1f} ft"],
        ["Oblique near / far width", f"{fp.get('oblique_near_width_ft', 0):.1f} ft / {fp.get('oblique_far_width_ft', 0):.1f} ft"],
        ["Exported footprints", str(len(flight.get("footprints", [])))],
    ]
    simple_table(footprint_rows)

    # 6. Coverage Heat Map
    heading(6, "Coverage Map")
    map_png = make_accuracy_map_png(s, geometry, flight, targets, acc_df, metric="image_count")
    story.append(Image(io.BytesIO(map_png), width=6.6 * inch, height=4.4 * inch))
    paragraph("The coverage map shows checkpoint image-count coverage using planned camera footprints only. No background imagery is used.")

    # 7. Viewing Geometry
    heading(7, "Viewing Geometry Analysis")
    if not acc_df.empty:
        dir_rows = [["Metric", "Value"],
                    ["Mean viewing-direction count", f"{acc_df['view_direction_count'].mean():.2f}"],
                    ["Single-direction checkpoints", str(int((acc_df['view_direction_count'] < 2).sum()))],
                    ["Common viewing directions", "; ".join(acc_df['view_directions'].value_counts().head(5).index.astype(str).tolist())]]
        simple_table(dir_rows)
    else:
        paragraph("No checkpoint viewing geometry was available.")

    # 8. GSD Analysis
    heading(8, "GSD Analysis")
    gsd_png = make_metric_histogram_png(acc_df, "local_gsd_cm", "Local GSD Distribution", "Local GSD (cm)")
    story.append(Image(io.BytesIO(gsd_png), width=5.8 * inch, height=3.3 * inch))
    if not acc_df.empty:
        gsd_rows = [["Metric", "Value"], ["Min / mean / max local GSD", f"{acc_df['local_gsd_cm'].min():.3f} / {acc_df['local_gsd_cm'].mean():.3f} / {acc_df['local_gsd_cm'].max():.3f} cm"]]
        simple_table(gsd_rows)

    # 9. Base/Height Analysis
    heading(9, "Base-to-Height Ratio Analysis")
    bh_png = make_metric_histogram_png(acc_df, "bh_ratio", "B/H Ratio Distribution", "B/H Ratio")
    story.append(Image(io.BytesIO(bh_png), width=5.8 * inch, height=3.3 * inch))
    if not acc_df.empty:
        bh_rows = [["Metric", "Value"], ["Min / mean / max B/H", f"{acc_df['bh_ratio'].min():.3f} / {acc_df['bh_ratio'].mean():.3f} / {acc_df['bh_ratio'].max():.3f}"], ["B/H < 0.15", str(int((acc_df['bh_ratio'] < 0.15).sum()))]]
        simple_table(bh_rows)

    # 10. GCP Analysis
    heading(10, "GCP Analysis")
    gcp_rows = [["Metric", "Value"], ["GCP count", str(len(targets.get("gcps", [])))], ["GCP spacing", f"{s.gcp_spacing_ft:g} ft"], ["GCP offset from road edge", f"{s.gcp_offset_from_road_edge_ft:g} ft"], ["GCP source", "Survey Control / GNSS / Total Station"]]
    if not acc_df.empty and "nearest_gcp_ft" in acc_df:
        gcp_rows += [["Nearest GCP distance: min / mean / max", f"{acc_df['nearest_gcp_ft'].min():.1f} / {acc_df['nearest_gcp_ft'].mean():.1f} / {acc_df['nearest_gcp_ft'].max():.1f} ft"]]
    simple_table(gcp_rows)

    # 11. Checkpoint Analysis
    heading(11, "Checkpoint Analysis")
    if not zone_summary.empty:
        rows = [zone_summary.columns.tolist()] + zone_summary.astype(str).values.tolist()
        simple_table(rows, widths=[1.35*inch,0.65*inch,0.75*inch,0.75*inch,0.75*inch,0.85*inch,0.85*inch,0.7*inch], font_size=6.5)
    if not acc_df.empty:
        cols = ["checkpoint_id", "zone", "image_count", "view_direction_count", "bh_ratio", "predicted_rmseh_cm", "predicted_rmsev_cm", "quality_class", "warnings"]
        sample = acc_df[cols].head(30).copy()
        rows = [cols] + sample.astype(str).values.tolist()
        simple_table(rows, widths=[0.72*inch,1.0*inch,0.55*inch,0.55*inch,0.5*inch,0.7*inch,0.7*inch,0.65*inch,1.3*inch], font_size=6.2)
        if len(acc_df) > 30:
            paragraph(f"Checkpoint table is truncated to the first 30 of {len(acc_df)} checkpoints. Use the CSV export for the complete table.")

    # 12. Predicted Accuracy Map
    heading(12, "Predicted Accuracy Maps")
    rmseh_png = make_accuracy_map_png(s, geometry, flight, targets, acc_df, metric="predicted_rmseh_cm")
    rmsev_png = make_accuracy_map_png(s, geometry, flight, targets, acc_df, metric="predicted_rmsev_cm")
    story.append(Image(io.BytesIO(rmseh_png), width=6.4 * inch, height=4.2 * inch))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Image(io.BytesIO(rmsev_png), width=6.4 * inch, height=4.2 * inch))

    # 13. Constraint Checker
    heading(13, "Constraint Checker")
    check_rows = [["Constraint", "Status", "Message"]] + [[c["Constraint"], c["Status"], c["Message"]] for c in checks]
    simple_table(check_rows, widths=[1.55*inch,0.75*inch,4.45*inch], font_size=7)

    # 14. Recommendation Engine
    heading(14, "Automatic Recommendation Engine")
    rec_rows = [["Priority", "Recommended Action", "Reason", "Expected Effect"]] + [[r["priority"], r["action"], r["reason"], r["expected_effect"]] for r in recs]
    simple_table(rec_rows, widths=[0.65*inch,1.8*inch,2.2*inch,2.1*inch], font_size=7)

    # 15. Scenario Comparison
    heading(15, "Scenario Comparison")
    if not comparison_df.empty:
        comp_png = make_scenario_comparison_png(comparison_df)
        story.append(Image(io.BytesIO(comp_png), width=6.3 * inch, height=3.3 * inch))
        comp_cols = comparison_df.columns.tolist()
        rows = [comp_cols] + comparison_df.head(12).astype(str).values.tolist()
        widths = [max(0.55, 6.8 / max(len(comp_cols), 1)) * inch for _ in comp_cols]
        simple_table(rows, widths=widths, font_size=5.7)
        paragraph("Scenario comparison includes saved scenarios from the current Streamlit session plus the current scenario. Save each scenario in Scenario Manager before generating the final comparison report.")
    else:
        paragraph("No saved scenarios are available for comparison.")

    # 16. Expected vs Measured Accuracy
    heading(16, "Expected vs. Measured Accuracy Calibration Plan")
    paragraph("After field processing, measured checkpoint residuals should be imported and compared with the pre-flight predictions. The recommended calibration workflow is: predicted RMSEH/RMSEV -> measured RMSEH/RMSEV -> calibration factors -> updated planning coefficients for future scenarios.")
    calib_rows = [["Future field", "Purpose"], ["Surveyed checkpoint coordinates", "Independent reference data"], ["UAS-derived checkpoint coordinates", "Processed product coordinates"], ["Residuals dX, dY, dZ", "Actual accuracy computation"], ["Prediction residual", "Model calibration and improvement"]]
    simple_table(calib_rows)

    # 17. Appendix
    heading(17, "Appendix: Scenario Parameters and Data Products")
    appendix_rows = [["Export", "Contents"], ["KMZ", "Centerline, roadway, ROW, flight lines, footprints, image centers, GCPs, checkpoints, camera rays"], ["Scenario JSON", "All scenario parameters and summary values"], ["GCP CSV", "GCP locations and source metadata"], ["Checkpoint CSV", "Checkpoint locations, zones, and source metadata"], ["Pre-flight Accuracy CSV", "Checkpoint-level predicted RMSE, coverage, B/H, GSD, and warnings"]]
    simple_table(appendix_rows)
    paragraph("This report is intended for design review and mission planning. Final ASPRS-style accuracy reporting requires independently surveyed checkpoints and post-processing residual statistics.")

    doc.build(story)
    return buf.getvalue()



# -----------------------------------------------------------------------------
# Version 3.1 ASPRS-informed planning metrics and HTML/PDF reporting.
# -----------------------------------------------------------------------------
def _safe_float(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def asprs_gsd_metrics(s: Scenario) -> Dict[str, float]:
    """Compute ASPRS-style nadir and oblique near/mid/far GSD metrics.

    Flat-earth planning model. DEM/height-model support can replace H per pixel later.
    GSDc is across line-of-sight. GSDl is along line-of-sight.
    """
    h_ft = max(1.0, s.altitude_ft - s.ground_height_for_projection_ft)
    h_mm = h_ft * 304.8
    f_mm = max(float(s.focal_length_mm), 1e-9)
    px_w_mm = float(s.sensor_width_mm) / max(float(s.image_width_px), 1.0)
    px_h_mm = float(s.sensor_height_mm) / max(float(s.image_height_px), 1.0)
    pixel_size_mm = (px_w_mm + px_h_mm) / 2.0 if px_h_mm > 0 else px_w_mm
    beta = math.radians(max(0.0, min(85.0, s.oblique_look_angle_deg)))
    half_hfov = math.radians(max(0.1, min(170.0, s.hfov_deg)) / 2.0)

    def gsd_for_theta(theta_rad: float) -> Tuple[float, float]:
        theta_rad = max(math.radians(0.0), min(math.radians(89.0), theta_rad))
        c = max(math.cos(theta_rad), 1e-6)
        gsdc_cm = (pixel_size_mm / f_mm) * h_mm / c / 10.0
        gsdl_cm = gsdc_cm / c
        return gsdc_cm, gsdl_cm

    nadir_c, nadir_l = gsd_for_theta(0.0)
    near_c, near_l = gsd_for_theta(max(0.0, beta - half_hfov))
    mid_c, mid_l = gsd_for_theta(beta)
    far_c, far_l = gsd_for_theta(min(math.radians(89.0), beta + half_hfov))
    values = [nadir_c, nadir_l, near_c, near_l, mid_c, mid_l, far_c, far_l]
    return {
        "pixel_size_width_um": px_w_mm * 1000.0,
        "pixel_size_height_um": px_h_mm * 1000.0,
        "pixel_size_mean_um": pixel_size_mm * 1000.0,
        "asprs_nadir_gsdc_cm": nadir_c,
        "asprs_nadir_gsdl_cm": nadir_l,
        "asprs_oblique_near_gsdc_cm": near_c,
        "asprs_oblique_near_gsdl_cm": near_l,
        "asprs_oblique_mid_gsdc_cm": mid_c,
        "asprs_oblique_mid_gsdl_cm": mid_l,
        "asprs_oblique_far_gsdc_cm": far_c,
        "asprs_oblique_far_gsdl_cm": far_l,
        "asprs_oblique_avg_gsd_cm": sum(values[2:]) / len(values[2:]),
        "asprs_project_min_gsd_cm": min(values),
        "asprs_project_max_gsd_cm": max(values),
    }


# Override previous camera_footprint with ASPRS GSD additions.
def camera_footprint(s: Scenario) -> Dict[str, float]:
    """Compute footprint metrics on flat horizontal projection plane plus ASPRS GSD metrics."""
    h = max(1.0, s.altitude_ft - s.ground_height_for_projection_ft)
    hfov = math.radians(s.hfov_deg)
    vfov = math.radians(s.vfov_deg)
    nadir_w = 2.0 * h * math.tan(hfov / 2.0)
    nadir_l = 2.0 * h * math.tan(vfov / 2.0)

    beta = math.radians(max(0.1, min(85.0, s.oblique_look_angle_deg)))
    ob_poly = projected_oblique_footprint((0.0, 0.0), 0.0, "Right", s)
    oblique_width_ft = polygon_extent_along_axis(ob_poly, 0.0)
    oblique_length_ft = polygon_extent_along_axis(ob_poly, math.pi / 2.0)

    near_angle = max(math.radians(0.1), beta - hfov / 2.0)
    far_angle = min(math.radians(89.0), beta + hfov / 2.0)
    near_distance_ft = h * math.tan(near_angle)
    far_distance_ft = h * math.tan(far_angle)
    near_width_ft = 2.0 * (h / max(math.cos(near_angle), 1e-6)) * math.tan(vfov / 2.0)
    far_width_ft = 2.0 * (h / max(math.cos(far_angle), 1e-6)) * math.tan(vfov / 2.0)

    gsd_cm = (s.sensor_width_mm / max(s.image_width_px, 1)) * (h * 304.8) / max(s.focal_length_mm, 1.0) / 10.0
    out = {
        "projection_height_ft": h,
        "ground_height_for_projection_ft": s.ground_height_for_projection_ft,
        "height_model_method": "Flat horizontal projection plane; DEM not used in this pre-flight version",
        "nadir_width_ft": nadir_w,
        "nadir_length_ft": nadir_l,
        "oblique_width_ft": oblique_width_ft,
        "oblique_length_ft": oblique_length_ft,
        "oblique_near_distance_ft": near_distance_ft,
        "oblique_far_distance_ft": far_distance_ft,
        "oblique_near_width_ft": near_width_ft,
        "oblique_far_width_ft": far_width_ft,
        "oblique_center_range_ft": h * math.tan(beta),
        "approx_nadir_gsd_cm": gsd_cm,
    }
    out.update(asprs_gsd_metrics(s))
    return out


def compute_asprs_overlap_metrics(s: Scenario, flight: Dict, fp: Dict[str, float]) -> Dict[str, object]:
    """Compute ASPRS-style planning overlap summaries from projected footprints.

    For side-looking obliques, along-track overlap is measured along image midline
    using footprint extent in flight direction. Side overlap between parallel lines
    is estimated from line spacing and near/far oblique footprint extent across the road.
    """
    rows = []
    for line in flight.get("lines", []) + flight.get("cross_lines", []):
        line_name = line.get("name", "")
        fts = [f for f in flight.get("footprints", []) if f.get("line") == line_name]
        fts = sorted(fts, key=lambda f: f.get("image_id", 0))
        vals = []
        for a, b in zip(fts[:-1], fts[1:]):
            spacing = distance((a["center_x"], a["center_y"]), (b["center_x"], b["center_y"]))
            extent_a = polygon_extent_along_axis(a["points"], a.get("heading", 0.0))
            extent_b = polygon_extent_along_axis(b["points"], b.get("heading", 0.0))
            extent = max((extent_a + extent_b) / 2.0, 1e-6)
            pct = max(0.0, min(100.0, (extent - spacing) / extent * 100.0))
            vals.append(pct)
        if vals:
            rows.append({
                "line": line_name,
                "type": line.get("type", ""),
                "asprs_overlap_basis": "midline along-track footprint overlap; trapezoid-aware planning estimate",
                "min_forward_overlap_pct": round(min(vals), 1),
                "avg_forward_overlap_pct": round(sum(vals) / len(vals), 1),
                "max_forward_overlap_pct": round(max(vals), 1),
                "image_pairs": len(vals),
            })

    side_overlap = None
    if s.lines_per_side > 1 and flight.get("side_line_spacing_ft") is not None:
        if "Oblique" in s.flight_mode:
            across_extent = max(fp.get("oblique_width_ft", 0.0), 1e-6)
            basis = "lateral oblique side overlap from far/near footprint extent between adjacent parallel lines"
        else:
            across_extent = max(fp.get("nadir_width_ft", 0.0), 1e-6)
            basis = "nadir side overlap from across-track footprint extent between adjacent parallel lines"
        pct = max(0.0, min(100.0, (across_extent - flight["side_line_spacing_ft"]) / across_extent * 100.0))
        side_overlap = {
            "side_overlap_pct_estimated": round(pct, 1),
            "side_overlap_distance_ft": round(across_extent - flight["side_line_spacing_ft"], 1),
            "line_spacing_ft": round(flight["side_line_spacing_ft"], 1),
            "basis": basis,
        }
    else:
        side_overlap = {
            "side_overlap_pct_estimated": None,
            "side_overlap_distance_ft": None,
            "line_spacing_ft": None,
            "basis": "N/A: one flight line per side",
        }

    if rows:
        all_avg = [r["avg_forward_overlap_pct"] for r in rows]
        summary = {
            "min_forward_overlap_pct": round(min(r["min_forward_overlap_pct"] for r in rows), 1),
            "avg_forward_overlap_pct": round(sum(all_avg) / len(all_avg), 1),
            "max_forward_overlap_pct": round(max(r["max_forward_overlap_pct"] for r in rows), 1),
        }
    else:
        summary = {"min_forward_overlap_pct": None, "avg_forward_overlap_pct": None, "max_forward_overlap_pct": None}
    summary.update(side_overlap)
    return {"rows": rows, "summary": summary}


def asprs_metadata_dict(s: Scenario, centerline_source: str, fp: Dict[str, float], overlap: Dict[str, object]) -> Dict[str, object]:
    """ASPRS-oriented pre-flight metadata package."""
    return {
        "standard_reference": "ASPRS Positional Accuracy Standards for Digital Geospatial Data, Edition 2 Version 2, Addendum VI; pre-flight planning subset",
        "height_model_used": fp.get("height_model_method"),
        "centerline_source": centerline_source,
        "platform": s.platform,
        "camera": s.camera,
        "principal_distance_mm": s.focal_length_mm,
        "frame_size_pixels": f"{s.image_width_px} x {s.image_height_px}",
        "sensor_size_mm": f"{s.sensor_width_mm:g} x {s.sensor_height_mm:g}",
        "pixel_size_um": f"{fp.get('pixel_size_width_um', 0):.3f} x {fp.get('pixel_size_height_um', 0):.3f}",
        "principal_point": "Unknown / camera database placeholder",
        "principal_point_reference_system": "Not specified in planning database",
        "distortion_parameters": "Unknown / not used in pre-flight geometric planning",
        "lens_field_of_view": f"HFOV {s.hfov_deg:g} deg x VFOV {s.vfov_deg:g} deg",
        "off_nadir_angle_deg": s.oblique_look_angle_deg,
        "look_angle_recommendation": "ASPRS notes typical oblique camera mounting is often 40-50 degrees; project-specific angles may be justified if accuracy is documented.",
        "image_rotation_after_collection": "No post-collection image rotation modeled",
        "image_format_compression": "Not applicable at pre-flight planning stage",
        "average_project_oblique_gsd_cm": round(fp.get("asprs_oblique_avg_gsd_cm", 0), 3),
        "project_min_gsd_cm": round(fp.get("asprs_project_min_gsd_cm", 0), 3),
        "project_max_gsd_cm": round(fp.get("asprs_project_max_gsd_cm", 0), 3),
        "forward_overlap_summary": overlap.get("summary", {}).get("avg_forward_overlap_pct"),
        "side_overlap_summary": overlap.get("summary", {}).get("side_overlap_pct_estimated"),
        "eo_file_contents_note": "Future EO export should include coordinate system, horizontal/vertical datum, units, rotation order, camera orientation, and off-nadir angle per image.",
    }


def scenario_summary_dict(s: Scenario, geometry: Dict, flight: Dict, targets: Dict, checks: List[Dict], fp: Dict[str, float]) -> Dict:
    overlap = compute_asprs_overlap_metrics(s, flight, fp)
    return {
        "scenario_name": s.scenario_name,
        "project_name": s.project_name,
        "platform": s.platform,
        "camera": s.camera,
        "altitude_ft": s.altitude_ft,
        "ground_height_for_projection_ft": s.ground_height_for_projection_ft,
        "effective_projection_height_ft": round(fp.get("projection_height_ft", s.altitude_ft), 2),
        "offset_from_road_edge_ft": s.offset_from_road_edge_ft,
        "flight_mode": s.flight_mode,
        "cross_flight": s.cross_flight,
        "forward_overlap_pct_input": s.forward_overlap_pct,
        "asprs_avg_forward_overlap_pct": overlap["summary"].get("avg_forward_overlap_pct"),
        "asprs_min_forward_overlap_pct": overlap["summary"].get("min_forward_overlap_pct"),
        "asprs_side_overlap_pct": overlap["summary"].get("side_overlap_pct_estimated"),
        "side_overlap_pct_input": s.side_overlap_pct,
        "path_length_ft": round(geometry["length_ft"], 2),
        "gcp_count": len(targets["gcps"]),
        "checkpoint_count": len(targets["checkpoints"]),
        "centerline_checkpoint_count": sum(1 for cp in targets["checkpoints"] if cp.get("zone") == "Roadway Center"),
        "road_edge_checkpoint_count": sum(1 for cp in targets["checkpoints"] if cp.get("zone") == "Roadway Edge Zone"),
        "outside_roadway_checkpoint_count": sum(1 for cp in targets["checkpoints"] if cp.get("zone") == "Outside Roadway"),
        "checkpoint_distribution": s.checkpoint_distribution,
        "effective_checkpoint_spacing_ft": round(targets.get("effective_checkpoint_spacing_ft", s.checkpoint_spacing_ft), 2),
        "estimated_image_count": len(flight["image_centers"]),
        "estimated_footprint_count": len(flight["footprints"]),
        "estimated_camera_orientation_count": len(flight.get("camera_orientations", [])),
        "side_line_spacing_ft": None if flight.get("side_line_spacing_ft") is None else round(flight["side_line_spacing_ft"], 2),
        "nadir_width_ft": round(fp["nadir_width_ft"], 2),
        "nadir_length_ft": round(fp["nadir_length_ft"], 2),
        "oblique_width_ft": round(fp["oblique_width_ft"], 2),
        "oblique_length_ft": round(fp["oblique_length_ft"], 2),
        "asprs_nadir_gsdc_cm": round(fp["asprs_nadir_gsdc_cm"], 3),
        "asprs_oblique_near_gsdc_cm": round(fp["asprs_oblique_near_gsdc_cm"], 3),
        "asprs_oblique_mid_gsdc_cm": round(fp["asprs_oblique_mid_gsdc_cm"], 3),
        "asprs_oblique_far_gsdc_cm": round(fp["asprs_oblique_far_gsdc_cm"], 3),
        "asprs_project_min_gsd_cm": round(fp["asprs_project_min_gsd_cm"], 3),
        "asprs_project_max_gsd_cm": round(fp["asprs_project_max_gsd_cm"], 3),
        "approx_nadir_gsd_cm": round(fp["approx_nadir_gsd_cm"], 3),
        "height_model_method": fp.get("height_model_method"),
        "status": scenario_status(checks),
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def build_full_html_report(
    s: Scenario,
    centerline_source: str,
    geometry: Dict,
    flight: Dict,
    targets: Dict,
    checks: List[Dict],
    fp: Dict[str, float],
    acc_df: pd.DataFrame,
    acc_summary: Dict[str, object],
    saved_scenarios: Optional[List[Dict[str, object]]] = None,
) -> bytes:
    """Build full 17-part HTML report with wide tables and embedded figures."""
    import base64
    def b64_png(png_bytes: bytes) -> str:
        return base64.b64encode(png_bytes).decode("ascii")
    def df_html(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "<p><em>No data available.</em></p>"
        return df.to_html(index=False, classes="data", border=0, escape=False)
    def kv_html(d: Dict[str, object]) -> str:
        return df_html(pd.DataFrame([{"Item": k, "Value": v} for k, v in d.items()]))
    def section(n, title, body):
        return f"<section><h2>Part {n}. {title}</h2>{body}</section>"

    overlap = compute_asprs_overlap_metrics(s, flight, fp)
    metadata = asprs_metadata_dict(s, centerline_source, fp, overlap)
    current_summary = scenario_summary_dict(s, geometry, flight, targets, checks, fp)
    current_summary.update({
        "overall_score": acc_summary.get("overall_score"),
        "mean_rmseh_cm": acc_summary.get("mean_rmseh_cm"),
        "mean_rmsev_cm": acc_summary.get("mean_rmsev_cm"),
        "preflight_status": acc_summary.get("status"),
        "oblique_look_angle_deg": s.oblique_look_angle_deg,
        "flight_side": s.flight_side,
    })
    comparison_df = make_scenario_comparison_df(saved_scenarios or [], current_summary)
    zone_summary = make_zone_summary_table(acc_df)
    recs = pd.DataFrame(make_recommendations(s, acc_summary, acc_df, checks))
    flight_distance = sum(polyline_length(line["points"]) for line in flight.get("lines", []) + flight.get("cross_lines", []))

    gsd_table = pd.DataFrame([
        {"Position": "Nadir", "GSD across LOS (cm)": fp.get("asprs_nadir_gsdc_cm"), "GSD along LOS (cm)": fp.get("asprs_nadir_gsdl_cm")},
        {"Position": "Oblique near", "GSD across LOS (cm)": fp.get("asprs_oblique_near_gsdc_cm"), "GSD along LOS (cm)": fp.get("asprs_oblique_near_gsdl_cm")},
        {"Position": "Oblique mid", "GSD across LOS (cm)": fp.get("asprs_oblique_mid_gsdc_cm"), "GSD along LOS (cm)": fp.get("asprs_oblique_mid_gsdl_cm")},
        {"Position": "Oblique far", "GSD across LOS (cm)": fp.get("asprs_oblique_far_gsdc_cm"), "GSD along LOS (cm)": fp.get("asprs_oblique_far_gsdl_cm")},
    ]).round(3)
    overlap_df = pd.DataFrame(overlap.get("rows", []))
    checks_df = pd.DataFrame(checks)

    css = """
    <style>
    body{font-family:Arial,Helvetica,sans-serif;margin:28px;color:#202124;line-height:1.35;}
    h1{font-size:28px;margin-bottom:4px;} h2{border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:30px;} h3{margin-top:18px;}
    .note{background:#f6f8fa;border-left:4px solid #4b8;padding:10px;margin:12px 0;}
    .warn{background:#fff3cd;border-left:4px solid #fa3;padding:10px;margin:12px 0;}
    table.data{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0;}
    table.data th{background:#e5e5e5;text-align:left;position:sticky;top:0;}
    table.data th,table.data td{border:1px solid #bbb;padding:5px;vertical-align:top;}
    .scroll{overflow-x:auto;border:1px solid #ddd;padding:6px;margin:8px 0;}
    img{max-width:100%;height:auto;border:1px solid #ccc;margin:8px 0;}
    .metricgrid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px;margin:10px 0;}
    .metric{background:#f8f9fa;border:1px solid #ddd;padding:10px;border-radius:6px;}
    .metric b{display:block;font-size:12px;color:#555}.metric span{font-size:22px;}
    @media print{body{margin:16px}.scroll{overflow:visible} table.data{font-size:9px} h2{page-break-before:auto}}
    </style>
    """
    metrics = f"""
    <div class='metricgrid'>
      <div class='metric'><b>Overall Score</b><span>{acc_summary.get('overall_score')}</span></div>
      <div class='metric'><b>Mean RMSEH</b><span>{acc_summary.get('mean_rmseh_cm')} cm</span></div>
      <div class='metric'><b>Mean RMSEV</b><span>{acc_summary.get('mean_rmsev_cm')} cm</span></div>
      <div class='metric'><b>Weak CPs</b><span>{acc_summary.get('weak_count')}</span></div>
    </div>
    """
    parts = []
    parts.append(section(1, "Executive Dashboard", metrics + kv_html({
        "Overall rating": acc_summary.get("status"),
        "Maximum RMSEH / RMSEV": f"{acc_summary.get('max_rmseh_cm')} cm / {acc_summary.get('max_rmsev_cm')} cm",
        "Image coverage": f"minimum {acc_summary.get('minimum_image_count')} images; average {acc_summary.get('average_image_count')} images",
        "Primary recommendation": recs.iloc[0]['action'] if not recs.empty else "Proceed with field validation",
    })))
    parts.append(section(2, "Project Metadata and Input Data", kv_html({
        "Project": s.project_name, "Scenario": s.scenario_name, "Description": s.description,
        "Centerline source": centerline_source, "Coordinate basis": "Local tangent plane from first centerline vertex",
        "Units": s.units, "Generated UTC": datetime.utcnow().isoformat(timespec="seconds") + "Z"
    })))
    parts.append(section(3, "Flight Planning Summary", kv_html({
        "Flight mode": s.flight_mode, "Flight side": s.flight_side, "Main flight lines": len(flight.get("lines", [])),
        "Cross flight lines": len(flight.get("cross_lines", [])), "Image centers": len(flight.get("image_centers", [])),
        "Approx. flight-line distance": f"{flight_distance:.1f} ft", "Altitude AGL": f"{s.altitude_ft:g} ft",
        "Offset from road edge": f"{s.offset_from_road_edge_ft:g} ft", "Input forward/side overlap": f"{s.forward_overlap_pct:g}% / {s.side_overlap_pct:g}%",
        "ASPRS estimated forward overlap avg": overlap['summary'].get('avg_forward_overlap_pct'),
        "ASPRS estimated side overlap": overlap['summary'].get('side_overlap_pct_estimated'),
        "Side overlap basis": overlap['summary'].get('basis')
    })))
    parts.append(section(4, "Camera Geometry and ASPRS Metadata", "<div class='scroll'>" + kv_html(metadata) + "</div>"))
    parts.append(section(5, "Footprint Analysis", kv_html({
        "Nadir width / length": f"{fp.get('nadir_width_ft',0):.1f} ft / {fp.get('nadir_length_ft',0):.1f} ft",
        "Oblique width / length": f"{fp.get('oblique_width_ft',0):.1f} ft / {fp.get('oblique_length_ft',0):.1f} ft",
        "Oblique near / far distance": f"{fp.get('oblique_near_distance_ft',0):.1f} ft / {fp.get('oblique_far_distance_ft',0):.1f} ft",
        "Oblique near / far width": f"{fp.get('oblique_near_width_ft',0):.1f} ft / {fp.get('oblique_far_width_ft',0):.1f} ft",
        "Height model": fp.get("height_model_method"), "Exported footprints": len(flight.get("footprints", []))
    })))
    parts.append(section(6, "Coverage Heat Map", f"<img src='data:image/png;base64,{b64_png(make_accuracy_map_png(s, geometry, flight, targets, acc_df, metric='image_count'))}'><p>Coverage is computed from planned footprint polygons under a flat-earth projection.</p>"))
    parts.append(section(7, "Viewing Geometry Analysis", ("<div class='scroll'>" + df_html(acc_df[["checkpoint_id","zone","image_count","view_direction_count","view_directions","warnings"]] if not acc_df.empty else acc_df) + "</div>")))
    parts.append(section(8, "ASPRS GSD Analysis", "<div class='scroll'>" + df_html(gsd_table) + "</div>" + f"<img src='data:image/png;base64,{b64_png(make_metric_histogram_png(acc_df,'local_gsd_cm','Local GSD Distribution','Local GSD (cm)'))}'>"))
    parts.append(section(9, "Base-to-Height Ratio Analysis", f"<img src='data:image/png;base64,{b64_png(make_metric_histogram_png(acc_df,'bh_ratio','B/H Ratio Distribution','B/H Ratio'))}'>" + (kv_html({"Min B/H": acc_df['bh_ratio'].min(), "Mean B/H": acc_df['bh_ratio'].mean(), "Max B/H": acc_df['bh_ratio'].max()}) if not acc_df.empty else "")))
    parts.append(section(10, "GCP Analysis", kv_html({
        "GCP count": len(targets.get("gcps", [])), "GCP spacing": f"{s.gcp_spacing_ft:g} ft",
        "GCP offset from road edge": f"{s.gcp_offset_from_road_edge_ft:g} ft",
        "Expected outside-roadway source": "GNSS / Total Station",
        "Nearest GCP distance mean": None if acc_df.empty else f"{acc_df['nearest_gcp_ft'].mean():.1f} ft"
    })))
    parts.append(section(11, "Checkpoint Analysis", "<h3>Zone Summary</h3><div class='scroll'>" + df_html(zone_summary) + "</div><h3>Checkpoint Screening Table</h3><div class='scroll'>" + df_html(acc_df.drop(columns=[c for c in ['x','y'] if c in acc_df.columns])) + "</div>"))
    parts.append(section(12, "Predicted Accuracy Maps", f"<h3>Predicted RMSEH</h3><img src='data:image/png;base64,{b64_png(make_accuracy_map_png(s, geometry, flight, targets, acc_df, metric='predicted_rmseh_cm'))}'><h3>Predicted RMSEV</h3><img src='data:image/png;base64,{b64_png(make_accuracy_map_png(s, geometry, flight, targets, acc_df, metric='predicted_rmsev_cm'))}'>"))
    parts.append(section(13, "Constraint Checker", "<div class='scroll'>" + df_html(checks_df) + "</div>"))
    parts.append(section(14, "Automatic Recommendation Engine", "<div class='scroll'>" + df_html(recs) + "</div>"))
    parts.append(section(15, "Scenario Comparison", f"<img src='data:image/png;base64,{b64_png(make_scenario_comparison_png(comparison_df))}'><div class='scroll'>{df_html(comparison_df)}</div><p>Save scenarios in Scenario Manager before generating the final comparison report.</p>"))
    parts.append(section(16, "Expected vs. Measured Accuracy Calibration Plan", kv_html({
        "Step 1": "Import surveyed checkpoint coordinates and UAS-derived checkpoint coordinates",
        "Step 2": "Compute residuals dX, dY, dZ and measured RMSEH/RMSEV",
        "Step 3": "Compare measured RMSE with pre-flight predicted RMSE",
        "Step 4": "Update horizontal, vertical, coverage, viewing-direction, GCP, and B/H coefficients",
    })))
    parts.append(section(17, "Appendix: Scenario Parameters and Data Products", "<h3>Scenario Summary</h3><div class='scroll'>" + kv_html(current_summary) + "</div><h3>ASPRS Overlap Details</h3><div class='scroll'>" + df_html(overlap_df) + "</div><h3>Exported Products</h3>KMZ, Scenario JSON, GCP CSV, Checkpoint CSV, Constraint CSV, Pre-flight Accuracy CSV, Full HTML Report, Executive PDF Summary."))

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{s.scenario_name} Pre-flight Accuracy Report</title>{css}</head><body>
    <h1>Oblique UAS Mission Planning and Pre-flight Accuracy Assessment Report</h1>
    <div class='note'>Full HTML report. This is the primary detailed report format. The PDF export is an executive summary to avoid wide-table layout issues.</div>
    {''.join(parts)}
    </body></html>"""
    return html.encode("utf-8")


# Override previous PDF report with compact executive summary PDF.
def build_preflight_pdf_report(
    s: Scenario,
    centerline_source: str,
    geometry: Dict,
    flight: Dict,
    targets: Dict,
    checks: List[Dict],
    fp: Dict[str, float],
    acc_df: pd.DataFrame,
    acc_summary: Dict[str, object],
    saved_scenarios: Optional[List[Dict[str, object]]] = None,
) -> bytes:
    """Build compact executive PDF summary. Full 17-part content is exported as HTML."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=0.6*inch, leftMargin=0.6*inch, topMargin=0.6*inch, bottomMargin=0.6*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=10))
    story = []
    def tbl(rows, widths=[2.4*inch,4.2*inch], fs=8):
        t=Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),fs),("VALIGN",(0,0),(-1,-1),"TOP")]))
        story.append(t); story.append(Spacer(1,0.12*inch))
    overlap = compute_asprs_overlap_metrics(s, flight, fp)
    metadata = asprs_metadata_dict(s, centerline_source, fp, overlap)
    recs = make_recommendations(s, acc_summary, acc_df, checks)
    story.append(Paragraph("Oblique UAS Pre-flight Accuracy Executive Summary", styles["Title"]))
    story.append(Paragraph(f"Scenario: {s.scenario_name}", styles["Heading2"]))
    story.append(Paragraph("Full 17-part details are provided in the HTML report. This PDF is intentionally compact for reliable printing.", styles["Small"]))
    story.append(Spacer(1,0.12*inch))
    tbl([["Metric","Value"],["Overall score",acc_summary.get("overall_score")],["Status",acc_summary.get("status")],["Mean RMSEH / RMSEV",f"{acc_summary.get('mean_rmseh_cm')} cm / {acc_summary.get('mean_rmsev_cm')} cm"],["Weak checkpoints",acc_summary.get("weak_count")],["Image coverage min/avg",f"{acc_summary.get('minimum_image_count')} / {acc_summary.get('average_image_count')}"],["Primary recommendation",recs[0]["action"] if recs else "Proceed with field validation"]])
    tbl([["ASPRS planning item","Scenario value"],["Height model",fp.get("height_model_method")],["Look angle",f"{s.oblique_look_angle_deg:g} deg"],["ASPRS GSD near/mid/far",f"{fp.get('asprs_oblique_near_gsdc_cm',0):.3f} / {fp.get('asprs_oblique_mid_gsdc_cm',0):.3f} / {fp.get('asprs_oblique_far_gsdc_cm',0):.3f} cm"],["ASPRS forward overlap avg",overlap['summary'].get('avg_forward_overlap_pct')],["ASPRS side overlap",overlap['summary'].get('side_overlap_pct_estimated')],["Metadata package", "Included in full HTML report"]])
    story.append(Paragraph("Planning Map", styles["Heading2"]))
    story.append(Image(io.BytesIO(make_accuracy_map_png(s, geometry, flight, targets, acc_df, metric="quality_score")), width=6.6*inch, height=4.4*inch))
    story.append(PageBreak())
    story.append(Paragraph("Key Recommendations", styles["Heading2"]))
    rec_rows = [["Priority","Action","Reason"]] + [[r["priority"], r["action"], r["reason"]] for r in recs[:8]]
    tbl(rec_rows, widths=[0.7*inch,2.6*inch,3.3*inch], fs=7)
    story.append(Paragraph("ASPRS Metadata Snapshot", styles["Heading2"]))
    meta_rows = [["Item","Value"]] + [[k,v] for k,v in list(metadata.items())[:14]]
    tbl(meta_rows, fs=7)
    story.append(Paragraph("Constraint Check Summary", styles["Heading2"]))
    chk_rows = [["Constraint","Status"]] + [[c["Constraint"], c["Status"]] for c in checks]
    tbl(chk_rows, widths=[4.8*inch,1.6*inch], fs=7)
    doc.build(story)
    return buf.getvalue()

# -----------------------------------------------------------------------------
# Streamlit GUI.
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Oblique UAS Planning Simulator", layout="wide")
st.title("Oblique UAS Highway Mapping Experiment Planning Simulator")
st.caption("Version 3.1 prototype — ASPRS-informed GSD/overlap/metadata, full HTML report, executive PDF summary, scenario comparison, KMZ/CSV/JSON export")

if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []

uploaded_centerline = None
with st.sidebar:
    st.header("Current Scenario")

# Tabs requested by the design document.
tabs = st.tabs([
    "Project", "Site", "UAS & Camera", "Flight Planning", "Survey Planning",
    "Footprint & Coverage", "Pre-flight Accuracy Report", "Constraint Check", "Scenario Manager", "Export"
])

with tabs[0]:
    st.subheader("Project")
    c1, c2 = st.columns(2)
    with c1:
        project_name = st.text_input("Project Name", "Cal Expo")
        scenario_name = st.text_input("Scenario Name", "Scenario_001")
        units = st.selectbox("Units", ["feet", "meters"], index=0)
    with c2:
        uploaded_centerline = st.file_uploader("Input Centerline KML/KMZ", type=["kml", "kmz"], help="Upload Path_Line.kmz or a KML containing a LineString.")
        output_folder = st.text_input("Output Folder", "outputs")
    description = st.text_area("Description", "")
    st.info("If no centerline is uploaded, the app uses a small built-in sample centerline near Cal Expo for testing. Flight direction is defined from the first centerline vertex to the last centerline vertex.")

with tabs[1]:
    st.subheader("Site")
    c1, c2, c3 = st.columns(3)
    roadway_width_ft = c1.number_input("Roadway Width (ft)", min_value=10.0, value=150.0, step=5.0)
    row_width_ft = c2.number_input("ROW Width (ft)", min_value=roadway_width_ft, value=250.0, step=10.0)
    station_interval_ft = c3.number_input("Station Interval (ft)", min_value=10.0, value=100.0, step=10.0)
    corridor_text = st.text_input("Corridor Widths (ft, comma-separated)", "50,100,150")
    corridor_widths_ft = tuple(float(v.strip()) for v in corridor_text.split(",") if v.strip())
    c4, c5 = st.columns(2)
    shoulder_width_ft = c4.number_input("Shoulder Width (ft, optional)", min_value=0.0, value=10.0, step=1.0)
    median_width_ft = c5.number_input("Median Width (ft, optional)", min_value=0.0, value=20.0, step=1.0)

with tabs[2]:
    st.subheader("UAS & Camera")
    c1, c2, c3 = st.columns(3)
    platform = c1.selectbox("Platform", list(CAMERA_MODELS.keys()), index=0)
    camera = c2.selectbox("Camera", list(CAMERA_MODELS[platform].keys()), index=0)
    cam = CAMERA_MODELS[platform][camera]
    oblique_look_angle_deg = c3.slider("Oblique Look Angle (deg)", 0, 70, 45)
    c3.caption("Default 45°. ASPRS notes typical oblique camera look angles are often 40–50°; other angles may be justified when accuracy is documented.")

    c4, c5, c6, c7 = st.columns(4)
    focal_length_mm = c4.number_input("Focal Length (mm)", value=float(cam.get("focal_length_mm", 0.0)), step=1.0)
    sensor_width_mm = c5.number_input("Sensor Width (mm)", value=float(cam.get("sensor_width_mm", 0.0)), step=0.1)
    sensor_height_mm = c6.number_input("Sensor Height (mm)", value=float(cam.get("sensor_height_mm", 0.0)), step=0.1)
    resolution_mp = c7.number_input("Resolution (MP)", value=float(cam.get("resolution_mp", 0.0)), step=1.0)

    c8, c9, c10, c11 = st.columns(4)
    image_width_px = int(c8.number_input("Image Width (px)", value=int(cam.get("image_width_px", 0)), step=100))
    image_height_px = int(c9.number_input("Image Height (px)", value=int(cam.get("image_height_px", 0)), step=100))
    hfov_deg = c10.number_input("Horizontal FOV (deg)", value=float(cam.get("hfov_deg", 73.7)), step=0.1)
    vfov_deg = c11.number_input("Vertical FOV (deg)", value=float(cam.get("vfov_deg", 53.1)), step=0.1)

with tabs[3]:
    st.subheader("Flight Planning")
    c1, c2, c3 = st.columns(3)
    flight_mode = c1.selectbox("Flight Mode", ["Oblique + Nadir", "Oblique only", "Nadir only"], index=0)
    flight_side = c2.selectbox("Flight Side", ["Both", "Left", "Right"], index=0)
    altitude_ft = c3.number_input("Flight Altitude AGL (ft)", min_value=50.0, max_value=400.0, value=200.0, step=10.0)

    ground_height_for_projection_ft = st.number_input(
        "Ground Height for Footprint Projection (ft)",
        min_value=-200.0,
        max_value=399.0,
        value=0.0,
        step=1.0,
        help="Footprints are projected to this horizontal plane. Effective projection height = Flight Altitude AGL - Ground Height.",
    )

    c4, c5, c6 = st.columns(3)
    offset_from_road_edge_ft = c4.number_input("Offset from Road Edge (ft)", min_value=0.0, value=100.0, step=5.0)
    lines_per_side = int(c5.number_input("Number of Flight Lines per Side", min_value=1, max_value=5, value=1, step=1))
    if lines_per_side == 1:
        c6.info("Side Overlap: N/A for one line per side")
    else:
        c6.info("Line spacing will be computed from side overlap")

    c7, c8 = st.columns(2)
    cross_flight_enabled = c7.checkbox("Cross Flight (Single)", value=True)
    cross_flight_type = "Single" if cross_flight_enabled else "None"
    cross_flight_angle_deg = c8.number_input("Cross Flight Angle (deg)", value=90.0, step=5.0)

    st.caption("Cross flight line will be placed near the middle of the corridor and flown perpendicular to the main flight direction.")

    st.divider()
    c9, c10 = st.columns(2)
    forward_overlap_pct = c9.slider("Forward Overlap (%)", 50, 95, 80)
    side_overlap_pct = c10.slider("Side Overlap (%)", 30, 90, 70, disabled=(lines_per_side == 1))

    st.info("All camera footprints are generated at every image center and projected to the selected ground-height plane. Flight direction follows the uploaded centerline start point to end point for nadir and oblique lines. Oblique footprints use the same image width/height axis convention as nadir footprints. Cross-flight footprints remain nadir rectangles. KMZ can also include 3D camera centers and orientation rays.")

    # Deprecated fields removed from GUI.
    cross_flight = cross_flight_enabled

with tabs[4]:
    st.subheader("Survey Planning")
    survey_mode = st.selectbox("Survey Mode", ["Cal Expo Validation Mode", "Highway Mode"], index=0)
    c1, c2, c3, c4 = st.columns(4)
    gcp_spacing_ft = c1.number_input("GCP Spacing (ft)", min_value=50.0, value=300.0, step=25.0)
    checkpoint_spacing_ft = c2.number_input("Checkpoint Spacing (ft)", min_value=25.0, value=300.0, step=25.0)
    gcp_offset_from_road_edge_ft = c3.number_input("GCP Offset from Road Edge (ft)", min_value=0.0, value=25.0, step=5.0)
    checkpoint_offset_from_road_edge_ft = c4.number_input("Checkpoint Offset from Road Edge (ft)", min_value=0.0, value=25.0, step=5.0)

    c5, c6, c7 = st.columns(3)
    safety_offset_ft = c5.number_input("Safety Offset (ft)", min_value=0.0, value=10.0, step=5.0)
    target_size_ft = c6.number_input("Target Size (ft)", min_value=0.5, value=2.0, step=0.5)
    placement_side = c7.selectbox("Placement Side", ["Both", "Left", "Right"], index=0)

    checkpoint_distribution = st.selectbox(
        "Checkpoint Distribution",
        [
            "Balanced: Centerline + Road Edges + Outside",
            "Centerline only",
            "Road edges only",
            "Outside roadway only",
            "Custom",
        ],
        index=0,
        help="Balanced mode places one centerline, one road-edge, and one outside-roadway checkpoint at each station. Edge/outside sides alternate left/right along the corridor.",
    )

    if checkpoint_distribution == "Custom":
        c8, c9, c10, c11 = st.columns(4)
        include_centerline_checkpoints = c8.checkbox("Roadway Center Checkpoints", value=True)
        include_edge_checkpoints = c9.checkbox("Roadway Edge-Zone Checkpoints", value=True)
        include_near_far_zone_checkpoints = c10.checkbox("Roadway Near/Far-Zone Checkpoints", value=False)
        include_outside_checkpoints = c11.checkbox("Outside-Roadway Checkpoints", value=True)
    else:
        include_centerline_checkpoints = checkpoint_distribution in ["Balanced: Centerline + Road Edges + Outside", "Centerline only"]
        include_edge_checkpoints = checkpoint_distribution in ["Balanced: Centerline + Road Edges + Outside", "Road edges only"]
        include_near_far_zone_checkpoints = False
        include_outside_checkpoints = checkpoint_distribution in ["Balanced: Centerline + Road Edges + Outside", "Outside roadway only"]
        st.caption("Custom zone checkboxes are hidden because a preset distribution mode is selected.")

    minimum_checkpoint_count = int(st.number_input("Minimum Total Checkpoint Count", min_value=1, value=50, step=1))
    st.info("Balanced mode distributes checkpoints among centerline, road-edge, and outside-roadway zones. Default spacing is 300 ft; if the total count is below the minimum, station spacing is automatically densified for export.")

with tabs[5]:
    st.subheader("Footprint & Coverage")
    c1, c2, c3 = st.columns(3)
    show_footprints = c1.checkbox("Show Footprints", value=True)
    show_image_centers = c2.checkbox("Show Image Centers", value=True)
    show_viewing_direction = c3.checkbox("Show Viewing Direction", value=True)
    show_camera_orientation_3d = st.checkbox("Export 3D Camera Centers & Orientation Rays", value=True)
    c4, c5 = st.columns(2)
    coverage_target = c4.selectbox("Coverage Target", ["Roadway", "ROW", "Corridor"], index=0)
    minimum_required_coverage_pct = c5.slider("Minimum Required Coverage (%)", 50, 100, 100)

# Build scenario and all derived data after collecting inputs.
s = Scenario(
    project_name=project_name, scenario_name=scenario_name, description=description, output_folder=output_folder, units=units,
    roadway_width_ft=roadway_width_ft, row_width_ft=row_width_ft, corridor_widths_ft=tuple(corridor_widths_ft),
    shoulder_width_ft=shoulder_width_ft, median_width_ft=median_width_ft, station_interval_ft=station_interval_ft,
    platform=platform, camera=camera, focal_length_mm=focal_length_mm, sensor_width_mm=sensor_width_mm,
    sensor_height_mm=sensor_height_mm, image_width_px=image_width_px, image_height_px=image_height_px,
    hfov_deg=hfov_deg, vfov_deg=vfov_deg, oblique_look_angle_deg=oblique_look_angle_deg,
    flight_mode=flight_mode, flight_side=flight_side, altitude_ft=altitude_ft,
    ground_height_for_projection_ft=ground_height_for_projection_ft,
    offset_from_road_edge_ft=offset_from_road_edge_ft, lines_per_side=lines_per_side,
    cross_flight=cross_flight,
    cross_flight_type=cross_flight_type, cross_flight_angle_deg=cross_flight_angle_deg,
    forward_overlap_pct=forward_overlap_pct, side_overlap_pct=side_overlap_pct,
    survey_mode=survey_mode,
    gcp_spacing_ft=gcp_spacing_ft, checkpoint_spacing_ft=checkpoint_spacing_ft,
    gcp_offset_from_road_edge_ft=gcp_offset_from_road_edge_ft,
    checkpoint_offset_from_road_edge_ft=checkpoint_offset_from_road_edge_ft, safety_offset_ft=safety_offset_ft,
    target_size_ft=target_size_ft, placement_side=placement_side,
    include_centerline_checkpoints=include_centerline_checkpoints,
    include_edge_checkpoints=include_edge_checkpoints,
    include_near_far_zone_checkpoints=include_near_far_zone_checkpoints,
    include_outside_checkpoints=include_outside_checkpoints,
    checkpoint_distribution=checkpoint_distribution,
    minimum_checkpoint_count=minimum_checkpoint_count,
    show_footprints=show_footprints, show_image_centers=show_image_centers,
    show_viewing_direction=show_viewing_direction, show_camera_orientation_3d=show_camera_orientation_3d, coverage_target=coverage_target,
    minimum_required_coverage_pct=minimum_required_coverage_pct,
)

try:
    center_lonlat, centerline_source = read_centerline_from_upload(uploaded_centerline)
    lon0, lat0 = center_lonlat[0]
    center_xy = [lonlat_to_xy(lon, lat, lon0, lat0) for lon, lat in center_lonlat]
    fp = camera_footprint(s)
    geometry = build_geometry(center_xy, s)
    flight = build_flight_plan(center_xy, s, fp)
    targets = build_targets(center_xy, s)
    checks = run_checks(s, geometry, flight, targets, fp)
    acc_df, acc_summary = preflight_accuracy_assessment(s, geometry, flight, targets, fp)
    status = scenario_status(checks)
except Exception as e:
    st.error(f"Could not build scenario: {e}")
    st.stop()

# Fill derived-output tabs.
with tabs[1]:
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Path Length", f"{geometry['length_ft']:.1f} ft")
    c2.metric("Roadway Width", f"{s.roadway_width_ft:.1f} ft")
    c3.metric("ROW Width", f"{s.row_width_ft:.1f} ft")
    c4.metric("Stations", len(geometry["stations"]))
    st.write(f"Centerline source: `{centerline_source}`")

with tabs[5]:
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nadir Footprint Width", f"{fp['nadir_width_ft']:.1f} ft")
    c2.metric("Nadir Footprint Length", f"{fp['nadir_length_ft']:.1f} ft")
    c3.metric("Oblique Footprint Width", f"{fp['oblique_width_ft']:.1f} ft")
    c4.metric("Effective Projection Height", f"{fp['projection_height_ft']:.1f} ft")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Oblique Near Width", f"{fp['oblique_near_width_ft']:.1f} ft")
    c6.metric("Oblique Far Width", f"{fp['oblique_far_width_ft']:.1f} ft")
    c7.metric("Near Distance", f"{fp['oblique_near_distance_ft']:.1f} ft")
    c8.metric("Far Distance", f"{fp['oblique_far_distance_ft']:.1f} ft")
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("ASPRS Nadir GSDc", f"{fp['asprs_nadir_gsdc_cm']:.3f} cm")
    c10.metric("ASPRS Near GSDc", f"{fp['asprs_oblique_near_gsdc_cm']:.3f} cm")
    c11.metric("ASPRS Mid GSDc", f"{fp['asprs_oblique_mid_gsdc_cm']:.3f} cm")
    c12.metric("ASPRS Far GSDc", f"{fp['asprs_oblique_far_gsdc_cm']:.3f} cm")
    overlap_metrics_display = compute_asprs_overlap_metrics(s, flight, fp)
    st.caption("Footprints use true frame-corner projection onto a flat horizontal planning plane. DEM/height-model support is reserved for a later version. GSD and overlap summaries follow the ASPRS Addendum VI planning concepts where applicable.")
    st.dataframe(pd.DataFrame(overlap_metrics_display.get("rows", [])), width="stretch")

with tabs[6]:
    st.subheader("Pre-flight Accuracy Report")
    st.caption("Planning-level accuracy screening based on GSD, image coverage, viewing geometry, B/H ratio, GCP proximity, and weak-zone detection. These are predicted values, not measured RMSE.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Score", f"{acc_summary.get('overall_score', 0)}")
    c2.metric("Mean RMSEH", f"{acc_summary.get('mean_rmseh_cm', 'N/A')} cm")
    c3.metric("Mean RMSEV", f"{acc_summary.get('mean_rmsev_cm', 'N/A')} cm")
    c4.metric("Weak CPs", f"{acc_summary.get('weak_count', 0)}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Min Image Count", f"{acc_summary.get('minimum_image_count', 0)}")
    c6.metric("Avg Image Count", f"{acc_summary.get('average_image_count', 0)}")
    c7.metric("Good / Moderate", f"{acc_summary.get('good_count', 0)} / {acc_summary.get('moderate_count', 0)}")
    c8.metric("Status", f"{acc_summary.get('status', 'N/A')}")

    st.markdown("### Accuracy Map")
    metric_to_map = st.selectbox("Map Metric", ["quality_score", "predicted_rmseh_cm", "predicted_rmsev_cm", "image_count", "bh_ratio", "local_gsd_cm"], index=0)
    map_png = make_accuracy_map_png(s, geometry, flight, targets, acc_df, metric=metric_to_map)
    st.image(map_png, caption=f"Plan-view accuracy map: {metric_to_map}", width="stretch")

    st.markdown("### Checkpoint Screening Table")
    display_cols = ["checkpoint_id", "zone", "source", "station_ft", "image_count", "view_direction_count", "view_directions", "local_gsd_cm", "bh_ratio", "nearest_gcp_ft", "predicted_rmseh_cm", "predicted_rmsev_cm", "quality_score", "quality_class", "warnings"]
    if not acc_df.empty:
        st.dataframe(acc_df[display_cols], width="stretch")
    else:
        st.warning("No checkpoints available for pre-flight accuracy assessment.")

    suggested_base_report = f"CalExpo_{safe_name(s.scenario_name)}_{safe_name(s.platform)}_{int(s.altitude_ft)}ft_{int(s.offset_from_road_edge_ft)}offset"
    report_pdf = build_preflight_pdf_report(s, centerline_source, geometry, flight, targets, checks, fp, acc_df, acc_summary, st.session_state.saved_scenarios)
    report_html = build_full_html_report(s, centerline_source, geometry, flight, targets, checks, fp, acc_df, acc_summary, st.session_state.saved_scenarios)
    st.markdown("### Automatic Recommendations")
    rec_df = pd.DataFrame(make_recommendations(s, acc_summary, acc_df, checks))
    st.dataframe(rec_df, width="stretch")

    st.markdown("### Scenario Comparison")
    current_comp_summary = scenario_summary_dict(s, geometry, flight, targets, checks, fp)
    current_comp_summary.update({
        "overall_score": acc_summary.get("overall_score"),
        "mean_rmseh_cm": acc_summary.get("mean_rmseh_cm"),
        "mean_rmsev_cm": acc_summary.get("mean_rmsev_cm"),
        "preflight_status": acc_summary.get("status"),
        "oblique_look_angle_deg": s.oblique_look_angle_deg,
        "flight_side": s.flight_side,
    })
    comparison_df_tab = make_scenario_comparison_df(st.session_state.saved_scenarios, current_comp_summary)
    if not comparison_df_tab.empty:
        st.dataframe(comparison_df_tab, width="stretch")
    st.caption("Save scenarios in Scenario Manager to build a multi-scenario comparison table in the PDF report.")

    st.download_button("Download Full HTML Report", data=report_html, file_name=f"{suggested_base_report}_preflight_accuracy_full_report.html", mime="text/html", key="preflight_report_html_download_tab")
    st.download_button("Download Executive PDF Summary", data=report_pdf, file_name=f"{suggested_base_report}_executive_preflight_summary.pdf", mime="application/pdf", key="preflight_report_pdf_download_tab")
    st.download_button("Download Pre-flight Accuracy CSV", data=df_to_csv_bytes(acc_df), file_name=f"{suggested_base_report}_preflight_accuracy.csv", mime="text/csv", key="preflight_report_csv_download_tab")

with tabs[7]:
    st.subheader("Constraint Check")
    st.dataframe(pd.DataFrame(checks), width="stretch")
    if status == "Ready":
        st.success("Scenario status: Ready")
    elif status == "Warning":
        st.warning("Scenario status: Warning")
    else:
        st.error("Scenario status: Error")

with tabs[8]:
    st.subheader("Scenario Manager")
    summary = scenario_summary_dict(s, geometry, flight, targets, checks, fp)
    summary.update({
        "overall_score": acc_summary.get("overall_score"),
        "mean_rmseh_cm": acc_summary.get("mean_rmseh_cm"),
        "mean_rmsev_cm": acc_summary.get("mean_rmsev_cm"),
        "max_rmseh_cm": acc_summary.get("max_rmseh_cm"),
        "max_rmsev_cm": acc_summary.get("max_rmsev_cm"),
        "weak_count": acc_summary.get("weak_count"),
        "preflight_status": acc_summary.get("status"),
        "oblique_look_angle_deg": s.oblique_look_angle_deg,
        "flight_side": s.flight_side,
    })
    csave, cclear = st.columns(2)
    if csave.button("Save Current Scenario to Session"):
        st.session_state.saved_scenarios = [r for r in st.session_state.saved_scenarios if r.get("scenario_name") != s.scenario_name]
        st.session_state.saved_scenarios.append(summary)
        st.success(f"Saved {s.scenario_name}")
    if cclear.button("Clear Saved Scenarios"):
        st.session_state.saved_scenarios = []
        st.info("Saved scenarios cleared.")
    comparison_df_ui = make_scenario_comparison_df(st.session_state.saved_scenarios, summary)
    if not comparison_df_ui.empty:
        st.markdown("### Scenario Comparison")
        st.dataframe(comparison_df_ui, width="stretch")
        st.download_button("Download Scenario Comparison CSV", data=df_to_csv_bytes(comparison_df_ui), file_name="scenario_comparison.csv", mime="text/csv", key="scenario_comparison_csv_download")
    else:
        st.info("No scenarios saved in this session yet.")

with tabs[9]:
    st.subheader("Export")
    summary = scenario_summary_dict(s, geometry, flight, targets, checks, fp)
    gcp_df = targets_to_df(targets["gcps"], lon0, lat0)
    cp_df = targets_to_df(targets["checkpoints"], lon0, lat0)
    checks_df = pd.DataFrame(checks)
    summary_df = pd.DataFrame([summary])

    suggested_base = f"CalExpo_{safe_name(s.scenario_name)}_{safe_name(s.platform)}_{int(s.altitude_ft)}ft_{int(s.offset_from_road_edge_ft)}offset"
    kmz_bytes = build_kmz_bytes(s, center_lonlat, geometry, flight, targets, checks, fp)
    scenario_json = json.dumps({**asdict(s), "summary": summary, "checks": checks}, indent=2).encode("utf-8")

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download KMZ", data=kmz_bytes, file_name=f"{suggested_base}.kmz", mime="application/vnd.google-earth.kmz", key="export_kmz_download")
    c2.download_button("Download Scenario JSON", data=scenario_json, file_name=f"{suggested_base}.json", mime="application/json", key="export_scenario_json_download")
    c3.download_button("Download Summary CSV", data=df_to_csv_bytes(summary_df), file_name=f"{suggested_base}_summary.csv", mime="text/csv", key="export_summary_csv_download")

    c4, c5, c6 = st.columns(3)
    c4.download_button("Download GCP CSV", data=df_to_csv_bytes(gcp_df), file_name=f"{suggested_base}_gcps.csv", mime="text/csv", key="export_gcp_csv_download")
    c5.download_button("Download Checkpoint CSV", data=df_to_csv_bytes(cp_df), file_name=f"{suggested_base}_checkpoints.csv", mime="text/csv", key="export_checkpoint_csv_download")
    c6.download_button("Download Constraint Check CSV", data=df_to_csv_bytes(checks_df), file_name=f"{suggested_base}_checks.csv", mime="text/csv", key="export_checks_csv_download")

    c7, c8 = st.columns(2)
    preflight_pdf = build_preflight_pdf_report(s, centerline_source, geometry, flight, targets, checks, fp, acc_df, acc_summary, st.session_state.saved_scenarios)
    preflight_html = build_full_html_report(s, centerline_source, geometry, flight, targets, checks, fp, acc_df, acc_summary, st.session_state.saved_scenarios)
    c7.download_button("Download Full HTML Report", data=preflight_html, file_name=f"{suggested_base}_preflight_accuracy_full_report.html", mime="text/html", key="export_preflight_html_download")
    c8.download_button("Download Executive PDF Summary", data=preflight_pdf, file_name=f"{suggested_base}_executive_preflight_summary.pdf", mime="application/pdf", key="export_preflight_pdf_download")
    st.download_button("Download Pre-flight Accuracy CSV", data=df_to_csv_bytes(acc_df), file_name=f"{suggested_base}_preflight_accuracy.csv", mime="text/csv", key="export_preflight_csv_download")

    st.write("Scenario summary")
    st.dataframe(summary_df, width="stretch")

# Sidebar summary and quick export.
with st.sidebar:
    st.markdown(f"**Scenario:** {s.scenario_name}")
    st.markdown(f"**Site:** {s.project_name}")
    st.markdown(f"**Platform:** {s.platform}")
    st.markdown(f"**Camera:** {s.camera}")
    st.markdown(f"**Altitude:** {s.altitude_ft:g} ft")
    st.markdown(f"**Offset:** {s.offset_from_road_edge_ft:g} ft")
    st.markdown(f"**Flight:** {s.flight_mode}")
    st.markdown(f"**Cross Flight:** {'Yes' if s.cross_flight else 'No'}")
    if flight.get("side_line_spacing_ft") is None:
        st.markdown(f"**Overlap:** {s.forward_overlap_pct:g} / N.A.")
    else:
        st.markdown(f"**Overlap:** {s.forward_overlap_pct:g} / {s.side_overlap_pct:g}")
        st.markdown(f"**Line Spacing:** {flight['side_line_spacing_ft']:.1f} ft")
    st.markdown(f"**GCPs:** {len(targets['gcps'])}")
    st.markdown(f"**Checkpoints:** {len(targets['checkpoints'])}")
    st.markdown(f"**Roadway CPs:** {sum(1 for cp in targets['checkpoints'] if str(cp.get('zone', '')).startswith('Roadway'))}")
    st.markdown(f"**Outside CPs:** {sum(1 for cp in targets['checkpoints'] if cp.get('zone') == 'Outside Roadway')}")
    st.markdown(f"**Status:** {status}")
    st.markdown(f"**Pre-flight Score:** {acc_summary.get('overall_score', 'N/A')}")
    st.divider()
    side_base = f"CalExpo_{safe_name(s.scenario_name)}_{int(s.altitude_ft)}ft_{int(s.offset_from_road_edge_ft)}offset"
    st.download_button("Generate / Download KMZ", data=build_kmz_bytes(s, center_lonlat, geometry, flight, targets, checks, fp), file_name=f"{side_base}.kmz", mime="application/vnd.google-earth.kmz", key="sidebar_kmz_download")
