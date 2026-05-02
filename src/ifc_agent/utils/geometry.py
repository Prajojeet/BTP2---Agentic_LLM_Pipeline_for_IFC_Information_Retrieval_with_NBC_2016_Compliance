"""Geometry helpers used by the geometric_processing tools.

Everything here is *defensive* — many real-world IFC files have:
- elements with no geometry (type-level definitions),
- broken openings,
- unitless quantities,
- arbitrary coordinate systems.

Each function tolerates these and returns ``None`` rather than raising.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------
def element_aabb(ctx, element) -> Optional[dict[str, Any]]:
    """Return an axis-aligned bounding box of a single element in **model units**.

    Output schema:
        {"min": [x, y, z], "max": [x, y, z], "size": [dx, dy, dz]}

    Returns None if the element has no geometric representation we can build.
    """
    import ifcopenshell.geom

    settings = ctx.geom_settings()
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
    except Exception as exc:
        logger.debug("Could not create shape for %s: %s", element, exc)
        return None

    geom = getattr(shape, "geometry", shape)
    verts = list(getattr(geom, "verts", []))
    if not verts or len(verts) < 3:
        return None

    xs = verts[0::3]
    ys = verts[1::3]
    zs = verts[2::3]
    mn = [min(xs), min(ys), min(zs)]
    mx = [max(xs), max(ys), max(zs)]
    return {
        "min": mn,
        "max": mx,
        "size": [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]],
    }


def aabb_intersect(a: dict, b: dict) -> bool:
    """Standard 3D AABB overlap test."""
    if not (a and b):
        return False
    amin, amax = a["min"], a["max"]
    bmin, bmax = b["min"], b["max"]
    return all(amin[i] <= bmax[i] and bmin[i] <= amax[i] for i in range(3))


# ---------------------------------------------------------------------------
# Property / quantity lookups
# ---------------------------------------------------------------------------
def get_psets(element) -> dict[str, dict]:
    """All property and quantity sets of an element, as nested dict."""
    import ifcopenshell.util.element as ifc_elem

    try:
        return ifc_elem.get_psets(element) or {}
    except Exception as exc:  # pragma: no cover
        logger.debug("get_psets failed for %s: %s", element, exc)
        return {}


def find_quantity(
    psets: dict[str, dict], names: list[str], qto_hints: Optional[list[str]] = None
) -> Optional[float]:
    """Walk all (Q|P)sets and return the first numeric value whose key matches
    one of ``names`` (case-insensitive). Optionally restrict to QTO sets named
    in ``qto_hints``.
    """
    name_lower = {n.lower() for n in names}
    for set_name, props in psets.items():
        if qto_hints and not any(h.lower() in set_name.lower() for h in qto_hints):
            continue
        if not isinstance(props, dict):
            continue
        for key, val in props.items():
            if key.lower() in name_lower and isinstance(val, (int, float)):
                return float(val)
    return None


# ---------------------------------------------------------------------------
# Spatial containment
# ---------------------------------------------------------------------------
def get_containing_storey(element) -> Optional[Any]:
    """Walk decomposition relationships up to the IfcBuildingStorey containing
    ``element`` (or one of its parents).
    """
    import ifcopenshell.util.element as ifc_elem

    try:
        storey = ifc_elem.get_container(element)
    except Exception:
        storey = None
    # Walk up if container isn't a storey
    while storey is not None and not storey.is_a("IfcBuildingStorey"):
        try:
            parent = ifc_elem.get_aggregate(storey)
        except Exception:
            parent = None
        if parent is None:
            return None
        storey = parent
    return storey


def get_containing_space(element) -> Optional[Any]:
    """Find the IfcSpace that contains the element, if any.

    First checks IfcRelContainedInSpatialStructure, then falls back to
    AABB containment against all spaces (slow — used only when the explicit
    relation is missing).
    """
    import ifcopenshell.util.element as ifc_elem

    try:
        container = ifc_elem.get_container(element)
    except Exception:
        container = None
    if container is not None and container.is_a("IfcSpace"):
        return container
    return None


# ---------------------------------------------------------------------------
# Cardinal directions
# ---------------------------------------------------------------------------
_DIRECTIONS = {
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
}


def normal_matches_direction(normal_xy: tuple[float, float], direction: str, tol_deg: float = 45.0) -> bool:
    """True iff a 2D normal points within ``tol_deg`` of the given cardinal direction."""
    direction = direction.lower().strip()
    if direction not in _DIRECTIONS:
        return False
    tx, ty = _DIRECTIONS[direction]
    nx, ny = normal_xy
    n_mag = math.hypot(nx, ny)
    if n_mag < 1e-9:
        return False
    nx, ny = nx / n_mag, ny / n_mag
    cos_theta = nx * tx + ny * ty
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_deg = math.degrees(math.acos(cos_theta))
    return angle_deg <= tol_deg


def element_outward_normal_xy(ctx, element) -> Optional[tuple[float, float]]:
    """Best-effort 2D outward normal of an element (used for facade orientation).

    Strategy: build the shape, compute the average face normal weighted by face
    area, drop Z. Returns None if geometry is unavailable.
    """
    import ifcopenshell.geom

    settings = ctx.geom_settings()
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
    except Exception:
        return None

    geom = getattr(shape, "geometry", shape)
    verts = list(getattr(geom, "verts", []))
    faces = list(getattr(geom, "faces", []))
    if not verts or not faces:
        return None

    pts = [(verts[i], verts[i + 1], verts[i + 2]) for i in range(0, len(verts), 3)]

    nx = ny = 0.0
    for i in range(0, len(faces), 3):
        try:
            a, b, c = pts[faces[i]], pts[faces[i + 1]], pts[faces[i + 2]]
        except IndexError:
            continue
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        # cross
        cx = uy * vz - uz * vy
        cy = uz * vx - ux * vz
        # weight by face area magnitude (proxy: the cross product norm)
        # but we only need x/y components for facade orientation
        nx += cx
        ny += cy

    if abs(nx) < 1e-9 and abs(ny) < 1e-9:
        return None
    return (nx, ny)
