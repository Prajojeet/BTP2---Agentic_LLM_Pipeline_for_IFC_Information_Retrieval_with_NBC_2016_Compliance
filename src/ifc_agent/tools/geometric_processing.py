"""Geometric Processing tools (11).

These tools operate on actual geometry: bounding boxes, intersections,
spatial containment, dimensions, accessibility checks.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..ifc_context import IFCContext
from ..utils.geometry import (
    aabb_intersect,
    element_aabb,
    find_quantity,
    get_psets,
)
from ..utils.geometry import get_containing_storey as _walk_to_storey
from ..utils.safe_serialize import truncate_list

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------
class _NoArgs(BaseModel):
    pass


class _GuidArgs(BaseModel):
    guid: str = Field(..., description="GlobalId (IfcGloballyUniqueId) of the element.")


class _TwoGuidArgs(BaseModel):
    guid_a: str = Field(..., description="GlobalId of the first element.")
    guid_b: str = Field(..., description="GlobalId of the second element.")


class _IfcClassArgs(BaseModel):
    ifc_class: str = Field(..., description="IFC class name, e.g. 'IfcDoor' or 'Door'.")


class _GuidListArgs(BaseModel):
    guids: list[str] = Field(..., description="List of element GlobalIds.")


class _StoreyPairArgs(BaseModel):
    storey_a: str = Field(..., description="Name of the first storey, e.g. 'Level 1'.")
    storey_b: str = Field(..., description="Name of the second storey, e.g. 'Level 2'.")


class _DoorAccessibilityArgs(BaseModel):
    guid: str = Field(..., description="GlobalId of the door.")
    min_clear_width_mm: float = Field(
        850.0,
        description="Minimum clear-opening width in millimetres. Default 850 mm (common accessibility threshold).",
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_geometric_processing_tools(ctx: IFCContext) -> list:
    """Build the 11 Geometric-Processing tools as StructuredTool instances."""

    scale = ctx.unit_scale_to_meters()

    # --- 1. get_element_bounding_box ---------------------------------------
    def get_element_bounding_box(guid: str) -> dict:
        """AABB of a single element in metres."""
        try:
            element = ctx.by_guid(guid)
        except ValueError as exc:
            return {"error": str(exc)}
        bbox = element_aabb(ctx, element)
        if bbox is None:
            return {"GlobalId": guid, "bounding_box": None, "note": "Element has no usable geometry."}
        # Convert to metres
        out = {
            "GlobalId": guid,
            "bounding_box_m": {
                "min": [v * scale for v in bbox["min"]],
                "max": [v * scale for v in bbox["max"]],
                "size": [v * scale for v in bbox["size"]],
            },
        }
        return out

    # --- 2. bounding_box_intersect -----------------------------------------
    def bounding_box_intersect(guid_a: str, guid_b: str) -> dict:
        """Test whether two elements' AABBs overlap."""
        try:
            ea = ctx.by_guid(guid_a)
            eb = ctx.by_guid(guid_b)
        except ValueError as exc:
            return {"error": str(exc)}
        ba = element_aabb(ctx, ea)
        bb = element_aabb(ctx, eb)
        if ba is None or bb is None:
            return {"intersect": False, "note": "One or both elements lack geometry."}
        return {"intersect": aabb_intersect(ba, bb)}

    # --- 3. get_containing_rooms_for_entities_type -------------------------
    def get_containing_rooms_for_entities_type(ifc_class: str) -> dict:
        """For every element of the given class, which IfcSpace contains it?"""
        return _containment_by_guids(
            ctx, [e.GlobalId for e in ctx.by_type(ifc_class)], ifc_class=ifc_class
        )

    # --- 4. get_containing_rooms_for_entity_guids --------------------------
    def get_containing_rooms_for_entity_guids(guids: list[str]) -> dict:
        """For every GlobalId in the list, which IfcSpace contains it?"""
        return _containment_by_guids(ctx, guids)

    # --- 5. get_containing_storey ------------------------------------------
    def get_containing_storey(guid: str) -> dict:
        """Return the IfcBuildingStorey containing the given element."""
        try:
            e = ctx.by_guid(guid)
        except ValueError as exc:
            return {"error": str(exc)}
        storey = _walk_to_storey(e)
        if storey is None:
            return {"GlobalId": guid, "storey": None}
        return {
            "GlobalId": guid,
            "storey": {
                "GlobalId": storey.GlobalId,
                "Name": getattr(storey, "Name", None),
                "Elevation_m": (
                    float(storey.Elevation) * scale
                    if isinstance(getattr(storey, "Elevation", None), (int, float))
                    else None
                ),
            },
        }

    # --- 6. get_door_dimensions --------------------------------------------
    def get_door_dimensions(guid: str) -> dict:
        """Width × height of a door in metres."""
        try:
            e = ctx.by_guid(guid)
        except ValueError as exc:
            return {"error": str(exc)}
        if not e.is_a("IfcDoor"):
            return {"error": f"Element {guid} is a {e.is_a()}, not IfcDoor."}

        # Prefer authored values
        width = getattr(e, "OverallWidth", None)
        height = getattr(e, "OverallHeight", None)
        if width is None or height is None:
            psets = get_psets(e)
            width = width if width is not None else find_quantity(psets, ["Width", "OverallWidth"])
            height = height if height is not None else find_quantity(psets, ["Height", "OverallHeight"])
        # Geometry fallback
        if width is None or height is None:
            bbox = element_aabb(ctx, e)
            if bbox is not None:
                dx, dy, dz = bbox["size"]
                width = width if width is not None else max(dx, dy)
                height = height if height is not None else dz
        return {
            "GlobalId": guid,
            "width_m": (float(width) * scale) if isinstance(width, (int, float)) else None,
            "height_m": (float(height) * scale) if isinstance(height, (int, float)) else None,
        }

    # --- 7. get_elements_in_room -------------------------------------------
    def get_elements_in_room(guid: str) -> dict:
        """List elements contained in a given IfcSpace.

        Combines ContainsElements (direct containment) with a bounding-box
        fallback for elements only related geometrically.
        """
        import ifcopenshell.util.element as ifc_elem

        try:
            space = ctx.by_guid(guid)
        except ValueError as exc:
            return {"error": str(exc)}
        if not space.is_a("IfcSpace"):
            return {"error": f"Element {guid} is a {space.is_a()}, not IfcSpace."}

        seen: dict[str, dict] = {}
        # 1. IfcRelContainedInSpatialStructure
        for rel in getattr(space, "ContainsElements", []) or []:
            for child in getattr(rel, "RelatedElements", []) or []:
                seen[child.GlobalId] = {
                    "GlobalId": child.GlobalId,
                    "type": child.is_a(),
                    "Name": getattr(child, "Name", None),
                    "via": "ContainsElements",
                }
        # 2. BoundedBy (IfcRelSpaceBoundary): walls, doors, windows touching the space
        for rel in getattr(space, "BoundedBy", []) or []:
            child = getattr(rel, "RelatedBuildingElement", None)
            if child is not None and child.GlobalId not in seen:
                seen[child.GlobalId] = {
                    "GlobalId": child.GlobalId,
                    "type": child.is_a(),
                    "Name": getattr(child, "Name", None),
                    "via": "BoundedBy",
                }

        return {
            "room": {"GlobalId": guid, "Name": getattr(space, "Name", None)},
            "n_elements": len(seen),
            "elements": list(seen.values())[:300],
        }

    # --- 8. get_floor_to_floor_height --------------------------------------
    def get_floor_to_floor_height(storey_a: str, storey_b: str) -> dict:
        """Vertical distance between two storeys (matched by Name), in metres."""
        storeys = ctx.by_type("IfcBuildingStorey")
        a = next((s for s in storeys if (s.Name or "").lower() == storey_a.lower()), None)
        b = next((s for s in storeys if (s.Name or "").lower() == storey_b.lower()), None)
        if a is None or b is None:
            return {
                "error": "Storey not found by name.",
                "available_storeys": [s.Name for s in storeys],
            }
        ea = float(getattr(a, "Elevation", 0.0) or 0.0) * scale
        eb = float(getattr(b, "Elevation", 0.0) or 0.0) * scale
        return {
            "storey_a": storey_a,
            "storey_b": storey_b,
            "elevation_a_m": ea,
            "elevation_b_m": eb,
            "distance_m": abs(ea - eb),
        }

    # --- 9. get_room_ceiling_height ----------------------------------------
    def get_room_ceiling_height(guid: str) -> dict:
        """Ceiling height of an IfcSpace in metres.

        Prefers Pset_SpaceCommon.Height / Qto_SpaceBaseQuantities.Height,
        falls back to AABB Z-extent.
        """
        try:
            space = ctx.by_guid(guid)
        except ValueError as exc:
            return {"error": str(exc)}
        if not space.is_a("IfcSpace"):
            return {"error": f"Element {guid} is a {space.is_a()}, not IfcSpace."}

        psets = get_psets(space)
        h = find_quantity(psets, ["Height", "FinishCeilingHeight", "NetCeilingHeight"])
        if h is not None:
            return {"GlobalId": guid, "ceiling_height_m": float(h) * scale, "source": "psets"}
        bbox = element_aabb(ctx, space)
        if bbox is None:
            return {"GlobalId": guid, "ceiling_height_m": None}
        return {
            "GlobalId": guid,
            "ceiling_height_m": float(bbox["size"][2]) * scale,
            "source": "geometry-aabb",
        }

    # --- 10. get_rooms_with_outdoor_access ---------------------------------
    def get_rooms_with_outdoor_access() -> dict:
        """Rooms that have at least one external door.

        Heuristic: a door is "external" if its Pset_DoorCommon.IsExternal is True,
        OR if the wall it fills has Pset_WallCommon.IsExternal == True.
        """
        rows: list[dict] = []
        for space in ctx.by_type("IfcSpace"):
            external_doors: list[dict] = []
            for rel in getattr(space, "BoundedBy", []) or []:
                child = getattr(rel, "RelatedBuildingElement", None)
                if child is None or not child.is_a("IfcDoor"):
                    continue
                if _door_is_external(child):
                    external_doors.append(
                        {"GlobalId": child.GlobalId, "Name": getattr(child, "Name", None)}
                    )
            if external_doors:
                rows.append(
                    {
                        "GlobalId": space.GlobalId,
                        "Name": getattr(space, "Name", None),
                        "n_external_doors": len(external_doors),
                        "external_doors": external_doors,
                    }
                )
        return truncate_list(rows, limit=200)

    # --- 11. check_door_accessibility --------------------------------------
    def check_door_accessibility(guid: str, min_clear_width_mm: float = 850.0) -> dict:
        """Pass/fail check for door clear width vs an accessibility threshold."""
        dims = get_door_dimensions(guid)
        if "error" in dims:
            return dims
        w_m = dims.get("width_m")
        if w_m is None:
            return {"GlobalId": guid, "passes": None, "reason": "Width could not be determined."}
        w_mm = float(w_m) * 1000.0
        return {
            "GlobalId": guid,
            "width_mm": round(w_mm, 1),
            "threshold_mm": min_clear_width_mm,
            "passes": w_mm >= min_clear_width_mm,
        }

    return [
        StructuredTool.from_function(
            get_element_bounding_box, name="get_element_bounding_box", args_schema=_GuidArgs
        ),
        StructuredTool.from_function(
            bounding_box_intersect, name="bounding_box_intersect", args_schema=_TwoGuidArgs
        ),
        StructuredTool.from_function(
            get_containing_rooms_for_entities_type,
            name="get_containing_rooms_for_entities_type",
            args_schema=_IfcClassArgs,
        ),
        StructuredTool.from_function(
            get_containing_rooms_for_entity_guids,
            name="get_containing_rooms_for_entity_guids",
            args_schema=_GuidListArgs,
        ),
        StructuredTool.from_function(
            get_containing_storey, name="get_containing_storey", args_schema=_GuidArgs
        ),
        StructuredTool.from_function(get_door_dimensions, name="get_door_dimensions", args_schema=_GuidArgs),
        StructuredTool.from_function(get_elements_in_room, name="get_elements_in_room", args_schema=_GuidArgs),
        StructuredTool.from_function(
            get_floor_to_floor_height, name="get_floor_to_floor_height", args_schema=_StoreyPairArgs
        ),
        StructuredTool.from_function(
            get_room_ceiling_height, name="get_room_ceiling_height", args_schema=_GuidArgs
        ),
        StructuredTool.from_function(
            get_rooms_with_outdoor_access, name="get_rooms_with_outdoor_access", args_schema=_NoArgs
        ),
        StructuredTool.from_function(
            check_door_accessibility, name="check_door_accessibility", args_schema=_DoorAccessibilityArgs
        ),
    ]


# ---------------------------------------------------------------------------
# Module-level helpers (used by tools above)
# ---------------------------------------------------------------------------
def _containment_by_guids(
    ctx: IFCContext, guids: list[str], ifc_class: Optional[str] = None
) -> dict:
    """Shared logic for tools 3 and 4."""
    import ifcopenshell.util.element as ifc_elem

    rows: list[dict] = []
    for gid in guids:
        try:
            element = ctx.by_guid(gid)
        except ValueError:
            rows.append({"GlobalId": gid, "room": None, "error": "not found"})
            continue
        # 1. direct containment
        try:
            container = ifc_elem.get_container(element)
        except Exception:
            container = None
        room: Optional[dict] = None
        if container is not None and container.is_a("IfcSpace"):
            room = {"GlobalId": container.GlobalId, "Name": container.Name}
        else:
            # 2. bounded-by walk: find a space whose BoundedBy lists this element
            for sp in ctx.by_type("IfcSpace"):
                for rel in getattr(sp, "BoundedBy", []) or []:
                    rb = getattr(rel, "RelatedBuildingElement", None)
                    if rb is not None and rb.GlobalId == gid:
                        room = {"GlobalId": sp.GlobalId, "Name": sp.Name}
                        break
                if room is not None:
                    break
        rows.append(
            {
                "GlobalId": gid,
                "type": element.is_a(),
                "room": room,
            }
        )
    out = {"n_elements": len(rows), "rows": rows[:300]}
    if ifc_class:
        out["ifc_class"] = ifc_class
    return out


def _door_is_external(door) -> bool:
    """True if door has IsExternal=True, or is hosted in an external wall."""
    psets = get_psets(door)
    door_common = psets.get("Pset_DoorCommon", {}) or {}
    if door_common.get("IsExternal") is True:
        return True
    # Walk through host wall
    for rel in getattr(door, "FillsVoids", []) or []:
        opening = getattr(rel, "RelatingOpeningElement", None)
        if opening is None:
            continue
        for void_rel in getattr(opening, "VoidsElements", []) or []:
            host = getattr(void_rel, "RelatingBuildingElement", None)
            if host is None:
                continue
            wall_common = get_psets(host).get("Pset_WallCommon", {}) or {}
            if wall_common.get("IsExternal") is True:
                return True
    return False
