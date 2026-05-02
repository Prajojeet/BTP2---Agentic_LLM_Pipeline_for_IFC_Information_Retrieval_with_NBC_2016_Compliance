"""Quantity Computation tools (8).

Sums, counts, and area/volume aggregations across the model. Where possible
we pull values from BaseQuantities pset (the "Q-set" / IfcElementQuantity)
because that's how authoring tools store designer-validated numbers.
Geometry is the fallback when quantities aren't authored.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..ifc_context import IFCContext
from ..utils.geometry import (
    element_aabb,
    element_outward_normal_xy,
    find_quantity,
    get_psets,
    normal_matches_direction,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------
class _NoArgs(BaseModel):
    pass


class _IfcClassArgs(BaseModel):
    ifc_class: str = Field(..., description="IFC class name, e.g. 'IfcWall' or 'Wall'.")


class _FacadeArgs(BaseModel):
    direction: str = Field(
        ..., description="Cardinal direction of the facade: 'north', 'south', 'east', or 'west'."
    )


class _QuantityArgs(BaseModel):
    guid: str = Field(..., description="GlobalId of the element to query.")
    quantity_name: str = Field(
        ...,
        description="Name of the quantity to extract, e.g. 'NetVolume', 'GrossArea', 'Length', 'Width'.",
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_quantity_computation_tools(ctx: IFCContext) -> list:
    """Build the 8 Quantity-Computation tools as StructuredTool instances."""

    scale = ctx.unit_scale_to_meters()

    # ---- helpers reused below -----------------------------------------------
    def _area_of(element) -> Optional[float]:
        """Best-effort area in m² for one element."""
        psets = get_psets(element)
        for names in (
            ["NetSideArea", "GrossSideArea"],          # walls
            ["NetArea", "GrossArea"],                  # generic
            ["NetFloorArea", "GrossFloorArea"],        # spaces / slabs
            ["TotalArea", "Area"],                     # fallback
        ):
            v = find_quantity(psets, names)
            if v is not None:
                # quantities are stored in model units — area unit² scale
                return float(v) * (scale ** 2)
        # Geometry fallback: 2D footprint area from AABB. Coarse, but better
        # than returning nothing.
        bbox = element_aabb(ctx, element)
        if bbox is None:
            return None
        dx, dy, _ = bbox["size"]
        return float(dx * dy) * (scale ** 2)

    def _volume_of(element) -> Optional[float]:
        psets = get_psets(element)
        for names in (
            ["NetVolume", "GrossVolume"],
            ["Volume"],
        ):
            v = find_quantity(psets, names)
            if v is not None:
                return float(v) * (scale ** 3)
        bbox = element_aabb(ctx, element)
        if bbox is None:
            return None
        dx, dy, dz = bbox["size"]
        return float(dx * dy * dz) * (scale ** 3)

    # --- 1. calculate_glazing_area -----------------------------------------
    def calculate_glazing_area() -> dict:
        """Total glazing area = sum of window + curtain-wall pane areas (m²)."""
        total = 0.0
        details: list[dict] = []
        for cls in ("IfcWindow", "IfcPlate"):  # IfcPlate covers curtain-wall panes
            elements = ctx.by_type(cls)
            for e in elements:
                # For IfcPlate, only count those that are part of curtain walls
                if cls == "IfcPlate":
                    pred_type = getattr(e, "PredefinedType", None)
                    if pred_type not in ("CURTAIN_PANEL", "GLASS_PANEL", "SHEET", None):
                        continue
                a = _area_of(e)
                if a is not None:
                    total += a
                    details.append({"GlobalId": e.GlobalId, "type": cls, "area_m2": round(a, 4)})
        return {
            "total_glazing_area_m2": round(total, 4),
            "n_elements": len(details),
            "elements": details[:50],
        }

    # --- 2. calculate_gross_floor_area -------------------------------------
    def calculate_gross_floor_area() -> dict:
        """GFA: prefers IfcSpace.GrossFloorArea, falls back to slab gross areas."""
        spaces = ctx.by_type("IfcSpace")
        total = 0.0
        n = 0
        for s in spaces:
            psets = get_psets(s)
            v = find_quantity(psets, ["GrossFloorArea", "GrossArea"], qto_hints=["Qto", "BaseQuantities"])
            if v is not None:
                total += float(v) * (scale ** 2)
                n += 1
        if n == 0:
            # fallback: sum slab gross areas
            for slab in ctx.by_type("IfcSlab"):
                psets = get_psets(slab)
                v = find_quantity(psets, ["GrossArea"], qto_hints=["Qto", "BaseQuantities"])
                if v is not None:
                    total += float(v) * (scale ** 2)
                    n += 1
        return {
            "gross_floor_area_m2": round(total, 4),
            "source": "IfcSpace.GrossFloorArea" if spaces else "IfcSlab.GrossArea",
            "n_contributing_elements": n,
        }

    # --- 3. calculate_usable_floor_area ------------------------------------
    def calculate_usable_floor_area() -> dict:
        """Usable / net floor area: sum of IfcSpace.NetFloorArea."""
        total = 0.0
        n = 0
        for s in ctx.by_type("IfcSpace"):
            psets = get_psets(s)
            v = find_quantity(
                psets, ["NetFloorArea", "NetArea"], qto_hints=["Qto", "BaseQuantities"]
            )
            if v is not None:
                total += float(v) * (scale ** 2)
                n += 1
        return {
            "usable_floor_area_m2": round(total, 4),
            "n_spaces_with_area": n,
        }

    # --- 4. count_windows_on_facade ----------------------------------------
    def count_windows_on_facade(direction: str) -> dict:
        """Count windows whose host wall faces the given cardinal direction."""
        direction = direction.lower().strip()
        if direction not in {"north", "south", "east", "west"}:
            return {"error": f"direction must be one of north/south/east/west (got {direction!r})"}

        windows = ctx.by_type("IfcWindow")
        matches: list[dict] = []
        for w in windows:
            # Try the host wall first — its normal is more reliable than the window's
            host = _get_host_wall(w)
            target = host if host is not None else w
            normal_xy = element_outward_normal_xy(ctx, target)
            if normal_xy is None:
                continue
            if normal_matches_direction(normal_xy, direction):
                matches.append(
                    {"GlobalId": w.GlobalId, "Name": getattr(w, "Name", None)}
                )

        return {
            "direction": direction,
            "count": len(matches),
            "windows": matches[:100],
        }

    # --- 5. extract_quantity_from_property_sets ----------------------------
    def extract_quantity_from_property_sets(guid: str, quantity_name: str) -> dict:
        """Return one named quantity from any pset/qset of an element."""
        try:
            element = ctx.by_guid(guid)
        except ValueError as exc:
            return {"error": str(exc)}
        psets = get_psets(element)
        v = find_quantity(psets, [quantity_name])
        return {
            "GlobalId": guid,
            "quantity_name": quantity_name,
            "value": v,
            "found": v is not None,
        }

    # --- 6. get_elements_area ----------------------------------------------
    def get_elements_area(ifc_class: str) -> dict:
        """Sum the area of all elements of the given class (m²)."""
        elements = ctx.by_type(ifc_class)
        total = 0.0
        n = 0
        for e in elements:
            a = _area_of(e)
            if a is not None:
                total += a
                n += 1
        return {
            "ifc_class": ifc_class,
            "n_elements_with_area": n,
            "n_total": len(elements),
            "total_area_m2": round(total, 4),
        }

    # --- 7. get_elements_volume --------------------------------------------
    def get_elements_volume(ifc_class: str) -> dict:
        """Sum the volume of all elements of the given class (m³)."""
        elements = ctx.by_type(ifc_class)
        total = 0.0
        n = 0
        for e in elements:
            v = _volume_of(e)
            if v is not None:
                total += v
                n += 1
        return {
            "ifc_class": ifc_class,
            "n_elements_with_volume": n,
            "n_total": len(elements),
            "total_volume_m3": round(total, 4),
        }

    # --- 8. get_pipe_length_by_type ----------------------------------------
    def get_pipe_length_by_type() -> dict:
        """Total pipe length grouped by ObjectType / pipe type, in metres."""
        rows: dict[str, dict] = {}
        for pipe in ctx.by_type("IfcPipeSegment"):
            ot = getattr(pipe, "ObjectType", None) or "<unspecified>"
            psets = get_psets(pipe)
            length = find_quantity(psets, ["Length"]) or 0.0
            length *= scale
            row = rows.setdefault(ot, {"type": ot, "count": 0, "total_length_m": 0.0})
            row["count"] += 1
            row["total_length_m"] += float(length)
        # Round for readability
        for r in rows.values():
            r["total_length_m"] = round(r["total_length_m"], 4)
        return {
            "n_types": len(rows),
            "by_type": list(rows.values()),
            "total_length_m": round(sum(r["total_length_m"] for r in rows.values()), 4),
        }

    return [
        StructuredTool.from_function(
            calculate_glazing_area, name="calculate_glazing_area", args_schema=_NoArgs
        ),
        StructuredTool.from_function(
            calculate_gross_floor_area, name="calculate_gross_floor_area", args_schema=_NoArgs
        ),
        StructuredTool.from_function(
            calculate_usable_floor_area, name="calculate_usable_floor_area", args_schema=_NoArgs
        ),
        StructuredTool.from_function(
            count_windows_on_facade, name="count_windows_on_facade", args_schema=_FacadeArgs
        ),
        StructuredTool.from_function(
            extract_quantity_from_property_sets,
            name="extract_quantity_from_property_sets",
            args_schema=_QuantityArgs,
        ),
        StructuredTool.from_function(
            get_elements_area, name="get_elements_area", args_schema=_IfcClassArgs
        ),
        StructuredTool.from_function(
            get_elements_volume, name="get_elements_volume", args_schema=_IfcClassArgs
        ),
        StructuredTool.from_function(
            get_pipe_length_by_type, name="get_pipe_length_by_type", args_schema=_NoArgs
        ),
    ]


# ---------------------------------------------------------------------------
# Module-level helper used inside tools
# ---------------------------------------------------------------------------
def _get_host_wall(window) -> Optional[object]:
    """Walk IfcRelFillsElement / IfcRelVoidsElement to find the wall hosting a window."""
    for rel in getattr(window, "FillsVoids", []) or []:
        opening = getattr(rel, "RelatingOpeningElement", None)
        if opening is None:
            continue
        for void_rel in getattr(opening, "VoidsElements", []) or []:
            host = getattr(void_rel, "RelatingBuildingElement", None)
            if host is not None and host.is_a("IfcWall"):
                return host
    return None
