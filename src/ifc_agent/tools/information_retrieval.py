"""Information Retrieval tools (10).

These tools answer "what is in this model?" questions without doing any
computation beyond reading attributes and property sets.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..ifc_context import IFCContext
from ..utils.geometry import get_psets
from ..utils.safe_serialize import to_jsonable, truncate_list

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument schemas (Pydantic v2)
# ---------------------------------------------------------------------------
class _IfcClassArgs(BaseModel):
    ifc_class: str = Field(
        ...,
        description="The IFC class name. Either with or without the 'Ifc' prefix, e.g. 'IfcWall' or 'Wall'.",
    )


class _GuidArgs(BaseModel):
    guid: str = Field(..., description="GlobalId (IfcGloballyUniqueId) of the element.")


class _NoArgs(BaseModel):
    pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_information_retrieval_tools(ctx: IFCContext) -> list:
    """Build the 10 Information-Retrieval tools as StructuredTool instances."""

    # --- 1. find_elements_by_ifc_class -------------------------------------
    def find_elements_by_ifc_class(ifc_class: str) -> dict:
        """Find every element of a given IFC class.

        Returns a dict with the count and a list of {GlobalId, Name, ObjectType}.
        """
        elements = ctx.by_type(ifc_class)
        items = [
            {
                "GlobalId": getattr(e, "GlobalId", None),
                "Name": getattr(e, "Name", None),
                "ObjectType": getattr(e, "ObjectType", None),
            }
            for e in elements
        ]
        return truncate_list(items, limit=300)

    # --- 2. get_element_properties -----------------------------------------
    def get_element_properties(guid: str) -> dict:
        """Return all property and quantity sets attached to one element."""
        try:
            element = ctx.by_guid(guid)
        except ValueError as exc:
            return {"error": str(exc)}
        psets = get_psets(element)
        return {
            "GlobalId": element.GlobalId,
            "type": element.is_a(),
            "Name": getattr(element, "Name", None),
            "psets": to_jsonable(psets),
        }

    # --- 3. get_available_models -------------------------------------------
    def get_available_models() -> dict:
        """List loaded IFC models. This app loads exactly one at a time."""
        return {
            "models": [
                {
                    "file": ctx.file_path.name,
                    "schema": ctx.schema,
                    "path": str(ctx.file_path),
                }
            ]
        }

    # --- 4. get_model_path -------------------------------------------------
    def get_model_path() -> dict:
        """Return the absolute path of the currently loaded IFC file."""
        return {"path": str(ctx.file_path)}

    # --- 5. list_object_types_for_ifc_entity -------------------------------
    def list_object_types_for_ifc_entity(ifc_class: str) -> dict:
        """Distinct ObjectType / type-name values for instances of an IFC class.

        Useful for "what kinds of doors do we have?"-style questions.
        """
        elements = ctx.by_type(ifc_class)
        seen: dict[str, int] = {}
        for e in elements:
            ot = getattr(e, "ObjectType", None) or "<unspecified>"
            seen[ot] = seen.get(ot, 0) + 1
        return {
            "ifc_class": ifc_class,
            "n_instances": len(elements),
            "object_types": [{"ObjectType": k, "count": v} for k, v in sorted(seen.items())],
        }

    # --- 6. list_rooms ------------------------------------------------------
    def list_rooms() -> dict:
        """Every IfcSpace with name, long name, and the storey it sits on."""
        import ifcopenshell.util.element as ifc_elem

        spaces = ctx.by_type("IfcSpace")
        rows = []
        for s in spaces:
            try:
                container = ifc_elem.get_container(s)
                storey_name = container.Name if container is not None else None
            except Exception:
                storey_name = None
            rows.append(
                {
                    "GlobalId": s.GlobalId,
                    "Name": getattr(s, "Name", None),
                    "LongName": getattr(s, "LongName", None),
                    "storey": storey_name,
                }
            )
        return truncate_list(rows, limit=500)

    # --- 7. get_storeys_names ----------------------------------------------
    def get_storeys_names() -> dict:
        """Every IfcBuildingStorey with name and elevation."""
        storeys = ctx.by_type("IfcBuildingStorey")
        rows = []
        scale = ctx.unit_scale_to_meters()
        for s in storeys:
            elev = getattr(s, "Elevation", None)
            rows.append(
                {
                    "GlobalId": s.GlobalId,
                    "Name": getattr(s, "Name", None),
                    "elevation_m": (float(elev) * scale) if isinstance(elev, (int, float)) else None,
                }
            )
        # Sort by elevation, ground storey first.
        rows.sort(key=lambda r: (r["elevation_m"] is None, r["elevation_m"] or 0.0))
        return {"count": len(rows), "storeys": rows}

    # --- 8. get_type_definitions_and_instances -----------------------------
    def get_type_definitions_and_instances(ifc_class: str) -> dict:
        """For an IFC class, list its IfcTypeObject definitions + instance counts.

        Walks IsDefinedBy / RelDefinesByType to build {type_name: [instance GUIDs]}.
        """
        import ifcopenshell.util.element as ifc_elem

        elements = ctx.by_type(ifc_class)
        by_type: dict[str, list[str]] = {}
        for e in elements:
            try:
                t = ifc_elem.get_type(e)
            except Exception:
                t = None
            key = getattr(t, "Name", None) or getattr(t, "GlobalId", None) or "<no type>"
            by_type.setdefault(key, []).append(e.GlobalId)
        return {
            "ifc_class": ifc_class,
            "type_definitions": [
                {"type_name": k, "instance_count": len(v), "instances": v[:50]}
                for k, v in sorted(by_type.items())
            ],
        }

    # --- 9. get_state ------------------------------------------------------
    def get_state() -> dict:
        """High-level summary of the loaded model."""
        proj = ctx.by_type("IfcProject")
        site = ctx.by_type("IfcSite")
        bld = ctx.by_type("IfcBuilding")
        st = ctx.by_type("IfcBuildingStorey")
        sp = ctx.by_type("IfcSpace")
        return {
            "file": ctx.file_path.name,
            "schema": ctx.schema,
            "project_name": proj[0].Name if proj else None,
            "n_sites": len(site),
            "n_buildings": len(bld),
            "n_storeys": len(st),
            "n_spaces": len(sp),
            "total_entities": len(list(ctx.model)),
        }

    # --- 10. is_georeferenced ----------------------------------------------
    def is_georeferenced() -> dict:
        """True iff the model has IfcMapConversion or IfcProjectedCRS."""
        try:
            map_conv = ctx.by_type("IfcMapConversion")
            crs = ctx.by_type("IfcProjectedCRS")
        except Exception:
            map_conv, crs = [], []
        return {
            "georeferenced": bool(map_conv or crs),
            "n_map_conversions": len(map_conv),
            "n_projected_crs": len(crs),
        }

    return [
        StructuredTool.from_function(
            find_elements_by_ifc_class, name="find_elements_by_ifc_class", args_schema=_IfcClassArgs
        ),
        StructuredTool.from_function(
            get_element_properties, name="get_element_properties", args_schema=_GuidArgs
        ),
        StructuredTool.from_function(
            get_available_models, name="get_available_models", args_schema=_NoArgs
        ),
        StructuredTool.from_function(get_model_path, name="get_model_path", args_schema=_NoArgs),
        StructuredTool.from_function(
            list_object_types_for_ifc_entity,
            name="list_object_types_for_ifc_entity",
            args_schema=_IfcClassArgs,
        ),
        StructuredTool.from_function(list_rooms, name="list_rooms", args_schema=_NoArgs),
        StructuredTool.from_function(get_storeys_names, name="get_storeys_names", args_schema=_NoArgs),
        StructuredTool.from_function(
            get_type_definitions_and_instances,
            name="get_type_definitions_and_instances",
            args_schema=_IfcClassArgs,
        ),
        StructuredTool.from_function(get_state, name="get_state", args_schema=_NoArgs),
        StructuredTool.from_function(is_georeferenced, name="is_georeferenced", args_schema=_NoArgs),
    ]
