"""Tool registry — exposes ``build_tools(ctx, findings_store=None)``.

The original 29 IFC tools (Hellin et al. 2025) plus 9 NBC 2016 Part 4 fire-safety
+ memory-recall tools when a findings store is supplied. The tool selector and
the ReAct executor both go through here.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool

from ..compliance import FindingsStore, build_compliance_tools
from ..ifc_context import IFCContext
from .geometric_processing import build_geometric_processing_tools
from .information_retrieval import build_information_retrieval_tools
from .quantity_computation import build_quantity_computation_tools


def build_tools(
    ctx: IFCContext,
    findings_store: Optional[FindingsStore] = None,
) -> list[BaseTool]:
    """Build all tools, each closed over the given IFCContext.

    The 29 retrieval tools come from Hellin et al. (2025), Table 1.
    When ``findings_store`` is given, 6 NBC 2016 Part 4 fire-safety checks +
    3 memory-recall tools are appended; the selector then has the option to
    pick them when the user asks about fire safety / egress / IS-code compliance.
    """
    tools: list[BaseTool] = (
        build_information_retrieval_tools(ctx)
        + build_quantity_computation_tools(ctx)
        + build_geometric_processing_tools(ctx)
    )
    if findings_store is not None:
        tools.extend(build_compliance_tools(ctx, findings_store))
    return tools


def tool_catalog(include_compliance: bool = True) -> list[dict[str, str]]:
    """Static catalogue used by the CoT tool-selector.

    Returns name + short description + category. The compliance entries are
    included by default so the selector can pick them when the user asks
    fire-safety / egress / NBC questions.
    """
    if include_compliance:
        return _CATALOG + _COMPLIANCE_CATALOG
    return list(_CATALOG)


_CATALOG: list[dict[str, str]] = [
    # --- Information Retrieval ---------------------------------------------
    {"category": "Information Retrieval", "name": "find_elements_by_ifc_class",
     "description": "Return GlobalIds + names of every IFC entity of a given class (e.g. 'IfcWall', 'IfcDoor')."},
    {"category": "Information Retrieval", "name": "get_element_properties",
     "description": "Return all property and quantity sets attached to a single element, identified by GlobalId."},
    {"category": "Information Retrieval", "name": "get_available_models",
     "description": "List the IFC files currently loaded into the session (this app loads exactly one at a time)."},
    {"category": "Information Retrieval", "name": "get_model_path",
     "description": "Return the file-system path of the currently loaded IFC model."},
    {"category": "Information Retrieval", "name": "list_object_types_for_ifc_entity",
     "description": "List the unique ObjectType / type-name values used by a given IFC class (e.g. distinct door types)."},
    {"category": "Information Retrieval", "name": "list_rooms",
     "description": "List every IfcSpace (room): GlobalId, Name, LongName, and storey it sits on."},
    {"category": "Information Retrieval", "name": "get_storeys_names",
     "description": "List every IfcBuildingStorey: GlobalId, Name, and Elevation."},
    {"category": "Information Retrieval", "name": "get_type_definitions_and_instances",
     "description": "For an IFC class, list its Type definitions and how many instances each Type drives."},
    {"category": "Information Retrieval", "name": "get_state",
     "description": "High-level model summary: schema, project name, site/building/storey counts, total entities."},
    {"category": "Information Retrieval", "name": "is_georeferenced",
     "description": "Check whether the model has georeferencing metadata (IfcMapConversion / IfcProjectedCRS)."},

    # --- Quantity Computation ----------------------------------------------
    {"category": "Quantity Computation", "name": "calculate_glazing_area",
     "description": "Total glazing (window+curtain-wall) area in square metres, summed across the whole model."},
    {"category": "Quantity Computation", "name": "calculate_gross_floor_area",
     "description": "Total gross floor area (GFA) in m² — sum of slab/space gross areas."},
    {"category": "Quantity Computation", "name": "calculate_usable_floor_area",
     "description": "Total usable floor area in m² — sum of net IfcSpace floor areas."},
    {"category": "Quantity Computation", "name": "count_windows_on_facade",
     "description": "Count windows on a given facade direction (north/south/east/west)."},
    {"category": "Quantity Computation", "name": "extract_quantity_from_property_sets",
     "description": "Extract a specific named quantity (e.g. 'NetVolume', 'Length') from one element's psets/qsets."},
    {"category": "Quantity Computation", "name": "get_elements_area",
     "description": "Sum the area in m² of all elements of a given IFC class (e.g. total wall area)."},
    {"category": "Quantity Computation", "name": "get_elements_volume",
     "description": "Sum the volume in m³ of all elements of a given IFC class (e.g. total concrete volume)."},
    {"category": "Quantity Computation", "name": "get_pipe_length_by_type",
     "description": "Total length of pipes (IfcPipeSegment) grouped by ObjectType / pipe type."},

    # --- Geometric Processing ----------------------------------------------
    {"category": "Geometric Processing", "name": "get_element_bounding_box",
     "description": "Axis-aligned bounding box (min, max, size) of a single element by GlobalId, in metres."},
    {"category": "Geometric Processing", "name": "bounding_box_intersect",
     "description": "Boolean test: do the AABBs of two elements (by GlobalIds) overlap?"},
    {"category": "Geometric Processing", "name": "get_containing_rooms_for_entities_type",
     "description": "For every element of a given class, return which IfcSpace contains it (e.g. which room each door is in)."},
    {"category": "Geometric Processing", "name": "get_containing_rooms_for_entity_guids",
     "description": "Given a list of element GlobalIds, return the containing IfcSpace for each."},
    {"category": "Geometric Processing", "name": "get_containing_storey",
     "description": "Return the IfcBuildingStorey containing the given element."},
    {"category": "Geometric Processing", "name": "get_door_dimensions",
     "description": "Width and height of a single door in metres, by GlobalId."},
    {"category": "Geometric Processing", "name": "get_elements_in_room",
     "description": "List all elements (doors, windows, furniture, …) inside a given IfcSpace by room GlobalId."},
    {"category": "Geometric Processing", "name": "get_floor_to_floor_height",
     "description": "Vertical distance in metres between two storeys, by storey names."},
    {"category": "Geometric Processing", "name": "get_room_ceiling_height",
     "description": "Ceiling height (m) of a single IfcSpace by GlobalId."},
    {"category": "Geometric Processing", "name": "get_rooms_with_outdoor_access",
     "description": "List rooms that contain at least one door connecting to the outside."},
    {"category": "Geometric Processing", "name": "check_door_accessibility",
     "description": "Check whether a door (by GlobalId) meets a minimum clear width (default 850 mm — accessibility)."},
]


_COMPLIANCE_CATALOG: list[dict[str, str]] = [
    # --- Fire Safety Compliance (NBC 2016 Part 4) -------------------------
    {"category": "Fire Safety Compliance", "name": "check_means_of_egress",
     "description": "NBC Part 4 4.4.2 — verify each storey has the required number of external exits for its occupant load. Pick when the user asks about exits, evacuation routes, or fire-egress count."},
    {"category": "Fire Safety Compliance", "name": "check_travel_distance",
     "description": "NBC Part 4 Table 9 — flag rooms whose distance to the nearest exit exceeds the occupancy limit. Pick for 'travel distance', 'how far is the exit', or evacuation-route length questions."},
    {"category": "Fire Safety Compliance", "name": "check_exit_widths",
     "description": "NBC Part 4 4.5 — verify external doors ≥ 1.0 m and stairs ≥ 1.5 m. Pick for exit-width / corridor-width / stair-width compliance questions."},
    {"category": "Fire Safety Compliance", "name": "check_dead_end_corridors",
     "description": "NBC Part 4 4.5.4 — heuristic flag of long single-door corridors as likely dead-ends > 6 m. Pick for dead-end / blind corridor questions."},
    {"category": "Fire Safety Compliance", "name": "check_compartmentation",
     "description": "NBC Part 4 3.4 — scan IfcWall.Pset_WallCommon.FireRating to check fire compartmentation. Pick for fire-rating, fire-resistance, or compartmentation questions."},
    {"category": "Fire Safety Compliance", "name": "check_refuge_area",
     "description": "NBC Part 4 4.4.5 — for buildings > 24 m, verify presence of a refuge area (IfcSpace named 'refuge'). Pick for high-rise refuge questions."},
    # --- Memory recall ------------------------------------------------------
    {"category": "Compliance Memory", "name": "list_compliance_findings",
     "description": "List every compliance finding recorded in this session (id, check, verdict, summary). Pick when the user asks 'what have we checked' or wants a recap."},
    {"category": "Compliance Memory", "name": "get_finding_details",
     "description": "Retrieve full details (params, failing elements) of one finding by its F-XXXXXXXX id. Pick when drilling into a previous check's results."},
    {"category": "Compliance Memory", "name": "clear_compliance_findings",
     "description": "Erase all compliance findings from session memory. Pick only when the user explicitly says to reset/clear."},
]


__all__ = ["build_tools", "tool_catalog"]
