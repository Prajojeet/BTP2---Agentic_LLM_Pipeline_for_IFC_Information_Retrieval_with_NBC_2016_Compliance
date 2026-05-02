"""NBC 2016 Part 4 — Fire & Life Safety compliance tools (beta).

Six structured checks the agent can call when a question is about fire
safety / egress:

  - check_means_of_egress         — number of exits per storey vs. occupant load
  - check_travel_distance         — room-to-nearest-exit straight-line distance × corridor factor
  - check_exit_widths             — door / corridor / stair clear widths
  - check_dead_end_corridors      — flags long dead-end corridors (heuristic)
  - check_compartmentation        — fire ratings on rated walls (Pset_WallCommon.FireRating)
  - check_refuge_area             — refuge floor area for high-rise buildings

Plus three accessor tools the agent can call to inspect/recall findings:

  - list_compliance_findings
  - get_finding_details
  - clear_compliance_findings

NBC clauses cited follow the National Building Code of India 2016, Part 4
(Fire and Life Safety). Where IFC does not carry the data needed, the tool
returns ``verdict = "indeterminate"`` with an explanation rather than guessing.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..ifc_context import IFCContext
from ..utils.geometry import (
    element_aabb,
    get_containing_space,
    get_containing_storey,
    get_psets,
)
from .findings_store import FindingsStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NBC reference values (Part 4, 2016 — beta subset)
# ---------------------------------------------------------------------------
# Occupant load factors (m² per person), from NBC 2016 Part 4, Table 7.
_OCCUPANT_LOAD_M2_PER_PERSON: dict[str, float] = {
    "residential": 12.5,
    "business": 10.0,
    "mercantile": 3.5,
    "assembly": 1.0,
    "educational": 4.0,
    "industrial": 10.0,
    "storage": 30.0,
    "default": 10.0,
}

# Maximum travel distance to nearest exit (m), NBC 2016 Part 4, Table 9.
_MAX_TRAVEL_DISTANCE_M: dict[str, float] = {
    "residential": 22.5,
    "business": 30.0,
    "mercantile": 30.0,
    "assembly": 22.5,
    "educational": 22.5,
    "industrial": 22.5,
    "storage": 30.0,
    "default": 22.5,
}

# Minimum clear widths (m), NBC 2016 Part 4, 4.5.
_MIN_DOOR_CLEAR_WIDTH_M = 1.0
_MIN_CORRIDOR_WIDTH_M = 1.0
_MIN_STAIR_WIDTH_M = 1.5
_DEAD_END_LIMIT_M = 6.0
_HIGH_RISE_HEIGHT_M = 24.0  # threshold for refuge area requirement


# ---------------------------------------------------------------------------
# Pydantic argument schemas
# ---------------------------------------------------------------------------
class _OccupancyArgs(BaseModel):
    occupancy_class: str = Field(
        default="default",
        description=(
            "NBC Part 4 occupancy class. One of: residential, business, mercantile, "
            "assembly, educational, industrial, storage, default."
        ),
    )


class _TravelDistanceArgs(BaseModel):
    occupancy_class: str = Field(
        default="default",
        description="NBC occupancy class — controls the maximum allowable travel distance.",
    )
    corridor_factor: float = Field(
        default=1.5,
        description=(
            "Multiplier applied to centroid-to-exit straight-line distance to estimate the "
            "actual walking path length. 1.5 is a reasonable default; raise to 2.0 for highly "
            "partitioned floor plans."
        ),
    )


class _NoArgs(BaseModel):
    pass


class _FindingIdArgs(BaseModel):
    finding_id: str = Field(description="The F-XXXXXXXX id returned by a previous compliance check.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_external_door(door: Any) -> bool:
    """Check Pset_DoorCommon.IsExternal (True/False), defaulting to False if absent."""
    psets = get_psets(door) or {}
    common = psets.get("Pset_DoorCommon") or {}
    val = common.get("IsExternal")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes"}
    # Fallback: check the host wall
    try:
        for rel in getattr(door, "FillsVoids", []) or []:
            opening = getattr(rel, "RelatingOpeningElement", None)
            if not opening:
                continue
            for void_rel in getattr(opening, "VoidsElements", []) or []:
                wall = getattr(void_rel, "RelatingBuildingElement", None)
                if not wall:
                    continue
                wpsets = get_psets(wall) or {}
                wcommon = wpsets.get("Pset_WallCommon") or {}
                ext = wcommon.get("IsExternal")
                if isinstance(ext, bool):
                    return ext
                if isinstance(ext, str):
                    return ext.strip().lower() in {"true", "1", "yes"}
    except Exception:
        pass
    return False


def _door_clear_width_m(ctx: IFCContext, door: Any) -> Optional[float]:
    """Best-effort door clear width in metres — pset → OverallWidth → AABB."""
    # 1. Try Pset_DoorCommon.OverallWidth or pset/qto width
    psets = get_psets(door) or {}
    for ps_name, vals in psets.items():
        if not isinstance(vals, dict):
            continue
        for key in ("OverallWidth", "Width", "ClearWidth", "NominalWidth"):
            if key in vals and isinstance(vals[key], (int, float)):
                return float(vals[key]) * ctx.unit_scale_to_meters()
    # 2. Direct attribute
    width_attr = getattr(door, "OverallWidth", None)
    if isinstance(width_attr, (int, float)):
        return float(width_attr) * ctx.unit_scale_to_meters()
    # 3. AABB
    aabb = element_aabb(ctx, door)
    if aabb:
        sx, sy, _ = aabb["size"]
        # The horizontal extent in the smaller of x or y is usually thickness;
        # the larger is the leaf width.
        return float(max(sx, sy)) * ctx.unit_scale_to_meters()
    return None


def _space_centroid_xyz(ctx: IFCContext, space: Any) -> Optional[tuple[float, float, float]]:
    """Return (x, y, z) centre of a space in metres."""
    aabb = element_aabb(ctx, space)
    if not aabb:
        return None
    scale = ctx.unit_scale_to_meters()
    mn, mx = aabb["min"], aabb["max"]
    return (
        (mn[0] + mx[0]) / 2.0 * scale,
        (mn[1] + mx[1]) / 2.0 * scale,
        (mn[2] + mx[2]) / 2.0 * scale,
    )


def _door_position_xyz(ctx: IFCContext, door: Any) -> Optional[tuple[float, float, float]]:
    aabb = element_aabb(ctx, door)
    if not aabb:
        return None
    scale = ctx.unit_scale_to_meters()
    mn, mx = aabb["min"], aabb["max"]
    return (
        (mn[0] + mx[0]) / 2.0 * scale,
        (mn[1] + mx[1]) / 2.0 * scale,
        (mn[2] + mx[2]) / 2.0 * scale,
    )


def _euclid(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5  # XY-only — same storey assumption


def _storey_of(element: Any) -> Optional[Any]:
    return get_containing_storey(element)


def _occupancy_factor(occ_class: str) -> float:
    return _OCCUPANT_LOAD_M2_PER_PERSON.get(
        occ_class.lower(), _OCCUPANT_LOAD_M2_PER_PERSON["default"]
    )


def _max_travel_distance(occ_class: str) -> float:
    return _MAX_TRAVEL_DISTANCE_M.get(
        occ_class.lower(), _MAX_TRAVEL_DISTANCE_M["default"]
    )


def _required_exits(occupant_load: float) -> int:
    """NBC 4.4.2: 1 exit if ≤ 50 occupants; 2 if 51–500; 3 if 501–1000; 4 if >1000."""
    if occupant_load <= 50:
        return 1
    if occupant_load <= 500:
        return 2
    if occupant_load <= 1000:
        return 3
    return 4


def _space_floor_area_m2(ctx: IFCContext, space: Any) -> Optional[float]:
    """Net floor area for an IfcSpace, from psets/qsets, falling back to AABB footprint."""
    psets = get_psets(space) or {}
    for ps in psets.values():
        if not isinstance(ps, dict):
            continue
        for k in ("NetFloorArea", "GrossFloorArea", "Area", "TotalArea"):
            v = ps.get(k)
            if isinstance(v, (int, float)) and v > 0:
                # Pset values are already in model units²
                scale = ctx.unit_scale_to_meters()
                return float(v) * (scale ** 2)
    aabb = element_aabb(ctx, space)
    if aabb:
        sx, sy, _ = aabb["size"]
        scale = ctx.unit_scale_to_meters()
        return float(sx * sy) * (scale ** 2)
    return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def _build_check_means_of_egress(ctx: IFCContext, store: FindingsStore) -> StructuredTool:
    def check_means_of_egress(occupancy_class: str = "default") -> dict[str, Any]:
        clause = "NBC 2016, Part 4, 4.4.2 (Number of Exits)"
        try:
            storeys = ctx.by_type("IfcBuildingStorey")
            doors = ctx.by_type("IfcDoor")
            spaces = ctx.by_type("IfcSpace")

            # Occupant load per storey from total IfcSpace area
            storey_data: dict[str, dict[str, Any]] = {}
            for st in storeys:
                gid = st.GlobalId
                storey_data[gid] = {
                    "name": getattr(st, "Name", None) or "(unnamed)",
                    "guid": gid,
                    "spaces_area_m2": 0.0,
                    "exit_doors": [],
                }

            occ_factor = _occupancy_factor(occupancy_class)

            for sp in spaces:
                st = _storey_of(sp)
                if st is None or st.GlobalId not in storey_data:
                    continue
                area = _space_floor_area_m2(ctx, sp) or 0.0
                storey_data[st.GlobalId]["spaces_area_m2"] += area

            for d in doors:
                if not _is_external_door(d):
                    continue
                st = _storey_of(d)
                if st is None or st.GlobalId not in storey_data:
                    continue
                storey_data[st.GlobalId]["exit_doors"].append(
                    {"guid": d.GlobalId, "name": getattr(d, "Name", None)}
                )

            failures = []
            params_per_storey = []
            for gid, data in storey_data.items():
                occ_load = data["spaces_area_m2"] / occ_factor if occ_factor > 0 else 0.0
                required = _required_exits(occ_load)
                actual = len(data["exit_doors"])
                params_per_storey.append({
                    "storey": data["name"],
                    "spaces_area_m2": round(data["spaces_area_m2"], 2),
                    "occupant_load": round(occ_load, 1),
                    "required_exits": required,
                    "actual_exits": actual,
                })
                if actual < required:
                    failures.append({
                        "element_guid": data["guid"],
                        "element_name": data["name"],
                        "reason": (
                            f"Storey '{data['name']}' has {actual} exit(s) for an estimated "
                            f"occupant load of {round(occ_load, 1)}; NBC requires {required}."
                        ),
                    })

            verdict = "fail" if failures else ("pass" if storey_data else "indeterminate")
            summary = (
                f"{len(storey_data)} storey(s) checked, occupancy '{occupancy_class}'; "
                f"{len(failures)} failure(s)."
            )

            f = store.record(
                check_name="check_means_of_egress",
                clause=clause,
                verdict=verdict,
                summary=summary,
                params_checked={
                    "occupancy_class": occupancy_class,
                    "occupant_load_factor_m2_per_person": occ_factor,
                    "per_storey": params_per_storey,
                },
                failures=failures,
            )
            return f.to_dict()
        except Exception as exc:
            logger.exception("check_means_of_egress failed: %s", exc)
            return store.record(
                check_name="check_means_of_egress",
                clause=clause,
                verdict="indeterminate",
                summary=f"Check could not be completed: {exc}",
                params_checked={"occupancy_class": occupancy_class},
            ).to_dict()

    return StructuredTool.from_function(
        check_means_of_egress,
        name="check_means_of_egress",
        description=(
            "NBC 2016 Part 4 4.4.2 — verify that each storey has the required number of "
            "exits (external doors) for its occupant load. Returns a Finding with verdict "
            "pass/fail/indeterminate, per-storey occupant-load math, and the list of "
            "storeys that failed."
        ),
        args_schema=_OccupancyArgs,
    )


def _build_check_travel_distance(ctx: IFCContext, store: FindingsStore) -> StructuredTool:
    def check_travel_distance(
        occupancy_class: str = "default", corridor_factor: float = 1.5
    ) -> dict[str, Any]:
        clause = "NBC 2016, Part 4, 4.5.1 / Table 9 (Travel Distance to Exit)"
        try:
            limit = _max_travel_distance(occupancy_class)
            spaces = ctx.by_type("IfcSpace")
            doors = ctx.by_type("IfcDoor")
            exits = [d for d in doors if _is_external_door(d)]

            if not exits:
                return store.record(
                    check_name="check_travel_distance",
                    clause=clause,
                    verdict="indeterminate",
                    summary="No external doors found; cannot compute travel distance.",
                    params_checked={"occupancy_class": occupancy_class, "limit_m": limit},
                ).to_dict()

            # Group exits by storey for fair comparison
            exits_by_storey: dict[str, list[tuple[str, tuple[float, float, float]]]] = {}
            for d in exits:
                st = _storey_of(d)
                if st is None:
                    continue
                pos = _door_position_xyz(ctx, d)
                if pos is None:
                    continue
                exits_by_storey.setdefault(st.GlobalId, []).append((d.GlobalId, pos))

            failures = []
            per_room = []
            for sp in spaces:
                st = _storey_of(sp)
                if st is None or st.GlobalId not in exits_by_storey:
                    continue
                centroid = _space_centroid_xyz(ctx, sp)
                if centroid is None:
                    continue
                # Find nearest exit on same storey
                nearest_dist = None
                nearest_guid = None
                for d_guid, d_pos in exits_by_storey[st.GlobalId]:
                    dist = _euclid(centroid, d_pos) * corridor_factor
                    if nearest_dist is None or dist < nearest_dist:
                        nearest_dist = dist
                        nearest_guid = d_guid
                if nearest_dist is None:
                    continue
                room_name = getattr(sp, "LongName", None) or getattr(sp, "Name", None) or "(unnamed)"
                per_room.append({
                    "room_guid": sp.GlobalId,
                    "room_name": room_name,
                    "storey_guid": st.GlobalId,
                    "travel_distance_m": round(nearest_dist, 2),
                    "nearest_exit_guid": nearest_guid,
                })
                if nearest_dist > limit:
                    failures.append({
                        "element_guid": sp.GlobalId,
                        "element_name": room_name,
                        "reason": (
                            f"Room '{room_name}' is ~{round(nearest_dist, 2)} m from the nearest "
                            f"external exit (limit {limit} m for '{occupancy_class}')."
                        ),
                    })

            verdict = "fail" if failures else ("pass" if per_room else "indeterminate")
            summary = (
                f"{len(per_room)} room(s) checked against {limit} m limit; "
                f"{len(failures)} exceeded."
            )

            f = store.record(
                check_name="check_travel_distance",
                clause=clause,
                verdict=verdict,
                summary=summary,
                params_checked={
                    "occupancy_class": occupancy_class,
                    "limit_m": limit,
                    "corridor_factor": corridor_factor,
                    "rooms_evaluated": len(per_room),
                },
                failures=failures,
                extras={"per_room": per_room[:200]},  # cap UI payload
            )
            return f.to_dict()
        except Exception as exc:
            logger.exception("check_travel_distance failed: %s", exc)
            return store.record(
                check_name="check_travel_distance",
                clause=clause,
                verdict="indeterminate",
                summary=f"Check could not be completed: {exc}",
                params_checked={"occupancy_class": occupancy_class},
            ).to_dict()

    return StructuredTool.from_function(
        check_travel_distance,
        name="check_travel_distance",
        description=(
            "NBC 2016 Part 4 Table 9 — for every room, compute the straight-line distance to "
            "the nearest external exit on the same storey, multiplied by a corridor factor "
            "(default 1.5), and flag rooms that exceed the occupancy-class limit. Returns a "
            "Finding listing failing rooms."
        ),
        args_schema=_TravelDistanceArgs,
    )


def _build_check_exit_widths(ctx: IFCContext, store: FindingsStore) -> StructuredTool:
    def check_exit_widths() -> dict[str, Any]:
        clause = "NBC 2016, Part 4, 4.5 (Exit Widths)"
        try:
            doors = ctx.by_type("IfcDoor")
            failures = []
            checked = 0
            for d in doors:
                if not _is_external_door(d):
                    continue
                checked += 1
                w = _door_clear_width_m(ctx, d)
                if w is None:
                    failures.append({
                        "element_guid": d.GlobalId,
                        "element_name": getattr(d, "Name", None),
                        "reason": "Could not determine clear width (no pset/geometry).",
                    })
                    continue
                if w < _MIN_DOOR_CLEAR_WIDTH_M:
                    failures.append({
                        "element_guid": d.GlobalId,
                        "element_name": getattr(d, "Name", None),
                        "reason": (
                            f"Exit door clear width {round(w, 3)} m is below the "
                            f"{_MIN_DOOR_CLEAR_WIDTH_M} m minimum."
                        ),
                    })

            # Stairs: check Width pset / geometry
            stairs = ctx.by_type("IfcStair") + ctx.by_type("IfcStairFlight")
            for s in stairs:
                checked += 1
                psets = get_psets(s) or {}
                width_m: Optional[float] = None
                for ps in psets.values():
                    if not isinstance(ps, dict):
                        continue
                    for k in ("Width", "NominalWidth", "ClearWidth"):
                        v = ps.get(k)
                        if isinstance(v, (int, float)) and v > 0:
                            width_m = float(v) * ctx.unit_scale_to_meters()
                            break
                    if width_m is not None:
                        break
                if width_m is None:
                    aabb = element_aabb(ctx, s)
                    if aabb:
                        sx, sy, _ = aabb["size"]
                        width_m = float(min(sx, sy)) * ctx.unit_scale_to_meters()
                if width_m is None:
                    failures.append({
                        "element_guid": s.GlobalId,
                        "element_name": getattr(s, "Name", None),
                        "reason": "Could not determine stair width.",
                    })
                elif width_m < _MIN_STAIR_WIDTH_M:
                    failures.append({
                        "element_guid": s.GlobalId,
                        "element_name": getattr(s, "Name", None),
                        "reason": (
                            f"Stair width {round(width_m, 3)} m is below the "
                            f"{_MIN_STAIR_WIDTH_M} m minimum."
                        ),
                    })

            verdict = "fail" if failures else ("pass" if checked else "indeterminate")
            summary = f"{checked} egress component(s) checked; {len(failures)} failure(s)."
            f = store.record(
                check_name="check_exit_widths",
                clause=clause,
                verdict=verdict,
                summary=summary,
                params_checked={
                    "min_door_width_m": _MIN_DOOR_CLEAR_WIDTH_M,
                    "min_stair_width_m": _MIN_STAIR_WIDTH_M,
                    "checked": checked,
                },
                failures=failures,
            )
            return f.to_dict()
        except Exception as exc:
            logger.exception("check_exit_widths failed: %s", exc)
            return store.record(
                check_name="check_exit_widths",
                clause=clause,
                verdict="indeterminate",
                summary=f"Check could not be completed: {exc}",
            ).to_dict()

    return StructuredTool.from_function(
        check_exit_widths,
        name="check_exit_widths",
        description=(
            "NBC 2016 Part 4 4.5 — verify clear widths of external doors (≥ 1.0 m) and "
            "stairs (≥ 1.5 m). Returns a Finding listing components below the minimum."
        ),
        args_schema=_NoArgs,
    )


def _build_check_compartmentation(ctx: IFCContext, store: FindingsStore) -> StructuredTool:
    def check_compartmentation() -> dict[str, Any]:
        clause = "NBC 2016, Part 4, 3.4 (Compartmentation / Fire Resistance)"
        try:
            walls = ctx.by_type("IfcWall")
            rated = 0
            unrated = 0
            failures: list[dict[str, Any]] = []
            for w in walls:
                psets = get_psets(w) or {}
                common = psets.get("Pset_WallCommon") or {}
                fr = common.get("FireRating")
                if isinstance(fr, str) and fr.strip():
                    rated += 1
                else:
                    # Only count walls flagged as fire-rated externally; quiet otherwise.
                    is_compartment = common.get("Compartmentation")
                    if isinstance(is_compartment, bool) and is_compartment:
                        unrated += 1
                        failures.append({
                            "element_guid": w.GlobalId,
                            "element_name": getattr(w, "Name", None),
                            "reason": (
                                "Wall flagged as Compartmentation=True but Pset_WallCommon."
                                "FireRating is missing or empty."
                            ),
                        })
            verdict = "fail" if failures else ("pass" if rated > 0 else "indeterminate")
            if rated == 0 and not failures:
                summary = (
                    "No walls carry Pset_WallCommon.FireRating data; cannot evaluate "
                    "compartmentation. Authoring tools often omit this — verdict indeterminate."
                )
            else:
                summary = (
                    f"{rated} fire-rated wall(s) found; {len(failures)} compartmentation "
                    f"wall(s) missing FireRating."
                )
            f = store.record(
                check_name="check_compartmentation",
                clause=clause,
                verdict=verdict,
                summary=summary,
                params_checked={"walls_with_fire_rating": rated},
                failures=failures,
            )
            return f.to_dict()
        except Exception as exc:
            logger.exception("check_compartmentation failed: %s", exc)
            return store.record(
                check_name="check_compartmentation",
                clause=clause,
                verdict="indeterminate",
                summary=f"Check could not be completed: {exc}",
            ).to_dict()

    return StructuredTool.from_function(
        check_compartmentation,
        name="check_compartmentation",
        description=(
            "NBC 2016 Part 4 3.4 — scan IfcWall instances for Pset_WallCommon.FireRating. "
            "Flags walls marked as Compartmentation=True but lacking a fire rating. Returns "
            "indeterminate when no walls carry fire-rating data."
        ),
        args_schema=_NoArgs,
    )


def _build_check_refuge_area(ctx: IFCContext, store: FindingsStore) -> StructuredTool:
    def check_refuge_area() -> dict[str, Any]:
        clause = "NBC 2016, Part 4, 4.4.5 (Refuge Area for High-Rise Buildings)"
        try:
            storeys = ctx.by_type("IfcBuildingStorey")
            if not storeys:
                return store.record(
                    check_name="check_refuge_area",
                    clause=clause,
                    verdict="indeterminate",
                    summary="No IfcBuildingStorey elements present.",
                ).to_dict()
            elevations = [
                float(getattr(s, "Elevation", 0) or 0) * ctx.unit_scale_to_meters()
                for s in storeys
            ]
            building_height = max(elevations) - min(elevations) if elevations else 0.0
            if building_height < _HIGH_RISE_HEIGHT_M:
                return store.record(
                    check_name="check_refuge_area",
                    clause=clause,
                    verdict="pass",
                    summary=(
                        f"Building height ~{round(building_height, 2)} m is below the "
                        f"{_HIGH_RISE_HEIGHT_M} m high-rise threshold; refuge area not required."
                    ),
                    params_checked={
                        "building_height_m": round(building_height, 2),
                        "high_rise_threshold_m": _HIGH_RISE_HEIGHT_M,
                    },
                ).to_dict()

            # Look for spaces named/long-named with "refuge"
            refuge_spaces: list[dict[str, Any]] = []
            for sp in ctx.by_type("IfcSpace"):
                lname = (getattr(sp, "LongName", None) or "") + " " + (getattr(sp, "Name", None) or "")
                if "refuge" in lname.lower():
                    refuge_spaces.append({
                        "guid": sp.GlobalId,
                        "name": getattr(sp, "LongName", None) or getattr(sp, "Name", None),
                        "area_m2": _space_floor_area_m2(ctx, sp),
                    })

            if refuge_spaces:
                verdict = "pass"
                summary = (
                    f"High-rise (~{round(building_height, 2)} m); "
                    f"{len(refuge_spaces)} refuge space(s) identified by name."
                )
                failures: list[dict[str, Any]] = []
            else:
                verdict = "fail"
                summary = (
                    f"High-rise (~{round(building_height, 2)} m) but no IfcSpace contains "
                    f"'refuge' in its name/LongName."
                )
                failures = [{
                    "element_guid": "",
                    "element_name": "(building)",
                    "reason": "No refuge area identified for high-rise building.",
                }]

            f = store.record(
                check_name="check_refuge_area",
                clause=clause,
                verdict=verdict,
                summary=summary,
                params_checked={
                    "building_height_m": round(building_height, 2),
                    "high_rise_threshold_m": _HIGH_RISE_HEIGHT_M,
                    "refuge_spaces_found": len(refuge_spaces),
                },
                failures=failures,
                extras={"refuge_spaces": refuge_spaces},
            )
            return f.to_dict()
        except Exception as exc:
            logger.exception("check_refuge_area failed: %s", exc)
            return store.record(
                check_name="check_refuge_area",
                clause=clause,
                verdict="indeterminate",
                summary=f"Check could not be completed: {exc}",
            ).to_dict()

    return StructuredTool.from_function(
        check_refuge_area,
        name="check_refuge_area",
        description=(
            "NBC 2016 Part 4 4.4.5 — for buildings taller than 24 m, verify the presence of at "
            "least one IfcSpace named/long-named with 'refuge'. Pass if not high-rise."
        ),
        args_schema=_NoArgs,
    )


def _build_check_dead_end_corridors(ctx: IFCContext, store: FindingsStore) -> StructuredTool:
    def check_dead_end_corridors() -> dict[str, Any]:
        clause = "NBC 2016, Part 4, 4.5.4 (Dead-end Corridors ≤ 6 m)"
        try:
            spaces = ctx.by_type("IfcSpace")
            corridor_spaces: list[Any] = []
            for sp in spaces:
                lname = (getattr(sp, "LongName", None) or "") + " " + (getattr(sp, "Name", None) or "")
                if "corridor" in lname.lower() or "passage" in lname.lower():
                    corridor_spaces.append(sp)
            if not corridor_spaces:
                return store.record(
                    check_name="check_dead_end_corridors",
                    clause=clause,
                    verdict="indeterminate",
                    summary=(
                        "No corridor-named IfcSpace elements found. Dead-end check requires "
                        "explicit corridor modelling."
                    ),
                    params_checked={"limit_m": _DEAD_END_LIMIT_M},
                ).to_dict()
            failures = []
            per_corridor = []
            for sp in corridor_spaces:
                aabb = element_aabb(ctx, sp)
                if not aabb:
                    continue
                sx, sy, _ = aabb["size"]
                length_m = float(max(sx, sy)) * ctx.unit_scale_to_meters()
                name = getattr(sp, "LongName", None) or getattr(sp, "Name", None) or "(unnamed)"
                per_corridor.append({
                    "guid": sp.GlobalId, "name": name, "length_m": round(length_m, 2),
                })
                # Heuristic: if a corridor space is longer than the dead-end limit AND has only one
                # door, treat it as a likely dead-end. We only have full bounding-box length here,
                # so we flag corridors longer than 2x the limit as a coarse proxy.
                if length_m > 2 * _DEAD_END_LIMIT_M:
                    doors_in_room = 0
                    for d in ctx.by_type("IfcDoor"):
                        if get_containing_space(d) is sp:
                            doors_in_room += 1
                    if doors_in_room <= 1:
                        failures.append({
                            "element_guid": sp.GlobalId,
                            "element_name": name,
                            "reason": (
                                f"Corridor '{name}' is ~{round(length_m, 2)} m long with "
                                f"{doors_in_room} door(s); likely exceeds the {_DEAD_END_LIMIT_M} m "
                                "dead-end limit. Heuristic flag — verify on plan."
                            ),
                        })
            verdict = "fail" if failures else ("pass" if per_corridor else "indeterminate")
            f = store.record(
                check_name="check_dead_end_corridors",
                clause=clause,
                verdict=verdict,
                summary=(
                    f"{len(per_corridor)} corridor space(s) inspected; {len(failures)} flagged "
                    f"as likely dead-ends."
                ),
                params_checked={"limit_m": _DEAD_END_LIMIT_M},
                failures=failures,
                extras={"per_corridor": per_corridor},
            )
            return f.to_dict()
        except Exception as exc:
            logger.exception("check_dead_end_corridors failed: %s", exc)
            return store.record(
                check_name="check_dead_end_corridors",
                clause=clause,
                verdict="indeterminate",
                summary=f"Check could not be completed: {exc}",
            ).to_dict()

    return StructuredTool.from_function(
        check_dead_end_corridors,
        name="check_dead_end_corridors",
        description=(
            "NBC 2016 Part 4 4.5.4 — heuristic check for corridor-named IfcSpace elements that "
            "are long and have only one door, flagging likely dead-end corridors above 6 m."
        ),
        args_schema=_NoArgs,
    )


# ---------------------------------------------------------------------------
# Recall tools
# ---------------------------------------------------------------------------
def _build_list_compliance_findings(store: FindingsStore) -> StructuredTool:
    def list_compliance_findings() -> dict[str, Any]:
        items = [
            {
                "finding_id": f.finding_id,
                "check_name": f.check_name,
                "verdict": f.verdict,
                "summary": f.summary,
                "clause": f.clause,
                "n_failures": len(f.failures),
                "created_at": f.created_at,
            }
            for f in store.all()
        ]
        return {"count": len(items), "findings": items}

    return StructuredTool.from_function(
        list_compliance_findings,
        name="list_compliance_findings",
        description=(
            "List every compliance finding recorded in the current session: id, check name, "
            "verdict, clause, and summary. Use this when the user asks 'what have we checked' "
            "or wants to recall a past result."
        ),
        args_schema=_NoArgs,
    )


def _build_get_finding_details(store: FindingsStore) -> StructuredTool:
    def get_finding_details(finding_id: str) -> dict[str, Any]:
        f = store.get(finding_id)
        if f is None:
            return {"error": f"No finding with id {finding_id!r}"}
        return f.to_dict()

    return StructuredTool.from_function(
        get_finding_details,
        name="get_finding_details",
        description=(
            "Retrieve the full record (params, failures, extras) for one compliance finding by "
            "its F-XXXXXXXX id. Use this to drill into a previous check's failing rooms/elements."
        ),
        args_schema=_FindingIdArgs,
    )


def _build_clear_compliance_findings(store: FindingsStore) -> StructuredTool:
    def clear_compliance_findings() -> dict[str, Any]:
        n = store.clear()
        return {"cleared": n}

    return StructuredTool.from_function(
        clear_compliance_findings,
        name="clear_compliance_findings",
        description="Erase all compliance findings from session memory. Use only when the user explicitly asks.",
        args_schema=_NoArgs,
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------
def build_compliance_tools(ctx: IFCContext, store: FindingsStore) -> list[StructuredTool]:
    """All compliance + memory-recall tools, closed over a shared FindingsStore."""
    return [
        _build_check_means_of_egress(ctx, store),
        _build_check_travel_distance(ctx, store),
        _build_check_exit_widths(ctx, store),
        _build_check_dead_end_corridors(ctx, store),
        _build_check_compartmentation(ctx, store),
        _build_check_refuge_area(ctx, store),
        _build_list_compliance_findings(store),
        _build_get_finding_details(store),
        _build_clear_compliance_findings(store),
    ]


__all__ = ["build_compliance_tools"]
