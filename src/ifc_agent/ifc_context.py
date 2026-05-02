"""IFCContext — the single source of truth for the loaded IFC model.

All tools accept an IFCContext (via closure) instead of opening the file themselves.
This keeps file I/O centralised and lets us cache expensive things like the geometry
iterator settings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import ifcopenshell

logger = logging.getLogger(__name__)


@dataclass
class IFCContext:
    """Holds the loaded IFC file plus cached helpers.

    Build it once per uploaded file with ``IFCContext.from_path(path)``.
    """

    file_path: Path
    model: ifcopenshell.file
    schema: str
    _geom_settings: Any = field(default=None, repr=False)
    _unit_scale: Optional[float] = field(default=None, repr=False)

    # ---------- Construction --------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path) -> "IFCContext":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"IFC file not found: {path}")
        logger.info("Opening IFC file: %s", path)
        model = ifcopenshell.open(str(path))
        return cls(file_path=path, model=model, schema=model.schema)

    # ---------- Cached helpers ------------------------------------------------
    def geom_settings(self) -> Any:
        """Lazy IfcOpenShell geometry settings — world coords, no openings."""
        if self._geom_settings is None:
            import ifcopenshell.geom

            settings = ifcopenshell.geom.settings()
            # Use named-API for portability across 0.7.x and 0.8.x:
            try:
                settings.set("use-world-coords", True)
            except Exception:
                # Older API
                try:
                    settings.set(settings.USE_WORLD_COORDS, True)
                except Exception:
                    pass
            self._geom_settings = settings
        return self._geom_settings

    def unit_scale_to_meters(self) -> float:
        """Multiplier converting model length units → metres.

        e.g. if the model is in millimetres, returns 0.001.
        """
        if self._unit_scale is None:
            try:
                import ifcopenshell.util.unit as ifc_unit

                self._unit_scale = ifc_unit.calculate_unit_scale(self.model)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not determine unit scale, assuming metres: %s", exc)
                self._unit_scale = 1.0
        return float(self._unit_scale)

    # ---------- Convenience lookups ------------------------------------------
    def by_guid(self, guid: str) -> Any:
        """Find an entity by its IfcGloballyUniqueId (GlobalId), or raise."""
        ent = self.model.by_guid(guid)
        if ent is None:
            raise ValueError(f"No element found with GlobalId {guid!r}")
        return ent

    def by_type(self, ifc_class: str) -> list[Any]:
        """All entities of an IFC class. Tolerates leading 'Ifc' or not."""
        cls = ifc_class if ifc_class.lower().startswith("ifc") else f"Ifc{ifc_class}"
        try:
            return list(self.model.by_type(cls))
        except RuntimeError:
            # IfcOpenShell raises RuntimeError for unknown classes
            return []
