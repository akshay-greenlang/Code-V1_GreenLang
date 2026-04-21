# -*- coding: utf-8 -*-
"""
Unit conversion graph (Phase F5).

Replaces the flat lookup tables in ``greenlang/factors/ontology/units.py``
with a directed conversion graph so callers can chain conversions across
different dimensions (energy, mass, volume) via BFS.

Example::

    graph = UnitGraph()
    # Convert 1 MMBtu of natural gas to kg of natural gas.
    kg = graph.convert(value=1.0, from_unit="MMBtu", to_unit="kg",
                       material="natural_gas")

Supported node types:

- Dimensionless — kg, g, t, L, m3, J, kJ, MJ, GJ, kWh, MWh, GWh, Btu, MMBtu, MMkcal
- Material-linked — kg ↔ L / m3 via density, GJ ↔ kg via heating value

The graph is built once at import; edges are registered via
``register_edge`` and resolved by ``shortest_path``.

Design intent: this replaces custom per-conversion code scattered in
parsers + Scope Engine. New unit combos get added by declaring an edge.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class Edge:
    source_unit: str
    target_unit: str
    factor: float                            # how to multiply value going source -> target
    requires_material: bool = False          # e.g. density
    requires_heating_value: bool = False
    requires_moisture: bool = False          # wet-biomass correction
    requires_oxidation: bool = False         # combustion completeness
    note: str = ""


class UnitConversionError(ValueError):
    pass


class UnitGraph:
    """Directed unit conversion graph."""

    def __init__(self) -> None:
        self._edges: Dict[str, List[Edge]] = {}
        self._bootstrap_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_edge(self, edge: Edge) -> None:
        self._edges.setdefault(edge.source_unit.lower(), []).append(edge)

    def shortest_path(
        self, *, from_unit: str, to_unit: str
    ) -> Optional[List[Edge]]:
        """BFS shortest path of edges from ``from_unit`` to ``to_unit``."""
        a, b = from_unit.lower(), to_unit.lower()
        if a == b:
            return []
        visited = {a}
        queue: deque[Tuple[str, List[Edge]]] = deque([(a, [])])
        while queue:
            current, path = queue.popleft()
            for edge in self._edges.get(current, []):
                if edge.target_unit.lower() in visited:
                    continue
                new_path = path + [edge]
                if edge.target_unit.lower() == b:
                    return new_path
                visited.add(edge.target_unit.lower())
                queue.append((edge.target_unit.lower(), new_path))
        return None

    def convert(
        self,
        *,
        value: float,
        from_unit: str,
        to_unit: str,
        material: Optional[str] = None,
        heating_value_mj_per_kg: Optional[float] = None,
        density_kg_per_l: Optional[float] = None,
        moisture_fraction: Optional[float] = None,
        oxidation_factor: Optional[float] = None,
        latent_heat_water_mj_per_kg: float = 2.45,
    ) -> float:
        """Convert ``value`` from one unit to another via the graph.

        Args:
            value: Numeric quantity to convert.
            from_unit, to_unit: Unit codes.
            material: Material code (enables material-linked edges).
            heating_value_mj_per_kg: LHV/HHV for mass<->energy edges.
            density_kg_per_l: Density for mass<->volume edges.
            moisture_fraction: Wet-biomass moisture fraction (0..1). If
                set, applies the as-received correction
                ``LHV_ar = LHV_dry * (1-M) - L*M`` to the heating value.
            oxidation_factor: Combustion completeness (0..1). If set,
                multiplies the effective heating value to reflect
                incomplete oxidation.
            latent_heat_water_mj_per_kg: Latent heat of vaporisation
                (default 2.45 MJ/kg).
        """
        path = self.shortest_path(from_unit=from_unit, to_unit=to_unit)
        if path is None:
            raise UnitConversionError(
                "no conversion path from %r -> %r" % (from_unit, to_unit)
            )
        current = value
        for edge in path:
            if edge.requires_material and material is None and density_kg_per_l is None:
                raise UnitConversionError(
                    "edge %s -> %s requires material context" % (edge.source_unit, edge.target_unit)
                )
            if edge.requires_heating_value and heating_value_mj_per_kg is None:
                raise UnitConversionError(
                    "edge %s -> %s requires heating_value_mj_per_kg"
                    % (edge.source_unit, edge.target_unit)
                )
            factor = edge.factor
            # Material-dependent edges: multiply by the appropriate property.
            if edge.requires_heating_value:
                hv = float(heating_value_mj_per_kg)
                if moisture_fraction is not None:
                    if not 0.0 <= moisture_fraction < 1.0:
                        raise UnitConversionError(
                            "moisture_fraction must be in [0,1), got %r"
                            % moisture_fraction
                        )
                    hv = max(
                        hv * (1.0 - moisture_fraction)
                        - latent_heat_water_mj_per_kg * moisture_fraction,
                        0.0,
                    )
                if oxidation_factor is not None:
                    if not 0.0 <= oxidation_factor <= 1.0:
                        raise UnitConversionError(
                            "oxidation_factor must be in [0,1], got %r"
                            % oxidation_factor
                        )
                    hv = hv * oxidation_factor
                # mass -> energy: kg x MJ/kg = MJ; energy -> mass uses reciprocal
                if edge.source_unit.lower() in {"kg", "g", "t"}:
                    factor = hv * edge.factor
                else:
                    # energy -> mass: divide by HV
                    if hv == 0:
                        raise UnitConversionError(
                            "effective heating value is zero; cannot invert"
                        )
                    factor = edge.factor / hv
            elif edge.requires_material and density_kg_per_l is not None:
                # mass <-> volume edge
                if edge.source_unit.lower() in {"kg", "g", "t"}:
                    # going mass -> volume: divide by density
                    factor = (1.0 / density_kg_per_l) * edge.factor
                else:
                    # going volume -> mass: multiply by density
                    factor = density_kg_per_l * edge.factor
            current = current * factor
        return current

    def edges(self) -> List[Edge]:
        out: List[Edge] = []
        for edges in self._edges.values():
            out.extend(edges)
        return out

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_defaults(self) -> None:
        # --- Mass ---
        self._symmetric("t", "kg", 1000.0)
        self._symmetric("kg", "g", 1000.0)
        self._symmetric("kg", "lb", 1.0 / 0.45359237)
        # --- Volume ---
        self._symmetric("m3", "L", 1000.0)
        self._symmetric("L", "mL", 1000.0)
        self._symmetric("L", "gallon", 1.0 / 3.78541)
        # --- Energy ---
        self._symmetric("MJ", "kJ", 1000.0)
        self._symmetric("GJ", "MJ", 1000.0)
        self._symmetric("MJ", "J", 1_000_000.0)
        self._symmetric("kWh", "MJ", 3.6)
        self._symmetric("MWh", "kWh", 1000.0)
        self._symmetric("GWh", "MWh", 1000.0)
        self._symmetric("MMBtu", "kWh", 293.07107017)       # 1 MMBtu = 293.071 kWh
        self._symmetric("MMBtu", "GJ", 1.0550559)
        self._symmetric("Btu", "MMBtu", 1.0 / 1_000_000)
        # --- Material-dependent edges ---
        # mass ↔ volume needs density
        self.register_edge(Edge(source_unit="kg", target_unit="L",
                                factor=1.0, requires_material=True,
                                note="mass → volume via density"))
        self.register_edge(Edge(source_unit="L", target_unit="kg",
                                factor=1.0, requires_material=True,
                                note="volume → mass via density"))
        # mass ↔ energy needs LHV
        self.register_edge(Edge(source_unit="kg", target_unit="MJ",
                                factor=1.0, requires_heating_value=True,
                                note="mass → energy via LHV"))
        self.register_edge(Edge(source_unit="MJ", target_unit="kg",
                                factor=1.0, requires_heating_value=True,
                                note="energy → mass via LHV (reciprocal)"))

    def _symmetric(self, a: str, b: str, factor_ab: float) -> None:
        """Register both a→b and b→a edges."""
        self.register_edge(Edge(source_unit=a, target_unit=b, factor=factor_ab))
        self.register_edge(Edge(source_unit=b, target_unit=a, factor=1.0 / factor_ab))


# Process-wide singleton; callers that need a custom graph instantiate their own.
DEFAULT_GRAPH = UnitGraph()


def convert(
    value: float, from_unit: str, to_unit: str, **kwargs
) -> float:
    """Convenience wrapper on the default graph."""
    return DEFAULT_GRAPH.convert(
        value=value, from_unit=from_unit, to_unit=to_unit, **kwargs
    )


__all__ = ["DEFAULT_GRAPH", "Edge", "UnitConversionError", "UnitGraph", "convert"]
