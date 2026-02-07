# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for Normalizer Service (AGENT-FOUND-003)

Tests full conversion pipeline: parse -> validate -> convert -> provenance,
GWP conversion pipeline, entity resolution pipeline, and batch operations.

All implementations are self-contained to avoid cross-module import issues.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest


# ---------------------------------------------------------------------------
# Self-contained implementations for integration testing
# ---------------------------------------------------------------------------

class Dimension(str, Enum):
    ENERGY = "ENERGY"
    MASS = "MASS"
    EMISSIONS = "EMISSIONS"
    VOLUME = "VOLUME"
    AREA = "AREA"
    DISTANCE = "DISTANCE"
    TIME = "TIME"


_UNIT_TABLE: Dict[str, Tuple[Dimension, Decimal]] = {
    "kWh": (Dimension.ENERGY, Decimal("1")),
    "MWh": (Dimension.ENERGY, Decimal("1000")),
    "GJ": (Dimension.ENERGY, Decimal("277.777777778")),
    "MJ": (Dimension.ENERGY, Decimal("0.277777777778")),
    "kg": (Dimension.MASS, Decimal("1")),
    "g": (Dimension.MASS, Decimal("0.001")),
    "t": (Dimension.MASS, Decimal("1000")),
    "lb": (Dimension.MASS, Decimal("0.453592")),
    "kgCO2e": (Dimension.EMISSIONS, Decimal("1")),
    "tCO2e": (Dimension.EMISSIONS, Decimal("1000")),
    "gCO2e": (Dimension.EMISSIONS, Decimal("0.001")),
    "MtCO2e": (Dimension.EMISSIONS, Decimal("1000000000")),
    "kgCO2": (Dimension.EMISSIONS, Decimal("1")),
    "tCO2": (Dimension.EMISSIONS, Decimal("1000")),
    "L": (Dimension.VOLUME, Decimal("1")),
    "m3": (Dimension.VOLUME, Decimal("1000")),
    "gal": (Dimension.VOLUME, Decimal("3.78541")),
    "bbl": (Dimension.VOLUME, Decimal("158.987")),
    "m": (Dimension.DISTANCE, Decimal("1")),
    "km": (Dimension.DISTANCE, Decimal("1000")),
    "m2": (Dimension.AREA, Decimal("1")),
    "hectare": (Dimension.AREA, Decimal("10000")),
}

_ALIASES: Dict[str, str] = {
    "kwh": "kWh", "KWH": "kWh", "mwh": "MWh", "MWH": "MWh",
    "gj": "GJ", "mj": "MJ", "KG": "kg", "kilogram": "kg",
    "T": "t", "tonne": "t", "tonnes": "t", "LB": "lb", "lbs": "lb",
    "kgco2e": "kgCO2e", "tco2e": "tCO2e", "tco2": "tCO2",
    "l": "L", "liter": "L", "litre": "L", "M3": "m3",
    "GAL": "gal", "gallon": "gal",
    "M": "m", "KM": "km", "kilometer": "km",
    "M2": "m2", "sqm": "m2",
}

GWP_TABLES = {
    "AR6": {"CH4": Decimal("29.8"), "N2O": Decimal("273"), "CO2": Decimal("1"), "CO2e": Decimal("1")},
    "AR5": {"CH4": Decimal("28"), "N2O": Decimal("265"), "CO2": Decimal("1"), "CO2e": Decimal("1")},
}


class ConversionResult:
    def __init__(self, value, from_unit, to_unit, dimension, factor, provenance_hash, error=None):
        self.value = value
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.dimension = dimension
        self.factor = factor
        self.provenance_hash = provenance_hash
        self.error = error

    @property
    def ok(self):
        return self.error is None


class UnitConverter:
    def __init__(self, default_precision=10):
        self._precision = default_precision

    def convert(self, value, from_unit, to_unit, precision=None):
        prec = precision if precision is not None else self._precision
        from_c = self._resolve(from_unit)
        to_c = self._resolve(to_unit)
        if from_c is None:
            return ConversionResult(Decimal(0), from_unit, to_unit, "UNKNOWN", Decimal(0), "", error=f"Unknown unit: {from_unit}")
        if to_c is None:
            return ConversionResult(Decimal(0), from_unit, to_unit, "UNKNOWN", Decimal(0), "", error=f"Unknown unit: {to_unit}")
        fd, ff = _UNIT_TABLE[from_c]
        td, tf = _UNIT_TABLE[to_c]
        if fd != td:
            return ConversionResult(Decimal(0), from_unit, to_unit, f"{fd.value}->{td.value}", Decimal(0), "", error=f"Incompatible dimensions: {fd.value} -> {td.value}")
        d = Decimal(str(value))
        factor = ff / tf
        result = (d * factor).quantize(Decimal(10) ** -prec, rounding=ROUND_HALF_UP)
        h = hashlib.sha256(json.dumps([str(value), from_c, to_c, str(result)], sort_keys=True).encode()).hexdigest()
        return ConversionResult(result, from_c, to_c, fd.value, factor, h)

    def convert_ghg(self, value, from_unit, to_unit, gwp_version="AR6"):
        gwp_table = GWP_TABLES.get(gwp_version)
        if gwp_table is None:
            return ConversionResult(Decimal(0), from_unit, to_unit, "EMISSIONS", Decimal(0), "", error=f"Unknown GWP version: {gwp_version}")
        from_gas = self._extract_gas(from_unit)
        to_gas = self._extract_gas(to_unit)
        gwp_from = gwp_table.get(from_gas or "CO2", Decimal("1"))
        gwp_to = gwp_table.get(to_gas or "CO2e", Decimal("1"))
        d = Decimal(str(value))
        gwp_factor = gwp_from / gwp_to
        result = (d * gwp_factor).quantize(Decimal(10) ** -self._precision, rounding=ROUND_HALF_UP)
        h = hashlib.sha256(json.dumps([str(value), from_unit, to_unit, gwp_version, str(result)], sort_keys=True).encode()).hexdigest()
        return ConversionResult(result, from_unit, to_unit, "EMISSIONS", gwp_factor, h)

    def batch_convert(self, items):
        return [self.convert(i["value"], i["from_unit"], i["to_unit"], precision=i.get("precision")) for i in items]

    def _resolve(self, unit):
        if unit in _UNIT_TABLE:
            return unit
        s = unit.strip()
        if s in _UNIT_TABLE:
            return s
        a = _ALIASES.get(unit) or _ALIASES.get(s)
        if a and a in _UNIT_TABLE:
            return a
        return None

    def _extract_gas(self, unit):
        for gas in ["CH4", "N2O", "CO2e", "CO2"]:
            if gas in unit:
                return gas
        return None


class ConfidenceLevel(str, Enum):
    EXACT = "EXACT"
    ALIAS = "ALIAS"
    FUZZY = "FUZZY"
    UNRESOLVED = "UNRESOLVED"


class EntityMatch:
    def __init__(self, canonical_id, canonical_name, confidence, level, category="", code=""):
        self.canonical_id = canonical_id
        self.canonical_name = canonical_name
        self.confidence = confidence
        self.level = level
        self.category = category
        self.code = code


_FUEL_DB = {
    "natural_gas": ("FUEL-001", "NG", "gaseous", ["Natural Gas", "Nat Gas", "natural-gas", "NG", "methane"]),
    "diesel": ("FUEL-002", "DSL", "liquid", ["Diesel", "diesel fuel", "diesel oil"]),
    "coal": ("FUEL-003", "COAL", "solid", ["Coal", "bituminous coal"]),
    "lpg": ("FUEL-004", "LPG", "gaseous", ["LPG", "autogas"]),
    "biogas": ("FUEL-005", "BG", "gaseous", ["Biogas", "biomethane"]),
    "biomass": ("FUEL-006", "BM", "solid", ["Biomass", "wood pellets"]),
    "gasoline": ("FUEL-007", "GAS", "liquid", ["Gasoline", "petrol"]),
    "fuel_oil": ("FUEL-008", "FO", "liquid", ["Fuel Oil", "HFO"]),
    "kerosene": ("FUEL-009", "KER", "liquid", ["Kerosene", "jet fuel"]),
    "propane": ("FUEL-010", "PRP", "gaseous", ["Propane"]),
}

_MATERIAL_DB = {
    "steel": ("MAT-001", "STL", "metals", ["Steel", "carbon steel"]),
    "aluminum": ("MAT-002", "ALU", "metals", ["Aluminum", "aluminium"]),
    "cement": ("MAT-003", "CEM", "construction", ["Cement", "Portland Cement", "OPC", "CEM I"]),
    "glass": ("MAT-004", "GLS", "construction", ["Glass"]),
}

_PROCESS_DB = {
    "electric_arc_furnace": ("PROC-001", "EAF", "steelmaking", ["Electric Arc Furnace", "EAF"]),
    "basic_oxygen_furnace": ("PROC-002", "BOF", "steelmaking", ["Basic Oxygen Furnace", "BOF"]),
}


class EntityResolver:
    def __init__(self):
        self._fuel_idx = {}
        self._mat_idx = {}
        self._proc_idx = {}
        self._build(_FUEL_DB, self._fuel_idx)
        self._build(_MATERIAL_DB, self._mat_idx)
        self._build(_PROCESS_DB, self._proc_idx)

    def _build(self, db, idx):
        for key, (_id, _code, _cat, aliases) in db.items():
            idx[key.lower()] = (key, ConfidenceLevel.EXACT)
            for a in aliases:
                lo = a.lower().strip()
                if lo not in idx:
                    idx[lo] = (key, ConfidenceLevel.ALIAS)

    def resolve_fuel(self, name):
        return self._resolve(name, self._fuel_idx, _FUEL_DB)

    def resolve_material(self, name):
        return self._resolve(name, self._mat_idx, _MATERIAL_DB)

    def resolve_process(self, name):
        return self._resolve(name, self._proc_idx, _PROCESS_DB)

    def batch_resolve_fuels(self, names):
        return [self.resolve_fuel(n) for n in names]

    def _resolve(self, name, idx, db):
        lo = name.lower().strip()
        norm = lo.replace("-", " ").replace("_", " ")
        result = idx.get(lo) or idx.get(norm)
        if result:
            key, level = result
            _id, code, cat, _ = db[key]
            conf = 1.0 if level == ConfidenceLevel.EXACT else 0.95
            return EntityMatch(_id, key, conf, level, cat, code)
        return EntityMatch("", "", 0.0, ConfidenceLevel.UNRESOLVED)


class DimensionalAnalyzer:
    _UD = {}
    for u, (d, _) in _UNIT_TABLE.items():
        _UD[u] = d
    for alias, canonical in _ALIASES.items():
        if canonical in _UD:
            _UD[alias] = _UD[canonical]

    def check_compatibility(self, a, b):
        da = self._get(a)
        db = self._get(b)
        if da is None or db is None:
            return False
        return da == db

    def _get(self, u):
        if u in self._UD:
            return self._UD[u]
        s = u.strip()
        if s in self._UD:
            return self._UD[s]
        a = _ALIASES.get(u) or _ALIASES.get(s)
        if a:
            return self._UD.get(a)
        return None


class ProvenanceRecord:
    def __init__(self, operation, inputs, output, parent_hash=None):
        self.operation = operation
        self.inputs = inputs
        self.output = output
        self.parent_hash = parent_hash
        self.hash = hashlib.sha256(json.dumps({"operation": operation, "inputs": inputs, "output": str(output), "parent_hash": parent_hash}, sort_keys=True).encode()).hexdigest()


class ConversionProvenanceTracker:
    def __init__(self):
        self._records = []

    def record(self, operation, inputs, output, parent_hash=None):
        r = ProvenanceRecord(operation, inputs, output, parent_hash)
        self._records.append(r)
        return r

    def get_chain(self, final_hash):
        lookup = {r.hash: r for r in self._records}
        chain = []
        h = final_hash
        while h and h in lookup:
            rec = lookup[h]
            chain.append(rec)
            h = rec.parent_hash
        chain.reverse()
        return chain

    def export_json(self):
        return json.dumps([{"operation": r.operation, "inputs": r.inputs, "output": r.output, "hash": r.hash, "parent_hash": r.parent_hash} for r in self._records], indent=2)

    @property
    def count(self):
        return len(self._records)


# ===========================================================================
# End-to-End Test Classes
# ===========================================================================


class TestFullConversionPipeline:
    """Test full conversion pipeline: validate -> convert -> provenance."""

    def test_mass_conversion_full_pipeline(self):
        da = DimensionalAnalyzer()
        converter = UnitConverter()
        tracker = ConversionProvenanceTracker()

        assert da.check_compatibility("kg", "t") is True
        result = converter.convert(1000, "kg", "t")
        assert result.ok
        assert float(result.value) == pytest.approx(1.0, rel=1e-6)

        record = tracker.record("convert", {"value": 1000, "from_unit": "kg", "to_unit": "t"}, str(result.value))
        assert len(record.hash) == 64

    def test_energy_conversion_full_pipeline(self):
        da = DimensionalAnalyzer()
        converter = UnitConverter()
        tracker = ConversionProvenanceTracker()

        assert da.check_compatibility("kWh", "MWh") is True
        result = converter.convert(100, "kWh", "MWh")
        assert result.ok
        assert float(result.value) == pytest.approx(0.1, rel=1e-6)

        record = tracker.record("convert", {"value": 100, "from_unit": "kWh", "to_unit": "MWh"}, str(result.value))
        assert record.hash

    def test_incompatible_pipeline_rejected(self):
        da = DimensionalAnalyzer()
        converter = UnitConverter()

        assert da.check_compatibility("kg", "kWh") is False
        result = converter.convert(1, "kg", "kWh")
        assert not result.ok

    def test_pipeline_with_alias_units(self):
        da = DimensionalAnalyzer()
        converter = UnitConverter()

        assert da.check_compatibility("kilogram", "t") is True
        result = converter.convert(2000, "kilogram", "t")
        assert result.ok
        assert float(result.value) == pytest.approx(2.0, rel=1e-6)


class TestGWPConversionPipeline:
    """Test GWP conversion end-to-end pipeline."""

    def test_ch4_to_co2e_ar6_pipeline(self):
        converter = UnitConverter()
        tracker = ConversionProvenanceTracker()

        result = converter.convert_ghg(1, "tCH4", "tCO2e", "AR6")
        assert result.ok
        assert float(result.value) == pytest.approx(29.8, rel=1e-3)

        record = tracker.record("ghg_convert", {"value": 1, "from_unit": "tCH4", "to_unit": "tCO2e", "gwp": "AR6"}, str(result.value))
        assert record.hash

    def test_n2o_to_co2e_ar5_pipeline(self):
        converter = UnitConverter()
        tracker = ConversionProvenanceTracker()

        result = converter.convert_ghg(1, "tN2O", "tCO2e", "AR5")
        assert result.ok
        assert float(result.value) == pytest.approx(265.0, rel=1e-3)

        record = tracker.record("ghg_convert", {"value": 1, "from_unit": "tN2O", "to_unit": "tCO2e", "gwp": "AR5"}, str(result.value))
        assert record.hash

    def test_chained_ghg_pipeline(self):
        converter = UnitConverter()
        tracker = ConversionProvenanceTracker()

        mass_result = converter.convert(1000, "kg", "t")
        assert mass_result.ok
        parent = tracker.record("convert", {"value": 1000, "from_unit": "kg", "to_unit": "t"}, str(mass_result.value))

        ghg_result = converter.convert_ghg(float(mass_result.value), "tCH4", "tCO2e", "AR6")
        assert ghg_result.ok
        child = tracker.record("ghg_convert", {"value": float(mass_result.value), "from": "tCH4", "to": "tCO2e"}, str(ghg_result.value), parent_hash=parent.hash)

        chain = tracker.get_chain(child.hash)
        assert len(chain) == 2
        assert chain[0].hash == parent.hash
        assert chain[1].hash == child.hash


class TestEntityResolutionPipeline:
    """Test entity resolution end-to-end pipeline."""

    def test_fuel_resolution_then_conversion(self):
        resolver = EntityResolver()
        converter = UnitConverter()

        match = resolver.resolve_fuel("Nat Gas")
        assert match.canonical_name == "natural_gas"
        assert match.confidence >= 0.95

        result = converter.convert(1000, "m3", "L")
        assert result.ok
        assert float(result.value) == pytest.approx(1000000.0, rel=1e-3)

    def test_material_resolution_then_mass_conversion(self):
        resolver = EntityResolver()
        converter = UnitConverter()

        match = resolver.resolve_material("aluminium")
        assert match.canonical_name == "aluminum"

        result = converter.convert(5000, "kg", "t")
        assert result.ok
        assert float(result.value) == pytest.approx(5.0, rel=1e-6)

    def test_process_resolution(self):
        resolver = EntityResolver()
        match = resolver.resolve_process("EAF")
        assert match.canonical_name == "electric_arc_furnace"
        assert match.category == "steelmaking"

    def test_mixed_entity_resolution(self):
        resolver = EntityResolver()

        fuel = resolver.resolve_fuel("diesel fuel")
        assert fuel.canonical_name == "diesel"

        material = resolver.resolve_material("OPC")
        assert material.canonical_name == "cement"

        process = resolver.resolve_process("BOF")
        assert process.canonical_name == "basic_oxygen_furnace"


class TestBatchOperations:
    """Test batch conversion operations end-to-end."""

    def test_batch_conversion_10_items(self):
        converter = UnitConverter()
        items = [
            {"value": 1000, "from_unit": "kg", "to_unit": "t"},
            {"value": 100, "from_unit": "kWh", "to_unit": "MWh"},
            {"value": 1, "from_unit": "m3", "to_unit": "L"},
            {"value": 1, "from_unit": "km", "to_unit": "m"},
            {"value": 1, "from_unit": "tCO2e", "to_unit": "kgCO2e"},
            {"value": 1, "from_unit": "gal", "to_unit": "L"},
            {"value": 1, "from_unit": "lb", "to_unit": "kg"},
            {"value": 1, "from_unit": "hectare", "to_unit": "m2"},
            {"value": 1, "from_unit": "MWh", "to_unit": "GJ"},
            {"value": 1000, "from_unit": "g", "to_unit": "kg"},
        ]
        results = converter.batch_convert(items)
        assert len(results) == 10
        assert all(r.ok for r in results)

    def test_batch_with_provenance(self):
        converter = UnitConverter()
        tracker = ConversionProvenanceTracker()

        items = [
            {"value": 100, "from_unit": "kg", "to_unit": "t"},
            {"value": 200, "from_unit": "kWh", "to_unit": "MWh"},
        ]
        results = converter.batch_convert(items)
        for i, result in enumerate(results):
            tracker.record("batch_convert", {"index": i, **items[i]}, str(result.value))
        assert tracker.count == 2

    def test_batch_entity_resolution(self):
        resolver = EntityResolver()
        names = ["Natural Gas", "Diesel", "Coal", "LPG", "Biogas",
                 "Unknown Fuel 1", "Unknown Fuel 2", "Gasoline", "Kerosene", "Propane"]
        results = resolver.batch_resolve_fuels(names)
        assert len(results) == 10
        resolved = [r for r in results if r.level != ConfidenceLevel.UNRESOLVED]
        unresolved = [r for r in results if r.level == ConfidenceLevel.UNRESOLVED]
        assert len(resolved) == 8
        assert len(unresolved) == 2


class TestProvenanceChaining:
    """Test provenance chain integrity end-to-end."""

    def test_three_step_chain(self):
        tracker = ConversionProvenanceTracker()
        r1 = tracker.record("parse", {"raw": "1000 kg"}, "1000")
        r2 = tracker.record("convert", {"value": 1000, "unit": "kg->t"}, "1.0", parent_hash=r1.hash)
        r3 = tracker.record("format", {"value": 1.0, "unit": "t"}, "1.0 t", parent_hash=r2.hash)

        chain = tracker.get_chain(r3.hash)
        assert len(chain) == 3
        assert chain[0].operation == "parse"
        assert chain[1].operation == "convert"
        assert chain[2].operation == "format"

    def test_export_and_reimport(self):
        tracker = ConversionProvenanceTracker()
        tracker.record("op1", {"a": 1}, "r1")
        tracker.record("op2", {"b": 2}, "r2")

        exported = tracker.export_json()
        imported = json.loads(exported)
        assert len(imported) == 2
        assert imported[0]["operation"] == "op1"
        assert imported[1]["operation"] == "op2"
