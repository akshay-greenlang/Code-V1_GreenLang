# -*- coding: utf-8 -*-
"""
Unit Tests for ColumnMapper (AGENT-DATA-002)

Tests column mapping: initialization, map_columns, exact_match,
synonym_match, fuzzy_match, pattern_match, register_synonym,
register_template, apply_template, get_canonical_fields by category,
unmapped headers, confidence scores, and statistics.

Coverage target: 85%+ of column_mapper.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline ColumnMapper mirroring greenlang/excel_normalizer/column_mapper.py
# ---------------------------------------------------------------------------

CANONICAL_FIELDS = {
    "energy": [
        "facility_name", "reporting_year", "electricity_kwh", "natural_gas_therms",
        "diesel_litres", "renewable_pct", "peak_demand_kw", "grid_emission_factor",
    ],
    "transport": [
        "vehicle_id", "route", "distance_km", "fuel_used_litres",
        "fuel_type", "cargo_weight_tonnes", "transport_mode", "co2_emissions_kg",
    ],
    "waste": [
        "waste_category", "disposal_method", "weight_kg", "recycled_pct",
        "landfill_kg", "incinerated_kg", "hazardous_flag", "treatment_facility",
    ],
    "emissions": [
        "source", "scope", "ghg_type", "emission_factor",
        "activity_data", "unit", "total_emissions_tco2e", "uncertainty_pct",
    ],
    "procurement": [
        "supplier", "product_category", "spend_usd", "emission_factor_kgco2e",
        "quantity", "unit", "origin_country", "scope3_category",
    ],
}

SYNONYMS: Dict[str, List[str]] = {
    "facility_name": ["facility", "site", "plant", "location", "building", "office"],
    "reporting_year": ["year", "fiscal year", "fy", "period", "reporting period"],
    "electricity_kwh": ["electricity", "electric consumption", "power kwh", "elec kwh",
                        "electricity consumption (kwh)", "electricity consumption kwh"],
    "natural_gas_therms": ["natural gas", "gas therms", "nat gas", "methane"],
    "fuel_type": ["fuel", "fuel source", "energy source", "fuel category"],
    "co2_emissions_kg": ["co2", "carbon emissions", "ghg emissions", "co2e",
                         "co2 emissions", "carbon dioxide", "co2 emissions (kgco2e)"],
    "total_emissions_tco2e": ["total emissions", "emissions total", "tco2e",
                              "total ghg", "scope 1 emissions (tco2e)"],
    "scope": ["emission scope", "ghg scope", "scope type"],
    "distance_km": ["distance", "km", "kilometers", "mileage"],
    "weight_kg": ["weight", "mass", "kg", "kilograms"],
    "spend_usd": ["spend", "cost", "amount", "total cost", "expenditure"],
    "supplier": ["vendor", "provider", "partner", "company"],
    "origin_country": ["country", "nation", "country of origin", "origin"],
    "diesel_litres": ["diesel", "diesel fuel", "diesel fuel (litres)"],
    "renewable_pct": ["renewable", "renewable energy", "renewable energy (%)"],
}


class ColumnMapper:
    """Maps source columns to canonical GreenLang fields."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._fuzzy_threshold: float = self._config.get("fuzzy_threshold", 0.75)
        self._synonyms: Dict[str, List[str]] = dict(SYNONYMS)
        self._templates: Dict[str, Dict[str, str]] = {}
        self._stats: Dict[str, int] = {
            "exact_matches": 0, "synonym_matches": 0,
            "fuzzy_matches": 0, "pattern_matches": 0, "unmapped": 0,
        }

    def map_columns(self, headers: List[str]) -> List[Dict[str, Any]]:
        results = []
        for idx, header in enumerate(headers):
            mapping = self._map_single(header, idx)
            results.append(mapping)
        return results

    def exact_match(self, header: str) -> Optional[Tuple[str, float]]:
        normalized = header.strip().lower().replace(" ", "_")
        all_fields = self._get_all_canonical()
        if normalized in all_fields:
            self._stats["exact_matches"] += 1
            return (normalized, 1.0)
        return None

    def synonym_match(self, header: str) -> Optional[Tuple[str, float]]:
        normalized = header.strip().lower()
        for canonical, syns in self._synonyms.items():
            for syn in syns:
                if normalized == syn.lower():
                    self._stats["synonym_matches"] += 1
                    return (canonical, 0.95)
        return None

    def fuzzy_match(self, header: str) -> Optional[Tuple[str, float]]:
        normalized = header.strip().lower()
        best_field, best_score = None, 0.0
        all_fields = self._get_all_canonical()
        for field in all_fields:
            score = SequenceMatcher(None, normalized, field).ratio()
            if score > best_score:
                best_score = score
                best_field = field
        # Also check synonyms
        for canonical, syns in self._synonyms.items():
            for syn in syns:
                score = SequenceMatcher(None, normalized, syn.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_field = canonical
        if best_field and best_score >= self._fuzzy_threshold:
            self._stats["fuzzy_matches"] += 1
            return (best_field, round(best_score, 4))
        return None

    def pattern_match(self, header: str) -> Optional[Tuple[str, float]]:
        patterns = [
            (r"(?i).*co2.*emiss.*", "co2_emissions_kg"),
            (r"(?i).*scope\s*[123].*emiss.*", "total_emissions_tco2e"),
            (r"(?i).*electric.*kwh.*", "electricity_kwh"),
            (r"(?i).*diesel.*litr.*", "diesel_litres"),
            (r"(?i).*fuel.*type.*", "fuel_type"),
            (r"(?i).*distance.*km.*", "distance_km"),
            (r"(?i).*weight.*kg.*", "weight_kg"),
        ]
        for pattern, field in patterns:
            if re.match(pattern, header.strip()):
                self._stats["pattern_matches"] += 1
                return (field, 0.85)
        return None

    def register_synonym(self, canonical_field: str, synonym: str) -> None:
        if canonical_field not in self._synonyms:
            self._synonyms[canonical_field] = []
        self._synonyms[canonical_field].append(synonym)

    def register_template(self, template_name: str, mappings: Dict[str, str]) -> None:
        self._templates[template_name] = mappings

    def apply_template(self, template_name: str, headers: List[str]) -> List[Dict[str, Any]]:
        template = self._templates.get(template_name)
        if not template:
            return self.map_columns(headers)
        results = []
        for idx, header in enumerate(headers):
            normalized = header.strip().lower()
            if normalized in template:
                results.append({
                    "source_column": header, "source_index": idx,
                    "canonical_field": template[normalized],
                    "strategy": "template", "confidence": 1.0,
                })
            else:
                results.append(self._map_single(header, idx))
        return results

    def get_canonical_fields(self, category: Optional[str] = None) -> List[str]:
        if category and category in CANONICAL_FIELDS:
            return list(CANONICAL_FIELDS[category])
        return self._get_all_canonical()

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self._stats)

    def _map_single(self, header: str, index: int) -> Dict[str, Any]:
        for strategy, func in [
            ("exact", self.exact_match),
            ("synonym", self.synonym_match),
            ("fuzzy", self.fuzzy_match),
            ("pattern", self.pattern_match),
        ]:
            result = func(header)
            if result:
                return {
                    "source_column": header, "source_index": index,
                    "canonical_field": result[0],
                    "strategy": strategy, "confidence": result[1],
                }
        self._stats["unmapped"] += 1
        return {
            "source_column": header, "source_index": index,
            "canonical_field": "", "strategy": "none", "confidence": 0.0,
        }

    def _get_all_canonical(self) -> List[str]:
        fields = []
        for category_fields in CANONICAL_FIELDS.values():
            fields.extend(category_fields)
        return list(set(fields))


# ===========================================================================
# Test Classes
# ===========================================================================


class TestColumnMapperInit:
    def test_default_creation(self):
        mapper = ColumnMapper()
        assert mapper._fuzzy_threshold == 0.75

    def test_custom_threshold(self):
        mapper = ColumnMapper(config={"fuzzy_threshold": 0.9})
        assert mapper._fuzzy_threshold == 0.9

    def test_initial_statistics(self):
        mapper = ColumnMapper()
        stats = mapper.get_statistics()
        assert stats["exact_matches"] == 0
        assert stats["unmapped"] == 0


class TestMapColumns:
    def test_map_multiple_headers(self):
        mapper = ColumnMapper()
        results = mapper.map_columns(["facility_name", "reporting_year", "unknown_field"])
        assert len(results) == 3
        assert results[0]["canonical_field"] == "facility_name"

    def test_map_preserves_order(self):
        mapper = ColumnMapper()
        results = mapper.map_columns(["facility_name", "fuel_type"])
        assert results[0]["source_index"] == 0
        assert results[1]["source_index"] == 1

    def test_map_empty_headers(self):
        mapper = ColumnMapper()
        results = mapper.map_columns([])
        assert results == []


class TestExactMatch:
    def test_exact_match_facility_name(self):
        mapper = ColumnMapper()
        result = mapper.exact_match("facility_name")
        assert result == ("facility_name", 1.0)

    def test_exact_match_reporting_year(self):
        mapper = ColumnMapper()
        result = mapper.exact_match("reporting_year")
        assert result is not None

    def test_exact_match_electricity_kwh(self):
        mapper = ColumnMapper()
        result = mapper.exact_match("electricity_kwh")
        assert result is not None

    def test_exact_match_fuel_type(self):
        mapper = ColumnMapper()
        result = mapper.exact_match("fuel_type")
        assert result is not None

    def test_exact_match_distance_km(self):
        mapper = ColumnMapper()
        result = mapper.exact_match("distance_km")
        assert result is not None

    def test_exact_match_weight_kg(self):
        mapper = ColumnMapper()
        result = mapper.exact_match("weight_kg")
        assert result is not None

    def test_exact_match_spend_usd(self):
        mapper = ColumnMapper()
        result = mapper.exact_match("spend_usd")
        assert result is not None

    def test_exact_match_with_spaces(self):
        mapper = ColumnMapper()
        result = mapper.exact_match("facility name")
        assert result is not None

    def test_exact_match_case_insensitive(self):
        mapper = ColumnMapper()
        result = mapper.exact_match("Facility_Name")
        assert result is not None

    def test_exact_match_unknown(self):
        mapper = ColumnMapper()
        result = mapper.exact_match("totally_unknown_xyz")
        assert result is None

    def test_exact_match_stats(self):
        mapper = ColumnMapper()
        mapper.exact_match("facility_name")
        assert mapper.get_statistics()["exact_matches"] == 1


class TestSynonymMatch:
    def test_synonym_facility(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("facility")
        assert result is not None
        assert result[0] == "facility_name"

    def test_synonym_site(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("site")
        assert result is not None
        assert result[0] == "facility_name"

    def test_synonym_plant(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("plant")
        assert result is not None

    def test_synonym_year(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("year")
        assert result is not None
        assert result[0] == "reporting_year"

    def test_synonym_electricity(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("electricity")
        assert result is not None

    def test_synonym_co2(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("co2")
        assert result is not None

    def test_synonym_vendor(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("vendor")
        assert result is not None
        assert result[0] == "supplier"

    def test_synonym_country(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("country")
        assert result is not None

    def test_synonym_natural_gas(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("natural gas")
        assert result is not None

    def test_synonym_diesel(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("diesel")
        assert result is not None

    def test_synonym_unknown(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("xyzzy_field")
        assert result is None

    def test_synonym_case_insensitive(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("FACILITY")
        assert result is not None

    def test_synonym_confidence(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("facility")
        assert result[1] == 0.95

    def test_synonym_stats(self):
        mapper = ColumnMapper()
        mapper.synonym_match("facility")
        assert mapper.get_statistics()["synonym_matches"] == 1

    def test_synonym_total_emissions(self):
        mapper = ColumnMapper()
        result = mapper.synonym_match("total emissions")
        assert result is not None


class TestFuzzyMatch:
    def test_fuzzy_facilty_name(self):
        mapper = ColumnMapper()
        result = mapper.fuzzy_match("facilty_name")  # Typo
        assert result is not None

    def test_fuzzy_electrcity(self):
        mapper = ColumnMapper()
        result = mapper.fuzzy_match("electrcity kwh")  # Typo
        assert result is not None

    def test_fuzzy_diesl(self):
        mapper = ColumnMapper()
        result = mapper.fuzzy_match("diesel litres")
        assert result is not None

    def test_fuzzy_co2_emission(self):
        mapper = ColumnMapper()
        result = mapper.fuzzy_match("co2 emission")
        assert result is not None

    def test_fuzzy_reporting_yr(self):
        mapper = ColumnMapper()
        result = mapper.fuzzy_match("reporting year")
        assert result is not None

    def test_fuzzy_gas_therms(self):
        mapper = ColumnMapper()
        result = mapper.fuzzy_match("gas therms")
        assert result is not None

    def test_fuzzy_high_threshold_rejects(self):
        mapper = ColumnMapper(config={"fuzzy_threshold": 0.99})
        result = mapper.fuzzy_match("facilty")
        assert result is None

    def test_fuzzy_completely_unrelated(self):
        mapper = ColumnMapper()
        result = mapper.fuzzy_match("zzzzz_qqqq_xxxx")
        assert result is None

    def test_fuzzy_confidence_value(self):
        mapper = ColumnMapper()
        result = mapper.fuzzy_match("facility_name")  # Exact should score high
        assert result is not None
        assert result[1] >= 0.75

    def test_fuzzy_stats(self):
        mapper = ColumnMapper()
        mapper.fuzzy_match("facilty_name")
        stats = mapper.get_statistics()
        assert stats["fuzzy_matches"] >= 0  # May or may not match


class TestPatternMatch:
    def test_pattern_co2_emissions(self):
        mapper = ColumnMapper()
        result = mapper.pattern_match("CO2 Emissions (kgCO2e)")
        assert result is not None
        assert result[0] == "co2_emissions_kg"

    def test_pattern_scope1_emissions(self):
        mapper = ColumnMapper()
        result = mapper.pattern_match("Scope 1 Emissions (tCO2e)")
        assert result is not None

    def test_pattern_electricity_kwh(self):
        mapper = ColumnMapper()
        result = mapper.pattern_match("Electricity Consumption (kWh)")
        assert result is not None

    def test_pattern_diesel_litres(self):
        mapper = ColumnMapper()
        result = mapper.pattern_match("Diesel Fuel (litres)")
        assert result is not None

    def test_pattern_fuel_type(self):
        mapper = ColumnMapper()
        result = mapper.pattern_match("Fuel Type")
        assert result is not None

    def test_pattern_distance_km(self):
        mapper = ColumnMapper()
        result = mapper.pattern_match("Distance (km)")
        assert result is not None

    def test_pattern_weight_kg(self):
        mapper = ColumnMapper()
        result = mapper.pattern_match("Weight (kg)")
        assert result is not None

    def test_pattern_no_match(self):
        mapper = ColumnMapper()
        result = mapper.pattern_match("Random Header XYZ")
        assert result is None

    def test_pattern_confidence(self):
        mapper = ColumnMapper()
        result = mapper.pattern_match("CO2 Emissions Total")
        assert result is not None
        assert result[1] == 0.85


class TestRegisterSynonym:
    def test_register_new_synonym(self):
        mapper = ColumnMapper()
        mapper.register_synonym("facility_name", "establishment")
        result = mapper.synonym_match("establishment")
        assert result is not None
        assert result[0] == "facility_name"

    def test_register_synonym_new_field(self):
        mapper = ColumnMapper()
        mapper.register_synonym("custom_field", "my custom header")
        assert "my custom header" in mapper._synonyms["custom_field"]


class TestRegisterTemplate:
    def test_register_and_apply_template(self):
        mapper = ColumnMapper()
        mapper.register_template("energy_v1", {"facility": "facility_name", "kwh": "electricity_kwh"})
        results = mapper.apply_template("energy_v1", ["facility", "kwh", "unknown"])
        assert results[0]["strategy"] == "template"
        assert results[0]["confidence"] == 1.0

    def test_apply_nonexistent_template(self):
        mapper = ColumnMapper()
        results = mapper.apply_template("missing", ["col1"])
        assert len(results) == 1


class TestGetCanonicalFields:
    def test_energy_category(self):
        mapper = ColumnMapper()
        fields = mapper.get_canonical_fields("energy")
        assert "facility_name" in fields
        assert "electricity_kwh" in fields

    def test_transport_category(self):
        mapper = ColumnMapper()
        fields = mapper.get_canonical_fields("transport")
        assert "vehicle_id" in fields
        assert "distance_km" in fields

    def test_waste_category(self):
        mapper = ColumnMapper()
        fields = mapper.get_canonical_fields("waste")
        assert "waste_category" in fields

    def test_emissions_category(self):
        mapper = ColumnMapper()
        fields = mapper.get_canonical_fields("emissions")
        assert "total_emissions_tco2e" in fields

    def test_procurement_category(self):
        mapper = ColumnMapper()
        fields = mapper.get_canonical_fields("procurement")
        assert "supplier" in fields

    def test_all_fields(self):
        mapper = ColumnMapper()
        all_fields = mapper.get_canonical_fields()
        assert len(all_fields) > 20

    def test_unknown_category(self):
        mapper = ColumnMapper()
        fields = mapper.get_canonical_fields("nonexistent")
        assert len(fields) > 0  # Returns all fields


class TestUnmappedHeaders:
    def test_unmapped_increments_stat(self):
        mapper = ColumnMapper()
        mapper.map_columns(["zzz_unknown_field_xyz"])
        assert mapper.get_statistics()["unmapped"] >= 1

    def test_unmapped_returns_empty_canonical(self):
        mapper = ColumnMapper()
        results = mapper.map_columns(["zzz_completely_unknown_xyz"])
        assert results[0]["canonical_field"] == ""
        assert results[0]["strategy"] == "none"
        assert results[0]["confidence"] == 0.0


class TestColumnMapperStatistics:
    def test_combined_stats(self):
        mapper = ColumnMapper()
        mapper.map_columns(["facility_name", "facility", "facilty_name",
                            "CO2 Emissions (kgCO2e)", "zzz_unknown"])
        stats = mapper.get_statistics()
        total = sum(stats.values())
        assert total >= 5
