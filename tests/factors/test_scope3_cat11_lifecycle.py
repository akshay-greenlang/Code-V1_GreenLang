# -*- coding: utf-8 -*-
"""Tests for Scope 3 Category 11 product-lifetime fields.

Covers:
* schema: the ``materials_products`` parameter group accepts the four
  new Cat 11 fields (product_lifetime_years, use_phase_energy_kwh,
  use_phase_frequency_per_year, end_of_life_allocation_method).
* data class: :class:`UsePhaseParameters` validates, enumerates end-of-life
  methods, and computes lifetime-energy correctly.
* method pack: ``CORPORATE_SCOPE3`` allows the expected families, lists
  Cat 11 in its reporting labels, and ``render_cat11_use_phase_block``
  surfaces lifecycle params in the audit template.
* resolution: a Cat 11 factor with use_phase attached round-trips through
  :class:`EmissionFactorRecord` and surfaces all four lifecycle parameters.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
)
from greenlang.data.emission_factor_record import (
    Boundary,
    DataQualityScore,
    EmissionFactorRecord,
    EndOfLifeAllocationMethod,
    GeographyLevel,
    GHGVectors,
    GWPSet,
    GWPValues,
    LicenseInfo,
    Methodology,
    Scope,
    SourceProvenance,
    UsePhaseParameters,
)
from greenlang.factors.method_packs.corporate import (
    CORPORATE_SCOPE3,
    render_cat11_use_phase_block,
)


# ---------------------------------------------------------------------------
# Schema JSON
# ---------------------------------------------------------------------------


_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "schemas"
    / "factor_record_v1.schema.json"
)


def _load_schema() -> dict:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class TestSchemaMaterialsProducts:
    def test_product_lifetime_fields_present(self):
        schema = _load_schema()
        materials_branch = next(
            b
            for b in schema["oneOf"]
            if b.get("title") == "materials_products parameters"
        )
        props = materials_branch["properties"]["parameters"]["properties"]
        assert "product_lifetime_years" in props
        assert "use_phase_energy_kwh" in props
        assert "use_phase_frequency_per_year" in props
        assert "end_of_life_allocation_method" in props

    def test_product_lifetime_years_minimum_is_zero(self):
        schema = _load_schema()
        materials_branch = next(
            b
            for b in schema["oneOf"]
            if b.get("title") == "materials_products parameters"
        )
        props = materials_branch["properties"]["parameters"]["properties"]
        assert props["product_lifetime_years"]["minimum"] == 0
        assert props["use_phase_energy_kwh"]["minimum"] == 0
        assert props["use_phase_frequency_per_year"]["minimum"] == 0

    def test_end_of_life_allocation_enum_values(self):
        schema = _load_schema()
        materials_branch = next(
            b
            for b in schema["oneOf"]
            if b.get("title") == "materials_products parameters"
        )
        props = materials_branch["properties"]["parameters"]["properties"]
        enum_values = props["end_of_life_allocation_method"]["enum"]
        # Required four + null tolerance.
        assert "100_1" in enum_values
        assert "50_50" in enum_values
        assert "avoided_burden" in enum_values
        assert "none" in enum_values


# ---------------------------------------------------------------------------
# UsePhaseParameters dataclass
# ---------------------------------------------------------------------------


class TestUsePhaseParameters:
    def test_all_fields_are_optional(self):
        up = UsePhaseParameters()
        assert up.product_lifetime_years is None
        assert up.use_phase_energy_kwh is None
        assert up.use_phase_frequency_per_year is None
        assert up.end_of_life_allocation_method is None

    def test_end_of_life_enum_roundtrip(self):
        up = UsePhaseParameters(
            end_of_life_allocation_method=EndOfLifeAllocationMethod.CUT_OFF_100_1
        )
        assert up.end_of_life_allocation_method.value == "100_1"

    def test_end_of_life_accepts_raw_string(self):
        up = UsePhaseParameters(end_of_life_allocation_method="50_50")
        assert up.end_of_life_allocation_method == EndOfLifeAllocationMethod.SPLIT_50_50

    def test_negative_lifetime_rejected(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            UsePhaseParameters(product_lifetime_years=-3.0)

    def test_lifetime_use_phase_energy(self):
        up = UsePhaseParameters(
            product_lifetime_years=14,
            use_phase_energy_kwh=1.1,
            use_phase_frequency_per_year=365,
        )
        # 14 * 1.1 * 365 = 5621.0
        assert up.lifetime_use_phase_energy_kwh() == pytest.approx(5621.0)

    def test_lifetime_use_phase_energy_none_when_incomplete(self):
        up = UsePhaseParameters(
            product_lifetime_years=14,
            use_phase_energy_kwh=1.1,
            # frequency missing
        )
        assert up.lifetime_use_phase_energy_kwh() is None


# ---------------------------------------------------------------------------
# CORPORATE_SCOPE3 audit template + allowed families
# ---------------------------------------------------------------------------


class TestCorporateScope3Cat11:
    def test_allowed_families_include_material_embodied(self):
        allowed = {f.value for f in CORPORATE_SCOPE3.selection_rule.allowed_families}
        assert FactorFamily.MATERIAL_EMBODIED.value in allowed
        # Cat 11 use-phase electricity draw
        assert FactorFamily.GRID_INTENSITY.value in allowed
        # Cat 1 land-use upstream
        assert FactorFamily.LAND_USE_REMOVALS.value in allowed

    def test_audit_template_has_cat11_placeholder(self):
        template = CORPORATE_SCOPE3.audit_text_template
        assert "{cat11_use_phase_block}" in template
        assert "{scope3_category}" in template

    def test_render_cat11_block_full(self):
        up = UsePhaseParameters(
            product_lifetime_years=14,
            use_phase_energy_kwh=1.1,
            use_phase_frequency_per_year=365,
            end_of_life_allocation_method=EndOfLifeAllocationMethod.CUT_OFF_100_1,
        )
        rendered = render_cat11_use_phase_block(up)
        assert "Cat 11 use-phase" in rendered
        assert "lifetime=14 yr" in rendered
        assert "per-use=1.1 kWh" in rendered
        assert "freq=365/yr" in rendered
        assert "EoL=100_1" in rendered

    def test_render_cat11_block_none_returns_empty(self):
        assert render_cat11_use_phase_block(None) == ""

    def test_render_cat11_block_partial(self):
        up = UsePhaseParameters(product_lifetime_years=10)
        rendered = render_cat11_use_phase_block(up)
        assert rendered.startswith("Cat 11 use-phase")
        assert "lifetime=10 yr" in rendered
        # No per-use / freq / EoL because they were None
        assert "per-use" not in rendered
        assert "freq=" not in rendered
        assert "EoL=" not in rendered

    def test_pack_version_bumped(self):
        # Adding Cat 11 surfacing requires a minor bump
        assert CORPORATE_SCOPE3.pack_version.split(".")[1] != "0"


# ---------------------------------------------------------------------------
# EmissionFactorRecord carries the lifecycle params
# ---------------------------------------------------------------------------


def _sample_cat11_record() -> EmissionFactorRecord:
    return EmissionFactorRecord(
        factor_id="EF:PACT:refrigerator-500L-eu",
        fuel_type="refrigerator_500l",
        unit="unit",
        geography="EU",
        geography_level=GeographyLevel.CONTINENT,
        vectors=GHGVectors(CO2=320.0, CH4=0.0, N2O=0.0, biogenic_CO2=5.0),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_3,
        boundary=Boundary.CRADLE_TO_GRAVE,
        provenance=SourceProvenance(
            source_org="Manufacturer",
            source_publication="Product carbon footprint disclosure",
            source_year=2024,
            methodology=Methodology.LCA,
        ),
        valid_from=date(2024, 1, 1),
        valid_to=date(2024, 12, 31),
        uncertainty_95ci=0.2,
        dqs=DataQualityScore(4, 3, 3, 3, 4),
        license_info=LicenseInfo(
            license="PACT Pathfinder",
            redistribution_allowed=False,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        method_profile=MethodProfile.CORPORATE_SCOPE3.value,
        factor_family=FactorFamily.MATERIAL_EMBODIED.value,
        factor_version="1.0.0",
        formula_type=FormulaType.LCA.value,
        use_phase=UsePhaseParameters(
            product_lifetime_years=14,
            use_phase_energy_kwh=1.1,
            use_phase_frequency_per_year=365,
            end_of_life_allocation_method=EndOfLifeAllocationMethod.CUT_OFF_100_1,
        ),
    )


class TestCat11FactorResolution:
    def test_cat11_factor_carries_lifecycle_params(self):
        rec = _sample_cat11_record()
        assert rec.use_phase is not None
        assert rec.use_phase.product_lifetime_years == 14
        assert rec.use_phase.use_phase_energy_kwh == 1.1
        assert rec.use_phase.use_phase_frequency_per_year == 365
        assert (
            rec.use_phase.end_of_life_allocation_method
            is EndOfLifeAllocationMethod.CUT_OFF_100_1
        )

    def test_lifetime_energy_computed(self):
        rec = _sample_cat11_record()
        assert rec.use_phase.lifetime_use_phase_energy_kwh() == pytest.approx(5621.0)

    def test_audit_template_renders_with_block(self):
        rec = _sample_cat11_record()
        template = CORPORATE_SCOPE3.audit_text_template
        rendered = template.format(
            scope3_category="11",
            calculation_method="hybrid",
            factor_id=rec.factor_id,
            dqs_score=75,
            cat11_use_phase_block=render_cat11_use_phase_block(rec.use_phase),
        )
        assert "Scope 3 Cat 11" in rendered
        assert "EF:PACT:refrigerator-500L-eu" in rendered
        assert "lifetime=14 yr" in rendered
        assert "EoL=100_1" in rendered

    def test_non_cat11_record_has_no_use_phase(self):
        # Default EmissionFactorRecord.use_phase is None for non-Cat-11 factors
        rec = _sample_cat11_record()
        rec.use_phase = None
        assert render_cat11_use_phase_block(rec.use_phase) == ""
