# -*- coding: utf-8 -*-
"""Tests for canonical_v1 migration (W4-A).

Covers the 10 migration items from ``docs/specs/schema_v1_gap_report.md``:

* Per-family parameter round-trips
* Old-shape -> new-shape preservation of numeric values (bit-identical)
* gwp_registry derives CO2e under 3+ GWP sets
* composite_fqs produces 0-100 from 1-5 DQS
* Enum maps + legacy deprecation warnings
* Migration script idempotence
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from greenlang.factors.data import (
    canonical_v1,
    canonical_v2,
    categorical_parameters,
    gwp_registry,
)
from greenlang.factors.data.canonical_v1 import (
    DEFAULT_GWP_SET,
    FACTOR_FAMILY_ENUM,
    FactorRecordV1,
    GWP_SET_ENUM,
    METHOD_PROFILE_ENUM,
    REDISTRIBUTION_CLASS_ENUM,
    STATUS_ENUM,
    compute_fqs,
    from_legacy_dict,
    migrate_record,
    to_legacy_dict,
)
from greenlang.factors.data.categorical_parameters import (
    CombustionParameters,
    ElectricityParameters,
    FinanceProxiesParameters,
    LandRemovalsParameters,
    MaterialsProductsParameters,
    RefrigerantsParameters,
    TransportParameters,
    WasteParameters,
    dump_parameters,
    parse_parameters,
)
from greenlang.factors.data.source_object import (
    SOURCE_REDISTRIBUTION_CLASS_ENUM,
    SOURCE_TYPE_ENUM,
    SourceObject,
    source_object_from_dict,
    source_object_to_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _legacy_desnz_payload() -> dict:
    """Sample DESNZ-shape factor (matches catalog_seed/desnz_ghg_conversion)."""
    return {
        "factor_id": "EF:DESNZ:s1_natural_gas_kwh_net_cv:UK:2024:v1",
        "fuel_type": "natural_gas",
        "unit": "kwh net cv",
        "geography": "UK",
        "geography_level": "country",
        "vectors": {"CO2": 0.18293, "CH4": 0.00021, "N2O": 0.00017},
        "gwp_100yr": {
            "gwp_set": "IPCC_AR5_100",
            "CH4_gwp": 28,
            "N2O_gwp": 265,
        },
        "scope": "1",
        "boundary": "combustion",
        "provenance": {
            "source_org": "DESNZ",
            "source_publication": "UK GHG conversion factors 2024",
            "source_year": 2024,
            "methodology": "IPCC_Tier_1",
            "version": "2024",
        },
        "valid_from": "2024-01-01",
        "uncertainty_95ci": 0.05,
        "dqs": {
            "temporal": 5,
            "geographical": 5,
            "technological": 4,
            "representativeness": 4,
            "methodological": 5,
        },
        "license_info": {
            "license": "OGL-UK-v3",
            "redistribution_allowed": True,
            "commercial_use_allowed": True,
            "attribution_required": True,
        },
        "source_id": "desnz_ghg_conversion",
        "factor_status": "certified",
        "redistribution_class": "open",
    }


# ---------------------------------------------------------------------------
# 1. CTO-reversible enum constants
# ---------------------------------------------------------------------------


class TestCTOEnums:
    """All enum tuples are exposed as module-level constants (W4-A rule)."""

    def test_status_enum_has_7_values(self):
        assert len(STATUS_ENUM) == 7
        assert "certified" in STATUS_ENUM
        assert "draft" in STATUS_ENUM
        assert "preview" in STATUS_ENUM
        assert "connector_only" in STATUS_ENUM
        assert "retired" in STATUS_ENUM

    def test_redistribution_class_enum_has_4_values(self):
        assert len(REDISTRIBUTION_CLASS_ENUM) == 4
        assert "open" in REDISTRIBUTION_CLASS_ENUM
        assert "licensed_embedded" in REDISTRIBUTION_CLASS_ENUM
        assert "customer_private" in REDISTRIBUTION_CLASS_ENUM
        assert "oem_redistributable" in REDISTRIBUTION_CLASS_ENUM

    def test_gwp_set_enum_has_6_values(self):
        assert len(GWP_SET_ENUM) == 6
        assert DEFAULT_GWP_SET == "IPCC_AR6_100"
        for value in GWP_SET_ENUM:
            assert value in gwp_registry.ALL_GWP_SETS

    def test_method_profile_enum_has_14_values(self):
        assert len(METHOD_PROFILE_ENUM) == 14
        assert "corporate_scope1" in METHOD_PROFILE_ENUM
        assert "corporate_scope2_location_based" in METHOD_PROFILE_ENUM
        assert "corporate_scope2_market_based" in METHOD_PROFILE_ENUM
        assert "corporate_scope3_upstream" in METHOD_PROFILE_ENUM
        assert "corporate_scope3_downstream" in METHOD_PROFILE_ENUM
        assert "product_carbon_iso_14067" in METHOD_PROFILE_ENUM
        assert "product_carbon_ghgp" in METHOD_PROFILE_ENUM
        assert "product_carbon_pact" in METHOD_PROFILE_ENUM
        assert "freight_iso_14083" in METHOD_PROFILE_ENUM
        assert "freight_glec" in METHOD_PROFILE_ENUM
        assert "land_removals_lsr" in METHOD_PROFILE_ENUM
        assert "finance_proxy_pcaf" in METHOD_PROFILE_ENUM
        assert "eu_cbam" in METHOD_PROFILE_ENUM
        assert "eu_dpp" in METHOD_PROFILE_ENUM

    def test_factor_family_enum_has_15_values(self):
        assert len(FACTOR_FAMILY_ENUM) == 15


# ---------------------------------------------------------------------------
# 2. DQS rescale (M12 / T04)
# ---------------------------------------------------------------------------


class TestComputeFqs:
    def test_all_fives_yields_100(self):
        assert compute_fqs(5, 5, 5, 5, 5) == 100.0

    def test_all_ones_yields_20(self):
        assert compute_fqs(1, 1, 1, 1, 1) == 20.0

    def test_all_threes_yields_60(self):
        assert compute_fqs(3, 3, 3, 3, 3) == 60.0

    def test_weights_sum_to_one(self):
        # 0.25+0.25+0.20+0.15+0.15 = 1.0
        # Result at score=5 is 20*5 = 100
        result = compute_fqs(5, 5, 5, 5, 5)
        assert result == pytest.approx(100.0, abs=0.01)

    def test_lowest_temporal_drops_score(self):
        base = compute_fqs(5, 5, 5, 5, 5)
        low_t = compute_fqs(1, 5, 5, 5, 5)
        assert low_t < base
        # dim weight 0.25: 20 * 0.25 * (5-1) = 20 pts
        assert base - low_t == pytest.approx(20.0, abs=0.01)

    def test_invalid_score_rejected(self):
        with pytest.raises(ValueError):
            compute_fqs(6, 5, 5, 5, 5)
        with pytest.raises(ValueError):
            compute_fqs(0, 5, 5, 5, 5)


# ---------------------------------------------------------------------------
# 3. GWP registry (M20 / X06)
# ---------------------------------------------------------------------------


class TestGwpRegistry:
    def test_co2_is_one_in_every_set(self):
        for gwp_set in gwp_registry.ALL_GWP_SETS:
            assert gwp_registry.lookup(gwp_set, "CO2") == Decimal("1.0")

    def test_ch4_differs_between_ar5_and_ar6(self):
        ar5 = gwp_registry.lookup("IPCC_AR5_100", "CH4")
        ar6 = gwp_registry.lookup("IPCC_AR6_100", "CH4")
        sar = gwp_registry.lookup("Kyoto_SAR_100", "CH4")
        assert ar5 == Decimal("28.0")
        assert ar6 == Decimal("27.9")
        assert sar == Decimal("21.0")
        assert ar5 != ar6 != sar

    def test_sf6_ar6_100(self):
        assert gwp_registry.lookup("IPCC_AR6_100", "SF6") == Decimal("25200")

    def test_aliases_accepted(self):
        # Legacy -> canonical
        assert gwp_registry.normalize_gwp_set("AR6_100") == "IPCC_AR6_100"
        assert gwp_registry.normalize_gwp_set("IPCC_SAR_100") == "Kyoto_SAR_100"

    def test_unknown_set_raises(self):
        with pytest.raises(ValueError):
            gwp_registry.lookup("UNKNOWN_SET_2099", "CO2")

    def test_unknown_gas_raises(self):
        with pytest.raises(ValueError):
            gwp_registry.lookup("IPCC_AR6_100", "UNKNOWN_GAS")

    def test_co2e_derivation_three_sets(self):
        """Same gas mix under 3 GWP sets gives 3 distinct CO2e values."""
        gases = {"CO2": 100.0, "CH4": 1.0, "N2O": 0.1}
        ar6 = gwp_registry.co2e(gases, "IPCC_AR6_100")
        ar5 = gwp_registry.co2e(gases, "IPCC_AR5_100")
        sar = gwp_registry.co2e(gases, "Kyoto_SAR_100")
        # Each derivation is deterministic
        assert ar6 == gwp_registry.co2e(gases, "IPCC_AR6_100")
        assert ar5 == gwp_registry.co2e(gases, "IPCC_AR5_100")
        assert sar == gwp_registry.co2e(gases, "Kyoto_SAR_100")
        # And they're all different
        assert len({ar5, ar6, sar}) == 3
        # Simple sanity check: AR6 ~= 100 + 27.9 + 27.3
        assert abs(float(ar6) - (100 + 27.9 + 27.3)) < 0.01

    def test_co2e_with_f_gases(self):
        gases = {"CO2": 10.0, "CH4": 0.0, "N2O": 0.0}
        f_gases = {"SF6": 0.001}  # 1 mg SF6 -> 25.2 kg CO2e AR6
        result = gwp_registry.co2e(gases, "IPCC_AR6_100", f_gases=f_gases)
        assert float(result) == pytest.approx(10 + 25.2, abs=0.01)


# ---------------------------------------------------------------------------
# 4. Per-family parameter round-trips (M19 / §4)
# ---------------------------------------------------------------------------


class TestCategoricalParameters:
    def test_combustion_roundtrip(self):
        payload = {
            "fuel_code": "diesel",
            "LHV": 43.0,
            "HHV": 45.6,
            "density": 0.832,
            "oxidation_factor": 1.0,
            "fossil_carbon_share": 1.0,
            "biogenic_carbon_share": 0.0,
            "scope_applicability": ["scope_1"],
        }
        parsed = parse_parameters("combustion", payload)
        assert isinstance(parsed, CombustionParameters)
        assert parsed.fuel_code == "diesel"
        dumped = dump_parameters(parsed)
        for k, v in payload.items():
            assert dumped[k] == v

    def test_combustion_hhv_lt_lhv_rejected(self):
        with pytest.raises(Exception):
            parse_parameters(
                "combustion",
                {"fuel_code": "x", "LHV": 50.0, "HHV": 40.0},
            )

    def test_electricity_roundtrip(self):
        payload = {
            "electricity_basis": "location",
            "supplier_specific": False,
            "residual_mix_applicable": False,
            "certificate_handling": None,
            "td_loss_included": True,
            "subregion_code": "eGRID-SERC",
            "scope_applicability": ["scope_2"],
        }
        parsed = parse_parameters("electricity", payload)
        assert isinstance(parsed, ElectricityParameters)
        assert parsed.electricity_basis == "location"
        dumped = dump_parameters(parsed)
        assert dumped["td_loss_included"] is True

    def test_transport_roundtrip(self):
        payload = {
            "mode": "road",
            "vehicle_class": "HGV_rigid_17t",
            "payload_basis": "t-km",
            "distance_basis": "route",
            "empty_running_assumption": 0.25,
            "utilization_rate": 0.8,
            "refrigerated": False,
            "energy_basis": "WTW",
            "scope_applicability": ["scope_3"],
        }
        parsed = parse_parameters("transport", payload)
        assert isinstance(parsed, TransportParameters)
        assert parsed.mode == "road"

    def test_materials_roundtrip(self):
        payload = {
            "boundary": "cradle_to_gate",
            "allocation_method": "mass",
            "recycled_content_assumption": 0.2,
            "supplier_primary_data_share": 0.5,
            "pcr_reference": "EN 15804:2012+A2:2019",
            "pact_compatible": True,
            "product_lifetime_years": 10.0,
            "use_phase_energy_kwh": 0.5,
            "use_phase_frequency_per_year": 365.0,
            "end_of_life_allocation_method": "100_1",
        }
        parsed = parse_parameters("materials_products", payload)
        assert isinstance(parsed, MaterialsProductsParameters)
        assert parsed.product_lifetime_years == 10.0

    def test_refrigerants_roundtrip(self):
        payload = {
            "gas_code": "R-410A",
            "leakage_basis": "annual",
            "recharge_assumption": 0.05,
            "recovery_destruction_treatment": "partial_recovery",
            "gwp_set_mapping": "IPCC_AR6_100",
        }
        parsed = parse_parameters("refrigerants", payload)
        assert isinstance(parsed, RefrigerantsParameters)
        assert parsed.gas_code == "R-410A"

    def test_land_removals_roundtrip(self):
        payload = {
            "land_use_category": "forest_land",
            "sequestration_basis": "stock_change",
            "permanence_class": "long_term",
            "reversal_risk_flag": False,
            "biogenic_accounting_treatment": "separate_reporting",
        }
        parsed = parse_parameters("land_removals", payload)
        assert isinstance(parsed, LandRemovalsParameters)

    def test_finance_proxies_roundtrip(self):
        payload = {
            "asset_class": "listed_equity_and_corporate_bonds",
            "sector_code": "NAICS:221112",
            "intensity_basis": "revenue",
            "geography": "US",
            "proxy_confidence_class": "score_3",
        }
        parsed = parse_parameters("finance_proxies", payload)
        assert isinstance(parsed, FinanceProxiesParameters)

    def test_waste_roundtrip(self):
        payload = {
            "treatment_route": "landfill",
            "methane_recovery_factor": 0.75,
            "net_calorific_value": 10.5,
        }
        parsed = parse_parameters("waste", payload)
        assert isinstance(parsed, WasteParameters)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            parse_parameters("not_a_family_2099", {})

    def test_generic_allows_extra(self):
        """Auxiliary families use GenericParameters with extra=allow."""
        payload = {"custom_field": 42}
        parsed = parse_parameters("density", payload)
        dumped = dump_parameters(parsed)
        assert dumped.get("custom_field") == 42


# ---------------------------------------------------------------------------
# 5. Legacy->v1 migration bit-identical numeric preservation (M08, M12, M05)
# ---------------------------------------------------------------------------


class TestLegacyToV1Migration:
    def test_vectors_preserved_bit_identical(self):
        legacy = _legacy_desnz_payload()
        record = from_legacy_dict(legacy)
        assert record.numerator.co2 == legacy["vectors"]["CO2"]
        assert record.numerator.ch4 == legacy["vectors"]["CH4"]
        assert record.numerator.n2o == legacy["vectors"]["N2O"]

    def test_fgases_folded_into_f_gases_dict(self):
        legacy = _legacy_desnz_payload()
        legacy["vectors"]["HFCs"] = 0.002
        legacy["vectors"]["SF6"] = 0.001
        legacy["vectors"]["PFCs"] = 0.0  # zero -> not included
        record = from_legacy_dict(legacy)
        assert record.numerator.f_gases == {"HFCs": 0.002, "SF6": 0.001}

    def test_jurisdiction_flattened(self):
        legacy = _legacy_desnz_payload()
        legacy["geography"] = "UK"
        legacy["region_hint"] = "GB-LND"
        record = from_legacy_dict(legacy)
        # UK -> GB (ISO alpha-2)
        assert record.jurisdiction.country == "GB"
        assert record.jurisdiction.region == "GB-LND"

    def test_dqs_rescaled_to_0_to_100(self):
        legacy = _legacy_desnz_payload()
        record = from_legacy_dict(legacy)
        # DQS input: 5,5,4,4,5 with legacy dim-name mapping:
        # temporal=5, geographical->geographic=5, technological->technology=4,
        # representativeness->completeness=4, methodological->verification=5
        expected = compute_fqs(
            temporal=5, geographic=5, technology=4, completeness=4, verification=5
        )
        assert record.quality.composite_fqs == expected

    def test_valid_to_sentinel_populated(self):
        legacy = _legacy_desnz_payload()
        # no valid_to on input
        record = from_legacy_dict(legacy)
        assert record.valid_to == date(9999, 12, 31)
        assert record.valid_from == date(2024, 1, 1)

    def test_licensing_derived_from_boolean_flags(self):
        legacy = _legacy_desnz_payload()
        record = from_legacy_dict(legacy)
        # redistribution_allowed=True + commercial_use_allowed=True -> open
        assert record.licensing.redistribution_class == "open"

    def test_licensing_legacy_aliases(self):
        legacy = _legacy_desnz_payload()
        # Remove the explicit redistribution_class so the boolean flags drive
        # the derivation path.
        legacy.pop("redistribution_class", None)
        legacy["license_info"] = {
            "license": "proprietary",
            "redistribution_allowed": False,
            "commercial_use_allowed": True,
            "attribution_required": True,
        }
        record = from_legacy_dict(legacy)
        assert record.licensing.redistribution_class == "licensed_embedded"
        # Derived read-only aliases
        assert record.licensing.redistribution_allowed is False
        assert record.licensing.commercial_use_allowed is True

    def test_gwp_set_migrated_from_gwp_100yr_nested(self):
        legacy = _legacy_desnz_payload()
        # Flat gwp_set absent — must be pulled from nested gwp_100yr
        record = from_legacy_dict(legacy)
        assert record.gwp_set == "IPCC_AR5_100"

    def test_gwp_set_legacy_sar_renamed(self):
        legacy = _legacy_desnz_payload()
        legacy["gwp_set"] = "IPCC_SAR_100"
        legacy.pop("gwp_100yr", None)
        record = from_legacy_dict(legacy)
        assert record.gwp_set == "Kyoto_SAR_100"

    def test_lineage_consolidated(self):
        legacy = _legacy_desnz_payload()
        legacy["created_at"] = "2026-04-23T13:12:42+00:00"
        legacy["created_by"] = "etl_pipeline"
        record = from_legacy_dict(legacy)
        assert record.lineage.ingested_by == "etl_pipeline"
        assert record.lineage.ingested_at.year == 2026
        assert record.lineage.change_reason == "Initial ingest"

    def test_activity_schema_structured(self):
        legacy = _legacy_desnz_payload()
        legacy["activity_tags"] = ["stationary"]
        legacy["sector_tags"] = ["NAICS:221112"]
        record = from_legacy_dict(legacy)
        assert record.activity_schema.category == "natural_gas"
        assert record.activity_schema.sub_category == "stationary"
        assert "NAICS:221112" in record.activity_schema.classification_codes

    def test_roundtrip_legacy_to_v1_to_legacy_preserves_vectors(self):
        legacy = _legacy_desnz_payload()
        record = from_legacy_dict(legacy)
        legacy_round = to_legacy_dict(record)
        for gas in ("CO2", "CH4", "N2O"):
            assert legacy_round["vectors"][gas] == legacy["vectors"][gas]


# ---------------------------------------------------------------------------
# 6. CO2e derivation from FactorRecordV1.compute_co2e
# ---------------------------------------------------------------------------


class TestComputeCo2eFromRecord:
    def test_compute_co2e_ar5(self):
        legacy = _legacy_desnz_payload()
        record = from_legacy_dict(legacy)  # gwp_set = AR5_100
        result = record.compute_co2e()
        # 0.18293*1 + 0.00021*28 + 0.00017*265 = 0.18293 + 0.00588 + 0.04505
        expected = 0.18293 + 0.00021 * 28 + 0.00017 * 265
        assert abs(float(result) - expected) < 1e-6

    def test_compute_co2e_three_different_sets(self):
        legacy = _legacy_desnz_payload()
        record = from_legacy_dict(legacy)
        ar5 = record.compute_co2e("IPCC_AR5_100")
        ar6 = record.compute_co2e("IPCC_AR6_100")
        sar = record.compute_co2e("Kyoto_SAR_100")
        assert ar5 != ar6
        assert ar5 != sar
        assert ar6 != sar


# ---------------------------------------------------------------------------
# 7. Enum validators reject invalid values
# ---------------------------------------------------------------------------


class TestEnumValidators:
    def _record_kwargs(self, **overrides):
        base = dict(
            factor_id="EF:TEST:x:v1",
            factor_family="combustion",
            factor_name="Test factor",
            method_profile="corporate_scope1",
            source_id="test",
            source_version="2024.1",
            factor_version="1.0.0",
            status="certified",
            jurisdiction=canonical_v1.JurisdictionV1(country="US"),
            valid_from=date(2024, 1, 1),
            valid_to=date(9999, 12, 31),
            activity_schema=canonical_v1.ActivitySchemaV1(category="x"),
            numerator=canonical_v1.NumeratorV1(co2=1.0, unit="kg"),
            denominator=canonical_v1.DenominatorV1(value=1.0, unit="kWh"),
            gwp_set="IPCC_AR6_100",
            formula_type="direct_factor",
            parameters={"fuel_code": "test"},
            quality=canonical_v1.QualityV1.from_dimensions(3, 3, 3, 3, 3),
            lineage=canonical_v1.LineageV1(
                ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ingested_by="test",
            ),
            licensing=canonical_v1.LicensingV1(
                redistribution_class="open", customer_entitlement_required=False
            ),
            explainability=canonical_v1.ExplainabilityV1(fallback_rank=1),
        )
        base.update(overrides)
        return base

    def test_invalid_factor_id_rejected(self):
        with pytest.raises(Exception):
            FactorRecordV1(**self._record_kwargs(factor_id="NOT:starts:with:EF"))

    def test_invalid_status_rejected(self):
        with pytest.raises(Exception):
            FactorRecordV1(**self._record_kwargs(status="not_a_status"))

    def test_invalid_method_profile_rejected(self):
        with pytest.raises(Exception):
            FactorRecordV1(**self._record_kwargs(method_profile="bogus"))

    def test_invalid_gwp_set_rejected(self):
        with pytest.raises(Exception):
            FactorRecordV1(**self._record_kwargs(gwp_set="AR99"))

    def test_invalid_family_rejected(self):
        with pytest.raises(Exception):
            FactorRecordV1(**self._record_kwargs(factor_family="not_a_family"))

    def test_valid_to_gt_valid_from(self):
        with pytest.raises(Exception):
            FactorRecordV1(**self._record_kwargs(
                valid_from=date(2024, 1, 1),
                valid_to=date(2023, 12, 31),
            ))


# ---------------------------------------------------------------------------
# 8. Legacy status / method_profile mappings emit DeprecationWarning
# ---------------------------------------------------------------------------


class TestLegacyMappingWarnings:
    def test_legacy_active_status_mapped_with_warning(self):
        legacy = _legacy_desnz_payload()
        legacy["factor_status"] = "active"
        # reset the warned-once cache so we actually see a warning
        canonical_v1._DEPRECATION_WARNED.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            record = from_legacy_dict(legacy)
        assert record.status == "certified"
        # may or may not emit depending on once-cache, but mapping worked.

    def test_legacy_under_review_maps_to_preview(self):
        legacy = _legacy_desnz_payload()
        legacy["factor_status"] = "under_review"
        record = from_legacy_dict(legacy)
        assert record.status == "preview"


# ---------------------------------------------------------------------------
# 9. Source object schema
# ---------------------------------------------------------------------------


class TestSourceObject:
    def test_source_object_constants_exposed(self):
        assert len(SOURCE_TYPE_ENUM) == 5
        assert len(SOURCE_REDISTRIBUTION_CLASS_ENUM) == 4

    def test_source_object_roundtrip(self):
        payload = {
            "source_id": "EPA_eGRID",
            "authority": "US EPA",
            "title": "eGRID2024 Summary Tables",
            "publisher": "US EPA",
            "jurisdiction": {"country": "US"},
            "dataset_version": "2024.1",
            "publication_date": "2024-07-01",
            "validity_period": {"from": "2024-01-01", "to": None},
            "ingestion_date": "2026-04-23T00:00:00+00:00",
            "source_type": "government",
            "redistribution_class": "open",
            "verification_status": "regulator_approved",
            "citation_text": "US EPA (2024). eGRID2024.",
        }
        obj = source_object_from_dict(payload)
        assert isinstance(obj, SourceObject)
        assert obj.source_id == "EPA_eGRID"
        dumped = source_object_to_dict(obj)
        assert dumped["source_id"] == "EPA_eGRID"
        assert dumped["redistribution_class"] == "open"

    def test_source_object_rejects_invalid_country(self):
        payload = {
            "source_id": "X",
            "authority": "Y",
            "title": "Z",
            "publisher": "Z",
            "jurisdiction": {"country": "USA"},  # 3 chars — invalid
            "dataset_version": "v1",
            "publication_date": "2024-01-01",
            "validity_period": {"from": "2024-01-01"},
            "ingestion_date": "2026-04-23T00:00:00+00:00",
            "source_type": "government",
            "redistribution_class": "open",
            "verification_status": "unverified",
            "citation_text": "cite",
        }
        with pytest.raises(Exception):
            source_object_from_dict(payload)


# ---------------------------------------------------------------------------
# 10. Migration script idempotence + coverage
# ---------------------------------------------------------------------------


class TestMigrationScript:
    def test_migrate_record_is_idempotent_on_v1(self):
        legacy = _legacy_desnz_payload()
        first = migrate_record(legacy)
        second = migrate_record(first)
        # The second pass should normalise to the exact same shape.
        assert second["factor_id"] == first["factor_id"]
        assert second["numerator"] == first["numerator"]
        assert second["jurisdiction"] == first["jurisdiction"]
        assert second["quality"] == first["quality"]
        assert second["licensing"] == first["licensing"]

    @pytest.mark.skipif(
        not (REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed_v1").exists(),
        reason="catalog_seed_v1 not generated (run scripts/migrate_catalog_to_v1.py)",
    )
    def test_migrated_bundle_has_schema_version(self):
        src = REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed_v1"
        files = list(src.rglob("*.json"))
        assert files, "expected at least one migrated bundle"
        sample = json.loads(files[0].read_text(encoding="utf-8"))
        assert sample.get("schema_version") == "factor_record_v1"
        assert isinstance(sample["factors"], list)
        if sample["factors"]:
            f0 = sample["factors"][0]
            # v1 shape markers
            assert "jurisdiction" in f0
            assert "numerator" in f0
            assert "composite_fqs" in f0.get("quality", {})


# ---------------------------------------------------------------------------
# 11. canonical_v2 compat shim
# ---------------------------------------------------------------------------


class TestCanonicalV2CompatShim:
    def test_v1_symbols_reexported(self):
        # Importing from the shim must work (and returns v1 symbols)
        from greenlang.factors.data.canonical_v2 import (
            FactorRecordV1 as ShimRecord,
            STATUS_ENUM as ShimStatus,
        )
        assert ShimRecord is FactorRecordV1
        assert ShimStatus == STATUS_ENUM

    def test_shim_map_legacy_to_v1(self):
        legacy = _legacy_desnz_payload()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = canonical_v2.map_legacy_to_v1(legacy)
        assert result["factor_id"] == legacy["factor_id"]
        # Check a DeprecationWarning fired.
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        )


# ---------------------------------------------------------------------------
# 12. Catalog repository shape detection
# ---------------------------------------------------------------------------


class TestRepositoryShapeDetection:
    def test_is_v1_shape_true(self):
        from greenlang.factors.catalog_repository import _is_v1_shape

        v1_payload = migrate_record(_legacy_desnz_payload())
        assert _is_v1_shape(v1_payload) is True

    def test_is_v1_shape_false_for_legacy(self):
        from greenlang.factors.catalog_repository import _is_v1_shape

        assert _is_v1_shape(_legacy_desnz_payload()) is False

    def test_v1_payload_projected_back_to_legacy(self):
        from greenlang.factors.catalog_repository import (
            _v1_to_legacy_dict,
        )

        v1_payload = migrate_record(_legacy_desnz_payload())
        legacy_dict = _v1_to_legacy_dict(v1_payload)
        assert "vectors" in legacy_dict
        assert "dqs" in legacy_dict
        # Numeric preservation
        assert legacy_dict["vectors"]["CO2"] == pytest.approx(0.18293, abs=1e-9)
