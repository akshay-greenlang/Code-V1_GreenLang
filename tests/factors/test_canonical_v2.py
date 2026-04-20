# -*- coding: utf-8 -*-
"""Phase F1 — CTO canonical-v2 extensions + non-negotiable enforcement."""
from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from greenlang.data.canonical_v2 import (
    ActivitySchema,
    ChangeLogEntry,
    ElectricityBasis,
    Explainability,
    FactorFamily,
    FactorParameters,
    FormulaType,
    Jurisdiction,
    MethodProfile,
    NonNegotiableViolation,
    PrimaryDataFlag,
    RawRecordRef,
    RedistributionClass,
    SourceType,
    UncertaintyDistribution,
    Verification,
    VerificationStatus,
    enforce_license_class_homogeneity,
    validate_non_negotiables,
)


# --------------------------------------------------------------------------
# Enumerations cover the CTO spec
# --------------------------------------------------------------------------


class TestEnumerationCoverage:
    def test_factor_family_has_15_entries(self):
        # CTO spec lists exactly 15 factor families.
        assert len(list(FactorFamily)) == 15

    def test_factor_family_names_match_cto_spec(self):
        expected = {
            "emissions",
            "energy_conversion",
            "carbon_content",
            "oxidation",
            "heating_value",
            "density",
            "refrigerant_gwp",
            "grid_intensity",
            "residual_mix",
            "transport_lane",
            "material_embodied",
            "waste_treatment",
            "land_use_removals",
            "finance_proxy",
            "classification_mapping",
        }
        assert {f.value for f in FactorFamily} == expected

    def test_method_profile_covers_7_cto_packs(self):
        values = {m.value for m in MethodProfile}
        for profile in (
            "corporate_scope1",
            "corporate_scope2_location_based",
            "corporate_scope2_market_based",
            "corporate_scope3",
            "product_carbon",
            "freight_iso_14083",
            "land_removals",
            "finance_proxy",
            "eu_cbam",
        ):
            assert profile in values, profile

    def test_formula_type_has_transport_chain(self):
        assert FormulaType.TRANSPORT_CHAIN.value == "transport_chain"

    def test_redistribution_class_four_cto_tiers_present(self):
        values = {r.value for r in RedistributionClass}
        for expected in ("open", "licensed", "customer_private", "oem_redistributable"):
            assert expected in values

    def test_uncertainty_distributions(self):
        values = {u.value for u in UncertaintyDistribution}
        assert {"log_normal", "triangular", "uniform", "normal"} <= values

    def test_electricity_basis_has_four(self):
        values = {b.value for b in ElectricityBasis}
        assert values == {
            "location_based",
            "market_based",
            "supplier_specific",
            "residual_mix",
        }


# --------------------------------------------------------------------------
# Sub-object shapes
# --------------------------------------------------------------------------


class TestSubObjectShapes:
    def test_jurisdiction_defaults_all_none(self):
        j = Jurisdiction()
        assert j.country is None and j.region is None and j.grid_region is None

    def test_activity_schema_requires_category(self):
        with pytest.raises(TypeError):
            ActivitySchema()  # type: ignore[call-arg]

    def test_parameters_defaults(self):
        p = FactorParameters()
        assert p.residual_mix_applicable is False
        assert p.electricity_basis is None
        assert p.scope_applicability == []

    def test_verification_default_status(self):
        v = Verification()
        assert v.status == VerificationStatus.UNVERIFIED

    def test_explainability_default_fallback_rank(self):
        e = Explainability()
        assert e.fallback_rank == 7  # global default (lowest priority)
        assert e.assumptions == []

    def test_change_log_entry_required_fields(self):
        entry = ChangeLogEntry(
            changed_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
            changed_by="methodology_lead",
            change_reason="annual refresh",
        )
        assert entry.changed_by == "methodology_lead"

    def test_raw_record_ref(self):
        r = RawRecordRef(
            raw_record_id="epa-hub:2024:diesel-001",
            raw_payload_hash="a" * 64,
            raw_format="csv",
        )
        assert r.storage_uri is None


# --------------------------------------------------------------------------
# enforce_license_class_homogeneity (non-negotiable #4)
# --------------------------------------------------------------------------


class TestLicenseClassHomogeneity:
    def test_empty_iterable_ok(self):
        enforce_license_class_homogeneity([])

    def test_single_class_ok(self):
        recs = [
            SimpleNamespace(redistribution_class="open"),
            SimpleNamespace(redistribution_class="open"),
        ]
        enforce_license_class_homogeneity(recs)

    def test_mixed_classes_raise(self):
        recs = [
            SimpleNamespace(redistribution_class="open"),
            SimpleNamespace(redistribution_class="licensed"),
        ]
        with pytest.raises(NonNegotiableViolation):
            enforce_license_class_homogeneity(recs)

    def test_legacy_license_class_attribute_read(self):
        recs = [
            SimpleNamespace(license_class="public_us_government"),
            SimpleNamespace(license_class="registry_terms"),
        ]
        with pytest.raises(NonNegotiableViolation):
            enforce_license_class_homogeneity(recs)

    def test_records_without_class_field_skipped(self):
        recs = [
            SimpleNamespace(other="ignored"),
            SimpleNamespace(redistribution_class="open"),
        ]
        enforce_license_class_homogeneity(recs)


# --------------------------------------------------------------------------
# validate_non_negotiables (per-record enforcement)
# --------------------------------------------------------------------------


def _good_record(**overrides):
    base = SimpleNamespace(
        factor_id="test-001",
        vectors=SimpleNamespace(CO2=0.5),           # #1 satisfied
        valid_from=date(2026, 1, 1),                # #5 satisfied
        release_version="1.0.0",                    # #5 + #2 satisfied
        factor_status="certified",
        provenance=SimpleNamespace(version="1"),
        compliance_frameworks=["GHG_Protocol"],
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class TestValidateNonNegotiables:
    def test_valid_record_passes(self):
        validate_non_negotiables(_good_record())

    def test_missing_vectors_raises_1(self):
        rec = _good_record(vectors=None)
        with pytest.raises(NonNegotiableViolation, match="non-negotiable #1"):
            validate_non_negotiables(rec)

    def test_missing_valid_from_raises_5(self):
        rec = _good_record(valid_from=None)
        with pytest.raises(NonNegotiableViolation, match="non-negotiable #5"):
            validate_non_negotiables(rec)

    def test_missing_source_version_raises_5(self):
        rec = _good_record(release_version=None, provenance=None, source_release=None)
        with pytest.raises(NonNegotiableViolation, match="non-negotiable #5"):
            validate_non_negotiables(rec)

    def test_certified_regulated_without_method_profile_raises_6(self):
        rec = _good_record(
            compliance_frameworks=["CBAM"],
            method_profile=None,
        )
        with pytest.raises(NonNegotiableViolation, match="non-negotiable #6"):
            validate_non_negotiables(rec)

    def test_certified_regulated_with_method_profile_passes(self):
        rec = _good_record(
            compliance_frameworks=["CBAM"],
            method_profile="eu_cbam",
        )
        validate_non_negotiables(rec)

    def test_preview_skips_version_check(self):
        rec = _good_record(factor_status="preview", factor_version=None, release_version=None)
        # Still requires valid_from + provenance, but NOT factor_version in preview.
        validate_non_negotiables(rec)


# --------------------------------------------------------------------------
# EmissionFactorRecord backward-compat additions
# --------------------------------------------------------------------------


def _minimal_efr():
    """Build a minimal EmissionFactorRecord exercising the legacy path.

    Helper duplicated from the existing test_emission_factor_record suite.
    """
    from greenlang.data.emission_factor_record import (
        Boundary,
        DataQualityScore,
        EmissionFactorRecord,
        GeographyLevel,
        GHGVectors,
        GWPSet,
        GWPValues,
        LicenseInfo,
        Methodology,
        Scope,
        SourceProvenance,
    )

    return EmissionFactorRecord(
        factor_id="EF:US:diesel:2024:v1",
        fuel_type="diesel",
        unit="gallons",
        geography="US",
        geography_level=GeographyLevel.COUNTRY,
        vectors=GHGVectors(CO2=10.18, CH4=0.00082, N2O=0.000164),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_1,
        boundary=Boundary.COMBUSTION,
        provenance=SourceProvenance(
            source_org="EPA",
            source_publication="GHG Inventories",
            source_year=2024,
            methodology=Methodology.IPCC_TIER_1,
        ),
        valid_from=date(2024, 1, 1),
        uncertainty_95ci=0.05,
        dqs=DataQualityScore(
            temporal=5, geographical=4, technological=4,
            representativeness=4, methodological=5,
        ),
        license_info=LicenseInfo(
            license="CC0-1.0",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=False,
        ),
    )


class TestEmissionFactorRecordBackwardCompat:
    def test_record_loads_without_new_fields(self):
        rec = _minimal_efr()
        assert rec.factor_family is None
        assert rec.method_profile is None
        assert rec.factor_version is None
        assert rec.jurisdiction is None
        assert rec.activity_schema is None
        assert rec.parameters is None
        assert rec.explainability is None
        assert rec.change_log == []

    def test_record_accepts_new_fields(self):
        rec = _minimal_efr()
        rec.factor_family = FactorFamily.EMISSIONS.value
        rec.method_profile = MethodProfile.CORPORATE_SCOPE1.value
        rec.factor_version = "1.0.0"
        rec.formula_type = FormulaType.COMBUSTION.value
        rec.jurisdiction = Jurisdiction(country="US")
        rec.activity_schema = ActivitySchema(
            category="stationary_combustion",
            sub_category="diesel",
            classification_codes=["NAICS:221112"],
        )
        rec.parameters = FactorParameters(
            scope_applicability=["scope1"],
            supplier_specific=False,
            biogenic_share=0.0,
        )
        rec.explainability = Explainability(
            assumptions=["Default EPA factor used when no primary data."],
            fallback_rank=5,
            rationale="EPA 2024 Tier-1 diesel factor.",
        )
        rec.primary_data_flag = PrimaryDataFlag.SECONDARY.value
        rec.uncertainty_distribution = UncertaintyDistribution.LOG_NORMAL.value
        rec.redistribution_class = RedistributionClass.OPEN.value
        rec.next_review_date = date(2027, 1, 1)
        rec.change_log.append(
            ChangeLogEntry(
                changed_at=datetime.now(timezone.utc),
                changed_by="methodology_lead",
                change_reason="initial import",
            )
        )
        # All assignments survive.
        assert rec.factor_family == "emissions"
        assert rec.jurisdiction.country == "US"
        assert rec.activity_schema.classification_codes == ["NAICS:221112"]
        assert rec.explainability.fallback_rank == 5
        assert len(rec.change_log) == 1


# --------------------------------------------------------------------------
# SourceRegistryEntry F1 extension fields are optional
# --------------------------------------------------------------------------


class TestSourceRegistryExtensions:
    def test_registry_entry_supports_new_fields(self):
        from greenlang.factors.source_registry import SourceRegistryEntry

        entry = SourceRegistryEntry(
            source_id="epa_hub",
            display_name="EPA GHG Hub",
            connector_only=False,
            license_class="public_us_government",
            redistribution_allowed=True,
            derivative_works_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
            citation_text="U.S. EPA.",
            cadence="quarterly",
            watch_mechanism="http_head",
            watch_url="https://www.epa.gov/ghgemissionfactors",
            watch_file_type=None,
            approval_required_for_certified=True,
            legal_signoff_artifact=None,
            legal_signoff_version=None,
            publisher="U.S. Environmental Protection Agency",
            jurisdiction="US",
            dataset_version="2024-Q4",
            publication_date="2024-10-15",
            validity_period="2024-01-01/2024-12-31",
            ingestion_date="2026-04-20T00:00:00+00:00",
            source_type=SourceType.GOVERNMENT.value,
            verification_status=VerificationStatus.REGULATOR_APPROVED.value,
            change_log_uri="https://www.epa.gov/ghgemissionfactors/changelog",
            legal_notes="Attribution required; commercial redistribution OK.",
        )
        assert entry.publisher == "U.S. Environmental Protection Agency"
        assert entry.source_type == "government"
        assert entry.verification_status == "regulator_approved"

    def test_existing_yaml_still_loads(self):
        """Phase F1 must not break the existing source_registry.yaml."""
        from greenlang.factors.source_registry import load_source_registry

        entries = load_source_registry()
        assert len(entries) > 0
        for entry in entries:
            # Legacy fields still populated.
            assert entry.source_id
            assert entry.display_name
            # New fields default to None when YAML doesn't carry them.
            assert entry.publisher is None or isinstance(entry.publisher, str)
            assert entry.source_type is None or isinstance(entry.source_type, str)
