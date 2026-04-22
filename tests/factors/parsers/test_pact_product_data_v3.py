# -*- coding: utf-8 -*-
"""Tests for the PACT v3 schema extensions in ``pact_product_data`` parser.

Covers:
* round-trip of a PACT v3 product-footprint row (all new fields present)
* enum validation for ``assurance_level`` (none / limited / reasonable + unknown)
* backward-compat path: v2 rows still parse and keep their semantics
* v2->v3 upgrade: a row that declares ``pcfSpec 3.x`` with v2-style supplier
  primary data share is promoted to the v3 ``primaryDataShare`` slot
"""
from __future__ import annotations

import pytest

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
    PrimaryDataFlag,
    RedistributionClass,
    VerificationStatus,
)
from greenlang.factors.ingestion.parsers.pact_product_data import (
    PACTAssuranceLevel,
    PACTDataQualityIndicatorV3,
    parse_pact_rows,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _v2_row() -> dict:
    return {
        "id": "urn:gl:pact:product:STEEL-V2",
        "productName": "Hot-rolled steel coil (v2)",
        "productCategoryCpc": "41237",
        "pcf": {
            "declaredUnit": "kg",
            "pCfExcludingBiogenic": "2.34",
            "biogenicCarbonEmissions": "0.05",
            "geographyCountrySubdivision": "DE",
            "supplierPrimaryDataShare": 0.6,
        },
        "companyName": "Acme Steel",
        "periodCoveredStart": "2024-01-01",
        "periodCoveredEnd": "2024-12-31",
        "version": 2,
        "pcfSpec": "2.0.0",
    }


def _v3_row() -> dict:
    return {
        "id": "urn:gl:pact:product:STEEL-V3",
        "productName": "Hot-rolled steel coil (v3)",
        "productCategoryCpc": "41237",
        "pcf": {
            "declaredUnit": "kg",
            "pCfExcludingBiogenic": "2.10",
            "biogenicCarbonEmissions": "0.04",
            "geographyCountrySubdivision": "DE",
            "primaryDataShare": 0.82,
            "dataQualityIndicator": {
                "coveragePercent": 95.0,
                "geographicalRepresentativeness": 4,
                "temporalRepresentativeness": 5,
                "technologicalRepresentativeness": 4,
                "dataQualityRating": 4,
            },
            "crossSectoralStandardsUsed": ["GHG Protocol Product Standard"],
            "productOrSectorSpecificRules": [
                {"operator": "PEF", "ruleNames": ["Steel PEFCR"], "version": "3.1"}
            ],
        },
        "assurance": {
            "coverage": "product line",
            "level": "reasonable",
            "provider": "TUV Rheinland",
        },
        "companyName": "Acme Steel",
        "periodCoveredStart": "2024-01-01",
        "periodCoveredEnd": "2024-12-31",
        "version": 1,
        "pcfSpec": "3.0.0",
    }


# ---------------------------------------------------------------------------
# Assurance enum validation
# ---------------------------------------------------------------------------


class TestAssuranceLevelEnum:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, PACTAssuranceLevel.NONE),
            ("", PACTAssuranceLevel.NONE),
            ("none", PACTAssuranceLevel.NONE),
            ("self_declared", PACTAssuranceLevel.NONE),
            ("self-declared", PACTAssuranceLevel.NONE),
            ("Limited", PACTAssuranceLevel.LIMITED),
            ("limited_assurance", PACTAssuranceLevel.LIMITED),
            ("REASONABLE", PACTAssuranceLevel.REASONABLE),
            ("reasonable_assurance", PACTAssuranceLevel.REASONABLE),
        ],
    )
    def test_parse_accepts_valid_variants(self, raw, expected):
        assert PACTAssuranceLevel.parse(raw) is expected

    def test_parse_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="Unknown PACT assurance level"):
            PACTAssuranceLevel.parse("absolute")

    def test_enum_values_match_pact_spec(self):
        # PACT v3 §4.3 formally allows exactly three values.
        assert {m.value for m in PACTAssuranceLevel} == {
            "none",
            "limited",
            "reasonable",
        }


# ---------------------------------------------------------------------------
# v3 round-trip
# ---------------------------------------------------------------------------


class TestPACTv3RoundTrip:
    def test_v3_row_parses_all_new_fields(self):
        records = parse_pact_rows([_v3_row()])
        assert len(records) == 1
        rec = records[0]

        # Baseline identity invariants
        assert rec.factor_family == FactorFamily.MATERIAL_EMBODIED.value
        assert rec.method_profile == MethodProfile.PRODUCT_CARBON.value
        assert rec.formula_type == FormulaType.LCA.value
        assert rec.redistribution_class == RedistributionClass.RESTRICTED.value

        # v3-specific tagging
        assert "pact_v3" in rec.tags
        assert "assurance_reasonable" in rec.tags
        assert "PACT_v3" in rec.compliance_frameworks
        assert rec.source_id == "pact_pathfinder"
        assert rec.source_release == "3.0.0"

        # Assurance -> Verification mapping
        assert rec.verification.status == VerificationStatus.REGULATOR_APPROVED
        assert rec.verification.verified_by == "TUV Rheinland"
        assert "assurance=reasonable" in (
            rec.verification.verification_reference or ""
        )

        # Primary data share >=0.8 -> PrimaryDataFlag.PRIMARY
        assert rec.primary_data_flag == PrimaryDataFlag.PRIMARY.value

        # DQI scores are propagated into the DQS block
        assert rec.dqs.geographical == 4
        assert rec.dqs.temporal == 5
        assert rec.dqs.technological == 4

        # Every new field shows up in the explainability assumptions
        joined = " | ".join(rec.explainability.assumptions)
        assert "Pathfinder Framework v3" in joined
        assert "Assurance level: reasonable" in joined
        assert "TUV Rheinland" in joined
        assert "0.82" in joined
        assert "DQI composite score: 4/5" in joined
        assert "Cross-sectoral standards used" in joined
        assert "version: 3.1" in joined

    def test_assurance_limited_maps_to_external_verified(self):
        row = _v3_row()
        row["assurance"]["level"] = "limited"
        row["assurance"]["provider"] = "DNV"
        rec = parse_pact_rows([row])[0]
        assert rec.verification.status == VerificationStatus.EXTERNAL_VERIFIED
        assert rec.verification.verified_by == "DNV"
        assert "assurance_limited" in rec.tags

    def test_assurance_none_on_v3_maps_to_unverified(self):
        row = _v3_row()
        row["assurance"] = {"level": "none"}
        rec = parse_pact_rows([row])[0]
        assert rec.verification.status == VerificationStatus.UNVERIFIED

    def test_invalid_assurance_level_is_skipped(self):
        row = _v3_row()
        row["assurance"]["level"] = "absolute"
        # Parser logs + skips; no record emitted.
        assert parse_pact_rows([row]) == []


# ---------------------------------------------------------------------------
# Backward compat for v2
# ---------------------------------------------------------------------------


class TestPACTv2BackCompat:
    def test_v2_row_still_parses(self):
        rec = parse_pact_rows([_v2_row()])[0]
        # v2 tag NOT present
        assert "pact_v3" not in rec.tags
        assert "PACT_v3" not in rec.compliance_frameworks
        # v2 source_id preserved for legacy callers
        assert rec.source_id == "pact_exchange"
        # v2 default verification status preserved
        assert rec.verification.status == VerificationStatus.EXTERNAL_VERIFIED
        # supplierPrimaryDataShare 0.6 -> supplier_specific True
        assert rec.parameters.supplier_specific is True

    def test_v2_row_default_pdf_flag(self):
        rec = parse_pact_rows([_v2_row()])[0]
        # supplierPrimaryDataShare 0.6 (<0.8) -> PRIMARY_MODELED
        assert rec.primary_data_flag == PrimaryDataFlag.PRIMARY_MODELED.value


# ---------------------------------------------------------------------------
# v2 -> v3 upgrade path
# ---------------------------------------------------------------------------


class TestV2toV3UpgradePath:
    def test_v3_spec_with_legacy_supplier_share_is_promoted(self):
        """A row with pcfSpec=3.0.0 but a v2-style supplierPrimaryDataShare
        field must treat that value as the v3 ``primaryDataShare``.
        """
        row = _v2_row()
        row["pcfSpec"] = "3.0.0"
        row["version"] = 1
        # No explicit v3 primaryDataShare key - legacy key only.
        rec = parse_pact_rows([row])[0]
        assert "pact_v3" in rec.tags

        # supplier_specific == True (share 0.6 >= 0.5)
        assert rec.parameters.supplier_specific is True

        # v2-style share appears in the v3 explainability text
        joined = " | ".join(rec.explainability.assumptions)
        assert "Primary data share (v3 §4.2.2): 0.60" in joined

    def test_v3_detection_via_datafields_without_spec_version(self):
        """Rows that are v3-shaped but still declare pcfSpec=2.x should
        still be recognized as v3 (duck typing of v3 payloads).
        """
        row = _v2_row()
        # Keep pcfSpec 2.0.0 but add a v3-only field.
        row["pcf"]["dataQualityIndicator"] = {
            "dataQualityRating": 3,
            "geographicalRepresentativeness": 3,
        }
        rec = parse_pact_rows([row])[0]
        assert "pact_v3" in rec.tags
        assert rec.dqs.representativeness == 3
        assert rec.dqs.geographical == 3

    def test_primary_data_share_v3_overrides_v2_when_both_present(self):
        row = _v3_row()
        row["pcf"]["supplierPrimaryDataShare"] = 0.1  # stale v2 key
        # v3 primaryDataShare=0.82 should win -> PRIMARY
        rec = parse_pact_rows([row])[0]
        assert rec.primary_data_flag == PrimaryDataFlag.PRIMARY.value


# ---------------------------------------------------------------------------
# DQI helper
# ---------------------------------------------------------------------------


class TestDQIHelper:
    def test_dqi_dataclass_initialises(self):
        dqi = PACTDataQualityIndicatorV3(
            coverage_percent=90.0,
            geographical=3,
            temporal=4,
            technological=2,
            data_quality_rating=3,
        )
        assert dqi.geographical == 3
        assert dqi.technological == 2
        assert dqi.data_quality_rating == 3

    def test_dqi_clamps_out_of_range_pedigree_scores(self):
        row = _v3_row()
        row["pcf"]["dataQualityIndicator"] = {
            "dataQualityRating": 9,           # > 5 should clamp
            "geographicalRepresentativeness": 0,  # < 1 should clamp
            "temporalRepresentativeness": 5,
            "technologicalRepresentativeness": 5,
        }
        rec = parse_pact_rows([row])[0]
        assert rec.dqs.representativeness == 5
        assert rec.dqs.geographical == 1
