# -*- coding: utf-8 -*-
"""
Unit tests for TCFD-to-ISSB Cross-Walk Engine.

Tests TCFD-to-ISSB mapping completeness, ISSB compliance checking,
gap identification, additional IFRS S2 requirements, migration pathway,
and dual scorecard with 26+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    TCFD_TO_IFRS_S2_MAPPING,
    TCFD_DISCLOSURES,
    ISSB_CROSS_INDUSTRY_METRICS,
    ISSBMetricType,
)
from services.models import (
    ISSBMapping,
    _new_id,
)


# ===========================================================================
# TCFD-to-ISSB Mapping Completeness
# ===========================================================================

class TestMappingCompleteness:
    """Test TCFD-to-ISSB mapping completeness."""

    def test_all_11_disclosures_mapped(self):
        for disclosure_ref in TCFD_DISCLOSURES:
            assert disclosure_ref in TCFD_TO_IFRS_S2_MAPPING, \
                f"Missing ISSB mapping for {disclosure_ref}"

    def test_mapping_count(self):
        assert len(TCFD_TO_IFRS_S2_MAPPING) == 11

    @pytest.mark.parametrize("disclosure_ref", list(TCFD_DISCLOSURES.keys()))
    def test_mapping_has_required_fields(self, disclosure_ref):
        mapping = TCFD_TO_IFRS_S2_MAPPING[disclosure_ref]
        assert "ifrs_s2_paragraph" in mapping
        assert "ifrs_s2_topic" in mapping
        assert "mapping_status" in mapping
        assert "notes" in mapping

    def test_issb_mapping_model_creation(self, sample_issb_mappings):
        assert len(sample_issb_mappings) == 3


# ===========================================================================
# ISSB Compliance Checking
# ===========================================================================

class TestISSBComplianceChecking:
    """Test ISSB compliance status checking."""

    @pytest.mark.parametrize("disclosure_ref,expected_status", [
        ("gov_a", "fully_mapped"),
        ("gov_b", "fully_mapped"),
        ("str_a", "fully_mapped"),
        ("str_b", "fully_mapped"),
        ("str_c", "enhanced"),
        ("rm_a", "fully_mapped"),
        ("rm_b", "fully_mapped"),
        ("rm_c", "fully_mapped"),
        ("mt_a", "enhanced"),
        ("mt_b", "enhanced"),
        ("mt_c", "enhanced"),
    ])
    def test_mapping_status(self, disclosure_ref, expected_status):
        mapping = TCFD_TO_IFRS_S2_MAPPING[disclosure_ref]
        assert mapping["mapping_status"] == expected_status

    def test_fully_mapped_count(self):
        fully_mapped = [
            ref for ref, m in TCFD_TO_IFRS_S2_MAPPING.items()
            if m["mapping_status"] == "fully_mapped"
        ]
        assert len(fully_mapped) == 7

    def test_enhanced_count(self):
        enhanced = [
            ref for ref, m in TCFD_TO_IFRS_S2_MAPPING.items()
            if m["mapping_status"] == "enhanced"
        ]
        assert len(enhanced) == 4

    @pytest.mark.parametrize("status", ["fully_mapped", "enhanced", "partial", "gap"])
    def test_valid_mapping_statuses(self, status):
        mapping = ISSBMapping(
            tcfd_disclosure_ref="gov_a",
            ifrs_s2_paragraph="5-6",
            mapping_status=status,
        )
        assert mapping.mapping_status == status


# ===========================================================================
# Gap Identification
# ===========================================================================

class TestGapIdentification:
    """Test ISSB gap identification."""

    def test_str_c_has_enhancement_note(self):
        mapping = TCFD_TO_IFRS_S2_MAPPING["str_c"]
        assert "resilience" in mapping["notes"].lower()

    def test_mt_b_scope3_requirement(self):
        mapping = TCFD_TO_IFRS_S2_MAPPING["mt_b"]
        assert "scope 3" in mapping["notes"].lower()

    def test_mt_a_cross_industry_metrics(self):
        mapping = TCFD_TO_IFRS_S2_MAPPING["mt_a"]
        assert "7 cross-industry" in mapping["notes"] or "cross-industry" in mapping["notes"]

    def test_gap_mapping_with_description(self, sample_issb_mappings):
        enhanced = [m for m in sample_issb_mappings if m.mapping_status == "enhanced"]
        for mapping in enhanced:
            assert mapping.gap_description is not None
            assert len(mapping.gap_description) > 0


# ===========================================================================
# Additional IFRS S2 Requirements
# ===========================================================================

class TestAdditionalRequirements:
    """Test additional IFRS S2 requirements beyond TCFD."""

    def test_seven_cross_industry_metrics(self):
        assert len(ISSB_CROSS_INDUSTRY_METRICS) == 7

    @pytest.mark.parametrize("metric_type", list(ISSBMetricType))
    def test_each_metric_has_paragraph_ref(self, metric_type):
        metric = ISSB_CROSS_INDUSTRY_METRICS[metric_type]
        assert "ifrs_s2_paragraph" in metric
        assert metric["ifrs_s2_paragraph"].startswith("29")

    def test_governance_paragraph_refs(self):
        gov_a = TCFD_TO_IFRS_S2_MAPPING["gov_a"]
        assert gov_a["ifrs_s2_paragraph"] == "5-6"

    def test_strategy_paragraph_refs(self):
        str_a = TCFD_TO_IFRS_S2_MAPPING["str_a"]
        assert str_a["ifrs_s2_paragraph"] == "10-12"


# ===========================================================================
# Migration Pathway
# ===========================================================================

class TestMigrationPathway:
    """Test TCFD-to-ISSB migration pathway."""

    def test_migration_action_in_mapping(self, sample_issb_mappings):
        enhanced = [m for m in sample_issb_mappings if m.mapping_status == "enhanced"]
        for mapping in enhanced:
            assert mapping.action_required is not None

    def test_fully_mapped_no_action(self):
        mapping = ISSBMapping(
            tcfd_disclosure_ref="gov_a",
            ifrs_s2_paragraph="5-6",
            mapping_status="fully_mapped",
        )
        assert mapping.action_required is None

    def test_gap_requires_action(self):
        mapping = ISSBMapping(
            tcfd_disclosure_ref="test",
            ifrs_s2_paragraph="99",
            mapping_status="gap",
            gap_description="Missing requirement",
            action_required="Implement missing disclosure",
        )
        assert mapping.action_required is not None
        assert len(mapping.action_required) > 0


# ===========================================================================
# Dual Scorecard
# ===========================================================================

class TestDualScorecard:
    """Test dual TCFD/ISSB compliance scorecard."""

    def test_tcfd_coverage(self):
        tcfd_disclosures = len(TCFD_DISCLOSURES)
        assert tcfd_disclosures == 11

    def test_issb_coverage(self):
        issb_mappings = len(TCFD_TO_IFRS_S2_MAPPING)
        assert issb_mappings == 11

    def test_tcfd_issb_alignment_percentage(self):
        fully_mapped = sum(
            1 for m in TCFD_TO_IFRS_S2_MAPPING.values()
            if m["mapping_status"] == "fully_mapped"
        )
        alignment_pct = (fully_mapped / len(TCFD_TO_IFRS_S2_MAPPING)) * 100
        assert alignment_pct > 50

    def test_mapping_model_timestamps(self, sample_issb_mappings):
        for mapping in sample_issb_mappings:
            assert mapping.created_at is not None
