# -*- coding: utf-8 -*-
"""
Unit tests for CompletenessValidationEngine - AGENT-EUDR-027

Tests completeness validation of Article 9 information elements including
weighted scoring, three-tier classification (INSUFFICIENT/PARTIAL/COMPLETE),
simplified due diligence thresholds, gap report generation with per-element
remediation actions, and validation statistics.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 5: Completeness Validation)
"""
from __future__ import annotations

from decimal import Decimal
from typing import Dict

import pytest

from greenlang.agents.eudr.information_gathering.completeness_validation_engine import (
    CompletenessValidationEngine,
)
from greenlang.agents.eudr.information_gathering.config import InformationGatheringConfig
from greenlang.agents.eudr.information_gathering.models import (
    Article9ElementName,
    Article9ElementStatus,
    CompletenessClassification,
    ElementStatus,
    EUDRCommodity,
)


@pytest.fixture
def engine(config) -> CompletenessValidationEngine:
    return CompletenessValidationEngine(config)


def _make_elements(
    statuses: Dict[str, ElementStatus],
    confidence: Decimal = Decimal("0.95"),
) -> Dict[str, Article9ElementStatus]:
    """Helper to build Article 9 element status dicts."""
    elements: Dict[str, Article9ElementStatus] = {}
    for elem in Article9ElementName:
        status = statuses.get(elem.value, ElementStatus.MISSING)
        elements[elem.value] = Article9ElementStatus(
            element_name=elem.value,
            status=status,
            source="test_source",
            value_summary=f"Test data for {elem.value}",
            confidence=confidence if status != ElementStatus.MISSING else Decimal("0"),
        )
    return elements


class TestValidateCompleteness:
    """Test full completeness validation flow."""

    def test_validate_completeness_complete(self, engine, sample_article9_elements):
        report = engine.validate_completeness(
            "OP-001", EUDRCommodity.COFFEE, sample_article9_elements
        )
        assert report.completeness_classification == CompletenessClassification.COMPLETE
        assert report.completeness_score >= Decimal("90")
        assert report.gap_report.total_gaps == 0
        assert len(report.provenance_hash) == 64

    def test_validate_completeness_partial(self, engine):
        statuses = {e.value: ElementStatus.COMPLETE for e in Article9ElementName}
        # Make 3 elements partial
        statuses["geolocation"] = ElementStatus.PARTIAL
        statuses["deforestation_free_evidence"] = ElementStatus.PARTIAL
        statuses["supply_chain_information"] = ElementStatus.PARTIAL
        elements = _make_elements(statuses)
        report = engine.validate_completeness(
            "OP-002", EUDRCommodity.COCOA, elements
        )
        assert report.completeness_classification == CompletenessClassification.PARTIAL
        assert Decimal("60") <= report.completeness_score < Decimal("90")

    def test_validate_completeness_insufficient(self, engine):
        statuses = {e.value: ElementStatus.MISSING for e in Article9ElementName}
        # Make only 3 complete
        statuses["product_description"] = ElementStatus.COMPLETE
        statuses["quantity"] = ElementStatus.COMPLETE
        statuses["buyer_identification"] = ElementStatus.COMPLETE
        elements = _make_elements(statuses)
        report = engine.validate_completeness(
            "OP-003", EUDRCommodity.SOYA, elements
        )
        assert report.completeness_classification == CompletenessClassification.INSUFFICIENT
        assert report.completeness_score < Decimal("60")

    def test_validate_completeness_all_missing(self, engine):
        elements: Dict[str, Article9ElementStatus] = {}
        report = engine.validate_completeness(
            "OP-004", EUDRCommodity.WOOD, elements
        )
        assert report.completeness_classification == CompletenessClassification.INSUFFICIENT
        assert report.completeness_score == Decimal("0.00")
        assert report.gap_report.total_gaps == 10

    def test_validate_completeness_tracks_history(self, engine, sample_article9_elements):
        engine.validate_completeness("OP-H1", EUDRCommodity.COFFEE, sample_article9_elements)
        engine.validate_completeness("OP-H2", EUDRCommodity.COCOA, sample_article9_elements)
        stats = engine.get_validation_stats()
        assert stats["total_validations"] == 2


class TestComputeElementScore:
    """Test individual element score computation."""

    def test_compute_element_score_complete(self, engine):
        element = Article9ElementStatus(
            element_name="product_description",
            status=ElementStatus.COMPLETE,
            confidence=Decimal("0.95"),
        )
        score = engine.compute_element_score(element)
        assert score == Decimal("0.9500")

    def test_compute_element_score_partial(self, engine):
        element = Article9ElementStatus(
            element_name="geolocation",
            status=ElementStatus.PARTIAL,
            confidence=Decimal("0.80"),
        )
        score = engine.compute_element_score(element)
        assert score == Decimal("0.4000")  # 0.5 * 0.80

    def test_compute_element_score_missing(self, engine):
        element = Article9ElementStatus(
            element_name="quantity",
            status=ElementStatus.MISSING,
            confidence=Decimal("0"),
        )
        score = engine.compute_element_score(element)
        assert score == Decimal("0")

    def test_compute_element_score_zero_confidence_complete(self, engine):
        element = Article9ElementStatus(
            element_name="quantity",
            status=ElementStatus.COMPLETE,
            confidence=Decimal("0"),
        )
        score = engine.compute_element_score(element)
        # Zero confidence defaults to 1.0 for complete elements
        assert score == Decimal("1.0000")


class TestClassifyCompleteness:
    """Test completeness classification thresholds."""

    def test_classify_completeness_insufficient(self, engine):
        result = engine.classify_completeness(Decimal("45"))
        assert result == CompletenessClassification.INSUFFICIENT

    def test_classify_completeness_partial(self, engine):
        result = engine.classify_completeness(Decimal("75"))
        assert result == CompletenessClassification.PARTIAL

    def test_classify_completeness_complete(self, engine):
        result = engine.classify_completeness(Decimal("95"))
        assert result == CompletenessClassification.COMPLETE

    def test_classify_completeness_boundary_insufficient(self, engine):
        result = engine.classify_completeness(Decimal("59.99"))
        assert result == CompletenessClassification.INSUFFICIENT

    def test_classify_completeness_boundary_partial(self, engine):
        result = engine.classify_completeness(Decimal("60"))
        assert result == CompletenessClassification.PARTIAL

    def test_classify_completeness_boundary_complete(self, engine):
        result = engine.classify_completeness(Decimal("90"))
        assert result == CompletenessClassification.COMPLETE


class TestSimplifiedDueDiligence:
    """Test simplified due diligence with relaxed thresholds."""

    def test_simplified_dd_lower_thresholds(self, engine):
        # Under standard: 55 would be INSUFFICIENT
        # Under simplified DD: 55 is PARTIAL (threshold is 50)
        result = engine.classify_completeness(Decimal("55"), is_simplified_dd=True)
        assert result == CompletenessClassification.PARTIAL

    def test_simplified_dd_complete_threshold(self, engine):
        # Under simplified DD: 80 is COMPLETE (vs 90 for standard)
        result = engine.classify_completeness(Decimal("80"), is_simplified_dd=True)
        assert result == CompletenessClassification.COMPLETE

    def test_simplified_dd_insufficient(self, engine):
        result = engine.classify_completeness(Decimal("45"), is_simplified_dd=True)
        assert result == CompletenessClassification.INSUFFICIENT

    def test_simplified_dd_in_validation(self, engine, sample_article9_elements):
        report = engine.validate_completeness(
            "OP-SDD", EUDRCommodity.COFFEE, sample_article9_elements,
            is_simplified_dd=True,
        )
        assert report.is_simplified_dd is True


class TestGapReport:
    """Test gap report generation."""

    def test_generate_gap_report_no_gaps(self, engine, sample_article9_elements):
        gap_report = engine.generate_gap_report(sample_article9_elements)
        assert gap_report.total_gaps == 0
        assert gap_report.items == []

    def test_generate_gap_report_with_gaps(self, engine):
        statuses = {e.value: ElementStatus.COMPLETE for e in Article9ElementName}
        statuses["geolocation"] = ElementStatus.MISSING
        statuses["deforestation_free_evidence"] = ElementStatus.PARTIAL
        elements = _make_elements(statuses)
        gap_report = engine.generate_gap_report(elements)
        assert gap_report.total_gaps == 2
        assert gap_report.critical_gaps >= 1  # geolocation missing is critical
        gap_names = [item.element_name for item in gap_report.items]
        assert "geolocation" in gap_names
        assert "deforestation_free_evidence" in gap_names

    def test_generate_gap_report_all_missing(self, engine):
        gap_report = engine.generate_gap_report({})
        assert gap_report.total_gaps == 10
        assert gap_report.critical_gaps >= 3  # Several critical elements

    def test_gap_report_severity_downgrade_for_partial(self, engine):
        statuses = {e.value: ElementStatus.COMPLETE for e in Article9ElementName}
        statuses["geolocation"] = ElementStatus.PARTIAL  # critical -> high when partial
        elements = _make_elements(statuses)
        gap_report = engine.generate_gap_report(elements)
        geo_gap = [i for i in gap_report.items if i.element_name == "geolocation"]
        assert len(geo_gap) == 1
        assert geo_gap[0].severity == "high"  # Downgraded from critical


class TestRemediationActions:
    """Test remediation action lookup."""

    def test_remediation_actions_missing(self, engine):
        for elem in Article9ElementName:
            action = engine.get_remediation_action(elem.value, ElementStatus.MISSING)
            assert len(action) > 0

    def test_remediation_actions_partial(self, engine):
        for elem in Article9ElementName:
            action = engine.get_remediation_action(elem.value, ElementStatus.PARTIAL)
            assert len(action) > 0

    def test_remediation_actions_complete(self, engine):
        action = engine.get_remediation_action("geolocation", ElementStatus.COMPLETE)
        assert action == ""


class TestValidationStats:
    """Test validation statistics."""

    def test_validation_stats(self, engine, sample_article9_elements):
        engine.validate_completeness("OP-S1", EUDRCommodity.COFFEE, sample_article9_elements)
        engine.validate_completeness("OP-S2", EUDRCommodity.COCOA, {})
        stats = engine.get_validation_stats()
        assert stats["total_validations"] == 2
        assert "complete" in stats["classification_breakdown"]
        assert "insufficient" in stats["classification_breakdown"]
        assert stats["average_score"] > 0

    def test_clear_history(self, engine, sample_article9_elements):
        engine.validate_completeness("OP-C1", EUDRCommodity.COFFEE, sample_article9_elements)
        engine.clear_history()
        stats = engine.get_validation_stats()
        assert stats["total_validations"] == 0
