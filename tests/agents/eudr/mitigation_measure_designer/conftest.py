# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-029 Mitigation Measure Designer tests.

Provides reusable test fixtures for config, models, strategy, and
workflow objects across all test modules.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List

from greenlang.agents.eudr.mitigation_measure_designer.config import (
    MitigationMeasureDesignerConfig,
    reset_config,
)
from greenlang.agents.eudr.mitigation_measure_designer.models import (
    Article11Category,
    EUDRCommodity,
    EvidenceType,
    MeasureEvidence,
    MeasurePriority,
    MeasureStatus,
    MeasureSummary,
    MeasureTemplate,
    MitigationMeasure,
    MitigationReport,
    MitigationStrategy,
    RiskDimension,
    RiskLevel,
    RiskTrigger,
    VerificationReport,
    VerificationResult,
    WorkflowState,
    WorkflowStatus,
)
from greenlang.agents.eudr.mitigation_measure_designer.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


# ---------------------------------------------------------------------------
# Auto-reset config singleton after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before/after each test."""
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> MitigationMeasureDesignerConfig:
    """Create a default MitigationMeasureDesignerConfig instance."""
    return MitigationMeasureDesignerConfig()


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_risk_trigger() -> RiskTrigger:
    """Create a sample RiskTrigger with HIGH risk and multiple elevated dimensions."""
    return RiskTrigger(
        assessment_id="assess-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.COFFEE,
        composite_score=Decimal("72"),
        risk_level=RiskLevel.HIGH,
        risk_dimensions={
            RiskDimension.COUNTRY: Decimal("65"),
            RiskDimension.SUPPLIER: Decimal("80"),
            RiskDimension.DEFORESTATION: Decimal("55"),
            RiskDimension.COMMODITY: Decimal("20"),
            RiskDimension.CORRUPTION: Decimal("45"),
        },
    )


@pytest.fixture
def sample_template() -> MeasureTemplate:
    """Create a sample MeasureTemplate for testing."""
    return MeasureTemplate(
        template_id="MMD-TPL-TEST-001",
        title="Test Supplier Verification",
        description="Verify supplier compliance with EUDR requirements.",
        article11_category=Article11Category.INDEPENDENT_AUDIT,
        applicable_dimensions=[RiskDimension.SUPPLIER, RiskDimension.COUNTRY],
        applicable_commodities=[],
        base_effectiveness=Decimal("25"),
        typical_timeline_days=30,
        evidence_requirements=["audit_report", "document"],
        regulatory_reference="EUDR Art. 11(2)(a)",
    )


@pytest.fixture
def sample_template_country() -> MeasureTemplate:
    """Create a sample MeasureTemplate targeting country dimension."""
    return MeasureTemplate(
        template_id="MMD-TPL-TEST-002",
        title="Country Governance Assessment",
        description="Assess country governance framework for EUDR compliance.",
        article11_category=Article11Category.ADDITIONAL_INFO,
        applicable_dimensions=[RiskDimension.COUNTRY],
        applicable_commodities=[],
        base_effectiveness=Decimal("20"),
        typical_timeline_days=21,
        evidence_requirements=["document"],
        regulatory_reference="EUDR Art. 11(2)(c)",
    )


@pytest.fixture
def sample_template_deforestation() -> MeasureTemplate:
    """Create a sample MeasureTemplate targeting deforestation dimension."""
    return MeasureTemplate(
        template_id="MMD-TPL-TEST-003",
        title="Enhanced Satellite Monitoring",
        description="Deploy satellite monitoring on supplier plots.",
        article11_category=Article11Category.OTHER_MEASURES,
        applicable_dimensions=[RiskDimension.DEFORESTATION],
        applicable_commodities=[],
        base_effectiveness=Decimal("28"),
        typical_timeline_days=30,
        evidence_requirements=["satellite_image", "document"],
        regulatory_reference="EUDR Art. 11(2)(c)",
    )


@pytest.fixture
def sample_measure() -> MitigationMeasure:
    """Create a sample MitigationMeasure in PROPOSED status."""
    return MitigationMeasure(
        measure_id="msr-test-001",
        strategy_id="stg-test-001",
        template_id="MMD-TPL-TEST-001",
        title="Test Supplier Verification",
        description="Verify supplier compliance with EUDR requirements.",
        article11_category=Article11Category.INDEPENDENT_AUDIT,
        target_dimension=RiskDimension.SUPPLIER,
        status=MeasureStatus.PROPOSED,
        priority=MeasurePriority.HIGH,
        expected_risk_reduction=Decimal("25"),
    )


@pytest.fixture
def sample_strategy(
    sample_risk_trigger: RiskTrigger,
    sample_measure: MitigationMeasure,
) -> MitigationStrategy:
    """Create a sample MitigationStrategy with one measure."""
    return MitigationStrategy(
        strategy_id="stg-test-001",
        workflow_id="wfl-test-001",
        risk_trigger=sample_risk_trigger,
        measures=[sample_measure],
        pre_mitigation_score=Decimal("72"),
        target_score=Decimal("30"),
        status=WorkflowStatus.STRATEGY_DESIGNED,
        provenance_hash="abc123def456",
    )


@pytest.fixture
def sample_workflow() -> WorkflowState:
    """Create a sample WorkflowState in INITIATED status."""
    return WorkflowState(
        workflow_id="wfl-test-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.COFFEE,
        status=WorkflowStatus.INITIATED,
    )


@pytest.fixture
def sample_verification_report() -> VerificationReport:
    """Create a sample VerificationReport with SUFFICIENT result."""
    return VerificationReport(
        verification_id="ver-test-001",
        strategy_id="stg-test-001",
        pre_score=Decimal("72"),
        post_score=Decimal("25"),
        risk_reduction=Decimal("47"),
        result=VerificationResult.SUFFICIENT,
        verified_by="AGENT-EUDR-029",
    )


@pytest.fixture
def sample_evidence() -> MeasureEvidence:
    """Create a sample MeasureEvidence item."""
    return MeasureEvidence(
        evidence_id="evd-test-001",
        measure_id="msr-test-001",
        evidence_type=EvidenceType.AUDIT_REPORT,
        title="Annual Supplier Audit Report",
        file_reference="s3://evidence/audit_2026.pdf",
        uploaded_by="auditor@greenlang.com",
    )


@pytest.fixture
def multiple_templates(
    sample_template: MeasureTemplate,
    sample_template_country: MeasureTemplate,
    sample_template_deforestation: MeasureTemplate,
) -> List[MeasureTemplate]:
    """Provide a list of multiple templates for testing."""
    return [sample_template, sample_template_country, sample_template_deforestation]
