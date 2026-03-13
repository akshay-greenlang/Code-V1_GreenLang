# -*- coding: utf-8 -*-
"""
Unit tests for Engine 9: Due Diligence Package Generator -- AGENT-EUDR-026

Tests DDS-compatible evidence package compilation from 25 agent outputs,
9-section DDS content per Article 12(2), SHA-256 per-artifact/per-section/
package-level integrity hashing, deterministic reproducibility, executive
summary generation, composite risk profile integration, per-section
completeness scoring, multi-language support (en/fr/de/es/pt), multiple
output formats (JSON/PDF/HTML/ZIP), package versioning, and simplified
vs standard package differences.

Test count: 90+ tests
Author: GreenLang Platform Team
Date: March 2026
"""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.due_diligence_orchestrator.due_diligence_package_generator import (
    DueDiligencePackageGenerator,
    _DDS_SECTIONS,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    AGENT_NAMES,
    ALL_EUDR_AGENTS,
    PHASE_1_AGENTS,
    PHASE_2_AGENTS,
    AgentExecutionRecord,
    AgentExecutionStatus,
    CompositeRiskProfile,
    DDSField,
    DDSSection,
    DueDiligencePackage,
    DueDiligencePhase,
    EUDRCommodity,
    MitigationDecision,
    QualityGateEvaluation,
    QualityGateId,
    QualityGateResultEnum,
    RiskScoreContribution,
    WorkflowState,
    WorkflowStatus,
    WorkflowType,
    _new_uuid,
    _utcnow,
)
from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
    set_config,
    reset_config,
)

from tests.agents.eudr.due_diligence_orchestrator.conftest import (
    DEFAULT_RISK_WEIGHTS,
    _make_phase1_output,
    _make_phase2_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_full_workflow_state(
    *,
    workflow_id: str = "wf-pkg-001",
    commodity: EUDRCommodity = EUDRCommodity.COCOA,
    workflow_type: WorkflowType = WorkflowType.STANDARD,
    operator_name: str = "Test Operator GmbH",
    country_codes: Optional[List[str]] = None,
    include_phase1: bool = True,
    include_phase2: bool = True,
) -> WorkflowState:
    """Build a WorkflowState with completed agent executions."""
    state = WorkflowState(
        workflow_id=workflow_id,
        definition_id="def-standard-001",
        status=WorkflowStatus.COMPLETING,
        workflow_type=workflow_type,
        commodity=commodity,
        current_phase=DueDiligencePhase.PACKAGE_GENERATION,
        operator_id="op-001",
        operator_name=operator_name,
        product_ids=["prod-001", "prod-002"],
        shipment_ids=["ship-001"],
        country_codes=country_codes if country_codes is not None else ["GH", "CI"],
        started_at=_utcnow() - timedelta(minutes=10),
        total_duration_ms=Decimal("600000"),
        progress_pct=Decimal("95"),
    )
    if include_phase1:
        for agent_id in PHASE_1_AGENTS:
            state.agent_executions[agent_id] = AgentExecutionRecord(
                workflow_id=workflow_id,
                agent_id=agent_id,
                status=AgentExecutionStatus.COMPLETED,
                duration_ms=Decimal("3000"),
                output_ref=f"s3://bucket/{workflow_id}/{agent_id}.json",
                output_summary=_make_phase1_output(agent_id),
            )
    if include_phase2:
        for agent_id in PHASE_2_AGENTS:
            state.agent_executions[agent_id] = AgentExecutionRecord(
                workflow_id=workflow_id,
                agent_id=agent_id,
                status=AgentExecutionStatus.COMPLETED,
                duration_ms=Decimal("2000"),
                output_ref=f"s3://bucket/{workflow_id}/{agent_id}.json",
                output_summary=_make_phase2_output(agent_id),
            )
    return state


def _build_agent_outputs(include_phase1: bool = True, include_phase2: bool = True) -> Dict[str, Dict[str, Any]]:
    """Build per-agent output data dictionaries."""
    outputs: Dict[str, Dict[str, Any]] = {}
    if include_phase1:
        for agent_id in PHASE_1_AGENTS:
            outputs[agent_id] = _make_phase1_output(agent_id)
    if include_phase2:
        for agent_id in PHASE_2_AGENTS:
            outputs[agent_id] = _make_phase2_output(agent_id)
    return outputs


def _build_risk_profile(
    *,
    composite_score: Decimal = Decimal("35.50"),
    risk_level: str = "standard",
    highest_dims: Optional[List[str]] = None,
) -> CompositeRiskProfile:
    """Build a CompositeRiskProfile with all 10 contributions."""
    contributions = []
    scores = {
        "EUDR-016": Decimal("40"), "EUDR-017": Decimal("35"),
        "EUDR-018": Decimal("30"), "EUDR-019": Decimal("25"),
        "EUDR-020": Decimal("45"), "EUDR-021": Decimal("30"),
        "EUDR-022": Decimal("35"), "EUDR-023": Decimal("40"),
        "EUDR-024": Decimal("20"), "EUDR-025": Decimal("25"),
    }
    for agent_id, raw in scores.items():
        w = DEFAULT_RISK_WEIGHTS[agent_id]
        contributions.append(RiskScoreContribution(
            agent_id=agent_id,
            agent_name=AGENT_NAMES.get(agent_id, agent_id),
            raw_score=raw,
            weight=w,
            weighted_score=raw * w,
        ))
    return CompositeRiskProfile(
        workflow_id="wf-pkg-001",
        contributions=contributions,
        composite_score=composite_score,
        risk_level=risk_level,
        highest_risk_dimensions=highest_dims or ["Deforestation Alert System", "Country Risk Evaluator"],
        all_dimensions_scored=True,
        coverage_pct=Decimal("100"),
    )


def _build_mitigation_decision(
    *,
    required: bool = True,
    level: str = "standard",
    pre_score: Decimal = Decimal("35.50"),
    post_score: Optional[Decimal] = Decimal("12.00"),
    strategies: Optional[List[str]] = None,
    adequacy_verified: bool = True,
) -> MitigationDecision:
    """Build a MitigationDecision."""
    return MitigationDecision(
        workflow_id="wf-pkg-001",
        mitigation_required=required,
        mitigation_level=level,
        pre_mitigation_score=pre_score,
        post_mitigation_score=post_score,
        mitigation_strategies=strategies or [
            "Enhanced supplier auditing",
            "Alternative sourcing from low-risk regions",
            "Third-party verification engagement",
        ],
        adequacy_verified=adequacy_verified,
    )


def _build_quality_gate_results() -> Dict[str, QualityGateEvaluation]:
    """Build passing quality gate evaluations."""
    return {
        "QG-1": QualityGateEvaluation(
            workflow_id="wf-pkg-001",
            gate_id=QualityGateId.QG1,
            phase_from=DueDiligencePhase.INFORMATION_GATHERING,
            phase_to=DueDiligencePhase.RISK_ASSESSMENT,
            result=QualityGateResultEnum.PASSED,
            weighted_score=Decimal("0.95"),
            threshold=Decimal("0.90"),
        ),
        "QG-2": QualityGateEvaluation(
            workflow_id="wf-pkg-001",
            gate_id=QualityGateId.QG2,
            phase_from=DueDiligencePhase.RISK_ASSESSMENT,
            phase_to=DueDiligencePhase.RISK_MITIGATION,
            result=QualityGateResultEnum.PASSED,
            weighted_score=Decimal("0.97"),
            threshold=Decimal("0.95"),
        ),
        "QG-3": QualityGateEvaluation(
            workflow_id="wf-pkg-001",
            gate_id=QualityGateId.QG3,
            phase_from=DueDiligencePhase.RISK_MITIGATION,
            phase_to=DueDiligencePhase.PACKAGE_GENERATION,
            result=QualityGateResultEnum.PASSED,
            weighted_score=Decimal("0.88"),
            threshold=Decimal("0.85"),
        ),
    }


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPackageGeneratorInit:
    """Test DueDiligencePackageGenerator initialization."""

    def test_init_with_default_config(self, default_config):
        """Generator initializes with global default config."""
        gen = DueDiligencePackageGenerator()
        assert gen is not None
        assert gen._config is not None

    def test_init_with_explicit_config(self, default_config):
        """Generator initializes with explicitly provided config."""
        gen = DueDiligencePackageGenerator(config=default_config)
        assert gen._config is default_config

    def test_init_with_custom_config(self):
        """Generator accepts a fully custom configuration."""
        cfg = DueDiligenceOrchestratorConfig(
            dds_schema_version="2.0.0",
            retention_years=7,
        )
        set_config(cfg)
        gen = DueDiligencePackageGenerator(config=cfg)
        assert gen._config.dds_schema_version == "2.0.0"
        assert gen._config.retention_years == 7


class TestDDSSectionGeneration:
    """Test 9 DDS sections per Article 12(2)."""

    def test_generates_exactly_9_sections(self, package_generator):
        """Package always contains exactly 9 DDS sections."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        assert len(pkg.sections) == 9

    def test_section_numbers_are_1_through_9(self, package_generator):
        """Section numbers are sequential from 1 to 9."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        numbers = [s.section_number for s in pkg.sections]
        assert numbers == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    @pytest.mark.parametrize("section_num,expected_title", [
        (1, "Operator Identification"),
        (2, "Product Description"),
        (3, "Country of Production"),
        (4, "Geolocation Data"),
        (5, "Quantity and Volume"),
        (6, "Date of Production"),
        (7, "Deforestation-Free Status"),
        (8, "Legal Compliance"),
        (9, "Risk Assessment and Mitigation"),
    ])
    def test_section_titles_match_article_12(
        self, package_generator, section_num, expected_title,
    ):
        """Each section title matches Article 12(2) requirements."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        section = pkg.sections[section_num - 1]
        assert section.title == expected_title

    def test_section_1_operator_identification(self, package_generator):
        """Section 1 contains operator name from workflow state."""
        state = _build_full_workflow_state(operator_name="Acme Coffee GmbH")
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        s1 = pkg.sections[0]
        assert s1.section_number == 1
        operator_field = next(
            (f for f in s1.fields if f.field_name == "operator_name"), None,
        )
        assert operator_field is not None
        assert operator_field.value == "Acme Coffee GmbH"
        assert operator_field.validated is True

    def test_section_3_country_of_production(self, package_generator):
        """Section 3 includes country codes from workflow state."""
        state = _build_full_workflow_state(country_codes=["BR", "CO"])
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        s3 = pkg.sections[2]
        country_field = next(
            (f for f in s3.fields if f.field_name == "country_of_production"), None,
        )
        assert country_field is not None
        assert country_field.value == ["BR", "CO"]

    def test_section_7_deforestation_free_status(self, package_generator):
        """Section 7 has 4 required fields for deforestation-free status."""
        state = _build_full_workflow_state()
        outputs = _build_agent_outputs()
        # Add deforestation-specific fields from agent outputs
        outputs["EUDR-003"]["deforestation_free_determination"] = "deforestation_free"
        outputs["EUDR-004"]["forest_cover_analysis"] = "stable"
        outputs["EUDR-005"]["land_use_change_analysis"] = "no_change"
        outputs["EUDR-003"]["satellite_evidence"] = "verified"
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=outputs,
        )
        s7 = pkg.sections[6]
        assert s7.section_number == 7
        assert len(s7.fields) == 4
        for field in s7.fields:
            assert field.validated is True

    def test_section_8_legal_compliance(self, package_generator):
        """Section 8 legal compliance references EUDR-023 and EUDR-024."""
        state = _build_full_workflow_state()
        outputs = _build_agent_outputs()
        outputs["EUDR-023"]["legal_compliance_status"] = "compliant"
        outputs["EUDR-023"]["applicable_legislation"] = "EU 2023/1115"
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=outputs,
        )
        s8 = pkg.sections[7]
        assert s8.section_number == 8
        status_field = next(
            (f for f in s8.fields if f.field_name == "legal_compliance_status"), None,
        )
        assert status_field is not None
        assert status_field.value == "compliant"

    def test_section_9_risk_assessment_from_profile(self, package_generator):
        """Section 9 extracts composite risk score from risk profile."""
        state = _build_full_workflow_state()
        risk_profile = _build_risk_profile(
            composite_score=Decimal("42.50"),
            risk_level="standard",
        )
        pkg = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
            agent_outputs=_build_agent_outputs(),
        )
        s9 = pkg.sections[8]
        assert s9.section_number == 9
        score_field = next(
            (f for f in s9.fields if f.field_name == "composite_risk_score"), None,
        )
        assert score_field is not None
        assert score_field.value == "42.50"
        assert score_field.validated is True

    def test_section_9_risk_dimension_breakdown(self, package_generator):
        """Section 9 includes risk dimension breakdown from profile."""
        state = _build_full_workflow_state()
        risk_profile = _build_risk_profile()
        pkg = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
            agent_outputs=_build_agent_outputs(),
        )
        s9 = pkg.sections[8]
        breakdown_field = next(
            (f for f in s9.fields if f.field_name == "risk_dimension_breakdown"), None,
        )
        assert breakdown_field is not None
        assert isinstance(breakdown_field.value, dict)
        # All 10 risk agent names should be present as keys
        assert len(breakdown_field.value) == 10

    def test_section_9_mitigation_measures(self, package_generator):
        """Section 9 includes mitigation strategies from decision."""
        state = _build_full_workflow_state()
        mitigation = _build_mitigation_decision(
            strategies=["Strategy A", "Strategy B"],
        )
        pkg = package_generator.generate_package(
            workflow_state=state,
            mitigation_decision=mitigation,
            agent_outputs=_build_agent_outputs(),
        )
        s9 = pkg.sections[8]
        mit_field = next(
            (f for f in s9.fields if f.field_name == "mitigation_measures"), None,
        )
        assert mit_field is not None
        assert mit_field.value == ["Strategy A", "Strategy B"]

    def test_section_9_residual_risk_score(self, package_generator):
        """Section 9 includes residual risk score post-mitigation."""
        state = _build_full_workflow_state()
        mitigation = _build_mitigation_decision(post_score=Decimal("8.50"))
        pkg = package_generator.generate_package(
            workflow_state=state,
            mitigation_decision=mitigation,
            agent_outputs=_build_agent_outputs(),
        )
        s9 = pkg.sections[8]
        residual_field = next(
            (f for f in s9.fields if f.field_name == "residual_risk_score"), None,
        )
        assert residual_field is not None
        assert residual_field.value == "8.50"

    def test_section_has_description(self, package_generator):
        """Each section has a non-empty description."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        for section in pkg.sections:
            assert section.description is not None
            assert len(section.description) > 10

    def test_section_evidence_refs_populated(self, package_generator):
        """Sections include evidence references from completed agents."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        # Section 1 sources from EUDR-001 and EUDR-008, both have output_ref
        s1 = pkg.sections[0]
        assert len(s1.evidence_refs) >= 1

    def test_section_agent_output_summaries(self, package_generator):
        """Sections include output summaries for available agents."""
        state = _build_full_workflow_state()
        outputs = _build_agent_outputs()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=outputs,
        )
        # Section 1 sources: EUDR-001, EUDR-008
        s1 = pkg.sections[0]
        assert "EUDR-001" in s1.agent_outputs
        assert s1.agent_outputs["EUDR-001"]["output_available"] is True

    def test_unvalidated_fields_have_notes(self, package_generator):
        """Fields without values get validation_notes explaining absence."""
        state = _build_full_workflow_state()
        # Empty outputs: no agent data to populate fields
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs={},
        )
        # Section 2 product description fields should be unvalidated
        s2 = pkg.sections[1]
        unvalidated = [f for f in s2.fields if not f.validated]
        for field in unvalidated:
            assert field.validation_notes is not None
            assert "not available" in field.validation_notes.lower()

    def test_field_name_variation_lookup(self, package_generator):
        """Field extraction tries name variations (underscore, hyphen, lower)."""
        state = _build_full_workflow_state()
        outputs = {
            "EUDR-001": {"tradename": "Premium Cocoa Beans"},
        }
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=outputs,
        )
        s2 = pkg.sections[1]
        trade_field = next(
            (f for f in s2.fields if f.field_name == "trade_name"), None,
        )
        assert trade_field is not None
        assert trade_field.value == "Premium Cocoa Beans"


class TestExecutiveSummary:
    """Test executive summary generation."""

    def test_summary_includes_commodity(self, package_generator):
        """Executive summary mentions the commodity."""
        state = _build_full_workflow_state(commodity=EUDRCommodity.COFFEE)
        pkg = package_generator.generate_package(workflow_state=state)
        assert "coffee" in pkg.executive_summary.lower()

    def test_summary_includes_operator(self, package_generator):
        """Executive summary mentions the operator name."""
        state = _build_full_workflow_state(operator_name="BeanTraders AG")
        pkg = package_generator.generate_package(workflow_state=state)
        assert "BeanTraders AG" in pkg.executive_summary

    def test_summary_includes_countries(self, package_generator):
        """Executive summary mentions countries of production."""
        state = _build_full_workflow_state(country_codes=["BR", "CO", "ET"])
        pkg = package_generator.generate_package(workflow_state=state)
        assert "BR" in pkg.executive_summary
        assert "CO" in pkg.executive_summary

    def test_summary_includes_risk_level(self, package_generator):
        """Executive summary reports the risk level when profile provided."""
        state = _build_full_workflow_state()
        risk_profile = _build_risk_profile(
            composite_score=Decimal("42.50"),
            risk_level="standard",
        )
        pkg = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
        )
        assert "42.50" in pkg.executive_summary
        assert "standard" in pkg.executive_summary.lower()

    def test_summary_includes_highest_risk_dimensions(self, package_generator):
        """Executive summary lists top risk dimensions."""
        state = _build_full_workflow_state()
        risk_profile = _build_risk_profile(
            highest_dims=["Deforestation Alert System", "Country Risk Evaluator"],
        )
        pkg = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
        )
        assert "Deforestation Alert System" in pkg.executive_summary

    def test_summary_with_mitigation_required(self, package_generator):
        """Summary describes mitigation when required."""
        state = _build_full_workflow_state()
        mitigation = _build_mitigation_decision(
            required=True,
            level="enhanced",
            post_score=Decimal("10.00"),
            strategies=["Strategy 1", "Strategy 2", "Strategy 3"],
            adequacy_verified=True,
        )
        pkg = package_generator.generate_package(
            workflow_state=state,
            mitigation_decision=mitigation,
        )
        assert "enhanced" in pkg.executive_summary.lower()
        assert "3" in pkg.executive_summary
        assert "10.00" in pkg.executive_summary
        assert "verified" in pkg.executive_summary.lower()

    def test_summary_no_mitigation_required(self, package_generator):
        """Summary states no mitigation when not required."""
        state = _build_full_workflow_state()
        mitigation = _build_mitigation_decision(
            required=False,
            level="none",
            post_score=None,
            strategies=[],
        )
        pkg = package_generator.generate_package(
            workflow_state=state,
            mitigation_decision=mitigation,
        )
        assert "no risk mitigation" in pkg.executive_summary.lower()

    def test_summary_mentions_workflow_type(self, package_generator):
        """Executive summary mentions the workflow type."""
        state = _build_full_workflow_state(
            workflow_type=WorkflowType.STANDARD,
        )
        pkg = package_generator.generate_package(workflow_state=state)
        assert "standard" in pkg.executive_summary.lower()

    def test_summary_mentions_agent_execution_count(self, package_generator):
        """Executive summary reports how many agents were executed."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        # 25 agents completed (15 phase1 + 10 phase2)
        assert "25" in pkg.executive_summary


class TestCompositeRiskProfile:
    """Test composite risk profile integration in the package."""

    def test_package_includes_risk_profile(self, package_generator):
        """Package stores the risk profile when provided."""
        state = _build_full_workflow_state()
        risk_profile = _build_risk_profile()
        pkg = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
        )
        assert pkg.risk_profile is not None
        assert pkg.risk_profile.composite_score == Decimal("35.50")

    def test_package_risk_profile_none_when_not_provided(self, package_generator):
        """Package has None risk_profile when not provided."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.risk_profile is None

    def test_risk_profile_hash_in_artifact_hashes(self, package_generator):
        """Artifact hashes include risk_profile hash when provided."""
        state = _build_full_workflow_state()
        risk_profile = _build_risk_profile()
        pkg = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
            agent_outputs=_build_agent_outputs(),
        )
        assert "risk_profile" in pkg.artifact_hashes
        assert len(pkg.artifact_hashes["risk_profile"]) == 64

    def test_no_risk_profile_hash_when_absent(self, package_generator):
        """No risk_profile key in artifact_hashes when profile not provided."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        assert "risk_profile" not in pkg.artifact_hashes

    def test_risk_level_classification_in_section_9(self, package_generator):
        """Section 9 includes the risk level classification string."""
        state = _build_full_workflow_state()
        risk_profile = _build_risk_profile(risk_level="high")
        pkg = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
        )
        s9 = pkg.sections[8]
        level_field = next(
            (f for f in s9.fields if f.field_name == "risk_level_classification"), None,
        )
        assert level_field is not None
        assert level_field.value == "high"

    def test_all_10_dimensions_in_breakdown(self, package_generator):
        """Risk dimension breakdown in section 9 covers all 10 risk agents."""
        state = _build_full_workflow_state()
        risk_profile = _build_risk_profile()
        pkg = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
        )
        s9 = pkg.sections[8]
        breakdown_field = next(
            (f for f in s9.fields if f.field_name == "risk_dimension_breakdown"), None,
        )
        assert breakdown_field is not None
        assert len(breakdown_field.value) == 10


class TestCompletenessScoring:
    """Test per-section completeness percentage scoring."""

    def test_full_completeness_with_all_outputs(self, package_generator):
        """Section with all fields populated has 100% completeness."""
        state = _build_full_workflow_state()
        # Populate all Section 1 fields
        outputs = _build_agent_outputs()
        outputs["EUDR-001"]["postal_address"] = "123 Main St, Hamburg"
        outputs["EUDR-001"]["email_address"] = "test@example.com"
        outputs["EUDR-008"]["eori_number"] = "DE1234567890"
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=outputs,
        )
        s1 = pkg.sections[0]
        assert s1.completeness_pct == Decimal("100.00")

    def test_zero_completeness_with_empty_outputs(self, package_generator):
        """Sections with no agent output have low completeness."""
        state = _build_full_workflow_state()
        # Remove operator_name to test: but operator_name comes from workflow state
        state_no_op = _build_full_workflow_state(operator_name=None)
        pkg = package_generator.generate_package(
            workflow_state=state_no_op,
            agent_outputs={},
        )
        # Section 2 (product description) should be 0% with no outputs
        s2 = pkg.sections[1]
        assert s2.completeness_pct == Decimal("0.00")

    def test_partial_completeness(self, package_generator):
        """Section with some fields populated has partial completeness."""
        state = _build_full_workflow_state()
        outputs = {
            "EUDR-001": {"product_description": "Cocoa Beans"},
        }
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=outputs,
        )
        s2 = pkg.sections[1]
        # product_description populated (1 of 4 fields)
        assert Decimal("0") < s2.completeness_pct < Decimal("100")

    def test_compute_package_completeness_average(self, package_generator):
        """Overall package completeness is the average of section scores."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        overall = package_generator.compute_package_completeness(pkg)
        assert isinstance(overall, Decimal)
        assert Decimal("0") <= overall <= Decimal("100")

    def test_compute_package_completeness_empty_sections(self, package_generator):
        """Package with no sections returns 0 completeness."""
        pkg = DueDiligencePackage(
            workflow_id="wf-empty",
            sections=[],
        )
        assert package_generator.compute_package_completeness(pkg) == Decimal("0")

    def test_get_incomplete_sections_returns_below_threshold(self, package_generator):
        """get_incomplete_sections returns sections below threshold."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs={},
        )
        incomplete = package_generator.get_incomplete_sections(
            pkg, threshold=Decimal("50"),
        )
        # With no agent outputs, most sections should be incomplete
        assert len(incomplete) > 0
        for item in incomplete:
            assert Decimal(item["completeness_pct"]) < Decimal("50")

    def test_get_incomplete_sections_includes_missing_fields(self, package_generator):
        """Incomplete section summaries list missing field names."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs={},
        )
        incomplete = package_generator.get_incomplete_sections(pkg)
        for item in incomplete:
            assert "missing_fields" in item
            assert isinstance(item["missing_fields"], list)

    def test_completeness_precision_two_decimal_places(self, package_generator):
        """Completeness percentages are quantized to 2 decimal places."""
        state = _build_full_workflow_state()
        outputs = {
            "EUDR-001": {"product_description": "Cocoa"},
        }
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=outputs,
        )
        for section in pkg.sections:
            # Check precision: value has at most 2 decimal places
            as_str = str(section.completeness_pct)
            if "." in as_str:
                decimal_part = as_str.split(".")[1]
                assert len(decimal_part) <= 2


class TestSHA256Hashing:
    """Test SHA-256 integrity hashing at artifact, section, and package levels."""

    def test_package_integrity_hash_is_64_hex_chars(self, package_generator):
        """Package integrity hash is a 64-character hex SHA-256."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        assert pkg.integrity_hash is not None
        assert len(pkg.integrity_hash) == 64
        assert all(c in "0123456789abcdef" for c in pkg.integrity_hash)

    def test_provenance_hash_equals_integrity_hash(self, package_generator):
        """Package provenance_hash is set equal to integrity_hash."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        assert pkg.provenance_hash == pkg.integrity_hash

    def test_per_section_hashes_present(self, package_generator):
        """Artifact hashes include one hash per DDS section."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        for i in range(1, 10):
            key = f"section_{i}"
            assert key in pkg.artifact_hashes
            assert len(pkg.artifact_hashes[key]) == 64

    def test_per_agent_output_hashes(self, package_generator):
        """Artifact hashes include one hash per agent output provided."""
        state = _build_full_workflow_state()
        outputs = _build_agent_outputs()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=outputs,
        )
        for agent_id in outputs:
            key = f"agent_output_{agent_id}"
            assert key in pkg.artifact_hashes
            assert len(pkg.artifact_hashes[key]) == 64

    def test_mitigation_decision_hash(self, package_generator):
        """Artifact hashes include mitigation_decision hash."""
        state = _build_full_workflow_state()
        mitigation = _build_mitigation_decision()
        pkg = package_generator.generate_package(
            workflow_state=state,
            mitigation_decision=mitigation,
            agent_outputs=_build_agent_outputs(),
        )
        assert "mitigation_decision" in pkg.artifact_hashes
        assert len(pkg.artifact_hashes["mitigation_decision"]) == 64

    def test_no_mitigation_hash_when_absent(self, package_generator):
        """No mitigation_decision key in hashes when not provided."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        assert "mitigation_decision" not in pkg.artifact_hashes

    def test_agent_output_hashes_sorted_by_agent_id(self, package_generator):
        """Agent output hashes are computed from sorted agent IDs."""
        state = _build_full_workflow_state()
        outputs = _build_agent_outputs()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=outputs,
        )
        agent_hash_keys = sorted(
            k for k in pkg.artifact_hashes if k.startswith("agent_output_")
        )
        expected_keys = sorted(f"agent_output_{aid}" for aid in outputs)
        assert agent_hash_keys == expected_keys

    def test_sha256_uses_canonical_json(self, package_generator):
        """Internal _sha256 uses canonical JSON (sorted keys, compact separators)."""
        data = {"b": 2, "a": 1}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        actual_hash = package_generator._sha256(data)
        assert actual_hash == expected_hash

    def test_sha256_deterministic_for_same_input(self, package_generator):
        """Same data always produces same hash."""
        data = {"key": "value", "num": 42}
        h1 = package_generator._sha256(data)
        h2 = package_generator._sha256(data)
        assert h1 == h2

    def test_sha256_different_for_different_input(self, package_generator):
        """Different data produces different hashes."""
        h1 = package_generator._sha256({"a": 1})
        h2 = package_generator._sha256({"a": 2})
        assert h1 != h2


class TestMultiLanguage:
    """Test multi-language support for package generation."""

    @pytest.mark.parametrize("language", ["en", "fr", "de", "es", "pt"])
    def test_all_supported_languages(self, package_generator, language):
        """Package generation succeeds for all 5 supported languages."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            language=language,
        )
        assert pkg.language == language

    def test_default_language_is_english(self, package_generator):
        """Default language is 'en' when not specified."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.language == "en"

    def test_language_stored_in_package(self, package_generator):
        """Selected language is stored in the package model."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            language="de",
        )
        assert pkg.language == "de"

    def test_different_language_same_sections(self, package_generator):
        """Different languages produce same number of sections."""
        state = _build_full_workflow_state()
        pkg_en = package_generator.generate_package(
            workflow_state=state,
            language="en",
        )
        pkg_fr = package_generator.generate_package(
            workflow_state=state,
            language="fr",
        )
        assert len(pkg_en.sections) == len(pkg_fr.sections)

    def test_language_does_not_affect_hash_determinism(self, package_generator):
        """Same language produces same package hash for same data."""
        state = _build_full_workflow_state()
        outputs = _build_agent_outputs()
        # Note: different language may yield different hash because language
        # is part of the package, but same language must be deterministic
        with patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.models._new_uuid",
            return_value="fixed-uuid",
        ):
            pkg1 = package_generator.generate_package(
                workflow_state=state,
                agent_outputs=outputs,
                language="fr",
            )
            pkg2 = package_generator.generate_package(
                workflow_state=state,
                agent_outputs=outputs,
                language="fr",
            )
        # Sections and artifacts are the same data, so artifact hashes match
        for key in pkg1.artifact_hashes:
            if key.startswith("section_") or key.startswith("agent_output_"):
                assert pkg1.artifact_hashes[key] == pkg2.artifact_hashes[key]


class TestPackageFormats:
    """Test package output format support."""

    def test_json_format_is_default(self, package_generator):
        """Package generates successfully for JSON use case."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        # Package is a Pydantic model, JSON-serializable
        data = pkg.model_dump()
        assert "sections" in data
        assert "integrity_hash" in data

    def test_package_serializes_to_valid_json(self, package_generator):
        """Package can be round-tripped through JSON."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
            risk_profile=_build_risk_profile(),
        )
        json_str = pkg.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["workflow_id"] == state.workflow_id
        assert len(parsed["sections"]) == 9

    def test_package_sections_serializable(self, package_generator):
        """Each section with fields is JSON-serializable."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        for section in pkg.sections:
            json_str = section.model_dump_json()
            parsed = json.loads(json_str)
            assert "section_number" in parsed
            assert "fields" in parsed

    def test_package_contains_download_urls_field(self, package_generator):
        """Package model has download_urls attribute (empty by default)."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        data = pkg.model_dump()
        assert "download_urls" in data

    def test_workflow_metadata_in_package(self, package_generator):
        """Package contains workflow_metadata dict."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert isinstance(pkg.workflow_metadata, dict)
        assert "workflow_id" in pkg.workflow_metadata
        assert "workflow_type" in pkg.workflow_metadata
        assert pkg.workflow_metadata["dds_schema_version"] is not None


class TestDeterminism:
    """Test deterministic package generation."""

    def test_same_inputs_same_artifact_hashes(self, package_generator):
        """Identical inputs produce identical artifact hashes."""
        state = _build_full_workflow_state()
        outputs = _build_agent_outputs()
        risk_profile = _build_risk_profile()
        mitigation = _build_mitigation_decision()

        pkg1 = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
            mitigation_decision=mitigation,
            agent_outputs=outputs,
        )
        pkg2 = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
            mitigation_decision=mitigation,
            agent_outputs=outputs,
        )
        # All artifact hashes must be identical
        assert pkg1.artifact_hashes == pkg2.artifact_hashes

    def test_section_hashes_deterministic(self, package_generator):
        """Section hashes are bit-perfect reproducible."""
        state = _build_full_workflow_state()
        outputs = _build_agent_outputs()

        pkg1 = package_generator.generate_package(
            workflow_state=state, agent_outputs=outputs,
        )
        pkg2 = package_generator.generate_package(
            workflow_state=state, agent_outputs=outputs,
        )
        for i in range(1, 10):
            assert pkg1.artifact_hashes[f"section_{i}"] == pkg2.artifact_hashes[f"section_{i}"]

    def test_executive_summary_deterministic(self, package_generator):
        """Same inputs produce identical executive summary text."""
        state = _build_full_workflow_state()
        risk_profile = _build_risk_profile()
        mitigation = _build_mitigation_decision()

        pkg1 = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
            mitigation_decision=mitigation,
        )
        pkg2 = package_generator.generate_package(
            workflow_state=state,
            risk_profile=risk_profile,
            mitigation_decision=mitigation,
        )
        assert pkg1.executive_summary == pkg2.executive_summary

    def test_different_workflow_id_different_package_hash(self, package_generator):
        """Different workflow IDs produce different package-level hashes."""
        state1 = _build_full_workflow_state(workflow_id="wf-aaa")
        state2 = _build_full_workflow_state(workflow_id="wf-bbb")
        outputs = _build_agent_outputs()

        pkg1 = package_generator.generate_package(
            workflow_state=state1, agent_outputs=outputs,
        )
        pkg2 = package_generator.generate_package(
            workflow_state=state2, agent_outputs=outputs,
        )
        # Package hash covers workflow_id, so must differ
        assert pkg1.integrity_hash != pkg2.integrity_hash

    def test_different_commodity_different_summary(self, package_generator):
        """Different commodities produce different executive summaries."""
        state1 = _build_full_workflow_state(commodity=EUDRCommodity.COCOA)
        state2 = _build_full_workflow_state(commodity=EUDRCommodity.SOYA)

        pkg1 = package_generator.generate_package(workflow_state=state1)
        pkg2 = package_generator.generate_package(workflow_state=state2)
        assert pkg1.executive_summary != pkg2.executive_summary


class TestPackageVersioning:
    """Test DDS schema versioning in packages."""

    def test_default_dds_schema_version(self, package_generator):
        """Package uses DDS schema version from config."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.dds_schema_version is not None
        assert len(pkg.dds_schema_version) > 0

    def test_custom_dds_schema_version(self):
        """Custom schema version propagates to package."""
        cfg = DueDiligenceOrchestratorConfig(dds_schema_version="2.1.0")
        set_config(cfg)
        gen = DueDiligencePackageGenerator(config=cfg)
        state = _build_full_workflow_state()
        pkg = gen.generate_package(workflow_state=state)
        assert pkg.dds_schema_version == "2.1.0"

    def test_schema_version_in_workflow_metadata(self, package_generator):
        """DDS schema version appears in workflow_metadata."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert "dds_schema_version" in pkg.workflow_metadata

    def test_retention_years_in_workflow_metadata(self, package_generator):
        """Retention years appears in workflow_metadata."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert "retention_years" in pkg.workflow_metadata
        assert pkg.workflow_metadata["retention_years"] == package_generator._config.retention_years


class TestSimplifiedPackage:
    """Test simplified due diligence package differences."""

    def test_simplified_workflow_produces_9_sections(self, package_generator):
        """Even simplified workflows produce all 9 DDS sections."""
        state = _build_full_workflow_state(
            workflow_type=WorkflowType.SIMPLIFIED,
            commodity=EUDRCommodity.WOOD,
        )
        pkg = package_generator.generate_package(workflow_state=state)
        assert len(pkg.sections) == 9

    def test_simplified_has_lower_completeness(self, package_generator):
        """Simplified workflow typically has lower completeness than standard."""
        # Standard with full outputs
        std_state = _build_full_workflow_state(workflow_type=WorkflowType.STANDARD)
        std_pkg = package_generator.generate_package(
            workflow_state=std_state,
            agent_outputs=_build_agent_outputs(),
        )
        # Simplified with only a few agents' outputs
        simp_state = _build_full_workflow_state(
            workflow_type=WorkflowType.SIMPLIFIED,
            include_phase2=False,
        )
        simp_outputs = {
            "EUDR-001": _make_phase1_output("EUDR-001"),
            "EUDR-002": _make_phase1_output("EUDR-002"),
            "EUDR-007": _make_phase1_output("EUDR-007"),
        }
        simp_pkg = package_generator.generate_package(
            workflow_state=simp_state,
            agent_outputs=simp_outputs,
        )
        std_completeness = package_generator.compute_package_completeness(std_pkg)
        simp_completeness = package_generator.compute_package_completeness(simp_pkg)
        assert simp_completeness <= std_completeness

    def test_simplified_workflow_type_in_package(self, package_generator):
        """Simplified workflow type is recorded in the package."""
        state = _build_full_workflow_state(workflow_type=WorkflowType.SIMPLIFIED)
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.workflow_type == WorkflowType.SIMPLIFIED

    def test_simplified_summary_mentions_simplified(self, package_generator):
        """Executive summary mentions 'simplified' workflow type."""
        state = _build_full_workflow_state(workflow_type=WorkflowType.SIMPLIFIED)
        pkg = package_generator.generate_package(workflow_state=state)
        assert "simplified" in pkg.executive_summary.lower()

    def test_simplified_fewer_executed_agents(self, package_generator):
        """Simplified package reports fewer executed agents."""
        state = _build_full_workflow_state(
            workflow_type=WorkflowType.SIMPLIFIED,
            include_phase2=False,
        )
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.total_agents_executed < 25

    def test_simplified_metadata_workflow_type(self, package_generator):
        """Workflow metadata records simplified type."""
        state = _build_full_workflow_state(workflow_type=WorkflowType.SIMPLIFIED)
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.workflow_metadata["workflow_type"] == "simplified"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_agent_outputs(self, package_generator):
        """Package generation works with no agent outputs."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs={},
        )
        assert len(pkg.sections) == 9
        assert pkg.integrity_hash is not None

    def test_none_agent_outputs(self, package_generator):
        """Package generation works when agent_outputs is None."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=None,
        )
        assert len(pkg.sections) == 9

    def test_no_agents_completed(self, package_generator):
        """Package generation works when no agents completed."""
        state = WorkflowState(
            workflow_id="wf-empty",
            definition_id="def-001",
            status=WorkflowStatus.COMPLETING,
            workflow_type=WorkflowType.STANDARD,
            commodity=EUDRCommodity.COCOA,
            current_phase=DueDiligencePhase.PACKAGE_GENERATION,
        )
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.total_agents_executed == 0
        assert len(pkg.sections) == 9

    def test_missing_operator_name(self, package_generator):
        """Package handles None operator name gracefully."""
        state = _build_full_workflow_state(operator_name=None)
        pkg = package_generator.generate_package(workflow_state=state)
        s1 = pkg.sections[0]
        operator_field = next(
            (f for f in s1.fields if f.field_name == "operator_name"), None,
        )
        assert operator_field is not None
        assert operator_field.validated is False

    def test_missing_country_codes(self, package_generator):
        """Package handles empty country codes list."""
        state = _build_full_workflow_state(country_codes=[])
        pkg = package_generator.generate_package(workflow_state=state)
        s3 = pkg.sections[2]
        country_field = next(
            (f for f in s3.fields if f.field_name == "country_of_production"), None,
        )
        # Empty list is falsy so field should not be validated
        assert country_field is not None
        assert country_field.validated is False

    def test_generated_by_propagated(self, package_generator):
        """generated_by parameter is stored in the package."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            generated_by="auditor@company.com",
        )
        assert pkg.generated_by == "auditor@company.com"

    def test_generated_at_is_utc(self, package_generator):
        """generated_at is a UTC timestamp."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.generated_at is not None
        assert pkg.generated_at.tzinfo is not None

    def test_quality_gate_results_in_package(self, package_generator):
        """Quality gate results are included in the package."""
        state = _build_full_workflow_state()
        qg_results = _build_quality_gate_results()
        pkg = package_generator.generate_package(
            workflow_state=state,
            quality_gate_results=qg_results,
        )
        assert len(pkg.quality_gate_results) == 3
        assert "QG-1" in pkg.quality_gate_results
        assert "QG-2" in pkg.quality_gate_results
        assert "QG-3" in pkg.quality_gate_results

    def test_empty_quality_gate_results(self, package_generator):
        """Package works with no quality gate results."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(
            workflow_state=state,
            quality_gate_results=None,
        )
        assert pkg.quality_gate_results == {}


class TestWorkflowMetadata:
    """Test workflow metadata inclusion in the package."""

    def test_metadata_has_workflow_id(self, package_generator):
        """Workflow metadata includes workflow_id."""
        state = _build_full_workflow_state(workflow_id="wf-meta-001")
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.workflow_metadata["workflow_id"] == "wf-meta-001"

    def test_metadata_has_definition_id(self, package_generator):
        """Workflow metadata includes definition_id."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.workflow_metadata["definition_id"] == "def-standard-001"

    def test_metadata_has_status(self, package_generator):
        """Workflow metadata includes current status."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.workflow_metadata["status"] == "completing"

    def test_metadata_has_current_phase(self, package_generator):
        """Workflow metadata includes current phase."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.workflow_metadata["current_phase"] == "package_generation"

    def test_metadata_has_operator_id(self, package_generator):
        """Workflow metadata includes operator_id."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.workflow_metadata["operator_id"] == "op-001"

    def test_metadata_has_product_and_shipment_ids(self, package_generator):
        """Workflow metadata includes product_ids and shipment_ids."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.workflow_metadata["product_ids"] == ["prod-001", "prod-002"]
        assert pkg.workflow_metadata["shipment_ids"] == ["ship-001"]

    def test_metadata_has_transition_and_checkpoint_counts(self, package_generator):
        """Workflow metadata includes transition and checkpoint counts."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert "transition_count" in pkg.workflow_metadata
        assert "checkpoint_count" in pkg.workflow_metadata

    def test_metadata_has_timestamps(self, package_generator):
        """Workflow metadata includes created_at and started_at."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert "created_at" in pkg.workflow_metadata
        assert "started_at" in pkg.workflow_metadata

    def test_metadata_total_duration(self, package_generator):
        """Workflow metadata includes total_duration_ms."""
        state = _build_full_workflow_state()
        pkg = package_generator.generate_package(workflow_state=state)
        assert pkg.workflow_metadata["total_duration_ms"] is not None


class TestAllCommodities:
    """Test package generation across all 7 EUDR commodities."""

    @pytest.mark.parametrize("commodity", list(EUDRCommodity))
    def test_package_generation_all_commodities(
        self, package_generator, commodity,
    ):
        """Package generation succeeds for all 7 EUDR commodities."""
        state = _build_full_workflow_state(commodity=commodity)
        pkg = package_generator.generate_package(
            workflow_state=state,
            agent_outputs=_build_agent_outputs(),
        )
        assert pkg.commodity == commodity
        assert len(pkg.sections) == 9
        assert pkg.integrity_hash is not None

    @pytest.mark.parametrize("commodity", list(EUDRCommodity))
    def test_executive_summary_mentions_commodity(
        self, package_generator, commodity,
    ):
        """Executive summary mentions each commodity by name."""
        state = _build_full_workflow_state(commodity=commodity)
        pkg = package_generator.generate_package(workflow_state=state)
        assert commodity.value in pkg.executive_summary.lower()


class TestDDSSectionDefinitions:
    """Test the static _DDS_SECTIONS definition."""

    def test_dds_sections_has_9_entries(self):
        """There are exactly 9 DDS section definitions."""
        assert len(_DDS_SECTIONS) == 9

    def test_dds_sections_sequential_numbers(self):
        """Section numbers are sequential 1 through 9."""
        numbers = [s["section_number"] for s in _DDS_SECTIONS]
        assert numbers == list(range(1, 10))

    def test_all_sections_have_fields(self):
        """Every section has at least one field definition."""
        for section_def in _DDS_SECTIONS:
            assert len(section_def["fields"]) >= 1, (
                f"Section {section_def['section_number']} has no fields"
            )

    def test_all_sections_have_source_agents(self):
        """Every section references at least one source agent."""
        for section_def in _DDS_SECTIONS:
            assert len(section_def["source_agents"]) >= 1

    def test_all_sections_have_article_ref(self):
        """Every section has an Article 12(2) reference."""
        for section_def in _DDS_SECTIONS:
            assert section_def["article_ref"].startswith("12(2)")

    def test_section_9_covers_all_risk_agents(self):
        """Section 9 references all 10 risk assessment agents (EUDR-016 to 025)."""
        section_9 = _DDS_SECTIONS[8]
        for i in range(16, 26):
            agent_id = f"EUDR-{i:03d}"
            assert agent_id in section_9["source_agents"], (
                f"{agent_id} missing from Section 9 source agents"
            )

    def test_required_fields_flagged_correctly(self):
        """All sections have at least one required field."""
        for section_def in _DDS_SECTIONS:
            required = [f for f in section_def["fields"] if f.get("required", True)]
            assert len(required) >= 1, (
                f"Section {section_def['section_number']} has no required fields"
            )
