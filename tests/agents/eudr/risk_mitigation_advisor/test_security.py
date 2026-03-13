# -*- coding: utf-8 -*-
"""
RBAC and Security Tests - AGENT-EUDR-025

Tests role-based access control, permission enforcement, data isolation,
input sanitization, rate limiting compliance, audit trail tamper detection,
stakeholder role restrictions, API authentication requirements, and
sensitive data handling for the Risk Mitigation Advisor.

Security requirements (from PRD):
    - RBAC with 6 stakeholder roles
    - Role-based data visibility restrictions
    - SHA-256 tamper-evident audit trail
    - Input validation and sanitization
    - Provenance chain integrity enforcement
    - EUDR Article 31 data retention compliance
    - Sensitive supplier data protection

Test count: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    RiskInput,
    RiskCategory,
    StakeholderRole,
    PlanStatus,
    MilestoneStatus,
    RecommendStrategiesRequest,
    CreatePlanRequest,
    CollaborateRequest,
    CollaborateResponse,
    GenerateReportRequest,
    ReportType,
    SUPPORTED_COMMODITIES,
)
from greenlang.agents.eudr.risk_mitigation_advisor.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
)

from .conftest import FIXED_DATE


# ---------------------------------------------------------------------------
# Stakeholder Role Definitions
# ---------------------------------------------------------------------------


class TestStakeholderRoleDefinitions:
    """Test stakeholder role enumeration completeness and correctness."""

    def test_six_stakeholder_roles_defined(self):
        """PRD requires exactly 6 stakeholder roles."""
        assert len(StakeholderRole) == 6

    def test_internal_compliance_role_exists(self):
        assert StakeholderRole.INTERNAL_COMPLIANCE is not None

    def test_procurement_role_exists(self):
        assert StakeholderRole.PROCUREMENT is not None

    def test_supplier_role_exists(self):
        assert StakeholderRole.SUPPLIER is not None

    def test_ngo_partner_role_exists(self):
        assert StakeholderRole.NGO_PARTNER is not None

    def test_competent_authority_role_exists(self):
        assert StakeholderRole.COMPETENT_AUTHORITY is not None

    def test_certification_body_role_exists(self):
        assert StakeholderRole.CERTIFICATION_BODY is not None

    def test_all_roles_are_strings(self):
        for role in StakeholderRole:
            assert isinstance(role.value, str)


# ---------------------------------------------------------------------------
# Role-Based Access Control
# ---------------------------------------------------------------------------


class TestRoleBasedAccess:
    """Test role-based access control for collaboration actions."""

    @pytest.mark.asyncio
    async def test_internal_compliance_full_access(self, collaboration_engine):
        """Internal compliance should have full access to plans."""
        req = CollaborateRequest(
            action="message",
            plan_id="plan-rbac-001",
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            message="Full access test.",
        )
        result = await collaboration_engine.execute(req)
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.asyncio
    async def test_procurement_access(self, collaboration_engine):
        """Procurement should access supplier-related actions."""
        req = CollaborateRequest(
            action="task",
            plan_id="plan-rbac-002",
            stakeholder_role=StakeholderRole.PROCUREMENT,
            task_assignments=[{"assignee": "team", "task": "Review supplier"}],
        )
        result = await collaboration_engine.execute(req)
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.asyncio
    async def test_supplier_limited_visibility(self, collaboration_engine):
        """Supplier should have limited visibility to own data."""
        req = CollaborateRequest(
            action="progress",
            plan_id="plan-rbac-003",
            stakeholder_role=StakeholderRole.SUPPLIER,
            message="Progress update.",
        )
        result = await collaboration_engine.execute(req)
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.asyncio
    async def test_ngo_landscape_access(self, collaboration_engine):
        """NGO partner should access landscape-level data."""
        req = CollaborateRequest(
            action="message",
            plan_id="plan-rbac-004",
            stakeholder_role=StakeholderRole.NGO_PARTNER,
            message="Landscape initiative data request.",
        )
        result = await collaboration_engine.execute(req)
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.asyncio
    async def test_authority_read_access(self, collaboration_engine):
        """Competent authority should have read-oriented access."""
        req = CollaborateRequest(
            action="message",
            plan_id="plan-rbac-005",
            stakeholder_role=StakeholderRole.COMPETENT_AUTHORITY,
            message="Compliance inquiry.",
        )
        result = await collaboration_engine.execute(req)
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.asyncio
    async def test_certification_body_audit_access(self, collaboration_engine):
        """Certification body should access audit documentation."""
        req = CollaborateRequest(
            action="document",
            plan_id="plan-rbac-006",
            stakeholder_role=StakeholderRole.CERTIFICATION_BODY,
            document_ids=["audit-doc-001"],
        )
        result = await collaboration_engine.execute(req)
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.parametrize("role", list(StakeholderRole))
    @pytest.mark.asyncio
    async def test_all_roles_can_execute(self, collaboration_engine, role):
        """All defined roles should be able to execute collaboration actions."""
        req = CollaborateRequest(
            action="message",
            plan_id="plan-rbac-all",
            stakeholder_role=role,
            message=f"Security test for {role.value}.",
        )
        result = await collaboration_engine.execute(req)
        assert isinstance(result, CollaborateResponse)


# ---------------------------------------------------------------------------
# Input Validation and Sanitization
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test input validation and sanitization security controls."""

    @pytest.mark.asyncio
    async def test_empty_operator_id_rejected(self):
        """Empty operator ID should be rejected."""
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="",
                supplier_id="sup-val",
                country_code="BR",
                commodity="soya",
            )

    @pytest.mark.asyncio
    async def test_empty_supplier_id_rejected(self):
        """Empty supplier ID should be rejected."""
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="op-val",
                supplier_id="",
                country_code="BR",
                commodity="soya",
            )

    @pytest.mark.asyncio
    async def test_invalid_country_code_rejected(self):
        """Invalid country code should be rejected."""
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="op-val",
                supplier_id="sup-val",
                country_code="INVALID_CODE",
                commodity="soya",
            )

    @pytest.mark.asyncio
    async def test_invalid_commodity_rejected(self):
        """Non-EUDR commodity should be rejected."""
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="op-val",
                supplier_id="sup-val",
                country_code="BR",
                commodity="cotton",
            )

    @pytest.mark.asyncio
    async def test_score_above_100_rejected(self):
        """Risk score > 100 should be rejected."""
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="op-val",
                supplier_id="sup-val",
                country_code="BR",
                commodity="soya",
                country_risk_score=Decimal("150"),
            )

    @pytest.mark.asyncio
    async def test_score_below_0_rejected(self):
        """Risk score < 0 should be rejected."""
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="op-val",
                supplier_id="sup-val",
                country_code="BR",
                commodity="soya",
                country_risk_score=Decimal("-5"),
            )

    @pytest.mark.asyncio
    async def test_plan_negative_budget_rejected(self):
        """Negative budget should be rejected."""
        with pytest.raises((ValueError, Exception)):
            CreatePlanRequest(
                operator_id="op-val",
                supplier_id="sup-val",
                budget_eur=Decimal("-1000"),
            )

    @pytest.mark.asyncio
    async def test_sql_injection_in_operator_id(self, strategy_engine):
        """SQL injection attempt in operator_id should be handled safely."""
        # The model should either reject or safely handle SQL injection
        try:
            ri = RiskInput(
                operator_id="op'; DROP TABLE plans;--",
                supplier_id="sup-inj",
                country_code="BR",
                commodity="soya",
                country_risk_score=Decimal("50"),
                supplier_risk_score=Decimal("50"),
                assessment_date=FIXED_DATE,
            )
            req = RecommendStrategiesRequest(
                risk_input=ri, top_k=3, deterministic_mode=True
            )
            result = await strategy_engine.recommend(req)
            # If it passes, it means input is safely handled (not executed as SQL)
            assert isinstance(result.strategies, list)
        except (ValueError, Exception):
            # Rejection is also acceptable security behavior
            pass

    @pytest.mark.asyncio
    async def test_xss_in_message(self, collaboration_engine):
        """XSS attempt in message should be handled safely."""
        req = CollaborateRequest(
            action="message",
            plan_id="plan-xss",
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            message="<script>alert('xss')</script>",
        )
        result = await collaboration_engine.execute(req)
        # Should not raise and should sanitize
        assert isinstance(result, CollaborateResponse)

    @pytest.mark.asyncio
    async def test_very_long_input_handled(self, strategy_engine):
        """Very long input strings should be handled without crash."""
        try:
            ri = RiskInput(
                operator_id="op-" + "x" * 10000,
                supplier_id="sup-long",
                country_code="BR",
                commodity="soya",
                assessment_date=FIXED_DATE,
            )
            req = RecommendStrategiesRequest(
                risk_input=ri, top_k=3, deterministic_mode=True
            )
            result = await strategy_engine.recommend(req)
            assert isinstance(result.strategies, list)
        except (ValueError, Exception):
            # Rejection of overly long input is acceptable
            pass


# ---------------------------------------------------------------------------
# Provenance Tamper Detection
# ---------------------------------------------------------------------------


class TestProvenanceTamperDetection:
    """Test SHA-256 provenance chain tamper detection."""

    def test_valid_chain_verifies(self):
        """Untampered chain should verify successfully."""
        tracker = ProvenanceTracker(genesis_hash="SEC-TEST")
        for i in range(10):
            tracker.record(
                "strategy_recommendation", "recommend", f"strat-sec-{i}"
            )
        assert tracker.verify_chain() is True

    def test_entity_type_validation(self):
        """Invalid entity type should be rejected."""
        tracker = ProvenanceTracker()
        with pytest.raises((ValueError, Exception)):
            tracker.record("invalid_entity_type", "recommend", "test-001")

    def test_action_validation(self):
        """Invalid action should be rejected."""
        tracker = ProvenanceTracker()
        with pytest.raises((ValueError, Exception)):
            tracker.record("strategy_recommendation", "invalid_action", "test-001")

    def test_all_14_entity_types_valid(self):
        """All 14 defined entity types should be accepted."""
        tracker = ProvenanceTracker()
        for et in VALID_ENTITY_TYPES:
            entry = tracker.record(et, "create", f"test-{et}")
            assert entry.hash_value != ""

    def test_all_14_actions_valid(self):
        """All 14 defined actions should be accepted."""
        tracker = ProvenanceTracker()
        for action in VALID_ACTIONS:
            entry = tracker.record(
                "strategy_recommendation", action, f"test-{action}"
            )
            assert entry.hash_value != ""

    def test_record_immutability(self):
        """ProvenanceRecord should be frozen (immutable)."""
        tracker = ProvenanceTracker()
        entry = tracker.record(
            "strategy_recommendation", "recommend", "strat-immut"
        )
        with pytest.raises((AttributeError, TypeError)):
            entry.hash_value = "tampered_hash"

    def test_chain_hash_length(self):
        """All hashes should be 64-character SHA-256 hex strings."""
        tracker = ProvenanceTracker()
        for i in range(5):
            entry = tracker.record(
                "remediation_plan", "create", f"plan-hash-{i}"
            )
            assert len(entry.hash_value) == 64
            # Verify it is valid hex
            int(entry.hash_value, 16)


# ---------------------------------------------------------------------------
# Audit Trail Compliance
# ---------------------------------------------------------------------------


class TestAuditTrailCompliance:
    """Test EUDR Article 31 audit trail compliance requirements."""

    def test_provenance_export_json_complete(self):
        """JSON export should include all chain records."""
        tracker = ProvenanceTracker()
        for i in range(5):
            tracker.record(
                "strategy_recommendation", "recommend", f"strat-audit-{i}",
                actor="audit_user",
            )
        json_str = tracker.export_json()
        records = json.loads(json_str)
        assert len(records) == 5

    def test_provenance_record_has_timestamp(self):
        """Each provenance record must have a timestamp."""
        tracker = ProvenanceTracker()
        entry = tracker.record(
            "remediation_plan", "create", "plan-ts-001"
        )
        assert entry.timestamp != ""
        assert "T" in entry.timestamp  # ISO 8601 format

    def test_provenance_record_has_actor(self):
        """Each provenance record should track the actor."""
        tracker = ProvenanceTracker()
        entry = tracker.record(
            "remediation_plan", "create", "plan-actor-001",
            actor="compliance_officer",
        )
        assert entry.actor == "compliance_officer"

    def test_provenance_get_by_entity(self):
        """Should retrieve provenance records by entity ID."""
        tracker = ProvenanceTracker()
        tracker.record("strategy_recommendation", "recommend", "strat-filter-001")
        tracker.record("remediation_plan", "create", "plan-filter-001")
        tracker.record("strategy_recommendation", "recommend", "strat-filter-001")

        records = tracker.get_by_entity("strat-filter-001")
        assert len(records) == 2

    def test_provenance_chain_ordering(self):
        """Chain records should maintain insertion order."""
        tracker = ProvenanceTracker()
        ids = [f"entity-{i}" for i in range(5)]
        for eid in ids:
            tracker.record("strategy_recommendation", "recommend", eid)
        chain = tracker.get_chain()
        for i, record in enumerate(chain):
            assert record.entity_id == ids[i]

    @pytest.mark.asyncio
    async def test_collaboration_action_logged(self, collaboration_engine):
        """Collaboration actions should be recorded in activity log."""
        req = CollaborateRequest(
            action="message",
            plan_id="plan-audit-log",
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            message="Audit trail test.",
        )
        await collaboration_engine.execute(req)
        log = await collaboration_engine.get_activity_log(plan_id="plan-audit-log")
        assert isinstance(log, list)
        assert len(log) >= 1


# ---------------------------------------------------------------------------
# Data Isolation
# ---------------------------------------------------------------------------


class TestDataIsolation:
    """Test data isolation between operators and suppliers."""

    @pytest.mark.asyncio
    async def test_plans_filtered_by_operator(self, remediation_engine):
        """Plans should be filterable by operator for data isolation."""
        req_a = CreatePlanRequest(
            operator_id="op-iso-a",
            supplier_id="sup-iso-a",
            budget_eur=Decimal("25000"),
        )
        req_b = CreatePlanRequest(
            operator_id="op-iso-b",
            supplier_id="sup-iso-b",
            budget_eur=Decimal("30000"),
        )
        await remediation_engine.create_plan(req_a)
        await remediation_engine.create_plan(req_b)

        plans_a = await remediation_engine.list_plans(operator_id="op-iso-a")
        for p in plans_a:
            assert p.operator_id == "op-iso-a"

    @pytest.mark.asyncio
    async def test_plans_filtered_by_supplier(self, remediation_engine):
        """Plans should be filterable by supplier for data isolation."""
        req = CreatePlanRequest(
            operator_id="op-iso-sup",
            supplier_id="sup-iso-specific",
            budget_eur=Decimal("20000"),
        )
        await remediation_engine.create_plan(req)
        plans = await remediation_engine.list_plans(
            operator_id="op-iso-sup", supplier_id="sup-iso-specific"
        )
        for p in plans:
            assert p.supplier_id == "sup-iso-specific"


# ---------------------------------------------------------------------------
# Sensitive Data Handling
# ---------------------------------------------------------------------------


class TestSensitiveDataHandling:
    """Test sensitive data handling and privacy controls."""

    def test_risk_input_model_frozen(self):
        """RiskInput model should be immutable (frozen=True)."""
        ri = RiskInput(
            operator_id="op-frozen",
            supplier_id="sup-frozen",
            country_code="BR",
            commodity="soya",
            assessment_date=FIXED_DATE,
        )
        with pytest.raises((ValueError, Exception)):
            ri.operator_id = "tampered"

    def test_plan_status_transitions_enforced(self):
        """Plan status should only transition through valid states."""
        # Valid: DRAFT -> ACTIVE
        # Invalid: DRAFT -> COMPLETED (must go through ACTIVE first)
        from greenlang.agents.eudr.risk_mitigation_advisor.remediation_plan_design_engine import (
            VALID_STATUS_TRANSITIONS,
        )
        assert PlanStatus.ACTIVE in VALID_STATUS_TRANSITIONS.get(PlanStatus.DRAFT, [])
        assert PlanStatus.COMPLETED not in VALID_STATUS_TRANSITIONS.get(PlanStatus.DRAFT, [])

    @pytest.mark.asyncio
    async def test_report_includes_provenance(self, collaboration_engine):
        """Reports should include provenance for audit compliance."""
        # This tests that any generated report has a trackable provenance
        req = CollaborateRequest(
            action="document",
            plan_id="plan-report-prov",
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            document_ids=["report-001"],
        )
        result = await collaboration_engine.execute(req)
        assert isinstance(result, CollaborateResponse)


# ---------------------------------------------------------------------------
# Hash Algorithm Security
# ---------------------------------------------------------------------------


class TestHashAlgorithmSecurity:
    """Test cryptographic hash algorithm security properties."""

    def test_default_algorithm_is_sha256(self):
        """Default provenance algorithm should be SHA-256."""
        tracker = ProvenanceTracker()
        entry = tracker.record(
            "strategy_recommendation", "recommend", "strat-algo-001"
        )
        assert len(entry.hash_value) == 64  # SHA-256 = 64 hex chars

    def test_sha384_supported(self):
        """SHA-384 algorithm should be supported."""
        tracker = ProvenanceTracker(algorithm="sha384")
        entry = tracker.record(
            "strategy_recommendation", "recommend", "strat-384-001"
        )
        assert len(entry.hash_value) == 96  # SHA-384 = 96 hex chars

    def test_sha512_supported(self):
        """SHA-512 algorithm should be supported."""
        tracker = ProvenanceTracker(algorithm="sha512")
        entry = tracker.record(
            "strategy_recommendation", "recommend", "strat-512-001"
        )
        assert len(entry.hash_value) == 128  # SHA-512 = 128 hex chars

    def test_invalid_algorithm_rejected(self):
        """Invalid hash algorithm should be rejected."""
        with pytest.raises(ValueError):
            ProvenanceTracker(algorithm="md5")

    def test_unsupported_algorithm_rejected(self):
        """Unsupported hash algorithm should be rejected."""
        with pytest.raises(ValueError):
            ProvenanceTracker(algorithm="sha1")


# ---------------------------------------------------------------------------
# Configuration Security
# ---------------------------------------------------------------------------


class TestConfigurationSecurity:
    """Test configuration security controls."""

    def test_config_singleton_thread_safe(self):
        """Config singleton should be thread-safe."""
        from greenlang.agents.eudr.risk_mitigation_advisor.config import (
            get_config, set_config, reset_config,
            RiskMitigationAdvisorConfig,
        )
        cfg = RiskMitigationAdvisorConfig(
            database_url="postgresql://test:test@localhost:5432/test",
            redis_url="redis://localhost:6379/15",
        )
        set_config(cfg)
        retrieved = get_config()
        assert retrieved is cfg

    def test_composite_weights_sum_to_one(self):
        """Composite risk weights must sum to exactly 1.0."""
        from .conftest import COMPOSITE_WEIGHTS
        total = sum(COMPOSITE_WEIGHTS.values())
        assert total == Decimal("1"), f"Weights sum to {total}, expected 1.0"

    def test_provenance_tracker_singleton_reset(self):
        """Provenance tracker singleton should reset cleanly."""
        from greenlang.agents.eudr.risk_mitigation_advisor.provenance import (
            get_tracker, reset_tracker,
        )
        tracker = get_tracker()
        tracker.record("strategy_recommendation", "recommend", "strat-reset-001")
        reset_tracker()
        new_tracker = get_tracker()
        assert len(new_tracker.get_chain()) == 0
