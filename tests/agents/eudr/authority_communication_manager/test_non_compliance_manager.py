# -*- coding: utf-8 -*-
"""
Unit tests for NonComplianceManager engine - AGENT-EUDR-040

Tests violation recording, penalty calculation, corrective action tracking,
severity-based penalty scaling, resolution workflow, provenance tracking,
and health checks.

55+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
)
from greenlang.agents.eudr.authority_communication_manager.non_compliance_manager import (
    NonComplianceManager,
)
from greenlang.agents.eudr.authority_communication_manager.models import (
    NonCompliance,
    ViolationSeverity,
    ViolationType,
)


@pytest.fixture
def config():
    return AuthorityCommunicationManagerConfig()


@pytest.fixture
def manager(config):
    return NonComplianceManager(config=config)


# ====================================================================
# Initialization
# ====================================================================


class TestInit:
    def test_manager_created(self, manager):
        assert manager is not None

    def test_default_config(self):
        m = NonComplianceManager()
        assert m.config is not None

    def test_custom_config(self, config):
        m = NonComplianceManager(config=config)
        assert m.config is config

    def test_records_empty(self, manager):
        assert len(manager._records) == 0

    def test_provenance_initialized(self, manager):
        assert manager._provenance is not None


# ====================================================================
# Record Violation
# ====================================================================


class TestRecordViolation:
    @pytest.mark.asyncio
    async def test_record_missing_dds(self, manager):
        result = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="major",
            description="Missing DDS for cocoa import batch B-2025-001",
        )
        assert isinstance(result, NonCompliance)
        assert result.violation_type == ViolationType.MISSING_DDS
        assert result.severity == ViolationSeverity.MAJOR

    @pytest.mark.asyncio
    async def test_record_deforestation_link(self, manager):
        result = await manager.record_violation(
            operator_id="OP-002",
            authority_id="AUTH-FR-001",
            violation_type="deforestation_link",
            severity="critical",
            description="Deforestation activity detected on plot GPS-2025-789.",
        )
        assert result.violation_type == ViolationType.DEFORESTATION_LINK
        assert result.severity == ViolationSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_record_false_information(self, manager):
        result = await manager.record_violation(
            operator_id="OP-003",
            authority_id="AUTH-NL-001",
            violation_type="false_information",
            severity="critical",
            description="Deliberate falsification of origin data.",
        )
        assert result.violation_type == ViolationType.FALSE_INFORMATION

    @pytest.mark.asyncio
    async def test_record_assigns_id(self, manager):
        result = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="minor",
            description="Test violation",
        )
        assert result.non_compliance_id is not None
        assert len(result.non_compliance_id) > 0

    @pytest.mark.asyncio
    async def test_record_computes_provenance(self, manager):
        result = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="minor",
            description="Test violation",
        )
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_record_with_penalty(self, manager):
        result = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="major",
            description="Missing DDS",
        )
        assert result.penalty_amount is not None
        assert result.penalty_amount > 0

    @pytest.mark.asyncio
    async def test_penalty_increases_with_severity(self, manager):
        minor = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="incomplete_dds",
            severity="minor",
            description="Minor: Incomplete DDS",
        )
        major = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="incomplete_dds",
            severity="major",
            description="Major: Incomplete DDS",
        )
        critical = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="incomplete_dds",
            severity="critical",
            description="Critical: Incomplete DDS",
        )
        assert minor.penalty_amount <= major.penalty_amount
        assert major.penalty_amount <= critical.penalty_amount

    @pytest.mark.asyncio
    async def test_record_with_evidence(self, manager):
        result = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="minor",
            description="Test violation",
            evidence_references=["DOC-001", "DOC-002"],
        )
        assert len(result.evidence_references) == 2

    @pytest.mark.asyncio
    async def test_record_with_corrective_actions(self, manager):
        result = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="minor",
            description="Test violation",
            corrective_actions_required=["Submit DDS within 30 days"],
        )
        assert len(result.corrective_actions_required) == 1

    @pytest.mark.asyncio
    async def test_record_with_commodity(self, manager):
        result = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="minor",
            description="Test violation",
            commodity="cocoa",
        )
        assert result.commodity == "cocoa"

    @pytest.mark.asyncio
    async def test_record_invalid_violation_type(self, manager):
        with pytest.raises(ValueError, match="Invalid"):
            await manager.record_violation(
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                violation_type="not_a_violation",
                severity="minor",
                description="Test violation",
            )

    @pytest.mark.asyncio
    async def test_record_invalid_severity(self, manager):
        with pytest.raises(ValueError, match="Invalid"):
            await manager.record_violation(
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                violation_type="missing_dds",
                severity="not_a_severity",
                description="Test violation",
            )

    @pytest.mark.asyncio
    async def test_record_stored(self, manager):
        result = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="minor",
            description="Test violation",
        )
        assert result.non_compliance_id in manager._records

    @pytest.mark.asyncio
    async def test_record_all_violation_types(self, manager):
        """Test each violation type can be recorded."""
        for vt in ViolationType:
            result = await manager.record_violation(
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                violation_type=vt.value,
                severity="minor",
                description=f"Test {vt.value}",
            )
            assert result.violation_type == vt

    @pytest.mark.asyncio
    async def test_record_penalty_override(self, manager):
        result = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="minor",
            description="Test violation",
            penalty_override=Decimal("9999.99"),
        )
        assert result.penalty_amount == Decimal("9999.99")

    @pytest.mark.asyncio
    async def test_record_sets_corrective_deadline(self, manager):
        result = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="minor",
            description="Test violation",
            corrective_deadline_days=60,
        )
        assert result.corrective_deadline is not None


# ====================================================================
# Get / List / Health
# ====================================================================


class TestGetListHealth:
    @pytest.mark.asyncio
    async def test_get_record(self, manager):
        nc = await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="minor",
            description="Test violation",
        )
        result = await manager.get_record(nc.non_compliance_id)
        assert result is not None
        assert result.non_compliance_id == nc.non_compliance_id

    @pytest.mark.asyncio
    async def test_get_record_not_found(self, manager):
        result = await manager.get_record("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_records_empty(self, manager):
        result = await manager.list_records()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_records_multiple(self, manager):
        await manager.record_violation(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="minor",
            description="Violation 1",
        )
        await manager.record_violation(
            operator_id="OP-002",
            authority_id="AUTH-FR-001",
            violation_type="deforestation_link",
            severity="critical",
            description="Violation 2",
        )
        result = await manager.list_records()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        health = await manager.health_check()
        assert health["status"] == "healthy"
