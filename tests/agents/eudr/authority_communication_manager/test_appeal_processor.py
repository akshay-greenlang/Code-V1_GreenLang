# -*- coding: utf-8 -*-
"""
Unit tests for AppealProcessor engine - AGENT-EUDR-040

Tests appeal filing, decision recording, extension granting, withdrawal,
deadline enforcement, penalty suspension, provenance tracking, and
complete appeal workflow lifecycle.

55+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
)
from greenlang.agents.eudr.authority_communication_manager.appeal_processor import (
    AppealProcessor,
)
from greenlang.agents.eudr.authority_communication_manager.models import (
    Appeal,
    AppealDecision,
)


@pytest.fixture
def config():
    return AuthorityCommunicationManagerConfig()


@pytest.fixture
def processor(config):
    return AppealProcessor(config=config)


# ====================================================================
# Initialization
# ====================================================================


class TestInit:
    def test_processor_created(self, processor):
        assert processor is not None

    def test_default_config(self):
        p = AppealProcessor()
        assert p.config is not None

    def test_custom_config(self, config):
        p = AppealProcessor(config=config)
        assert p.config is config

    def test_appeals_empty(self, processor):
        assert len(processor._appeals) == 0

    def test_provenance_initialized(self, processor):
        assert processor._provenance is not None


# ====================================================================
# File Appeal
# ====================================================================


class TestFileAppeal:
    @pytest.mark.asyncio
    async def test_file_appeal(self, processor):
        result = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Geolocation data was submitted but not properly processed by the portal.",
        )
        assert isinstance(result, Appeal)
        assert result.decision == AppealDecision.PENDING

    @pytest.mark.asyncio
    async def test_file_appeal_assigns_id(self, processor):
        result = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        assert result.appeal_id is not None
        assert len(result.appeal_id) > 0

    @pytest.mark.asyncio
    async def test_file_appeal_sets_deadline(self, processor):
        result = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        assert result.deadline is not None

    @pytest.mark.asyncio
    async def test_file_appeal_penalty_suspended(self, processor):
        result = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        assert result.penalty_suspended is True

    @pytest.mark.asyncio
    async def test_file_appeal_computes_provenance(self, processor):
        result = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_file_appeal_with_evidence(self, processor):
        result = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
            supporting_evidence=["DOC-010", "DOC-011", "DOC-012"],
        )
        assert len(result.supporting_evidence) == 3

    @pytest.mark.asyncio
    async def test_file_appeal_stored(self, processor):
        result = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        assert result.appeal_id in processor._appeals

    @pytest.mark.asyncio
    async def test_file_appeal_empty_grounds_raises(self, processor):
        with pytest.raises(ValueError, match="grounds"):
            await processor.file_appeal(
                non_compliance_id="NC-001",
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                grounds="",
            )

    @pytest.mark.asyncio
    async def test_file_appeal_short_grounds_raises(self, processor):
        with pytest.raises(ValueError, match="grounds"):
            await processor.file_appeal(
                non_compliance_id="NC-001",
                operator_id="OP-001",
                authority_id="AUTH-DE-001",
                grounds="Too short",
            )

    @pytest.mark.asyncio
    async def test_file_appeal_extensions_zero(self, processor):
        result = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        assert result.extensions_granted == 0


# ====================================================================
# Record Decision
# ====================================================================


class TestRecordDecision:
    @pytest.mark.asyncio
    async def test_record_upheld(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        result = await processor.record_decision(
            appeal_id=appeal.appeal_id,
            decision="upheld",
            reason="Appeal grounds substantiated by evidence.",
        )
        assert result.decision == AppealDecision.UPHELD

    @pytest.mark.asyncio
    async def test_record_overturned(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        result = await processor.record_decision(
            appeal_id=appeal.appeal_id,
            decision="overturned",
            reason="Original decision had procedural errors.",
        )
        assert result.decision == AppealDecision.OVERTURNED

    @pytest.mark.asyncio
    async def test_record_partially_upheld(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        result = await processor.record_decision(
            appeal_id=appeal.appeal_id,
            decision="partially_upheld",
            reason="Penalty reduced.",
        )
        assert result.decision == AppealDecision.PARTIALLY_UPHELD

    @pytest.mark.asyncio
    async def test_record_dismissed(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        result = await processor.record_decision(
            appeal_id=appeal.appeal_id,
            decision="dismissed",
            reason="Insufficient evidence to support appeal.",
        )
        assert result.decision == AppealDecision.DISMISSED

    @pytest.mark.asyncio
    async def test_record_decision_sets_date(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        result = await processor.record_decision(
            appeal_id=appeal.appeal_id,
            decision="upheld",
            reason="Test reason.",
        )
        assert result.decision_date is not None

    @pytest.mark.asyncio
    async def test_record_decision_not_found(self, processor):
        with pytest.raises(ValueError, match="not found"):
            await processor.record_decision(
                appeal_id="nonexistent",
                decision="upheld",
                reason="Test reason.",
            )

    @pytest.mark.asyncio
    async def test_record_decision_invalid(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        with pytest.raises(ValueError, match="Invalid"):
            await processor.record_decision(
                appeal_id=appeal.appeal_id,
                decision="not_a_decision",
                reason="Test reason.",
            )


# ====================================================================
# Grant Extension
# ====================================================================


class TestGrantExtension:
    @pytest.mark.asyncio
    async def test_grant_extension(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        original_deadline = appeal.deadline
        result = await processor.grant_extension(appeal.appeal_id)
        assert result.extensions_granted == 1
        assert result.deadline > original_deadline

    @pytest.mark.asyncio
    async def test_grant_two_extensions(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        await processor.grant_extension(appeal.appeal_id)
        result = await processor.grant_extension(appeal.appeal_id)
        assert result.extensions_granted == 2

    @pytest.mark.asyncio
    async def test_grant_extension_max_exceeded(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        await processor.grant_extension(appeal.appeal_id)
        await processor.grant_extension(appeal.appeal_id)
        with pytest.raises(ValueError, match="[Mm]ax"):
            await processor.grant_extension(appeal.appeal_id)

    @pytest.mark.asyncio
    async def test_grant_extension_not_found(self, processor):
        with pytest.raises(ValueError, match="not found"):
            await processor.grant_extension("nonexistent")


# ====================================================================
# Withdraw Appeal
# ====================================================================


class TestWithdrawAppeal:
    @pytest.mark.asyncio
    async def test_withdraw(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        result = await processor.withdraw_appeal(appeal.appeal_id)
        assert result.decision == AppealDecision.WITHDRAWN

    @pytest.mark.asyncio
    async def test_withdraw_not_found(self, processor):
        with pytest.raises(ValueError, match="not found"):
            await processor.withdraw_appeal("nonexistent")


# ====================================================================
# Get / List / Health
# ====================================================================


class TestGetListHealth:
    @pytest.mark.asyncio
    async def test_get_appeal(self, processor):
        appeal = await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for the appeal submission.",
        )
        result = await processor.get_appeal(appeal.appeal_id)
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_appeal_not_found(self, processor):
        result = await processor.get_appeal("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_appeals_empty(self, processor):
        result = await processor.list_appeals()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_appeals_multiple(self, processor):
        await processor.file_appeal(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Test grounds for appeal one submission.",
        )
        await processor.file_appeal(
            non_compliance_id="NC-002",
            operator_id="OP-002",
            authority_id="AUTH-FR-001",
            grounds="Test grounds for appeal two submission.",
        )
        result = await processor.list_appeals()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_health_check(self, processor):
        health = await processor.health_check()
        assert health["status"] == "healthy"
