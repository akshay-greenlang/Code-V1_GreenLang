# -*- coding: utf-8 -*-
"""
Unit tests for VersionController - AGENT-EUDR-037

Tests version creation, amendments, digital signatures, signature validation,
and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.agents.eudr.due_diligence_statement_creator.config import DDSCreatorConfig
from greenlang.agents.eudr.due_diligence_statement_creator.version_controller import VersionController
from greenlang.agents.eudr.due_diligence_statement_creator.models import (
    AmendmentReason, AmendmentRecord, DDSStatus,
    DigitalSignature, SignatureType, StatementVersion,
)


@pytest.fixture
def config():
    return DDSCreatorConfig()


@pytest.fixture
def controller(config):
    return VersionController(config=config)


class TestCreateVersion:
    @pytest.mark.asyncio
    async def test_returns_statement_version(self, controller):
        ver = await controller.create_version(
            statement_id="DDS-001", version_number=1, created_by="OP-001")
        assert isinstance(ver, StatementVersion)

    @pytest.mark.asyncio
    async def test_version_id_prefix(self, controller):
        ver = await controller.create_version(
            statement_id="DDS-001", version_number=1)
        assert ver.version_id.startswith("VER-")

    @pytest.mark.asyncio
    async def test_statement_id_set(self, controller):
        ver = await controller.create_version(
            statement_id="DDS-XYZ", version_number=1)
        assert ver.statement_id == "DDS-XYZ"

    @pytest.mark.asyncio
    async def test_version_number_set(self, controller):
        ver = await controller.create_version(
            statement_id="DDS-001", version_number=3)
        assert ver.version_number == 3

    @pytest.mark.asyncio
    async def test_draft_status(self, controller):
        ver = await controller.create_version(
            statement_id="DDS-001", version_number=1)
        assert ver.status == DDSStatus.DRAFT

    @pytest.mark.asyncio
    async def test_created_by_set(self, controller):
        ver = await controller.create_version(
            statement_id="DDS-001", version_number=1, created_by="OP-USER")
        assert ver.created_by == "OP-USER"

    @pytest.mark.asyncio
    async def test_amendment_reason_parsed(self, controller):
        ver = await controller.create_version(
            statement_id="DDS-001", version_number=2,
            amendment_reason="correction_of_error")
        assert ver.amendment_reason == AmendmentReason.CORRECTION_OF_ERROR

    @pytest.mark.asyncio
    async def test_invalid_amendment_reason_ignored(self, controller):
        ver = await controller.create_version(
            statement_id="DDS-001", version_number=2,
            amendment_reason="invalid_reason")
        assert ver.amendment_reason is None

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, controller):
        ver = await controller.create_version(
            statement_id="DDS-001", version_number=1)
        assert len(ver.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_supersedes_version_set(self, controller):
        v1 = await controller.create_version(
            statement_id="DDS-001", version_number=1)
        v2 = await controller.create_version(
            statement_id="DDS-001", version_number=2)
        assert v2.supersedes_version == v1.version_id

    @pytest.mark.asyncio
    async def test_version_count_increments(self, controller):
        await controller.create_version(statement_id="DDS-001", version_number=1)
        await controller.create_version(statement_id="DDS-001", version_number=2)
        health = await controller.health_check()
        assert health["versions_created"] == 2


class TestGetVersions:
    @pytest.mark.asyncio
    async def test_get_versions_returns_list(self, controller):
        await controller.create_version(statement_id="DDS-001", version_number=1)
        versions = await controller.get_versions("DDS-001")
        assert isinstance(versions, list)
        assert len(versions) == 1

    @pytest.mark.asyncio
    async def test_get_versions_sorted(self, controller):
        await controller.create_version(statement_id="DDS-001", version_number=2)
        await controller.create_version(statement_id="DDS-001", version_number=1)
        versions = await controller.get_versions("DDS-001")
        assert versions[0].version_number <= versions[1].version_number

    @pytest.mark.asyncio
    async def test_get_versions_nonexistent(self, controller):
        versions = await controller.get_versions("DDS-NONEXISTENT")
        assert versions == []


class TestGetLatestVersion:
    @pytest.mark.asyncio
    async def test_returns_latest(self, controller):
        await controller.create_version(statement_id="DDS-001", version_number=1)
        await controller.create_version(statement_id="DDS-001", version_number=2)
        latest = await controller.get_latest_version("DDS-001")
        assert latest is not None
        assert latest.version_number == 2

    @pytest.mark.asyncio
    async def test_returns_none_for_nonexistent(self, controller):
        latest = await controller.get_latest_version("DDS-NONEXISTENT")
        assert latest is None


class TestCreateAmendment:
    @pytest.mark.asyncio
    async def test_returns_amendment_record(self, controller):
        amd = await controller.create_amendment(
            statement_id="DDS-001", reason="correction_of_error",
            description="Fix operator address", previous_version=1)
        assert isinstance(amd, AmendmentRecord)

    @pytest.mark.asyncio
    async def test_amendment_id_prefix(self, controller):
        amd = await controller.create_amendment(
            statement_id="DDS-001", reason="correction_of_error",
            description="Fix", previous_version=1)
        assert amd.amendment_id.startswith("AMD-")

    @pytest.mark.asyncio
    async def test_new_version_created(self, controller):
        amd = await controller.create_amendment(
            statement_id="DDS-001", reason="correction_of_error",
            description="Fix", previous_version=1)
        assert amd.new_version == 2
        versions = await controller.get_versions("DDS-001")
        assert len(versions) == 1

    @pytest.mark.asyncio
    async def test_amendment_reason_parsed(self, controller):
        amd = await controller.create_amendment(
            statement_id="DDS-001", reason="additional_information",
            description="Add data", previous_version=1)
        assert amd.reason == AmendmentReason.ADDITIONAL_INFORMATION

    @pytest.mark.asyncio
    async def test_invalid_reason_defaults(self, controller):
        amd = await controller.create_amendment(
            statement_id="DDS-001", reason="invalid_xyz",
            description="Fix", previous_version=1)
        assert amd.reason == AmendmentReason.CORRECTION_OF_ERROR

    @pytest.mark.asyncio
    async def test_changed_fields_set(self, controller):
        amd = await controller.create_amendment(
            statement_id="DDS-001", reason="correction_of_error",
            description="Fix", previous_version=1,
            changed_fields=["operator_address", "operator_name"])
        assert "operator_address" in amd.changed_fields

    @pytest.mark.asyncio
    async def test_changed_by_set(self, controller):
        amd = await controller.create_amendment(
            statement_id="DDS-001", reason="correction_of_error",
            description="Fix", previous_version=1,
            changed_by="admin@company.com")
        assert amd.changed_by == "admin@company.com"

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, controller):
        amd = await controller.create_amendment(
            statement_id="DDS-001", reason="correction_of_error",
            description="Fix", previous_version=1)
        assert len(amd.provenance_hash) == 64


class TestGetAmendments:
    @pytest.mark.asyncio
    async def test_get_amendments(self, controller):
        await controller.create_amendment(
            statement_id="DDS-001", reason="correction_of_error",
            description="Fix 1", previous_version=1)
        await controller.create_amendment(
            statement_id="DDS-001", reason="additional_information",
            description="Add info", previous_version=2)
        amendments = await controller.get_amendments("DDS-001")
        assert len(amendments) == 2

    @pytest.mark.asyncio
    async def test_get_amendments_nonexistent(self, controller):
        amendments = await controller.get_amendments("DDS-NONEXISTENT")
        assert amendments == []


class TestApplySignature:
    @pytest.mark.asyncio
    async def test_returns_digital_signature(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith")
        assert isinstance(sig, DigitalSignature)

    @pytest.mark.asyncio
    async def test_signature_id_prefix(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith")
        assert sig.signature_id.startswith("SIG-")

    @pytest.mark.asyncio
    async def test_signer_name_set(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="Jane Doe")
        assert sig.signer_name == "Jane Doe"

    @pytest.mark.asyncio
    async def test_default_signature_type(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith")
        assert sig.signature_type == SignatureType.QUALIFIED_ELECTRONIC

    @pytest.mark.asyncio
    async def test_custom_signature_type(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith",
            signature_type="advanced_electronic")
        assert sig.signature_type == SignatureType.ADVANCED_ELECTRONIC

    @pytest.mark.asyncio
    async def test_invalid_signature_type_defaults(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith",
            signature_type="invalid_xyz")
        assert sig.signature_type == SignatureType.QUALIFIED_ELECTRONIC

    @pytest.mark.asyncio
    async def test_is_valid_true(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith")
        assert sig.is_valid is True

    @pytest.mark.asyncio
    async def test_validity_period(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith")
        assert sig.valid_from is not None
        assert sig.valid_until is not None
        delta = sig.valid_until - sig.valid_from
        assert delta.days == 365

    @pytest.mark.asyncio
    async def test_algorithm_from_config(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith")
        assert sig.algorithm == "RSA-SHA256"

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith")
        assert len(sig.provenance_hash) == 64


class TestValidateSignature:
    @pytest.mark.asyncio
    async def test_valid_signature(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith")
        result = await controller.validate_signature(sig)
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_expired_signature(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith")
        sig.valid_until = datetime.now(timezone.utc) - timedelta(days=1)
        result = await controller.validate_signature(sig)
        assert result["valid"] is False
        assert any("expired" in i.lower() for i in result["issues"])

    @pytest.mark.asyncio
    async def test_non_qualified_signature_fails(self, controller):
        sig = await controller.apply_signature(
            statement_id="DDS-001", signer_name="John Smith",
            signature_type="simple_electronic")
        result = await controller.validate_signature(sig)
        assert result["valid"] is False
        assert any("qualified" in i.lower() for i in result["issues"])


class TestVersionControllerHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, controller):
        health = await controller.health_check()
        assert health["engine"] == "VersionController"
        assert health["status"] == "healthy"
        assert health["versions_created"] == 0
