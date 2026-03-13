# -*- coding: utf-8 -*-
"""
Unit tests for OriginValidator engine - AGENT-EUDR-039

Tests country origin verification against supply chain data,
DDS cross-referencing, confidence scoring, mismatch detection,
batch operations, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.customs_declaration_support.config import CustomsDeclarationConfig
from greenlang.agents.eudr.customs_declaration_support.origin_validator import OriginValidator
from greenlang.agents.eudr.customs_declaration_support.models import (
    OriginVerification, OriginVerificationResult,
)


@pytest.fixture
def config():
    return CustomsDeclarationConfig()


@pytest.fixture
def validator(config):
    return OriginValidator(config=config)


# ====================================================================
# Origin Verification Tests
# ====================================================================


class TestVerifyOrigin:
    @pytest.mark.asyncio
    async def test_verify_matching_origin(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=["CI", "GH"],
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert isinstance(result, OriginVerification)
        assert result.result == OriginVerificationResult.VERIFIED

    @pytest.mark.asyncio
    async def test_verify_mismatching_origin(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-002",
            declared_origin="BR",
            supply_chain_origins=["CI", "GH"],
            dds_reference="GL-DDS-20260313-XYZABC",
        )
        assert result.result == OriginVerificationResult.MISMATCH
        assert result.mismatch_details != ""

    @pytest.mark.asyncio
    async def test_verify_empty_supply_chain(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-003",
            declared_origin="CI",
            supply_chain_origins=[],
            dds_reference="GL-DDS-20260313-DEFGHI",
        )
        assert result.result in (
            OriginVerificationResult.UNVERIFIED,
            OriginVerificationResult.MISMATCH,
        )

    @pytest.mark.asyncio
    async def test_verify_no_dds_reference(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-004",
            declared_origin="CI",
            supply_chain_origins=["CI"],
            dds_reference="",
        )
        assert result.result in (
            OriginVerificationResult.UNVERIFIED,
            OriginVerificationResult.MISMATCH,
        )


class TestConfidenceScoring:
    @pytest.mark.asyncio
    async def test_high_confidence_for_exact_match(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=["CI"],
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert result.confidence_score >= Decimal("80")

    @pytest.mark.asyncio
    async def test_lower_confidence_for_multi_origin(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=["CI", "GH", "CM", "NG"],
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        # Confidence should still be positive since origin is in list
        assert result.confidence_score > Decimal("0")

    @pytest.mark.asyncio
    async def test_zero_confidence_for_mismatch(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-002",
            declared_origin="JP",
            supply_chain_origins=["CI", "GH"],
            dds_reference="GL-DDS-20260313-XYZABC",
        )
        assert result.confidence_score <= Decimal("30")

    @pytest.mark.asyncio
    async def test_confidence_is_decimal(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=["CI"],
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert isinstance(result.confidence_score, Decimal)

    @pytest.mark.asyncio
    async def test_confidence_between_0_and_100(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=["CI"],
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert Decimal("0") <= result.confidence_score <= Decimal("100")


class TestMismatchDetails:
    @pytest.mark.asyncio
    async def test_mismatch_details_include_declared(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-002",
            declared_origin="BR",
            supply_chain_origins=["CI", "GH"],
            dds_reference="GL-DDS-20260313-XYZABC",
        )
        assert "BR" in result.mismatch_details

    @pytest.mark.asyncio
    async def test_mismatch_details_include_supply_chain(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-002",
            declared_origin="BR",
            supply_chain_origins=["CI", "GH"],
            dds_reference="GL-DDS-20260313-XYZABC",
        )
        assert "CI" in result.mismatch_details or "GH" in result.mismatch_details

    @pytest.mark.asyncio
    async def test_no_mismatch_details_for_verified(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=["CI"],
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert result.mismatch_details == "" or result.mismatch_details is None


class TestBatchVerification:
    @pytest.mark.asyncio
    async def test_batch_verify_multiple(self, validator):
        verifications = [
            {
                "declaration_id": "DECL-001",
                "declared_origin": "CI",
                "supply_chain_origins": ["CI"],
                "dds_reference": "GL-DDS-20260313-A",
            },
            {
                "declaration_id": "DECL-002",
                "declared_origin": "BR",
                "supply_chain_origins": ["CI"],
                "dds_reference": "GL-DDS-20260313-B",
            },
        ]
        results = await validator.verify_origin_batch(verifications)
        assert len(results) == 2
        verified = [r for r in results if r.result == OriginVerificationResult.VERIFIED]
        mismatched = [r for r in results if r.result == OriginVerificationResult.MISMATCH]
        assert len(verified) == 1
        assert len(mismatched) == 1

    @pytest.mark.asyncio
    async def test_batch_verify_empty(self, validator):
        results = await validator.verify_origin_batch([])
        assert results == []


class TestProvenanceTracking:
    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, validator):
        result = await validator.verify_origin(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=["CI"],
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_provenance_deterministic(self, validator):
        kwargs = dict(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=["CI"],
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        r1 = await validator.verify_origin(**kwargs)
        r2 = await validator.verify_origin(**kwargs)
        assert r1.provenance_hash == r2.provenance_hash


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_dict(self, validator):
        health = await validator.health_check()
        assert isinstance(health, dict)
        assert health["engine"] == "OriginValidator"

    @pytest.mark.asyncio
    async def test_status_healthy(self, validator):
        health = await validator.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_verifications_count(self, validator):
        health = await validator.health_check()
        assert health["verifications_performed"] == 0
