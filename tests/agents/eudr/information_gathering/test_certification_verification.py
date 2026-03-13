# -*- coding: utf-8 -*-
"""
Unit tests for CertificationVerificationEngine - AGENT-EUDR-027

Tests certificate verification against 6 certification bodies (FSC, RSPO,
PEFC, Rainforest Alliance, UTZ, EU Organic), cache behavior, batch
verification, expiring certificate monitoring, supplier certificate
lookups, compliance matrix generation, and engine statistics.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 2: Certification Verification)
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.information_gathering.certification_verification_engine import (
    CertificationVerificationEngine,
    VerificationCache,
)
from greenlang.agents.eudr.information_gathering.config import InformationGatheringConfig
from greenlang.agents.eudr.information_gathering.models import (
    CertificateVerificationResult,
    CertificationBody,
    CertVerificationStatus,
    EUDRCommodity,
)


class TestCertificationVerificationEngineInit:
    """Test engine initialization."""

    def test_engine_initialization(self, config):
        engine = CertificationVerificationEngine(config)
        stats = engine.get_verification_stats()
        assert stats["adapters_registered"] == 6
        assert stats["total_verified"] == 0

    def test_engine_with_default_config(self):
        engine = CertificationVerificationEngine()
        stats = engine.get_verification_stats()
        assert stats["adapters_registered"] == 6


class TestVerifyCertificate:
    """Test single certificate verification operations."""

    @pytest.mark.asyncio
    async def test_verify_certificate_valid_fsc(self, config):
        engine = CertificationVerificationEngine(config)
        result = await engine.verify_certificate("FSC-C012345", CertificationBody.FSC)
        assert result.verification_status == CertVerificationStatus.VALID
        assert result.holder_name == "Sample Forest Products Ltd"
        assert EUDRCommodity.WOOD in result.commodity_scope
        assert result.chain_of_custody_model == "Transfer"
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_verify_certificate_not_found(self, config):
        engine = CertificationVerificationEngine(config)
        result = await engine.verify_certificate("INVALID-001", CertificationBody.FSC)
        assert result.verification_status == CertVerificationStatus.NOT_FOUND
        assert result.holder_name == ""

    @pytest.mark.asyncio
    async def test_verify_certificate_expired_utz(self, config):
        engine = CertificationVerificationEngine(config)
        result = await engine.verify_certificate("UTZ-12345", CertificationBody.UTZ)
        assert result.verification_status == CertVerificationStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_verify_certificate_rspo(self, config):
        engine = CertificationVerificationEngine(config)
        result = await engine.verify_certificate("RSPO-54321", CertificationBody.RSPO)
        assert result.verification_status == CertVerificationStatus.VALID
        assert EUDRCommodity.OIL_PALM in result.commodity_scope

    @pytest.mark.asyncio
    async def test_verify_certificate_pefc(self, config):
        engine = CertificationVerificationEngine(config)
        result = await engine.verify_certificate("PEFC/01-23-456", CertificationBody.PEFC)
        assert result.verification_status == CertVerificationStatus.VALID
        assert result.chain_of_custody_model == "Percentage"

    @pytest.mark.asyncio
    async def test_verify_certificate_rainforest_alliance(self, config):
        engine = CertificationVerificationEngine(config)
        result = await engine.verify_certificate("RA-2026-001", CertificationBody.RAINFOREST_ALLIANCE)
        assert result.verification_status == CertVerificationStatus.VALID
        assert EUDRCommodity.COCOA in result.commodity_scope

    @pytest.mark.asyncio
    async def test_verify_certificate_eu_organic(self, config):
        engine = CertificationVerificationEngine(config)
        result = await engine.verify_certificate("EU-BIO-123", CertificationBody.EU_ORGANIC)
        assert result.verification_status == CertVerificationStatus.VALID
        assert result.chain_of_custody_model == "Identity Preserved"

    @pytest.mark.asyncio
    async def test_verify_certificate_san_valid(self, config):
        engine = CertificationVerificationEngine(config)
        result = await engine.verify_certificate("SAN-2026-001", CertificationBody.UTZ)
        assert result.verification_status == CertVerificationStatus.VALID


class TestVerificationCache:
    """Test certificate verification cache."""

    @pytest.mark.asyncio
    async def test_verification_cache(self, config):
        engine = CertificationVerificationEngine(config)
        cert_id = "FSC-C999999"
        result1 = await engine.verify_certificate(cert_id, CertificationBody.FSC)
        result2 = await engine.verify_certificate(cert_id, CertificationBody.FSC)
        # Second call should return cached result
        assert result1.verification_status == result2.verification_status
        assert result1.provenance_hash == result2.provenance_hash

    def test_verification_cache_operations(self):
        cache = VerificationCache(default_ttl_seconds=3600)
        result = CertificateVerificationResult(
            certificate_id="TEST-001",
            certification_body=CertificationBody.FSC,
            verification_status=CertVerificationStatus.VALID,
        )
        cache.put("TEST-001", CertificationBody.FSC, result)
        assert cache.get("TEST-001", CertificationBody.FSC) is not None
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0


class TestBatchVerify:
    """Test batch certificate verification."""

    @pytest.mark.asyncio
    async def test_batch_verify(self, config):
        engine = CertificationVerificationEngine(config)
        certs = [
            ("FSC-C012345", CertificationBody.FSC),
            ("RSPO-54321", CertificationBody.RSPO),
            ("INVALID-001", CertificationBody.PEFC),
        ]
        results = await engine.batch_verify(certs)
        assert len(results) == 3
        assert results[0].verification_status == CertVerificationStatus.VALID
        assert results[1].verification_status == CertVerificationStatus.VALID
        assert results[2].verification_status == CertVerificationStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_batch_verify_empty(self, config):
        engine = CertificationVerificationEngine(config)
        results = await engine.batch_verify([])
        assert results == []


class TestExpiringCertificates:
    """Test expiring certificate monitoring."""

    @pytest.mark.asyncio
    async def test_get_expiring_certificates(self, config):
        engine = CertificationVerificationEngine(config)
        # Verify some certificates first to populate history
        await engine.verify_certificate("FSC-C012345", CertificationBody.FSC)
        await engine.verify_certificate("RA-2026-001", CertificationBody.RAINFOREST_ALLIANCE)
        # RA has 165 days validity -> within 180-day window
        expiring = engine.get_expiring_certificates(days_ahead=180)
        assert isinstance(expiring, list)
        # The RA cert should be expiring within 180 days
        ra_certs = [c for c in expiring if c.certification_body == CertificationBody.RAINFOREST_ALLIANCE]
        assert len(ra_certs) >= 1

    @pytest.mark.asyncio
    async def test_get_expiring_certificates_none(self, config):
        engine = CertificationVerificationEngine(config)
        # No certs verified -> no expiring
        expiring = engine.get_expiring_certificates(days_ahead=90)
        assert expiring == []


class TestSupplierCertificates:
    """Test supplier certificate lookups."""

    @pytest.mark.asyncio
    async def test_get_supplier_certificates(self, config):
        engine = CertificationVerificationEngine(config)
        await engine.verify_certificate("FSC-C012345", CertificationBody.FSC)
        results = engine.get_supplier_certificates("Sample Forest")
        assert len(results) >= 1
        assert results[0].holder_name == "Sample Forest Products Ltd"

    @pytest.mark.asyncio
    async def test_get_supplier_certificates_no_match(self, config):
        engine = CertificationVerificationEngine(config)
        results = engine.get_supplier_certificates("NonExistent Corp")
        assert results == []


class TestVerificationStats:
    """Test engine statistics."""

    @pytest.mark.asyncio
    async def test_get_verification_stats(self, config):
        engine = CertificationVerificationEngine(config)
        await engine.verify_certificate("FSC-C012345", CertificationBody.FSC)
        await engine.verify_certificate("INVALID-001", CertificationBody.FSC)
        stats = engine.get_verification_stats()
        assert stats["total_verified"] == 2
        assert "valid" in stats["status_breakdown"]
        assert "not_found" in stats["status_breakdown"]


class TestCacheClearAndHistory:
    """Test cache and history clearing."""

    @pytest.mark.asyncio
    async def test_clear_cache(self, config):
        engine = CertificationVerificationEngine(config)
        await engine.verify_certificate("FSC-C012345", CertificationBody.FSC)
        engine.clear_cache()
        stats = engine.get_verification_stats()
        assert stats["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_clear_history(self, config):
        engine = CertificationVerificationEngine(config)
        await engine.verify_certificate("FSC-C012345", CertificationBody.FSC)
        engine.clear_history()
        stats = engine.get_verification_stats()
        assert stats["total_verified"] == 0
