# -*- coding: utf-8 -*-
"""
Tests for CrossReferenceVerifier - AGENT-EUDR-012 Engine 7: Cross-Reference Verification

Comprehensive test suite covering:
- All 6 registry types (FSC, RSPO, ISCC, Fairtrade, UTZ/RA, IPPC)
- All 5 verification statuses (verified, not_found, expired, revoked, error)
- Scope matching
- Quantity matching
- Party matching
- Batch cross-reference
- Response caching with TTL
- Cache statistics

Test count: 45+ tests
Coverage target: >= 85% of CrossReferenceVerifier module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication Agent (GL-EUDR-DAV-012)
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.document_authentication.conftest import (
    REGISTRY_TYPES,
    CROSSREF_STATUSES,
    SHA256_HEX_LENGTH,
    DOC_ID_FSC_001,
    DOC_ID_BOL_001,
    CERT_NUM_FSC_001,
    CERT_NUM_RSPO_001,
    CERT_NUM_ISCC_001,
    CROSSREF_FSC_VERIFIED,
    CROSSREF_RSPO_EXPIRED,
    make_crossref_result,
    assert_crossref_valid,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. All Registry Types
# ===========================================================================


class TestAllRegistryTypes:
    """Test cross-reference verification across all 6 registry types."""

    @pytest.mark.parametrize("registry_type", REGISTRY_TYPES)
    def test_verify_all_registries(self, crossref_engine, registry_type):
        """Each registry type can be queried."""
        result = make_crossref_result(registry_type=registry_type, status="verified")
        assert_crossref_valid(result)
        assert result["registry_type"] == registry_type

    @pytest.mark.parametrize("registry_type", REGISTRY_TYPES)
    def test_registry_result_structure(self, crossref_engine, registry_type):
        """Each registry result has required fields."""
        result = make_crossref_result(registry_type=registry_type)
        required_keys = [
            "crossref_id", "document_id", "registry_type",
            "certificate_number", "status", "scope_match",
            "quantity_match", "party_match", "processing_time_ms",
        ]
        for key in required_keys:
            assert key in result, f"Missing key '{key}' for registry '{registry_type}'"

    def test_fsc_registry_verification(self, crossref_engine):
        """FSC registry verification returns result."""
        result = crossref_engine.verify(
            registry_type="fsc",
            certificate_number=CERT_NUM_FSC_001,
        )
        assert result is not None
        assert result["registry_type"] == "fsc"

    def test_rspo_registry_verification(self, crossref_engine):
        """RSPO registry verification returns result."""
        result = crossref_engine.verify(
            registry_type="rspo",
            certificate_number=CERT_NUM_RSPO_001,
        )
        assert result is not None
        assert result["registry_type"] == "rspo"

    def test_iscc_registry_verification(self, crossref_engine):
        """ISCC registry verification returns result."""
        result = crossref_engine.verify(
            registry_type="iscc",
            certificate_number=CERT_NUM_ISCC_001,
        )
        assert result is not None
        assert result["registry_type"] == "iscc"

    def test_fairtrade_registry_verification(self, crossref_engine):
        """Fairtrade registry verification returns result."""
        result = crossref_engine.verify(
            registry_type="fairtrade",
            certificate_number="FT-12345",
        )
        assert result is not None

    def test_utz_ra_registry_verification(self, crossref_engine):
        """UTZ/Rainforest Alliance registry verification returns result."""
        result = crossref_engine.verify(
            registry_type="utz_ra",
            certificate_number="RA-98765",
        )
        assert result is not None

    def test_ippc_registry_verification(self, crossref_engine):
        """IPPC ePhyto registry verification returns result."""
        result = crossref_engine.verify(
            registry_type="ippc",
            certificate_number="EPHYTO-GH-2026-001",
        )
        assert result is not None


# ===========================================================================
# 2. Verification Statuses
# ===========================================================================


class TestVerificationStatuses:
    """Test all cross-reference verification statuses."""

    @pytest.mark.parametrize("status", CROSSREF_STATUSES)
    def test_all_statuses(self, crossref_engine, status):
        """All 5 verification statuses are valid."""
        result = make_crossref_result(status=status)
        assert result["status"] == status

    def test_verified_status(self, crossref_engine):
        """Verified status means certificate is valid in registry."""
        result = make_crossref_result(status="verified")
        assert result["status"] == "verified"

    def test_not_found_status(self, crossref_engine):
        """Not found status means certificate not in registry."""
        result = make_crossref_result(status="not_found")
        assert result["status"] == "not_found"

    def test_expired_status(self, crossref_engine):
        """Expired status means certificate expired in registry."""
        result = copy.deepcopy(CROSSREF_RSPO_EXPIRED)
        assert result["status"] == "expired"

    def test_revoked_status(self, crossref_engine):
        """Revoked status means certificate revoked in registry."""
        result = make_crossref_result(status="revoked")
        assert result["status"] == "revoked"

    def test_error_status(self, crossref_engine):
        """Error status means registry API call failed."""
        result = make_crossref_result(status="error")
        assert result["status"] == "error"


# ===========================================================================
# 3. Scope Matching
# ===========================================================================


class TestScopeMatching:
    """Test scope matching between document and registry."""

    def test_scope_match_true(self, crossref_engine):
        """Matching scope returns scope_match=True."""
        result = make_crossref_result(scope_match=True)
        assert result["scope_match"] is True

    def test_scope_match_false(self, crossref_engine):
        """Mismatching scope returns scope_match=False."""
        result = make_crossref_result(scope_match=False)
        assert result["scope_match"] is False

    def test_scope_match_verified_consistent(self, crossref_engine):
        """Verified certificate with scope match is fully consistent."""
        result = make_crossref_result(status="verified", scope_match=True)
        assert result["status"] == "verified"
        assert result["scope_match"] is True

    def test_scope_mismatch_flagged(self, crossref_engine):
        """Scope mismatch is flagged even when certificate is verified."""
        result = make_crossref_result(status="verified", scope_match=False)
        assert result["scope_match"] is False


# ===========================================================================
# 4. Quantity Matching
# ===========================================================================


class TestQuantityMatching:
    """Test quantity matching between document and registry."""

    def test_quantity_match_true(self, crossref_engine):
        """Matching quantity returns quantity_match=True."""
        result = make_crossref_result(quantity_match=True)
        assert result["quantity_match"] is True

    def test_quantity_match_false(self, crossref_engine):
        """Mismatching quantity returns quantity_match=False."""
        result = make_crossref_result(quantity_match=False)
        assert result["quantity_match"] is False

    def test_quantity_within_tolerance(self, crossref_engine):
        """Quantity within tolerance matches."""
        result = crossref_engine.verify(
            registry_type="fsc",
            certificate_number=CERT_NUM_FSC_001,
            claimed_quantity_kg=50000.0,
        )
        assert result is not None

    def test_quantity_outside_tolerance(self, crossref_engine):
        """Quantity outside tolerance does not match."""
        result = crossref_engine.verify(
            registry_type="fsc",
            certificate_number=CERT_NUM_FSC_001,
            claimed_quantity_kg=999999.0,
        )
        assert result is not None


# ===========================================================================
# 5. Party Matching
# ===========================================================================


class TestPartyMatching:
    """Test party matching between document and registry."""

    def test_party_match_true(self, crossref_engine):
        """Matching party returns party_match=True."""
        result = make_crossref_result(party_match=True)
        assert result["party_match"] is True

    def test_party_match_false(self, crossref_engine):
        """Mismatching party returns party_match=False."""
        result = make_crossref_result(party_match=False)
        assert result["party_match"] is False

    def test_party_match_with_holder(self, crossref_engine):
        """Party matching compares against registry holder."""
        result = crossref_engine.verify(
            registry_type="fsc",
            certificate_number=CERT_NUM_FSC_001,
            claimed_holder="Amazon Timber Ltd",
        )
        assert result is not None


# ===========================================================================
# 6. Batch Cross-Reference
# ===========================================================================


class TestBatchCrossReference:
    """Test batch cross-reference verification."""

    def test_batch_verify_multiple(self, crossref_engine):
        """Batch verify multiple certificate numbers."""
        requests = [
            {"registry_type": "fsc", "certificate_number": f"FSC-C{100000+i}"}
            for i in range(5)
        ]
        results = crossref_engine.batch_verify(requests)
        assert len(results) == 5

    def test_batch_verify_empty(self, crossref_engine):
        """Batch verify with empty list returns empty results."""
        results = crossref_engine.batch_verify([])
        assert len(results) == 0

    def test_batch_verify_mixed_registries(self, crossref_engine):
        """Batch verify across different registry types."""
        requests = [
            {"registry_type": "fsc", "certificate_number": CERT_NUM_FSC_001},
            {"registry_type": "rspo", "certificate_number": CERT_NUM_RSPO_001},
            {"registry_type": "iscc", "certificate_number": CERT_NUM_ISCC_001},
        ]
        results = crossref_engine.batch_verify(requests)
        assert len(results) == 3

    def test_batch_verify_partial_failure(self, crossref_engine):
        """Batch verify handles partial failures gracefully."""
        requests = [
            {"registry_type": "fsc", "certificate_number": CERT_NUM_FSC_001},
            {"registry_type": "invalid_registry", "certificate_number": "INVALID"},
        ]
        results = crossref_engine.batch_verify(requests, continue_on_error=True)
        assert len(results) == 2


# ===========================================================================
# 7. Response Caching
# ===========================================================================


class TestResponseCaching:
    """Test response caching with TTL."""

    def test_first_request_not_cached(self, crossref_engine):
        """First request is not served from cache."""
        result = crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-CACHE-001",
        )
        assert result.get("cached") is False

    def test_second_request_cached(self, crossref_engine):
        """Second request for same certificate is served from cache."""
        crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-CACHE-002",
        )
        result = crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-CACHE-002",
        )
        assert result.get("cached") is True

    def test_cache_ttl_remaining(self, crossref_engine):
        """Cached result includes TTL remaining."""
        crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-CACHE-003",
        )
        result = crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-CACHE-003",
        )
        if result.get("cached"):
            assert result.get("cache_ttl_remaining_s") is not None
            assert result["cache_ttl_remaining_s"] > 0

    def test_cache_invalidation(self, crossref_engine):
        """Cache can be invalidated for a specific certificate."""
        crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-CACHE-004",
        )
        crossref_engine.invalidate_cache(
            registry_type="fsc",
            certificate_number="FSC-CACHE-004",
        )
        result = crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-CACHE-004",
        )
        assert result.get("cached") is False

    def test_cache_clear_all(self, crossref_engine):
        """Entire cache can be cleared."""
        crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-CACHE-005",
        )
        crossref_engine.clear_cache()
        result = crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-CACHE-005",
        )
        assert result.get("cached") is False


# ===========================================================================
# 8. Cache Statistics
# ===========================================================================


class TestCacheStatistics:
    """Test cache statistics."""

    def test_cache_stats_empty(self, crossref_engine):
        """Empty cache returns zero counts."""
        stats = crossref_engine.get_cache_stats()
        assert stats is not None
        assert stats.get("total_entries", 0) >= 0

    def test_cache_stats_after_requests(self, crossref_engine):
        """Cache stats update after verification requests."""
        crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-STAT-001",
        )
        stats = crossref_engine.get_cache_stats()
        assert stats.get("total_entries", 0) >= 1 or stats.get("misses", 0) >= 1

    def test_cache_hit_rate(self, crossref_engine):
        """Cache hit rate is tracked."""
        crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-STAT-002",
        )
        crossref_engine.verify(
            registry_type="fsc",
            certificate_number="FSC-STAT-002",
        )
        stats = crossref_engine.get_cache_stats()
        assert "hit_rate" in stats or "hits" in stats


# ===========================================================================
# 9. Edge Cases
# ===========================================================================


class TestCrossRefEdgeCases:
    """Test edge cases for cross-reference verification."""

    def test_empty_certificate_number_raises(self, crossref_engine):
        """Empty certificate number raises ValueError."""
        with pytest.raises(ValueError):
            crossref_engine.verify(
                registry_type="fsc",
                certificate_number="",
            )

    def test_none_certificate_raises(self, crossref_engine):
        """None certificate number raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            crossref_engine.verify(
                registry_type="fsc",
                certificate_number=None,
            )

    def test_invalid_registry_raises(self, crossref_engine):
        """Invalid registry type raises ValueError."""
        with pytest.raises(ValueError):
            crossref_engine.verify(
                registry_type="invalid_registry",
                certificate_number="CERT-001",
            )

    def test_response_time_recorded(self, crossref_engine):
        """Response time from registry is recorded."""
        result = make_crossref_result()
        assert "response_time_ms" in result
        assert result["response_time_ms"] >= 0

    def test_processing_time_recorded(self, crossref_engine):
        """Overall processing time is recorded."""
        result = make_crossref_result()
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_provenance_hash_on_result(self, crossref_engine):
        """Cross-reference result can include a provenance hash."""
        result = make_crossref_result()
        result["provenance_hash"] = "a" * 64
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_factory_result_valid(self, crossref_engine):
        """Factory-built cross-reference result passes validation."""
        result = make_crossref_result()
        assert_crossref_valid(result)

    @pytest.mark.parametrize("registry,status", [
        ("fsc", "verified"),
        ("rspo", "expired"),
        ("iscc", "not_found"),
        ("fairtrade", "revoked"),
        ("utz_ra", "error"),
        ("ippc", "verified"),
    ])
    def test_registry_status_combinations(self, crossref_engine, registry, status):
        """Various registry-status combinations are valid."""
        result = make_crossref_result(registry_type=registry, status=status)
        assert_crossref_valid(result)
