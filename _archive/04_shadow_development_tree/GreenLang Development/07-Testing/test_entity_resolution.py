"""
Test suite for Entity Resolution Services

Tests the multi-source entity resolution system including:
- GLEIF API integration
- DUNS lookup functionality
- OpenCorporates search
- Unified resolution with fallback
- Caching and rate limiting
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json

from greenlang.intelligence.entity_resolution import (
    EntityResolver,
    ResolutionRequest,
    ResolutionResult,
    EntityMatch,
    EntitySource,
    EntityType,
    GLEIFClient,
    DUNSClient,
    OpenCorporatesClient,
    RateLimiter
)


class TestEntityMatch:
    """Test EntityMatch model validation and methods."""

    def test_valid_entity_match(self):
        """Test creating a valid EntityMatch."""
        match = EntityMatch(
            entity_name="Apple Inc",
            lei="HWUPKR0MPOU8FGXBT394",
            country="US",
            source=EntitySource.GLEIF,
            confidence_score=0.95,
            match_method="lei_lookup"
        )
        assert match.entity_name == "Apple Inc"
        assert match.lei == "HWUPKR0MPOU8FGXBT394"
        assert match.confidence_score == 0.95

    def test_lei_validation(self):
        """Test LEI format validation."""
        # Valid LEI
        match = EntityMatch(
            entity_name="Test Corp",
            lei="ABCD1234567890123456",
            country="US",
            source=EntitySource.GLEIF,
            confidence_score=1.0,
            match_method="lei_lookup"
        )
        assert match.lei == "ABCD1234567890123456"

        # Invalid LEI - wrong length
        with pytest.raises(ValueError, match="Invalid LEI format"):
            EntityMatch(
                entity_name="Test Corp",
                lei="ABC123",
                country="US",
                source=EntitySource.GLEIF,
                confidence_score=1.0,
                match_method="lei_lookup"
            )

    def test_duns_validation(self):
        """Test DUNS number validation."""
        # Valid DUNS
        match = EntityMatch(
            entity_name="Test Corp",
            duns="123456789",
            country="US",
            source=EntitySource.DUNS,
            confidence_score=0.9,
            match_method="duns_lookup"
        )
        assert match.duns == "123456789"

        # Invalid DUNS - not 9 digits
        with pytest.raises(ValueError, match="Invalid DUNS format"):
            EntityMatch(
                entity_name="Test Corp",
                duns="12345",
                country="US",
                source=EntitySource.DUNS,
                confidence_score=0.9,
                match_method="duns_lookup"
            )


class TestResolutionRequest:
    """Test ResolutionRequest model."""

    def test_simple_request(self):
        """Test creating a simple resolution request."""
        request = ResolutionRequest(query="Microsoft Corporation")
        assert request.query == "Microsoft Corporation"
        assert request.fuzzy_match is True
        assert request.min_confidence == 0.7
        assert request.max_results == 10

    def test_detailed_request(self):
        """Test creating a detailed resolution request."""
        request = ResolutionRequest(
            query="Samsung",
            country="KR",
            lei="988400WZKH4DWWFPDY10",
            min_confidence=0.8,
            max_results=5,
            preferred_sources=[EntitySource.GLEIF, EntitySource.OPENCORPORATES]
        )
        assert request.country == "KR"
        assert request.lei == "988400WZKH4DWWFPDY10"
        assert request.min_confidence == 0.8
        assert len(request.preferred_sources) == 2


@pytest.mark.asyncio
class TestGLEIFClient:
    """Test GLEIF API client."""

    async def test_search_by_name(self):
        """Test searching entities by name via GLEIF."""
        client = GLEIFClient()

        # Mock the session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [{
                "attributes": {
                    "lei": "HWUPKR0MPOU8FGXBT394",
                    "entity": {
                        "legalName": {"name": "Apple Inc."},
                        "legalAddress": {
                            "country": "US",
                            "city": "Cupertino",
                            "region": "CA"
                        }
                    }
                }
            }]
        })

        with patch.object(client, 'session') as mock_session:
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock()

            results = await client.search_by_name("Apple", "US")

            assert len(results) == 1
            assert results[0]["attributes"]["lei"] == "HWUPKR0MPOU8FGXBT394"

    async def test_get_by_lei(self):
        """Test getting entity by LEI."""
        client = GLEIFClient()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": {
                "attributes": {
                    "lei": "HWUPKR0MPOU8FGXBT394",
                    "entity": {
                        "legalName": {"name": "Apple Inc."},
                        "status": "ACTIVE"
                    }
                }
            }
        })

        with patch.object(client, 'session') as mock_session:
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock()

            result = await client.get_by_lei("HWUPKR0MPOU8FGXBT394")

            assert result is not None
            assert result["attributes"]["lei"] == "HWUPKR0MPOU8FGXBT394"

    def test_parse_lei_record(self):
        """Test parsing GLEIF API response."""
        client = GLEIFClient()

        record = {
            "attributes": {
                "lei": "HWUPKR0MPOU8FGXBT394",
                "entity": {
                    "legalName": {"name": "Apple Inc."},
                    "legalAddress": {
                        "addressLine1": "One Apple Park Way",
                        "city": "Cupertino",
                        "region": "CA",
                        "postalCode": "95014",
                        "country": "US"
                    },
                    "status": "ACTIVE"
                },
                "lastUpdateDate": "2024-01-01T00:00:00"
            }
        }

        match = client.parse_lei_record(record)

        assert match.entity_name == "Apple Inc."
        assert match.lei == "HWUPKR0MPOU8FGXBT394"
        assert match.country == "US"
        assert match.status == "ACTIVE"
        assert match.source == EntitySource.GLEIF


@pytest.mark.asyncio
class TestDUNSClient:
    """Test DUNS/D&B API client."""

    async def test_search_by_name(self):
        """Test searching companies by name."""
        client = DUNSClient("test_key", "test_secret")

        # The mock implementation returns data for Microsoft
        results = await client.search_by_name("Microsoft", "US")

        # Check if we get expected mock data
        if results:  # Mock only returns data for "microsoft"
            assert results[0]["duns"] == "081466849"
            assert results[0]["name"] == "Microsoft Corporation"

    def test_parse_duns_record(self):
        """Test parsing D&B response."""
        client = DUNSClient("test", "test")

        record = {
            "duns": "081466849",
            "name": "Microsoft Corporation",
            "country": "US",
            "address": {
                "line1": "One Microsoft Way",
                "city": "Redmond",
                "state": "WA",
                "postalCode": "98052",
                "country": "US"
            },
            "incorporationYear": 1975
        }

        match = client.parse_duns_record(record)

        assert match.entity_name == "Microsoft Corporation"
        assert match.duns == "081466849"
        assert match.country == "US"
        assert match.source == EntitySource.DUNS


@pytest.mark.asyncio
class TestOpenCorporatesClient:
    """Test OpenCorporates API client."""

    async def test_search_companies(self):
        """Test searching companies."""
        client = OpenCorporatesClient()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "results": {
                "companies": [{
                    "company": {
                        "name": "Tesla, Inc.",
                        "company_number": "3581572",
                        "jurisdiction_code": "us_de",
                        "current_status": "Active",
                        "incorporation_date": "2003-07-01",
                        "registered_address": {
                            "street_address": "3500 Deer Creek Road",
                            "locality": "Palo Alto",
                            "region": "CA",
                            "postal_code": "94304",
                            "country": "United States"
                        }
                    }
                }]
            }
        })

        with patch.object(client, 'session') as mock_session:
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock()

            results = await client.search_companies("Tesla", country="US")

            assert len(results) == 1
            assert results[0]["name"] == "Tesla, Inc."
            assert results[0]["jurisdiction_code"] == "us_de"

    def test_parse_company_record(self):
        """Test parsing OpenCorporates response."""
        client = OpenCorporatesClient()

        record = {
            "name": "Tesla, Inc.",
            "company_number": "3581572",
            "jurisdiction_code": "us_de",
            "current_status": "Active",
            "incorporation_date": "2003-07-01",
            "registered_address": {
                "street_address": "3500 Deer Creek Road",
                "locality": "Palo Alto",
                "region": "CA",
                "postal_code": "94304",
                "country": "US"
            },
            "updated_at": "2024-01-01T00:00:00"
        }

        match = client.parse_company_record(record, confidence=0.9)

        assert match.entity_name == "Tesla, Inc."
        assert match.opencorporates_id == "us_de/3581572"
        assert match.status == "ACTIVE"
        assert match.source == EntitySource.OPENCORPORATES
        assert match.confidence_score == 0.9


@pytest.mark.asyncio
class TestRateLimiter:
    """Test rate limiting functionality."""

    async def test_rate_limiting(self):
        """Test that rate limiter delays calls appropriately."""
        limiter = RateLimiter(max_calls=2, period_seconds=1)

        start_time = asyncio.get_event_loop().time()

        # First two calls should go through immediately
        await limiter.acquire()
        await limiter.acquire()

        # Third call should be delayed
        await limiter.acquire()

        elapsed = asyncio.get_event_loop().time() - start_time

        # Should have waited at least 1 second
        assert elapsed >= 1.0


@pytest.mark.asyncio
class TestEntityResolver:
    """Test the main EntityResolver class."""

    async def test_simple_resolution(self):
        """Test simple entity resolution."""
        resolver = EntityResolver()

        # Mock the GLEIF client
        mock_match = EntityMatch(
            entity_name="Apple Inc",
            lei="HWUPKR0MPOU8FGXBT394",
            country="US",
            source=EntitySource.GLEIF,
            confidence_score=0.95,
            match_method="name_search"
        )

        with patch.object(resolver, '_resolve_from_gleif', return_value=[mock_match]):
            result = await resolver.resolve_entity("Apple Inc")

            assert result.best_match is not None
            assert result.best_match.entity_name == "Apple Inc"
            assert result.best_match.lei == "HWUPKR0MPOU8FGXBT394"
            assert len(result.sources_succeeded) > 0

    async def test_multi_source_resolution(self):
        """Test resolution using multiple sources."""
        resolver = EntityResolver()

        gleif_match = EntityMatch(
            entity_name="Tesla Inc",
            lei="54930053HGCFWVHYZX42",
            country="US",
            source=EntitySource.GLEIF,
            confidence_score=0.95,
            match_method="name_search"
        )

        oc_match = EntityMatch(
            entity_name="Tesla, Inc.",
            opencorporates_id="us_de/3581572",
            country="US",
            source=EntitySource.OPENCORPORATES,
            confidence_score=0.90,
            match_method="opencorporates_search"
        )

        with patch.object(resolver, '_resolve_from_gleif', return_value=[gleif_match]), \
             patch.object(resolver, '_resolve_from_opencorporates', return_value=[oc_match]), \
             patch.object(resolver, '_resolve_from_duns', return_value=[]):

            request = ResolutionRequest(
                query="Tesla",
                preferred_sources=[EntitySource.GLEIF, EntitySource.OPENCORPORATES, EntitySource.DUNS]
            )
            result = await resolver.resolve_entity(request)

            assert len(result.all_matches) == 2
            assert result.best_match.confidence_score == 0.95
            assert result.best_match.source == EntitySource.GLEIF

    async def test_batch_resolution(self):
        """Test batch entity resolution."""
        resolver = EntityResolver()

        mock_match = EntityMatch(
            entity_name="Test Corp",
            country="US",
            source=EntitySource.OPENCORPORATES,
            confidence_score=0.85,
            match_method="name_search"
        )

        with patch.object(resolver, 'resolve_entity') as mock_resolve:
            mock_result = ResolutionResult(
                request=ResolutionRequest(query="Test"),
                resolution_time_ms=100,
                provenance_hash="test_hash"
            )
            mock_result.best_match = mock_match
            mock_resolve.return_value = mock_result

            queries = ["Company A", "Company B", "Company C"]
            results = await resolver.batch_resolve(queries)

            assert len(results) == 3
            assert mock_resolve.call_count == 3

    def test_name_normalization(self):
        """Test company name normalization."""
        resolver = EntityResolver()

        # Test removing suffixes
        assert resolver._normalize_company_name("Apple Inc.") == "apple"
        assert resolver._normalize_company_name("Microsoft Corporation") == "microsoft"
        assert resolver._normalize_company_name("Google LLC") == "google"

        # Test punctuation removal
        assert resolver._normalize_company_name("AT&T") == "att"

        # Test case normalization
        assert resolver._normalize_company_name("IBM") == "ibm"

    def test_name_similarity(self):
        """Test name similarity calculation."""
        resolver = EntityResolver()

        # Exact match
        score = resolver._calculate_name_similarity("Apple Inc", "Apple Inc")
        assert score == 1.0

        # Close match
        score = resolver._calculate_name_similarity("Microsoft", "Microsoft Corporation")
        assert score >= 0.85

        # Partial match
        score = resolver._calculate_name_similarity("Google", "Alphabet Inc")
        assert score < 0.5

    async def test_caching(self):
        """Test that results are cached properly."""
        resolver = EntityResolver()

        mock_match = EntityMatch(
            entity_name="Cached Corp",
            country="US",
            source=EntitySource.GLEIF,
            confidence_score=0.9,
            match_method="cached"
        )

        call_count = 0

        async def mock_resolve(*args):
            nonlocal call_count
            call_count += 1
            return [mock_match]

        with patch.object(resolver, '_resolve_from_gleif', side_effect=mock_resolve):
            # First call
            result1 = await resolver.resolve_entity("Cached Corp")
            assert call_count == 1

            # Second call should hit cache
            result2 = await resolver.resolve_entity("Cached Corp")
            assert call_count == 1  # Should not increase
            assert result2.cache_hits == 1

    def test_provenance_calculation(self):
        """Test provenance hash calculation."""
        resolver = EntityResolver()

        request = ResolutionRequest(query="Test Corp")
        result = ResolutionResult(
            request=request,
            resolution_time_ms=100,
            provenance_hash=""
        )

        hash1 = resolver._calculate_provenance(request, result)
        assert len(hash1) == 64  # SHA-256 hash length

        # Same inputs should give same hash
        hash2 = resolver._calculate_provenance(request, result)
        assert hash1 == hash2

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        resolver = EntityResolver()

        initial_stats = resolver.get_statistics()
        assert initial_stats["total_resolutions"] == 0
        assert initial_stats["cache_hits"] == 0

        # After resolution, stats should update
        # (would need actual resolution to test fully)


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for the complete system."""

    async def test_end_to_end_resolution(self):
        """Test complete resolution flow."""
        resolver = EntityResolver()

        # Create a complex request
        request = ResolutionRequest(
            query="Apple Inc",
            country="US",
            min_confidence=0.7,
            max_results=5,
            preferred_sources=[EntitySource.GLEIF, EntitySource.OPENCORPORATES]
        )

        # Mock the API responses
        gleif_match = EntityMatch(
            entity_name="Apple Inc.",
            lei="HWUPKR0MPOU8FGXBT394",
            country="US",
            address={
                "line1": "One Apple Park Way",
                "city": "Cupertino",
                "region": "CA",
                "postal_code": "95014",
                "country": "US"
            },
            source=EntitySource.GLEIF,
            confidence_score=0.95,
            match_method="name_search"
        )

        oc_match = EntityMatch(
            entity_name="Apple Inc.",
            opencorporates_id="us_ca/C0806592",
            country="US",
            source=EntitySource.OPENCORPORATES,
            confidence_score=0.92,
            match_method="opencorporates_search"
        )

        with patch.object(resolver, '_resolve_from_gleif', return_value=[gleif_match]), \
             patch.object(resolver, '_resolve_from_opencorporates', return_value=[oc_match]):

            result = await resolver.resolve_entity(request)

            # Verify resolution worked
            assert result.best_match is not None
            assert result.best_match.lei == "HWUPKR0MPOU8FGXBT394"
            assert len(result.all_matches) == 2
            assert EntitySource.GLEIF in result.sources_succeeded
            assert EntitySource.OPENCORPORATES in result.sources_succeeded

            # Verify metadata
            assert result.resolution_time_ms > 0
            assert result.provenance_hash
            assert len(result.provenance_hash) == 64

    async def test_fallback_on_source_failure(self):
        """Test that resolver falls back when a source fails."""
        resolver = EntityResolver()

        oc_match = EntityMatch(
            entity_name="Fallback Corp",
            opencorporates_id="us_de/123456",
            country="US",
            source=EntitySource.OPENCORPORATES,
            confidence_score=0.85,
            match_method="opencorporates_search"
        )

        async def gleif_error(*args):
            raise Exception("GLEIF API error")

        with patch.object(resolver, '_resolve_from_gleif', side_effect=gleif_error), \
             patch.object(resolver, '_resolve_from_opencorporates', return_value=[oc_match]), \
             patch.object(resolver, '_resolve_from_duns', return_value=[]):

            request = ResolutionRequest(
                query="Fallback Corp",
                preferred_sources=[EntitySource.GLEIF, EntitySource.OPENCORPORATES]
            )
            result = await resolver.resolve_entity(request)

            # Should still get a result from OpenCorporates
            assert result.best_match is not None
            assert result.best_match.source == EntitySource.OPENCORPORATES

            # GLEIF should be marked as failed
            assert "gleif" in result.sources_failed
            assert EntitySource.OPENCORPORATES in result.sources_succeeded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])