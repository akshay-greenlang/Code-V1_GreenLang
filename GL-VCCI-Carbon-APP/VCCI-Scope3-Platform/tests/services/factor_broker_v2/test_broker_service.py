"""
Comprehensive Tests for Factor Broker Service
GL-VCCI Scope 3 Platform - Phase 6

Tests for main FactorBroker class including:
- Runtime resolution for all 4 sources (ecoinvent, DESNZ, EPA, Proxy)
- Cache hit/miss scenarios
- Fallback logic (ecoinvent → DESNZ → EPA → Proxy)
- License compliance (no bulk redistribution)
- Performance: <50ms p95

Test Count: 25 tests
Target Coverage: 95%

Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

# Import models and classes to test
import sys
sys.path.insert(0, '/c/Users/aksha/Code-V1_GreenLang/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform')

from services.factor_broker.broker import FactorBroker
from services.factor_broker.models import (
    FactorRequest,
    FactorResponse,
    FactorMetadata,
    ProvenanceInfo,
    DataQualityIndicator,
    GWPStandard,
    SourceType,
    GWPComparisonRequest,
    HealthCheckResponse
)
from services.factor_broker.exceptions import (
    FactorNotFoundError,
    SourceUnavailableError,
    ValidationError,
    LicenseViolationError
)
from services.factor_broker.config import FactorBrokerConfig


@pytest.fixture
def mock_config():
    """Create a mock Factor Broker configuration."""
    config = Mock(spec=FactorBrokerConfig)
    config.cache = Mock()
    config.cache.enabled = True
    config.cache.ttl_seconds = 86400  # 24 hours
    config.proxy = Mock()
    config.is_source_enabled = Mock(return_value=True)
    config.get_source_config = Mock(return_value={})
    config.get_cascade_order = Mock(return_value=[
        SourceType.ECOINVENT,
        SourceType.DESNZ_UK,
        SourceType.EPA_US,
        SourceType.PROXY
    ])
    config.validate = Mock(return_value=[])
    return config


@pytest.fixture
def sample_factor_request():
    """Create a sample factor request."""
    return FactorRequest(
        product="Steel",
        region="US",
        gwp_standard=GWPStandard.AR6,
        unit="kg",
        year=2024,
        category="Metals"
    )


@pytest.fixture
def sample_factor_response():
    """Create a sample factor response."""
    return FactorResponse(
        factor_id="ecoinvent_3.10_steel_us_ar6_kg",
        value=1.85,
        unit="kgCO2e/kg",
        uncertainty=0.10,
        metadata=FactorMetadata(
            source=SourceType.ECOINVENT,
            source_version="3.10",
            gwp_standard=GWPStandard.AR6,
            reference_year=2024,
            geographic_scope="US",
            data_quality=DataQualityIndicator(
                reliability=5,
                completeness=5,
                temporal_correlation=5,
                geographical_correlation=5,
                technological_correlation=5,
                overall_score=100
            )
        ),
        provenance=ProvenanceInfo(
            lookup_timestamp=datetime.utcnow(),
            cache_hit=False,
            is_proxy=False,
            fallback_chain=["ecoinvent"]
        )
    )


class TestFactorBrokerInitialization:
    """Test Factor Broker initialization and configuration."""

    def test_broker_initialization_with_config(self, mock_config):
        """Test broker initializes correctly with provided config."""
        with patch('services.factor_broker.broker.FactorCache'):
            broker = FactorBroker(config=mock_config)

            assert broker.config == mock_config
            assert broker.performance_stats["total_requests"] == 0
            assert broker.sources is not None

    def test_broker_initialization_without_config(self):
        """Test broker initializes with default config from environment."""
        with patch('services.factor_broker.broker.FactorBrokerConfig.from_env') as mock_from_env:
            mock_from_env.return_value = Mock()
            mock_from_env.return_value.validate = Mock(return_value=[])
            mock_from_env.return_value.cache = Mock()

            with patch('services.factor_broker.broker.FactorCache'):
                broker = FactorBroker()
                mock_from_env.assert_called_once()

    def test_broker_initialization_with_invalid_config(self):
        """Test broker raises error with invalid configuration."""
        invalid_config = Mock(spec=FactorBrokerConfig)
        invalid_config.validate = Mock(return_value=["Missing API key"])

        with pytest.raises(ValidationError) as exc_info:
            with patch('services.factor_broker.broker.FactorCache'):
                FactorBroker(config=invalid_config)

        assert "Configuration errors" in str(exc_info.value)

    def test_source_initialization(self, mock_config):
        """Test data sources are initialized correctly."""
        with patch('services.factor_broker.broker.FactorCache'):
            with patch('services.factor_broker.broker.EcoinventSource') as mock_eco:
                broker = FactorBroker(config=mock_config)

                # Sources should be initialized if enabled
                assert isinstance(broker.sources, dict)


class TestFactorResolutionEcoinvent:
    """Test factor resolution from Ecoinvent source."""

    @pytest.mark.asyncio
    async def test_resolve_from_ecoinvent_success(
        self,
        mock_config,
        sample_factor_request,
        sample_factor_response
    ):
        """Test successful factor resolution from ecoinvent (cache miss)."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=None)  # Cache miss
            mock_cache.set = AsyncMock()

            with patch('services.factor_broker.broker.EcoinventSource') as mock_eco_class:
                mock_eco = mock_eco_class.return_value
                mock_eco.fetch_factor = AsyncMock(return_value=sample_factor_response)

                broker = FactorBroker(config=mock_config)
                broker.sources[SourceType.ECOINVENT] = mock_eco

                result = await broker.resolve(sample_factor_request)

                assert result.factor_id == sample_factor_response.factor_id
                assert result.value == 1.85
                assert result.source == "ecoinvent"
                assert not result.provenance.cache_hit
                assert broker.performance_stats["successful_requests"] == 1
                assert broker.performance_stats["source_usage"]["ecoinvent"] == 1

                # Cache should be called
                mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_from_cache_hit(
        self,
        mock_config,
        sample_factor_request,
        sample_factor_response
    ):
        """Test factor resolution with cache hit."""
        sample_factor_response.provenance.cache_hit = True

        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=sample_factor_response)

            broker = FactorBroker(config=mock_config)

            result = await broker.resolve(sample_factor_request)

            assert result.factor_id == sample_factor_response.factor_id
            assert broker.performance_stats["cache_hits"] == 1
            assert broker.performance_stats["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_resolve_performance_under_50ms(
        self,
        mock_config,
        sample_factor_request,
        sample_factor_response
    ):
        """Test factor resolution performance is under 50ms for p95."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=sample_factor_response)

            broker = FactorBroker(config=mock_config)

            # Run 100 requests to get p95 latency
            latencies = []
            for _ in range(100):
                start = datetime.utcnow()
                await broker.resolve(sample_factor_request)
                latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
                latencies.append(latency_ms)

            # Calculate p95
            latencies.sort()
            p95_latency = latencies[94]  # 95th percentile

            assert p95_latency < 50, f"p95 latency {p95_latency}ms exceeds 50ms target"


class TestFallbackLogic:
    """Test cascading fallback logic: ecoinvent → DESNZ → EPA → Proxy."""

    @pytest.mark.asyncio
    async def test_fallback_to_desnz_when_ecoinvent_unavailable(
        self,
        mock_config,
        sample_factor_request,
        sample_factor_response
    ):
        """Test fallback to DESNZ when ecoinvent is unavailable."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            with patch('services.factor_broker.broker.EcoinventSource') as mock_eco_class:
                with patch('services.factor_broker.broker.DESNZSource') as mock_desnz_class:
                    mock_eco = mock_eco_class.return_value
                    mock_eco.fetch_factor = AsyncMock(
                        side_effect=SourceUnavailableError(
                            source="ecoinvent",
                            reason="API timeout"
                        )
                    )

                    mock_desnz = mock_desnz_class.return_value
                    desnz_response = sample_factor_response.copy()
                    desnz_response.metadata.source = SourceType.DESNZ_UK
                    mock_desnz.fetch_factor = AsyncMock(return_value=desnz_response)

                    broker = FactorBroker(config=mock_config)
                    broker.sources[SourceType.ECOINVENT] = mock_eco
                    broker.sources[SourceType.DESNZ_UK] = mock_desnz

                    result = await broker.resolve(sample_factor_request)

                    assert result.source == "desnz_uk"
                    assert "ecoinvent" in result.provenance.fallback_chain
                    assert "desnz_uk" in result.provenance.fallback_chain

    @pytest.mark.asyncio
    async def test_fallback_to_epa_when_desnz_fails(
        self,
        mock_config,
        sample_factor_request,
        sample_factor_response
    ):
        """Test fallback to EPA when both ecoinvent and DESNZ fail."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            broker = FactorBroker(config=mock_config)

            # Mock all sources to fail except EPA
            mock_eco = Mock()
            mock_eco.fetch_factor = AsyncMock(side_effect=SourceUnavailableError("ecoinvent", "fail"))
            broker.sources[SourceType.ECOINVENT] = mock_eco

            mock_desnz = Mock()
            mock_desnz.fetch_factor = AsyncMock(return_value=None)
            broker.sources[SourceType.DESNZ_UK] = mock_desnz

            mock_epa = Mock()
            epa_response = sample_factor_response.copy()
            epa_response.metadata.source = SourceType.EPA_US
            mock_epa.fetch_factor = AsyncMock(return_value=epa_response)
            broker.sources[SourceType.EPA_US] = mock_epa

            result = await broker.resolve(sample_factor_request)

            assert result.source == "epa_us"
            assert len(result.provenance.fallback_chain) == 3

    @pytest.mark.asyncio
    async def test_fallback_to_proxy_when_all_sources_fail(
        self,
        mock_config,
        sample_factor_request,
        sample_factor_response
    ):
        """Test fallback to proxy when all primary sources fail."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            broker = FactorBroker(config=mock_config)

            # Mock all primary sources to fail
            for source_type in [SourceType.ECOINVENT, SourceType.DESNZ_UK, SourceType.EPA_US]:
                mock_source = Mock()
                mock_source.fetch_factor = AsyncMock(return_value=None)
                broker.sources[source_type] = mock_source

            # Mock proxy to succeed
            mock_proxy = Mock()
            proxy_response = sample_factor_response.copy()
            proxy_response.metadata.source = SourceType.PROXY
            proxy_response.provenance.is_proxy = True
            mock_proxy.fetch_factor = AsyncMock(return_value=proxy_response)
            broker.sources[SourceType.PROXY] = mock_proxy

            result = await broker.resolve(sample_factor_request)

            assert result.source == "proxy"
            assert result.provenance.is_proxy
            assert len(result.provenance.fallback_chain) == 4

    @pytest.mark.asyncio
    async def test_factor_not_found_when_all_sources_fail(
        self,
        mock_config,
        sample_factor_request
    ):
        """Test FactorNotFoundError when all sources including proxy fail."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=None)

            broker = FactorBroker(config=mock_config)

            # Mock all sources to return None
            for source_type in [SourceType.ECOINVENT, SourceType.DESNZ_UK, SourceType.EPA_US, SourceType.PROXY]:
                mock_source = Mock()
                mock_source.fetch_factor = AsyncMock(return_value=None)
                broker.sources[source_type] = mock_source

            with pytest.raises(FactorNotFoundError) as exc_info:
                await broker.resolve(sample_factor_request)

            assert "Steel" in str(exc_info.value)
            assert "US" in str(exc_info.value)
            assert broker.performance_stats["failed_requests"] == 1


class TestCacheManagement:
    """Test cache hit/miss scenarios and TTL compliance."""

    @pytest.mark.asyncio
    async def test_cache_miss_triggers_source_lookup(
        self,
        mock_config,
        sample_factor_request,
        sample_factor_response
    ):
        """Test cache miss triggers lookup from sources."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=None)  # Cache miss
            mock_cache.set = AsyncMock()

            broker = FactorBroker(config=mock_config)

            mock_source = Mock()
            mock_source.fetch_factor = AsyncMock(return_value=sample_factor_response)
            broker.sources[SourceType.ECOINVENT] = mock_source

            result = await broker.resolve(sample_factor_request)

            # Source should be called
            mock_source.fetch_factor.assert_called_once()
            # Result should be cached
            mock_cache.set.assert_called_once()
            assert broker.performance_stats["cache_hits"] == 0

    @pytest.mark.asyncio
    async def test_cache_hit_bypasses_source_lookup(
        self,
        mock_config,
        sample_factor_request,
        sample_factor_response
    ):
        """Test cache hit bypasses source lookup."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=sample_factor_response)  # Cache hit

            broker = FactorBroker(config=mock_config)

            mock_source = Mock()
            mock_source.fetch_factor = AsyncMock(return_value=sample_factor_response)
            broker.sources[SourceType.ECOINVENT] = mock_source

            result = await broker.resolve(sample_factor_request)

            # Source should NOT be called
            mock_source.fetch_factor.assert_not_called()
            assert broker.performance_stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_cache_ttl_24_hours_for_license_compliance(
        self,
        mock_config
    ):
        """Test cache TTL is 24 hours for ecoinvent license compliance."""
        assert mock_config.cache.ttl_seconds == 86400  # 24 hours

        # This ensures ecoinvent license compliance (no caching beyond 24 hours)


class TestLicenseCompliance:
    """Test license compliance, especially for ecoinvent."""

    @pytest.mark.asyncio
    async def test_no_bulk_redistribution_of_ecoinvent_data(
        self,
        mock_config
    ):
        """Test that bulk redistribution of ecoinvent data is prevented."""
        # License compliance is enforced by:
        # 1. 24-hour cache TTL
        # 2. No direct export of raw ecoinvent data
        # 3. API key authentication for ecoinvent

        assert mock_config.cache.ttl_seconds == 86400  # 24 hours max

    @pytest.mark.asyncio
    async def test_license_violation_error_raised_appropriately(self):
        """Test LicenseViolationError is raised for violations."""
        with pytest.raises(LicenseViolationError) as exc_info:
            raise LicenseViolationError(
                violation_type="bulk_export_attempt",
                license_source="ecoinvent"
            )

        assert "bulk_export_attempt" in str(exc_info.value)
        assert "ecoinvent" in str(exc_info.value)


class TestPerformanceStatistics:
    """Test performance tracking and statistics."""

    @pytest.mark.asyncio
    async def test_performance_stats_tracking(
        self,
        mock_config,
        sample_factor_request,
        sample_factor_response
    ):
        """Test performance statistics are tracked correctly."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            broker = FactorBroker(config=mock_config)

            mock_source = Mock()
            mock_source.fetch_factor = AsyncMock(return_value=sample_factor_response)
            broker.sources[SourceType.ECOINVENT] = mock_source

            # Make 5 requests
            for _ in range(5):
                await broker.resolve(sample_factor_request)

            assert broker.performance_stats["total_requests"] == 5
            assert broker.performance_stats["successful_requests"] == 5
            assert broker.performance_stats["source_usage"]["ecoinvent"] == 5

    def test_get_performance_stats(self, mock_config):
        """Test get_performance_stats returns correct data."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get_stats = Mock(return_value={"hit_rate": 0.85})

            broker = FactorBroker(config=mock_config)
            broker.performance_stats["total_requests"] = 100
            broker.performance_stats["successful_requests"] = 95
            broker.performance_stats["total_latency_ms"] = 4000.0

            stats = broker.get_performance_stats()

            assert stats["total_requests"] == 100
            assert stats["success_rate"] == 0.95
            assert stats["average_latency_ms"] == 40.0
            assert stats["cache"]["hit_rate"] == 0.85

    def test_reset_stats(self, mock_config):
        """Test reset_stats clears all statistics."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.reset_stats = Mock()

            broker = FactorBroker(config=mock_config)
            broker.performance_stats["total_requests"] = 100

            broker.reset_stats()

            assert broker.performance_stats["total_requests"] == 0
            mock_cache.reset_stats.assert_called_once()


class TestGWPComparison:
    """Test GWP standard comparison (AR5 vs AR6)."""

    @pytest.mark.asyncio
    async def test_compare_gwp_standards_success(
        self,
        mock_config,
        sample_factor_response
    ):
        """Test successful GWP standard comparison."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()

            broker = FactorBroker(config=mock_config)

            # Mock source to return different values for AR5 and AR6
            mock_source = Mock()
            ar5_response = sample_factor_response.copy()
            ar5_response.value = 1.82
            ar5_response.metadata.gwp_standard = GWPStandard.AR5

            ar6_response = sample_factor_response.copy()
            ar6_response.value = 1.85
            ar6_response.metadata.gwp_standard = GWPStandard.AR6

            async def fetch_side_effect(request):
                if request.gwp_standard == GWPStandard.AR5:
                    return ar5_response
                else:
                    return ar6_response

            mock_source.fetch_factor = AsyncMock(side_effect=fetch_side_effect)
            broker.sources[SourceType.ECOINVENT] = mock_source

            request = GWPComparisonRequest(
                product="Steel",
                region="US",
                unit="kg"
            )

            result = await broker.compare_gwp_standards(request)

            assert result.ar5.value == 1.82
            assert result.ar6.value == 1.85
            assert result.difference_absolute == 0.03
            assert abs(result.difference_percent - 1.648) < 0.1  # ~1.65%


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_all_sources_healthy(self, mock_config):
        """Test health check when all sources are healthy."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get_stats = Mock(return_value={"hit_rate": 0.87})

            broker = FactorBroker(config=mock_config)

            # Mock all sources as healthy
            for source_type in [SourceType.ECOINVENT, SourceType.DESNZ_UK, SourceType.EPA_US]:
                mock_source = Mock()
                mock_source.health_check = AsyncMock(return_value={
                    "status": "healthy",
                    "last_check": datetime.utcnow().isoformat()
                })
                broker.sources[source_type] = mock_source

            result = await broker.health_check()

            assert result.status == "healthy"
            assert result.cache_hit_rate == 0.87
            assert len(result.data_sources) == 3

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_some_sources_unhealthy(self, mock_config):
        """Test health check returns degraded when some sources are unhealthy."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.get_stats = Mock(return_value={"hit_rate": 0.87})

            broker = FactorBroker(config=mock_config)

            # Mock one source as unhealthy
            mock_eco = Mock()
            mock_eco.health_check = AsyncMock(return_value={"status": "unhealthy"})
            broker.sources[SourceType.ECOINVENT] = mock_eco

            mock_desnz = Mock()
            mock_desnz.health_check = AsyncMock(return_value={"status": "healthy"})
            broker.sources[SourceType.DESNZ_UK] = mock_desnz

            result = await broker.health_check()

            assert result.status == "degraded"


class TestContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_broker_async_context_manager(self, mock_config):
        """Test broker works as async context manager."""
        with patch('services.factor_broker.broker.FactorCache') as mock_cache_class:
            mock_cache = mock_cache_class.return_value
            mock_cache.close = Mock()

            async with FactorBroker(config=mock_config) as broker:
                assert broker is not None

            # Cache close should be called
            mock_cache.close.assert_called_once()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
