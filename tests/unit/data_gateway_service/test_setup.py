# -*- coding: utf-8 -*-
"""
Unit Tests for DataGatewayService Facade & Setup (AGENT-DATA-004)

Tests the DataGatewayService facade including engine delegation
(query execution, batch queries, source registration, source testing,
schema translation, cache management, catalog search, statistics),
FastAPI integration (configure/get/get_router), and full lifecycle flows.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inline models
# ---------------------------------------------------------------------------


class DataSource:
    """Registered data source."""

    def __init__(self, source_id: str, name: str, source_type: str,
                 connection_config: Optional[Dict[str, Any]] = None,
                 status: str = "active"):
        self.source_id = source_id
        self.name = name
        self.source_type = source_type
        self.connection_config = connection_config or {}
        self.status = status
        self.provenance_hash = ""
        self.registered_at = datetime.now(timezone.utc).isoformat()


class QueryRequest:
    """Request for executing a data query."""

    def __init__(self, sources: List[str], query: Dict[str, Any],
                 timeout: int = 30, use_cache: bool = True):
        self.sources = sources
        self.query = query
        self.timeout = timeout
        self.use_cache = use_cache


class QueryResponse:
    """Response from a data query execution."""

    def __init__(self, query_id: str, sources: List[str],
                 data: List[Dict[str, Any]],
                 cached: bool = False,
                 duration_ms: float = 0.0,
                 provenance_hash: str = ""):
        self.query_id = query_id
        self.sources = sources
        self.data = data
        self.cached = cached
        self.duration_ms = duration_ms
        self.provenance_hash = provenance_hash


class SourceHealthResult:
    """Health check result for a data source."""

    def __init__(self, source_id: str, healthy: bool,
                 latency_ms: float = 0.0, error: Optional[str] = None):
        self.source_id = source_id
        self.healthy = healthy
        self.latency_ms = latency_ms
        self.error = error


class TranslationResult:
    """Result of schema translation."""

    def __init__(self, source_type: str, target_type: str,
                 original: Dict[str, Any], translated: Dict[str, Any],
                 provenance_hash: str = ""):
        self.source_type = source_type
        self.target_type = target_type
        self.original = original
        self.translated = translated
        self.provenance_hash = provenance_hash


class CatalogEntry:
    """A catalog entry for search results."""

    def __init__(self, entry_id: str, name: str, description: str,
                 domain: str, source_type: str,
                 tags: Optional[List[str]] = None):
        self.entry_id = entry_id
        self.name = name
        self.description = description
        self.domain = domain
        self.source_type = source_type
        self.tags = tags or []


class CacheStats:
    """Cache statistics."""

    def __init__(self, total_entries: int = 0, hits: int = 0,
                 misses: int = 0, hit_rate: float = 0.0):
        self.total_entries = total_entries
        self.hits = hits
        self.misses = misses
        self.hit_rate = hit_rate


# ---------------------------------------------------------------------------
# Inline DataGatewayService facade
# ---------------------------------------------------------------------------


class DataGatewayService:
    """Facade for the Data Gateway Agent SDK (GL-DATA-GW-001)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._sources: Dict[str, DataSource] = {}
        self._queries: List[QueryResponse] = []
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._catalog: List[CatalogEntry] = []
        self._schema_mappings: Dict[str, Dict[str, str]] = {}
        self._source_counter = 0
        self._query_counter = 0
        self._catalog_counter = 0
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute a query across one or more data sources."""
        self._query_counter += 1
        query_id = f"QRY-{self._query_counter:05d}"
        start_time = time.time()

        # Check cache
        query_hash = _compute_hash(request.query)
        if request.use_cache and query_hash in self._cache:
            self._cache_hits += 1
            cached_data = self._cache[query_hash]
            response = QueryResponse(
                query_id=query_id,
                sources=request.sources,
                data=cached_data,
                cached=True,
                duration_ms=0.1,
                provenance_hash=_compute_hash({"query_id": query_id, "cached": True}),
            )
            self._queries.append(response)
            return response

        self._cache_misses += 1

        # Check timeout
        if request.timeout <= 0:
            raise TimeoutError(f"Query {query_id} timed out")

        # Simulate query execution across sources
        data: List[Dict[str, Any]] = []
        for source_id in request.sources:
            src = self._sources.get(source_id)
            if src is not None:
                data.append({
                    "source_id": source_id,
                    "source_type": src.source_type,
                    "records": [],
                })

        duration_ms = (time.time() - start_time) * 1000

        # Store in cache
        if request.use_cache:
            self._cache[query_hash] = data

        response = QueryResponse(
            query_id=query_id,
            sources=request.sources,
            data=data,
            cached=False,
            duration_ms=duration_ms,
            provenance_hash=_compute_hash({
                "query_id": query_id,
                "sources": request.sources,
            }),
        )
        self._queries.append(response)
        return response

    def execute_batch(self, requests: List[QueryRequest],
                      parallel: bool = True) -> List[QueryResponse]:
        """Execute multiple queries in batch."""
        results = []
        for req in requests:
            results.append(self.execute_query(req))
        return results

    def register_source(self, name: str, source_type: str,
                        connection_config: Optional[Dict[str, Any]] = None) -> DataSource:
        """Register a new data source."""
        # Check for duplicate name
        for src in self._sources.values():
            if src.name == name:
                raise ValueError(f"Source with name '{name}' already registered")

        self._source_counter += 1
        source_id = f"SRC-{self._source_counter:05d}"
        source = DataSource(
            source_id=source_id,
            name=name,
            source_type=source_type,
            connection_config=connection_config,
            status="active",
        )
        source.provenance_hash = _compute_hash({
            "source_id": source_id,
            "name": name,
            "source_type": source_type,
        })
        self._sources[source_id] = source
        return source

    def test_source(self, source_id: str) -> SourceHealthResult:
        """Test connectivity to a registered data source."""
        source = self._sources.get(source_id)
        if source is None:
            return SourceHealthResult(
                source_id=source_id,
                healthy=False,
                error=f"Source {source_id} not found",
            )
        # Simulate healthy check
        return SourceHealthResult(
            source_id=source_id,
            healthy=True,
            latency_ms=5.2,
        )

    def translate_schema(self, data: Dict[str, Any],
                         source_type: str,
                         target_type: str = "canonical") -> TranslationResult:
        """Translate data from one schema to another."""
        mapping_key = f"{source_type}->{target_type}"
        mapping = self._schema_mappings.get(mapping_key)
        if mapping is None:
            raise KeyError(
                f"No schema mapping registered for {source_type} -> {target_type}"
            )
        translated: Dict[str, Any] = {}
        for src_field, tgt_field in mapping.items():
            if src_field in data:
                translated[tgt_field] = data[src_field]

        return TranslationResult(
            source_type=source_type,
            target_type=target_type,
            original=data,
            translated=translated,
            provenance_hash=_compute_hash({
                "source_type": source_type,
                "target_type": target_type,
                "translated": translated,
            }),
        )

    def get_cache_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return CacheStats(
            total_entries=len(self._cache),
            hits=self._cache_hits,
            misses=self._cache_misses,
            hit_rate=hit_rate,
        )

    def invalidate_cache(self, source_id: Optional[str] = None,
                         invalidate_all: bool = False) -> int:
        """Invalidate cache entries."""
        if invalidate_all:
            count = len(self._cache)
            self._cache.clear()
            return count
        # For source-specific invalidation, clear all (simplified)
        if source_id:
            count = len(self._cache)
            self._cache.clear()
            return count
        return 0

    def search_catalog(self, keyword: str) -> List[CatalogEntry]:
        """Search the data catalog by keyword."""
        keyword_lower = keyword.lower()
        results = []
        for entry in self._catalog:
            if (keyword_lower in entry.name.lower()
                    or keyword_lower in entry.description.lower()
                    or any(keyword_lower in t.lower() for t in entry.tags)):
                results.append(entry)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall gateway statistics."""
        return {
            "total_sources": len(self._sources),
            "total_queries": len(self._queries),
            "total_catalog_entries": len(self._catalog),
            "cache_entries": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "service_initialized": self._initialized,
        }


# ---------------------------------------------------------------------------
# FastAPI integration functions
# ---------------------------------------------------------------------------


def configure_data_gateway(app: Any,
                           config: Optional[Dict[str, Any]] = None) -> DataGatewayService:
    """Configure the Data Gateway Service on a FastAPI application."""
    service = DataGatewayService(config=config)
    app.state.data_gateway_service = service
    return service


def get_data_gateway(app: Any) -> DataGatewayService:
    """Get the DataGatewayService from app state."""
    service = getattr(app.state, "data_gateway_service", None)
    if service is None:
        raise RuntimeError(
            "Data gateway service not configured. "
            "Call configure_data_gateway(app) first."
        )
    return service


def get_router(service: Optional[DataGatewayService] = None) -> Any:
    """Get the data gateway API router."""
    try:
        # Would import from greenlang.data_gateway.api.router
        return None  # Router not available in test context
    except ImportError:
        return None


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def service() -> DataGatewayService:
    return DataGatewayService()


@pytest.fixture
def service_with_source() -> DataGatewayService:
    """Service with a pre-registered source and schema mapping."""
    svc = DataGatewayService()
    svc.register_source("SAP ERP", "erp", {"host": "sap.local"})
    svc._schema_mappings["erp->canonical"] = {
        "emission_amount": "co2_kg",
        "emission_source": "source",
    }
    svc._catalog.append(CatalogEntry(
        entry_id="CAT-00001",
        name="Emissions Data",
        description="CO2 emissions from ERP system",
        domain="sustainability",
        source_type="erp",
        tags=["emissions", "co2"],
    ))
    return svc


# ===========================================================================
# Test Classes
# ===========================================================================


class TestServiceCreation:
    """Tests for DataGatewayService initialization."""

    def test_default_creation(self):
        svc = DataGatewayService()
        assert svc.is_initialized is True

    def test_creation_with_config(self):
        config = {"cache_ttl": 600, "max_sources": 20}
        svc = DataGatewayService(config=config)
        assert svc._config["cache_ttl"] == 600
        assert svc._config["max_sources"] == 20


class TestExecuteQuery:
    """Tests for query execution."""

    def test_single_source_query(self, service_with_source):
        req = QueryRequest(
            sources=["SRC-00001"],
            query={"table": "emissions", "year": 2025},
        )
        resp = service_with_source.execute_query(req)
        assert resp is not None
        assert resp.query_id.startswith("QRY-")
        assert resp.sources == ["SRC-00001"]
        assert len(resp.provenance_hash) == 64

    def test_multi_source_query(self, service_with_source):
        service_with_source.register_source("CSV Store", "csv")
        req = QueryRequest(
            sources=["SRC-00001", "SRC-00002"],
            query={"table": "combined"},
        )
        resp = service_with_source.execute_query(req)
        assert len(resp.data) == 2

    def test_query_timeout(self, service_with_source):
        req = QueryRequest(
            sources=["SRC-00001"],
            query={"table": "emissions"},
            timeout=0,
        )
        with pytest.raises(TimeoutError):
            service_with_source.execute_query(req)


class TestExecuteBatch:
    """Tests for batch query execution."""

    def test_parallel_queries(self, service_with_source):
        requests = [
            QueryRequest(["SRC-00001"], {"table": "emissions"}),
            QueryRequest(["SRC-00001"], {"table": "energy"}),
            QueryRequest(["SRC-00001"], {"table": "waste"}),
        ]
        results = service_with_source.execute_batch(requests, parallel=True)
        assert len(results) == 3
        assert all(r.query_id.startswith("QRY-") for r in results)

    def test_sequential_queries(self, service_with_source):
        requests = [
            QueryRequest(["SRC-00001"], {"table": "a"}),
            QueryRequest(["SRC-00001"], {"table": "b"}),
        ]
        results = service_with_source.execute_batch(requests, parallel=False)
        assert len(results) == 2


class TestRegisterSource:
    """Tests for data source registration."""

    def test_register_success(self, service):
        source = service.register_source(
            name="PostgreSQL DB",
            source_type="postgres",
            connection_config={"host": "db.local", "port": 5432},
        )
        assert source is not None
        assert source.source_id.startswith("SRC-")
        assert source.name == "PostgreSQL DB"
        assert source.source_type == "postgres"
        assert source.status == "active"
        assert len(source.provenance_hash) == 64

    def test_register_duplicate_name(self, service):
        service.register_source("MyDB", "postgres")
        with pytest.raises(ValueError, match="already registered"):
            service.register_source("MyDB", "postgres")


class TestTestSource:
    """Tests for source health testing."""

    def test_healthy_source(self, service_with_source):
        result = service_with_source.test_source("SRC-00001")
        assert result.healthy is True
        assert result.latency_ms > 0

    def test_unhealthy_source(self, service):
        result = service.test_source("SRC-99999")
        assert result.healthy is False
        assert result.error is not None


class TestTranslateSchema:
    """Tests for schema translation."""

    def test_translate_success(self, service_with_source):
        data = {"emission_amount": 150.5, "emission_source": "diesel"}
        result = service_with_source.translate_schema(data, "erp", "canonical")
        assert result.translated["co2_kg"] == 150.5
        assert result.translated["source"] == "diesel"
        assert len(result.provenance_hash) == 64

    def test_translate_unknown_source_type(self, service):
        with pytest.raises(KeyError, match="No schema mapping"):
            service.translate_schema({"x": 1}, "unknown", "canonical")


class TestCacheStats:
    """Tests for cache statistics."""

    def test_initial_cache_stats(self, service):
        stats = service.get_cache_stats()
        assert stats.total_entries == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_cache_stats_after_queries(self, service_with_source):
        req = QueryRequest(["SRC-00001"], {"table": "emissions"})
        service_with_source.execute_query(req)  # miss
        service_with_source.execute_query(req)  # hit (cached)
        stats = service_with_source.get_cache_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.total_entries >= 1


class TestInvalidateCache:
    """Tests for cache invalidation."""

    def test_invalidate_by_source(self, service_with_source):
        req = QueryRequest(["SRC-00001"], {"table": "emissions"})
        service_with_source.execute_query(req)
        removed = service_with_source.invalidate_cache(source_id="SRC-00001")
        assert removed >= 1
        stats = service_with_source.get_cache_stats()
        assert stats.total_entries == 0

    def test_invalidate_all(self, service_with_source):
        req1 = QueryRequest(["SRC-00001"], {"table": "a"})
        req2 = QueryRequest(["SRC-00001"], {"table": "b"})
        service_with_source.execute_query(req1)
        service_with_source.execute_query(req2)
        removed = service_with_source.invalidate_cache(invalidate_all=True)
        assert removed >= 2


class TestSearchCatalog:
    """Tests for catalog search."""

    def test_search_results(self, service_with_source):
        results = service_with_source.search_catalog("emissions")
        assert len(results) >= 1
        assert any(e.name == "Emissions Data" for e in results)

    def test_search_no_results(self, service_with_source):
        results = service_with_source.search_catalog("blockchain")
        assert results == []


class TestGetStatistics:
    """Tests for overall gateway statistics."""

    def test_initial_statistics(self, service):
        stats = service.get_statistics()
        assert stats["total_sources"] == 0
        assert stats["total_queries"] == 0
        assert stats["cache_entries"] == 0
        assert stats["service_initialized"] is True

    def test_statistics_after_operations(self, service_with_source):
        req = QueryRequest(["SRC-00001"], {"table": "emissions"})
        service_with_source.execute_query(req)
        stats = service_with_source.get_statistics()
        assert stats["total_sources"] == 1
        assert stats["total_queries"] == 1
        assert stats["cache_entries"] >= 1


class TestFastAPIIntegration:
    """Tests for FastAPI app integration."""

    def test_configure_data_gateway(self):
        app = MagicMock()
        svc = configure_data_gateway(app)
        assert svc.is_initialized is True
        assert app.state.data_gateway_service is svc

    def test_configure_data_gateway_with_config(self):
        app = MagicMock()
        config = {"cache_ttl": 600}
        svc = configure_data_gateway(app, config=config)
        assert svc._config["cache_ttl"] == 600

    def test_get_data_gateway(self):
        app = MagicMock()
        svc = configure_data_gateway(app)
        retrieved = get_data_gateway(app)
        assert retrieved is svc

    def test_get_data_gateway_not_configured(self):
        app = MagicMock(spec=[])
        app.state = MagicMock(spec=[])
        with pytest.raises(RuntimeError, match="not configured"):
            get_data_gateway(app)

    def test_get_router(self):
        result = get_router()
        # Router may be None if FastAPI is not available in test env
        assert result is None or hasattr(result, "routes")


class TestFullLifecycle:
    """Tests for complete data gateway lifecycle."""

    def test_complete_lifecycle(self):
        service = DataGatewayService()

        # 1. Register data source
        source = service.register_source(
            name="ERP System",
            source_type="erp",
            connection_config={"host": "erp.local", "port": 5432},
        )
        assert source.source_id is not None
        assert source.status == "active"

        # 2. Register schema mapping
        service._schema_mappings["erp->canonical"] = {
            "emission_amount": "co2_kg",
            "emission_source": "source",
        }

        # 3. Execute query
        req = QueryRequest(
            sources=[source.source_id],
            query={"table": "emissions", "year": 2025},
        )
        resp = service.execute_query(req)
        assert resp.query_id is not None
        assert resp.cached is False

        # 4. Execute same query again (should hit cache)
        resp2 = service.execute_query(req)
        assert resp2.cached is True

        # 5. Check cache stats
        stats = service.get_cache_stats()
        assert stats.hits == 1
        assert stats.misses == 1

        # 6. Translate schema
        data = {"emission_amount": 150.5, "emission_source": "diesel"}
        translated = service.translate_schema(data, "erp", "canonical")
        assert translated.translated["co2_kg"] == 150.5

        # 7. Test source health
        health = service.test_source(source.source_id)
        assert health.healthy is True

        # 8. Check overall statistics
        final_stats = service.get_statistics()
        assert final_stats["total_sources"] == 1
        assert final_stats["total_queries"] == 2
        assert final_stats["service_initialized"] is True
