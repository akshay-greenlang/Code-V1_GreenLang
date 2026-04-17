# -*- coding: utf-8 -*-
"""Tests for Phase 10: Performance optimizer, index manager, search cache, regulatory tagger."""

from __future__ import annotations

import time

import pytest

from greenlang.factors.index_manager import (
    CATALOG_INDEXES,
    IndexDefinition,
    IndexManager,
    IndexStatus,
    IndexType,
)
from greenlang.factors.performance_optimizer import PerformanceOptimizer, PerformanceReport
from greenlang.factors.regulatory_tagger import (
    RegulatoryFramework,
    RegulatoryTagger,
    TaggingRule,
)
from greenlang.factors.search_cache import CacheStats, SearchCache


# ── PerformanceOptimizer ─────────────────────────────────────────────


class TestPerformanceOptimizer:
    def test_record_query(self):
        opt = PerformanceOptimizer()
        p = opt.record_query("SELECT * FROM factors WHERE id=1", 12.5, 1, used_index=True)
        assert p.query_hash
        assert p.execution_time_ms == 12.5

    def test_slow_query_detection(self):
        opt = PerformanceOptimizer(slow_threshold_ms=100.0)
        opt.record_query("SELECT * FROM factors", 150.0, 5000)
        report = opt.get_report()
        assert report.slow_query_count == 1

    def test_report_empty(self):
        opt = PerformanceOptimizer()
        report = opt.get_report()
        assert report.total_queries == 0

    def test_report_percentiles(self):
        opt = PerformanceOptimizer()
        for i in range(100):
            opt.record_query(f"q{i}", float(i), i)
        report = opt.get_report()
        assert report.total_queries == 100
        assert report.p95_latency_ms >= 90
        assert report.avg_latency_ms > 0

    def test_plan_cache(self):
        opt = PerformanceOptimizer()
        opt.cache_plan("SELECT 1", {"plan": "index_scan"})
        assert opt.get_cached_plan("SELECT 1") == {"plan": "index_scan"}
        assert opt.get_cached_plan("SELECT 2") is None

    def test_optimal_batch_size(self):
        assert PerformanceOptimizer.optimal_batch_size(10000, 500) > 0
        assert PerformanceOptimizer.optimal_batch_size(50, 500) == 50

    def test_report_to_dict(self):
        opt = PerformanceOptimizer()
        opt.record_query("q1", 10.0, 5)
        d = opt.get_report().to_dict()
        assert "total_queries" in d
        assert "recommendations" in d

    def test_clear(self):
        opt = PerformanceOptimizer()
        opt.record_query("q1", 10.0, 1)
        opt.cache_plan("q1", {})
        opt.clear()
        assert opt.get_report().total_queries == 0
        assert opt.get_cached_plan("q1") is None

    def test_eviction(self):
        opt = PerformanceOptimizer()
        opt.MAX_PROFILES = 10
        for i in range(20):
            opt.record_query(f"q{i}", float(i), 1)
        report = opt.get_report()
        assert report.total_queries == 10

    def test_recommendations_low_index_hit(self):
        opt = PerformanceOptimizer()
        for i in range(20):
            opt.record_query(f"q{i}", 10.0, 1, used_index=False)
        report = opt.get_report()
        assert any("Index hit" in r for r in report.recommendations)


# ── IndexManager ─────────────────────────────────────────────────────


class TestIndexManager:
    def test_default_indexes(self):
        mgr = IndexManager()
        indexes = mgr.list_indexes()
        assert len(indexes) == len(CATALOG_INDEXES)

    def test_add_custom_index(self):
        mgr = IndexManager()
        idx = IndexDefinition(name="idx_custom", table="test", columns=["col1"])
        mgr.add_index(idx)
        assert mgr.get_index("idx_custom") is not None

    def test_remove_index(self):
        mgr = IndexManager()
        removed = mgr.remove_index("idx_factors_source_id")
        assert removed is not None
        assert removed.status == IndexStatus.DROPPED

    def test_generate_create_sql(self):
        mgr = IndexManager()
        sqls = mgr.generate_create_sql()
        assert len(sqls) > 0
        assert all("CREATE" in s for s in sqls)

    def test_gin_index_sql(self):
        mgr = IndexManager()
        idx = mgr.get_index("idx_factors_search_gin")
        assert idx is not None
        sql = idx.to_sql("postgresql")
        assert "USING gin" in sql

    def test_partial_index_sql(self):
        mgr = IndexManager()
        idx = mgr.get_index("idx_factors_status")
        assert idx is not None
        sql = idx.to_sql()
        assert "WHERE" in sql

    def test_update_stats(self):
        mgr = IndexManager()
        mgr.update_stats("idx_factors_source_id", size_bytes=50000, scan_count=100)
        idx = mgr.get_index("idx_factors_source_id")
        assert idx.status == IndexStatus.ACTIVE
        assert idx.size_bytes == 50000

    def test_recommendations_unused(self):
        mgr = IndexManager()
        mgr.update_stats("idx_factors_source_id", size_bytes=2_000_000, scan_count=0)
        recs = mgr.get_recommendations()
        assert any(r.action == "drop" for r in recs)

    def test_recommendations_invalid(self):
        mgr = IndexManager()
        idx = mgr.get_index("idx_factors_category")
        idx.status = IndexStatus.INVALID
        recs = mgr.get_recommendations()
        assert any(r.action == "rebuild" for r in recs)


# ── SearchCache ──────────────────────────────────────────────────────


class TestSearchCache:
    def test_put_and_get(self):
        cache = SearchCache(l1_max=100, default_ttl=60)
        params = {"query": "electricity", "geo": "US"}
        cache.put(params, [{"factor_id": "EF:1"}])
        result = cache.get(params)
        assert result == [{"factor_id": "EF:1"}]

    def test_cache_miss(self):
        cache = SearchCache()
        assert cache.get({"query": "nonexistent"}) is None

    def test_ttl_expiration(self):
        cache = SearchCache(default_ttl=0)
        params = {"query": "expired"}
        cache.put(params, [1, 2, 3], ttl=0)
        time.sleep(0.01)
        assert cache.get(params) is None

    def test_lru_eviction(self):
        cache = SearchCache(l1_max=3)
        for i in range(5):
            cache.put({"q": i}, f"result_{i}")
        # First two should be evicted
        assert cache.get({"q": 0}) is None
        assert cache.get({"q": 1}) is None
        assert cache.get({"q": 4}) is not None

    def test_invalidate(self):
        cache = SearchCache()
        params = {"q": "test"}
        cache.put(params, "data")
        assert cache.invalidate(params)
        assert cache.get(params) is None

    def test_invalidate_edition(self):
        cache = SearchCache(l1_max=100)
        for i in range(10):
            cache.put({"q": i}, f"r{i}")
        count = cache.invalidate_edition("ed-2026")
        assert count == 10
        assert cache.stats.l1_size == 0

    def test_stats(self):
        cache = SearchCache()
        cache.put({"q": "a"}, "data")
        cache.get({"q": "a"})  # hit
        cache.get({"q": "b"})  # miss
        assert cache.stats.l1_hits == 1
        assert cache.stats.l1_misses == 1

    def test_stats_to_dict(self):
        cache = SearchCache()
        d = cache.stats.to_dict()
        assert "l1_hit_ratio" in d
        assert "total_hit_ratio" in d

    def test_clear(self):
        cache = SearchCache()
        cache.put({"q": "a"}, "data")
        cache.clear()
        assert cache.get({"q": "a"}) is None


# ── RegulatoryTagger ─────────────────────────────────────────────────


class TestRegulatoryTagger:
    def _make_factor(self, **overrides) -> dict:
        base = {
            "factor_id": "EF:TEST:001",
            "category": "energy",
            "source_id": "defra_2025",
            "geography": "GB",
            "scope": "1",
            "status": "certified",
            "data_quality_score": 4.0,
        }
        base.update(overrides)
        return base

    def test_tag_ghg_protocol(self):
        tagger = RegulatoryTagger()
        result = tagger.tag_factor(self._make_factor())
        frameworks = [f.value for f in result.frameworks]
        assert "ghg_protocol" in frameworks

    def test_tag_csrd(self):
        tagger = RegulatoryTagger()
        result = tagger.tag_factor(self._make_factor(category="transport"))
        frameworks = [f.value for f in result.frameworks]
        assert "csrd_esrs" in frameworks

    def test_tag_cbam(self):
        tagger = RegulatoryTagger()
        result = tagger.tag_factor(self._make_factor(category="cement"))
        frameworks = [f.value for f in result.frameworks]
        assert "cbam" in frameworks

    def test_tag_uk_secr(self):
        tagger = RegulatoryTagger()
        result = tagger.tag_factor(self._make_factor(source_id="defra_2025", geography="GB"))
        frameworks = [f.value for f in result.frameworks]
        assert "uk_secr" in frameworks

    def test_tag_eudr(self):
        tagger = RegulatoryTagger()
        result = tagger.tag_factor(self._make_factor(category="palm_oil"))
        frameworks = [f.value for f in result.frameworks]
        assert "eudr" in frameworks

    def test_tag_iso_requires_certified(self):
        tagger = RegulatoryTagger()
        uncertified = self._make_factor(status="draft", data_quality_score=4.0)
        result = tagger.tag_factor(uncertified)
        frameworks = [f.value for f in result.frameworks]
        assert "iso_14064" not in frameworks

    def test_tag_batch(self):
        tagger = RegulatoryTagger()
        factors = [self._make_factor(factor_id=f"EF:{i}") for i in range(5)]
        results = tagger.tag_batch(factors)
        assert len(results) == 5

    def test_coverage_report(self):
        tagger = RegulatoryTagger()
        factors = [
            self._make_factor(category="energy", scope="1"),
            self._make_factor(category="cement", scope="1"),
            self._make_factor(category="palm_oil", scope="3"),
        ]
        report = tagger.coverage_report(factors)
        assert report["total_factors"] == 3
        assert "framework_coverage" in report
        assert report["framework_coverage"]["ghg_protocol"]["count"] == 3

    def test_filter_by_framework(self):
        tagger = RegulatoryTagger()
        factors = [
            self._make_factor(category="cement"),
            self._make_factor(category="energy"),
            self._make_factor(category="palm_oil"),
        ]
        cbam = tagger.filter_by_framework(factors, RegulatoryFramework.CBAM)
        assert len(cbam) == 1
        assert cbam[0]["category"] == "cement"

    def test_custom_rule(self):
        tagger = RegulatoryTagger(rules=[])
        tagger.add_rule(TaggingRule(
            framework=RegulatoryFramework.GHG_PROTOCOL,
            description="Custom",
            match_categories={"custom"},
        ))
        result = tagger.tag_factor(self._make_factor(category="custom"))
        assert RegulatoryFramework.GHG_PROTOCOL in result.frameworks

    def test_no_match(self):
        tagger = RegulatoryTagger(rules=[])
        result = tagger.tag_factor(self._make_factor())
        assert len(result.frameworks) == 0
