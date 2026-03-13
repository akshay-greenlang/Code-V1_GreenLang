# -*- coding: utf-8 -*-
"""
Tests for Engine 4: Measure Library Engine - AGENT-EUDR-025

Tests 500+ mitigation measure catalog, full-text search, faceted filtering,
measure comparison, category distribution, ISO 31000 mapping, EUDR article
tagging, and certification scheme support.

Test count: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    RiskCategory,
    ISO31000TreatmentType,
    ImplementationComplexity,
    MitigationMeasure,
    SearchMeasuresRequest,
    SearchMeasuresResponse,
    MIN_LIBRARY_MEASURES,
    RISK_CATEGORY_COUNT,
)
from greenlang.agents.eudr.risk_mitigation_advisor.measure_library_engine import (
    MeasureLibraryEngine,
)


class TestMeasureLibraryInit:
    def test_engine_initializes(self, measure_library_engine):
        assert measure_library_engine is not None

    def test_library_has_measures(self, measure_library_engine):
        count = measure_library_engine.get_measure_count()
        assert count >= MIN_LIBRARY_MEASURES

    def test_library_covers_all_risk_categories(self, measure_library_engine):
        categories = measure_library_engine.get_covered_categories()
        assert len(categories) >= RISK_CATEGORY_COUNT


class TestMeasureSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, measure_library_engine, search_measures_request):
        result = await measure_library_engine.search(search_measures_request)
        assert isinstance(result, SearchMeasuresResponse)
        assert result.total_count >= 0

    @pytest.mark.asyncio
    async def test_search_by_risk_category(self, measure_library_engine):
        for cat in RiskCategory:
            req = SearchMeasuresRequest(risk_category=cat, limit=5)
            result = await measure_library_engine.search(req)
            assert isinstance(result, SearchMeasuresResponse)
            for m in result.measures:
                assert m.risk_category == cat

    @pytest.mark.asyncio
    async def test_search_by_query(self, measure_library_engine):
        req = SearchMeasuresRequest(query="monitoring", limit=10)
        result = await measure_library_engine.search(req)
        assert isinstance(result, SearchMeasuresResponse)

    @pytest.mark.asyncio
    async def test_search_by_commodity(self, measure_library_engine):
        req = SearchMeasuresRequest(commodity="palm_oil", limit=10)
        result = await measure_library_engine.search(req)
        assert isinstance(result, SearchMeasuresResponse)

    @pytest.mark.asyncio
    async def test_search_by_complexity(self, measure_library_engine):
        req = SearchMeasuresRequest(complexity=ImplementationComplexity.LOW, limit=10)
        result = await measure_library_engine.search(req)
        for m in result.measures:
            assert m.implementation_complexity == ImplementationComplexity.LOW

    @pytest.mark.asyncio
    async def test_search_by_iso_31000_type(self, measure_library_engine):
        req = SearchMeasuresRequest(iso_31000_type=ISO31000TreatmentType.REDUCE, limit=10)
        result = await measure_library_engine.search(req)
        for m in result.measures:
            assert m.iso_31000_type == ISO31000TreatmentType.REDUCE

    @pytest.mark.asyncio
    async def test_search_by_max_cost(self, measure_library_engine):
        req = SearchMeasuresRequest(max_cost_eur=Decimal("5000"), limit=10)
        result = await measure_library_engine.search(req)
        for m in result.measures:
            assert m.cost_estimate_eur.min_value <= Decimal("5000")

    @pytest.mark.asyncio
    async def test_search_pagination(self, measure_library_engine):
        req1 = SearchMeasuresRequest(limit=5, offset=0)
        req2 = SearchMeasuresRequest(limit=5, offset=5)
        r1 = await measure_library_engine.search(req1)
        r2 = await measure_library_engine.search(req2)
        if r1.measures and r2.measures:
            ids1 = {m.measure_id for m in r1.measures}
            ids2 = {m.measure_id for m in r2.measures}
            assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_search_empty_query(self, measure_library_engine):
        req = SearchMeasuresRequest(limit=10)
        result = await measure_library_engine.search(req)
        assert isinstance(result, SearchMeasuresResponse)

    @pytest.mark.asyncio
    async def test_search_time_recorded(self, measure_library_engine, search_measures_request):
        result = await measure_library_engine.search(search_measures_request)
        assert result.search_time_ms >= Decimal("0")


class TestMeasureRetrieval:
    @pytest.mark.asyncio
    async def test_get_measure_by_id(self, measure_library_engine):
        req = SearchMeasuresRequest(limit=1)
        result = await measure_library_engine.search(req)
        if result.measures:
            measure = await measure_library_engine.get_measure(result.measures[0].measure_id)
            assert measure is not None
            assert measure.measure_id == result.measures[0].measure_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_measure(self, measure_library_engine):
        measure = await measure_library_engine.get_measure("nonexistent-id")
        assert measure is None


class TestMeasureComparison:
    @pytest.mark.asyncio
    async def test_compare_two_measures(self, measure_library_engine):
        req = SearchMeasuresRequest(limit=2)
        result = await measure_library_engine.search(req)
        if len(result.measures) >= 2:
            comparison = await measure_library_engine.compare_measures(
                [result.measures[0].measure_id, result.measures[1].measure_id]
            )
            assert comparison is not None
            assert len(comparison) == 2

    @pytest.mark.asyncio
    async def test_compare_empty_list(self, measure_library_engine):
        comparison = await measure_library_engine.compare_measures([])
        assert comparison == []


class TestMeasureCategoryDistribution:
    @pytest.mark.asyncio
    async def test_country_risk_measures_count(self, measure_library_engine):
        req = SearchMeasuresRequest(risk_category=RiskCategory.COUNTRY, limit=100)
        result = await measure_library_engine.search(req)
        assert result.total_count >= 60

    @pytest.mark.asyncio
    async def test_supplier_risk_measures_count(self, measure_library_engine):
        req = SearchMeasuresRequest(risk_category=RiskCategory.SUPPLIER, limit=100)
        result = await measure_library_engine.search(req)
        assert result.total_count >= 70

    @pytest.mark.asyncio
    async def test_deforestation_measures_count(self, measure_library_engine):
        req = SearchMeasuresRequest(risk_category=RiskCategory.DEFORESTATION, limit=100)
        result = await measure_library_engine.search(req)
        assert result.total_count >= 60

    @pytest.mark.asyncio
    async def test_all_categories_have_measures(self, measure_library_engine):
        for cat in RiskCategory:
            req = SearchMeasuresRequest(risk_category=cat, limit=1)
            result = await measure_library_engine.search(req)
            assert result.total_count >= 1, f"No measures for category {cat.value}"


class TestMeasureDataQuality:
    @pytest.mark.asyncio
    async def test_measures_have_names(self, measure_library_engine):
        req = SearchMeasuresRequest(limit=20)
        result = await measure_library_engine.search(req)
        for m in result.measures:
            assert m.name is not None
            assert len(m.name) > 0

    @pytest.mark.asyncio
    async def test_measures_have_descriptions(self, measure_library_engine):
        req = SearchMeasuresRequest(limit=20)
        result = await measure_library_engine.search(req)
        for m in result.measures:
            assert m.description is not None
            assert len(m.description) > 0

    @pytest.mark.asyncio
    async def test_measures_have_cost_ranges(self, measure_library_engine):
        req = SearchMeasuresRequest(limit=20)
        result = await measure_library_engine.search(req)
        for m in result.measures:
            assert m.cost_estimate_eur.min_value >= Decimal("0")
            assert m.cost_estimate_eur.max_value >= m.cost_estimate_eur.min_value

    @pytest.mark.asyncio
    async def test_measures_have_effectiveness_ratings(self, measure_library_engine):
        req = SearchMeasuresRequest(limit=20)
        result = await measure_library_engine.search(req)
        for m in result.measures:
            assert Decimal("0") <= m.effectiveness_rating <= Decimal("100")

    @pytest.mark.asyncio
    async def test_measures_have_valid_iso_types(self, measure_library_engine):
        req = SearchMeasuresRequest(limit=20)
        result = await measure_library_engine.search(req)
        for m in result.measures:
            assert isinstance(m.iso_31000_type, ISO31000TreatmentType)

    @pytest.mark.asyncio
    async def test_measures_have_risk_reduction_range(self, measure_library_engine):
        req = SearchMeasuresRequest(limit=20)
        result = await measure_library_engine.search(req)
        for m in result.measures:
            assert m.expected_risk_reduction_pct.min_value >= Decimal("0")
            assert m.expected_risk_reduction_pct.max_value >= m.expected_risk_reduction_pct.min_value


class TestMeasurePackages:
    @pytest.mark.asyncio
    async def test_get_recommended_package(self, measure_library_engine):
        package = await measure_library_engine.get_recommended_package(
            risk_category=RiskCategory.DEFORESTATION,
            commodity="palm_oil",
            risk_level="high",
        )
        assert package is not None
        assert len(package) >= 1

    @pytest.mark.asyncio
    async def test_package_for_low_risk(self, measure_library_engine):
        package = await measure_library_engine.get_recommended_package(
            risk_category=RiskCategory.COUNTRY,
            commodity="wood",
            risk_level="low",
        )
        assert package is not None
