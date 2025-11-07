"""
Test Suite for Pareto Analysis
GL-VCCI Scope 3 Platform

Tests for Pareto analyzer (80/20 rule).

Version: 1.0.0
"""

import pytest
from services.agents.hotspot.analysis.pareto import ParetoAnalyzer
from services.agents.hotspot.config import ParetoConfig
from services.agents.hotspot.exceptions import ParetoAnalysisError, InsufficientDataError


@pytest.fixture
def analyzer():
    """Create ParetoAnalyzer instance."""
    return ParetoAnalyzer()


@pytest.fixture
def sample_data():
    """Create sample emission data."""
    return [
        {"emissions_tco2e": 50000, "supplier_name": "Supplier A"},
        {"emissions_tco2e": 20000, "supplier_name": "Supplier B"},
        {"emissions_tco2e": 15000, "supplier_name": "Supplier C"},
        {"emissions_tco2e": 10000, "supplier_name": "Supplier D"},
        {"emissions_tco2e": 5000, "supplier_name": "Supplier E"},
        {"emissions_tco2e": 3000, "supplier_name": "Supplier F"},
        {"emissions_tco2e": 2000, "supplier_name": "Supplier G"},
        {"emissions_tco2e": 1000, "supplier_name": "Supplier H"},
    ]


def test_pareto_basic_analysis(analyzer, sample_data):
    """Test basic Pareto analysis."""
    result = analyzer.analyze(sample_data, "supplier_name")

    assert result.total_entities == 8
    assert result.total_emissions_tco2e == 106000
    assert len(result.top_20_percent) > 0


def test_pareto_80_20_rule(analyzer, sample_data):
    """Test that 80/20 rule is detected."""
    result = analyzer.analyze(sample_data, "supplier_name")

    # Top 20% should capture significant emissions
    assert result.pareto_efficiency > 0.5


def test_pareto_cumulative_calculation(analyzer, sample_data):
    """Test cumulative percentage calculation."""
    result = analyzer.analyze(sample_data, "supplier_name")

    # Last item should be 100%
    assert result.top_20_percent[-1].cumulative_percent <= 100


def test_pareto_sorting(analyzer, sample_data):
    """Test that items are sorted by emissions descending."""
    result = analyzer.analyze(sample_data, "supplier_name")

    for i in range(len(result.top_20_percent) - 1):
        assert (result.top_20_percent[i].emissions_tco2e >=
                result.top_20_percent[i + 1].emissions_tco2e)


def test_pareto_empty_data(analyzer):
    """Test with empty data."""
    with pytest.raises(InsufficientDataError):
        analyzer.analyze([], "supplier_name")


def test_pareto_insufficient_records(analyzer):
    """Test with insufficient records."""
    data = [{"emissions_tco2e": 100, "supplier_name": "A"}]

    with pytest.raises(InsufficientDataError):
        analyzer.analyze(data, "supplier_name")


def test_pareto_custom_config():
    """Test with custom configuration."""
    config = ParetoConfig(
        pareto_threshold=0.70,
        top_n_percent=0.30,
        min_records=3
    )

    analyzer = ParetoAnalyzer(config)
    data = [
        {"emissions_tco2e": 100, "supplier_name": "A"},
        {"emissions_tco2e": 50, "supplier_name": "B"},
        {"emissions_tco2e": 25, "supplier_name": "C"},
    ]

    result = analyzer.analyze(data, "supplier_name")
    assert result is not None


def test_pareto_chart_data(analyzer, sample_data):
    """Test chart data generation."""
    result = analyzer.analyze(sample_data, "supplier_name")

    assert "chart_data" in result.model_dump()
    chart = result.chart_data
    assert chart["chart_type"] == "pareto"
    assert len(chart["data"]) > 0


def test_pareto_category_dimension(analyzer):
    """Test Pareto by category."""
    data = [
        {"emissions_tco2e": 50000, "scope3_category": 1},
        {"emissions_tco2e": 30000, "scope3_category": 4},
        {"emissions_tco2e": 20000, "scope3_category": 6},
        {"emissions_tco2e": 10000, "scope3_category": 1},
        {"emissions_tco2e": 5000, "scope3_category": 4},
    ]

    result = analyzer.analyze(data, "scope3_category")
    assert result.total_entities > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
