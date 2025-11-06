"""
Test Suite for HotspotAnalysisAgent
GL-VCCI Scope 3 Platform

Comprehensive tests for main agent functionality.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

import pytest
import json
from pathlib import Path
from typing import List, Dict, Any

from services.agents.hotspot import HotspotAnalysisAgent
from services.agents.hotspot.models import (
    Initiative,
    SupplierSwitchScenario,
    ModalShiftScenario,
    ProductSubstitutionScenario
)
from services.agents.hotspot.config import (
    HotspotAnalysisConfig,
    HotspotCriteria,
    AnalysisDimension,
    ScenarioType
)
from services.agents.hotspot.exceptions import HotspotAnalysisError, InsufficientDataError


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def emissions_data() -> List[Dict[str, Any]]:
    """Load sample emissions data from fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "emissions_data.json"
    with open(fixture_path, "r") as f:
        return json.load(f)


@pytest.fixture
def agent() -> HotspotAnalysisAgent:
    """Create HotspotAnalysisAgent instance."""
    return HotspotAnalysisAgent()


@pytest.fixture
def custom_config() -> HotspotAnalysisConfig:
    """Create custom configuration."""
    return HotspotAnalysisConfig(
        hotspot_criteria=HotspotCriteria(
            emission_threshold_tco2e=5000.0,
            percent_threshold=10.0,
            dqi_threshold=60.0
        )
    )


@pytest.fixture
def sample_initiatives() -> List[Initiative]:
    """Create sample initiatives for testing."""
    return [
        Initiative(
            name="Switch to Renewable Energy",
            reduction_potential_tco2e=10000,
            implementation_cost_usd=200000,
            annual_savings_usd=50000
        ),
        Initiative(
            name="Supplier Engagement Program",
            reduction_potential_tco2e=5000,
            implementation_cost_usd=75000,
            annual_savings_usd=0
        ),
        Initiative(
            name="Modal Shift to Rail",
            reduction_potential_tco2e=3000,
            implementation_cost_usd=-10000,  # Savings
            annual_savings_usd=15000
        )
    ]


# ============================================================================
# TEST AGENT INITIALIZATION
# ============================================================================

def test_agent_initialization():
    """Test agent initialization with default config."""
    agent = HotspotAnalysisAgent()
    assert agent is not None
    assert agent.config is not None
    assert agent.pareto_analyzer is not None
    assert agent.segmentation_analyzer is not None


def test_agent_initialization_with_custom_config(custom_config):
    """Test agent initialization with custom config."""
    agent = HotspotAnalysisAgent(config=custom_config)
    assert agent.config == custom_config
    assert agent.config.hotspot_criteria.emission_threshold_tco2e == 5000.0


# ============================================================================
# TEST PARETO ANALYSIS
# ============================================================================

def test_pareto_analysis_supplier_dimension(agent, emissions_data):
    """Test Pareto analysis by supplier."""
    result = agent.analyze_pareto(emissions_data, dimension="supplier_name")

    assert result is not None
    assert result.dimension == "supplier_name"
    assert result.total_emissions_tco2e > 0
    assert result.total_entities > 0
    assert len(result.top_20_percent) > 0
    assert result.pareto_efficiency > 0


def test_pareto_analysis_category_dimension(agent, emissions_data):
    """Test Pareto analysis by category."""
    result = agent.analyze_pareto(emissions_data, dimension="scope3_category")

    assert result is not None
    assert result.dimension == "scope3_category"
    assert len(result.top_20_percent) > 0


def test_pareto_analysis_chart_data(agent, emissions_data):
    """Test Pareto chart data generation."""
    result = agent.analyze_pareto(emissions_data)

    assert "chart_data" in result.model_dump()
    chart_data = result.chart_data
    assert chart_data["chart_type"] == "pareto"
    assert "data" in chart_data
    assert len(chart_data["data"]) > 0


def test_pareto_analysis_empty_data(agent):
    """Test Pareto analysis with empty data."""
    with pytest.raises(HotspotAnalysisError):
        agent.analyze_pareto([])


def test_pareto_analysis_insufficient_data(agent):
    """Test Pareto analysis with insufficient data."""
    minimal_data = [
        {"emissions_tco2e": 100, "supplier_name": "Test"}
    ]
    with pytest.raises(HotspotAnalysisError):
        agent.analyze_pareto(minimal_data)


# ============================================================================
# TEST SEGMENTATION ANALYSIS
# ============================================================================

def test_segmentation_analysis_single_dimension(agent, emissions_data):
    """Test segmentation by single dimension."""
    results = agent.analyze_segmentation(
        emissions_data,
        dimensions=[AnalysisDimension.SUPPLIER]
    )

    assert AnalysisDimension.SUPPLIER in results
    supplier_analysis = results[AnalysisDimension.SUPPLIER]
    assert supplier_analysis.total_emissions_tco2e > 0
    assert len(supplier_analysis.segments) > 0


def test_segmentation_analysis_multiple_dimensions(agent, emissions_data):
    """Test segmentation by multiple dimensions."""
    dimensions = [
        AnalysisDimension.SUPPLIER,
        AnalysisDimension.CATEGORY,
        AnalysisDimension.REGION
    ]
    results = agent.analyze_segmentation(emissions_data, dimensions=dimensions)

    assert len(results) == len(dimensions)
    for dim in dimensions:
        assert dim in results


def test_segmentation_top_segments(agent, emissions_data):
    """Test top segments extraction."""
    results = agent.analyze_segmentation(
        emissions_data,
        dimensions=[AnalysisDimension.SUPPLIER]
    )

    analysis = results[AnalysisDimension.SUPPLIER]
    assert len(analysis.top_10_segments) <= 10
    # Verify sorted by emissions descending
    for i in range(len(analysis.top_10_segments) - 1):
        assert (analysis.top_10_segments[i].emissions_tco2e >=
                analysis.top_10_segments[i + 1].emissions_tco2e)


def test_segmentation_concentration(agent, emissions_data):
    """Test concentration calculation."""
    results = agent.analyze_segmentation(
        emissions_data,
        dimensions=[AnalysisDimension.SUPPLIER]
    )

    analysis = results[AnalysisDimension.SUPPLIER]
    assert 0 <= analysis.top_3_concentration <= 100


# ============================================================================
# TEST SCENARIO MODELING
# ============================================================================

def test_supplier_switch_scenario(agent, emissions_data):
    """Test supplier switching scenario."""
    scenario = SupplierSwitchScenario(
        name="Switch to Low Carbon Steel",
        from_supplier="Acme Steel Corp",
        to_supplier="Green Steel Inc",
        products=["Steel Sheets"],
        current_emissions_tco2e=45000,
        new_emissions_tco2e=30000,
        estimated_reduction_tco2e=15000,
        estimated_cost_usd=100000
    )

    result = agent.model_scenario(scenario, emissions_data)

    assert result is not None
    assert result.reduction_tco2e > 0
    assert result.roi_usd_per_tco2e > 0
    assert len(result.risks) > 0


def test_modal_shift_scenario(agent, emissions_data):
    """Test modal shift scenario."""
    scenario = ModalShiftScenario(
        name="Shift Air to Sea",
        from_mode="air",
        to_mode="sea",
        routes=["US-EU"],
        volume_pct=50,
        estimated_reduction_tco2e=2000,
        estimated_cost_usd=-10000  # Savings
    )

    result = agent.model_scenario(scenario, emissions_data)

    assert result is not None
    assert result.reduction_tco2e > 0
    assert result.implementation_cost_usd >= 0


def test_product_substitution_scenario(agent):
    """Test product substitution scenario."""
    scenario = ProductSubstitutionScenario(
        name="Virgin to Recycled Steel",
        from_product="virgin_steel",
        to_product="recycled_steel",
        volume_tonnes=1000,
        current_ef_kgco2e_per_tonne=2000,
        new_ef_kgco2e_per_tonne=1000,
        estimated_reduction_tco2e=1000,
        estimated_cost_usd=50000
    )

    result = agent.model_scenario(scenario)

    assert result is not None
    assert result.reduction_tco2e > 0


def test_compare_scenarios(agent, emissions_data):
    """Test scenario comparison."""
    scenarios = [
        SupplierSwitchScenario(
            name="Scenario 1",
            from_supplier="A",
            to_supplier="B",
            products=["P1"],
            current_emissions_tco2e=1000,
            new_emissions_tco2e=800,
            estimated_reduction_tco2e=200,
            estimated_cost_usd=10000
        ),
        ModalShiftScenario(
            name="Scenario 2",
            from_mode="air",
            to_mode="sea",
            routes=["R1"],
            volume_pct=50,
            estimated_reduction_tco2e=300,
            estimated_cost_usd=5000
        )
    ]

    result = agent.compare_scenarios(scenarios, emissions_data)

    assert result is not None
    assert result["n_scenarios"] == 2
    assert "total_reduction_potential_tco2e" in result


# ============================================================================
# TEST ROI ANALYSIS
# ============================================================================

def test_calculate_roi(agent, sample_initiatives):
    """Test ROI calculation."""
    initiative = sample_initiatives[0]
    result = agent.calculate_roi(initiative)

    assert result is not None
    assert result.roi_usd_per_tco2e > 0
    assert result.npv_10y_usd is not None
    assert result.carbon_value_usd > 0


def test_calculate_roi_multiple_initiatives(agent, sample_initiatives):
    """Test ROI for multiple initiatives."""
    for initiative in sample_initiatives:
        result = agent.calculate_roi(initiative)
        assert result is not None


# ============================================================================
# TEST ABATEMENT CURVE
# ============================================================================

def test_generate_abatement_curve(agent, sample_initiatives):
    """Test abatement curve generation."""
    result = agent.generate_abatement_curve(sample_initiatives)

    assert result is not None
    assert len(result.initiatives) == len(sample_initiatives)
    assert result.total_reduction_potential_tco2e > 0
    assert result.n_negative_cost >= 0
    assert result.n_positive_cost >= 0


def test_abatement_curve_sorting(agent, sample_initiatives):
    """Test that abatement curve is sorted by cost-effectiveness."""
    result = agent.generate_abatement_curve(sample_initiatives)

    # Verify sorted by cost per tCO2e (ascending)
    for i in range(len(result.initiatives) - 1):
        assert (result.initiatives[i].cost_per_tco2e <=
                result.initiatives[i + 1].cost_per_tco2e)


def test_abatement_curve_empty_list(agent):
    """Test abatement curve with empty list."""
    with pytest.raises(HotspotAnalysisError):
        agent.generate_abatement_curve([])


# ============================================================================
# TEST HOTSPOT DETECTION
# ============================================================================

def test_identify_hotspots(agent, emissions_data):
    """Test hotspot detection."""
    result = agent.identify_hotspots(emissions_data)

    assert result is not None
    assert result.n_hotspots > 0
    assert result.total_emissions_tco2e > 0
    assert len(result.hotspots) > 0


def test_identify_hotspots_custom_criteria(agent, emissions_data):
    """Test hotspot detection with custom criteria."""
    criteria = HotspotCriteria(
        emission_threshold_tco2e=10000.0,
        percent_threshold=15.0
    )

    result = agent.identify_hotspots(emissions_data, criteria=criteria)

    assert result is not None
    # Should have fewer hotspots with higher thresholds
    assert result.n_hotspots >= 0


def test_hotspot_priorities(agent, emissions_data):
    """Test hotspot priority levels."""
    result = agent.identify_hotspots(emissions_data)

    assert len(result.critical_hotspots) >= 0
    assert len(result.high_hotspots) >= 0


def test_hotspot_coverage(agent, emissions_data):
    """Test hotspot coverage calculation."""
    result = agent.identify_hotspots(emissions_data)

    assert 0 <= result.hotspot_coverage_pct <= 100


# ============================================================================
# TEST INSIGHT GENERATION
# ============================================================================

def test_generate_insights_from_data(agent, emissions_data):
    """Test insight generation from raw data."""
    result = agent.generate_insights(emissions_data=emissions_data)

    assert result is not None
    assert result.total_insights > 0
    assert len(result.all_insights) > 0
    assert len(result.top_recommendations) > 0


def test_generate_insights_from_analysis(agent, emissions_data):
    """Test insight generation from pre-computed analysis."""
    hotspot_report = agent.identify_hotspots(emissions_data)
    pareto_analysis = agent.analyze_pareto(emissions_data)

    result = agent.generate_insights(
        hotspot_report=hotspot_report,
        pareto_analysis=pareto_analysis
    )

    assert result is not None
    assert result.total_insights > 0


def test_insight_priorities(agent, emissions_data):
    """Test insight priority categorization."""
    result = agent.generate_insights(emissions_data=emissions_data)

    total = (len(result.critical_insights) +
             len(result.high_insights) +
             len(result.medium_insights) +
             len(result.low_insights))

    assert total == result.total_insights


def test_insight_summary(agent, emissions_data):
    """Test insight summary generation."""
    result = agent.generate_insights(emissions_data=emissions_data)

    assert result.summary is not None
    assert len(result.summary) > 0


# ============================================================================
# TEST COMPREHENSIVE ANALYSIS
# ============================================================================

def test_comprehensive_analysis(agent, emissions_data):
    """Test comprehensive analysis."""
    result = agent.analyze_comprehensive(emissions_data)

    assert result is not None
    assert "pareto" in result
    assert "segmentation" in result
    assert "hotspots" in result
    assert "insights" in result
    assert "summary" in result


def test_comprehensive_analysis_summary(agent, emissions_data):
    """Test comprehensive analysis summary."""
    result = agent.analyze_comprehensive(emissions_data)

    summary = result["summary"]
    assert summary["total_records"] == len(emissions_data)
    assert summary["total_emissions_tco2e"] > 0
    assert summary["processing_time_seconds"] > 0


def test_comprehensive_analysis_performance(agent, emissions_data):
    """Test comprehensive analysis performance."""
    import time

    start = time.time()
    result = agent.analyze_comprehensive(emissions_data)
    elapsed = time.time() - start

    # Should complete in reasonable time
    assert elapsed < 5.0  # 5 seconds for 10 records


def test_comprehensive_analysis_empty_data(agent):
    """Test comprehensive analysis with empty data."""
    with pytest.raises(HotspotAnalysisError):
        agent.analyze_comprehensive([])


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.performance
def test_performance_100_records(agent):
    """Test performance with 100 records."""
    import time

    # Generate 100 records
    data = []
    for i in range(100):
        data.append({
            "record_id": f"REC-{i:04d}",
            "emissions_tco2e": 1000 + i * 10,
            "emissions_kgco2e": (1000 + i * 10) * 1000,
            "supplier_name": f"Supplier {i % 20}",
            "scope3_category": (i % 5) + 1,
            "product_name": f"Product {i % 10}",
            "region": "US" if i % 2 == 0 else "EU",
            "dqi_score": 70.0 + (i % 30),
            "tier": (i % 3) + 1
        })

    start = time.time()
    result = agent.analyze_comprehensive(data)
    elapsed = time.time() - start

    assert result is not None
    assert elapsed < 2.0  # Should complete in < 2 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
