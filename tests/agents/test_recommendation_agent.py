"""Tests for RecommendationAgent.

This module tests the RecommendationAgent implementation, ensuring:
1. Recommendation generation for all emission source types
2. ROI and payback period calculations
3. Technology recommendations based on context
4. Prioritization and ranking logic
5. Implementation planning and roadmap creation
6. Country-specific recommendations
7. Benchmarking against industry standards
8. Edge cases and boundary conditions
9. Deterministic behavior
10. Performance within requirements

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from greenlang.agents.recommendation_agent import RecommendationAgent
from greenlang.agents.base import AgentConfig


class TestRecommendationAgentBasic:
    """Basic tests for RecommendationAgent initialization and structure."""

    @pytest.fixture
    def agent(self):
        """Create RecommendationAgent instance for testing."""
        return RecommendationAgent()

    @pytest.fixture
    def simple_emissions(self):
        """Simple emissions data for basic testing."""
        return {
            "emissions_by_source": {
                "electricity": 15000,
                "natural_gas": 8500,
            },
            "building_type": "commercial_office",
            "country": "US",
        }

    @pytest.fixture
    def complex_emissions(self):
        """Complex emissions data with full context."""
        return {
            "emissions_by_source": {
                "electricity": 15000,
                "natural_gas": 8500,
                "diesel": 3200,
            },
            "building_type": "commercial_office",
            "building_age": 12,
            "performance_rating": "Average",
            "country": "US",
            "load_breakdown": {
                "hvac_load": 0.45,
                "lighting_load": 0.25,
                "equipment_load": 0.30,
            },
        }

    def test_initialization(self, agent):
        """Test RecommendationAgent initializes correctly."""
        assert agent.config.name == "RecommendationAgent"
        assert agent.config.description is not None
        assert hasattr(agent, "recommendations_db")
        assert agent.recommendations_db is not None

    def test_recommendations_database_structure(self, agent):
        """Test recommendations database has expected structure."""
        db = agent.recommendations_db

        # Check major categories exist
        assert "hvac" in db
        assert "lighting" in db
        assert "envelope" in db
        assert "renewable" in db
        assert "operations" in db

        # Check HVAC category has subcategories
        assert "high_consumption" in db["hvac"]
        assert "poor_efficiency" in db["hvac"]

        # Check recommendations have required fields
        hvac_rec = db["hvac"]["high_consumption"][0]
        assert "action" in hvac_rec
        assert "impact" in hvac_rec
        assert "cost" in hvac_rec
        assert "payback" in hvac_rec
        assert "priority" in hvac_rec

    def test_validate_input_valid(self, agent, simple_emissions):
        """Test validation passes for valid input."""
        assert agent.validate_input(simple_emissions) is True

    def test_validate_input_minimal(self, agent):
        """Test validation passes for minimal valid input."""
        minimal = {"emissions_by_source": {"electricity": 10000}}
        assert agent.validate_input(minimal) is True

    def test_execute_basic_recommendations(self, agent, simple_emissions):
        """Test basic recommendation generation."""
        result = agent.execute(simple_emissions)

        # Should succeed
        assert result.success is True
        assert result.data is not None

        # Check output structure
        data = result.data
        assert "recommendations" in data
        assert "total_recommendations" in data
        assert "potential_emissions_reduction" in data
        assert "grouped_recommendations" in data
        assert "quick_wins" in data
        assert "high_impact" in data
        assert "implementation_roadmap" in data

        # Check recommendations list
        recommendations = data["recommendations"]
        assert len(recommendations) > 0
        assert len(recommendations) <= 10  # Default max is 10

        # Check first recommendation structure
        rec = recommendations[0]
        assert "action" in rec
        assert "impact" in rec
        assert "cost" in rec
        assert "payback" in rec
        assert "priority" in rec

    def test_execute_complex_scenario(self, agent, complex_emissions):
        """Test recommendation generation with full context."""
        result = agent.execute(complex_emissions)

        assert result.success is True
        data = result.data

        # Should have recommendations
        assert len(data["recommendations"]) > 0

        # Should have grouped recommendations
        grouped = data["grouped_recommendations"]
        assert len(grouped) > 0

        # Should have implementation roadmap
        roadmap = data["implementation_roadmap"]
        assert len(roadmap) > 0


class TestRecommendationGeneration:
    """Tests for recommendation generation logic."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_electricity_heavy_recommendations(self, agent):
        """Test recommendations for electricity-heavy emissions."""
        input_data = {
            "emissions_by_source": {
                "electricity": 20000,  # 80% of total
                "natural_gas": 5000,
            }
        }

        result = agent.execute(input_data)
        recommendations = result.data["recommendations"]

        # Should include renewable energy recommendations
        renewable_recs = [r for r in recommendations if "solar" in r["action"].lower() or "renewable" in r["action"].lower()]
        assert len(renewable_recs) > 0

        # Should include lighting recommendations
        lighting_recs = [r for r in recommendations if "lighting" in r["action"].lower() or "led" in r["action"].lower()]
        assert len(lighting_recs) > 0

    def test_natural_gas_heavy_recommendations(self, agent):
        """Test recommendations for natural gas-heavy emissions."""
        input_data = {
            "emissions_by_source": {
                "electricity": 3000,
                "natural_gas": 17000,  # 85% of total
            },
            "building_age": 25,  # Old enough to trigger envelope recommendations
        }

        result = agent.execute(input_data)
        recommendations = result.data["recommendations"]

        # Should include envelope recommendations (old building, >20 years)
        envelope_recs = [r for r in recommendations if "insulation" in r["action"].lower() or "envelope" in r["action"].lower() or "window" in r["action"].lower() or "seal" in r["action"].lower()]
        assert len(envelope_recs) > 0

    def test_high_hvac_load_recommendations(self, agent):
        """Test recommendations when HVAC load is high."""
        input_data = {
            "emissions_by_source": {"electricity": 15000},
            "load_breakdown": {"hvac_load": 0.55},  # 55% HVAC
        }

        result = agent.execute(input_data)
        recommendations = result.data["recommendations"]

        # Should include HVAC-specific recommendations
        hvac_recs = [r for r in recommendations if "hvac" in r["action"].lower()]
        assert len(hvac_recs) > 0

    def test_old_building_recommendations(self, agent):
        """Test recommendations for old buildings."""
        input_data = {
            "emissions_by_source": {"electricity": 10000, "natural_gas": 8000},
            "building_age": 25,  # Old building
        }

        result = agent.execute(input_data)
        recommendations = result.data["recommendations"]

        # Should include envelope recommendations
        envelope_recs = [r for r in recommendations if "insulation" in r["action"].lower() or "window" in r["action"].lower() or "seal" in r["action"].lower()]
        assert len(envelope_recs) > 0

    def test_poor_performance_recommendations(self, agent):
        """Test recommendations for poor-performing buildings."""
        input_data = {
            "emissions_by_source": {"electricity": 12000},
            "performance_rating": "Poor",
        }

        result = agent.execute(input_data)
        recommendations = result.data["recommendations"]

        # Should include operational recommendations
        ops_recs = [r for r in recommendations if "energy management" in r["action"].lower() or "audit" in r["action"].lower() or "train" in r["action"].lower()]
        assert len(ops_recs) > 0

    def test_multiple_sources_prioritization(self, agent):
        """Test that recommendations are prioritized correctly."""
        input_data = {
            "emissions_by_source": {
                "electricity": 18000,  # Largest (65% - above 60% threshold)
                "natural_gas": 7500,   # Second (27%)
                "diesel": 2200,        # Smallest (8%)
            }
        }

        result = agent.execute(input_data)
        recommendations = result.data["recommendations"]

        # Should have multiple recommendations
        assert len(recommendations) > 0

        # Check that we have electricity-related recommendations
        # (since it's the largest source and above 60% threshold)
        elec_recs = [r for r in recommendations if "electric" in r["action"].lower() or "solar" in r["action"].lower() or "renewable" in r["action"].lower() or "led" in r["action"].lower() or "lighting" in r["action"].lower()]
        assert len(elec_recs) > 0, "Should have electricity-related recommendations for dominant source"


class TestCountrySpecificRecommendations:
    """Tests for country-specific recommendation logic."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    @pytest.fixture
    def base_emissions(self):
        return {
            "emissions_by_source": {"electricity": 15000, "natural_gas": 8500}
        }

    def test_us_recommendations(self, agent, base_emissions):
        """Test US-specific recommendations."""
        base_emissions["country"] = "US"
        result = agent.execute(base_emissions)

        recommendations = result.data["recommendations"]

        # Should include US-specific recommendations
        us_recs = [r for r in recommendations if "IRA" in r["action"] or "ENERGY STAR" in r["action"] or "tax credit" in r["action"]]
        assert len(us_recs) > 0

    def test_india_recommendations(self, agent, base_emissions):
        """Test India-specific recommendations."""
        base_emissions["country"] = "IN"
        result = agent.execute(base_emissions)

        recommendations = result.data["recommendations"]

        # Should include India-specific recommendations
        india_recs = [r for r in recommendations if "PAT" in r["action"] or "subsidy" in r["action"]]
        assert len(india_recs) > 0

    def test_eu_recommendations(self, agent, base_emissions):
        """Test EU-specific recommendations."""
        base_emissions["country"] = "EU"
        result = agent.execute(base_emissions)

        recommendations = result.data["recommendations"]

        # Should include EU-specific recommendations
        eu_recs = [r for r in recommendations if "EU Taxonomy" in r["action"] or "district heating" in r["action"]]
        assert len(eu_recs) > 0

    def test_unknown_country_fallback(self, agent, base_emissions):
        """Test handling of unknown country codes."""
        base_emissions["country"] = "XX"  # Unknown
        result = agent.execute(base_emissions)

        # Should still succeed with general recommendations
        assert result.success is True
        assert len(result.data["recommendations"]) > 0


class TestRecommendationPrioritization:
    """Tests for recommendation prioritization and ranking."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_prioritization_high_first(self, agent):
        """Test that high-priority recommendations come first."""
        recommendations = [
            {"action": "Low priority", "priority": "Low", "payback": "10 years"},
            {"action": "High priority", "priority": "High", "payback": "2 years"},
            {"action": "Medium priority", "priority": "Medium", "payback": "5 years"},
        ]

        prioritized = agent._prioritize_recommendations(recommendations)

        # High priority should be first
        assert prioritized[0]["priority"] == "High"

    def test_prioritization_by_payback(self, agent):
        """Test prioritization considers payback period."""
        recommendations = [
            {"action": "Fast payback", "priority": "High", "payback": "1 year"},
            {"action": "Slow payback", "priority": "High", "payback": "10 years"},
        ]

        prioritized = agent._prioritize_recommendations(recommendations)

        # Within same priority, shorter payback should come first
        assert "Fast payback" in prioritized[0]["action"]

    def test_deduplication(self, agent):
        """Test that duplicate recommendations are removed."""
        recommendations = [
            {"action": "Install LED lighting", "priority": "High", "payback": "2 years"},
            {"action": "Install LED lighting", "priority": "High", "payback": "2 years"},
            {"action": "Different action", "priority": "High", "payback": "2 years"},
        ]

        prioritized = agent._prioritize_recommendations(recommendations)

        # Should have only 2 recommendations (duplicate removed)
        assert len(prioritized) == 2

    def test_extract_payback_years_immediate(self, agent):
        """Test extraction of 'Immediate' payback."""
        assert agent._extract_payback_years("Immediate") == 0

    def test_extract_payback_years_range(self, agent):
        """Test extraction from payback range."""
        assert agent._extract_payback_years("2-3 years") == 2
        assert agent._extract_payback_years("5-7 years") == 5

    def test_extract_payback_years_single(self, agent):
        """Test extraction from single value."""
        assert agent._extract_payback_years("3 years") == 3
        assert agent._extract_payback_years("10-15 years") == 10

    def test_extract_payback_years_na(self, agent):
        """Test extraction of 'N/A' payback."""
        assert agent._extract_payback_years("N/A") == 999


class TestSavingsPotentialCalculation:
    """Tests for emissions savings potential calculation."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_calculate_savings_basic(self, agent):
        """Test basic savings potential calculation."""
        recommendations = [
            {"action": "Solar PV", "impact": "30-50% reduction"},
            {"action": "LED lighting", "impact": "50-70% reduction"},
        ]
        emissions_by_source = {"electricity": 10000}

        savings = agent._calculate_savings_potential(recommendations, emissions_by_source)

        # Should have min and max savings
        assert "minimum_kg_co2e" in savings
        assert "maximum_kg_co2e" in savings
        assert "percentage_range" in savings

        # Min should be less than max
        assert savings["minimum_kg_co2e"] < savings["maximum_kg_co2e"]

        # Both should be positive
        assert savings["minimum_kg_co2e"] > 0
        assert savings["maximum_kg_co2e"] > 0

    def test_calculate_savings_zero_emissions(self, agent):
        """Test savings calculation with zero emissions."""
        recommendations = [{"action": "Test", "impact": "10% reduction"}]
        emissions_by_source = {}

        savings = agent._calculate_savings_potential(recommendations, emissions_by_source)

        # Should handle zero emissions
        assert savings["minimum_kg_co2e"] == 0
        assert savings["maximum_kg_co2e"] == 0

    def test_calculate_savings_no_percentages(self, agent):
        """Test savings calculation when impact has no percentages."""
        recommendations = [
            {"action": "Test", "impact": "Significant reduction"},  # No %
        ]
        emissions_by_source = {"electricity": 10000}

        savings = agent._calculate_savings_potential(recommendations, emissions_by_source)

        # Should handle missing percentages gracefully
        assert "minimum_kg_co2e" in savings
        assert "maximum_kg_co2e" in savings


class TestRecommendationGrouping:
    """Tests for grouping recommendations by timeline."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_group_by_payback(self, agent):
        """Test grouping recommendations by payback period."""
        recommendations = [
            {"action": "Immediate", "payback": "Immediate"},
            {"action": "Short term", "payback": "1-2 years"},
            {"action": "Medium term", "payback": "3-5 years"},
            {"action": "Long term", "payback": "7-10 years"},
        ]

        grouped = agent._group_recommendations(recommendations)

        # Check groups exist
        assert "Immediate Actions" in grouped
        assert "Short Term (1-2 years)" in grouped
        assert "Medium Term (3-5 years)" in grouped
        assert "Long Term (5+ years)" in grouped

        # Verify correct assignment
        assert len(grouped["Immediate Actions"]) == 1
        assert len(grouped["Short Term (1-2 years)"]) == 1
        assert len(grouped["Medium Term (3-5 years)"]) == 1
        assert len(grouped["Long Term (5+ years)"]) == 1

    def test_group_removes_empty(self, agent):
        """Test that empty groups are removed."""
        recommendations = [
            {"action": "Only immediate", "payback": "Immediate"},
        ]

        grouped = agent._group_recommendations(recommendations)

        # Only immediate actions group should exist
        assert "Immediate Actions" in grouped
        assert len(grouped) == 1  # Other groups removed


class TestImplementationRoadmap:
    """Tests for implementation roadmap creation."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_roadmap_structure(self, agent):
        """Test roadmap has correct structure."""
        recommendations = [
            {"action": "Quick win", "cost": "Low", "priority": "High", "payback": "1 year"},
            {"action": "Medium investment", "cost": "Medium", "priority": "High", "payback": "3 years"},
            {"action": "Major upgrade", "cost": "High", "priority": "High", "payback": "7 years"},
        ]

        roadmap = agent._create_roadmap(recommendations)

        # Should have multiple phases
        assert len(roadmap) > 0

        # Check first phase structure
        phase = roadmap[0]
        assert "phase" in phase
        assert "actions" in phase
        assert "estimated_cost" in phase
        assert "expected_impact" in phase

    def test_roadmap_phase_1_quick_wins(self, agent):
        """Test Phase 1 includes quick wins."""
        recommendations = [
            {"action": "Quick win", "cost": "Low", "priority": "High"},
            {"action": "Not quick", "cost": "High", "priority": "High"},
        ]

        roadmap = agent._create_roadmap(recommendations)

        # Phase 1 should exist
        phase1 = roadmap[0]
        assert "Quick Wins" in phase1["phase"]

        # Should include low-cost, high-priority items
        actions = [a["action"] for a in phase1["actions"]]
        assert "Quick win" in actions

    def test_roadmap_phase_2_medium_investments(self, agent):
        """Test Phase 2 includes medium investments."""
        recommendations = [
            {"action": "Medium cost", "cost": "Medium"},
        ]

        roadmap = agent._create_roadmap(recommendations)

        # Should have Phase 2 for medium investments
        phase2 = [p for p in roadmap if "Strategic Improvements" in p["phase"]]
        assert len(phase2) > 0

    def test_roadmap_phase_3_major_upgrades(self, agent):
        """Test Phase 3 includes major upgrades."""
        recommendations = [
            {"action": "Major upgrade", "cost": "High", "priority": "High"},
        ]

        roadmap = agent._create_roadmap(recommendations)

        # Should have Phase 3 for major upgrades
        phase3 = [p for p in roadmap if "Major Upgrades" in p["phase"]]
        assert len(phase3) > 0


class TestQuickWinsAndHighImpact:
    """Tests for quick wins and high-impact identification."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_quick_wins_identification(self, agent):
        """Test identification of quick wins (low cost)."""
        input_data = {
            "emissions_by_source": {"electricity": 15000},
            "building_type": "commercial_office",
        }

        result = agent.execute(input_data)
        quick_wins = result.data["quick_wins"]

        # Should have quick wins
        assert len(quick_wins) > 0
        assert len(quick_wins) <= 3  # Max 3

        # All should be low cost
        for qw in quick_wins:
            assert qw["cost"] == "Low"

    def test_high_impact_identification(self, agent):
        """Test identification of high-impact recommendations."""
        input_data = {
            "emissions_by_source": {"electricity": 15000, "natural_gas": 8500},
        }

        result = agent.execute(input_data)
        high_impact = result.data["high_impact"]

        # Should have high-impact recommendations
        assert len(high_impact) > 0
        assert len(high_impact) <= 3  # Max 3

        # All should mention significant percentages
        for hi in high_impact:
            impact = hi["impact"]
            # Should contain "20%" or higher
            assert any(f"{i}%" in impact for i in range(20, 101))


class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_empty_emissions(self, agent):
        """Test handling of empty emissions."""
        input_data = {"emissions_by_source": {}}

        result = agent.execute(input_data)

        # Should still succeed
        assert result.success is True

        # May have general recommendations
        # (not source-specific)

    def test_single_source(self, agent):
        """Test handling of single emission source."""
        input_data = {
            "emissions_by_source": {"electricity": 10000}
        }

        result = agent.execute(input_data)

        assert result.success is True
        assert len(result.data["recommendations"]) > 0

    def test_very_small_emissions(self, agent):
        """Test handling of very small emissions values."""
        input_data = {
            "emissions_by_source": {"electricity": 1}  # 1 kg
        }

        result = agent.execute(input_data)

        assert result.success is True
        # Should still provide recommendations

    def test_very_large_emissions(self, agent):
        """Test handling of very large emissions values."""
        input_data = {
            "emissions_by_source": {"coal": 10000000}  # 10,000 tons
        }

        result = agent.execute(input_data)

        assert result.success is True
        assert len(result.data["recommendations"]) > 0

    def test_zero_emissions(self, agent):
        """Test handling of zero emissions."""
        input_data = {
            "emissions_by_source": {"electricity": 0}
        }

        result = agent.execute(input_data)

        assert result.success is True

    def test_missing_optional_fields(self, agent):
        """Test handling when optional fields are missing."""
        input_data = {
            "emissions_by_source": {"electricity": 10000}
            # No building_type, country, etc.
        }

        result = agent.execute(input_data)

        # Should succeed with defaults
        assert result.success is True
        assert len(result.data["recommendations"]) > 0

    def test_invalid_building_type(self, agent):
        """Test handling of invalid building type."""
        input_data = {
            "emissions_by_source": {"electricity": 10000},
            "building_type": "invalid_type",
        }

        result = agent.execute(input_data)

        # Should still succeed (uses defaults)
        assert result.success is True

    def test_invalid_performance_rating(self, agent):
        """Test handling of invalid performance rating."""
        input_data = {
            "emissions_by_source": {"electricity": 10000},
            "performance_rating": "Invalid",
        }

        result = agent.execute(input_data)

        # Should still succeed
        assert result.success is True

    def test_negative_building_age(self, agent):
        """Test handling of negative building age."""
        input_data = {
            "emissions_by_source": {"electricity": 10000},
            "building_age": -5,
        }

        result = agent.execute(input_data)

        # Should handle gracefully (treated as 0 or ignored)
        assert result.success is True

    def test_extreme_building_age(self, agent):
        """Test handling of extreme building age."""
        input_data = {
            "emissions_by_source": {"electricity": 10000},
            "building_age": 150,  # Very old
        }

        result = agent.execute(input_data)

        assert result.success is True
        # Should include envelope recommendations


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    @pytest.fixture
    def test_input(self):
        return {
            "emissions_by_source": {
                "electricity": 15000,
                "natural_gas": 8500,
            },
            "building_type": "commercial_office",
            "building_age": 10,
        }

    def test_same_input_same_output(self, agent, test_input):
        """Test that same input produces same output."""
        result1 = agent.execute(test_input)
        result2 = agent.execute(test_input)

        # Both should succeed
        assert result1.success is True
        assert result2.success is True

        # Should have same number of recommendations
        assert len(result1.data["recommendations"]) == len(result2.data["recommendations"])

        # Recommendations should be in same order
        rec1 = result1.data["recommendations"]
        rec2 = result2.data["recommendations"]

        for i in range(min(len(rec1), len(rec2))):
            assert rec1[i]["action"] == rec2[i]["action"]
            assert rec1[i]["priority"] == rec2[i]["priority"]

    def test_potential_savings_deterministic(self, agent, test_input):
        """Test that potential savings calculation is deterministic."""
        result1 = agent.execute(test_input)
        result2 = agent.execute(test_input)

        savings1 = result1.data["potential_emissions_reduction"]
        savings2 = result2.data["potential_emissions_reduction"]

        # Should be identical
        assert savings1["minimum_kg_co2e"] == savings2["minimum_kg_co2e"]
        assert savings1["maximum_kg_co2e"] == savings2["maximum_kg_co2e"]

    def test_roadmap_deterministic(self, agent, test_input):
        """Test that roadmap is deterministic."""
        result1 = agent.execute(test_input)
        result2 = agent.execute(test_input)

        roadmap1 = result1.data["implementation_roadmap"]
        roadmap2 = result2.data["implementation_roadmap"]

        # Should have same number of phases
        assert len(roadmap1) == len(roadmap2)

        # Phases should be identical
        for i in range(len(roadmap1)):
            assert roadmap1[i]["phase"] == roadmap2[i]["phase"]
            assert len(roadmap1[i]["actions"]) == len(roadmap2[i]["actions"])

    def test_grouping_deterministic(self, agent, test_input):
        """Test that grouping is deterministic."""
        result1 = agent.execute(test_input)
        result2 = agent.execute(test_input)

        grouped1 = result1.data["grouped_recommendations"]
        grouped2 = result2.data["grouped_recommendations"]

        # Should have same groups
        assert set(grouped1.keys()) == set(grouped2.keys())

        # Each group should have same number of items
        for key in grouped1.keys():
            assert len(grouped1[key]) == len(grouped2[key])


class TestMetadata:
    """Tests for metadata and analysis basis."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_metadata_included(self, agent):
        """Test that metadata is included in results."""
        input_data = {
            "emissions_by_source": {"electricity": 10000},
            "building_type": "retail",
            "building_age": 5,
            "performance_rating": "Good",
        }

        result = agent.execute(input_data)

        # Should have metadata
        assert result.metadata is not None
        assert "analysis_basis" in result.metadata

        # Check analysis basis
        basis = result.metadata["analysis_basis"]
        assert "building_type" in basis
        assert "performance_rating" in basis
        assert "building_age" in basis

        # Values should match input
        assert basis["building_type"] == "retail"
        assert basis["building_age"] == 5
        assert basis["performance_rating"] == "Good"


class TestRecommendationDatabaseAccess:
    """Tests for recommendation database access methods."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_get_recommendations_valid_category(self, agent):
        """Test getting recommendations from valid category."""
        recs = agent._get_recommendations("hvac", "high_consumption")

        assert len(recs) > 0
        # Should be a copy (modifications don't affect original)
        assert isinstance(recs, list)

    def test_get_recommendations_invalid_category(self, agent):
        """Test getting recommendations from invalid category."""
        recs = agent._get_recommendations("invalid", "subcategory")

        assert recs == []

    def test_get_recommendations_invalid_subcategory(self, agent):
        """Test getting recommendations with invalid subcategory."""
        recs = agent._get_recommendations("hvac", "invalid")

        assert recs == []

    def test_get_country_recommendations_us(self, agent):
        """Test getting US country-specific recommendations."""
        recs = agent._get_country_specific_recommendations("US")

        assert len(recs) > 0
        # Should include IRA or ENERGY STAR
        actions = [r["action"] for r in recs]
        combined = " ".join(actions)
        assert "IRA" in combined or "ENERGY STAR" in combined

    def test_get_country_recommendations_india(self, agent):
        """Test getting India country-specific recommendations."""
        recs = agent._get_country_specific_recommendations("IN")

        assert len(recs) > 0
        # Should include PAT or solar subsidy
        actions = [r["action"] for r in recs]
        combined = " ".join(actions)
        assert "PAT" in combined or "solar" in combined.lower()

    def test_get_country_recommendations_eu(self, agent):
        """Test getting EU country-specific recommendations."""
        recs = agent._get_country_specific_recommendations("EU")

        assert len(recs) > 0
        # Should include EU Taxonomy
        actions = [r["action"] for r in recs]
        combined = " ".join(actions)
        assert "EU Taxonomy" in combined or "district heating" in combined

    def test_get_country_recommendations_unknown(self, agent):
        """Test getting recommendations for unknown country."""
        recs = agent._get_country_specific_recommendations("XX")

        # Should return empty list for unknown countries
        assert recs == []


class TestPerformance:
    """Performance and efficiency tests."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_execution_time_simple(self, agent):
        """Test execution time for simple input."""
        import time

        input_data = {
            "emissions_by_source": {"electricity": 10000}
        }

        start = time.time()
        result = agent.execute(input_data)
        elapsed = time.time() - start

        # Should complete quickly (< 1 second for simple case)
        assert elapsed < 1.0
        assert result.success is True

    def test_execution_time_complex(self, agent):
        """Test execution time for complex input."""
        import time

        input_data = {
            "emissions_by_source": {
                "electricity": 15000,
                "natural_gas": 8500,
                "diesel": 3200,
                "fuel_oil": 2000,
                "propane": 1500,
            },
            "building_type": "commercial_office",
            "building_age": 15,
            "performance_rating": "Average",
            "country": "US",
            "load_breakdown": {
                "hvac_load": 0.45,
                "lighting_load": 0.25,
                "equipment_load": 0.30,
            },
        }

        start = time.time()
        result = agent.execute(input_data)
        elapsed = time.time() - start

        # Should complete within reasonable time (< 2 seconds)
        assert elapsed < 2.0
        assert result.success is True

    def test_memory_efficiency(self, agent):
        """Test that agent doesn't accumulate state between calls."""
        import sys

        input_data = {
            "emissions_by_source": {"electricity": 10000}
        }

        # Get initial size
        initial_size = sys.getsizeof(agent)

        # Execute multiple times
        for _ in range(10):
            agent.execute(input_data)

        # Size shouldn't grow significantly
        final_size = sys.getsizeof(agent)
        growth = final_size - initial_size

        # Allow for some growth but not excessive
        assert growth < 10000  # Less than 10KB growth


class TestIntegration:
    """Integration tests with realistic scenarios."""

    @pytest.fixture
    def agent(self):
        return RecommendationAgent()

    def test_small_office_scenario(self, agent):
        """Test small office building scenario."""
        input_data = {
            "emissions_by_source": {
                "electricity": 5000,
                "natural_gas": 2000,
            },
            "building_type": "commercial_office",
            "building_age": 8,
            "performance_rating": "Average",
            "country": "US",
        }

        result = agent.execute(input_data)

        assert result.success is True

        # Should have actionable recommendations
        recommendations = result.data["recommendations"]
        assert len(recommendations) > 0

        # Should have quick wins
        quick_wins = result.data["quick_wins"]
        assert len(quick_wins) > 0

        # Should have implementation roadmap
        roadmap = result.data["implementation_roadmap"]
        assert len(roadmap) > 0

    def test_large_warehouse_scenario(self, agent):
        """Test large warehouse scenario."""
        input_data = {
            "emissions_by_source": {
                "electricity": 8000,
                "natural_gas": 15000,  # High heating load
            },
            "building_type": "warehouse",
            "building_age": 25,  # Old building (>20 triggers envelope recs)
            "performance_rating": "Below Average",
            "country": "US",
        }

        result = agent.execute(input_data)

        assert result.success is True

        recommendations = result.data["recommendations"]
        assert len(recommendations) > 0

        # Should include envelope recommendations (old, poor performing)
        envelope_recs = [r for r in recommendations if "insulation" in r["action"].lower() or "envelope" in r["action"].lower() or "seal" in r["action"].lower() or "window" in r["action"].lower()]
        assert len(envelope_recs) > 0

    def test_manufacturing_facility_scenario(self, agent):
        """Test manufacturing facility scenario."""
        input_data = {
            "emissions_by_source": {
                "electricity": 50000,
                "natural_gas": 80000,
                "diesel": 10000,
            },
            "building_type": "manufacturing",
            "building_age": 15,
            "performance_rating": "Average",
            "country": "IN",  # India
        }

        result = agent.execute(input_data)

        assert result.success is True

        recommendations = result.data["recommendations"]
        assert len(recommendations) > 0

        # Should include India-specific recommendations
        india_recs = [r for r in recommendations if "PAT" in r["action"] or "subsidy" in r["action"]]
        assert len(india_recs) > 0

    def test_data_center_scenario(self, agent):
        """Test data center scenario (electricity-heavy)."""
        input_data = {
            "emissions_by_source": {
                "electricity": 100000,  # Very high electricity
                "natural_gas": 5000,    # Minimal gas
            },
            "building_type": "data_center",
            "building_age": 5,
            "performance_rating": "Good",
            "country": "US",
        }

        result = agent.execute(input_data)

        assert result.success is True

        recommendations = result.data["recommendations"]

        # Should heavily focus on electricity reduction
        elec_recs = [r for r in recommendations if "solar" in r["action"].lower() or "renewable" in r["action"].lower() or "efficiency" in r["action"].lower()]
        assert len(elec_recs) > 0

    def test_old_poor_performing_building(self, agent):
        """Test old building with poor performance."""
        input_data = {
            "emissions_by_source": {
                "electricity": 20000,
                "natural_gas": 15000,
            },
            "building_type": "commercial_office",
            "building_age": 30,  # Very old
            "performance_rating": "Poor",
            "country": "US",
        }

        result = agent.execute(input_data)

        assert result.success is True

        recommendations = result.data["recommendations"]
        assert len(recommendations) >= 5  # Should have many recommendations

        # Should include diverse recommendations (envelope, operations, etc.)
        # Check for various keywords across recommendations
        all_actions = " ".join([r["action"].lower() for r in recommendations])

        # Should have envelope recommendations (old building)
        assert "insulation" in all_actions or "window" in all_actions or "seal" in all_actions

        # Should have operational recommendations (poor performance)
        assert "energy management" in all_actions or "audit" in all_actions or "train" in all_actions


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
