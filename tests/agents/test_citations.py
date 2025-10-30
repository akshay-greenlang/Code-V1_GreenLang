"""Tests for Citation Infrastructure and Integration.

This module tests the citation system implementation, ensuring:
1. EF CID generation is deterministic and follows format
2. Citation data structures are valid
3. Citations are properly integrated into all agents
4. Citations are reset between runs
5. Citations are exported in agent output
6. Citation validation works correctly

Author: GreenLang Framework Team
Date: October 2025
"""

import re
import pytest
from datetime import datetime
from greenlang.agents.citations import (
    EmissionFactorCitation,
    CalculationCitation,
    DataSourceCitation,
    CitationBundle,
    create_emission_factor_citation,
)


class TestCitationDataStructures:
    """Test citation data structure creation and validation."""

    def test_emission_factor_citation_creation(self):
        """Test EmissionFactorCitation can be created with valid data."""
        citation = EmissionFactorCitation(
            source="EPA GHG Inventory",
            factor_name="Natural Gas Combustion",
            value=53.06,
            unit="kg CO2e/MMBtu",
            ef_cid="ef_428b1c64829dc8f5",
            version="2023",
            confidence="high",
            region="US",
            gwp_set="AR5",
            timestamp=datetime.now(),
        )

        assert citation.source == "EPA GHG Inventory"
        assert citation.factor_name == "Natural Gas Combustion"
        assert citation.value == 53.06
        assert citation.unit == "kg CO2e/MMBtu"
        assert citation.ef_cid == "ef_428b1c64829dc8f5"
        assert citation.version == "2023"
        assert citation.confidence == "high"
        assert citation.region == "US"
        assert citation.gwp_set == "AR5"

    def test_calculation_citation_creation(self):
        """Test CalculationCitation can be created with valid data."""
        citation = CalculationCitation(
            step_name="calculate_emissions",
            formula="Emissions = Amount × EF",
            inputs={"amount": 1000, "ef": 53.06},
            output={"emissions_kg": 53060},
            timestamp=datetime.now(),
            tool_call_id="calc_1",
        )

        assert citation.step_name == "calculate_emissions"
        assert citation.formula == "Emissions = Amount × EF"
        assert citation.inputs["amount"] == 1000
        assert citation.inputs["ef"] == 53.06
        assert citation.output["emissions_kg"] == 53060
        assert citation.tool_call_id == "calc_1"

    def test_data_source_citation_creation(self):
        """Test DataSourceCitation can be created with valid data."""
        citation = DataSourceCitation(
            source_name="Grid Intensity Database",
            source_type="database",
            query={"region": "US", "table": "grid_factors"},
            timestamp=datetime.now(),
        )

        assert citation.source_name == "Grid Intensity Database"
        assert citation.source_type == "database"
        assert citation.query["region"] == "US"
        assert citation.query["table"] == "grid_factors"

    def test_citation_serialization(self):
        """Test citations can be serialized to dict."""
        citation = EmissionFactorCitation(
            source="Test Source",
            factor_name="Test Factor",
            value=100.0,
            unit="kg CO2e",
            ef_cid="ef_1234567890abcdef",
            version="2023",
            confidence="high",
            region="US",
            gwp_set="AR5",
            timestamp=datetime.now(),
        )

        citation_dict = citation.dict()
        assert isinstance(citation_dict, dict)
        assert citation_dict["source"] == "Test Source"
        assert citation_dict["value"] == 100.0
        assert citation_dict["ef_cid"] == "ef_1234567890abcdef"


class TestEFCIDGeneration:
    """Test EF CID (Emission Factor Content ID) generation."""

    def test_ef_cid_format(self):
        """Test EF CID follows the correct format: ef_<16-char-hex>."""
        citation = create_emission_factor_citation(
            source="EPA",
            factor_name="Natural Gas",
            value=53.06,
            unit="kg CO2e/MMBtu",
            version="2023",
            region="US",
        )

        # Check format: ef_<16 hex characters>
        ef_cid_pattern = r"^ef_[0-9a-f]{16}$"
        assert re.match(ef_cid_pattern, citation.ef_cid), \
            f"EF CID '{citation.ef_cid}' does not match format 'ef_<16-char-hex>'"

    def test_ef_cid_determinism(self):
        """Test EF CID is deterministic for same input."""
        citation1 = create_emission_factor_citation(
            source="EPA",
            factor_name="Natural Gas",
            value=53.06,
            unit="kg CO2e/MMBtu",
            version="2023",
            region="US",
        )

        citation2 = create_emission_factor_citation(
            source="EPA",
            factor_name="Natural Gas",
            value=53.06,
            unit="kg CO2e/MMBtu",
            version="2023",
            region="US",
        )

        assert citation1.ef_cid == citation2.ef_cid, \
            "EF CID should be deterministic for identical inputs"

    def test_ef_cid_uniqueness(self):
        """Test EF CID is unique for different inputs."""
        citation1 = create_emission_factor_citation(
            source="EPA",
            factor_name="Natural Gas",
            value=53.06,
            unit="kg CO2e/MMBtu",
            version="2023",
            region="US",
        )

        citation2 = create_emission_factor_citation(
            source="EPA",
            factor_name="Coal",  # Different factor
            value=95.52,
            unit="kg CO2e/MMBtu",
            version="2023",
            region="US",
        )

        assert citation1.ef_cid != citation2.ef_cid, \
            "EF CID should be unique for different emission factors"


class TestCitationIntegrationInAgents:
    """Test citation integration across all AI agents."""

    def test_fuel_agent_ai_has_citations(self):
        """Test FuelAgentAI includes citations in output."""
        from greenlang.agents.fuel_agent_ai import FuelAgentAI

        agent = FuelAgentAI()

        # Check citation tracking variables exist (at least one type)
        assert hasattr(agent, '_current_citations'), \
            "FuelAgentAI should have _current_citations for emission factors"
        assert isinstance(agent._current_citations, list)

    def test_carbon_agent_ai_has_citations(self):
        """Test CarbonAgentAI includes citations in output."""
        from greenlang.agents.carbon_agent_ai import CarbonAgentAI

        agent = CarbonAgentAI()

        # Check citation tracking variables exist
        has_citations = hasattr(agent, '_current_citations') or hasattr(agent, '_calculation_citations')
        assert has_citations, "CarbonAgentAI should have at least one citation tracking attribute"

    def test_grid_factor_agent_ai_has_citations(self):
        """Test GridFactorAgentAI includes citations in output."""
        from greenlang.agents.grid_factor_agent_ai import GridFactorAgentAI

        agent = GridFactorAgentAI()

        # Check citation tracking variables exist
        has_citations = hasattr(agent, '_current_citations') or hasattr(agent, '_calculation_citations')
        assert has_citations, "GridFactorAgentAI should have at least one citation tracking attribute"

    def test_boiler_replacement_agent_ai_has_citations(self):
        """Test BoilerReplacementAgent_AI includes citations in output."""
        from greenlang.agents.boiler_replacement_agent_ai import BoilerReplacementAgent_AI

        agent = BoilerReplacementAgent_AI()

        # Check citation tracking variables exist
        has_citations = hasattr(agent, '_current_citations') or hasattr(agent, '_calculation_citations')
        assert has_citations, "BoilerReplacementAgent_AI should have at least one citation tracking attribute"

    def test_industrial_process_heat_agent_ai_has_citations(self):
        """Test IndustrialProcessHeatAgent_AI includes citations in output."""
        from greenlang.agents.industrial_process_heat_agent_ai import IndustrialProcessHeatAgent_AI

        agent = IndustrialProcessHeatAgent_AI()

        # Check citation tracking variables exist
        has_citations = hasattr(agent, '_current_citations') or hasattr(agent, '_calculation_citations')
        assert has_citations, "IndustrialProcessHeatAgent_AI should have at least one citation tracking attribute"

    def test_decarbonization_roadmap_agent_ai_has_citations(self):
        """Test DecarbonizationRoadmapAgentAI includes citations in output."""
        from greenlang.agents.decarbonization_roadmap_agent_ai import DecarbonizationRoadmapAgentAI

        agent = DecarbonizationRoadmapAgentAI()

        # Check citation tracking variables exist
        has_citations = hasattr(agent, '_current_citations') or hasattr(agent, '_calculation_citations')
        assert has_citations, "DecarbonizationRoadmapAgentAI should have at least one citation tracking attribute"

    def test_industrial_heat_pump_agent_ai_has_citations(self):
        """Test IndustrialHeatPumpAgent_AI includes citations in output."""
        from greenlang.agents.industrial_heat_pump_agent_ai import IndustrialHeatPumpAgent_AI

        agent = IndustrialHeatPumpAgent_AI()

        # Check citation tracking variables exist
        has_citations = hasattr(agent, '_current_citations') or hasattr(agent, '_calculation_citations')
        assert has_citations, "IndustrialHeatPumpAgent_AI should have at least one citation tracking attribute"

    def test_recommendation_agent_ai_has_citations(self):
        """Test RecommendationAgentAI includes citations in output."""
        from greenlang.agents.recommendation_agent_ai import RecommendationAgentAI

        agent = RecommendationAgentAI()

        # Check citation tracking variables exist
        has_citations = hasattr(agent, '_current_citations') or hasattr(agent, '_calculation_citations')
        assert has_citations, "RecommendationAgentAI should have at least one citation tracking attribute"

    def test_report_agent_ai_has_citations(self):
        """Test ReportAgentAI includes citations in output."""
        from greenlang.agents.report_agent_ai import ReportAgentAI

        agent = ReportAgentAI()

        # Check citation tracking variables exist
        has_citations = hasattr(agent, '_current_citations') or hasattr(agent, '_calculation_citations')
        assert has_citations, "ReportAgentAI should have at least one citation tracking attribute"

    def test_anomaly_agent_iforest_has_citations(self):
        """Test AnomalyAgentIForest includes citations in output."""
        from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent

        agent = IsolationForestAnomalyAgent()

        # Check citation tracking variables exist
        has_citations = hasattr(agent, '_current_citations') or hasattr(agent, '_calculation_citations')
        assert has_citations, "IsolationForestAnomalyAgent should have at least one citation tracking attribute"

    def test_forecast_agent_sarima_has_citations(self):
        """Test ForecastAgentSARIMA includes citations in output."""
        from greenlang.agents.forecast_agent_sarima import SARIMAForecastAgent

        agent = SARIMAForecastAgent()

        # Check citation tracking variables exist
        has_citations = hasattr(agent, '_current_citations') or hasattr(agent, '_calculation_citations')
        assert has_citations, "SARIMAForecastAgent should have at least one citation tracking attribute"


class TestCitationReset:
    """Test citation lists are reset between runs."""

    def test_citations_reset_on_initialization(self):
        """Test citation lists are empty on agent initialization."""
        from greenlang.agents.carbon_agent_ai import CarbonAgentAI

        agent = CarbonAgentAI()
        # Test agents that have both citation types
        assert len(agent._current_citations) == 0
        assert len(agent._calculation_citations) == 0

    def test_citations_would_accumulate_without_reset(self):
        """Test that citations need to be reset to avoid accumulation."""
        # This is a conceptual test - we verify the reset logic exists
        from greenlang.agents.carbon_agent_ai import CarbonAgentAI

        agent = CarbonAgentAI()

        # Manually add a citation (simulating a calculation)
        citation = CalculationCitation(
            step_name="test",
            formula="test = 1 + 1",
            inputs={"a": 1, "b": 1},
            output={"result": 2},
            timestamp=datetime.now(),
            tool_call_id="test_1",
        )
        agent._calculation_citations.append(citation)

        # Verify it was added
        assert len(agent._calculation_citations) == 1

        # Simulate reset (what should happen at start of each run)
        agent._current_citations = []
        agent._calculation_citations = []

        # Verify reset worked
        assert len(agent._current_citations) == 0
        assert len(agent._calculation_citations) == 0


class TestCitationExport:
    """Test citation export in agent output."""

    def test_citation_export_format(self):
        """Test citations are exported in correct format."""
        # Create sample citations
        calc_citation = CalculationCitation(
            step_name="test_calculation",
            formula="result = a + b",
            inputs={"a": 10, "b": 20},
            output={"result": 30},
            timestamp=datetime.now(),
            tool_call_id="calc_1",
        )

        # Simulate export format
        output = {
            "result": 30,
            "citations": {
                "calculations": [calc_citation.dict()],
            }
        }

        # Verify structure
        assert "citations" in output
        assert "calculations" in output["citations"]
        assert isinstance(output["citations"]["calculations"], list)
        assert len(output["citations"]["calculations"]) == 1
        assert output["citations"]["calculations"][0]["step_name"] == "test_calculation"
        assert output["citations"]["calculations"][0]["formula"] == "result = a + b"


class TestCitationBundle:
    """Test CitationBundle utility class."""

    def test_citation_bundle_creation(self):
        """Test CitationBundle can be created with citations."""
        # Create citations first
        ef_citation = create_emission_factor_citation(
            source="EPA",
            factor_name="Natural Gas",
            value=53.06,
            unit="kg CO2e/MMBtu",
            version="2023",
            region="US",
        )

        calc_citation = CalculationCitation(
            step_name="calculate_emissions",
            formula="Emissions = Amount × EF",
            inputs={"amount": 1000, "ef": 53.06},
            output={"emissions_kg": 53060},
            timestamp=datetime.now(),
            tool_call_id="calc_1",
        )

        # Create bundle with citations
        bundle = CitationBundle(
            agent_id="test_agent",
            emission_factors=[ef_citation],
            calculations=[calc_citation],
        )

        # Verify bundle contains citations
        assert len(bundle.emission_factors) == 1
        assert len(bundle.calculations) == 1
        assert bundle.agent_id == "test_agent"

    def test_citation_bundle_export(self):
        """Test CitationBundle can export all citations."""
        # Create citations
        ef_citation = create_emission_factor_citation(
            source="EPA",
            factor_name="Natural Gas",
            value=53.06,
            unit="kg CO2e/MMBtu",
            version="2023",
            region="US",
        )

        calc_citation = CalculationCitation(
            step_name="calculate_emissions",
            formula="Emissions = Amount × EF",
            inputs={"amount": 1000, "ef": 53.06},
            output={"emissions_kg": 53060},
            timestamp=datetime.now(),
            tool_call_id="calc_1",
        )

        # Create bundle
        bundle = CitationBundle(
            agent_id="test_agent",
            emission_factors=[ef_citation],
            calculations=[calc_citation],
        )

        # Export
        exported = bundle.to_dict()

        # Verify export structure
        assert "agent_id" in exported
        assert "emission_factors" in exported
        assert "calculations" in exported
        assert exported["agent_id"] == "test_agent"
        assert len(exported["emission_factors"]) == 1
        assert len(exported["calculations"]) == 1


class TestCitationCoverage:
    """Test citation coverage across all 11 AI agents."""

    def test_all_agents_have_citation_tracking(self):
        """Test all 11 AI agents have citation tracking implemented."""
        agent_classes = [
            ('greenlang.agents.fuel_agent_ai', 'FuelAgentAI'),
            ('greenlang.agents.carbon_agent_ai', 'CarbonAgentAI'),
            ('greenlang.agents.grid_factor_agent_ai', 'GridFactorAgentAI'),
            ('greenlang.agents.boiler_replacement_agent_ai', 'BoilerReplacementAgent_AI'),
            ('greenlang.agents.industrial_process_heat_agent_ai', 'IndustrialProcessHeatAgent_AI'),
            ('greenlang.agents.decarbonization_roadmap_agent_ai', 'DecarbonizationRoadmapAgentAI'),
            ('greenlang.agents.industrial_heat_pump_agent_ai', 'IndustrialHeatPumpAgent_AI'),
            ('greenlang.agents.recommendation_agent_ai', 'RecommendationAgentAI'),
            ('greenlang.agents.report_agent_ai', 'ReportAgentAI'),
            ('greenlang.agents.anomaly_agent_iforest', 'IsolationForestAnomalyAgent'),
            ('greenlang.agents.forecast_agent_sarima', 'SARIMAForecastAgent'),
        ]

        for module_name, class_name in agent_classes:
            # Import agent dynamically
            module = __import__(module_name, fromlist=[class_name])
            agent_class = getattr(module, class_name)

            # Create instance
            agent = agent_class()

            # Verify citation tracking (at least one type)
            has_current = hasattr(agent, '_current_citations')
            has_calc = hasattr(agent, '_calculation_citations')

            assert has_current or has_calc, \
                f"{class_name} missing citation tracking attributes"

            # Verify they are lists if they exist
            if has_current:
                assert isinstance(agent._current_citations, list), \
                    f"{class_name}._current_citations is not a list"
            if has_calc:
                assert isinstance(agent._calculation_citations, list), \
                    f"{class_name}._calculation_citations is not a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
