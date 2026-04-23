"""
Tests for Engineering Rationale Generator Module

Comprehensive tests for EngineeringRationaleGenerator including:
- Rationale generation for different calculation types
- Standard citations and principles
- Compliance framework mapping
- Markdown formatting

Author: GreenLang AI Team
"""

import pytest
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from explainability.explanation_schemas import (
    StandardSource,
    StandardCitation,
    ThermodynamicPrinciple,
    EngineeringRationale,
)
from explainability.engineering_rationale import (
    CalculationType,
    EngineeringRationaleGenerator,
    RationaleConfig,
    STANDARD_CITATIONS,
    THERMODYNAMIC_PRINCIPLES,
    get_all_standard_sources,
    get_citations_by_source,
    format_principle_as_markdown,
    format_rationale_as_markdown,
)


# Test fixtures

@pytest.fixture
def generator():
    """Create a rationale generator."""
    return EngineeringRationaleGenerator()


@pytest.fixture
def custom_config():
    """Create custom configuration."""
    return RationaleConfig(
        include_formulas=True,
        include_citations=True,
        include_assumptions=True,
        include_limitations=True,
        max_citations=5
    )


@pytest.fixture
def sample_inputs():
    """Sample calculation inputs."""
    return {
        "fuel_type": "natural_gas",
        "mass_flow": 100.0,
        "temperature": 350.0,
        "pressure": 101325.0
    }


@pytest.fixture
def sample_outputs():
    """Sample calculation outputs."""
    return {
        "efficiency": 0.85,
        "emissions": 1200.0,
        "heat_output": 5000.0
    }


class TestRationaleConfig:
    """Tests for RationaleConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RationaleConfig()

        assert config.include_formulas is True
        assert config.include_citations is True
        assert config.include_assumptions is True
        assert config.include_limitations is True
        assert config.max_citations == 10
        assert config.confidence_level == 0.95

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RationaleConfig(
            include_formulas=False,
            max_citations=5,
            confidence_level=0.99
        )

        assert config.include_formulas is False
        assert config.max_citations == 5
        assert config.confidence_level == 0.99


class TestEngineeringRationaleGenerator:
    """Tests for EngineeringRationaleGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = EngineeringRationaleGenerator(
            agent_id="GL-TEST",
            agent_version="1.0.0"
        )

        assert generator.agent_id == "GL-TEST"
        assert generator.agent_version == "1.0.0"

    def test_initialization_with_config(self, custom_config):
        """Test generator initialization with custom config."""
        generator = EngineeringRationaleGenerator(config=custom_config)

        assert generator.config.max_citations == 5

    def test_generate_rationale_combustion(
        self,
        generator,
        sample_inputs,
        sample_outputs
    ):
        """Test rationale generation for combustion calculation."""
        rationale = generator.generate_rationale(
            calculation_type=CalculationType.COMBUSTION,
            inputs=sample_inputs,
            outputs=sample_outputs
        )

        assert rationale.calculation_type == "combustion"
        assert rationale.validation_status == "PASS"
        assert len(rationale.methodology) > 0
        assert len(rationale.citations) > 0
        assert rationale.provenance_hash is not None

    def test_generate_rationale_emission(
        self,
        generator,
        sample_inputs,
        sample_outputs
    ):
        """Test rationale generation for emission calculation."""
        rationale = generator.generate_rationale(
            calculation_type=CalculationType.EMISSION_CALCULATION,
            inputs=sample_inputs,
            outputs=sample_outputs
        )

        assert rationale.calculation_type == "emission_calculation"
        assert len(rationale.citations) > 0
        # Should include EPA and IPCC citations
        citation_sources = [c.source for c in rationale.citations]
        assert StandardSource.EPA in citation_sources or StandardSource.IPCC in citation_sources

    def test_generate_rationale_steam_properties(
        self,
        generator,
        sample_inputs,
        sample_outputs
    ):
        """Test rationale generation for steam properties calculation."""
        rationale = generator.generate_rationale(
            calculation_type=CalculationType.STEAM_PROPERTIES,
            inputs=sample_inputs,
            outputs=sample_outputs
        )

        assert rationale.calculation_type == "steam_properties"
        # Should include IAPWS citations
        citation_sources = [c.source for c in rationale.citations]
        assert StandardSource.IAPWS in citation_sources

    def test_generate_rationale_with_custom_assumptions(
        self,
        generator,
        sample_inputs,
        sample_outputs
    ):
        """Test rationale generation with custom assumptions."""
        custom_assumptions = [
            "Steady-state operation assumed",
            "No fouling present"
        ]

        rationale = generator.generate_rationale(
            calculation_type=CalculationType.HEAT_TRANSFER,
            inputs=sample_inputs,
            outputs=sample_outputs,
            custom_assumptions=custom_assumptions
        )

        assert "Steady-state operation assumed" in rationale.assumptions
        assert "No fouling present" in rationale.assumptions

    def test_generate_rationale_with_custom_limitations(
        self,
        generator,
        sample_inputs,
        sample_outputs
    ):
        """Test rationale generation with custom limitations."""
        custom_limitations = [
            "Results valid for operating range only"
        ]

        rationale = generator.generate_rationale(
            calculation_type=CalculationType.EFFICIENCY_CALCULATION,
            inputs=sample_inputs,
            outputs=sample_outputs,
            custom_limitations=custom_limitations
        )

        assert "Results valid for operating range only" in rationale.limitations

    def test_generate_rationale_with_additional_citations(
        self,
        generator,
        sample_inputs,
        sample_outputs
    ):
        """Test rationale generation with additional citations."""
        additional_citation = StandardCitation(
            source=StandardSource.ISO,
            standard_id="ISO 9001",
            year=2015,
            title="Quality Management Systems"
        )

        rationale = generator.generate_rationale(
            calculation_type=CalculationType.EFFICIENCY_CALCULATION,
            inputs=sample_inputs,
            outputs=sample_outputs,
            additional_citations=[additional_citation]
        )

        citation_ids = [c.standard_id for c in rationale.citations]
        assert "ISO 9001" in citation_ids

    def test_generate_rationale_validation_status_fail(self, generator):
        """Test rationale with empty outputs fails validation."""
        rationale = generator.generate_rationale(
            calculation_type=CalculationType.COMBUSTION,
            inputs={"fuel": "gas"},
            outputs={}  # Empty outputs
        )

        assert rationale.validation_status == "FAIL"

    def test_generate_thermodynamic_explanation(self, generator):
        """Test thermodynamic explanation generation."""
        explanation = generator.generate_thermodynamic_explanation(
            principle_key="first_law_energy_balance",
            inputs={"Q": 1000, "W": 200},
            result=800
        )

        assert "First Law" in explanation["principle_name"]
        assert "formula" in explanation
        assert explanation["result"] == 800

    def test_generate_thermodynamic_explanation_unknown_principle(self, generator):
        """Test thermodynamic explanation with unknown principle."""
        with pytest.raises(ValueError, match="Unknown principle"):
            generator.generate_thermodynamic_explanation(
                principle_key="unknown_principle",
                inputs={},
                result=0
            )

    def test_get_standard_citation(self, generator):
        """Test getting a standard citation."""
        citation = generator.get_standard_citation("ASME_PTC_4")

        assert citation is not None
        assert citation.source == StandardSource.ASME
        assert "PTC 4" in citation.standard_id

    def test_get_standard_citation_unknown(self, generator):
        """Test getting unknown citation returns None."""
        citation = generator.get_standard_citation("UNKNOWN_STANDARD")

        assert citation is None

    def test_get_principle(self, generator):
        """Test getting a thermodynamic principle."""
        principle = generator.get_principle("first_law_energy_balance")

        assert principle is not None
        assert "First Law" in principle.name

    def test_get_principle_unknown(self, generator):
        """Test getting unknown principle returns None."""
        principle = generator.get_principle("unknown_principle")

        assert principle is None

    def test_format_citation_list(self, generator):
        """Test citation list formatting."""
        citations = [
            STANDARD_CITATIONS["ASME_PTC_4"],
            STANDARD_CITATIONS["EPA_AP42"]
        ]

        formatted = generator.format_citation_list(citations)

        assert "ASME" in formatted
        assert "EPA" in formatted
        assert ";" in formatted

    def test_generate_compliance_mapping_emissions(self, generator):
        """Test compliance mapping for emission calculations."""
        mapping = generator.generate_compliance_mapping(
            CalculationType.EMISSION_CALCULATION
        )

        assert "GHG Protocol" in mapping
        assert "ISO 14064" in mapping
        assert "EPA GHGRP" in mapping
        assert len(mapping["GHG Protocol"]) > 0

    def test_generate_compliance_mapping_efficiency(self, generator):
        """Test compliance mapping for efficiency calculations."""
        mapping = generator.generate_compliance_mapping(
            CalculationType.EFFICIENCY_CALCULATION
        )

        assert "ISO 50001" in mapping
        assert len(mapping["ISO 50001"]) > 0


class TestStandardCitations:
    """Tests for standard citations database."""

    def test_citations_exist(self):
        """Test that expected citations exist."""
        assert "ASME_PTC_4" in STANDARD_CITATIONS
        assert "EPA_AP42" in STANDARD_CITATIONS
        assert "IAPWS_IF97" in STANDARD_CITATIONS
        assert "ISO_14064" in STANDARD_CITATIONS
        assert "IPCC_2006" in STANDARD_CITATIONS

    def test_citation_format(self):
        """Test citation formatting."""
        citation = STANDARD_CITATIONS["ASME_PTC_4"]
        formatted = citation.format_citation()

        assert "ASME" in formatted
        assert "PTC 4" in formatted

    def test_citation_to_dict(self):
        """Test citation dictionary conversion."""
        citation = STANDARD_CITATIONS["ASME_PTC_4"]
        result = citation.to_dict()

        assert "source" in result
        assert "standard_id" in result
        assert result["source"] == "ASME"


class TestThermodynamicPrinciples:
    """Tests for thermodynamic principles database."""

    def test_principles_exist(self):
        """Test that expected principles exist."""
        assert "first_law_energy_balance" in THERMODYNAMIC_PRINCIPLES
        assert "heat_transfer_conduction" in THERMODYNAMIC_PRINCIPLES
        assert "steam_enthalpy" in THERMODYNAMIC_PRINCIPLES
        assert "combustion_efficiency" in THERMODYNAMIC_PRINCIPLES

    def test_principle_structure(self):
        """Test principle structure."""
        principle = THERMODYNAMIC_PRINCIPLES["first_law_energy_balance"]

        assert principle.name is not None
        assert principle.formula is not None
        assert principle.description is not None
        assert len(principle.variables) > 0

    def test_principle_to_dict(self):
        """Test principle dictionary conversion."""
        principle = THERMODYNAMIC_PRINCIPLES["first_law_energy_balance"]
        result = principle.to_dict()

        assert "name" in result
        assert "formula" in result
        assert "variables" in result


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_all_standard_sources(self):
        """Test getting all standard sources."""
        sources = get_all_standard_sources()

        assert "ASME" in sources
        assert "EPA" in sources
        assert "NIST" in sources
        assert "IAPWS" in sources
        assert "ISO" in sources

    def test_get_citations_by_source(self):
        """Test getting citations by source."""
        asme_citations = get_citations_by_source(StandardSource.ASME)

        assert len(asme_citations) > 0
        for citation in asme_citations:
            assert citation.source == StandardSource.ASME

    def test_format_principle_as_markdown(self):
        """Test Markdown formatting for principle."""
        principle = THERMODYNAMIC_PRINCIPLES["first_law_energy_balance"]
        markdown = format_principle_as_markdown(principle)

        assert "###" in markdown  # Header
        assert "**Formula:**" in markdown
        assert "**Variables:**" in markdown

    def test_format_rationale_as_markdown(self, generator, sample_inputs, sample_outputs):
        """Test Markdown formatting for rationale."""
        rationale = generator.generate_rationale(
            calculation_type=CalculationType.COMBUSTION,
            inputs=sample_inputs,
            outputs=sample_outputs
        )

        markdown = format_rationale_as_markdown(rationale)

        assert "# Engineering Rationale" in markdown
        assert "## Summary" in markdown
        assert "## Methodology" in markdown
        assert "## Input Parameters" in markdown
        assert "## Output Results" in markdown


class TestEngineeringRationaleSchema:
    """Tests for EngineeringRationale data schema."""

    def test_rationale_creation(self):
        """Test EngineeringRationale creation."""
        rationale = EngineeringRationale(
            rationale_id="test123",
            calculation_type="combustion",
            summary="Test calculation",
            methodology=["Step 1", "Step 2"],
            principles=[],
            citations=[],
            assumptions=["Assumption 1"],
            limitations=["Limitation 1"],
            input_parameters={"fuel": "gas"},
            output_results={"efficiency": 0.85}
        )

        assert rationale.rationale_id == "test123"
        assert rationale.validation_status == "PASS"
        assert rationale.provenance_hash is not None

    def test_rationale_to_dict(self):
        """Test EngineeringRationale serialization."""
        rationale = EngineeringRationale(
            rationale_id="test123",
            calculation_type="combustion",
            summary="Test calculation",
            methodology=["Step 1"],
            principles=[],
            citations=[],
            assumptions=[],
            limitations=[],
            input_parameters={},
            output_results={"result": 1.0}
        )

        result = rationale.to_dict()

        assert "rationale_id" in result
        assert "calculation_type" in result
        assert "provenance_hash" in result

    def test_rationale_provenance_hash_deterministic(self):
        """Test provenance hash is deterministic."""
        rationale1 = EngineeringRationale(
            rationale_id="test123",
            calculation_type="combustion",
            summary="Test",
            methodology=[],
            principles=[],
            citations=[],
            assumptions=[],
            limitations=[],
            input_parameters={"x": 1},
            output_results={"y": 2}
        )

        rationale2 = EngineeringRationale(
            rationale_id="test123",
            calculation_type="combustion",
            summary="Test",
            methodology=[],
            principles=[],
            citations=[],
            assumptions=[],
            limitations=[],
            input_parameters={"x": 1},
            output_results={"y": 2}
        )

        assert rationale1.provenance_hash == rationale2.provenance_hash


class TestCalculationType:
    """Tests for CalculationType enum."""

    def test_all_types_defined(self):
        """Test all expected calculation types exist."""
        expected_types = [
            "HEAT_TRANSFER",
            "COMBUSTION",
            "STEAM_PROPERTIES",
            "EMISSION_CALCULATION",
            "EFFICIENCY_CALCULATION",
            "PINCH_ANALYSIS",
            "HEAT_EXCHANGER",
            "INSULATION",
            "STEAM_TRAP",
            "WATER_TREATMENT",
            "FUEL_OPTIMIZATION",
            "PREDICTIVE_MAINTENANCE"
        ]

        for type_name in expected_types:
            assert hasattr(CalculationType, type_name)

    def test_type_values(self):
        """Test calculation type values."""
        assert CalculationType.COMBUSTION.value == "combustion"
        assert CalculationType.EMISSION_CALCULATION.value == "emission_calculation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
