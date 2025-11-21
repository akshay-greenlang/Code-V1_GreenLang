# -*- coding: utf-8 -*-
"""
Integration Tests for Shared Services
Demonstrates cross-service integration patterns

Version: 1.0.0
"""

import pytest
import asyncio
from greenlang.services import (
    FactorBroker,
    FactorRequest,
    EntityResolver,
    SupplierEntity,
    MonteCarloSimulator,
    PedigreeMatrixEvaluator,
    PedigreeScore,
)
from greenlang.agents.templates import (
    IntakeAgent,
    CalculatorAgent,
    ReportingAgent,
)


class TestFactorBrokerIntegration:
    """Test Factor Broker service integration."""

    @pytest.mark.asyncio
    async def test_basic_factor_resolution(self):
        """Test basic emission factor resolution."""
        broker = FactorBroker()

        request = FactorRequest(
            product="Steel",
            region="US",
            gwp_standard="AR6"
        )

        # This would normally resolve from real sources
        # For testing, we'd use mocked sources
        # response = await broker.resolve(request)
        # assert response.value > 0
        # assert response.metadata.source in ["ecoinvent", "desnz_uk", "epa_us", "proxy"]


class TestEntityMDMIntegration:
    """Test Entity MDM service integration."""

    def test_entity_resolution(self):
        """Test entity resolution pipeline."""
        # This would test with actual Weaviate instance
        # For now, demonstrate the API
        entity = SupplierEntity(
            entity_id="test_123",
            name="Apple Inc",
        )

        # resolver = EntityResolver()
        # result = resolver.resolve(entity)
        # assert result.status in ["auto_matched", "pending_review", "no_match"]


class TestMethodologiesIntegration:
    """Test Methodologies service integration."""

    def test_pedigree_matrix(self):
        """Test pedigree matrix evaluation."""
        evaluator = PedigreeMatrixEvaluator()

        pedigree = PedigreeScore(
            reliability=5,
            completeness=5,
            temporal=5,
            geographical=5,
            technological=5
        )

        # Validate score
        assert evaluator.validate_pedigree_score(pedigree)

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        simulator = MonteCarloSimulator(seed=42)

        # This would run actual simulation
        # result = simulator.simulate(
        #     value=100.0,
        #     uncertainty=0.10,
        #     iterations=10000
        # )
        # assert result.mean == pytest.approx(100.0, rel=0.05)


class TestFullStackIntegration:
    """Test full stack integration across all services."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow: Intake -> Calculate -> Report."""

        # Step 1: Data Intake
        intake_agent = IntakeAgent()

        # Would ingest real data
        # intake_result = await intake_agent.ingest(
        #     file_path="test_data.csv",
        #     validate=True
        # )
        # assert intake_result.success

        # Step 2: Calculations
        calculator_agent = CalculatorAgent()

        # Register formula
        calculator_agent.register_formula(
            name="scope3_emissions",
            formula=lambda quantity, factor: quantity * factor,
            required_inputs=["quantity", "factor"]
        )

        calc_result = await calculator_agent.calculate(
            formula_name="scope3_emissions",
            inputs={"quantity": 1000, "factor": 1.85},
            with_uncertainty=False
        )

        assert calc_result.success
        assert calc_result.value == 1850.0

        # Step 3: Reporting
        reporting_agent = ReportingAgent()

        # Would generate real report
        # report_result = await reporting_agent.generate_report(
        #     data=calculations_df,
        #     format="excel",
        #     check_compliance=["ghg_protocol"]
        # )
        # assert report_result.success


class TestServiceComposition:
    """Test service composition patterns."""

    @pytest.mark.asyncio
    async def test_factor_broker_with_uncertainty(self):
        """Test Factor Broker + Methodologies integration."""

        # This demonstrates how services compose
        # broker = FactorBroker()
        # simulator = MonteCarloSimulator()

        # Get factor
        # factor_response = await broker.resolve(request)

        # Quantify uncertainty
        # uncertainty_result = simulator.simulate(
        #     value=factor_response.value,
        #     uncertainty=factor_response.uncertainty,
        #     iterations=10000
        # )

        # Combined result
        # assert uncertainty_result.p95 > factor_response.value


class TestAgentTemplates:
    """Test agent template patterns."""

    @pytest.mark.asyncio
    async def test_intake_agent_csv(self):
        """Test CSV intake with validation."""
        import pandas as pd

        # Create test data
        test_data = pd.DataFrame({
            "product": ["Steel", "Aluminum"],
            "quantity": [100, 200],
            "unit": ["kg", "kg"]
        })

        agent = IntakeAgent(
            schema={
                "required": ["product", "quantity", "unit"],
                "types": {
                    "product": "string",
                    "quantity": "number",
                    "unit": "string"
                }
            }
        )

        result = await agent.ingest(
            data=test_data,
            validate=True
        )

        assert result.success
        assert result.rows_read == 2
        assert result.rows_valid == 2

    @pytest.mark.asyncio
    async def test_calculator_agent_provenance(self):
        """Test calculator with full provenance."""
        agent = CalculatorAgent()

        # Register formula
        agent.register_formula(
            name="simple_multiply",
            formula=lambda a, b: a * b,
            required_inputs=["a", "b"]
        )

        result = await agent.calculate(
            formula_name="simple_multiply",
            inputs={"a": 10, "b": 5}
        )

        assert result.success
        assert result.value == 50
        assert result.provenance is not None
        assert result.provenance.formula == "simple_multiply"
        assert result.provenance.hash is not None

    @pytest.mark.asyncio
    async def test_reporting_agent_formats(self):
        """Test reporting agent with multiple formats."""
        import pandas as pd

        test_data = pd.DataFrame({
            "category": ["Scope 1", "Scope 2", "Scope 3"],
            "emissions": [100, 200, 300]
        })

        agent = ReportingAgent()

        # Test JSON
        result = await agent.generate_report(
            data=test_data,
            format="json"
        )
        assert result.success
        assert result.data is not None

        # Test CSV
        result = await agent.generate_report(
            data=test_data,
            format="csv"
        )
        assert result.success
        assert result.data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
