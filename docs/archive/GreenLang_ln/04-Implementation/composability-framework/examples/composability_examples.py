# -*- coding: utf-8 -*-
"""
GreenLang Composability Framework Examples

This module demonstrates various patterns for composing GreenLang agents
using the GLEL (GreenLang Expression Language) framework.

Examples include:
1. Sequential chaining with pipe operator
2. Parallel execution patterns
3. Error handling and retry logic
4. Streaming and async patterns
5. Map-reduce patterns
6. Conditional branching
7. Zero-hallucination wrappers
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any
from pydantic import BaseModel, Field

from greenlang.determinism import deterministic_random
from greenlang.core.composability import (
    AgentRunnable,
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
    RetryRunnable,
    FallbackRunnable,
    ZeroHallucinationWrapper,
    RunnableConfig,
    create_sequential_chain,
    create_parallel_chain,
    create_map_reduce_chain
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Mock Agent Classes for Demonstration
# ============================================================================

class IntakeAgent:
    """Mock intake agent for data collection."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process intake data."""
        logger.info(f"IntakeAgent processing: {input_data.get('company_name', 'Unknown')}")

        # Simulate data intake and validation
        return {
            "company_name": input_data.get("company_name"),
            "reporting_period": input_data.get("reporting_period"),
            "activity_data": {
                "electricity_kwh": 50000,
                "natural_gas_mmbtu": 1000,
                "fleet_miles": 25000
            },
            "intake_timestamp": DeterministicClock.now().isoformat(),
            "status": "VALIDATED"
        }


class ValidationAgent:
    """Mock validation agent for data quality checks."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against business rules."""
        logger.info("ValidationAgent checking data quality")

        # Simulate validation logic
        validation_results = []

        # Check required fields
        required_fields = ["company_name", "reporting_period", "activity_data"]
        for field in required_fields:
            if field in input_data:
                validation_results.append({"rule": f"{field}_present", "status": "PASS"})
            else:
                validation_results.append({"rule": f"{field}_present", "status": "FAIL"})

        # Add validation results to output
        input_data["validation_results"] = validation_results
        input_data["validation_status"] = "PASS" if all(
            r["status"] == "PASS" for r in validation_results
        ) else "FAIL"

        return input_data


class CalculationAgent:
    """Mock calculation agent with zero-hallucination guarantee."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform deterministic calculations.
        ZERO HALLUCINATION - Only uses formulas and database lookups.
        """
        logger.info("CalculationAgent performing deterministic calculations")

        activity_data = input_data.get("activity_data", {})

        # Deterministic calculations using emission factors
        emissions = {
            "scope2_electricity": activity_data.get("electricity_kwh", 0) * 0.4,  # kg CO2e
            "scope1_natural_gas": activity_data.get("natural_gas_mmbtu", 0) * 53.06,  # kg CO2e
            "scope1_fleet": activity_data.get("fleet_miles", 0) * 0.4,  # kg CO2e
        }

        # Calculate totals
        emissions["scope1_total"] = emissions["scope1_natural_gas"] + emissions["scope1_fleet"]
        emissions["scope2_total"] = emissions["scope2_electricity"]
        emissions["total_emissions"] = emissions["scope1_total"] + emissions["scope2_total"]

        # Add calculation metadata for provenance
        input_data["emissions"] = emissions
        input_data["calculation_method"] = "IPCC_2021_TIER1"
        input_data["calculation_timestamp"] = DeterministicClock.now().isoformat()

        return input_data


class ReportingAgent:
    """Mock reporting agent for output generation."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reporting outputs."""
        logger.info("ReportingAgent generating report")

        emissions = input_data.get("emissions", {})

        # Create summary report
        report = {
            "company_name": input_data.get("company_name"),
            "reporting_period": input_data.get("reporting_period"),
            "total_emissions_kg_co2e": emissions.get("total_emissions", 0),
            "scope1_emissions_kg_co2e": emissions.get("scope1_total", 0),
            "scope2_emissions_kg_co2e": emissions.get("scope2_total", 0),
            "validation_status": input_data.get("validation_status"),
            "report_generated_at": DeterministicClock.now().isoformat(),
            "report_format": "GRI_STANDARD",
            "full_data": input_data  # Include complete data for audit trail
        }

        return report


class ComplianceAgent:
    """Mock compliance checking agent."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance against regulations."""
        logger.info("ComplianceAgent checking regulatory compliance")

        total_emissions = input_data.get("emissions", {}).get("total_emissions", 0)

        # Mock compliance checks
        compliance_checks = []

        # EU taxonomy alignment
        if total_emissions < 100000:
            compliance_checks.append({
                "regulation": "EU_TAXONOMY",
                "status": "ALIGNED",
                "threshold": 100000
            })
        else:
            compliance_checks.append({
                "regulation": "EU_TAXONOMY",
                "status": "NOT_ALIGNED",
                "threshold": 100000,
                "gap": total_emissions - 100000
            })

        # CSRD reporting requirements
        compliance_checks.append({
            "regulation": "CSRD",
            "status": "COMPLIANT",
            "requirements_met": ["double_materiality", "value_chain", "targets"]
        })

        return {
            "compliance_checks": compliance_checks,
            "overall_compliance": all(c["status"] in ["ALIGNED", "COMPLIANT"]
                                     for c in compliance_checks),
            "input_data": input_data
        }


class RiskAssessmentAgent:
    """Mock risk assessment agent."""

    async def aprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async risk assessment."""
        logger.info("RiskAssessmentAgent evaluating climate risks")

        # Simulate async processing
        await asyncio.sleep(0.1)

        emissions = input_data.get("emissions", {}).get("total_emissions", 0)

        # Mock risk scoring
        risk_score = min(100, emissions / 1000)  # Simple linear scoring

        risks = {
            "transition_risk": {
                "score": risk_score,
                "category": "HIGH" if risk_score > 70 else "MEDIUM" if risk_score > 30 else "LOW",
                "factors": ["carbon_pricing", "regulation", "market_shifts"]
            },
            "physical_risk": {
                "score": 45,  # Mock value
                "category": "MEDIUM",
                "factors": ["flooding", "extreme_heat", "supply_chain"]
            },
            "overall_risk_score": (risk_score + 45) / 2
        }

        return {
            "risk_assessment": risks,
            "assessment_date": DeterministicClock.now().isoformat(),
            "input_data": input_data
        }


# ============================================================================
# Example 1: Sequential Chaining with Pipe Operator
# ============================================================================

def example_sequential_chain():
    """Demonstrate sequential chaining using the pipe operator."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Sequential Chain with Pipe Operator")
    print("="*80)

    # Create agents
    intake = IntakeAgent()
    validation = ValidationAgent()
    calculation = CalculationAgent()
    reporting = ReportingAgent()

    # Create chain using pipe operator (|)
    chain = (
        AgentRunnable(intake) |
        AgentRunnable(validation) |
        AgentRunnable(calculation) |
        AgentRunnable(reporting)
    )

    # Input data
    input_data = {
        "company_name": "GreenCorp Industries",
        "reporting_period": "2024-Q1"
    }

    # Execute chain
    result = chain.invoke(input_data)

    print(f"\nFinal Report:")
    print(f"  Company: {result['company_name']}")
    print(f"  Period: {result['reporting_period']}")
    print(f"  Total Emissions: {result['total_emissions_kg_co2e']:,.2f} kg CO2e")
    print(f"  Validation Status: {result['validation_status']}")

    return result


# ============================================================================
# Example 2: Parallel Execution Pattern
# ============================================================================

async def example_parallel_execution():
    """Demonstrate parallel execution of multiple agents."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Parallel Execution Pattern")
    print("="*80)

    # Prepare input data with emissions calculated
    calculation = CalculationAgent()
    intake = IntakeAgent()

    base_data = {
        "company_name": "ParallelCorp",
        "reporting_period": "2024-Q1"
    }

    # First get activity data and calculate emissions
    intake_data = intake.process(base_data)
    calculated_data = calculation.process(intake_data)

    # Create parallel execution for compliance and risk assessment
    parallel = RunnableParallel({
        "compliance": AgentRunnable(ComplianceAgent()),
        "risk": AgentRunnable(RiskAssessmentAgent()),
        "reporting": AgentRunnable(ReportingAgent())
    })

    # Execute in parallel
    results = await parallel.ainvoke(calculated_data)

    print(f"\nParallel Execution Results:")
    print(f"  Branches executed: {len(results) - 1}")  # -1 for _parallel_provenance
    print(f"  Compliance Status: {results['compliance']['overall_compliance']}")
    print(f"  Risk Category: {results['risk']['risk_assessment']['transition_risk']['category']}")
    print(f"  Report Generated: {results['reporting']['report_generated_at']}")

    return results


# ============================================================================
# Example 3: Error Handling and Retry Logic
# ============================================================================

class UnreliableAgent:
    """Mock agent that randomly fails to demonstrate retry logic."""

    def __init__(self, failure_rate: float = 0.5):
        self.failure_rate = failure_rate
        self.attempt_count = 0

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with random failures."""
        self.attempt_count += 1
        logger.info(f"UnreliableAgent attempt #{self.attempt_count}")

        import random
        if deterministic_random().random() < self.failure_rate and self.attempt_count < 3:
            raise RuntimeError(f"Random failure on attempt {self.attempt_count}")

        return {
            **input_data,
            "processed": True,
            "attempts_required": self.attempt_count
        }


def example_retry_logic():
    """Demonstrate retry logic for handling failures."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Error Handling and Retry Logic")
    print("="*80)

    # Create unreliable agent with retry wrapper
    unreliable = UnreliableAgent(failure_rate=0.7)
    reliable = AgentRunnable(unreliable).with_retry(max_retries=3, delay_ms=500)

    # Create chain with retry logic
    chain = (
        AgentRunnable(IntakeAgent()) |
        reliable |  # This step will retry on failure
        AgentRunnable(CalculationAgent())
    )

    input_data = {
        "company_name": "RetryTest Corp",
        "reporting_period": "2024-Q1"
    }

    try:
        result = chain.invoke(input_data)
        print(f"\nSuccess after {result.get('attempts_required', 1)} attempt(s)")
        print(f"  Total Emissions: {result['emissions']['total_emissions']:,.2f} kg CO2e")
    except Exception as e:
        print(f"\nFailed after all retries: {str(e)}")

    return result


# ============================================================================
# Example 4: Streaming and Async Patterns
# ============================================================================

async def example_streaming():
    """Demonstrate streaming results from a chain."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Streaming Pattern")
    print("="*80)

    # Create chain
    chain = RunnableSequence([
        AgentRunnable(IntakeAgent()),
        AgentRunnable(ValidationAgent()),
        AgentRunnable(CalculationAgent()),
        AgentRunnable(ReportingAgent())
    ])

    input_data = {
        "company_name": "StreamCorp",
        "reporting_period": "2024-Q1"
    }

    print("\nStreaming results from each step:")
    async for chunk in chain.astream(input_data):
        print(f"  Step: {chunk['step']}")
        if 'output' in chunk and 'emissions' in chunk['output']:
            emissions = chunk['output']['emissions'].get('total_emissions', 0)
            print(f"    -> Emissions calculated: {emissions:,.2f} kg CO2e")


# ============================================================================
# Example 5: Map-Reduce Pattern
# ============================================================================

class AggregationAgent:
    """Mock agent for aggregating multiple results."""

    def process(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple facility results."""
        logger.info(f"AggregationAgent aggregating {len(input_data)} results")

        total_emissions = sum(
            item.get('emissions', {}).get('total_emissions', 0)
            for item in input_data
        )

        scope1_total = sum(
            item.get('emissions', {}).get('scope1_total', 0)
            for item in input_data
        )

        scope2_total = sum(
            item.get('emissions', {}).get('scope2_total', 0)
            for item in input_data
        )

        return {
            "aggregated_results": {
                "facility_count": len(input_data),
                "total_emissions": total_emissions,
                "scope1_total": scope1_total,
                "scope2_total": scope2_total,
                "aggregation_timestamp": DeterministicClock.now().isoformat()
            },
            "facility_details": input_data
        }


async def example_map_reduce():
    """Demonstrate map-reduce pattern for processing multiple facilities."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Map-Reduce Pattern")
    print("="*80)

    # Create processing pipeline for individual facilities
    facility_processor = (
        AgentRunnable(IntakeAgent()) |
        AgentRunnable(ValidationAgent()) |
        AgentRunnable(CalculationAgent())
    )

    # Create aggregation agent
    aggregator = AgentRunnable(AggregationAgent())

    # Create map-reduce chain
    map_reduce_chain = create_map_reduce_chain(
        mapper=facility_processor,
        reducer=aggregator
    )

    # Multiple facility inputs
    facilities = [
        {"company_name": "Facility A", "reporting_period": "2024-Q1"},
        {"company_name": "Facility B", "reporting_period": "2024-Q1"},
        {"company_name": "Facility C", "reporting_period": "2024-Q1"},
    ]

    # Execute map-reduce
    result = await map_reduce_chain.ainvoke(facilities)

    print(f"\nMap-Reduce Results:")
    print(f"  Facilities processed: {result['aggregated_results']['facility_count']}")
    print(f"  Total emissions: {result['aggregated_results']['total_emissions']:,.2f} kg CO2e")
    print(f"  Scope 1: {result['aggregated_results']['scope1_total']:,.2f} kg CO2e")
    print(f"  Scope 2: {result['aggregated_results']['scope2_total']:,.2f} kg CO2e")

    return result


# ============================================================================
# Example 6: Conditional Branching
# ============================================================================

def example_conditional_branching():
    """Demonstrate conditional branching based on data."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Conditional Branching")
    print("="*80)

    # Create different processing paths
    small_company_chain = (
        AgentRunnable(ValidationAgent()) |
        AgentRunnable(CalculationAgent())
    )

    large_company_chain = (
        AgentRunnable(ValidationAgent()) |
        AgentRunnable(CalculationAgent()) |
        AgentRunnable(ComplianceAgent())
    )

    # Create branching logic
    def is_large_company(input_data: Dict[str, Any]) -> bool:
        # Check if it's a large company based on activity data
        activity = input_data.get("activity_data", {})
        electricity = activity.get("electricity_kwh", 0)
        return electricity > 100000

    branch = RunnableBranch(
        branches=[
            (is_large_company, large_company_chain),
        ],
        default=small_company_chain
    )

    # Test with different company sizes
    print("\nProcessing small company:")
    small_input = IntakeAgent().process({
        "company_name": "SmallCo",
        "reporting_period": "2024-Q1"
    })
    small_result = branch.invoke(small_input)
    print(f"  Emissions: {small_result['emissions']['total_emissions']:,.2f} kg CO2e")
    print(f"  Compliance checks: {'compliance_checks' in small_result}")

    # Modify to simulate large company
    large_input = IntakeAgent().process({
        "company_name": "LargeCo",
        "reporting_period": "2024-Q1"
    })
    large_input["activity_data"]["electricity_kwh"] = 150000
    large_result = branch.invoke(large_input)
    print(f"\nProcessing large company:")
    print(f"  Emissions: {large_result['input_data']['emissions']['total_emissions']:,.2f} kg CO2e")
    print(f"  Compliance checks: {len(large_result.get('compliance_checks', []))}")

    return small_result, large_result


# ============================================================================
# Example 7: Zero-Hallucination Wrapper
# ============================================================================

def example_zero_hallucination():
    """Demonstrate zero-hallucination wrapper for calculation integrity."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Zero-Hallucination Wrapper")
    print("="*80)

    # Create calculation agent with zero-hallucination wrapper
    calculation = CalculationAgent()

    # Define validation rules
    def validate_numeric_inputs(input_data: Dict[str, Any]) -> bool:
        """Ensure all activity data is numeric."""
        activity = input_data.get("activity_data", {})
        for key, value in activity.items():
            if not isinstance(value, (int, float)):
                return False
        return True

    def validate_positive_values(input_data: Dict[str, Any]) -> bool:
        """Ensure all values are non-negative."""
        activity = input_data.get("activity_data", {})
        for value in activity.values():
            if isinstance(value, (int, float)) and value < 0:
                return False
        return True

    # Wrap with zero-hallucination guarantees
    safe_calculation = ZeroHallucinationWrapper(
        AgentRunnable(calculation),
        validation_rules=[validate_numeric_inputs, validate_positive_values]
    )

    # Create chain with zero-hallucination wrapper
    chain = (
        AgentRunnable(IntakeAgent()) |
        AgentRunnable(ValidationAgent()) |
        safe_calculation |  # Zero-hallucination guaranteed
        AgentRunnable(ReportingAgent())
    )

    input_data = {
        "company_name": "SafeCalc Corp",
        "reporting_period": "2024-Q1"
    }

    result = chain.invoke(input_data)

    print(f"\nZero-Hallucination Results:")
    print(f"  Company: {result['company_name']}")
    print(f"  Calculation Method: {result['full_data']['calculation_method']}")
    print(f"  Total Emissions: {result['total_emissions_kg_co2e']:,.2f} kg CO2e")
    print(f"  Validation: All calculations are deterministic (no LLM involvement)")

    return result


# ============================================================================
# Example 8: Lambda Functions in Chains
# ============================================================================

def example_lambda_functions():
    """Demonstrate using lambda functions in chains."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Lambda Functions in Chains")
    print("="*80)

    # Create lambda functions for simple transformations
    def add_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing metadata."""
        return {
            **data,
            "processed_at": DeterministicClock.now().isoformat(),
            "processing_version": "1.0.0"
        }

    def calculate_intensity(data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate emission intensity metrics."""
        emissions = data.get("emissions", {})
        total = emissions.get("total_emissions", 0)

        # Mock revenue for intensity calculation
        revenue = 1000000  # $1M

        return {
            **data,
            "intensity_metrics": {
                "emissions_per_revenue": total / revenue if revenue > 0 else 0,
                "unit": "kg CO2e per USD"
            }
        }

    # Create chain with lambda functions
    chain = (
        AgentRunnable(IntakeAgent()) |
        RunnableLambda(add_metadata, name="AddMetadata") |
        AgentRunnable(CalculationAgent()) |
        RunnableLambda(calculate_intensity, name="CalculateIntensity") |
        AgentRunnable(ReportingAgent())
    )

    input_data = {
        "company_name": "Lambda Corp",
        "reporting_period": "2024-Q1"
    }

    result = chain.invoke(input_data)

    print(f"\nLambda Chain Results:")
    print(f"  Processing Version: {result['full_data']['processing_version']}")
    print(f"  Total Emissions: {result['total_emissions_kg_co2e']:,.2f} kg CO2e")
    intensity = result['full_data']['intensity_metrics']['emissions_per_revenue']
    print(f"  Emission Intensity: {intensity:.6f} kg CO2e/USD")

    return result


# ============================================================================
# Example 9: Fallback Patterns
# ============================================================================

class PrimaryDataSource:
    """Mock primary data source that might fail."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Try to fetch from primary source."""
        import random
        if deterministic_random().random() < 0.7:  # 70% failure rate
            raise ConnectionError("Primary data source unavailable")

        return {
            **input_data,
            "emission_factors": {
                "electricity": 0.4,
                "natural_gas": 53.06,
                "fleet": 0.4
            },
            "source": "primary_database"
        }


class FallbackDataSource:
    """Mock fallback data source."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch from fallback source."""
        logger.info("Using fallback data source")

        return {
            **input_data,
            "emission_factors": {
                "electricity": 0.42,  # Slightly different factors
                "natural_gas": 53.0,
                "fleet": 0.41
            },
            "source": "fallback_cache"
        }


def example_fallback():
    """Demonstrate fallback pattern for resilience."""
    print("\n" + "="*80)
    print("EXAMPLE 9: Fallback Pattern")
    print("="*80)

    # Create data source with fallback
    primary = AgentRunnable(PrimaryDataSource())
    fallback = AgentRunnable(FallbackDataSource())

    data_source = FallbackRunnable(primary, fallback)

    # Create chain with fallback data source
    chain = (
        AgentRunnable(IntakeAgent()) |
        data_source |  # Will use fallback if primary fails
        AgentRunnable(CalculationAgent())
    )

    input_data = {
        "company_name": "Resilient Corp",
        "reporting_period": "2024-Q1"
    }

    result = chain.invoke(input_data)

    print(f"\nFallback Pattern Results:")
    print(f"  Data Source Used: {result.get('source', 'unknown')}")
    print(f"  Emission Factors Retrieved: {result.get('emission_factors') is not None}")
    print(f"  Total Emissions: {result['emissions']['total_emissions']:,.2f} kg CO2e")

    return result


# ============================================================================
# Example 10: Complex Pipeline with All Features
# ============================================================================

async def example_complex_pipeline():
    """Demonstrate a complex pipeline using multiple patterns."""
    print("\n" + "="*80)
    print("EXAMPLE 10: Complex Pipeline with All Features")
    print("="*80)

    # Phase 1: Data collection with retry
    intake_with_retry = AgentRunnable(IntakeAgent()).with_retry(max_retries=2)

    # Phase 2: Parallel validation and enrichment
    parallel_processing = RunnableParallel({
        "validation": AgentRunnable(ValidationAgent()),
        "emission_factors": FallbackRunnable(
            AgentRunnable(PrimaryDataSource()),
            AgentRunnable(FallbackDataSource())
        )
    })

    # Phase 3: Zero-hallucination calculation
    safe_calc = ZeroHallucinationWrapper(
        AgentRunnable(CalculationAgent()),
        validation_rules=[
            lambda x: x.get("validation", {}).get("validation_status") == "PASS"
        ]
    )

    # Phase 4: Parallel compliance and risk assessment
    parallel_assessment = RunnableParallel({
        "compliance": AgentRunnable(ComplianceAgent()),
        "risk": AgentRunnable(RiskAssessmentAgent())
    })

    # Phase 5: Final reporting
    reporting = AgentRunnable(ReportingAgent())

    # Build the complex pipeline
    async def process_complex_pipeline(input_data: Dict[str, Any]):
        """Execute the complex pipeline."""

        # Step 1: Intake with retry
        print("  Step 1: Data intake...")
        intake_result = intake_with_retry.invoke(input_data)

        # Step 2: Parallel validation and enrichment
        print("  Step 2: Parallel validation and enrichment...")
        parallel_result = await parallel_processing.ainvoke(intake_result)

        # Merge results
        merged_data = {
            **intake_result,
            **parallel_result["validation"],
            **parallel_result.get("emission_factors", {})
        }

        # Step 3: Safe calculation
        print("  Step 3: Zero-hallucination calculation...")
        calc_result = safe_calc.invoke(merged_data)

        # Step 4: Parallel assessments
        print("  Step 4: Parallel compliance and risk assessment...")
        assessment_result = await parallel_assessment.ainvoke(calc_result)

        # Step 5: Final reporting
        print("  Step 5: Generating final report...")
        final_data = {
            **calc_result,
            "compliance": assessment_result["compliance"],
            "risk": assessment_result["risk"]
        }
        report = reporting.invoke(final_data)

        return report

    # Execute the complex pipeline
    input_data = {
        "company_name": "Complex Corp",
        "reporting_period": "2024-Q1"
    }

    result = await process_complex_pipeline(input_data)

    print(f"\nComplex Pipeline Results:")
    print(f"  Company: {result['company_name']}")
    print(f"  Total Emissions: {result['total_emissions_kg_co2e']:,.2f} kg CO2e")
    print(f"  Validation Status: {result['validation_status']}")
    print(f"  Compliance Status: {result['full_data']['compliance']['overall_compliance']}")
    risk_score = result['full_data']['risk']['risk_assessment']['overall_risk_score']
    print(f"  Overall Risk Score: {risk_score:.2f}/100")

    return result


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("GreenLang Composability Framework Examples")
    print("Demonstrating GLEL (GreenLang Expression Language)")
    print("="*80)

    # Run synchronous examples
    example_sequential_chain()
    example_retry_logic()
    example_conditional_branching()
    example_zero_hallucination()
    example_lambda_functions()
    example_fallback()

    # Run async examples
    await example_parallel_execution()
    await example_streaming()
    await example_map_reduce()
    await example_complex_pipeline()

    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())