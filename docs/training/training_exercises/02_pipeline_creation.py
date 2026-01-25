#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 02: Pipeline Creation

Difficulty: Intermediate
Duration: 45 minutes
Target Audience: Developers

Learning Objectives:
- Understand agent pipelines
- Create multi-stage processing workflows
- Implement error handling in pipelines
- Use parallel processing stages

Prerequisites:
- Completed Exercise 01
- Completed 03_developer_training.md
"""

import sys
from typing import Dict, Any, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# =============================================================================
# EXERCISE SETUP (Do not modify)
# =============================================================================

@dataclass
class AgentResult:
    """Result from agent processing."""
    output: Dict[str, Any]
    success: bool = True
    error: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """Process input and return result."""
        pass


class ValidationAgent(BaseAgent):
    """Validates input data."""

    REQUIRED_FIELDS = ["fuel_type", "quantity"]
    VALID_FUEL_TYPES = ["diesel", "natural_gas", "coal", "gasoline"]

    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in input_data:
                return AgentResult(
                    output=input_data,
                    success=False,
                    error=f"Missing required field: {field}"
                )

        # Validate fuel type
        if input_data["fuel_type"] not in self.VALID_FUEL_TYPES:
            return AgentResult(
                output=input_data,
                success=False,
                error=f"Invalid fuel type: {input_data['fuel_type']}"
            )

        # Validate quantity
        if input_data["quantity"] <= 0:
            return AgentResult(
                output=input_data,
                success=False,
                error="Quantity must be positive"
            )

        return AgentResult(
            output={**input_data, "validated": True},
            success=True,
            metadata={"validation_passed": True}
        )


class EmissionFactorAgent(BaseAgent):
    """Looks up emission factors."""

    FACTORS = {
        "diesel": 2.68,
        "natural_gas": 1.93,
        "coal": 3.45,
        "gasoline": 2.31,
    }

    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        fuel_type = input_data.get("fuel_type")
        ef = self.FACTORS.get(fuel_type)

        if ef is None:
            return AgentResult(
                output=input_data,
                success=False,
                error=f"No emission factor for: {fuel_type}"
            )

        return AgentResult(
            output={**input_data, "emission_factor": ef},
            success=True
        )


class CalculationAgent(BaseAgent):
    """Calculates emissions."""

    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        quantity = input_data.get("quantity", 0)
        ef = input_data.get("emission_factor", 0)

        emissions = quantity * ef

        return AgentResult(
            output={
                **input_data,
                "emissions_kg_co2e": round(emissions, 2)
            },
            success=True
        )


class QualityScoreAgent(BaseAgent):
    """Calculates data quality score."""

    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        score = 1.0

        # Deduct for missing optional fields
        if "unit" not in input_data:
            score -= 0.1
        if "source" not in input_data:
            score -= 0.1
        if "date" not in input_data:
            score -= 0.1

        return AgentResult(
            output={**input_data, "quality_score": round(score, 2)},
            success=True
        )


class ReportingAgent(BaseAgent):
    """Formats final report."""

    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        report = {
            "fuel_type": input_data.get("fuel_type"),
            "quantity": input_data.get("quantity"),
            "emissions_kg_co2e": input_data.get("emissions_kg_co2e"),
            "quality_score": input_data.get("quality_score", 1.0),
            "status": "COMPLETE"
        }

        return AgentResult(output=report, success=True)


class Pipeline:
    """Simple pipeline implementation for exercises."""

    def __init__(self, name: str, agents: List[BaseAgent] = None):
        self.name = name
        self.agents = agents or []

    def add(self, agent: BaseAgent) -> "Pipeline":
        """Add agent to pipeline."""
        self.agents.append(agent)
        return self

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute pipeline on input data."""
        current_data = input_data

        for agent in self.agents:
            result = agent.process(current_data)

            if not result.success:
                return AgentResult(
                    output=current_data,
                    success=False,
                    error=f"Pipeline failed at {agent.name}: {result.error}"
                )

            current_data = result.output

        return AgentResult(output=current_data, success=True)


def verify_answer(task_num: int, answer: Any, expected: Any, tolerance: float = 0) -> bool:
    """Verify if the answer is correct."""
    correct = False

    if isinstance(expected, float) and tolerance > 0:
        correct = abs(answer - expected) <= tolerance
    else:
        correct = answer == expected

    if correct:
        print(f"  Task {task_num}: CORRECT!")
        return True
    else:
        print(f"  Task {task_num}: INCORRECT")
        print(f"    Your answer: {answer}")
        print(f"    Expected: {expected}")
        return False


# =============================================================================
# EXERCISES BEGIN HERE
# =============================================================================

def main():
    print("=" * 70)
    print("EXERCISE 02: Pipeline Creation")
    print("=" * 70)
    print()

    score = 0
    total_tasks = 5

    # -------------------------------------------------------------------------
    # TASK 1: Create Basic Pipeline
    # -------------------------------------------------------------------------
    print("TASK 1: Create a Basic Pipeline")
    print("-" * 40)
    print("Create a pipeline with these agents in order:")
    print("  1. ValidationAgent")
    print("  2. EmissionFactorAgent")
    print("  3. CalculationAgent")
    print()


    # YOUR CODE HERE:
    pipeline = Pipeline("basic_pipeline")
    pipeline.add(ValidationAgent("validator"))
    pipeline.add(EmissionFactorAgent("ef_lookup"))
    pipeline.add(CalculationAgent("calculator"))

    # Test the pipeline
    test_input = {"fuel_type": "diesel", "quantity": 1000}
    result = pipeline.execute(test_input)

    task1_answer = result.output.get("emissions_kg_co2e")

    if verify_answer(1, task1_answer, 2680.0):
        score += 1
    print()

    # -------------------------------------------------------------------------
    # TASK 2: Add Quality Scoring
    # -------------------------------------------------------------------------
    print("TASK 2: Extended Pipeline with Quality Scoring")
    print("-" * 40)
    print("Extend the pipeline to include QualityScoreAgent after calculation.")
    print("Test with input that has all optional fields.")
    print()


    # YOUR CODE HERE:
    extended_pipeline = Pipeline("extended_pipeline")
    extended_pipeline.add(ValidationAgent("validator"))
    extended_pipeline.add(EmissionFactorAgent("ef_lookup"))
    extended_pipeline.add(CalculationAgent("calculator"))
    extended_pipeline.add(QualityScoreAgent("quality"))

    # Input with all optional fields
    complete_input = {
        "fuel_type": "diesel",
        "quantity": 500,
        "unit": "liters",
        "source": "fleet_vehicles",
        "date": "2025-12-07"
    }
    result = extended_pipeline.execute(complete_input)

    task2_answer = result.output.get("quality_score")

    # With all optional fields present, score should be 1.0
    if verify_answer(2, task2_answer, 1.0):
        score += 1
    print()

    # -------------------------------------------------------------------------
    # TASK 3: Handle Validation Errors
    # -------------------------------------------------------------------------
    print("TASK 3: Pipeline Error Handling")
    print("-" * 40)
    print("What happens when invalid input is passed?")
    print("Test with: negative quantity")
    print()


    # YOUR CODE HERE:
    invalid_input = {"fuel_type": "diesel", "quantity": -100}
    result = extended_pipeline.execute(invalid_input)

    task3_answer = result.success

    # Pipeline should fail on validation
    if verify_answer(3, task3_answer, False):
        score += 1
    print(f"  Error message: {result.error}")
    print()

    # -------------------------------------------------------------------------
    # TASK 4: Create Full Pipeline
    # -------------------------------------------------------------------------
    print("TASK 4: Create Full Pipeline with Reporting")
    print("-" * 40)
    print("Create a complete pipeline that ends with ReportingAgent.")
    print("Execute with natural_gas, 250 liters.")
    print()


    # YOUR CODE HERE:
    full_pipeline = Pipeline("full_pipeline")
    full_pipeline.add(ValidationAgent("validator"))
    full_pipeline.add(EmissionFactorAgent("ef_lookup"))
    full_pipeline.add(CalculationAgent("calculator"))
    full_pipeline.add(QualityScoreAgent("quality"))
    full_pipeline.add(ReportingAgent("reporter"))

    test_input = {"fuel_type": "natural_gas", "quantity": 250}
    result = full_pipeline.execute(test_input)

    # Expected: 250 * 1.93 = 482.5
    task4_answer = result.output.get("emissions_kg_co2e")

    if verify_answer(4, task4_answer, 482.5):
        score += 1
    print(f"  Report status: {result.output.get('status')}")
    print()

    # -------------------------------------------------------------------------
    # TASK 5: Process Multiple Inputs
    # -------------------------------------------------------------------------
    print("TASK 5: Batch Processing")
    print("-" * 40)
    print("Process multiple inputs through the pipeline.")
    print("Sum up the total emissions from all successful calculations.")
    print()


    # YOUR CODE HERE:
    batch_inputs = [
        {"fuel_type": "diesel", "quantity": 100},
        {"fuel_type": "natural_gas", "quantity": 200},
        {"fuel_type": "gasoline", "quantity": 150},
        {"fuel_type": "invalid_fuel", "quantity": 100},  # This should fail
    ]

    total_emissions = 0.0
    successful_count = 0

    for input_data in batch_inputs:
        result = full_pipeline.execute(input_data)
        if result.success:
            total_emissions += result.output.get("emissions_kg_co2e", 0)
            successful_count += 1

    task5_answer = round(total_emissions, 2)

    # Expected: 268.0 (diesel) + 386.0 (gas) + 346.5 (gasoline) = 1000.5
    if verify_answer(5, task5_answer, 1000.5):
        score += 1
    print(f"  Successful calculations: {successful_count}/4")
    print()

    # -------------------------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------------------------
    print("=" * 70)
    print(f"EXERCISE COMPLETE: {score}/{total_tasks} tasks correct")
    print("=" * 70)

    if score == total_tasks:
        print("Excellent! You have mastered pipeline creation.")
        print("Continue to Exercise 05: API Integration")
    elif score >= 3:
        print("Good progress! Review the incorrect answers and try again.")
    else:
        print("Keep practicing! Review the developer training materials.")

    return score == total_tasks


# =============================================================================
# BONUS CHALLENGES
# =============================================================================

def bonus_challenges():
    """Optional advanced challenges."""
    print()
    print("=" * 70)
    print("BONUS CHALLENGES")
    print("=" * 70)
    print()

    print("Challenge A: Create a ConditionalAgent that only runs if a condition is met")
    print("Challenge B: Implement parallel processing for independent calculations")
    print("Challenge C: Add retry logic to the pipeline for transient failures")
    print("Challenge D: Create a pipeline that branches based on fuel type")
    print()


# =============================================================================
# SOLUTIONS
# =============================================================================

SOLUTIONS = """
TASK 1 Solution:
    pipeline = Pipeline("basic_pipeline")
    pipeline.add(ValidationAgent("validator"))
    pipeline.add(EmissionFactorAgent("ef_lookup"))
    pipeline.add(CalculationAgent("calculator"))
    result = pipeline.execute({"fuel_type": "diesel", "quantity": 1000})
    # Answer: 2680.0

TASK 2 Solution:
    extended_pipeline = Pipeline("extended_pipeline")
    extended_pipeline.add(ValidationAgent("validator"))
    extended_pipeline.add(EmissionFactorAgent("ef_lookup"))
    extended_pipeline.add(CalculationAgent("calculator"))
    extended_pipeline.add(QualityScoreAgent("quality"))

    complete_input = {
        "fuel_type": "diesel", "quantity": 500,
        "unit": "liters", "source": "fleet", "date": "2025-12-07"
    }
    result = extended_pipeline.execute(complete_input)
    # Answer: quality_score = 1.0

TASK 3 Solution:
    invalid_input = {"fuel_type": "diesel", "quantity": -100}
    result = extended_pipeline.execute(invalid_input)
    # Answer: result.success = False

TASK 4 Solution:
    full_pipeline = Pipeline("full_pipeline")
    # Add all 5 agents
    result = full_pipeline.execute({"fuel_type": "natural_gas", "quantity": 250})
    # Answer: 482.5 kg CO2e

TASK 5 Solution:
    total = 0
    for input_data in batch_inputs:
        result = full_pipeline.execute(input_data)
        if result.success:
            total += result.output.get("emissions_kg_co2e", 0)
    # Answer: 1000.5 (268.0 + 386.0 + 346.5)
"""


if __name__ == "__main__":
    if "--show-solutions" in sys.argv:
        print(SOLUTIONS)
    elif "--bonus" in sys.argv:
        bonus_challenges()
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        success = main()
        sys.exit(0 if success else 1)
