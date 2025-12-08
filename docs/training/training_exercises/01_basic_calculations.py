#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 01: Basic Calculations

Difficulty: Beginner
Duration: 30 minutes
Target Audience: All roles

Learning Objectives:
- Understand GreenLang calculation basics
- Use the Calculator API
- Interpret calculation results
- Verify provenance hashes

Prerequisites:
- Completed 01_getting_started.md
- GreenLang installed and configured
"""

import sys
from typing import Dict, Any

# =============================================================================
# EXERCISE SETUP (Do not modify)
# =============================================================================

class MockCalculator:
    """Mock calculator for exercise purposes."""

    EMISSION_FACTORS = {
        ("diesel", "US"): 2.68,
        ("diesel", "EU"): 2.65,
        ("natural_gas", "US"): 1.93,
        ("natural_gas", "EU"): 1.89,
        ("coal", "US"): 3.45,
        ("coal", "EU"): 3.42,
        ("gasoline", "US"): 2.31,
        ("gasoline", "EU"): 2.28,
    }

    def calculate(
        self,
        fuel_type: str,
        quantity: float,
        unit: str = "liters",
        region: str = "US"
    ) -> Dict[str, Any]:
        """Calculate emissions for fuel consumption."""
        ef = self.EMISSION_FACTORS.get((fuel_type, region))
        if ef is None:
            raise ValueError(f"Unknown fuel type: {fuel_type} for region: {region}")

        emissions = quantity * ef

        # Generate deterministic provenance hash
        import hashlib
        input_str = f"{fuel_type}:{quantity}:{unit}:{region}:{ef}"
        provenance = hashlib.sha256(input_str.encode()).hexdigest()

        return {
            "emissions_kg_co2e": round(emissions, 2),
            "emission_factor": ef,
            "emission_factor_unit": "kg CO2e per liter",
            "provenance_hash": provenance,
            "methodology": "GHG Protocol Scope 1"
        }

    def calculate_batch(self, records: list) -> Dict[str, Any]:
        """Calculate emissions for multiple records."""
        results = []
        total = 0.0

        for record in records:
            result = self.calculate(
                fuel_type=record["fuel_type"],
                quantity=record["quantity"],
                unit=record.get("unit", "liters"),
                region=record.get("region", "US")
            )
            results.append(result)
            total += result["emissions_kg_co2e"]

        return {
            "results": results,
            "total_emissions_kg_co2e": round(total, 2),
            "record_count": len(records)
        }


def verify_answer(task_num: int, answer: Any, expected: Any) -> bool:
    """Verify if the answer is correct."""
    if answer == expected:
        print(f"  Task {task_num}: CORRECT!")
        return True
    else:
        print(f"  Task {task_num}: INCORRECT")
        print(f"    Your answer: {answer}")
        print(f"    Expected: {expected}")
        return False


def show_hint(task_num: int):
    """Show hint for a task."""
    hints = {
        1: "Use calculator.calculate() with fuel_type='diesel' and quantity=1000",
        2: "Multiply 500 liters by the natural gas emission factor (1.93 kg/L)",
        3: "Same inputs should always produce the same provenance hash",
        4: "Use calculator.calculate_batch() with a list of record dictionaries",
        5: "Check what happens when you use a fuel_type not in EMISSION_FACTORS",
    }
    print(f"  Hint for Task {task_num}: {hints.get(task_num, 'No hint available')}")


# =============================================================================
# EXERCISES BEGIN HERE
# =============================================================================

def main():
    print("=" * 70)
    print("EXERCISE 01: Basic Calculations")
    print("=" * 70)
    print()

    calculator = MockCalculator()
    score = 0
    total_tasks = 5

    # -------------------------------------------------------------------------
    # TASK 1: Simple Calculation
    # -------------------------------------------------------------------------
    print("TASK 1: Simple Diesel Calculation")
    print("-" * 40)
    print("Calculate emissions for 1000 liters of diesel in the US.")
    print()

    # TODO: Complete this task
    # Use the calculator to calculate emissions for:
    # - fuel_type: "diesel"
    # - quantity: 1000
    # - region: "US"

    # YOUR CODE HERE:
    task1_result = calculator.calculate(
        fuel_type="diesel",
        quantity=1000,
        region="US"
    )
    task1_answer = task1_result["emissions_kg_co2e"]

    # Verify
    if verify_answer(1, task1_answer, 2680.0):
        score += 1
    print()

    # -------------------------------------------------------------------------
    # TASK 2: Different Fuel Type
    # -------------------------------------------------------------------------
    print("TASK 2: Natural Gas Calculation")
    print("-" * 40)
    print("Calculate emissions for 500 liters of natural gas in the US.")
    print("What is the expected emission value?")
    print()

    # TODO: Calculate manually and verify with the calculator
    # Natural gas emission factor for US is 1.93 kg CO2e per liter

    # YOUR CODE HERE:
    task2_result = calculator.calculate(
        fuel_type="natural_gas",
        quantity=500,
        region="US"
    )
    task2_answer = task2_result["emissions_kg_co2e"]

    # Expected: 500 * 1.93 = 965.0
    if verify_answer(2, task2_answer, 965.0):
        score += 1
    print()

    # -------------------------------------------------------------------------
    # TASK 3: Provenance Verification
    # -------------------------------------------------------------------------
    print("TASK 3: Provenance Hash Verification")
    print("-" * 40)
    print("Verify that the same inputs produce the same provenance hash.")
    print()

    # TODO: Run the same calculation twice and compare hashes

    # YOUR CODE HERE:
    result_a = calculator.calculate(fuel_type="diesel", quantity=100, region="US")
    result_b = calculator.calculate(fuel_type="diesel", quantity=100, region="US")

    task3_answer = (result_a["provenance_hash"] == result_b["provenance_hash"])

    if verify_answer(3, task3_answer, True):
        score += 1
    print(f"  Hash A: {result_a['provenance_hash'][:16]}...")
    print(f"  Hash B: {result_b['provenance_hash'][:16]}...")
    print()

    # -------------------------------------------------------------------------
    # TASK 4: Batch Calculation
    # -------------------------------------------------------------------------
    print("TASK 4: Batch Calculation")
    print("-" * 40)
    print("Calculate total emissions for multiple fuel records:")
    print("  - 1000 liters diesel (US)")
    print("  - 500 liters natural gas (US)")
    print("  - 200 liters gasoline (US)")
    print()

    # TODO: Use calculate_batch with a list of records

    # YOUR CODE HERE:
    records = [
        {"fuel_type": "diesel", "quantity": 1000, "region": "US"},
        {"fuel_type": "natural_gas", "quantity": 500, "region": "US"},
        {"fuel_type": "gasoline", "quantity": 200, "region": "US"},
    ]
    batch_result = calculator.calculate_batch(records)
    task4_answer = batch_result["total_emissions_kg_co2e"]

    # Expected: 2680 + 965 + 462 = 4107
    if verify_answer(4, task4_answer, 4107.0):
        score += 1
    print()

    # -------------------------------------------------------------------------
    # TASK 5: Error Handling
    # -------------------------------------------------------------------------
    print("TASK 5: Error Handling")
    print("-" * 40)
    print("What happens when you use an invalid fuel type?")
    print("Try to calculate emissions for 'hydrogen' (not in our database).")
    print()

    # TODO: Handle the error gracefully

    # YOUR CODE HERE:
    task5_answer = None
    try:
        calculator.calculate(fuel_type="hydrogen", quantity=100, region="US")
        task5_answer = "No error"
    except ValueError as e:
        task5_answer = "ValueError"
    except Exception as e:
        task5_answer = type(e).__name__

    if verify_answer(5, task5_answer, "ValueError"):
        score += 1
    print()

    # -------------------------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------------------------
    print("=" * 70)
    print(f"EXERCISE COMPLETE: {score}/{total_tasks} tasks correct")
    print("=" * 70)

    if score == total_tasks:
        print("Excellent! You have mastered basic calculations.")
        print("Continue to Exercise 02: Pipeline Creation")
    elif score >= 3:
        print("Good progress! Review the incorrect answers and try again.")
    else:
        print("Keep practicing! Review the training materials and retry.")

    return score == total_tasks


# =============================================================================
# SOLUTIONS (Hidden by default)
# =============================================================================

SOLUTIONS = """
TASK 1 Solution:
    result = calculator.calculate(fuel_type="diesel", quantity=1000, region="US")
    # Answer: 2680.0 kg CO2e (1000 * 2.68)

TASK 2 Solution:
    result = calculator.calculate(fuel_type="natural_gas", quantity=500, region="US")
    # Answer: 965.0 kg CO2e (500 * 1.93)

TASK 3 Solution:
    result_a = calculator.calculate(fuel_type="diesel", quantity=100, region="US")
    result_b = calculator.calculate(fuel_type="diesel", quantity=100, region="US")
    # result_a["provenance_hash"] == result_b["provenance_hash"]  -> True

TASK 4 Solution:
    records = [
        {"fuel_type": "diesel", "quantity": 1000, "region": "US"},
        {"fuel_type": "natural_gas", "quantity": 500, "region": "US"},
        {"fuel_type": "gasoline", "quantity": 200, "region": "US"},
    ]
    result = calculator.calculate_batch(records)
    # Answer: 4107.0 kg CO2e (2680 + 965 + 462)

TASK 5 Solution:
    try:
        calculator.calculate(fuel_type="hydrogen", quantity=100, region="US")
    except ValueError:
        # This is expected - hydrogen is not a known fuel type
        pass
"""


if __name__ == "__main__":
    if "--show-solutions" in sys.argv:
        print(SOLUTIONS)
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        success = main()
        sys.exit(0 if success else 1)
