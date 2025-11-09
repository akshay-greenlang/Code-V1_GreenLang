"""
Example 5: Parallel Batch Processing
=====================================

Demonstrates parallel batch processing with CalculatorAgent.
"""

import asyncio
from datetime import datetime
from greenlang.agents.templates import CalculatorAgent


async def main():
    """Run batch processing example."""
    # Create calculator agent
    agent = CalculatorAgent(config={
        "thread_workers": 4,
        "process_workers": 2
    })

    # Register calculation formula
    def calculate_emissions(activity: float, factor: float) -> float:
        return activity * factor

    agent.register_formula(
        "emissions",
        calculate_emissions,
        required_inputs=["activity", "factor"]
    )

    # Prepare batch inputs
    inputs_list = [
        {"activity": 1000, "factor": 2.5},
        {"activity": 1500, "factor": 3.2},
        {"activity": 800, "factor": 1.8},
        {"activity": 2000, "factor": 2.9},
        {"activity": 1200, "factor": 2.1},
    ]

    # Sequential processing
    print("\nSequential Processing:")
    start = datetime.now()
    sequential_results = []
    for inputs in inputs_list:
        result = await agent.calculate("emissions", inputs)
        sequential_results.append(result.value)
    sequential_duration = (datetime.now() - start).total_seconds()
    print(f"  Duration: {sequential_duration:.3f}s")
    print(f"  Results: {sequential_results}")

    # Parallel processing
    print("\nParallel Processing:")
    start = datetime.now()
    parallel_results = await agent.batch_calculate(
        formula_name="emissions",
        inputs_list=inputs_list,
        parallel=True,
        use_processes=False
    )
    parallel_duration = (datetime.now() - start).total_seconds()
    print(f"  Duration: {parallel_duration:.3f}s")
    print(f"  Results: {[r.value for r in parallel_results]}")

    # Speedup
    speedup = sequential_duration / parallel_duration
    print(f"\nSpeedup: {speedup:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
