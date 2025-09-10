"""Example 26: Long-run smoke test (100 executions; no memory blow-up)."""

import pytest
import gc
import psutil
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.fuel_agent import FuelAgent
except Exception:
    FuelAgent = None

@pytest.mark.example
@pytest.mark.timeout(60)  # 60 second timeout
def test_long_run_no_memory_leak():
    """100 executions without memory blow-up."""
    if FuelAgent is None:
        pytest.skip("FuelAgent not importable")
    
    try:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    except Exception:
        # If psutil not available, just run the test without memory check
        initial_memory = None
    
    agent = FuelAgent()
    results = []
    
    # Run 100 iterations
    for i in range(100):
        result = agent.run({
            "fuel_type": "electricity",
            "amount": 1000 + i,
            "unit": "kWh",
            "country": "US"
        })
        
        # Store only success status to avoid keeping all results in memory
        results.append(result["success"])
        
        # Periodic garbage collection
        if i % 20 == 0:
            gc.collect()
    
    # Check all succeeded
    assert all(results), "Some calculations failed"
    assert len(results) == 100, "Not all iterations completed"
    
    # Check memory usage if psutil available
    if initial_memory is not None:
        gc.collect()  # Force collection before measuring
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Allow up to 50MB growth (reasonable for Python runtime)
        assert memory_growth < 50, f"Memory grew by {memory_growth:.1f}MB"

@pytest.mark.example
def test_rapid_fire_calculations():
    """Rapid sequential calculations remain stable."""
    if FuelAgent is None:
        pytest.skip("FuelAgent not importable")
    
    agent = FuelAgent()
    
    # Rapid fire 50 calculations
    for _ in range(50):
        result = agent.run({
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "IN"
        })
        assert result["success"] is True
        assert result["data"]["co2e_emissions_kg"] > 0