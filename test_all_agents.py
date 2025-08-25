#!/usr/bin/env python3
"""Test script to verify all agents are properly registered and accessible"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from greenlang.agents import (
    BaseAgent,
    FuelAgent,
    BoilerAgent,
    CarbonAgent,
    InputValidatorAgent,
    ReportAgent,
    BenchmarkAgent,
    GridFactorAgent,
    BuildingProfileAgent,
    IntensityAgent,
    RecommendationAgent
)
from greenlang.core.orchestrator import Orchestrator

def test_all_agents():
    """Test that all agents can be imported and registered"""
    
    print("Testing all GreenLang agents...")
    print("=" * 50)
    
    # List of all agents
    agents = [
        ("validator", InputValidatorAgent),
        ("fuel", FuelAgent),
        ("boiler", BoilerAgent),
        ("carbon", CarbonAgent),
        ("report", ReportAgent),
        ("benchmark", BenchmarkAgent),
        ("grid_factor", GridFactorAgent),
        ("building_profile", BuildingProfileAgent),
        ("intensity", IntensityAgent),
        ("recommendation", RecommendationAgent),
    ]
    
    # Test imports
    print("\n1. Testing imports:")
    for agent_id, agent_class in agents:
        try:
            assert agent_class is not None
            print(f"   [OK] {agent_class.__name__} imported successfully")
        except Exception as e:
            print(f"   [FAIL] Failed to import {agent_id}: {e}")
            return False
    
    # Test registration
    print("\n2. Testing agent registration:")
    orchestrator = Orchestrator()
    
    for agent_id, agent_class in agents:
        try:
            agent_instance = agent_class()
            orchestrator.register_agent(agent_id, agent_instance)
            print(f"   [OK] {agent_class.__name__} registered as '{agent_id}'")
        except Exception as e:
            print(f"   [FAIL] Failed to register {agent_class.__name__}: {e}")
            return False
    
    # Verify all agents are registered
    print("\n3. Verifying registered agents:")
    registered_count = len(orchestrator.agents)
    print(f"   Total agents registered: {registered_count}")
    
    if registered_count == len(agents):
        print(f"   [OK] All {len(agents)} agents successfully registered!")
    else:
        print(f"   [FAIL] Expected {len(agents)} agents, but only {registered_count} registered")
        return False
    
    print("\n4. Registered agents list:")
    for agent_id in orchestrator.agents:
        agent = orchestrator.agents[agent_id]
        print(f"   - {agent_id}: {agent.__class__.__name__}")
    
    print("\n" + "=" * 50)
    print("[SUCCESS] All tests passed! All agents are properly configured.")
    return True

if __name__ == "__main__":
    success = test_all_agents()
    exit(0 if success else 1)