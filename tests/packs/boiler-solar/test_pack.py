# -*- coding: utf-8 -*-
"""
Basic tests for boiler-solar pack.
"""

def test_pack_loads():
    """Test that the pack can be loaded."""
    from core.greenlang.packs.loader_simple import load_manifest
    import os
    
    # Get pack directory
    pack_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Load manifest
    manifest = load_manifest(pack_dir)
    
    assert manifest.name == "boiler-solar"
    assert manifest.version == "1.0.0"
    assert manifest.kind == "pack"
    assert manifest.license == "Apache-2.0"


def test_agents_exist():
    """Test that all agents exist."""
    import os
    
    pack_dir = os.path.dirname(os.path.dirname(__file__))
    agents_dir = os.path.join(pack_dir, "agents")
    
    expected_agents = [
        "boiler_efficiency.py",
        "solar_potential.py",
        "report_generator.py"
    ]
    
    for agent in expected_agents:
        agent_path = os.path.join(agents_dir, agent)
        assert os.path.exists(agent_path), f"Agent not found: {agent}"


def test_pipeline_exists():
    """Test that pipeline file exists."""
    import os
    
    pack_dir = os.path.dirname(os.path.dirname(__file__))
    pipeline_path = os.path.join(pack_dir, "gl.yaml")
    
    assert os.path.exists(pipeline_path), "Pipeline file gl.yaml not found"