# -*- coding: utf-8 -*-
"""AgentSpec v2 Migration Validation Tests - Batch 1.

This test suite validates the AgentSpec v2 pack.yaml files for the first
batch of migrated agents:
- CarbonAgentAI
- GridFactorAgentAI
- BoilerReplacementAgentAI

Tests verify:
1. Pack.yaml files exist and are valid YAML
2. Required AgentSpec v2 sections are present
3. Input/output schemas are well-formed
4. Compliance fields are correct
5. Agents can be wrapped with AgentSpecV2Wrapper

Author: GreenLang Framework Team
Date: 2025-10-26 (Phase 2 - STANDARDIZATION)
Status: Production Ready
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Any

from greenlang.agents.agentspec_v2_compat import wrap_agent_v2
from greenlang.agents.carbon_agent_ai import CarbonAgentAI
from greenlang.agents.grid_factor_agent_ai import GridFactorAgentAI
from greenlang.agents.boiler_replacement_agent_ai import BoilerReplacementAgent_AI


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def pack_root():
    """Get the packs directory root."""
    return Path(__file__).parent.parent.parent / "packs"


@pytest.fixture
def carbon_ai_pack(pack_root):
    """Load CarbonAgentAI pack.yaml."""
    pack_path = pack_root / "carbon_ai" / "pack.yaml"
    with open(pack_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def grid_factor_ai_pack(pack_root):
    """Load GridFactorAgentAI pack.yaml."""
    pack_path = pack_root / "grid_factor_ai" / "pack.yaml"
    with open(pack_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def boiler_replacement_ai_pack(pack_root):
    """Load BoilerReplacementAgentAI pack.yaml."""
    pack_path = pack_root / "boiler_replacement_ai" / "pack.yaml"
    with open(pack_path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================================
# Pack.yaml Structure Tests
# ==============================================================================

def test_carbon_ai_pack_exists(pack_root):
    """Test that CarbonAgentAI pack.yaml exists."""
    pack_path = pack_root / "carbon_ai" / "pack.yaml"
    assert pack_path.exists(), f"Pack file not found: {pack_path}"


def test_grid_factor_ai_pack_exists(pack_root):
    """Test that GridFactorAgentAI pack.yaml exists."""
    pack_path = pack_root / "grid_factor_ai" / "pack.yaml"
    assert pack_path.exists(), f"Pack file not found: {pack_path}"


def test_boiler_replacement_ai_pack_exists(pack_root):
    """Test that BoilerReplacementAgentAI pack.yaml exists."""
    pack_path = pack_root / "boiler_replacement_ai" / "pack.yaml"
    assert pack_path.exists(), f"Pack file not found: {pack_path}"


# ==============================================================================
# Schema Version Tests
# ==============================================================================

@pytest.mark.parametrize("pack_fixture,agent_name", [
    ("carbon_ai_pack", "CarbonAgentAI"),
    ("grid_factor_ai_pack", "GridFactorAgentAI"),
    ("boiler_replacement_ai_pack", "BoilerReplacementAgentAI"),
])
def test_pack_schema_version(pack_fixture, agent_name, request):
    """Test that pack has correct schema_version."""
    pack = request.getfixturevalue(pack_fixture)
    assert "schema_version" in pack, f"{agent_name}: Missing schema_version"
    assert pack["schema_version"] == "2.0.0", f"{agent_name}: Wrong schema version"


# ==============================================================================
# Required Sections Tests
# ==============================================================================

@pytest.mark.parametrize("pack_fixture,agent_name", [
    ("carbon_ai_pack", "CarbonAgentAI"),
    ("grid_factor_ai_pack", "GridFactorAgentAI"),
    ("boiler_replacement_ai_pack", "BoilerReplacementAgentAI"),
])
def test_pack_has_required_sections(pack_fixture, agent_name, request):
    """Test that pack has all required AgentSpec v2 sections."""
    pack = request.getfixturevalue(pack_fixture)

    required_sections = [
        "schema_version",
        "id",
        "name",
        "version",
        "summary",
        "compute",
        "ai",
        "realtime",
        "provenance",
        "metadata",
    ]

    for section in required_sections:
        assert section in pack, f"{agent_name}: Missing required section '{section}'"


# ==============================================================================
# Compute Section Tests
# ==============================================================================

@pytest.mark.parametrize("pack_fixture,agent_name", [
    ("carbon_ai_pack", "CarbonAgentAI"),
    ("grid_factor_ai_pack", "GridFactorAgentAI"),
    ("boiler_replacement_ai_pack", "BoilerReplacementAgentAI"),
])
def test_compute_section_structure(pack_fixture, agent_name, request):
    """Test compute section has required fields."""
    pack = request.getfixturevalue(pack_fixture)
    compute = pack["compute"]

    assert "entrypoint" in compute, f"{agent_name}: Missing entrypoint"
    assert "deterministic" in compute, f"{agent_name}: Missing deterministic flag"
    assert "inputs" in compute, f"{agent_name}: Missing inputs schema"
    assert "outputs" in compute, f"{agent_name}: Missing outputs schema"

    # Verify deterministic is True
    assert compute["deterministic"] is True, f"{agent_name}: Should be deterministic"


@pytest.mark.parametrize("pack_fixture,agent_name,expected_inputs", [
    ("carbon_ai_pack", "CarbonAgentAI", ["emissions"]),
    ("grid_factor_ai_pack", "GridFactorAgentAI", ["country", "fuel_type", "unit"]),
    ("boiler_replacement_ai_pack", "BoilerReplacementAgentAI", [
        "boiler_type", "fuel_type", "rated_capacity_kw", "age_years"
    ]),
])
def test_input_schema_has_required_fields(pack_fixture, agent_name, expected_inputs, request):
    """Test that input schema has expected fields."""
    pack = request.getfixturevalue(pack_fixture)
    inputs = pack["compute"]["inputs"]

    for field in expected_inputs:
        assert field in inputs, f"{agent_name}: Missing input field '{field}'"
        assert "dtype" in inputs[field], f"{agent_name}: Input '{field}' missing dtype"
        assert "description" in inputs[field], f"{agent_name}: Input '{field}' missing description"


@pytest.mark.parametrize("pack_fixture,agent_name,expected_outputs", [
    ("carbon_ai_pack", "CarbonAgentAI", [
        "total_co2e_kg", "total_co2e_tons", "emissions_breakdown"
    ]),
    ("grid_factor_ai_pack", "GridFactorAgentAI", [
        "emission_factor", "source", "country"
    ]),
    ("boiler_replacement_ai_pack", "BoilerReplacementAgentAI", [
        "current_efficiency", "recommended_technology", "simple_payback_years"
    ]),
])
def test_output_schema_has_required_fields(pack_fixture, agent_name, expected_outputs, request):
    """Test that output schema has expected fields."""
    pack = request.getfixturevalue(pack_fixture)
    outputs = pack["compute"]["outputs"]

    for field in expected_outputs:
        assert field in outputs, f"{agent_name}: Missing output field '{field}'"
        assert "dtype" in outputs[field], f"{agent_name}: Output '{field}' missing dtype"
        assert "description" in outputs[field], f"{agent_name}: Output '{field}' missing description"


# ==============================================================================
# AI Section Tests
# ==============================================================================

@pytest.mark.parametrize("pack_fixture,agent_name", [
    ("carbon_ai_pack", "CarbonAgentAI"),
    ("grid_factor_ai_pack", "GridFactorAgentAI"),
    ("boiler_replacement_ai_pack", "BoilerReplacementAgentAI"),
])
def test_ai_section_structure(pack_fixture, agent_name, request):
    """Test AI section has required fields."""
    pack = request.getfixturevalue(pack_fixture)
    ai = pack["ai"]

    assert "system_prompt" in ai, f"{agent_name}: Missing system_prompt"
    assert "budget" in ai, f"{agent_name}: Missing budget"
    assert "tools" in ai, f"{agent_name}: Missing tools list"

    # Verify system prompt is non-empty
    assert len(ai["system_prompt"]) > 100, f"{agent_name}: System prompt too short"

    # Verify budget has max_usd_per_run
    assert "max_usd_per_run" in ai["budget"], f"{agent_name}: Missing max_usd_per_run"


# ==============================================================================
# Provenance Section Tests
# ==============================================================================

@pytest.mark.parametrize("pack_fixture,agent_name", [
    ("carbon_ai_pack", "CarbonAgentAI"),
    ("grid_factor_ai_pack", "GridFactorAgentAI"),
    ("boiler_replacement_ai_pack", "BoilerReplacementAgentAI"),
])
def test_provenance_section_structure(pack_fixture, agent_name, request):
    """Test provenance section has required fields."""
    pack = request.getfixturevalue(pack_fixture)
    provenance = pack["provenance"]

    assert "ef_pinning" in provenance, f"{agent_name}: Missing ef_pinning"
    assert "gwp_set" in provenance, f"{agent_name}: Missing gwp_set"
    assert "citation_required" in provenance, f"{agent_name}: Missing citation_required"
    assert "determinism_required" in provenance, f"{agent_name}: Missing determinism_required"

    # Verify flags are correct
    assert provenance["citation_required"] is True, f"{agent_name}: Citations should be required"
    assert provenance["determinism_required"] is True, f"{agent_name}: Determinism should be required"


# ==============================================================================
# Metadata Tests
# ==============================================================================

@pytest.mark.parametrize("pack_fixture,agent_name", [
    ("carbon_ai_pack", "CarbonAgentAI"),
    ("grid_factor_ai_pack", "GridFactorAgentAI"),
    ("boiler_replacement_ai_pack", "BoilerReplacementAgentAI"),
])
def test_metadata_has_compliance(pack_fixture, agent_name, request):
    """Test metadata has compliance field with AgentSpec_v2."""
    pack = request.getfixturevalue(pack_fixture)
    metadata = pack["metadata"]

    assert "compliance" in metadata, f"{agent_name}: Missing compliance field"
    compliance_list = metadata["compliance"]

    assert "AgentSpec_v2" in compliance_list, \
        f"{agent_name}: Should declare AgentSpec_v2 compliance"


# ==============================================================================
# Agent Wrapping Tests (Integration)
# ==============================================================================

def test_carbon_ai_can_be_wrapped(pack_root):
    """Test that CarbonAgentAI can be wrapped with AgentSpecV2Wrapper."""
    agent = CarbonAgentAI()
    pack_path = pack_root / "carbon_ai"

    # This should not raise an exception
    try:
        wrapped = wrap_agent_v2(agent, pack_path=pack_path, enable_validation=False)
        assert wrapped is not None
    except Exception as e:
        pytest.fail(f"Failed to wrap CarbonAgentAI: {e}")


def test_grid_factor_ai_can_be_wrapped(pack_root):
    """Test that GridFactorAgentAI can be wrapped with AgentSpecV2Wrapper."""
    agent = GridFactorAgentAI()
    pack_path = pack_root / "grid_factor_ai"

    # This should not raise an exception
    try:
        wrapped = wrap_agent_v2(agent, pack_path=pack_path, enable_validation=False)
        assert wrapped is not None
    except Exception as e:
        pytest.fail(f"Failed to wrap GridFactorAgentAI: {e}")


def test_boiler_replacement_ai_can_be_wrapped(pack_root):
    """Test that BoilerReplacementAgentAI can be wrapped with AgentSpecV2Wrapper."""
    agent = BoilerReplacementAgent_AI()
    pack_path = pack_root / "boiler_replacement_ai"

    # This should not raise an exception
    try:
        wrapped = wrap_agent_v2(agent, pack_path=pack_path, enable_validation=False)
        assert wrapped is not None
    except Exception as e:
        pytest.fail(f"Failed to wrap BoilerReplacementAgentAI: {e}")


# ==============================================================================
# Summary Test
# ==============================================================================

def test_batch1_migration_summary(pack_root):
    """Print summary of Batch 1 migration status."""
    agents = [
        ("carbon_ai", "CarbonAgentAI"),
        ("grid_factor_ai", "GridFactorAgentAI"),
        ("boiler_replacement_ai", "BoilerReplacementAgentAI"),
    ]

    print("\n" + "="*80)
    print("AgentSpec v2 Migration - Batch 1 Summary")
    print("="*80)

    for pack_dir, agent_name in agents:
        pack_path = pack_root / pack_dir / "pack.yaml"
        status = "✅ COMPLETE" if pack_path.exists() else "❌ MISSING"
        print(f"{agent_name:40s} {status}")

    print("="*80)
    print("Phase 2 Progress: 3/12 agents migrated (25%)")
    print("="*80 + "\n")
