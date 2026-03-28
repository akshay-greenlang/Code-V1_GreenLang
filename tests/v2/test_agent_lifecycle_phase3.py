from pathlib import Path

from greenlang.v2.agent_lifecycle import (
    enforce_agent_state_for_runtime,
    validate_agent_registry,
)


def test_phase3_agent_registry_validates() -> None:
    registry = Path("greenlang/agents/v2_agent_registry.yaml")
    result = validate_agent_registry(registry)
    assert result.ok, result.errors
    assert result.entries


def test_runtime_enforcement_blocks_retired_agent() -> None:
    registry = Path("greenlang/agents/v2_agent_registry.yaml")
    try:
        enforce_agent_state_for_runtime("greenlang.agent.retired.taxonomy-v1", registry)
        assert False, "retired agent should be blocked"
    except ValueError as exc:
        assert "retired" in str(exc)


def test_runtime_enforcement_allows_production_agent() -> None:
    registry = Path("greenlang/agents/v2_agent_registry.yaml")
    enforce_agent_state_for_runtime("greenlang.agent.runtime.orchestrator", registry)
