from datetime import date

import pytest

from greenlang.v2.agent_registry import AgentRegistryEntry


def test_production_agent_requires_owner_and_support() -> None:
    entry = AgentRegistryEntry(
        agent_id="gl.agent.runtime",
        owner_team="platform-runtime",
        support_channel="#gl-runtime",
        current_version="2.1.0",
        state="production",
    )
    assert entry.state == "production"


def test_deprecated_agent_requires_replacement_and_date() -> None:
    with pytest.raises(ValueError):
        AgentRegistryEntry(
            agent_id="gl.agent.legacy",
            owner_team="platform-runtime",
            support_channel="#gl-runtime",
            current_version="1.9.0",
            state="deprecated",
        )

    entry = AgentRegistryEntry(
        agent_id="gl.agent.legacy",
        owner_team="platform-runtime",
        support_channel="#gl-runtime",
        current_version="1.9.0",
        state="deprecated",
        replacement_agent_id="gl.agent.runtime.v2",
        deprecation_date=date(2026, 12, 31),
    )
    assert entry.replacement_agent_id == "gl.agent.runtime.v2"

