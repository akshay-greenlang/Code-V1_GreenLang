# -*- coding: utf-8 -*-
"""V2 agent lifecycle registry validation and runtime policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .agent_registry import AgentRegistryEntry

DEFAULT_AGENT_REGISTRY = Path("greenlang/agents/v2_agent_registry.yaml")


@dataclass
class AgentLifecycleValidation:
    ok: bool
    errors: list[str]
    entries: list[AgentRegistryEntry]


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("agent registry yaml root must be a mapping")
    return loaded


def load_agent_registry(path: Path = DEFAULT_AGENT_REGISTRY) -> list[dict[str, Any]]:
    payload = _read_yaml(path)
    agents = payload.get("agents", [])
    if not isinstance(agents, list):
        raise ValueError("agent registry field 'agents' must be a list")
    return [item for item in agents if isinstance(item, dict)]


def validate_agent_registry(path: Path = DEFAULT_AGENT_REGISTRY) -> AgentLifecycleValidation:
    errors: list[str] = []
    validated: list[AgentRegistryEntry] = []
    if not path.exists():
        return AgentLifecycleValidation(ok=False, errors=[f"missing agent registry: {path.as_posix()}"], entries=[])

    seen_ids: set[str] = set()
    by_id: dict[str, AgentRegistryEntry] = {}
    try:
        raw_entries = load_agent_registry(path)
    except Exception as exc:
        return AgentLifecycleValidation(ok=False, errors=[str(exc)], entries=[])

    for idx, raw in enumerate(raw_entries):
        try:
            entry = AgentRegistryEntry.model_validate(raw)
        except Exception as exc:
            errors.append(f"entry[{idx}] invalid: {exc}")
            continue
        if entry.agent_id in seen_ids:
            errors.append(f"duplicate agent_id: {entry.agent_id}")
            continue
        seen_ids.add(entry.agent_id)
        validated.append(entry)
        by_id[entry.agent_id] = entry

    for entry in validated:
        if entry.state == "deprecated":
            replacement = by_id.get(entry.replacement_agent_id or "")
            if replacement is None:
                errors.append(
                    f"{entry.agent_id}: replacement_agent_id '{entry.replacement_agent_id}' "
                    "is not defined in registry"
                )
            elif replacement.state == "retired":
                errors.append(
                    f"{entry.agent_id}: replacement_agent_id '{entry.replacement_agent_id}' cannot be retired"
                )
        if entry.state == "retired" and not entry.replacement_agent_id:
            errors.append(f"{entry.agent_id}: retired agents must include replacement_agent_id")

    return AgentLifecycleValidation(ok=not errors, errors=errors, entries=validated)


def enforce_agent_state_for_runtime(agent_id: str, path: Path = DEFAULT_AGENT_REGISTRY) -> None:
    result = validate_agent_registry(path)
    if not result.ok:
        raise ValueError(f"agent lifecycle registry invalid: {result.errors}")
    by_id = {entry.agent_id: entry for entry in result.entries}
    entry = by_id.get(agent_id)
    if not entry:
        return
    if entry.state == "retired":
        raise ValueError(
            f"agent '{agent_id}' is retired and blocked from runtime resolution; "
            f"use replacement '{entry.replacement_agent_id}'"
        )
    if entry.state == "deprecated" and not entry.replacement_agent_id:
        raise ValueError(
            f"agent '{agent_id}' is deprecated and missing replacement; runtime resolution denied"
        )
