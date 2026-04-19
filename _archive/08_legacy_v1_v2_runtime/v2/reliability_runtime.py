# -*- coding: utf-8 -*-
"""Runtime helpers for V2 connector reliability profile enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .reliability import ConnectorReliabilityProfile

DEFAULT_CONNECTOR_REGISTRY = Path("applications/connectors/v2_connector_registry.yaml")


class ErrorClass(str, Enum):
    TRANSIENT = "transient"
    THROTTLING = "throttling"
    AUTH = "auth"
    SCHEMA = "schema"
    PERMANENT = "permanent"


@dataclass
class ConnectorRegistryValidation:
    ok: bool
    errors: list[str]
    profiles: list[ConnectorReliabilityProfile]


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("connector registry yaml root must be a mapping")
    return loaded


def load_connector_registry(path: Path = DEFAULT_CONNECTOR_REGISTRY) -> list[dict[str, Any]]:
    payload = _read_yaml(path)
    connectors = payload.get("connectors", [])
    if not isinstance(connectors, list):
        raise ValueError("connector registry field 'connectors' must be a list")
    return [item for item in connectors if isinstance(item, dict)]


def validate_connector_registry(path: Path = DEFAULT_CONNECTOR_REGISTRY) -> ConnectorRegistryValidation:
    if not path.exists():
        return ConnectorRegistryValidation(
            ok=False,
            errors=[f"missing connector registry: {path.as_posix()}"],
            profiles=[],
        )
    errors: list[str] = []
    profiles: list[ConnectorReliabilityProfile] = []
    seen: set[str] = set()
    try:
        connectors = load_connector_registry(path)
    except Exception as exc:
        return ConnectorRegistryValidation(ok=False, errors=[str(exc)], profiles=[])
    for idx, item in enumerate(connectors):
        connector_id = str(item.get("connector_id", "")).strip()
        owner = str(item.get("owner_team", "")).strip()
        support = str(item.get("support_channel", "")).strip()
        profile_payload = item.get("reliability_profile", {})
        if not connector_id:
            errors.append(f"entry[{idx}] connector_id is required")
            continue
        if connector_id in seen:
            errors.append(f"duplicate connector_id in registry: {connector_id}")
            continue
        seen.add(connector_id)
        if not owner:
            errors.append(f"{connector_id}: owner_team is required")
        if not support:
            errors.append(f"{connector_id}: support_channel is required")
        if not isinstance(profile_payload, dict):
            errors.append(f"{connector_id}: reliability_profile must be a mapping")
            continue
        try:
            profile = ConnectorReliabilityProfile(
                connector_id=connector_id,
                owner_team=owner,
                retry=profile_payload.get("retry", {}),
                timeout=profile_payload.get("timeout", {}),
                circuit_breaker=profile_payload.get("circuit_breaker", {}),
                idempotency_required=bool(profile_payload.get("idempotency_required", True)),
                dead_letter_enabled=bool(profile_payload.get("dead_letter_enabled", True)),
            )
            profiles.append(profile)
        except Exception as exc:
            errors.append(f"{connector_id}: {exc}")
    return ConnectorRegistryValidation(ok=not errors, errors=errors, profiles=profiles)


def get_reliability_profile(
    connector_id: str,
    path: Path = DEFAULT_CONNECTOR_REGISTRY,
) -> ConnectorReliabilityProfile | None:
    result = validate_connector_registry(path)
    if not result.ok:
        raise ValueError(f"connector registry invalid: {result.errors}")
    for profile in result.profiles:
        if profile.connector_id == connector_id:
            return profile
    return None


def classify_connector_error(exc: Exception) -> ErrorClass:
    message = str(exc).lower()
    if "429" in message or "throttle" in message or "rate limit" in message:
        return ErrorClass.THROTTLING
    if "auth" in message or "unauthorized" in message or "forbidden" in message:
        return ErrorClass.AUTH
    if "schema" in message or "validation" in message:
        return ErrorClass.SCHEMA
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return ErrorClass.TRANSIENT
    return ErrorClass.PERMANENT
