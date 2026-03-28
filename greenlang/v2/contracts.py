# -*- coding: utf-8 -*-
"""GreenLang v2 contract models and validators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


PACK_LIFECYCLE_VALUES = Literal["experimental", "candidate", "supported", "regulated-critical", "deprecated"]


class V2PackSecurity(BaseModel):
    signed: bool = Field(..., description="True when pack is cryptographically signed.")
    signatures: list[str] = Field(default_factory=list)
    sbom: str | None = None
    provenance: str | None = None


class V2PackMetadata(BaseModel):
    owner_team: str
    support_channel: str
    lifecycle: PACK_LIFECYCLE_VALUES
    quality_tier: Literal["experimental", "candidate", "supported", "regulated-critical"]

    @field_validator("owner_team", "support_channel")
    @classmethod
    def _required_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


class V2PackContract(BaseModel):
    contract_version: Literal["2.0"]
    name: str
    app_id: str
    version: str
    kind: Literal["pack"]
    runtime: Literal["greenlang-v2"]
    entry_pipeline: str
    metadata: V2PackMetadata
    security: V2PackSecurity

    @field_validator("name", "app_id", "entry_pipeline")
    @classmethod
    def _required_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


class V2PipelineStage(BaseModel):
    id: str
    type: Literal["validate", "compute", "policy", "export", "audit"]


class V2RuntimeConventions(BaseModel):
    command: str
    success_exit_code: int = 0
    blocked_exit_code: int = 4
    artifact_contract: list[str] = Field(default_factory=list)

    @field_validator("command")
    @classmethod
    def _required_command(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("command must be a non-empty string")
        return value.strip()


class V2PipelineContract(BaseModel):
    contract_version: Literal["2.0"]
    app_id: str
    pipeline_id: str
    runtime: Literal["greenlang-v2"]
    stages: list[V2PipelineStage]
    runtime_conventions: V2RuntimeConventions

    @field_validator("app_id", "pipeline_id")
    @classmethod
    def _required_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("stages")
    @classmethod
    def _require_standard_stages(cls, stages: list[V2PipelineStage]) -> list[V2PipelineStage]:
        expected = ["validate", "compute", "policy", "export", "audit"]
        got = [stage.type for stage in stages]
        if got != expected:
            raise ValueError(f"stages must exactly match {expected}, got {got}")
        return stages


@dataclass
class ValidationFinding:
    path: str
    ok: bool
    errors: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("yaml root must be a mapping")
    return loaded


def validate_v2_pack(path: Path) -> ValidationFinding:
    errors: list[str] = []
    try:
        data = _load_yaml(path)
        model = V2PackContract.model_validate(data)
        pack_dir = path.parent
        for signature in model.security.signatures:
            if not (pack_dir / signature).exists():
                errors.append(f"missing signature file: {signature}")
        if model.metadata.quality_tier in {"supported", "regulated-critical"} and not model.security.signed:
            errors.append("supported and regulated-critical tiers require signed=true")
        if model.security.signed and not model.security.signatures:
            errors.append("security.signed=true requires at least one signature")
    except Exception as exc:  # pragma: no cover
        errors.append(str(exc))
    return ValidationFinding(path=str(path), ok=not errors, errors=errors)


def validate_v2_pipeline(path: Path) -> ValidationFinding:
    errors: list[str] = []
    try:
        data = _load_yaml(path)
        model = V2PipelineContract.model_validate(data)
        if not model.runtime_conventions.artifact_contract:
            errors.append("runtime_conventions.artifact_contract must be non-empty")
    except Exception as exc:  # pragma: no cover
        errors.append(str(exc))
    return ValidationFinding(path=str(path), ok=not errors, errors=errors)

