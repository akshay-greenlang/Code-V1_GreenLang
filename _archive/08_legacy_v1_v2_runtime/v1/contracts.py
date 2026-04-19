# -*- coding: utf-8 -*-
"""
GreenLang v1 contract models and validators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class V1PackSecurity(BaseModel):
    signed: bool = Field(
        ...,
        description="True when pack is signed and signature evidence is present.",
    )
    signatures: list[str] = Field(
        default_factory=list,
        description="List of signature files in the pack directory.",
    )
    sbom: str | None = Field(
        default=None,
        description="Optional SBOM artifact path.",
    )


class V1PackMetadata(BaseModel):
    owner_team: str
    support_channel: str
    lifecycle: Literal["draft", "candidate", "supported", "deprecated"]

    @field_validator("owner_team", "support_channel")
    @classmethod
    def _required_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


class V1PackContract(BaseModel):
    contract_version: Literal["1.0"]
    name: str
    app_id: str
    version: str
    kind: Literal["pack"]
    runtime: Literal["greenlang-v1"]
    entry_pipeline: str
    metadata: V1PackMetadata
    security: V1PackSecurity

    @field_validator("name", "app_id", "entry_pipeline")
    @classmethod
    def _required_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()


class V1PipelineStage(BaseModel):
    id: str
    type: Literal["validate", "compute", "policy", "export", "audit"]


class V1RuntimeConventions(BaseModel):
    command: str = Field(
        ...,
        description="Canonical command grammar used for this app profile.",
    )
    success_exit_code: int = Field(default=0)
    blocked_exit_code: int = Field(default=4)
    artifact_contract: list[str] = Field(default_factory=list)

    @field_validator("command")
    @classmethod
    def _required_command(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("command must be a non-empty string")
        return value.strip()


class V1PipelineContract(BaseModel):
    contract_version: Literal["1.0"]
    app_id: str
    pipeline_id: str
    runtime: Literal["greenlang-v1"]
    stages: list[V1PipelineStage]
    runtime_conventions: V1RuntimeConventions

    @field_validator("app_id", "pipeline_id")
    @classmethod
    def _required_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("stages")
    @classmethod
    def _require_standard_stages(
        cls, stages: list[V1PipelineStage]
    ) -> list[V1PipelineStage]:
        expected = ["validate", "compute", "policy", "export", "audit"]
        got = [stage.type for stage in stages]
        if got != expected:
            raise ValueError(
                f"stages must exactly match {expected}, got {got}"
            )
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


def validate_v1_pack(path: Path) -> ValidationFinding:
    errors: list[str] = []
    try:
        data = _load_yaml(path)
        model = V1PackContract.model_validate(data)
        pack_dir = path.parent
        for signature in model.security.signatures:
            if not (pack_dir / signature).exists():
                errors.append(f"missing signature file: {signature}")
        if model.security.signed and not model.security.signatures:
            errors.append("security.signed=true requires at least one signature")
    except Exception as exc:  # pragma: no cover - handled by tests
        errors.append(str(exc))
    return ValidationFinding(path=str(path), ok=not errors, errors=errors)


def validate_v1_pipeline(path: Path) -> ValidationFinding:
    errors: list[str] = []
    try:
        data = _load_yaml(path)
        model = V1PipelineContract.model_validate(data)
        if not model.runtime_conventions.artifact_contract:
            errors.append("runtime_conventions.artifact_contract must be non-empty")
    except Exception as exc:  # pragma: no cover - handled by tests
        errors.append(str(exc))
    return ValidationFinding(path=str(path), ok=not errors, errors=errors)

