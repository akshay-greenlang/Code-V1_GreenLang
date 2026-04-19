# -*- coding: utf-8 -*-
"""V2 agent registry policy models."""

from __future__ import annotations

from datetime import date
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal


AgentState = Literal["incubating", "qualified", "production", "deprecated", "retired"]


class AgentRegistryEntry(BaseModel):
    agent_id: str
    owner_team: str
    support_channel: str
    current_version: str
    state: AgentState
    deprecation_date: date | None = None
    replacement_agent_id: str | None = None

    @field_validator("agent_id", "owner_team", "support_channel", "current_version")
    @classmethod
    def _required_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("replacement_agent_id")
    @classmethod
    def _replacement_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("replacement_agent_id must be non-empty if provided")
        return value.strip()

    @model_validator(mode="after")
    def _deprecated_requirements(self):
        if self.state == "deprecated":
            if self.deprecation_date is None:
                raise ValueError("deprecation_date is required when state is deprecated")
            if not self.replacement_agent_id:
                raise ValueError("replacement_agent_id is required when state is deprecated")
        return self

