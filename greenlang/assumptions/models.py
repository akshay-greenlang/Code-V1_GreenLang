# -*- coding: utf-8 -*-
"""
Assumptions Registry Data Models - AGENT-FOUND-004: Assumptions Registry

Pydantic v2 data models for the Assumptions Registry SDK. These models
are clean SDK versions that mirror the foundation agent enumerations
and models while providing a stable public API.

Models:
    - Enums: AssumptionDataType, AssumptionCategory, ScenarioType,
             ChangeType, ValidationSeverity
    - Core: ValidationRule, ValidationResult, AssumptionMetadata,
            AssumptionVersion, Assumption, Scenario
    - Audit: ChangeLogEntry
    - Graph: DependencyNode
    - Analysis: SensitivityResult
    - Value: AssumptionValue

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-004 Assumptions Registry
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enumerations
# =============================================================================


class AssumptionDataType(str, Enum):
    """Supported data types for assumption values."""
    FLOAT = "float"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    DATE = "date"
    LIST_FLOAT = "list_float"
    LIST_STRING = "list_string"
    DICT = "dict"


class AssumptionCategory(str, Enum):
    """Categories of assumptions for organization."""
    EMISSION_FACTOR = "emission_factor"
    CONVERSION_FACTOR = "conversion_factor"
    ECONOMIC = "economic"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    CLIMATE = "climate"
    ENERGY = "energy"
    TRANSPORT = "transport"
    WASTE = "waste"
    WATER = "water"
    CUSTOM = "custom"


class ScenarioType(str, Enum):
    """Pre-defined scenario types."""
    BASELINE = "baseline"
    OPTIMISTIC = "optimistic"
    CONSERVATIVE = "conservative"
    BEST_CASE = "best_case"
    WORST_CASE = "worst_case"
    REGULATORY = "regulatory"
    CUSTOM = "custom"


class ChangeType(str, Enum):
    """Types of changes to assumptions."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SCENARIO_OVERRIDE = "scenario_override"
    INHERIT = "inherit"
    REVERT = "revert"


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# Core Data Models
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class ValidationRule(BaseModel):
    """Validation rule for an assumption value."""
    rule_id: str = Field(..., description="Unique rule identifier")
    description: str = Field(..., description="Human-readable rule description")
    min_value: Optional[float] = Field(None, description="Minimum allowed value")
    max_value: Optional[float] = Field(None, description="Maximum allowed value")
    allowed_values: Optional[List[Any]] = Field(None, description="List of allowed values")
    regex_pattern: Optional[str] = Field(None, description="Regex pattern for string values")
    custom_validator: Optional[str] = Field(None, description="Name of custom validator function")
    severity: ValidationSeverity = Field(
        default=ValidationSeverity.ERROR,
        description="Severity if validation fails",
    )

    model_config = {"extra": "forbid"}


class ValidationResult(BaseModel):
    """Result of validating an assumption value."""
    is_valid: bool = Field(..., description="Overall validation status")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    info: List[str] = Field(default_factory=list, description="Informational messages")
    rules_checked: List[str] = Field(default_factory=list, description="Rules that were checked")

    model_config = {"extra": "forbid"}


class AssumptionMetadata(BaseModel):
    """Metadata for an assumption."""
    source: str = Field(..., description="Source of the assumption (e.g., EPA, IPCC)")
    source_url: Optional[str] = Field(None, description="URL to source document")
    source_year: Optional[int] = Field(None, description="Year of source publication")
    methodology: Optional[str] = Field(None, description="Methodology used")
    geographic_scope: Optional[str] = Field(None, description="Geographic applicability")
    temporal_scope: Optional[str] = Field(None, description="Temporal applicability")
    uncertainty_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Uncertainty percentage",
    )
    confidence_level: Optional[str] = Field(
        None, description="Confidence level (e.g., high, medium, low)",
    )
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    notes: Optional[str] = Field(None, description="Additional notes")
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata fields",
    )

    model_config = {"extra": "forbid"}


class AssumptionVersion(BaseModel):
    """A single version of an assumption value."""
    version_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique version ID",
    )
    version_number: int = Field(..., ge=1, description="Sequential version number")
    value: Any = Field(..., description="The assumption value")
    effective_from: datetime = Field(..., description="When this version becomes effective")
    effective_until: Optional[datetime] = Field(None, description="When this version expires")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    created_by: str = Field(..., description="User who created this version")
    change_reason: str = Field(..., description="Reason for the change")
    change_type: ChangeType = Field(..., description="Type of change")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")
    parent_version_id: Optional[str] = Field(None, description="Previous version ID")
    scenario_id: Optional[str] = Field(None, description="Scenario this version applies to")

    model_config = {"extra": "forbid"}


class Assumption(BaseModel):
    """Complete assumption definition with all versions and metadata."""
    assumption_id: str = Field(..., description="Unique assumption identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description")
    category: AssumptionCategory = Field(..., description="Assumption category")
    data_type: AssumptionDataType = Field(..., description="Data type of the value")
    unit: Optional[str] = Field(None, description="Unit of measurement")

    # Current value (baseline scenario)
    current_value: Any = Field(..., description="Current value for baseline scenario")
    default_value: Any = Field(..., description="Default/fallback value")

    # Version history
    versions: List[AssumptionVersion] = Field(
        default_factory=list, description="Version history",
    )

    # Validation
    validation_rules: List[ValidationRule] = Field(
        default_factory=list, description="Validation rules",
    )

    # Metadata
    metadata: AssumptionMetadata = Field(..., description="Assumption metadata")

    # Dependency tracking
    depends_on: List[str] = Field(
        default_factory=list, description="Assumptions this depends on",
    )
    used_by: List[str] = Field(
        default_factory=list, description="Calculations using this assumption",
    )

    # Inheritance
    parent_assumption_id: Optional[str] = Field(
        None, description="Parent assumption for inheritance",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=_utcnow, description="Last update timestamp")

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash of current state")

    model_config = {"extra": "forbid"}

    @field_validator("assumption_id")
    @classmethod
    def validate_assumption_id(cls, v: str) -> str:
        """Validate assumption ID format."""
        if not v or len(v) < 3:
            raise ValueError("assumption_id must be at least 3 characters")
        if not v.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError(
                "assumption_id must be alphanumeric with underscores, hyphens, or dots"
            )
        return v


class Scenario(BaseModel):
    """A scenario containing assumption overrides."""
    scenario_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique scenario ID",
    )
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    scenario_type: ScenarioType = Field(..., description="Type of scenario")

    # Overrides: assumption_id -> value
    overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Assumption value overrides",
    )

    # Metadata
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    created_by: str = Field(..., description="User who created the scenario")
    is_active: bool = Field(default=True, description="Whether scenario is active")
    parent_scenario_id: Optional[str] = Field(
        None, description="Parent scenario for inheritance",
    )

    # Tags for filtering
    tags: List[str] = Field(default_factory=list, description="Scenario tags")

    model_config = {"extra": "forbid"}


class ChangeLogEntry(BaseModel):
    """Audit log entry for assumption changes."""
    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique log ID",
    )
    timestamp: datetime = Field(default_factory=_utcnow, description="Change timestamp")
    user_id: str = Field(..., description="User who made the change")
    change_type: ChangeType = Field(..., description="Type of change")
    assumption_id: str = Field(..., description="Affected assumption ID")
    scenario_id: Optional[str] = Field(None, description="Affected scenario ID")

    # Change details
    old_value: Optional[Any] = Field(None, description="Previous value")
    new_value: Optional[Any] = Field(None, description="New value")
    change_reason: str = Field(..., description="Reason for change")

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")

    model_config = {"extra": "forbid"}


class DependencyNode(BaseModel):
    """Node in the dependency graph."""
    assumption_id: str = Field(..., description="Assumption identifier")
    calculation_ids: List[str] = Field(
        default_factory=list, description="Calculations using this assumption",
    )
    upstream: List[str] = Field(
        default_factory=list, description="Assumptions this depends on",
    )
    downstream: List[str] = Field(
        default_factory=list, description="Assumptions depending on this",
    )

    model_config = {"extra": "forbid"}


class SensitivityResult(BaseModel):
    """Result of sensitivity analysis for an assumption."""
    assumption_id: str = Field(..., description="Assumption identifier")
    baseline_value: Any = Field(..., description="Current baseline value")
    scenario_values: Dict[str, Any] = Field(
        default_factory=dict, description="Values by scenario name",
    )
    min_value: Optional[float] = Field(None, description="Minimum across scenarios")
    max_value: Optional[float] = Field(None, description="Maximum across scenarios")
    range_value: Optional[float] = Field(None, description="Range across scenarios")
    range_pct: Optional[float] = Field(None, description="Range as percentage of baseline")
    dependency_count: int = Field(default=0, description="Number of dependent calculations")
    dependent_calculations: List[str] = Field(
        default_factory=list, description="Calculations affected",
    )

    model_config = {"extra": "forbid"}


class AssumptionValue(BaseModel):
    """Resolved value for an assumption, possibly with scenario override."""
    assumption_id: str = Field(..., description="Assumption identifier")
    value: Any = Field(..., description="The resolved value")
    value_source: str = Field(
        default="baseline", description="Where the value came from",
    )
    unit: Optional[str] = Field(None, description="Unit of measurement")
    data_type: str = Field(default="float", description="Data type of the value")

    model_config = {"extra": "forbid"}


__all__ = [
    # Enumerations
    "AssumptionDataType",
    "AssumptionCategory",
    "ScenarioType",
    "ChangeType",
    "ValidationSeverity",
    # Core models
    "ValidationRule",
    "ValidationResult",
    "AssumptionMetadata",
    "AssumptionVersion",
    "Assumption",
    "Scenario",
    # Audit models
    "ChangeLogEntry",
    # Graph models
    "DependencyNode",
    # Analysis models
    "SensitivityResult",
    # Value models
    "AssumptionValue",
]
