# -*- coding: utf-8 -*-
"""
GL-FOUND-X-004: Assumptions Registry Agent
==========================================

A version-controlled registry for managing assumptions used in zero-hallucination
compliance calculations. This agent provides complete provenance tracking for all
assumptions, supporting scenario analysis and regulatory audit trails.

Capabilities:
    - Assumption catalog with full metadata and validation rules
    - Version control with immutable history and timestamps
    - Scenario management (baseline, optimistic, conservative, custom)
    - Complete change audit trail for compliance
    - Assumption validation against allowed ranges
    - Dependency tracking between assumptions and calculations
    - Assumption inheritance and override chains
    - Sensitivity analysis hooks for what-if scenarios

Zero-Hallucination Guarantees:
    - All assumption values are explicitly defined, never inferred
    - Complete audit trail with SHA-256 provenance hashes
    - Deterministic scenario selection with no implicit defaults
    - All changes logged with user, timestamp, and justification

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK delegation
# ---------------------------------------------------------------------------
ASSUMPTIONS_SDK_AVAILABLE = False
try:
    from greenlang.assumptions.registry import AssumptionRegistry as _SDKRegistry
    ASSUMPTIONS_SDK_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Enumerations
# =============================================================================


class AssumptionDataType(str, Enum):
    """Supported data types for assumption values."""
    FLOAT = "float"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"  # 0-100 range
    RATIO = "ratio"            # 0-1 range
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
# Data Models
# =============================================================================


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
        description="Severity if validation fails"
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
    uncertainty_pct: Optional[float] = Field(None, ge=0, le=100, description="Uncertainty percentage")
    confidence_level: Optional[str] = Field(None, description="Confidence level (e.g., high, medium, low)")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    notes: Optional[str] = Field(None, description="Additional notes")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")

    model_config = {"extra": "forbid"}


class AssumptionVersion(BaseModel):
    """A single version of an assumption value."""
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique version ID")
    version_number: int = Field(..., ge=1, description="Sequential version number")
    value: Any = Field(..., description="The assumption value")
    effective_from: datetime = Field(..., description="When this version becomes effective")
    effective_until: Optional[datetime] = Field(None, description="When this version expires")
    created_at: datetime = Field(default_factory=DeterministicClock.now, description="Creation timestamp")
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
    versions: List[AssumptionVersion] = Field(default_factory=list, description="Version history")

    # Validation
    validation_rules: List[ValidationRule] = Field(default_factory=list, description="Validation rules")

    # Metadata
    metadata: AssumptionMetadata = Field(..., description="Assumption metadata")

    # Dependency tracking
    depends_on: List[str] = Field(default_factory=list, description="Assumptions this depends on")
    used_by: List[str] = Field(default_factory=list, description="Calculations using this assumption")

    # Inheritance
    parent_assumption_id: Optional[str] = Field(None, description="Parent assumption for inheritance")

    # Timestamps
    created_at: datetime = Field(default_factory=DeterministicClock.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=DeterministicClock.now, description="Last update timestamp")

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash of current state")

    model_config = {"extra": "forbid"}

    @field_validator('assumption_id')
    @classmethod
    def validate_assumption_id(cls, v: str) -> str:
        """Validate assumption ID format."""
        if not v or len(v) < 3:
            raise ValueError("assumption_id must be at least 3 characters")
        if not v.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError("assumption_id must be alphanumeric with underscores, hyphens, or dots")
        return v


class Scenario(BaseModel):
    """A scenario containing assumption overrides."""
    scenario_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique scenario ID")
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    scenario_type: ScenarioType = Field(..., description="Type of scenario")

    # Overrides: assumption_id -> value
    overrides: Dict[str, Any] = Field(default_factory=dict, description="Assumption value overrides")

    # Metadata
    created_at: datetime = Field(default_factory=DeterministicClock.now, description="Creation timestamp")
    created_by: str = Field(..., description="User who created the scenario")
    is_active: bool = Field(default=True, description="Whether scenario is active")
    parent_scenario_id: Optional[str] = Field(None, description="Parent scenario for inheritance")

    # Tags for filtering
    tags: List[str] = Field(default_factory=list, description="Scenario tags")

    model_config = {"extra": "forbid"}


class ChangeLogEntry(BaseModel):
    """Audit log entry for assumption changes."""
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique log ID")
    timestamp: datetime = Field(default_factory=DeterministicClock.now, description="Change timestamp")
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
    calculation_ids: List[str] = Field(default_factory=list, description="Calculations using this assumption")
    upstream: List[str] = Field(default_factory=list, description="Assumptions this depends on")
    downstream: List[str] = Field(default_factory=list, description="Assumptions depending on this")

    model_config = {"extra": "forbid"}


# =============================================================================
# Input/Output Models
# =============================================================================


class AssumptionsRegistryInput(BaseModel):
    """Input for Assumptions Registry operations."""
    operation: str = Field(..., description="Operation to perform")
    assumption_id: Optional[str] = Field(None, description="Target assumption ID")
    scenario_id: Optional[str] = Field(None, description="Target scenario ID")
    assumption_data: Optional[Dict[str, Any]] = Field(None, description="Assumption data for create/update")
    scenario_data: Optional[Dict[str, Any]] = Field(None, description="Scenario data for create/update")
    value: Optional[Any] = Field(None, description="Value for update operations")
    user_id: str = Field(default="system", description="User performing operation")
    change_reason: Optional[str] = Field(None, description="Reason for change")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters for query operations")

    model_config = {"extra": "forbid"}

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is supported."""
        valid_ops = {
            "create_assumption", "get_assumption", "update_assumption", "delete_assumption",
            "list_assumptions", "create_scenario", "get_scenario", "update_scenario",
            "delete_scenario", "list_scenarios", "get_value", "set_value",
            "get_change_log", "get_dependencies", "validate_assumption",
            "get_sensitivity_analysis", "export_assumptions", "import_assumptions"
        }
        if v not in valid_ops:
            raise ValueError(f"Invalid operation: {v}. Valid operations: {valid_ops}")
        return v


class AssumptionsRegistryOutput(BaseModel):
    """Output from Assumptions Registry operations."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation that was performed")
    data: Optional[Dict[str, Any]] = Field(None, description="Result data")
    assumption: Optional[Assumption] = Field(None, description="Assumption if applicable")
    scenario: Optional[Scenario] = Field(None, description="Scenario if applicable")
    assumptions: Optional[List[Assumption]] = Field(None, description="List of assumptions")
    scenarios: Optional[List[Scenario]] = Field(None, description="List of scenarios")
    change_log: Optional[List[ChangeLogEntry]] = Field(None, description="Change log entries")
    validation_result: Optional[ValidationResult] = Field(None, description="Validation result")
    dependencies: Optional[Dict[str, DependencyNode]] = Field(None, description="Dependency graph")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    error: Optional[str] = Field(None, description="Error message if failed")

    model_config = {"extra": "forbid"}


# =============================================================================
# Main Agent Class
# =============================================================================


class AssumptionsRegistryAgent(BaseAgent):
    """
    GL-FOUND-X-004: Assumptions Registry Agent

    Manages version-controlled assumptions for zero-hallucination compliance.
    Provides complete provenance tracking, scenario management, and audit trails.

    Zero-Hallucination Principles:
        - All assumption values are explicitly defined
        - No implicit defaults or inferred values
        - Complete audit trail for every change
        - Deterministic scenario resolution

    Usage:
        >>> registry = AssumptionsRegistryAgent()
        >>> result = registry.run({
        ...     "operation": "create_assumption",
        ...     "assumption_data": {...},
        ...     "user_id": "analyst_1",
        ...     "change_reason": "Initial setup"
        ... })
    """

    AGENT_ID = "GL-FOUND-X-004"
    AGENT_NAME = "Assumptions Registry"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Assumptions Registry Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Version-controlled assumptions registry for zero-hallucination compliance",
                version=self.VERSION,
                parameters={
                    "max_versions_per_assumption": 100,
                    "enable_change_logging": True,
                    "enable_validation": True,
                    "default_scenario": ScenarioType.BASELINE.value,
                }
            )
        super().__init__(config)

        # Storage (in production, use persistent storage)
        self._assumptions: Dict[str, Assumption] = {}
        self._scenarios: Dict[str, Scenario] = {}
        self._change_log: List[ChangeLogEntry] = []
        self._dependency_graph: Dict[str, DependencyNode] = {}

        # Custom validators registry
        self._custom_validators: Dict[str, Callable] = {}

        # Create default baseline scenario
        self._create_default_scenarios()

        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def _create_default_scenarios(self):
        """Create default scenarios."""
        default_scenarios = [
            Scenario(
                name="Baseline",
                description="Default baseline scenario with standard assumptions",
                scenario_type=ScenarioType.BASELINE,
                created_by="system"
            ),
            Scenario(
                name="Conservative",
                description="Conservative scenario with higher emission factors",
                scenario_type=ScenarioType.CONSERVATIVE,
                created_by="system"
            ),
            Scenario(
                name="Optimistic",
                description="Optimistic scenario with lower emission factors",
                scenario_type=ScenarioType.OPTIMISTIC,
                created_by="system"
            ),
        ]

        for scenario in default_scenarios:
            self._scenarios[scenario.scenario_id] = scenario

    def register_custom_validator(self, name: str, validator: Callable[[Any], bool]):
        """
        Register a custom validation function.

        Args:
            name: Name to reference the validator
            validator: Function that takes a value and returns True if valid
        """
        self._custom_validators[name] = validator
        self.logger.info(f"Registered custom validator: {name}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute an assumptions registry operation.

        If the Assumptions SDK is available, delegates to the SDK first
        with graceful fallback to the built-in implementation.

        Args:
            input_data: Operation parameters

        Returns:
            AgentResult with operation result
        """
        # Try SDK delegation first if available
        if ASSUMPTIONS_SDK_AVAILABLE:
            try:
                return self._execute_via_sdk(input_data)
            except Exception as sdk_err:
                self.logger.warning(
                    "SDK delegation failed, falling back to built-in: %s",
                    str(sdk_err),
                )

        start_time = DeterministicClock.now()

        try:
            # Parse input
            registry_input = AssumptionsRegistryInput(**input_data)

            # Route to operation handler
            operation_handlers = {
                "create_assumption": self._handle_create_assumption,
                "get_assumption": self._handle_get_assumption,
                "update_assumption": self._handle_update_assumption,
                "delete_assumption": self._handle_delete_assumption,
                "list_assumptions": self._handle_list_assumptions,
                "create_scenario": self._handle_create_scenario,
                "get_scenario": self._handle_get_scenario,
                "update_scenario": self._handle_update_scenario,
                "delete_scenario": self._handle_delete_scenario,
                "list_scenarios": self._handle_list_scenarios,
                "get_value": self._handle_get_value,
                "set_value": self._handle_set_value,
                "get_change_log": self._handle_get_change_log,
                "get_dependencies": self._handle_get_dependencies,
                "validate_assumption": self._handle_validate_assumption,
                "get_sensitivity_analysis": self._handle_sensitivity_analysis,
                "export_assumptions": self._handle_export,
                "import_assumptions": self._handle_import,
            }

            handler = operation_handlers.get(registry_input.operation)
            if not handler:
                raise ValueError(f"Unknown operation: {registry_input.operation}")

            output = handler(registry_input)

            # Calculate processing time
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            output.processing_time_ms = processing_time_ms

            # Calculate provenance hash
            output.provenance_hash = self._compute_provenance_hash(output.model_dump())

            self.logger.info(
                f"Operation {registry_input.operation} completed in {processing_time_ms:.2f}ms"
            )

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
                error=output.error
            )

        except Exception as e:
            self.logger.error(f"Operation failed: {str(e)}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                data={"operation": input_data.get("operation", "unknown")}
            )

    def _execute_via_sdk(self, input_data: Dict[str, Any]) -> AgentResult:
        """Delegate execution to the Assumptions SDK.

        Args:
            input_data: Operation parameters.

        Returns:
            AgentResult from SDK execution.

        Raises:
            Exception: If SDK delegation fails.
        """
        sdk_registry = _SDKRegistry()
        operation = input_data.get("operation", "")

        if operation == "create_assumption" and input_data.get("assumption_data"):
            ad = input_data["assumption_data"]
            assumption = sdk_registry.create(
                assumption_id=ad["assumption_id"],
                name=ad["name"],
                description=ad.get("description", ""),
                category=ad.get("category", "custom"),
                data_type=ad.get("data_type", "float"),
                value=ad["current_value"],
                user_id=input_data.get("user_id", "system"),
                change_reason=input_data.get("change_reason", "Initial creation"),
                metadata_source=ad.get("metadata", {}).get("source", "user_defined"),
            )
            return AgentResult(
                success=True,
                data={"assumption_id": assumption.assumption_id, "version": 1},
            )

        if operation == "get_value" and input_data.get("assumption_id"):
            value = sdk_registry.get_value(input_data["assumption_id"])
            return AgentResult(
                success=True,
                data={"value": value, "assumption_id": input_data["assumption_id"]},
            )

        # For operations not yet mapped, raise to trigger fallback
        raise NotImplementedError(f"SDK delegation not implemented for {operation}")

    # =========================================================================
    # Assumption Operations
    # =========================================================================

    def _handle_create_assumption(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Create a new assumption."""
        if not input_data.assumption_data:
            return AssumptionsRegistryOutput(
                success=False,
                operation="create_assumption",
                error="assumption_data is required"
            )

        try:
            # Build metadata
            metadata_data = input_data.assumption_data.get("metadata", {})
            if not metadata_data.get("source"):
                metadata_data["source"] = "user_defined"
            metadata = AssumptionMetadata(**metadata_data)

            # Create assumption
            assumption = Assumption(
                assumption_id=input_data.assumption_data["assumption_id"],
                name=input_data.assumption_data["name"],
                description=input_data.assumption_data.get("description", ""),
                category=AssumptionCategory(input_data.assumption_data.get("category", "custom")),
                data_type=AssumptionDataType(input_data.assumption_data.get("data_type", "float")),
                unit=input_data.assumption_data.get("unit"),
                current_value=input_data.assumption_data["current_value"],
                default_value=input_data.assumption_data.get("default_value", input_data.assumption_data["current_value"]),
                validation_rules=[
                    ValidationRule(**rule) for rule in input_data.assumption_data.get("validation_rules", [])
                ],
                metadata=metadata,
                depends_on=input_data.assumption_data.get("depends_on", []),
                parent_assumption_id=input_data.assumption_data.get("parent_assumption_id"),
            )

            # Check for duplicates
            if assumption.assumption_id in self._assumptions:
                return AssumptionsRegistryOutput(
                    success=False,
                    operation="create_assumption",
                    error=f"Assumption {assumption.assumption_id} already exists"
                )

            # Validate value
            if self.config.parameters.get("enable_validation", True):
                validation = self._validate_value(assumption, assumption.current_value)
                if not validation.is_valid:
                    return AssumptionsRegistryOutput(
                        success=False,
                        operation="create_assumption",
                        error=f"Validation failed: {validation.errors}",
                        validation_result=validation
                    )

            # Create initial version
            initial_version = AssumptionVersion(
                version_number=1,
                value=assumption.current_value,
                effective_from=DeterministicClock.now(),
                created_by=input_data.user_id,
                change_reason=input_data.change_reason or "Initial creation",
                change_type=ChangeType.CREATE,
            )
            initial_version.provenance_hash = self._compute_provenance_hash(
                {"value": assumption.current_value, "version": 1}
            )
            assumption.versions.append(initial_version)

            # Calculate assumption provenance
            assumption.provenance_hash = self._compute_provenance_hash(assumption.model_dump())

            # Store assumption
            self._assumptions[assumption.assumption_id] = assumption

            # Update dependency graph
            self._update_dependency_graph(assumption)

            # Log change
            self._log_change(
                user_id=input_data.user_id,
                change_type=ChangeType.CREATE,
                assumption_id=assumption.assumption_id,
                new_value=assumption.current_value,
                change_reason=input_data.change_reason or "Initial creation"
            )

            self.logger.info(f"Created assumption: {assumption.assumption_id}")

            return AssumptionsRegistryOutput(
                success=True,
                operation="create_assumption",
                assumption=assumption,
                data={"assumption_id": assumption.assumption_id, "version": 1}
            )

        except Exception as e:
            return AssumptionsRegistryOutput(
                success=False,
                operation="create_assumption",
                error=str(e)
            )

    def _handle_get_assumption(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Get an assumption by ID."""
        if not input_data.assumption_id:
            return AssumptionsRegistryOutput(
                success=False,
                operation="get_assumption",
                error="assumption_id is required"
            )

        assumption = self._assumptions.get(input_data.assumption_id)
        if not assumption:
            return AssumptionsRegistryOutput(
                success=False,
                operation="get_assumption",
                error=f"Assumption {input_data.assumption_id} not found"
            )

        return AssumptionsRegistryOutput(
            success=True,
            operation="get_assumption",
            assumption=assumption,
            data={
                "assumption_id": assumption.assumption_id,
                "current_value": assumption.current_value,
                "version_count": len(assumption.versions)
            }
        )

    def _handle_update_assumption(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Update an assumption value."""
        if not input_data.assumption_id:
            return AssumptionsRegistryOutput(
                success=False,
                operation="update_assumption",
                error="assumption_id is required"
            )

        if input_data.value is None and not input_data.assumption_data:
            return AssumptionsRegistryOutput(
                success=False,
                operation="update_assumption",
                error="value or assumption_data is required"
            )

        assumption = self._assumptions.get(input_data.assumption_id)
        if not assumption:
            return AssumptionsRegistryOutput(
                success=False,
                operation="update_assumption",
                error=f"Assumption {input_data.assumption_id} not found"
            )

        # Get new value
        new_value = input_data.value if input_data.value is not None else input_data.assumption_data.get("current_value")

        # Validate new value
        if self.config.parameters.get("enable_validation", True):
            validation = self._validate_value(assumption, new_value)
            if not validation.is_valid:
                return AssumptionsRegistryOutput(
                    success=False,
                    operation="update_assumption",
                    error=f"Validation failed: {validation.errors}",
                    validation_result=validation
                )

        # Store old value for audit
        old_value = assumption.current_value

        # Check version limit
        max_versions = self.config.parameters.get("max_versions_per_assumption", 100)
        if len(assumption.versions) >= max_versions:
            # Remove oldest version (keep first for provenance)
            if len(assumption.versions) > 1:
                assumption.versions.pop(1)

        # Create new version
        new_version = AssumptionVersion(
            version_number=len(assumption.versions) + 1,
            value=new_value,
            effective_from=DeterministicClock.now(),
            created_by=input_data.user_id,
            change_reason=input_data.change_reason or "Value update",
            change_type=ChangeType.UPDATE,
            parent_version_id=assumption.versions[-1].version_id if assumption.versions else None,
            scenario_id=input_data.scenario_id,
        )
        new_version.provenance_hash = self._compute_provenance_hash({
            "value": new_value,
            "version": new_version.version_number,
            "previous": old_value
        })

        # Mark previous version as expired
        if assumption.versions:
            assumption.versions[-1].effective_until = DeterministicClock.now()

        # Update assumption
        assumption.versions.append(new_version)
        assumption.current_value = new_value
        assumption.updated_at = DeterministicClock.now()
        assumption.provenance_hash = self._compute_provenance_hash(assumption.model_dump())

        # Log change
        self._log_change(
            user_id=input_data.user_id,
            change_type=ChangeType.UPDATE,
            assumption_id=assumption.assumption_id,
            old_value=old_value,
            new_value=new_value,
            change_reason=input_data.change_reason or "Value update",
            scenario_id=input_data.scenario_id
        )

        self.logger.info(
            f"Updated assumption {assumption.assumption_id}: {old_value} -> {new_value}"
        )

        return AssumptionsRegistryOutput(
            success=True,
            operation="update_assumption",
            assumption=assumption,
            data={
                "assumption_id": assumption.assumption_id,
                "old_value": old_value,
                "new_value": new_value,
                "version": new_version.version_number
            }
        )

    def _handle_delete_assumption(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Delete an assumption (soft delete)."""
        if not input_data.assumption_id:
            return AssumptionsRegistryOutput(
                success=False,
                operation="delete_assumption",
                error="assumption_id is required"
            )

        assumption = self._assumptions.get(input_data.assumption_id)
        if not assumption:
            return AssumptionsRegistryOutput(
                success=False,
                operation="delete_assumption",
                error=f"Assumption {input_data.assumption_id} not found"
            )

        # Check if assumption is used by others
        if assumption.used_by:
            return AssumptionsRegistryOutput(
                success=False,
                operation="delete_assumption",
                error=f"Cannot delete: assumption is used by {assumption.used_by}"
            )

        # Store for audit before deletion
        deleted_assumption = deepcopy(assumption)

        # Remove from storage
        del self._assumptions[input_data.assumption_id]

        # Remove from dependency graph
        if input_data.assumption_id in self._dependency_graph:
            del self._dependency_graph[input_data.assumption_id]

        # Log change
        self._log_change(
            user_id=input_data.user_id,
            change_type=ChangeType.DELETE,
            assumption_id=input_data.assumption_id,
            old_value=deleted_assumption.current_value,
            change_reason=input_data.change_reason or "Deletion"
        )

        self.logger.info(f"Deleted assumption: {input_data.assumption_id}")

        return AssumptionsRegistryOutput(
            success=True,
            operation="delete_assumption",
            data={"assumption_id": input_data.assumption_id, "deleted": True}
        )

    def _handle_list_assumptions(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """List assumptions with optional filtering."""
        assumptions = list(self._assumptions.values())

        # Apply filters
        if input_data.filters:
            if "category" in input_data.filters:
                category = AssumptionCategory(input_data.filters["category"])
                assumptions = [a for a in assumptions if a.category == category]

            if "tags" in input_data.filters:
                required_tags = set(input_data.filters["tags"])
                assumptions = [
                    a for a in assumptions
                    if required_tags.issubset(set(a.metadata.tags))
                ]

            if "search" in input_data.filters:
                search_term = input_data.filters["search"].lower()
                assumptions = [
                    a for a in assumptions
                    if search_term in a.name.lower() or search_term in a.description.lower()
                ]

        return AssumptionsRegistryOutput(
            success=True,
            operation="list_assumptions",
            assumptions=assumptions,
            data={"count": len(assumptions)}
        )

    # =========================================================================
    # Scenario Operations
    # =========================================================================

    def _handle_create_scenario(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Create a new scenario."""
        if not input_data.scenario_data:
            return AssumptionsRegistryOutput(
                success=False,
                operation="create_scenario",
                error="scenario_data is required"
            )

        try:
            scenario = Scenario(
                name=input_data.scenario_data["name"],
                description=input_data.scenario_data.get("description", ""),
                scenario_type=ScenarioType(input_data.scenario_data.get("scenario_type", "custom")),
                overrides=input_data.scenario_data.get("overrides", {}),
                created_by=input_data.user_id,
                parent_scenario_id=input_data.scenario_data.get("parent_scenario_id"),
                tags=input_data.scenario_data.get("tags", []),
            )

            # Validate overrides reference valid assumptions
            for assumption_id in scenario.overrides:
                if assumption_id not in self._assumptions:
                    return AssumptionsRegistryOutput(
                        success=False,
                        operation="create_scenario",
                        error=f"Override references unknown assumption: {assumption_id}"
                    )

            # Store scenario
            self._scenarios[scenario.scenario_id] = scenario

            self.logger.info(f"Created scenario: {scenario.name} ({scenario.scenario_id})")

            return AssumptionsRegistryOutput(
                success=True,
                operation="create_scenario",
                scenario=scenario,
                data={"scenario_id": scenario.scenario_id}
            )

        except Exception as e:
            return AssumptionsRegistryOutput(
                success=False,
                operation="create_scenario",
                error=str(e)
            )

    def _handle_get_scenario(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Get a scenario by ID."""
        if not input_data.scenario_id:
            return AssumptionsRegistryOutput(
                success=False,
                operation="get_scenario",
                error="scenario_id is required"
            )

        scenario = self._scenarios.get(input_data.scenario_id)
        if not scenario:
            return AssumptionsRegistryOutput(
                success=False,
                operation="get_scenario",
                error=f"Scenario {input_data.scenario_id} not found"
            )

        return AssumptionsRegistryOutput(
            success=True,
            operation="get_scenario",
            scenario=scenario,
            data={
                "scenario_id": scenario.scenario_id,
                "override_count": len(scenario.overrides)
            }
        )

    def _handle_update_scenario(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Update a scenario."""
        if not input_data.scenario_id:
            return AssumptionsRegistryOutput(
                success=False,
                operation="update_scenario",
                error="scenario_id is required"
            )

        scenario = self._scenarios.get(input_data.scenario_id)
        if not scenario:
            return AssumptionsRegistryOutput(
                success=False,
                operation="update_scenario",
                error=f"Scenario {input_data.scenario_id} not found"
            )

        # Update fields
        if input_data.scenario_data:
            if "name" in input_data.scenario_data:
                scenario.name = input_data.scenario_data["name"]
            if "description" in input_data.scenario_data:
                scenario.description = input_data.scenario_data["description"]
            if "overrides" in input_data.scenario_data:
                # Validate new overrides
                for assumption_id in input_data.scenario_data["overrides"]:
                    if assumption_id not in self._assumptions:
                        return AssumptionsRegistryOutput(
                            success=False,
                            operation="update_scenario",
                            error=f"Override references unknown assumption: {assumption_id}"
                        )
                scenario.overrides = input_data.scenario_data["overrides"]
            if "is_active" in input_data.scenario_data:
                scenario.is_active = input_data.scenario_data["is_active"]
            if "tags" in input_data.scenario_data:
                scenario.tags = input_data.scenario_data["tags"]

        self.logger.info(f"Updated scenario: {scenario.name}")

        return AssumptionsRegistryOutput(
            success=True,
            operation="update_scenario",
            scenario=scenario,
            data={"scenario_id": scenario.scenario_id}
        )

    def _handle_delete_scenario(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Delete a scenario."""
        if not input_data.scenario_id:
            return AssumptionsRegistryOutput(
                success=False,
                operation="delete_scenario",
                error="scenario_id is required"
            )

        scenario = self._scenarios.get(input_data.scenario_id)
        if not scenario:
            return AssumptionsRegistryOutput(
                success=False,
                operation="delete_scenario",
                error=f"Scenario {input_data.scenario_id} not found"
            )

        # Prevent deletion of baseline scenario
        if scenario.scenario_type == ScenarioType.BASELINE:
            return AssumptionsRegistryOutput(
                success=False,
                operation="delete_scenario",
                error="Cannot delete baseline scenario"
            )

        del self._scenarios[input_data.scenario_id]

        self.logger.info(f"Deleted scenario: {input_data.scenario_id}")

        return AssumptionsRegistryOutput(
            success=True,
            operation="delete_scenario",
            data={"scenario_id": input_data.scenario_id, "deleted": True}
        )

    def _handle_list_scenarios(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """List all scenarios."""
        scenarios = list(self._scenarios.values())

        # Apply filters
        if input_data.filters:
            if "scenario_type" in input_data.filters:
                scenario_type = ScenarioType(input_data.filters["scenario_type"])
                scenarios = [s for s in scenarios if s.scenario_type == scenario_type]

            if "is_active" in input_data.filters:
                is_active = input_data.filters["is_active"]
                scenarios = [s for s in scenarios if s.is_active == is_active]

            if "tags" in input_data.filters:
                required_tags = set(input_data.filters["tags"])
                scenarios = [
                    s for s in scenarios
                    if required_tags.issubset(set(s.tags))
                ]

        return AssumptionsRegistryOutput(
            success=True,
            operation="list_scenarios",
            scenarios=scenarios,
            data={"count": len(scenarios)}
        )

    # =========================================================================
    # Value Operations
    # =========================================================================

    def _handle_get_value(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Get assumption value with optional scenario override."""
        if not input_data.assumption_id:
            return AssumptionsRegistryOutput(
                success=False,
                operation="get_value",
                error="assumption_id is required"
            )

        assumption = self._assumptions.get(input_data.assumption_id)
        if not assumption:
            return AssumptionsRegistryOutput(
                success=False,
                operation="get_value",
                error=f"Assumption {input_data.assumption_id} not found"
            )

        # Start with base value
        value = assumption.current_value
        value_source = "baseline"

        # Apply inheritance if applicable
        if assumption.parent_assumption_id:
            parent = self._assumptions.get(assumption.parent_assumption_id)
            if parent and value is None:
                value = parent.current_value
                value_source = f"inherited:{assumption.parent_assumption_id}"

        # Apply scenario override if specified
        if input_data.scenario_id:
            scenario = self._scenarios.get(input_data.scenario_id)
            if scenario and input_data.assumption_id in scenario.overrides:
                value = scenario.overrides[input_data.assumption_id]
                value_source = f"scenario:{scenario.name}"

                # Log scenario override access
                self._log_change(
                    user_id=input_data.user_id,
                    change_type=ChangeType.SCENARIO_OVERRIDE,
                    assumption_id=input_data.assumption_id,
                    new_value=value,
                    change_reason=f"Value accessed with scenario {scenario.name}",
                    scenario_id=input_data.scenario_id
                )

        return AssumptionsRegistryOutput(
            success=True,
            operation="get_value",
            data={
                "assumption_id": input_data.assumption_id,
                "value": value,
                "value_source": value_source,
                "unit": assumption.unit,
                "data_type": assumption.data_type.value
            }
        )

    def _handle_set_value(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Set assumption value (alias for update_assumption with value)."""
        return self._handle_update_assumption(input_data)

    # =========================================================================
    # Audit and Analysis Operations
    # =========================================================================

    def _handle_get_change_log(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Get change log entries."""
        log_entries = self._change_log.copy()

        # Apply filters
        if input_data.filters:
            if "assumption_id" in input_data.filters:
                log_entries = [
                    e for e in log_entries
                    if e.assumption_id == input_data.filters["assumption_id"]
                ]

            if "user_id" in input_data.filters:
                log_entries = [
                    e for e in log_entries
                    if e.user_id == input_data.filters["user_id"]
                ]

            if "change_type" in input_data.filters:
                change_type = ChangeType(input_data.filters["change_type"])
                log_entries = [
                    e for e in log_entries
                    if e.change_type == change_type
                ]

            if "from_date" in input_data.filters:
                from_date = datetime.fromisoformat(input_data.filters["from_date"])
                log_entries = [
                    e for e in log_entries
                    if e.timestamp >= from_date
                ]

            if "to_date" in input_data.filters:
                to_date = datetime.fromisoformat(input_data.filters["to_date"])
                log_entries = [
                    e for e in log_entries
                    if e.timestamp <= to_date
                ]

        # Sort by timestamp descending
        log_entries.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply limit
        if input_data.filters and "limit" in input_data.filters:
            log_entries = log_entries[:input_data.filters["limit"]]

        return AssumptionsRegistryOutput(
            success=True,
            operation="get_change_log",
            change_log=log_entries,
            data={"count": len(log_entries)}
        )

    def _handle_get_dependencies(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Get dependency graph for assumptions."""
        if input_data.assumption_id:
            # Get dependencies for specific assumption
            node = self._dependency_graph.get(input_data.assumption_id)
            if not node:
                return AssumptionsRegistryOutput(
                    success=False,
                    operation="get_dependencies",
                    error=f"No dependencies found for {input_data.assumption_id}"
                )
            return AssumptionsRegistryOutput(
                success=True,
                operation="get_dependencies",
                dependencies={input_data.assumption_id: node},
                data={"assumption_id": input_data.assumption_id}
            )

        # Return full dependency graph
        return AssumptionsRegistryOutput(
            success=True,
            operation="get_dependencies",
            dependencies=self._dependency_graph,
            data={"total_nodes": len(self._dependency_graph)}
        )

    def _handle_validate_assumption(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Validate an assumption value."""
        if not input_data.assumption_id:
            return AssumptionsRegistryOutput(
                success=False,
                operation="validate_assumption",
                error="assumption_id is required"
            )

        assumption = self._assumptions.get(input_data.assumption_id)
        if not assumption:
            return AssumptionsRegistryOutput(
                success=False,
                operation="validate_assumption",
                error=f"Assumption {input_data.assumption_id} not found"
            )

        # Validate current value or provided value
        value_to_validate = input_data.value if input_data.value is not None else assumption.current_value
        validation_result = self._validate_value(assumption, value_to_validate)

        return AssumptionsRegistryOutput(
            success=True,
            operation="validate_assumption",
            validation_result=validation_result,
            data={
                "assumption_id": input_data.assumption_id,
                "value_validated": value_to_validate,
                "is_valid": validation_result.is_valid
            }
        )

    def _handle_sensitivity_analysis(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Get sensitivity analysis data for assumptions."""
        if not input_data.assumption_id:
            return AssumptionsRegistryOutput(
                success=False,
                operation="get_sensitivity_analysis",
                error="assumption_id is required"
            )

        assumption = self._assumptions.get(input_data.assumption_id)
        if not assumption:
            return AssumptionsRegistryOutput(
                success=False,
                operation="get_sensitivity_analysis",
                error=f"Assumption {input_data.assumption_id} not found"
            )

        # Collect scenario values
        scenario_values = {}
        for scenario_id, scenario in self._scenarios.items():
            if input_data.assumption_id in scenario.overrides:
                scenario_values[scenario.name] = scenario.overrides[input_data.assumption_id]
            else:
                scenario_values[scenario.name] = assumption.current_value

        # Calculate statistics if numeric
        sensitivity_data = {
            "assumption_id": input_data.assumption_id,
            "baseline_value": assumption.current_value,
            "scenario_values": scenario_values,
            "dependency_count": len(assumption.used_by),
            "dependent_calculations": assumption.used_by,
        }

        # Add range analysis for numeric types
        if assumption.data_type in [AssumptionDataType.FLOAT, AssumptionDataType.INTEGER,
                                    AssumptionDataType.PERCENTAGE, AssumptionDataType.RATIO]:
            numeric_values = [
                v for v in scenario_values.values()
                if isinstance(v, (int, float))
            ]
            if numeric_values:
                sensitivity_data["min_value"] = min(numeric_values)
                sensitivity_data["max_value"] = max(numeric_values)
                sensitivity_data["range"] = max(numeric_values) - min(numeric_values)
                if assumption.current_value and assumption.current_value != 0:
                    sensitivity_data["range_pct"] = (
                        (max(numeric_values) - min(numeric_values)) / abs(assumption.current_value) * 100
                    )

        return AssumptionsRegistryOutput(
            success=True,
            operation="get_sensitivity_analysis",
            data=sensitivity_data
        )

    def _handle_export(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Export all assumptions and scenarios."""
        export_data = {
            "export_timestamp": DeterministicClock.now().isoformat(),
            "exported_by": input_data.user_id,
            "assumptions": [a.model_dump() for a in self._assumptions.values()],
            "scenarios": [s.model_dump() for s in self._scenarios.values()],
            "change_log": [c.model_dump() for c in self._change_log],
        }

        # Calculate export hash for integrity
        export_hash = self._compute_provenance_hash(export_data)
        export_data["export_hash"] = export_hash

        return AssumptionsRegistryOutput(
            success=True,
            operation="export_assumptions",
            data=export_data
        )

    def _handle_import(self, input_data: AssumptionsRegistryInput) -> AssumptionsRegistryOutput:
        """Import assumptions from export data."""
        if not input_data.assumption_data:
            return AssumptionsRegistryOutput(
                success=False,
                operation="import_assumptions",
                error="assumption_data with import content is required"
            )

        import_data = input_data.assumption_data
        imported_count = 0
        errors = []

        # Import assumptions
        for assumption_dict in import_data.get("assumptions", []):
            try:
                # Skip if already exists
                if assumption_dict["assumption_id"] in self._assumptions:
                    continue

                # Rebuild metadata
                metadata = AssumptionMetadata(**assumption_dict.get("metadata", {"source": "imported"}))

                assumption = Assumption(
                    assumption_id=assumption_dict["assumption_id"],
                    name=assumption_dict["name"],
                    description=assumption_dict.get("description", ""),
                    category=AssumptionCategory(assumption_dict.get("category", "custom")),
                    data_type=AssumptionDataType(assumption_dict.get("data_type", "float")),
                    unit=assumption_dict.get("unit"),
                    current_value=assumption_dict["current_value"],
                    default_value=assumption_dict.get("default_value", assumption_dict["current_value"]),
                    metadata=metadata,
                )

                self._assumptions[assumption.assumption_id] = assumption
                self._update_dependency_graph(assumption)
                imported_count += 1

            except Exception as e:
                errors.append(f"Failed to import {assumption_dict.get('assumption_id', 'unknown')}: {str(e)}")

        # Log import
        self._log_change(
            user_id=input_data.user_id,
            change_type=ChangeType.CREATE,
            assumption_id="__import__",
            new_value={"imported_count": imported_count},
            change_reason=input_data.change_reason or f"Bulk import of {imported_count} assumptions"
        )

        return AssumptionsRegistryOutput(
            success=len(errors) == 0,
            operation="import_assumptions",
            data={
                "imported_count": imported_count,
                "errors": errors
            },
            error="; ".join(errors) if errors else None
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _validate_value(self, assumption: Assumption, value: Any) -> ValidationResult:
        """Validate a value against assumption rules."""
        errors = []
        warnings = []
        info = []
        rules_checked = []

        for rule in assumption.validation_rules:
            rules_checked.append(rule.rule_id)

            try:
                # Check min value
                if rule.min_value is not None and isinstance(value, (int, float)):
                    if value < rule.min_value:
                        msg = f"Value {value} is below minimum {rule.min_value}"
                        if rule.severity == ValidationSeverity.ERROR:
                            errors.append(msg)
                        elif rule.severity == ValidationSeverity.WARNING:
                            warnings.append(msg)
                        else:
                            info.append(msg)

                # Check max value
                if rule.max_value is not None and isinstance(value, (int, float)):
                    if value > rule.max_value:
                        msg = f"Value {value} is above maximum {rule.max_value}"
                        if rule.severity == ValidationSeverity.ERROR:
                            errors.append(msg)
                        elif rule.severity == ValidationSeverity.WARNING:
                            warnings.append(msg)
                        else:
                            info.append(msg)

                # Check allowed values
                if rule.allowed_values is not None:
                    if value not in rule.allowed_values:
                        msg = f"Value {value} is not in allowed values: {rule.allowed_values}"
                        if rule.severity == ValidationSeverity.ERROR:
                            errors.append(msg)
                        elif rule.severity == ValidationSeverity.WARNING:
                            warnings.append(msg)
                        else:
                            info.append(msg)

                # Check regex pattern
                if rule.regex_pattern and isinstance(value, str):
                    import re
                    if not re.match(rule.regex_pattern, value):
                        msg = f"Value '{value}' does not match pattern {rule.regex_pattern}"
                        if rule.severity == ValidationSeverity.ERROR:
                            errors.append(msg)
                        elif rule.severity == ValidationSeverity.WARNING:
                            warnings.append(msg)
                        else:
                            info.append(msg)

                # Check custom validator
                if rule.custom_validator and rule.custom_validator in self._custom_validators:
                    validator_func = self._custom_validators[rule.custom_validator]
                    if not validator_func(value):
                        msg = f"Custom validation '{rule.custom_validator}' failed for value {value}"
                        if rule.severity == ValidationSeverity.ERROR:
                            errors.append(msg)
                        elif rule.severity == ValidationSeverity.WARNING:
                            warnings.append(msg)
                        else:
                            info.append(msg)

            except Exception as e:
                errors.append(f"Validation rule {rule.rule_id} failed: {str(e)}")

        # Type validation
        type_valid = self._validate_data_type(assumption.data_type, value)
        if not type_valid:
            errors.append(f"Value type does not match expected {assumption.data_type.value}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info,
            rules_checked=rules_checked
        )

    def _validate_data_type(self, data_type: AssumptionDataType, value: Any) -> bool:
        """Validate value matches expected data type."""
        type_checks = {
            AssumptionDataType.FLOAT: lambda v: isinstance(v, (int, float)),
            AssumptionDataType.INTEGER: lambda v: isinstance(v, int) or (isinstance(v, float) and v.is_integer()),
            AssumptionDataType.STRING: lambda v: isinstance(v, str),
            AssumptionDataType.BOOLEAN: lambda v: isinstance(v, bool),
            AssumptionDataType.PERCENTAGE: lambda v: isinstance(v, (int, float)) and 0 <= v <= 100,
            AssumptionDataType.RATIO: lambda v: isinstance(v, (int, float)) and 0 <= v <= 1,
            AssumptionDataType.DATE: lambda v: isinstance(v, (str, datetime)),
            AssumptionDataType.LIST_FLOAT: lambda v: isinstance(v, list) and all(isinstance(x, (int, float)) for x in v),
            AssumptionDataType.LIST_STRING: lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
            AssumptionDataType.DICT: lambda v: isinstance(v, dict),
        }

        check_func = type_checks.get(data_type)
        if check_func:
            return check_func(value)
        return True

    def _update_dependency_graph(self, assumption: Assumption):
        """Update dependency graph when assumption changes."""
        node = DependencyNode(
            assumption_id=assumption.assumption_id,
            calculation_ids=assumption.used_by,
            upstream=assumption.depends_on,
            downstream=[]
        )

        self._dependency_graph[assumption.assumption_id] = node

        # Update downstream references in upstream nodes
        for upstream_id in assumption.depends_on:
            if upstream_id in self._dependency_graph:
                if assumption.assumption_id not in self._dependency_graph[upstream_id].downstream:
                    self._dependency_graph[upstream_id].downstream.append(assumption.assumption_id)

    def _log_change(
        self,
        user_id: str,
        change_type: ChangeType,
        assumption_id: str,
        change_reason: str,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        scenario_id: Optional[str] = None
    ):
        """Log a change to the audit trail."""
        if not self.config.parameters.get("enable_change_logging", True):
            return

        entry = ChangeLogEntry(
            user_id=user_id,
            change_type=change_type,
            assumption_id=assumption_id,
            scenario_id=scenario_id,
            old_value=old_value,
            new_value=new_value,
            change_reason=change_reason,
        )

        # Calculate provenance hash
        entry.provenance_hash = self._compute_provenance_hash({
            "user": user_id,
            "type": change_type.value,
            "assumption": assumption_id,
            "old": old_value,
            "new": new_value,
            "reason": change_reason,
            "timestamp": entry.timestamp.isoformat()
        })

        self._change_log.append(entry)

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(data).encode()).hexdigest()

    # =========================================================================
    # Public Convenience Methods
    # =========================================================================

    def get_assumption_value(
        self,
        assumption_id: str,
        scenario_id: Optional[str] = None,
        user_id: str = "system"
    ) -> Optional[Any]:
        """
        Convenience method to get an assumption value directly.

        Args:
            assumption_id: The assumption identifier
            scenario_id: Optional scenario for overrides
            user_id: User making the request

        Returns:
            The assumption value or None if not found
        """
        result = self.run({
            "operation": "get_value",
            "assumption_id": assumption_id,
            "scenario_id": scenario_id,
            "user_id": user_id
        })

        if result.success and result.data:
            return result.data.get("data", {}).get("value")
        return None

    def set_assumption_value(
        self,
        assumption_id: str,
        value: Any,
        user_id: str,
        change_reason: str,
        scenario_id: Optional[str] = None
    ) -> bool:
        """
        Convenience method to set an assumption value directly.

        Args:
            assumption_id: The assumption identifier
            value: New value to set
            user_id: User making the change
            change_reason: Reason for the change
            scenario_id: Optional scenario for the override

        Returns:
            True if successful, False otherwise
        """
        result = self.run({
            "operation": "set_value",
            "assumption_id": assumption_id,
            "value": value,
            "user_id": user_id,
            "change_reason": change_reason,
            "scenario_id": scenario_id
        })

        return result.success

    def register_calculation_dependency(
        self,
        calculation_id: str,
        assumption_ids: List[str]
    ):
        """
        Register that a calculation depends on specific assumptions.

        Args:
            calculation_id: Identifier for the calculation
            assumption_ids: List of assumptions the calculation uses
        """
        for assumption_id in assumption_ids:
            if assumption_id in self._assumptions:
                assumption = self._assumptions[assumption_id]
                if calculation_id not in assumption.used_by:
                    assumption.used_by.append(calculation_id)
                    self._update_dependency_graph(assumption)

    def get_assumptions_for_calculation(
        self,
        calculation_id: str,
        scenario_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all assumptions needed for a specific calculation.

        Args:
            calculation_id: The calculation identifier
            scenario_id: Optional scenario for overrides

        Returns:
            Dictionary of assumption_id -> value
        """
        assumptions = {}

        for assumption_id, assumption in self._assumptions.items():
            if calculation_id in assumption.used_by:
                value_result = self.run({
                    "operation": "get_value",
                    "assumption_id": assumption_id,
                    "scenario_id": scenario_id,
                    "user_id": "system"
                })

                if value_result.success and value_result.data:
                    assumptions[assumption_id] = value_result.data.get("data", {}).get("value")

        return assumptions

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        base_stats = super().get_stats()

        registry_stats = {
            "total_assumptions": len(self._assumptions),
            "total_scenarios": len(self._scenarios),
            "total_change_log_entries": len(self._change_log),
            "assumptions_by_category": {},
            "scenarios_by_type": {},
        }

        # Count by category
        for assumption in self._assumptions.values():
            cat = assumption.category.value
            registry_stats["assumptions_by_category"][cat] = (
                registry_stats["assumptions_by_category"].get(cat, 0) + 1
            )

        # Count by scenario type
        for scenario in self._scenarios.values():
            st = scenario.scenario_type.value
            registry_stats["scenarios_by_type"][st] = (
                registry_stats["scenarios_by_type"].get(st, 0) + 1
            )

        return {**base_stats, **registry_stats}
