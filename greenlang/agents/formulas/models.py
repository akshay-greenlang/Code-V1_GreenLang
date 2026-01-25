"""
Formula Versioning Data Models

This module defines Pydantic models for formula versioning, validation,
and execution tracking.

All models include complete type safety and validation to prevent
data integrity issues.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, date
from enum import Enum
import json


class VersionStatus(str, Enum):
    """Formula version lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class CalculationType(str, Enum):
    """Supported calculation types."""
    SUM = "sum"
    SUBTRACTION = "subtraction"
    MULTIPLICATION = "multiplication"
    DIVISION = "division"
    PERCENTAGE = "percentage"
    DATABASE_LOOKUP = "database_lookup"
    DATABASE_LOOKUP_AND_MULTIPLY = "database_lookup_and_multiply"
    CONDITIONAL_SUM = "conditional_sum"
    GROUP_BY_COUNT = "group_by_count"
    RATIO_SCALED = "ratio_scaled"
    CUSTOM_EXPRESSION = "custom_expression"


class ExecutionStatus(str, Enum):
    """Formula execution result status."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"


class DependencyType(str, Enum):
    """Formula dependency type."""
    REQUIRED = "required"
    OPTIONAL = "optional"


class FormulaCategory(str, Enum):
    """Formula categorization."""
    EMISSIONS = "emissions"
    ENERGY = "energy"
    WATER = "water"
    WASTE = "waste"
    WORKFORCE = "workforce"
    EFFICIENCY = "efficiency"
    COST = "cost"
    COMPLIANCE = "compliance"
    UTILITY = "utility"


class ValidationRules(BaseModel):
    """Validation constraints for formula inputs and outputs."""
    min_value: Optional[float] = Field(None, description="Minimum allowed value")
    max_value: Optional[float] = Field(None, description="Maximum allowed value")
    required: bool = Field(True, description="Whether value is required")
    allow_zero: bool = Field(True, description="Whether zero is valid")
    allow_negative: bool = Field(False, description="Whether negative values allowed")
    precision: Optional[int] = Field(None, ge=0, le=10, description="Decimal precision")

    @field_validator('max_value')
    @classmethod
    def validate_max_greater_than_min(cls, v, info):
        """Ensure max_value > min_value if both specified."""
        if v is not None and info.data.get('min_value') is not None:
            if v <= info.data['min_value']:
                raise ValueError("max_value must be greater than min_value")
        return v


class FormulaMetadata(BaseModel):
    """Core formula metadata (table: formulas)."""
    id: Optional[int] = Field(None, description="Database ID")
    formula_code: str = Field(..., min_length=1, max_length=50, description="Unique formula code")
    formula_name: str = Field(..., min_length=1, max_length=200, description="Human-readable name")
    category: FormulaCategory = Field(..., description="Formula category")
    description: Optional[str] = Field(None, description="Detailed description")
    standard_reference: Optional[str] = Field(None, max_length=200, description="Regulatory standard reference")
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="system", max_length=100)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class FormulaVersion(BaseModel):
    """
    Formula version with complete calculation specification.

    This model represents a specific version of a formula, including
    the expression, inputs, validation rules, and metadata.
    """
    id: Optional[int] = Field(None, description="Database ID")
    formula_id: int = Field(..., description="Foreign key to formulas table")
    version_number: int = Field(..., ge=1, description="Version number (1, 2, 3...)")

    # Formula definition
    formula_expression: str = Field(..., description="Mathematical expression or calculation logic")
    calculation_type: CalculationType = Field(..., description="Type of calculation")
    required_inputs: List[str] = Field(..., min_items=0, description="Required input field names")
    optional_inputs: List[str] = Field(default_factory=list, description="Optional input field names")
    output_unit: Optional[str] = Field(None, max_length=50, description="Unit of output")
    output_type: str = Field(default="numeric", description="Output data type")

    # Validation and safety
    validation_rules: Optional[ValidationRules] = Field(None, description="Input/output validation rules")
    deterministic: bool = Field(True, description="Is calculation deterministic?")
    zero_hallucination: bool = Field(True, description="Is zero-hallucination safe?")

    # Version lifecycle
    version_status: VersionStatus = Field(default=VersionStatus.DRAFT)
    effective_from: Optional[date] = Field(None, description="Version becomes active on this date")
    effective_to: Optional[date] = Field(None, description="Version expires on this date")

    # Documentation
    change_notes: Optional[str] = Field(None, description="What changed in this version")
    example_calculation: Optional[str] = Field(None, description="Example showing inputs -> output")

    # A/B testing
    ab_test_group: Optional[str] = Field(None, max_length=50, description="A/B test group assignment")
    ab_traffic_weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Traffic weight for A/B testing")

    # Performance tracking
    avg_execution_time_ms: Optional[float] = Field(None, ge=0, description="Average execution time")
    execution_count: int = Field(default=0, ge=0, description="Number of times executed")

    # Audit
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="system", max_length=100)

    class Config:
        use_enum_values = True

    @field_validator('required_inputs', 'optional_inputs')
    @classmethod
    def validate_input_names(cls, v):
        """Validate input names are valid Python identifiers."""
        for name in v:
            if not name.replace('_', '').replace('[', '').replace(']', '').isalnum():
                raise ValueError(f"Invalid input name: {name}")
        return v

    @model_validator(mode='after')
    def validate_effective_dates(self):
        """Ensure effective_to > effective_from."""
        if self.effective_from and self.effective_to:
            if self.effective_to <= self.effective_from:
                raise ValueError("effective_to must be after effective_from")
        return self

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for database storage."""
        data = self.dict()

        # Convert required_inputs and optional_inputs to JSON strings
        data['required_inputs'] = json.dumps(data['required_inputs'])
        data['optional_inputs'] = json.dumps(data['optional_inputs'])

        # Convert validation_rules to JSON string
        if data['validation_rules']:
            data['validation_rules'] = json.dumps(data['validation_rules'])

        # Convert dates to ISO format strings
        if data['effective_from']:
            data['effective_from'] = data['effective_from'].isoformat()
        if data['effective_to']:
            data['effective_to'] = data['effective_to'].isoformat()

        return data


class FormulaDependency(BaseModel):
    """Formula dependency relationship."""
    id: Optional[int] = Field(None, description="Database ID")
    formula_version_id: int = Field(..., description="Formula version that has dependency")
    depends_on_formula_code: str = Field(..., description="Formula code this depends on")
    depends_on_version_number: Optional[int] = Field(None, description="Specific version (null=latest)")
    dependency_type: DependencyType = Field(default=DependencyType.REQUIRED)

    class Config:
        use_enum_values = True


class FormulaExecutionResult(BaseModel):
    """
    Result of formula execution with complete provenance.

    This model tracks every formula execution for audit trails,
    performance monitoring, and debugging.
    """
    id: Optional[int] = Field(None, description="Database ID")
    formula_version_id: int = Field(..., description="Version used for calculation")

    # Execution context
    execution_timestamp: datetime = Field(default_factory=datetime.now)
    agent_name: Optional[str] = Field(None, max_length=100)
    calculation_id: Optional[str] = Field(None, max_length=64, description="Links to broader calculation")
    user_id: Optional[str] = Field(None, max_length=100)

    # Input/Output
    input_data: Dict[str, Any] = Field(..., description="Input values used")
    output_value: Any = Field(..., description="Calculated result")
    input_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 of input")
    output_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 of output")

    # Performance
    execution_time_ms: float = Field(..., ge=0, description="Execution duration")

    # Status
    execution_status: ExecutionStatus = Field(default=ExecutionStatus.SUCCESS)
    error_message: Optional[str] = Field(None, description="Error details if failed")

    class Config:
        use_enum_values = True

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for database storage."""
        data = self.dict()

        # Convert input_data and output_value to JSON strings
        data['input_data'] = json.dumps(data['input_data'])
        data['output_value'] = json.dumps(data['output_value'])

        return data


class ABTest(BaseModel):
    """A/B test configuration and results."""
    id: Optional[int] = Field(None, description="Database ID")
    test_name: str = Field(..., min_length=1, max_length=100, description="Unique test name")
    formula_code: str = Field(..., description="Formula being tested")

    # Test configuration
    control_version_id: int = Field(..., description="Control version ID")
    variant_version_id: int = Field(..., description="Variant version ID")
    traffic_split: float = Field(default=0.5, ge=0.0, le=1.0, description="% to variant")

    # Test lifecycle
    test_status: Literal["draft", "running", "completed", "cancelled"] = Field(default="draft")
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Results
    control_executions: int = Field(default=0, ge=0)
    variant_executions: int = Field(default=0, ge=0)
    control_avg_time_ms: Optional[float] = Field(None, ge=0)
    variant_avg_time_ms: Optional[float] = Field(None, ge=0)

    # Decision
    winning_version_id: Optional[int] = None
    decision_notes: Optional[str] = None

    # Audit
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="system", max_length=100)

    @field_validator('variant_version_id')
    @classmethod
    def validate_different_versions(cls, v, info):
        """Ensure control and variant are different versions."""
        if v == info.data.get('control_version_id'):
            raise ValueError("Variant version must be different from control version")
        return v


class FormulaMigration(BaseModel):
    """Track formula migration from external sources."""
    id: Optional[int] = Field(None, description="Database ID")
    migration_name: str = Field(..., max_length=100)
    source_type: Literal["yaml", "python", "manual"] = Field(...)
    source_file: Optional[str] = Field(None, max_length=500)
    formulas_migrated: int = Field(default=0, ge=0)
    migration_status: Literal["pending", "in_progress", "completed", "failed"] = Field(default="pending")
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="system", max_length=100)


class FormulaComparisonResult(BaseModel):
    """Result of comparing two formula versions."""
    formula_code: str
    version_a: int
    version_b: int

    # Differences
    expression_changed: bool
    inputs_changed: bool
    output_unit_changed: bool
    validation_rules_changed: bool

    # Details
    expression_diff: Optional[str] = None
    added_inputs: List[str] = Field(default_factory=list)
    removed_inputs: List[str] = Field(default_factory=list)

    # Performance comparison
    avg_time_diff_ms: Optional[float] = None
    avg_time_diff_pct: Optional[float] = None
