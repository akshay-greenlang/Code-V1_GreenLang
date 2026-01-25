# -*- coding: utf-8 -*-
"""
Base Template Classes for GreenLang Agent Factory
=================================================

Provides base classes and data models for agent templates.
These classes define the structure and contracts for all agent templates.

Features:
- Type-safe schema definitions with Pydantic
- Formula and standard reference tracking
- Safety constraint enforcement
- Zero-hallucination configuration
- Provenance tracking

Copyright (c) 2024 GreenLang. All rights reserved.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from pydantic import BaseModel, Field, validator


# =============================================================================
# Enums
# =============================================================================

class FieldType(str, Enum):
    """Data types for schema fields."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LIST = "list"
    DICT = "dict"
    OBJECT = "object"
    ENUM = "enum"
    FILE = "file"
    IMAGE = "image"


class ProcessingStepType(str, Enum):
    """Types of processing steps."""
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    CALCULATION = "calculation"
    AGGREGATION = "aggregation"
    LOOKUP = "lookup"
    CLASSIFICATION = "classification"
    DECISION = "decision"
    OUTPUT = "output"


class ConstraintSeverity(str, Enum):
    """Severity levels for safety constraints."""
    CRITICAL = "critical"  # Must halt processing
    ERROR = "error"  # Must flag and may halt
    WARNING = "warning"  # Flag but continue
    INFO = "info"  # Log only


class CalculationMode(str, Enum):
    """Modes for zero-hallucination calculations."""
    DETERMINISTIC = "deterministic"  # Only deterministic calculations
    FORMULA_ONLY = "formula_only"  # Only predefined formulas
    LOOKUP_ONLY = "lookup_only"  # Only database/table lookups
    HYBRID = "hybrid"  # Combination with validation


# =============================================================================
# Schema Models
# =============================================================================

class SchemaField(BaseModel):
    """Definition of a schema field."""
    name: str = Field(..., description="Field name")
    field_type: FieldType = Field(..., alias="type", description="Data type")
    description: str = Field(..., description="Field description")
    required: bool = Field(default=True, description="Whether field is required")
    default: Optional[Any] = Field(None, description="Default value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    allowed_values: Optional[List[Any]] = Field(None, description="Enumerated values")
    pattern: Optional[str] = Field(None, description="Regex pattern for strings")
    nested_schema: Optional[str] = Field(None, description="Reference to nested schema")

    class Config:
        populate_by_name = True


class AgentInputSchema(BaseModel):
    """
    Input schema definition for an agent.

    Defines all input fields with types, validation rules, and metadata.
    """
    name: str = Field(..., description="Schema name")
    description: str = Field(..., description="Schema description")
    version: str = Field(default="1.0.0", description="Schema version")
    fields: Dict[str, SchemaField] = Field(default_factory=dict, description="Field definitions")
    required_fields: List[str] = Field(default_factory=list, description="Required field names")
    validation_rules: List[str] = Field(default_factory=list, description="Cross-field validation rules")

    def get_required_fields(self) -> List[SchemaField]:
        """Get list of required fields."""
        return [f for f in self.fields.values() if f.required]

    def get_optional_fields(self) -> List[SchemaField]:
        """Get list of optional fields."""
        return [f for f in self.fields.values() if not f.required]

    def to_pydantic_model(self) -> Type[BaseModel]:
        """Generate a Pydantic model class from this schema."""
        field_definitions = {}
        for name, field in self.fields.items():
            python_type = self._get_python_type(field.field_type)
            if field.required:
                field_definitions[name] = (python_type, Field(..., description=field.description))
            else:
                field_definitions[name] = (Optional[python_type], Field(field.default, description=field.description))

        # Create dynamic model
        from pydantic import create_model
        return create_model(f"{self.name}Model", **field_definitions)

    @staticmethod
    def _get_python_type(field_type: FieldType) -> type:
        """Map FieldType to Python type."""
        type_map = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: float,
            FieldType.BOOLEAN: bool,
            FieldType.DATE: str,
            FieldType.DATETIME: datetime,
            FieldType.LIST: list,
            FieldType.DICT: dict,
            FieldType.OBJECT: dict,
            FieldType.ENUM: str,
            FieldType.FILE: str,
            FieldType.IMAGE: bytes,
        }
        return type_map.get(field_type, Any)


class AgentOutputSchema(BaseModel):
    """
    Output schema definition for an agent.

    Defines all output fields with types and provenance requirements.
    """
    name: str = Field(..., description="Schema name")
    description: str = Field(..., description="Schema description")
    version: str = Field(default="1.0.0", description="Schema version")
    fields: Dict[str, SchemaField] = Field(default_factory=dict, description="Field definitions")
    provenance_fields: List[str] = Field(
        default_factory=lambda: ["provenance_hash", "calculation_timestamp", "source_references"],
        description="Fields for audit trail"
    )


# =============================================================================
# Formula and Standard References
# =============================================================================

@dataclass
class FormulaReference:
    """
    Reference to a calculation formula with standards citation.

    Attributes:
        formula_id: Unique identifier for the formula
        name: Human-readable formula name
        equation: Mathematical equation (LaTeX or plain text)
        variables: Variable definitions with types and units
        source_standard: Standard that defines this formula
        source_section: Section/clause within the standard
        application: When to use this formula
        limitations: Known limitations or assumptions
        implementation: Python function reference
    """
    formula_id: str
    name: str
    equation: str
    variables: Dict[str, Dict[str, str]]  # {name: {description, unit, type}}
    source_standard: str
    source_section: str
    application: str
    limitations: List[str] = field(default_factory=list)
    implementation: Optional[str] = None  # Module path to function
    uncertainty: Optional[float] = None  # Typical uncertainty %

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "id": self.formula_id,
            "name": self.name,
            "equation": self.equation,
            "variables": self.variables,
            "source": {
                "standard": self.source_standard,
                "section": self.source_section,
            },
            "application": self.application,
            "limitations": self.limitations,
            "implementation": self.implementation,
            "uncertainty": self.uncertainty,
        }


@dataclass
class StandardReference:
    """
    Reference to an industry standard.

    Attributes:
        standard_code: Standard identifier (e.g., "ASME PTC 4")
        title: Full standard title
        edition: Edition/year
        sections: Applicable sections
        requirements: Key requirements from this standard
        compliance_level: Required, recommended, or optional
    """
    standard_code: str
    title: str
    edition: str
    sections: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    compliance_level: str = "required"  # required, recommended, optional

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "code": self.standard_code,
            "title": self.title,
            "edition": self.edition,
            "sections": self.sections,
            "requirements": self.requirements,
            "compliance_level": self.compliance_level,
        }


# =============================================================================
# Safety Constraints
# =============================================================================

@dataclass
class SafetyConstraint:
    """
    Safety constraint definition.

    Enforces safety rules during agent processing with configurable
    severity and actions.

    Attributes:
        constraint_id: Unique identifier
        name: Human-readable name
        description: Detailed description
        condition: Condition expression (Python-like)
        action: Action to take when violated
        severity: Severity level
        standard_reference: Standard that mandates this constraint
        parameters: Constraint parameters (thresholds, limits)
    """
    constraint_id: str
    name: str
    description: str
    condition: str  # e.g., "excess_air >= 0 and excess_air <= 50"
    action: str  # e.g., "reject", "warn", "flag", "escalate"
    severity: ConstraintSeverity
    standard_reference: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    message_template: str = "Constraint {constraint_id} violated: {condition}"

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate the constraint condition.

        Args:
            context: Dictionary of values for evaluation

        Returns:
            True if constraint is satisfied, False otherwise
        """
        try:
            # Safe evaluation with limited namespace
            safe_context = {
                **context,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
            }
            return bool(eval(self.condition, {"__builtins__": {}}, safe_context))
        except Exception:
            return False

    def format_violation_message(self, context: Dict[str, Any]) -> str:
        """Format violation message with context."""
        return self.message_template.format(
            constraint_id=self.constraint_id,
            condition=self.condition,
            **context
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "id": self.constraint_id,
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "action": self.action,
            "severity": self.severity.value,
            "standard_reference": self.standard_reference,
            "parameters": self.parameters,
        }


# =============================================================================
# Zero-Hallucination Configuration
# =============================================================================

@dataclass
class ZeroHallucinationConfig:
    """
    Configuration for zero-hallucination guarantees.

    Ensures all numeric calculations are deterministic and auditable.

    Attributes:
        enabled: Whether zero-hallucination mode is active
        calculation_mode: Mode for calculations
        allowed_operations: List of allowed calculation types
        forbidden_operations: List of forbidden operations
        require_formula_citation: Require formula source for all calculations
        require_provenance: Require SHA-256 provenance hash
        llm_allowed_for: Operations where LLM is allowed (non-numeric)
    """
    enabled: bool = True
    calculation_mode: CalculationMode = CalculationMode.DETERMINISTIC
    allowed_operations: List[str] = field(default_factory=lambda: [
        "arithmetic",
        "formula_evaluation",
        "table_lookup",
        "interpolation",
        "unit_conversion",
        "aggregation",
    ])
    forbidden_operations: List[str] = field(default_factory=lambda: [
        "llm_numeric_calculation",
        "ml_prediction_for_compliance",
        "unvalidated_external_api",
    ])
    require_formula_citation: bool = True
    require_provenance: bool = True
    llm_allowed_for: List[str] = field(default_factory=lambda: [
        "classification",
        "entity_resolution",
        "narrative_generation",
        "materiality_assessment",
    ])

    def validate_operation(self, operation: str) -> bool:
        """Check if an operation is allowed."""
        if not self.enabled:
            return True
        if operation in self.forbidden_operations:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "enabled": self.enabled,
            "calculation_mode": self.calculation_mode.value,
            "allowed_operations": self.allowed_operations,
            "forbidden_operations": self.forbidden_operations,
            "require_formula_citation": self.require_formula_citation,
            "require_provenance": self.require_provenance,
            "llm_allowed_for": self.llm_allowed_for,
        }


# =============================================================================
# Processing Step Definition
# =============================================================================

@dataclass
class ProcessingStep:
    """
    Definition of a processing step in an agent pipeline.

    Attributes:
        step_id: Unique step identifier
        name: Human-readable name
        step_type: Type of processing
        description: What this step does
        inputs: Input field names
        outputs: Output field names
        formula_refs: Formulas used in this step
        constraints: Safety constraints applied
        implementation: Python function reference
    """
    step_id: str
    name: str
    step_type: ProcessingStepType
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    formula_refs: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    implementation: Optional[str] = None
    error_handling: str = "propagate"  # propagate, skip, default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "id": self.step_id,
            "name": self.name,
            "type": self.step_type.value,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "formula_refs": self.formula_refs,
            "constraints": self.constraints,
            "implementation": self.implementation,
            "error_handling": self.error_handling,
        }


# =============================================================================
# Base Agent Template
# =============================================================================

class BaseAgentTemplate(ABC):
    """
    Abstract base class for all agent templates.

    Provides common structure and methods for template definition.
    Subclasses must implement category-specific requirements.
    """

    def __init__(
        self,
        template_id: str,
        name: str,
        description: str,
        version: str = "1.0.0",
    ):
        """
        Initialize base template.

        Args:
            template_id: Unique template identifier
            name: Human-readable template name
            description: Template description
            version: Semantic version string
        """
        self.template_id = template_id
        self.name = name
        self.description = description
        self.version = version

        # Schema definitions
        self.input_schema: Optional[AgentInputSchema] = None
        self.output_schema: Optional[AgentOutputSchema] = None

        # References
        self.formulas: List[FormulaReference] = []
        self.standards: List[StandardReference] = []
        self.constraints: List[SafetyConstraint] = []

        # Processing
        self.processing_steps: List[ProcessingStep] = []

        # Configuration
        self.zero_hallucination = ZeroHallucinationConfig()

        # Metadata
        self.tags: List[str] = []
        self.equipment_types: List[str] = []
        self.industries: List[str] = []
        self.author: str = "GreenLang Team"
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()

    @property
    @abstractmethod
    def category(self) -> str:
        """Return the template category."""
        pass

    @abstractmethod
    def define_input_schema(self) -> AgentInputSchema:
        """Define the input schema for this template."""
        pass

    @abstractmethod
    def define_output_schema(self) -> AgentOutputSchema:
        """Define the output schema for this template."""
        pass

    @abstractmethod
    def define_formulas(self) -> List[FormulaReference]:
        """Define the formulas used by this template."""
        pass

    @abstractmethod
    def define_constraints(self) -> List[SafetyConstraint]:
        """Define safety constraints for this template."""
        pass

    @abstractmethod
    def define_processing_steps(self) -> List[ProcessingStep]:
        """Define processing steps for this template."""
        pass

    def build(self) -> Dict[str, Any]:
        """
        Build the complete template structure.

        Returns:
            Complete template as dictionary for YAML serialization
        """
        self.input_schema = self.define_input_schema()
        self.output_schema = self.define_output_schema()
        self.formulas = self.define_formulas()
        self.constraints = self.define_constraints()
        self.processing_steps = self.define_processing_steps()

        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for YAML serialization."""
        return {
            "metadata": {
                "template_id": self.template_id,
                "name": self.name,
                "description": self.description,
                "category": self.category,
                "version": self.version,
                "status": "production",
                "author": self.author,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "tags": self.tags,
                "equipment_types": self.equipment_types,
                "industries": self.industries,
                "applicable_standards": [s.standard_code for s in self.standards],
            },
            "input_schema": {
                "name": self.input_schema.name if self.input_schema else "",
                "description": self.input_schema.description if self.input_schema else "",
                "fields": {
                    name: {
                        "type": f.field_type.value,
                        "description": f.description,
                        "required": f.required,
                        "unit": f.unit,
                        "min_value": f.min_value,
                        "max_value": f.max_value,
                    }
                    for name, f in (self.input_schema.fields.items() if self.input_schema else {})
                },
            },
            "output_schema": {
                "name": self.output_schema.name if self.output_schema else "",
                "description": self.output_schema.description if self.output_schema else "",
                "fields": {
                    name: {
                        "type": f.field_type.value,
                        "description": f.description,
                    }
                    for name, f in (self.output_schema.fields.items() if self.output_schema else {})
                },
            },
            "formulas": [f.to_dict() for f in self.formulas],
            "standards": [s.to_dict() for s in self.standards],
            "safety_constraints": [c.to_dict() for c in self.constraints],
            "processing": {
                "steps": [s.to_dict() for s in self.processing_steps],
            },
            "zero_hallucination": self.zero_hallucination.to_dict(),
        }

    def calculate_content_hash(self) -> str:
        """Calculate SHA-256 hash of template content."""
        import json
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# Category-Specific Base Templates
# =============================================================================

class EfficiencyAgentTemplate(BaseAgentTemplate):
    """Base template for efficiency calculation agents."""

    @property
    def category(self) -> str:
        return "efficiency"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.industries = ["oil_gas", "power_generation", "petrochemical", "manufacturing"]


class SafetyAgentTemplate(BaseAgentTemplate):
    """Base template for safety assessment agents."""

    @property
    def category(self) -> str:
        return "safety"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.industries = ["oil_gas", "chemical", "power_generation", "manufacturing"]

        # Safety agents have stricter constraints by default
        self.zero_hallucination.calculation_mode = CalculationMode.DETERMINISTIC


class EmissionsAgentTemplate(BaseAgentTemplate):
    """Base template for emissions calculation agents."""

    @property
    def category(self) -> str:
        return "emissions"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.industries = ["all"]

        # Emissions calculations require provenance for regulatory compliance
        self.zero_hallucination.require_provenance = True
        self.zero_hallucination.require_formula_citation = True


class MaintenanceAgentTemplate(BaseAgentTemplate):
    """Base template for maintenance and inspection agents."""

    @property
    def category(self) -> str:
        return "maintenance"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.industries = ["oil_gas", "power_generation", "petrochemical", "manufacturing"]

        # Maintenance agents can use ML for condition assessment
        self.zero_hallucination.llm_allowed_for.extend([
            "condition_assessment",
            "defect_classification",
            "maintenance_recommendation",
        ])
