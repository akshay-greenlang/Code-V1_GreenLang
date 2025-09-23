"""Core type definitions and contracts for GreenLang.

This module provides the foundational types used throughout the GreenLang system,
ensuring type safety and clear contracts between components.
"""

from __future__ import annotations
from typing import (
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import TypedDict, NotRequired, Annotated

# ==============================================================================
# Unit Type Annotations
# ==============================================================================

# Energy and emissions units with semantic meaning
KgCO2e = Annotated[float, "kg CO2e"]
TonsCO2e = Annotated[float, "tons CO2e"]
KWh = Annotated[float, "kWh"]
MWh = Annotated[float, "MWh"]
Therms = Annotated[float, "therms"]
Liters = Annotated[float, "liters"]
Gallons = Annotated[float, "gallons"]
SquareFeet = Annotated[float, "square feet"]
SquareMeters = Annotated[float, "square meters"]

# ==============================================================================
# Literal Types for Enumerations
# ==============================================================================

# Supported country codes (extend as dataset grows)
CountryCode = Literal[
    "US",
    "IN",
    "EU",
    "CN",
    "JP",
    "BR",
    "KR",
    "UK",
    "DE",
    "CA",
    "AU",
    "FR",
    "IT",
    "ES",
    "MX",
    "ID",
    "RU",
    "SA",
    "ZA",
    "EG",
    "NG",
]

# Energy units
EnergyUnit = Literal["kWh", "MWh", "GWh", "therms", "MMBtu", "GJ", "kcal"]

# Volume units
VolumeUnit = Literal["liters", "gallons", "m3", "cubic_feet", "barrels"]

# Mass units
MassUnit = Literal["kg", "tons", "metric_tons", "pounds", "short_tons"]

# Area units
AreaUnit = Literal["sqft", "sqm", "acres", "hectares"]

# All units combined
UnitStr = Union[EnergyUnit, VolumeUnit, MassUnit, AreaUnit]

# Fuel types
FuelType = Literal[
    "electricity",
    "natural_gas",
    "diesel",
    "gasoline",
    "propane",
    "fuel_oil",
    "coal",
    "biomass",
    "solar_pv_generation",
    "district_heating",
]

# Building types
BuildingType = Literal[
    "commercial_office",
    "hospital",
    "data_center",
    "retail",
    "warehouse",
    "hotel",
    "education",
    "restaurant",
    "industrial",
    "residential",
    "mixed_use",
    "laboratory",
    "museum",
    "stadium",
]

# Benchmark ratings
BenchmarkRating = Literal["excellent", "good", "average", "below_average", "poor"]

# Report formats
ReportFormat = Literal["json", "markdown", "html", "pdf", "excel"]

# ==============================================================================
# Error Handling Types
# ==============================================================================


class ErrorInfo(TypedDict):
    """Structured error information with context."""

    type: str  # Error type (e.g., "ValidationError", "DataError")
    message: str  # Human-readable error message
    context: NotRequired[Dict[str, object]]  # Additional context
    agent_id: NotRequired[str]  # Agent that raised the error
    step: NotRequired[str]  # Workflow step where error occurred
    traceback: NotRequired[str]  # Stack trace for debugging


# ==============================================================================
# Result Types (Success/Failure Pattern)
# ==============================================================================

T_co = TypeVar("T_co", covariant=True)  # Result payload type


class SuccessResult(TypedDict, Generic[T_co]):
    """Successful operation result with typed data."""

    success: Literal[True]
    data: T_co
    metadata: NotRequired[Dict[str, object]]  # Optional metadata


class FailureResult(TypedDict):
    """Failed operation result with error information."""

    success: Literal[False]
    error: ErrorInfo
    metadata: NotRequired[Dict[str, object]]  # Optional metadata


# Union type for all agent results
AgentResult = Union[SuccessResult[T_co], FailureResult]

# ==============================================================================
# Agent Protocol Definition
# ==============================================================================

InT = TypeVar("InT", contravariant=True)  # Input type
OutT = TypeVar("OutT", covariant=True)  # Output type


class Agent(Protocol[InT, OutT]):
    """Protocol defining the contract all agents must follow."""

    agent_id: str
    name: str
    version: str

    def run(self, payload: InT) -> AgentResult[OutT]:
        """Execute the agent with typed input and output."""
        ...

    def validate(self, payload: InT) -> bool:
        """Validate input payload before execution."""
        ...


# ==============================================================================
# Common Data Structures
# ==============================================================================


class Quantity(TypedDict):
    """Represents a measurement with value and unit."""

    value: float
    unit: UnitStr


class Location(TypedDict):
    """Geographic location information."""

    country: CountryCode
    region: NotRequired[str]
    city: NotRequired[str]
    postal_code: NotRequired[str]
    coordinates: NotRequired[Tuple[float, float]]  # (latitude, longitude)


class DateRange(TypedDict):
    """Date range for time-based queries."""

    start: str  # ISO 8601 date string
    end: str  # ISO 8601 date string


class EmissionFactorInfo(TypedDict):
    """Emission factor with metadata."""

    emission_factor: float
    unit: str
    source: str
    version: str
    last_updated: str
    confidence: NotRequired[float]  # 0.0 to 1.0
    notes: NotRequired[str]


# ==============================================================================
# Workflow Types
# ==============================================================================


class WorkflowStep(TypedDict):
    """Single step in a workflow."""

    name: str
    agent_id: str
    input_mapping: NotRequired[Dict[str, str]]  # Map context to agent input
    output_key: NotRequired[str]  # Where to store output in context
    condition: NotRequired[str]  # Conditional execution expression
    retry_count: NotRequired[int]  # Number of retries on failure
    timeout: NotRequired[float]  # Timeout in seconds


class WorkflowDefinition(TypedDict):
    """Complete workflow definition."""

    name: str
    version: str
    description: NotRequired[str]
    steps: List[WorkflowStep]
    input_schema: NotRequired[Dict[str, object]]  # JSON Schema
    output_schema: NotRequired[Dict[str, object]]  # JSON Schema


# ==============================================================================
# Validation Types
# ==============================================================================


class ValidationError(TypedDict):
    """Validation error details."""

    field: str
    message: str
    expected: NotRequired[object]
    actual: NotRequired[object]


class ValidationResult(TypedDict):
    """Result of validation operation."""

    valid: bool
    errors: List[ValidationError]
    warnings: NotRequired[List[str]]


# ==============================================================================
# Type Guards (Runtime Type Checking)
# ==============================================================================


def is_success_result(result: AgentResult[object]) -> bool:
    """Type guard to check if result is successful."""
    return result.get("success", False) is True


def is_failure_result(result: AgentResult[object]) -> bool:
    """Type guard to check if result is failure."""
    return result.get("success", True) is False


# ==============================================================================
# Utility Type Aliases
# ==============================================================================

# JSON-compatible types
JSONValue = Union[
    str, int, float, bool, None, Dict[str, "JSONValue"], List["JSONValue"]
]
JSONObject = Dict[str, JSONValue]

# Configuration types
ConfigDict = Mapping[str, object]
ParameterDict = Dict[str, Union[str, int, float, bool, None]]

# Data collections
EmissionsDict = Dict[FuelType, KgCO2e]
FactorsDict = Dict[Tuple[CountryCode, FuelType], float]

# ==============================================================================
# Export all public types
# ==============================================================================

__all__ = [
    # Units
    "KgCO2e",
    "TonsCO2e",
    "KWh",
    "MWh",
    "Therms",
    "Liters",
    "Gallons",
    "SquareFeet",
    "SquareMeters",
    # Literals
    "CountryCode",
    "EnergyUnit",
    "VolumeUnit",
    "MassUnit",
    "AreaUnit",
    "UnitStr",
    "FuelType",
    "BuildingType",
    "BenchmarkRating",
    "ReportFormat",
    # Core types
    "ErrorInfo",
    "SuccessResult",
    "FailureResult",
    "AgentResult",
    "Agent",
    "Quantity",
    "Location",
    "DateRange",
    "EmissionFactorInfo",
    # Workflow types
    "WorkflowStep",
    "WorkflowDefinition",
    # Validation types
    "ValidationError",
    "ValidationResult",
    # Type guards
    "is_success_result",
    "is_failure_result",
    # Utility types
    "JSONValue",
    "JSONObject",
    "ConfigDict",
    "ParameterDict",
    "EmissionsDict",
    "FactorsDict",
]
