"""Type definitions for all GreenLang agents.

This module defines the input and output types for each agent,
ensuring type safety and clear contracts.
"""

from typing import List, Dict, Union, Tuple
from typing_extensions import TypedDict, NotRequired, Literal
from ..types import (
    CountryCode,
    UnitStr,
    FuelType,
    BuildingType,
    BenchmarkRating,
    ReportFormat,
    KgCO2e,
    Quantity,
    Location,
    EmissionFactorInfo,
)

# ==============================================================================
# FuelAgent Types
# ==============================================================================


class FuelInput(TypedDict):
    """Input for FuelAgent."""

    fuel_type: FuelType
    consumption: Quantity
    country: CountryCode
    year: NotRequired[int]  # For historical factors


class FuelOutput(TypedDict):
    """Output from FuelAgent."""

    co2e_emissions_kg: KgCO2e
    fuel_type: FuelType
    consumption_value: float
    consumption_unit: UnitStr
    emission_factor: float
    emission_factor_unit: str
    source: str
    version: str
    last_updated: str
    confidence: NotRequired[float]


# ==============================================================================
# BoilerAgent Types
# ==============================================================================


class BoilerInput(TypedDict):
    """Input for BoilerAgent."""

    boiler_type: Literal[
        "condensing",
        "standard",
        "low_efficiency",
        "modern",
        "traditional",
        "pulverized",
        "stoker",
        "hand_fired",
        "resistance",
        "heat_pump",
    ]
    fuel_type: FuelType
    thermal_output: NotRequired[
        Quantity
    ]  # Either thermal output OR fuel consumption required
    fuel_consumption: NotRequired[
        Quantity
    ]  # Either thermal output OR fuel consumption required
    efficiency: NotRequired[float]  # 0-1 scale, optional
    age: NotRequired[Literal["new", "medium", "old"]]
    country: NotRequired[CountryCode]
    year: NotRequired[int]
    operating_hours: NotRequired[float]  # Annual operating hours


class BoilerOutput(TypedDict):
    """Output from BoilerAgent."""

    co2e_emissions_kg: KgCO2e
    boiler_type: str
    fuel_type: FuelType
    fuel_consumption_value: float
    fuel_consumption_unit: UnitStr
    thermal_output_value: float
    thermal_output_unit: UnitStr
    efficiency: float
    thermal_efficiency_percent: float
    emission_factor: float
    emission_factor_unit: str
    fuel_intensity: float  # Fuel per unit thermal output
    emission_intensity: float  # Emissions per unit thermal output
    performance_rating: BenchmarkRating
    recommendations: List[Dict[str, str]]
    source: str
    version: str
    last_updated: str
    confidence: NotRequired[float]


# ==============================================================================
# GridFactorAgent Types
# ==============================================================================


class GridFactorInput(TypedDict):
    """Input for GridFactorAgent."""

    country: CountryCode
    fuel_type: FuelType
    unit: UnitStr
    year: NotRequired[int]
    month: NotRequired[int]


class GridFactorOutput(EmissionFactorInfo):
    """Output from GridFactorAgent (extends EmissionFactorInfo)."""

    country: CountryCode
    fuel_type: FuelType
    grid_mix: NotRequired[Dict[str, float]]  # Renewable %, fossil %, etc.


# ==============================================================================
# InputValidatorAgent Types
# ==============================================================================


class RawBuildingInput(TypedDict):
    """Raw input data for a building (before validation)."""

    metadata: Dict[str, object]  # Flexible structure
    energy_consumption: Dict[str, object]  # Various fuel types
    water_usage: NotRequired[Dict[str, object]]
    waste_generation: NotRequired[Dict[str, object]]


class NormalizedBuildingInput(TypedDict):
    """Normalized building input after validation."""

    metadata: "BuildingMetadata"
    energy_consumption: Dict[FuelType, Quantity]
    water_usage: NotRequired[Quantity]
    waste_generation: NotRequired[Quantity]
    validation_warnings: NotRequired[List[str]]


class BuildingMetadata(TypedDict):
    """Structured building metadata."""

    building_type: BuildingType
    area: float
    area_unit: Literal["sqft", "sqm"]
    location: Location
    occupancy: NotRequired[int]
    floor_count: NotRequired[int]
    building_age: NotRequired[int]
    certification: NotRequired[str]  # LEED, Energy Star, etc.


# ==============================================================================
# CarbonAgent Types
# ==============================================================================


class EmissionItem(TypedDict):
    """Single emission source."""

    fuel: FuelType
    co2e_emissions_kg: KgCO2e
    scope: NotRequired[Literal["1", "2", "3"]]  # GHG Protocol scopes


class CarbonInput(TypedDict):
    """Input for CarbonAgent."""

    emissions: List[EmissionItem]
    include_offsets: NotRequired[bool]
    group_by_scope: NotRequired[bool]


class CarbonOutput(TypedDict):
    """Output from CarbonAgent."""

    total_co2e_kg: KgCO2e
    total_co2e_tons: float
    by_fuel: Dict[FuelType, KgCO2e]
    by_fuel_percent: Dict[FuelType, float]
    by_scope: NotRequired[Dict[str, KgCO2e]]
    largest_source: FuelType
    smallest_source: FuelType


# ==============================================================================
# IntensityAgent Types
# ==============================================================================


class IntensityInput(TypedDict):
    """Input for IntensityAgent."""

    total_co2e_kg: KgCO2e
    area: float
    area_unit: Literal["sqft", "sqm"]
    occupancy: NotRequired[int]
    operating_hours: NotRequired[float]  # Annual hours


class IntensityOutput(TypedDict):
    """Output from IntensityAgent."""

    co2e_per_sqft: float
    co2e_per_sqm: float
    co2e_per_person: NotRequired[float]
    co2e_per_hour: NotRequired[float]
    energy_use_intensity: NotRequired[float]  # EUI if energy data available


# ==============================================================================
# BenchmarkAgent Types
# ==============================================================================


class BenchmarkInput(TypedDict):
    """Input for BenchmarkAgent."""

    co2e_per_sqft: float
    building_type: BuildingType
    country: NotRequired[CountryCode]
    climate_zone: NotRequired[str]
    year: NotRequired[int]


class BenchmarkOutput(TypedDict):
    """Output from BenchmarkAgent."""

    rating: BenchmarkRating
    score: float  # 0-100
    threshold_excellent: float
    threshold_good: float
    threshold_average: float
    threshold_poor: float
    percentile: NotRequired[float]  # Where this building ranks
    similar_buildings_avg: NotRequired[float]
    improvement_potential: float  # kg CO2e possible reduction


# ==============================================================================
# RecommendationAgent Types
# ==============================================================================


class RecommendationInput(TypedDict):
    """Input for RecommendationAgent."""

    rating: BenchmarkRating
    by_fuel: Dict[FuelType, KgCO2e]
    co2e_per_sqft: float
    building_type: BuildingType
    building_age: NotRequired[int]
    climate_zone: NotRequired[str]


class Recommendation(TypedDict):
    """Single recommendation."""

    id: str
    category: Literal["quick_win", "medium_term", "long_term"]
    action: str
    estimated_savings_percent: float
    estimated_cost: NotRequired[str]  # Low/Medium/High
    payback_years: NotRequired[float]
    implementation_difficulty: NotRequired[Literal["easy", "moderate", "complex"]]


class RecommendationOutput(TypedDict):
    """Output from RecommendationAgent."""

    recommendations: List[Recommendation]
    quick_wins: List[Recommendation]
    total_savings_potential: float  # kg CO2e
    implementation_roadmap: NotRequired[List[str]]  # Ordered steps


# ==============================================================================
# ReportAgent Types
# ==============================================================================


class ReportInput(TypedDict):
    """Input for ReportAgent."""

    emissions: CarbonOutput
    intensity: IntensityOutput
    benchmark: BenchmarkOutput
    recommendations: NotRequired[List[Recommendation]]
    format: ReportFormat
    include_charts: NotRequired[bool]
    executive_summary: NotRequired[bool]


class ReportOutput(TypedDict):
    """Output from ReportAgent."""

    report: Union[str, Dict[str, object]]  # String for MD/HTML, dict for JSON
    format: ReportFormat
    generated_at: str  # ISO 8601 timestamp
    report_id: NotRequired[str]
    file_path: NotRequired[str]  # If saved to disk


# ==============================================================================
# BuildingProfileAgent Types
# ==============================================================================


class BuildingProfileInput(TypedDict):
    """Input for BuildingProfileAgent."""

    building_type: BuildingType
    area: float
    area_unit: Literal["sqft", "sqm"]
    location: Location
    age: NotRequired[int]
    occupancy: NotRequired[int]


class BuildingProfileOutput(TypedDict):
    """Output from BuildingProfileAgent."""

    profile_id: str
    expected_eui: float  # Energy Use Intensity
    expected_emissions_range: Tuple[float, float]  # (min, max)
    similar_buildings_count: int
    typical_fuel_mix: Dict[FuelType, float]  # Percentages
    recommended_targets: Dict[str, float]


# ==============================================================================
# AssistantAgent Types (for LLM integration)
# ==============================================================================


class AssistantInput(TypedDict):
    """Input for AI Assistant."""

    query: str
    context: NotRequired[Dict[str, object]]
    mode: NotRequired[Literal["analysis", "recommendation", "explanation"]]
    max_tokens: NotRequired[int]


class AssistantOutput(TypedDict):
    """Output from AI Assistant."""

    response: str
    confidence: float
    sources: NotRequired[List[str]]
    follow_up_questions: NotRequired[List[str]]


# ==============================================================================
# Workflow Types
# ==============================================================================


class WorkflowInput(TypedDict):
    """Generic workflow input."""

    building_data: RawBuildingInput
    workflow_config: NotRequired[Dict[str, object]]
    output_format: NotRequired[ReportFormat]


class WorkflowOutput(TypedDict):
    """Complete workflow output."""

    emissions: CarbonOutput
    intensity: IntensityOutput
    benchmark: BenchmarkOutput
    recommendations: List[Recommendation]
    report: NotRequired[ReportOutput]
    execution_time: float
    steps_completed: List[str]


# ==============================================================================
# Portfolio Types (for multiple buildings)
# ==============================================================================


class PortfolioInput(TypedDict):
    """Input for portfolio analysis."""

    buildings: List[RawBuildingInput]
    aggregation_method: NotRequired[Literal["sum", "average", "weighted"]]
    comparison_enabled: NotRequired[bool]


class PortfolioOutput(TypedDict):
    """Output from portfolio analysis."""

    total_emissions: KgCO2e
    total_area: float
    average_intensity: float
    building_results: List[WorkflowOutput]
    best_performer: NotRequired[str]  # Building ID
    worst_performer: NotRequired[str]  # Building ID
    portfolio_rating: BenchmarkRating


# ==============================================================================
# Export all types
# ==============================================================================

__all__ = [
    # Fuel
    "FuelInput",
    "FuelOutput",
    # Boiler
    "BoilerInput",
    "BoilerOutput",
    # Grid Factor
    "GridFactorInput",
    "GridFactorOutput",
    # Validator
    "RawBuildingInput",
    "NormalizedBuildingInput",
    "BuildingMetadata",
    # Carbon
    "EmissionItem",
    "CarbonInput",
    "CarbonOutput",
    # Intensity
    "IntensityInput",
    "IntensityOutput",
    # Benchmark
    "BenchmarkInput",
    "BenchmarkOutput",
    # Recommendations
    "RecommendationInput",
    "Recommendation",
    "RecommendationOutput",
    # Report
    "ReportInput",
    "ReportOutput",
    # Building Profile
    "BuildingProfileInput",
    "BuildingProfileOutput",
    # Assistant
    "AssistantInput",
    "AssistantOutput",
    # Workflow
    "WorkflowInput",
    "WorkflowOutput",
    # Portfolio
    "PortfolioInput",
    "PortfolioOutput",
]
