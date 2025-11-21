# -*- coding: utf-8 -*-
"""
Scope3ReportingAgent Data Models
GL-VCCI Scope 3 Platform

Pydantic models for report generation inputs and outputs.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

from .config import ReportStandard, ExportFormat, ValidationLevel


# ============================================================================
# INPUT MODELS
# ============================================================================

class CompanyInfo(BaseModel):
    """Company information for report headers."""

    name: str = Field(..., description="Company legal name")
    registration_number: Optional[str] = Field(None, description="Company registration number")
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    website: Optional[str] = Field(None, description="Company website")

    # Reporting details
    reporting_year: int = Field(..., description="Reporting year")
    fiscal_year_start: Optional[str] = Field(None, description="Fiscal year start (MM-DD)")
    fiscal_year_end: Optional[str] = Field(None, description="Fiscal year end (MM-DD)")

    # Organizational details
    number_of_employees: Optional[int] = Field(None, ge=0, description="Total employees")
    annual_revenue_usd: Optional[float] = Field(None, ge=0, description="Annual revenue (USD)")
    industry_sector: Optional[str] = Field(None, description="Primary industry sector")

    # Contact
    contact_name: Optional[str] = Field(None)
    contact_email: Optional[str] = Field(None)

    # Branding
    logo_path: Optional[str] = Field(None, description="Path to company logo")
    primary_color: Optional[str] = Field(None, description="Primary brand color (hex)")


class EmissionsData(BaseModel):
    """Emissions data for reporting."""

    # Scope 1
    scope1_tco2e: float = Field(..., ge=0, description="Scope 1 emissions (tCO2e)")
    scope1_sources: Optional[List[Dict[str, Any]]] = Field(None, description="Scope 1 breakdown")

    # Scope 2
    scope2_location_tco2e: float = Field(..., ge=0, description="Scope 2 location-based (tCO2e)")
    scope2_market_tco2e: float = Field(..., ge=0, description="Scope 2 market-based (tCO2e)")
    scope2_sources: Optional[List[Dict[str, Any]]] = Field(None, description="Scope 2 breakdown")

    # Scope 3
    scope3_tco2e: float = Field(..., ge=0, description="Total Scope 3 emissions (tCO2e)")
    scope3_categories: Dict[int, float] = Field(..., description="Scope 3 by category (tCO2e)")
    scope3_details: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed Scope 3 data")

    # Total
    total_tco2e: Optional[float] = Field(None, description="Total emissions (auto-calculated)")

    # Quality
    avg_dqi_score: float = Field(..., ge=0, le=100, description="Average DQI score")
    data_quality_by_scope: Optional[Dict[str, float]] = Field(None, description="DQI by scope")

    # Uncertainty
    uncertainty_results: Optional[Dict[str, Any]] = Field(None, description="Uncertainty analysis")

    # Provenance
    provenance_chains: Optional[List[Dict[str, Any]]] = Field(None, description="Provenance chains")

    # Temporal
    reporting_period_start: datetime = Field(..., description="Period start")
    reporting_period_end: datetime = Field(..., description="Period end")

    # Comparative
    prior_year_emissions: Optional[Dict[str, float]] = Field(None, description="Prior year data")
    yoy_change_pct: Optional[float] = Field(None, description="Year-over-year change %")


class EnergyData(BaseModel):
    """Energy consumption data for reporting."""

    total_energy_mwh: float = Field(..., ge=0, description="Total energy consumption (MWh)")

    # By source
    renewable_energy_mwh: float = Field(default=0.0, ge=0, description="Renewable energy (MWh)")
    non_renewable_energy_mwh: float = Field(default=0.0, ge=0, description="Non-renewable (MWh)")

    # Percentages
    renewable_pct: Optional[float] = Field(None, ge=0, le=100, description="Renewable %")

    # Breakdown
    energy_by_source: Optional[Dict[str, float]] = Field(None, description="Energy by source")
    energy_by_facility: Optional[Dict[str, float]] = Field(None, description="Energy by facility")

    # Intensity
    energy_intensity_per_revenue: Optional[float] = Field(None, description="MWh per $M revenue")
    energy_intensity_per_fte: Optional[float] = Field(None, description="MWh per FTE")


class IntensityMetrics(BaseModel):
    """Carbon intensity metrics."""

    # Per revenue
    tco2e_per_million_usd: Optional[float] = Field(None, description="tCO2e per $M revenue")

    # Per employee
    tco2e_per_fte: Optional[float] = Field(None, description="tCO2e per FTE")

    # Per product/unit
    tco2e_per_unit: Optional[float] = Field(None, description="tCO2e per product unit")
    unit_name: Optional[str] = Field(None, description="Product unit name")

    # Custom metrics
    custom_metrics: Optional[Dict[str, float]] = Field(None, description="Custom intensity metrics")


class RisksOpportunities(BaseModel):
    """Climate risks and opportunities for IFRS S2."""

    physical_risks: List[Dict[str, Any]] = Field(default_factory=list, description="Physical climate risks")
    transition_risks: List[Dict[str, Any]] = Field(default_factory=list, description="Transition risks")
    opportunities: List[Dict[str, Any]] = Field(default_factory=list, description="Climate opportunities")

    financial_impact_assessment: Optional[str] = Field(None, description="Financial impact summary")
    resilience_strategy: Optional[str] = Field(None, description="Climate resilience strategy")


class TransportData(BaseModel):
    """Transport emissions data for ISO 14083."""

    transport_by_mode: Dict[str, Dict[str, Any]] = Field(..., description="Breakdown by mode")
    total_tonne_km: float = Field(..., ge=0, description="Total tonne-kilometers")
    total_emissions_tco2e: float = Field(..., ge=0, description="Total transport emissions")

    # Data quality
    emission_factors_used: List[Dict[str, Any]] = Field(..., description="Emission factors")
    data_quality_score: float = Field(..., ge=0, le=100, description="Data quality score")

    # Conformance
    methodology: str = Field(default="ISO 14083:2023", description="Calculation methodology")
    calculation_results: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed results")


# ============================================================================
# VALIDATION MODELS
# ============================================================================

class ValidationCheck(BaseModel):
    """Individual validation check result."""

    check_name: str = Field(..., description="Check name")
    status: str = Field(..., description="PASS/FAIL/WARNING")
    message: Optional[str] = Field(None, description="Status message")
    severity: str = Field(default="info", description="Severity level")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class ValidationResult(BaseModel):
    """Complete validation result."""

    is_valid: bool = Field(..., description="Overall validation status")
    validation_level: ValidationLevel = Field(..., description="Validation level used")

    # Checks
    checks: List[ValidationCheck] = Field(..., description="Individual checks")
    passed_checks: int = Field(..., ge=0, description="Number passed")
    failed_checks: int = Field(..., ge=0, description="Number failed")
    warnings: int = Field(..., ge=0, description="Number of warnings")

    # Standard-specific
    standard: Optional[ReportStandard] = Field(None, description="Standard validated for")
    completeness_pct: Optional[float] = Field(None, ge=0, le=100, description="Data completeness %")

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")

    # Metadata
    validated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# OUTPUT MODELS
# ============================================================================

class ChartInfo(BaseModel):
    """Generated chart information."""

    chart_id: str = Field(..., description="Chart identifier")
    chart_type: str = Field(..., description="Chart type")
    title: str = Field(..., description="Chart title")

    # File paths
    image_path: Optional[str] = Field(None, description="Path to chart image")
    data_path: Optional[str] = Field(None, description="Path to chart data")

    # Metadata
    width: Optional[int] = Field(None, description="Width in pixels")
    height: Optional[int] = Field(None, description="Height in pixels")
    format: str = Field(default="png", description="Image format")


class ReportMetadata(BaseModel):
    """Report generation metadata."""

    report_id: str = Field(..., description="Unique report identifier")
    standard: ReportStandard = Field(..., description="Reporting standard")
    export_format: ExportFormat = Field(..., description="Export format")

    # Generation info
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: Optional[str] = Field(None, description="User/system that generated report")
    generation_time_seconds: Optional[float] = Field(None, description="Generation time")

    # Version
    agent_version: str = Field(default="1.0.0", description="Agent version")
    standard_version: Optional[str] = Field(None, description="Standard version")

    # Data coverage
    reporting_period: str = Field(..., description="Reporting period")
    data_sources: List[str] = Field(default_factory=list, description="Data sources")

    # Quality
    validation_passed: bool = Field(..., description="Validation status")
    data_quality_score: float = Field(..., ge=0, le=100, description="Overall DQI score")


class ReportResult(BaseModel):
    """Result of report generation."""

    success: bool = Field(..., description="Generation success")
    metadata: ReportMetadata = Field(..., description="Report metadata")

    # Output files
    file_path: Optional[str] = Field(None, description="Primary output file path")
    additional_files: List[str] = Field(default_factory=list, description="Additional files")

    # Charts
    charts: List[ChartInfo] = Field(default_factory=list, description="Generated charts")

    # Content summary
    sections_generated: List[str] = Field(default_factory=list, description="Report sections")
    tables_count: int = Field(default=0, ge=0, description="Number of tables")
    charts_count: int = Field(default=0, ge=0, description="Number of charts")

    # Validation
    validation_result: Optional[ValidationResult] = Field(None, description="Pre-generation validation")

    # Warnings/Errors
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    errors: List[str] = Field(default_factory=list, description="Errors")

    # Content (for API)
    content: Optional[Dict[str, Any]] = Field(None, description="Report content (JSON)")


class BatchReportResult(BaseModel):
    """Result of batch report generation."""

    total_reports: int = Field(..., ge=0, description="Total reports requested")
    successful_reports: int = Field(..., ge=0, description="Successfully generated")
    failed_reports: int = Field(..., ge=0, description="Failed generation")

    results: List[ReportResult] = Field(..., description="Individual results")

    total_generation_time: float = Field(..., ge=0, description="Total time (seconds)")

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_reports == 0:
            return 0.0
        return self.successful_reports / self.total_reports


__all__ = [
    # Input models
    "CompanyInfo",
    "EmissionsData",
    "EnergyData",
    "IntensityMetrics",
    "RisksOpportunities",
    "TransportData",

    # Validation models
    "ValidationCheck",
    "ValidationResult",

    # Output models
    "ChartInfo",
    "ReportMetadata",
    "ReportResult",
    "BatchReportResult",
]
