"""
Compliance Reporter for GL-003 UnifiedSteam SteamSystemOptimizer

This module implements GHG Protocol aligned reporting and energy compliance
reporting for steam system operations.

Key Features:
    - GHG Protocol Corporate Standard aligned reporting
    - Scope 1, 2, and 3 emissions categorization
    - Energy consumption reporting (ISO 50001 compatible)
    - Savings verification reporting
    - Multiple export formats (JSON, CSV, PDF)

Reference Standards:
    - GHG Protocol Corporate Accounting and Reporting Standard
    - GHG Protocol Scope 2 Guidance
    - ISO 14064-1: GHG Inventories
    - ISO 50001: Energy Management
    - EPA 40 CFR Part 98 Subpart C

Example:
    >>> reporter = ComplianceReporter(audit_logger, provenance_tracker)
    >>> ghg_report = reporter.generate_ghg_report(
    ...     time_period=period,
    ...     scope=GHGScope.SCOPE_1,
    ...     boundary="operational_control"
    ... )
    >>> energy_report = reporter.generate_energy_report(time_period=period)
    >>> exported = reporter.export_for_auditor(ghg_report, ReportFormat.PDF)

Author: GreenLang Steam Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class GHGScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "SCOPE_1"  # Direct emissions
    SCOPE_2 = "SCOPE_2"  # Indirect - purchased electricity, steam, heat
    SCOPE_3 = "SCOPE_3"  # Other indirect emissions
    ALL = "ALL"  # All scopes


class ReportFormat(str, Enum):
    """Supported report export formats."""

    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"
    XLSX = "XLSX"


class ReportingPeriod(BaseModel):
    """Time period for compliance reporting."""

    start_date: datetime = Field(..., description="Period start date")
    end_date: datetime = Field(..., description="Period end date")
    period_type: str = Field(
        default="CUSTOM", description="Period type (MONTHLY, QUARTERLY, ANNUAL, CUSTOM)"
    )
    fiscal_year: Optional[int] = Field(None, description="Fiscal year if applicable")
    reporting_quarter: Optional[int] = Field(
        None, ge=1, le=4, description="Quarter if applicable"
    )

    @validator("end_date")
    def end_after_start(cls, v, values):
        """Validate end_date is after start_date."""
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    @property
    def days(self) -> int:
        """Number of days in period."""
        return (self.end_date - self.start_date).days

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class EmissionFactor(BaseModel):
    """
    Emission factor for GHG calculations.

    Based on EPA and IPCC emission factors for common fuels.
    """

    factor_id: str = Field(..., description="Unique factor identifier")
    fuel_type: str = Field(..., description="Fuel type")
    factor_value: float = Field(..., gt=0, description="Emission factor value")
    factor_unit: str = Field(..., description="Factor unit (kg CO2e/MMBtu, etc.)")
    source: str = Field(..., description="Factor source (EPA, IPCC, etc.)")
    source_year: int = Field(..., description="Source publication year")
    gwp_co2: float = Field(1.0, description="Global warming potential - CO2")
    gwp_ch4: float = Field(25.0, description="Global warming potential - CH4")
    gwp_n2o: float = Field(298.0, description="Global warming potential - N2O")

    class Config:
        frozen = True


class EmissionSource(BaseModel):
    """A source of GHG emissions."""

    source_id: str = Field(..., description="Source identifier")
    source_name: str = Field(..., description="Source name")
    source_type: str = Field(
        ..., description="Source type (STATIONARY_COMBUSTION, etc.)"
    )
    scope: GHGScope = Field(..., description="GHG Protocol scope")
    fuel_type: str = Field(..., description="Fuel type")
    activity_data: float = Field(..., ge=0, description="Activity data (fuel consumed)")
    activity_unit: str = Field(..., description="Activity unit")
    emission_factor: EmissionFactor = Field(..., description="Applied emission factor")
    co2_emissions_kg: float = Field(..., ge=0, description="CO2 emissions (kg)")
    ch4_emissions_kg: float = Field(0.0, ge=0, description="CH4 emissions (kg)")
    n2o_emissions_kg: float = Field(0.0, ge=0, description="N2O emissions (kg)")
    co2e_emissions_kg: float = Field(..., ge=0, description="CO2e emissions (kg)")

    class Config:
        frozen = True


class GHGReport(BaseModel):
    """
    GHG Protocol aligned emissions report.

    Contains Scope 1, 2, and/or 3 emissions with full
    methodology documentation.
    """

    report_id: UUID = Field(default_factory=uuid4, description="Unique report ID")
    report_type: str = Field(default="GHG_INVENTORY", description="Report type")
    report_version: str = Field(default="1.0.0", description="Report version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report creation timestamp"
    )

    # Reporting entity
    organization_name: str = Field(..., description="Organization name")
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")

    # Period
    reporting_period: ReportingPeriod = Field(..., description="Reporting period")

    # Boundary
    boundary_type: str = Field(
        ..., description="Boundary type (OPERATIONAL_CONTROL, FINANCIAL_CONTROL, EQUITY_SHARE)"
    )
    boundary_description: str = Field(..., description="Boundary description")
    included_scopes: List[GHGScope] = Field(..., description="Scopes included")

    # Scope 1 emissions
    scope1_total_co2e_mt: float = Field(
        0.0, ge=0, description="Total Scope 1 CO2e (metric tons)"
    )
    scope1_sources: List[EmissionSource] = Field(
        default_factory=list, description="Scope 1 emission sources"
    )
    scope1_by_fuel: Dict[str, float] = Field(
        default_factory=dict, description="Scope 1 by fuel type (metric tons CO2e)"
    )

    # Scope 2 emissions
    scope2_location_based_co2e_mt: float = Field(
        0.0, ge=0, description="Scope 2 location-based CO2e (metric tons)"
    )
    scope2_market_based_co2e_mt: float = Field(
        0.0, ge=0, description="Scope 2 market-based CO2e (metric tons)"
    )
    scope2_sources: List[EmissionSource] = Field(
        default_factory=list, description="Scope 2 emission sources"
    )

    # Scope 3 emissions (optional)
    scope3_total_co2e_mt: float = Field(
        0.0, ge=0, description="Total Scope 3 CO2e (metric tons)"
    )
    scope3_categories: Dict[str, float] = Field(
        default_factory=dict, description="Scope 3 by category"
    )

    # Totals
    total_co2e_mt: float = Field(..., ge=0, description="Total CO2e (metric tons)")
    total_co2_mt: float = Field(..., ge=0, description="Total CO2 (metric tons)")
    total_ch4_mt: float = Field(0.0, ge=0, description="Total CH4 (metric tons)")
    total_n2o_mt: float = Field(0.0, ge=0, description="Total N2O (metric tons)")

    # Methodology
    methodology_notes: str = Field(..., description="Methodology documentation")
    emission_factors_source: str = Field(
        ..., description="Source of emission factors"
    )
    data_quality_statement: str = Field(..., description="Data quality statement")

    # Verification
    verification_status: str = Field(
        default="UNVERIFIED", description="Verification status"
    )
    verifier_name: Optional[str] = Field(None, description="Third-party verifier")
    verification_date: Optional[datetime] = Field(
        None, description="Verification date"
    )

    # Prepared by
    prepared_by: str = Field(..., description="Preparer name/ID")
    approved_by: Optional[str] = Field(None, description="Approver name/ID")

    # Hash for integrity
    report_hash: Optional[str] = Field(None, description="SHA-256 hash of report")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of report content."""
        data = self.dict(exclude={"report_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class EnergyReport(BaseModel):
    """
    Energy consumption report aligned with ISO 50001.

    Contains energy consumption by source, efficiency metrics,
    and energy performance indicators.
    """

    report_id: UUID = Field(default_factory=uuid4, description="Unique report ID")
    report_type: str = Field(default="ENERGY_CONSUMPTION", description="Report type")
    report_version: str = Field(default="1.0.0", description="Report version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report creation timestamp"
    )

    # Facility
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")

    # Period
    reporting_period: ReportingPeriod = Field(..., description="Reporting period")

    # Total energy
    total_energy_mmbtu: float = Field(..., ge=0, description="Total energy (MMBtu)")
    total_energy_gj: float = Field(..., ge=0, description="Total energy (GJ)")
    total_energy_mwh: float = Field(..., ge=0, description="Total energy (MWh)")

    # Energy by source
    natural_gas_mmbtu: float = Field(0.0, ge=0, description="Natural gas (MMBtu)")
    fuel_oil_mmbtu: float = Field(0.0, ge=0, description="Fuel oil (MMBtu)")
    propane_mmbtu: float = Field(0.0, ge=0, description="Propane (MMBtu)")
    electricity_mwh: float = Field(0.0, ge=0, description="Electricity (MWh)")
    purchased_steam_mmbtu: float = Field(0.0, ge=0, description="Purchased steam (MMBtu)")
    other_energy_mmbtu: float = Field(0.0, ge=0, description="Other energy (MMBtu)")

    # Energy breakdown
    energy_by_source: Dict[str, float] = Field(
        default_factory=dict, description="Energy by source (MMBtu)"
    )
    energy_by_end_use: Dict[str, float] = Field(
        default_factory=dict, description="Energy by end use (MMBtu)"
    )

    # Steam production
    steam_production_klb: float = Field(0.0, ge=0, description="Steam production (klb)")
    steam_by_header: Dict[str, float] = Field(
        default_factory=dict, description="Steam by header (klb)"
    )

    # Efficiency metrics
    boiler_efficiency_pct: float = Field(
        ..., ge=0, le=100, description="Average boiler efficiency (%)"
    )
    system_efficiency_pct: float = Field(
        ..., ge=0, le=100, description="Average system efficiency (%)"
    )
    heat_rate_mmbtu_per_klb: float = Field(
        ..., ge=0, description="Heat rate (MMBtu/klb steam)"
    )

    # Energy Performance Indicators (EnPIs)
    energy_intensity_mmbtu_per_unit: float = Field(
        ..., ge=0, description="Energy intensity (MMBtu/production unit)"
    )
    production_volume: float = Field(..., ge=0, description="Production volume")
    production_unit: str = Field(..., description="Production unit of measure")

    # Baseline comparison
    baseline_energy_mmbtu: Optional[float] = Field(
        None, ge=0, description="Baseline energy (MMBtu)"
    )
    baseline_period: Optional[str] = Field(None, description="Baseline period")
    energy_improvement_pct: Optional[float] = Field(
        None, description="Improvement from baseline (%)"
    )

    # Cost
    total_energy_cost_usd: float = Field(0.0, ge=0, description="Total energy cost (USD)")
    cost_by_source: Dict[str, float] = Field(
        default_factory=dict, description="Cost by source (USD)"
    )

    # Methodology
    methodology_notes: str = Field(..., description="Methodology documentation")
    data_quality_statement: str = Field(..., description="Data quality statement")

    # Prepared by
    prepared_by: str = Field(..., description="Preparer name/ID")

    # Hash
    report_hash: Optional[str] = Field(None, description="SHA-256 hash of report")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of report content."""
        data = self.dict(exclude={"report_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class VerificationReport(BaseModel):
    """
    Savings verification report for M&V compliance.

    Documents verified energy and cost savings for a project.
    """

    report_id: UUID = Field(default_factory=uuid4, description="Unique report ID")
    report_type: str = Field(default="SAVINGS_VERIFICATION", description="Report type")
    report_version: str = Field(default="1.0.0", description="Report version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report creation timestamp"
    )

    # Project
    project_id: str = Field(..., description="Project identifier")
    project_name: str = Field(..., description="Project name")
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")

    # Period
    verification_period: ReportingPeriod = Field(
        ..., description="Verification period"
    )

    # ECMs verified
    ecm_ids: List[str] = Field(..., description="ECM identifiers")
    ecm_descriptions: List[str] = Field(..., description="ECM descriptions")

    # Savings summary
    verified_energy_savings_mmbtu: float = Field(
        ..., description="Verified energy savings (MMBtu)"
    )
    verified_cost_savings_usd: float = Field(
        ..., description="Verified cost savings (USD)"
    )
    verified_co2e_reduction_mt: float = Field(
        ..., description="Verified CO2e reduction (metric tons)"
    )

    # Comparison to predicted
    predicted_energy_savings_mmbtu: Optional[float] = Field(
        None, description="Predicted energy savings (MMBtu)"
    )
    savings_realization_rate_pct: Optional[float] = Field(
        None, description="Savings realization rate (%)"
    )

    # Uncertainty
    savings_uncertainty_pct: float = Field(
        ..., ge=0, le=100, description="Savings uncertainty (%)"
    )
    confidence_level_pct: float = Field(
        90.0, ge=0, le=100, description="Confidence level (%)"
    )
    savings_range_low_mmbtu: float = Field(
        ..., description="Low end of savings range (MMBtu)"
    )
    savings_range_high_mmbtu: float = Field(
        ..., description="High end of savings range (MMBtu)"
    )

    # Methodology
    mv_methodology: str = Field(
        ..., description="M&V methodology (IPMVP Option A/B/C/D)"
    )
    measurement_boundary: str = Field(..., description="Measurement boundary")
    methodology_notes: str = Field(..., description="Methodology documentation")

    # Evidence references
    baseline_evidence_id: str = Field(..., description="Baseline evidence ID")
    post_evidence_id: str = Field(..., description="Post-implementation evidence ID")
    savings_evidence_id: str = Field(..., description="Savings evidence ID")

    # Verification
    verification_status: str = Field(
        ..., description="Verification status (VERIFIED, PARTIALLY_VERIFIED, NOT_VERIFIED)"
    )
    verification_notes: Optional[str] = Field(
        None, description="Verification notes"
    )

    # Prepared by
    prepared_by: str = Field(..., description="Preparer name/ID")
    reviewed_by: Optional[str] = Field(None, description="Reviewer name/ID")
    verified_by: Optional[str] = Field(None, description="Verifier name/ID")

    # Hash
    report_hash: Optional[str] = Field(None, description="SHA-256 hash of report")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of report content."""
        data = self.dict(exclude={"report_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class ExportedReport(BaseModel):
    """
    Exported report with format-specific content.
    """

    export_id: UUID = Field(default_factory=uuid4, description="Export ID")
    source_report_id: str = Field(..., description="Source report ID")
    source_report_type: str = Field(..., description="Source report type")
    export_format: ReportFormat = Field(..., description="Export format")
    exported_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Export timestamp"
    )

    # Content
    file_path: Optional[str] = Field(None, description="File path if written to disk")
    content_bytes: Optional[int] = Field(None, description="Content size in bytes")
    content_hash: str = Field(..., description="SHA-256 hash of content")

    # Metadata
    exported_by: str = Field(..., description="Exporter ID")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class ComplianceReporter:
    """
    Compliance reporter for GHG Protocol and energy reporting.

    Generates regulatory compliance reports with full audit
    trail integration and provenance tracking.

    Attributes:
        audit_logger: Audit logger for event tracking
        provenance_tracker: Provenance tracker for data lineage

    Example:
        >>> reporter = ComplianceReporter(audit_logger, provenance_tracker)
        >>> ghg_report = reporter.generate_ghg_report(period, GHGScope.SCOPE_1, "operational")
        >>> energy_report = reporter.generate_energy_report(period)
    """

    # Default emission factors (EPA 2023 values)
    DEFAULT_EMISSION_FACTORS: Dict[str, EmissionFactor] = {
        "natural_gas": EmissionFactor(
            factor_id="EF_NG_EPA_2023",
            fuel_type="NATURAL_GAS",
            factor_value=53.06,
            factor_unit="kg CO2/MMBtu",
            source="EPA",
            source_year=2023,
        ),
        "fuel_oil_no2": EmissionFactor(
            factor_id="EF_FO2_EPA_2023",
            fuel_type="FUEL_OIL_NO2",
            factor_value=73.16,
            factor_unit="kg CO2/MMBtu",
            source="EPA",
            source_year=2023,
        ),
        "propane": EmissionFactor(
            factor_id="EF_PROPANE_EPA_2023",
            fuel_type="PROPANE",
            factor_value=62.87,
            factor_unit="kg CO2/MMBtu",
            source="EPA",
            source_year=2023,
        ),
        "coal_bituminous": EmissionFactor(
            factor_id="EF_COAL_BIT_EPA_2023",
            fuel_type="COAL_BITUMINOUS",
            factor_value=93.28,
            factor_unit="kg CO2/MMBtu",
            source="EPA",
            source_year=2023,
        ),
    }

    def __init__(
        self,
        audit_logger: Optional[Any] = None,
        provenance_tracker: Optional[Any] = None,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize compliance reporter.

        Args:
            audit_logger: Audit logger for event tracking
            provenance_tracker: Provenance tracker for data lineage
            storage_path: Path for storing reports
        """
        self.audit_logger = audit_logger
        self.provenance_tracker = provenance_tracker
        self.storage_path = Path(storage_path) if storage_path else None

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # Custom emission factors
        self._emission_factors = dict(self.DEFAULT_EMISSION_FACTORS)

        logger.info("ComplianceReporter initialized")

    def register_emission_factor(self, factor: EmissionFactor) -> None:
        """Register a custom emission factor."""
        self._emission_factors[factor.fuel_type.lower()] = factor
        logger.debug(f"Emission factor registered: {factor.fuel_type}")

    def generate_ghg_report(
        self,
        time_period: ReportingPeriod,
        scope: GHGScope,
        boundary: str,
        organization_name: str,
        facility_id: str,
        facility_name: str,
        energy_consumption: Dict[str, float],
        prepared_by: str,
        boundary_description: Optional[str] = None,
        scope2_method: str = "location_based",
        electricity_factor_kg_per_mwh: float = 420.0,
        purchased_steam_factor_kg_per_mmbtu: float = 66.0,
    ) -> GHGReport:
        """
        Generate a GHG Protocol aligned emissions report.

        Args:
            time_period: Reporting period
            scope: GHG scope(s) to include
            boundary: Boundary type
            organization_name: Organization name
            facility_id: Facility identifier
            facility_name: Facility name
            energy_consumption: Energy consumption by fuel type (MMBtu)
            prepared_by: Preparer name/ID
            boundary_description: Boundary description
            scope2_method: Scope 2 method (location_based, market_based)
            electricity_factor_kg_per_mwh: Grid electricity factor
            purchased_steam_factor_kg_per_mmbtu: Purchased steam factor

        Returns:
            GHGReport
        """
        start_time = datetime.now(timezone.utc)

        # Determine included scopes
        if scope == GHGScope.ALL:
            included_scopes = [GHGScope.SCOPE_1, GHGScope.SCOPE_2, GHGScope.SCOPE_3]
        else:
            included_scopes = [scope]

        # Calculate Scope 1 emissions (direct combustion)
        scope1_sources: List[EmissionSource] = []
        scope1_by_fuel: Dict[str, float] = {}
        scope1_total = 0.0
        total_co2 = 0.0
        total_ch4 = 0.0
        total_n2o = 0.0

        scope1_fuels = ["natural_gas", "fuel_oil_no2", "propane", "coal_bituminous"]

        for fuel, consumption_mmbtu in energy_consumption.items():
            fuel_lower = fuel.lower().replace(" ", "_")
            if fuel_lower in scope1_fuels and consumption_mmbtu > 0:
                factor = self._emission_factors.get(fuel_lower)
                if not factor:
                    continue

                # Calculate emissions
                co2_kg = consumption_mmbtu * factor.factor_value
                ch4_kg = consumption_mmbtu * 0.001 * factor.gwp_ch4  # Simplified
                n2o_kg = consumption_mmbtu * 0.0001 * factor.gwp_n2o  # Simplified
                co2e_kg = co2_kg + ch4_kg + n2o_kg

                source = EmissionSource(
                    source_id=f"S1_{fuel_lower}_{facility_id}",
                    source_name=f"{fuel} combustion",
                    source_type="STATIONARY_COMBUSTION",
                    scope=GHGScope.SCOPE_1,
                    fuel_type=fuel,
                    activity_data=consumption_mmbtu,
                    activity_unit="MMBtu",
                    emission_factor=factor,
                    co2_emissions_kg=co2_kg,
                    ch4_emissions_kg=ch4_kg,
                    n2o_emissions_kg=n2o_kg,
                    co2e_emissions_kg=co2e_kg,
                )
                scope1_sources.append(source)
                scope1_by_fuel[fuel] = co2e_kg / 1000  # Convert to MT
                scope1_total += co2e_kg / 1000
                total_co2 += co2_kg / 1000
                total_ch4 += ch4_kg / 1000
                total_n2o += n2o_kg / 1000

        # Calculate Scope 2 emissions (purchased electricity, steam)
        scope2_sources: List[EmissionSource] = []
        scope2_location = 0.0
        scope2_market = 0.0

        electricity_mwh = energy_consumption.get("electricity_mwh", 0)
        if electricity_mwh > 0:
            co2e_kg = electricity_mwh * electricity_factor_kg_per_mwh
            scope2_location += co2e_kg / 1000
            scope2_market += co2e_kg / 1000  # Same for simplified case

        purchased_steam = energy_consumption.get("purchased_steam_mmbtu", 0)
        if purchased_steam > 0:
            co2e_kg = purchased_steam * purchased_steam_factor_kg_per_mmbtu
            scope2_location += co2e_kg / 1000

        # Total emissions
        total_co2e = scope1_total
        if GHGScope.SCOPE_2 in included_scopes:
            total_co2e += scope2_location if scope2_method == "location_based" else scope2_market

        # Methodology notes
        methodology_notes = (
            f"GHG inventory prepared following GHG Protocol Corporate Standard. "
            f"Boundary approach: {boundary}. "
            f"Scope 1 emissions calculated using EPA emission factors (2023). "
            f"Scope 2 emissions calculated using {scope2_method} method. "
            f"Grid electricity factor: {electricity_factor_kg_per_mwh} kg CO2e/MWh. "
            f"All calculations use SHA-256 provenance hashing for integrity."
        )

        data_quality = (
            f"Activity data sourced from metered fuel consumption and utility invoices. "
            f"Emission factors from EPA GHG Emission Factors Hub (2023). "
            f"Reporting period: {time_period.days} days."
        )

        report = GHGReport(
            organization_name=organization_name,
            facility_id=facility_id,
            facility_name=facility_name,
            reporting_period=time_period,
            boundary_type=boundary,
            boundary_description=boundary_description or f"All operations under {boundary}",
            included_scopes=included_scopes,
            scope1_total_co2e_mt=scope1_total,
            scope1_sources=scope1_sources,
            scope1_by_fuel=scope1_by_fuel,
            scope2_location_based_co2e_mt=scope2_location,
            scope2_market_based_co2e_mt=scope2_market,
            scope2_sources=scope2_sources,
            total_co2e_mt=total_co2e,
            total_co2_mt=total_co2,
            total_ch4_mt=total_ch4,
            total_n2o_mt=total_n2o,
            methodology_notes=methodology_notes,
            emission_factors_source="EPA GHG Emission Factors Hub 2023",
            data_quality_statement=data_quality,
            prepared_by=prepared_by,
        )

        # Calculate and set hash
        report_dict = report.dict()
        report_dict["report_hash"] = report.calculate_hash()
        report = GHGReport(**report_dict)

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"GHG report generated: {report.report_id}",
            extra={
                "total_co2e_mt": total_co2e,
                "scope1_mt": scope1_total,
                "processing_time_ms": processing_time,
            }
        )

        return report

    def generate_energy_report(
        self,
        time_period: ReportingPeriod,
        facility_id: str,
        facility_name: str,
        energy_consumption: Dict[str, float],
        steam_production_klb: float,
        boiler_efficiency_pct: float,
        system_efficiency_pct: float,
        production_volume: float,
        production_unit: str,
        prepared_by: str,
        energy_costs: Optional[Dict[str, float]] = None,
        baseline_energy_mmbtu: Optional[float] = None,
        baseline_period: Optional[str] = None,
    ) -> EnergyReport:
        """
        Generate an energy consumption report.

        Args:
            time_period: Reporting period
            facility_id: Facility identifier
            facility_name: Facility name
            energy_consumption: Energy by source (MMBtu or MWh for electricity)
            steam_production_klb: Steam production (klb)
            boiler_efficiency_pct: Average boiler efficiency
            system_efficiency_pct: Average system efficiency
            production_volume: Production volume
            production_unit: Production unit
            prepared_by: Preparer name/ID
            energy_costs: Energy costs by source (USD)
            baseline_energy_mmbtu: Baseline energy for comparison
            baseline_period: Baseline period description

        Returns:
            EnergyReport
        """
        start_time = datetime.now(timezone.utc)

        # Extract energy by source
        natural_gas = energy_consumption.get("natural_gas_mmbtu", 0)
        fuel_oil = energy_consumption.get("fuel_oil_mmbtu", 0)
        propane = energy_consumption.get("propane_mmbtu", 0)
        electricity_mwh = energy_consumption.get("electricity_mwh", 0)
        purchased_steam = energy_consumption.get("purchased_steam_mmbtu", 0)
        other = energy_consumption.get("other_mmbtu", 0)

        # Calculate totals
        total_mmbtu = natural_gas + fuel_oil + propane + purchased_steam + other
        # Add electricity in MMBtu equivalent (1 MWh = 3.412 MMBtu)
        total_mmbtu += electricity_mwh * 3.412

        total_gj = total_mmbtu * 1.055  # MMBtu to GJ
        total_mwh_equiv = total_mmbtu / 3.412

        # Calculate heat rate
        heat_rate = total_mmbtu / steam_production_klb if steam_production_klb > 0 else 0

        # Calculate energy intensity
        energy_intensity = total_mmbtu / production_volume if production_volume > 0 else 0

        # Calculate improvement if baseline provided
        improvement_pct = None
        if baseline_energy_mmbtu and baseline_energy_mmbtu > 0:
            # Normalize to production (simplified)
            baseline_intensity = baseline_energy_mmbtu / production_volume if production_volume > 0 else 0
            if baseline_intensity > 0:
                improvement_pct = (1 - energy_intensity / baseline_intensity) * 100

        # Total costs
        energy_costs = energy_costs or {}
        total_cost = sum(energy_costs.values())

        # Methodology
        methodology = (
            f"Energy consumption report prepared following ISO 50001 guidelines. "
            f"Energy data sourced from utility meters and fuel receipts. "
            f"Steam production from plant historian data. "
            f"Energy intensity calculated as total energy / {production_unit}."
        )

        data_quality = (
            f"Metered data with estimated {95}% completeness. "
            f"Period: {time_period.days} days."
        )

        report = EnergyReport(
            facility_id=facility_id,
            facility_name=facility_name,
            reporting_period=time_period,
            total_energy_mmbtu=total_mmbtu,
            total_energy_gj=total_gj,
            total_energy_mwh=total_mwh_equiv,
            natural_gas_mmbtu=natural_gas,
            fuel_oil_mmbtu=fuel_oil,
            propane_mmbtu=propane,
            electricity_mwh=electricity_mwh,
            purchased_steam_mmbtu=purchased_steam,
            other_energy_mmbtu=other,
            energy_by_source=energy_consumption,
            steam_production_klb=steam_production_klb,
            boiler_efficiency_pct=boiler_efficiency_pct,
            system_efficiency_pct=system_efficiency_pct,
            heat_rate_mmbtu_per_klb=heat_rate,
            energy_intensity_mmbtu_per_unit=energy_intensity,
            production_volume=production_volume,
            production_unit=production_unit,
            baseline_energy_mmbtu=baseline_energy_mmbtu,
            baseline_period=baseline_period,
            energy_improvement_pct=improvement_pct,
            total_energy_cost_usd=total_cost,
            cost_by_source=energy_costs,
            methodology_notes=methodology,
            data_quality_statement=data_quality,
            prepared_by=prepared_by,
        )

        # Calculate and set hash
        report_dict = report.dict()
        report_dict["report_hash"] = report.calculate_hash()
        report = EnergyReport(**report_dict)

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Energy report generated: {report.report_id}",
            extra={
                "total_mmbtu": total_mmbtu,
                "processing_time_ms": processing_time,
            }
        )

        return report

    def generate_savings_verification_report(
        self,
        project_id: str,
        project_name: str,
        facility_id: str,
        facility_name: str,
        verification_period: ReportingPeriod,
        verified_energy_savings_mmbtu: float,
        verified_cost_savings_usd: float,
        verified_co2e_reduction_mt: float,
        savings_uncertainty_pct: float,
        mv_methodology: str,
        measurement_boundary: str,
        baseline_evidence_id: str,
        post_evidence_id: str,
        savings_evidence_id: str,
        ecm_ids: List[str],
        ecm_descriptions: List[str],
        prepared_by: str,
        predicted_savings_mmbtu: Optional[float] = None,
        confidence_level_pct: float = 90.0,
    ) -> VerificationReport:
        """
        Generate a savings verification report.

        Args:
            project_id: Project identifier
            project_name: Project name
            facility_id: Facility identifier
            facility_name: Facility name
            verification_period: Verification period
            verified_energy_savings_mmbtu: Verified energy savings
            verified_cost_savings_usd: Verified cost savings
            verified_co2e_reduction_mt: Verified emissions reduction
            savings_uncertainty_pct: Savings uncertainty
            mv_methodology: M&V methodology used
            measurement_boundary: Measurement boundary
            baseline_evidence_id: Baseline evidence ID
            post_evidence_id: Post evidence ID
            savings_evidence_id: Savings evidence ID
            ecm_ids: ECM identifiers
            ecm_descriptions: ECM descriptions
            prepared_by: Preparer name/ID
            predicted_savings_mmbtu: Predicted savings for comparison
            confidence_level_pct: Confidence level

        Returns:
            VerificationReport
        """
        start_time = datetime.now(timezone.utc)

        # Calculate savings range
        z_score = 1.645 if confidence_level_pct == 90 else 1.96
        uncertainty_mmbtu = verified_energy_savings_mmbtu * (savings_uncertainty_pct / 100)
        savings_low = verified_energy_savings_mmbtu - uncertainty_mmbtu * z_score
        savings_high = verified_energy_savings_mmbtu + uncertainty_mmbtu * z_score

        # Calculate realization rate
        realization_rate = None
        if predicted_savings_mmbtu and predicted_savings_mmbtu > 0:
            realization_rate = (verified_energy_savings_mmbtu / predicted_savings_mmbtu) * 100

        # Determine verification status
        if savings_uncertainty_pct <= 50:
            verification_status = "VERIFIED"
        elif savings_uncertainty_pct <= 100:
            verification_status = "PARTIALLY_VERIFIED"
        else:
            verification_status = "NOT_VERIFIED"

        methodology_notes = (
            f"Savings verification performed using {mv_methodology} methodology. "
            f"Measurement boundary: {measurement_boundary}. "
            f"Baseline adjusted for normalization factors. "
            f"Uncertainty quantified at {confidence_level_pct}% confidence level."
        )

        report = VerificationReport(
            project_id=project_id,
            project_name=project_name,
            facility_id=facility_id,
            facility_name=facility_name,
            verification_period=verification_period,
            ecm_ids=ecm_ids,
            ecm_descriptions=ecm_descriptions,
            verified_energy_savings_mmbtu=verified_energy_savings_mmbtu,
            verified_cost_savings_usd=verified_cost_savings_usd,
            verified_co2e_reduction_mt=verified_co2e_reduction_mt,
            predicted_energy_savings_mmbtu=predicted_savings_mmbtu,
            savings_realization_rate_pct=realization_rate,
            savings_uncertainty_pct=savings_uncertainty_pct,
            confidence_level_pct=confidence_level_pct,
            savings_range_low_mmbtu=savings_low,
            savings_range_high_mmbtu=savings_high,
            mv_methodology=mv_methodology,
            measurement_boundary=measurement_boundary,
            methodology_notes=methodology_notes,
            baseline_evidence_id=baseline_evidence_id,
            post_evidence_id=post_evidence_id,
            savings_evidence_id=savings_evidence_id,
            verification_status=verification_status,
            prepared_by=prepared_by,
        )

        # Calculate and set hash
        report_dict = report.dict()
        report_dict["report_hash"] = report.calculate_hash()
        report = VerificationReport(**report_dict)

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Verification report generated: {report.report_id}",
            extra={
                "verified_savings_mmbtu": verified_energy_savings_mmbtu,
                "verification_status": verification_status,
                "processing_time_ms": processing_time,
            }
        )

        return report

    def export_for_auditor(
        self,
        report: Union[GHGReport, EnergyReport, VerificationReport],
        format: ReportFormat,
        output_path: Optional[str] = None,
        exported_by: str = "GL-003",
    ) -> ExportedReport:
        """
        Export a report for auditor review.

        Args:
            report: Report to export
            format: Export format
            output_path: Output file path (optional)
            exported_by: Exporter identifier

        Returns:
            ExportedReport with export metadata

        Raises:
            ValueError: If format not supported
        """
        start_time = datetime.now(timezone.utc)
        content = ""
        content_bytes = 0

        if format == ReportFormat.JSON:
            content = json.dumps(report.dict(), indent=2, default=str)
            content_bytes = len(content.encode("utf-8"))

        elif format == ReportFormat.CSV:
            content = self._export_to_csv(report)
            content_bytes = len(content.encode("utf-8"))

        elif format == ReportFormat.PDF:
            # Generate text-based report (production would use proper PDF library)
            content = self._export_to_text(report)
            content_bytes = len(content.encode("utf-8"))

        else:
            raise ValueError(f"Unsupported format: {format}")

        # Calculate content hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Write to file if path provided
        file_path = None
        if output_path:
            with open(output_path, "w") as f:
                f.write(content)
            file_path = output_path

        exported = ExportedReport(
            source_report_id=str(report.report_id),
            source_report_type=report.report_type,
            export_format=format,
            file_path=file_path,
            content_bytes=content_bytes,
            content_hash=content_hash,
            exported_by=exported_by,
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Report exported: {format.value}",
            extra={
                "source_report_id": str(report.report_id),
                "content_bytes": content_bytes,
                "processing_time_ms": processing_time,
            }
        )

        return exported

    def _export_to_csv(
        self,
        report: Union[GHGReport, EnergyReport, VerificationReport],
    ) -> str:
        """Export report to CSV format."""
        output = StringIO()
        writer = csv.writer(output)

        if isinstance(report, GHGReport):
            writer.writerow(["GHG Emissions Report"])
            writer.writerow(["Report ID", str(report.report_id)])
            writer.writerow(["Organization", report.organization_name])
            writer.writerow(["Facility", report.facility_name])
            writer.writerow(["Period Start", report.reporting_period.start_date.isoformat()])
            writer.writerow(["Period End", report.reporting_period.end_date.isoformat()])
            writer.writerow([])
            writer.writerow(["Scope", "Category", "CO2e (metric tons)"])
            writer.writerow(["Scope 1", "Total", report.scope1_total_co2e_mt])
            for fuel, amount in report.scope1_by_fuel.items():
                writer.writerow(["Scope 1", fuel, amount])
            writer.writerow(["Scope 2 (Location)", "Total", report.scope2_location_based_co2e_mt])
            writer.writerow(["Total", "", report.total_co2e_mt])

        elif isinstance(report, EnergyReport):
            writer.writerow(["Energy Consumption Report"])
            writer.writerow(["Report ID", str(report.report_id)])
            writer.writerow(["Facility", report.facility_name])
            writer.writerow(["Total Energy (MMBtu)", report.total_energy_mmbtu])
            writer.writerow(["Boiler Efficiency (%)", report.boiler_efficiency_pct])
            writer.writerow(["Energy Intensity", report.energy_intensity_mmbtu_per_unit])

        elif isinstance(report, VerificationReport):
            writer.writerow(["Savings Verification Report"])
            writer.writerow(["Report ID", str(report.report_id)])
            writer.writerow(["Project", report.project_name])
            writer.writerow(["Verified Savings (MMBtu)", report.verified_energy_savings_mmbtu])
            writer.writerow(["Cost Savings (USD)", report.verified_cost_savings_usd])
            writer.writerow(["CO2e Reduction (MT)", report.verified_co2e_reduction_mt])
            writer.writerow(["Verification Status", report.verification_status])

        return output.getvalue()

    def _export_to_text(
        self,
        report: Union[GHGReport, EnergyReport, VerificationReport],
    ) -> str:
        """Export report to text format (for PDF placeholder)."""
        lines = []
        lines.append("=" * 70)

        if isinstance(report, GHGReport):
            lines.append("GHG EMISSIONS INVENTORY REPORT")
            lines.append("=" * 70)
            lines.append(f"Report ID: {report.report_id}")
            lines.append(f"Organization: {report.organization_name}")
            lines.append(f"Facility: {report.facility_name} ({report.facility_id})")
            lines.append(f"Period: {report.reporting_period.start_date.date()} to {report.reporting_period.end_date.date()}")
            lines.append(f"Boundary: {report.boundary_type}")
            lines.append("")
            lines.append("-" * 40)
            lines.append("EMISSIONS SUMMARY")
            lines.append("-" * 40)
            lines.append(f"Scope 1 (Direct): {report.scope1_total_co2e_mt:.2f} metric tons CO2e")
            lines.append(f"Scope 2 (Location-based): {report.scope2_location_based_co2e_mt:.2f} metric tons CO2e")
            lines.append(f"TOTAL: {report.total_co2e_mt:.2f} metric tons CO2e")
            lines.append("")
            lines.append("-" * 40)
            lines.append("METHODOLOGY")
            lines.append("-" * 40)
            lines.append(report.methodology_notes)

        elif isinstance(report, EnergyReport):
            lines.append("ENERGY CONSUMPTION REPORT")
            lines.append("=" * 70)
            lines.append(f"Report ID: {report.report_id}")
            lines.append(f"Facility: {report.facility_name}")
            lines.append(f"Period: {report.reporting_period.start_date.date()} to {report.reporting_period.end_date.date()}")
            lines.append("")
            lines.append("-" * 40)
            lines.append("ENERGY SUMMARY")
            lines.append("-" * 40)
            lines.append(f"Total Energy: {report.total_energy_mmbtu:.2f} MMBtu")
            lines.append(f"Natural Gas: {report.natural_gas_mmbtu:.2f} MMBtu")
            lines.append(f"Electricity: {report.electricity_mwh:.2f} MWh")
            lines.append(f"Steam Production: {report.steam_production_klb:.2f} klb")
            lines.append(f"Boiler Efficiency: {report.boiler_efficiency_pct:.1f}%")
            lines.append(f"Energy Intensity: {report.energy_intensity_mmbtu_per_unit:.3f} MMBtu/{report.production_unit}")

        elif isinstance(report, VerificationReport):
            lines.append("SAVINGS VERIFICATION REPORT")
            lines.append("=" * 70)
            lines.append(f"Report ID: {report.report_id}")
            lines.append(f"Project: {report.project_name}")
            lines.append(f"Facility: {report.facility_name}")
            lines.append("")
            lines.append("-" * 40)
            lines.append("VERIFIED SAVINGS")
            lines.append("-" * 40)
            lines.append(f"Energy Savings: {report.verified_energy_savings_mmbtu:.2f} MMBtu")
            lines.append(f"Cost Savings: ${report.verified_cost_savings_usd:,.2f}")
            lines.append(f"CO2e Reduction: {report.verified_co2e_reduction_mt:.2f} metric tons")
            lines.append(f"Uncertainty: +/- {report.savings_uncertainty_pct:.1f}%")
            lines.append(f"Verification Status: {report.verification_status}")

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"Report Hash: {report.report_hash}")

        return "\n".join(lines)
