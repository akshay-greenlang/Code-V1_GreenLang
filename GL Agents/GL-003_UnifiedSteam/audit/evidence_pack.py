"""
Evidence Packager for GL-003 UnifiedSteam SteamSystemOptimizer

This module implements M&V (Measurement & Verification) evidence packaging
following IPMVP (International Performance Measurement and Verification
Protocol) guidelines for steam system energy savings verification.

Key Features:
    - Baseline period evidence packaging
    - Post-implementation evidence packaging
    - Savings evidence calculation and packaging
    - M&V report generation with methodology notes
    - Digital signatures for evidence integrity
    - Support for IPMVP Option A, B, C, and D methodologies

Reference Standards:
    - IPMVP Core Concepts (EVO 10000)
    - ASHRAE Guideline 14
    - FEMP M&V Guidelines

Example:
    >>> packager = EvidencePackager(storage_path="/audit/evidence")
    >>> baseline = packager.create_baseline_evidence(
    ...     baseline_period=period,
    ...     measurements=measurements,
    ...     calculations=calculations
    ... )
    >>> post = packager.create_post_implementation_evidence(
    ...     post_period=period,
    ...     measurements=measurements,
    ...     calculations=calculations
    ... )
    >>> savings = packager.create_savings_evidence(baseline, post, factors)
    >>> report = packager.package_mv_report(savings, methodology_notes)

Author: GreenLang Steam Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import zipfile
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class MVMethodology(str, Enum):
    """IPMVP M&V Methodology Options."""

    OPTION_A = "OPTION_A"  # Retrofit Isolation: Key Parameter Measurement
    OPTION_B = "OPTION_B"  # Retrofit Isolation: All Parameter Measurement
    OPTION_C = "OPTION_C"  # Whole Facility
    OPTION_D = "OPTION_D"  # Calibrated Simulation


class NormalizationFactor(BaseModel):
    """
    Factor used to normalize baseline and post-implementation data.

    Common factors include production volume, heating/cooling degree days,
    occupancy, and operating hours.
    """

    factor_id: str = Field(..., description="Unique factor identifier")
    factor_name: str = Field(..., description="Human-readable factor name")
    factor_type: str = Field(
        ..., description="Type (PRODUCTION, WEATHER, OCCUPANCY, HOURS, CUSTOM)"
    )
    unit: str = Field(..., description="Unit of measurement")

    # Baseline values
    baseline_value: float = Field(..., description="Average baseline period value")
    baseline_data_points: int = Field(0, ge=0, description="Number of baseline data points")

    # Post-implementation values
    post_value: float = Field(..., description="Average post-implementation value")
    post_data_points: int = Field(0, ge=0, description="Number of post data points")

    # Normalization
    normalization_coefficient: Optional[float] = Field(
        None, description="Regression coefficient if applicable"
    )
    r_squared: Optional[float] = Field(
        None, ge=0, le=1, description="R-squared of regression"
    )

    class Config:
        frozen = True


class MeasurementRecord(BaseModel):
    """Record of a measurement for M&V purposes."""

    measurement_id: str = Field(..., description="Unique measurement ID")
    timestamp: datetime = Field(..., description="Measurement timestamp")
    tag_id: str = Field(..., description="Measurement point tag")
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Engineering unit")
    quality: str = Field(default="GOOD", description="Data quality flag")
    source: str = Field(..., description="Data source (HISTORIAN, METER, MANUAL)")

    # Calibration info
    calibration_date: Optional[datetime] = Field(
        None, description="Last calibration date"
    )
    accuracy_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Measurement accuracy %"
    )

    class Config:
        frozen = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class CalculationRecord(BaseModel):
    """Record of a calculation for M&V purposes."""

    calculation_id: str = Field(..., description="Unique calculation ID")
    calculation_type: str = Field(..., description="Type of calculation")
    timestamp: datetime = Field(..., description="Calculation timestamp")

    # Inputs and outputs
    inputs: Dict[str, Any] = Field(..., description="Input values")
    outputs: Dict[str, Any] = Field(..., description="Output values")

    # Provenance
    formula_id: str = Field(..., description="Formula identifier")
    formula_version: str = Field(..., description="Formula version")
    inputs_hash: str = Field(..., description="SHA-256 hash of inputs")
    outputs_hash: str = Field(..., description="SHA-256 hash of outputs")

    class Config:
        frozen = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class EvidencePeriod(BaseModel):
    """Time period definition for M&V evidence."""

    start_date: datetime = Field(..., description="Period start date")
    end_date: datetime = Field(..., description="Period end date")
    duration_days: int = Field(..., ge=1, description="Period duration in days")
    description: Optional[str] = Field(None, description="Period description")

    @validator("end_date")
    def end_after_start(cls, v, values):
        """Validate end_date is after start_date."""
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    @validator("duration_days", always=True)
    def calculate_duration(cls, v, values):
        """Calculate or validate duration."""
        if "start_date" in values and "end_date" in values:
            actual = (values["end_date"] - values["start_date"]).days
            if v != actual:
                return actual
        return v

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BaselineEvidence(BaseModel):
    """
    Evidence package for the baseline period.

    Contains all measurements, calculations, and metadata
    for the pre-implementation baseline period.
    """

    evidence_id: UUID = Field(default_factory=uuid4, description="Unique evidence ID")
    evidence_type: str = Field(default="BASELINE", description="Evidence type")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )

    # Period
    baseline_period: EvidencePeriod = Field(..., description="Baseline period definition")

    # Energy consumption
    total_energy_mmbtu: float = Field(..., description="Total energy consumption (MMBtu)")
    energy_by_fuel: Dict[str, float] = Field(
        default_factory=dict, description="Energy by fuel type (MMBtu)"
    )

    # Steam production
    total_steam_production_klb: float = Field(
        ..., description="Total steam production (klb)"
    )
    steam_by_header: Dict[str, float] = Field(
        default_factory=dict, description="Steam by header (klb)"
    )

    # Efficiency metrics
    average_boiler_efficiency: float = Field(
        ..., ge=0, le=100, description="Average boiler efficiency (%)"
    )
    average_system_efficiency: float = Field(
        ..., ge=0, le=100, description="Average system efficiency (%)"
    )

    # Measurements
    measurements: List[MeasurementRecord] = Field(
        default_factory=list, description="Measurement records"
    )
    measurement_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Measurement statistics"
    )

    # Calculations
    calculations: List[CalculationRecord] = Field(
        default_factory=list, description="Calculation records"
    )

    # Normalization factors
    normalization_factors: Dict[str, float] = Field(
        default_factory=dict, description="Average normalization factor values"
    )

    # Data quality
    data_completeness_pct: float = Field(
        100.0, ge=0, le=100, description="Data completeness %"
    )
    data_quality_notes: List[str] = Field(
        default_factory=list, description="Data quality notes"
    )

    # Hash for integrity
    evidence_hash: Optional[str] = Field(None, description="SHA-256 hash of evidence")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of evidence content."""
        data = self.dict(exclude={"evidence_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class PostEvidence(BaseModel):
    """
    Evidence package for the post-implementation period.

    Contains all measurements, calculations, and metadata
    for the post-implementation reporting period.
    """

    evidence_id: UUID = Field(default_factory=uuid4, description="Unique evidence ID")
    evidence_type: str = Field(default="POST_IMPLEMENTATION", description="Evidence type")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )

    # Period
    post_period: EvidencePeriod = Field(..., description="Post period definition")

    # Implementation details
    implementation_date: datetime = Field(
        ..., description="Date ECM was implemented"
    )
    ecm_description: str = Field(..., description="Description of ECM implemented")
    ecm_ids: List[str] = Field(
        default_factory=list, description="ECM identifiers"
    )

    # Energy consumption
    total_energy_mmbtu: float = Field(..., description="Total energy consumption (MMBtu)")
    energy_by_fuel: Dict[str, float] = Field(
        default_factory=dict, description="Energy by fuel type (MMBtu)"
    )

    # Steam production
    total_steam_production_klb: float = Field(
        ..., description="Total steam production (klb)"
    )
    steam_by_header: Dict[str, float] = Field(
        default_factory=dict, description="Steam by header (klb)"
    )

    # Efficiency metrics
    average_boiler_efficiency: float = Field(
        ..., ge=0, le=100, description="Average boiler efficiency (%)"
    )
    average_system_efficiency: float = Field(
        ..., ge=0, le=100, description="Average system efficiency (%)"
    )

    # Measurements
    measurements: List[MeasurementRecord] = Field(
        default_factory=list, description="Measurement records"
    )
    measurement_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Measurement statistics"
    )

    # Calculations
    calculations: List[CalculationRecord] = Field(
        default_factory=list, description="Calculation records"
    )

    # Normalization factors
    normalization_factors: Dict[str, float] = Field(
        default_factory=dict, description="Average normalization factor values"
    )

    # Data quality
    data_completeness_pct: float = Field(
        100.0, ge=0, le=100, description="Data completeness %"
    )
    data_quality_notes: List[str] = Field(
        default_factory=list, description="Data quality notes"
    )

    # Hash for integrity
    evidence_hash: Optional[str] = Field(None, description="SHA-256 hash of evidence")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of evidence content."""
        data = self.dict(exclude={"evidence_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class SavingsEvidence(BaseModel):
    """
    Evidence package for calculated energy savings.

    Contains baseline-adjusted savings calculations with
    normalization factor adjustments.
    """

    evidence_id: UUID = Field(default_factory=uuid4, description="Unique evidence ID")
    evidence_type: str = Field(default="SAVINGS", description="Evidence type")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )

    # References
    baseline_evidence_id: str = Field(..., description="Baseline evidence ID")
    post_evidence_id: str = Field(..., description="Post evidence ID")

    # Methodology
    methodology: MVMethodology = Field(..., description="IPMVP methodology used")
    measurement_boundary: str = Field(..., description="Measurement boundary description")

    # Baseline-adjusted energy
    baseline_adjusted_energy_mmbtu: float = Field(
        ..., description="Baseline energy adjusted to post-period conditions (MMBtu)"
    )

    # Actual post-period energy
    post_period_energy_mmbtu: float = Field(
        ..., description="Actual post-period energy (MMBtu)"
    )

    # Calculated savings
    gross_savings_mmbtu: float = Field(
        ..., description="Gross energy savings (MMBtu)"
    )
    gross_savings_pct: float = Field(
        ..., description="Gross savings as percentage of baseline"
    )

    # Adjustments
    routine_adjustments_mmbtu: float = Field(
        0.0, description="Routine adjustments (MMBtu)"
    )
    non_routine_adjustments_mmbtu: float = Field(
        0.0, description="Non-routine adjustments (MMBtu)"
    )

    # Net savings
    net_savings_mmbtu: float = Field(..., description="Net energy savings (MMBtu)")
    net_savings_pct: float = Field(
        ..., description="Net savings as percentage of baseline"
    )

    # Uncertainty
    savings_uncertainty_pct: float = Field(
        ..., ge=0, le=100, description="Savings uncertainty (%)"
    )
    confidence_level_pct: float = Field(
        90.0, ge=0, le=100, description="Confidence level (%)"
    )
    savings_lower_bound_mmbtu: float = Field(
        ..., description="Lower bound of savings (MMBtu)"
    )
    savings_upper_bound_mmbtu: float = Field(
        ..., description="Upper bound of savings (MMBtu)"
    )

    # Cost savings
    energy_cost_savings_usd: float = Field(
        ..., description="Energy cost savings (USD)"
    )
    avoided_costs_usd: float = Field(
        0.0, description="Other avoided costs (USD)"
    )
    total_cost_savings_usd: float = Field(
        ..., description="Total cost savings (USD)"
    )

    # Emissions reduction
    co2e_reduction_metric_tons: float = Field(
        ..., description="CO2e reduction (metric tons)"
    )

    # Normalization factors applied
    normalization_factors: List[NormalizationFactor] = Field(
        default_factory=list, description="Normalization factors applied"
    )

    # Calculation details
    calculation_methodology: str = Field(
        ..., description="Description of savings calculation methodology"
    )
    adjustments_methodology: Optional[str] = Field(
        None, description="Description of adjustments methodology"
    )

    # Evidence hashes
    baseline_hash: str = Field(..., description="Hash of baseline evidence")
    post_hash: str = Field(..., description="Hash of post evidence")
    evidence_hash: Optional[str] = Field(None, description="SHA-256 hash of this evidence")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of evidence content."""
        data = self.dict(exclude={"evidence_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class MVReport(BaseModel):
    """
    Complete M&V Report for regulatory submission.

    Contains all evidence packages with methodology documentation
    following IPMVP guidelines.
    """

    report_id: UUID = Field(default_factory=uuid4, description="Unique report ID")
    report_version: str = Field(default="1.0.0", description="Report version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report creation timestamp"
    )

    # Project information
    project_id: str = Field(..., description="Project identifier")
    project_name: str = Field(..., description="Project name")
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")

    # Report period
    reporting_period: EvidencePeriod = Field(..., description="Reporting period")

    # Methodology
    methodology: MVMethodology = Field(..., description="IPMVP methodology")
    methodology_notes: str = Field(..., description="Methodology documentation")
    measurement_boundary: str = Field(..., description="Measurement boundary")

    # Evidence packages
    baseline_evidence: BaselineEvidence = Field(..., description="Baseline evidence")
    post_evidence: PostEvidence = Field(..., description="Post-implementation evidence")
    savings_evidence: SavingsEvidence = Field(..., description="Savings evidence")

    # Summary results
    total_savings_mmbtu: float = Field(..., description="Total energy savings (MMBtu)")
    total_cost_savings_usd: float = Field(..., description="Total cost savings (USD)")
    total_co2e_reduction_mt: float = Field(
        ..., description="Total CO2e reduction (metric tons)"
    )

    # Data quality statement
    data_quality_statement: str = Field(
        ..., description="Data quality and completeness statement"
    )

    # Reviewer information
    prepared_by: str = Field(..., description="Preparer name/ID")
    reviewed_by: Optional[str] = Field(None, description="Reviewer name/ID")
    approved_by: Optional[str] = Field(None, description="Approver name/ID")

    # Signatures
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


class SignedEvidence(BaseModel):
    """
    Digitally signed evidence package.

    Contains the evidence with HMAC signature for integrity verification.
    """

    evidence_id: UUID = Field(default_factory=uuid4, description="Unique ID")
    evidence_type: str = Field(..., description="Type of evidence contained")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Signing timestamp"
    )

    # Original evidence hash
    evidence_hash: str = Field(..., description="SHA-256 hash of original evidence")

    # Signature
    signature: str = Field(..., description="HMAC-SHA256 signature")
    signature_algorithm: str = Field(
        default="HMAC-SHA256", description="Signature algorithm"
    )

    # Signer information
    signer_id: str = Field(..., description="Signer identifier")
    signer_role: str = Field(..., description="Signer role")

    # Verification status
    is_verified: bool = Field(False, description="Signature verified")
    verification_timestamp: Optional[datetime] = Field(
        None, description="When signature was verified"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class EvidencePackager:
    """
    M&V Evidence Packager for steam system savings verification.

    Creates IPMVP-compliant evidence packages for baseline periods,
    post-implementation periods, and calculated savings.

    Attributes:
        storage_path: Path for storing evidence packages

    Example:
        >>> packager = EvidencePackager(storage_path="/audit/evidence")
        >>> baseline = packager.create_baseline_evidence(...)
        >>> post = packager.create_post_implementation_evidence(...)
        >>> savings = packager.create_savings_evidence(baseline, post, factors)
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        signing_key: Optional[str] = None,
    ):
        """
        Initialize evidence packager.

        Args:
            storage_path: Path for storing evidence packages
            signing_key: Key for HMAC signing (keep secure!)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._signing_key = signing_key

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "EvidencePackager initialized",
            extra={"storage_path": str(self.storage_path)}
        )

    def create_baseline_evidence(
        self,
        baseline_period: EvidencePeriod,
        measurements: List[MeasurementRecord],
        calculations: List[CalculationRecord],
        total_energy_mmbtu: float,
        total_steam_production_klb: float,
        average_boiler_efficiency: float,
        average_system_efficiency: float,
        energy_by_fuel: Optional[Dict[str, float]] = None,
        steam_by_header: Optional[Dict[str, float]] = None,
        normalization_factors: Optional[Dict[str, float]] = None,
        data_quality_notes: Optional[List[str]] = None,
    ) -> BaselineEvidence:
        """
        Create baseline period evidence package.

        Args:
            baseline_period: Period definition
            measurements: List of measurement records
            calculations: List of calculation records
            total_energy_mmbtu: Total energy consumption
            total_steam_production_klb: Total steam production
            average_boiler_efficiency: Average boiler efficiency
            average_system_efficiency: Average system efficiency
            energy_by_fuel: Energy breakdown by fuel type
            steam_by_header: Steam breakdown by header
            normalization_factors: Normalization factor values
            data_quality_notes: Data quality notes

        Returns:
            BaselineEvidence package
        """
        # Calculate data completeness
        expected_points = baseline_period.duration_days * 24  # hourly data
        actual_points = len(measurements)
        completeness = min(100.0, (actual_points / expected_points) * 100)

        # Create measurement summary
        measurement_summary = self._create_measurement_summary(measurements)

        evidence = BaselineEvidence(
            baseline_period=baseline_period,
            total_energy_mmbtu=total_energy_mmbtu,
            energy_by_fuel=energy_by_fuel or {},
            total_steam_production_klb=total_steam_production_klb,
            steam_by_header=steam_by_header or {},
            average_boiler_efficiency=average_boiler_efficiency,
            average_system_efficiency=average_system_efficiency,
            measurements=measurements,
            measurement_summary=measurement_summary,
            calculations=calculations,
            normalization_factors=normalization_factors or {},
            data_completeness_pct=completeness,
            data_quality_notes=data_quality_notes or [],
        )

        # Calculate and set hash
        evidence_dict = evidence.dict()
        evidence_dict["evidence_hash"] = evidence.calculate_hash()
        evidence = BaselineEvidence(**evidence_dict)

        logger.info(
            f"Baseline evidence created: {evidence.evidence_id}",
            extra={
                "period_days": baseline_period.duration_days,
                "measurements": len(measurements),
                "completeness_pct": completeness,
            }
        )

        return evidence

    def create_post_implementation_evidence(
        self,
        post_period: EvidencePeriod,
        measurements: List[MeasurementRecord],
        calculations: List[CalculationRecord],
        implementation_date: datetime,
        ecm_description: str,
        total_energy_mmbtu: float,
        total_steam_production_klb: float,
        average_boiler_efficiency: float,
        average_system_efficiency: float,
        ecm_ids: Optional[List[str]] = None,
        energy_by_fuel: Optional[Dict[str, float]] = None,
        steam_by_header: Optional[Dict[str, float]] = None,
        normalization_factors: Optional[Dict[str, float]] = None,
        data_quality_notes: Optional[List[str]] = None,
    ) -> PostEvidence:
        """
        Create post-implementation period evidence package.

        Args:
            post_period: Period definition
            measurements: List of measurement records
            calculations: List of calculation records
            implementation_date: Date ECM was implemented
            ecm_description: Description of ECM
            total_energy_mmbtu: Total energy consumption
            total_steam_production_klb: Total steam production
            average_boiler_efficiency: Average boiler efficiency
            average_system_efficiency: Average system efficiency
            ecm_ids: ECM identifiers
            energy_by_fuel: Energy breakdown by fuel type
            steam_by_header: Steam breakdown by header
            normalization_factors: Normalization factor values
            data_quality_notes: Data quality notes

        Returns:
            PostEvidence package
        """
        # Calculate data completeness
        expected_points = post_period.duration_days * 24
        actual_points = len(measurements)
        completeness = min(100.0, (actual_points / expected_points) * 100)

        # Create measurement summary
        measurement_summary = self._create_measurement_summary(measurements)

        evidence = PostEvidence(
            post_period=post_period,
            implementation_date=implementation_date,
            ecm_description=ecm_description,
            ecm_ids=ecm_ids or [],
            total_energy_mmbtu=total_energy_mmbtu,
            energy_by_fuel=energy_by_fuel or {},
            total_steam_production_klb=total_steam_production_klb,
            steam_by_header=steam_by_header or {},
            average_boiler_efficiency=average_boiler_efficiency,
            average_system_efficiency=average_system_efficiency,
            measurements=measurements,
            measurement_summary=measurement_summary,
            calculations=calculations,
            normalization_factors=normalization_factors or {},
            data_completeness_pct=completeness,
            data_quality_notes=data_quality_notes or [],
        )

        # Calculate and set hash
        evidence_dict = evidence.dict()
        evidence_dict["evidence_hash"] = evidence.calculate_hash()
        evidence = PostEvidence(**evidence_dict)

        logger.info(
            f"Post-implementation evidence created: {evidence.evidence_id}",
            extra={
                "period_days": post_period.duration_days,
                "measurements": len(measurements),
                "completeness_pct": completeness,
            }
        )

        return evidence

    def create_savings_evidence(
        self,
        baseline: BaselineEvidence,
        post: PostEvidence,
        normalization_factors: List[NormalizationFactor],
        methodology: MVMethodology = MVMethodology.OPTION_B,
        measurement_boundary: str = "Steam system boundary",
        routine_adjustments_mmbtu: float = 0.0,
        non_routine_adjustments_mmbtu: float = 0.0,
        energy_price_per_mmbtu: float = 10.0,
        co2e_factor_kg_per_mmbtu: float = 53.06,  # Natural gas default
        confidence_level_pct: float = 90.0,
    ) -> SavingsEvidence:
        """
        Create savings evidence by comparing baseline to post-implementation.

        Performs baseline adjustment using normalization factors and
        calculates savings with uncertainty quantification.

        Args:
            baseline: Baseline evidence
            post: Post-implementation evidence
            normalization_factors: Factors for baseline adjustment
            methodology: IPMVP methodology used
            measurement_boundary: Description of measurement boundary
            routine_adjustments_mmbtu: Routine adjustments (MMBtu)
            non_routine_adjustments_mmbtu: Non-routine adjustments (MMBtu)
            energy_price_per_mmbtu: Energy price for cost calculations
            co2e_factor_kg_per_mmbtu: CO2e emission factor
            confidence_level_pct: Confidence level for uncertainty

        Returns:
            SavingsEvidence package
        """
        # Calculate baseline-adjusted energy
        # For each normalization factor, adjust baseline to post conditions
        baseline_adjusted = baseline.total_energy_mmbtu
        for factor in normalization_factors:
            if factor.normalization_coefficient and factor.baseline_value != 0:
                # Adjust baseline energy based on factor change
                factor_ratio = factor.post_value / factor.baseline_value
                adjustment = (factor_ratio - 1) * factor.normalization_coefficient * baseline.total_energy_mmbtu
                baseline_adjusted += adjustment

        # Calculate gross savings
        gross_savings = baseline_adjusted - post.total_energy_mmbtu
        gross_savings_pct = (gross_savings / baseline_adjusted * 100) if baseline_adjusted != 0 else 0

        # Apply adjustments
        net_savings = gross_savings - routine_adjustments_mmbtu - non_routine_adjustments_mmbtu
        net_savings_pct = (net_savings / baseline_adjusted * 100) if baseline_adjusted != 0 else 0

        # Calculate uncertainty (simplified - production would use more sophisticated methods)
        # Using square root of sum of squares of individual uncertainties
        baseline_uncertainty = 1 - (baseline.data_completeness_pct / 100) * 0.1
        post_uncertainty = 1 - (post.data_completeness_pct / 100) * 0.1
        combined_uncertainty = ((baseline_uncertainty ** 2 + post_uncertainty ** 2) ** 0.5) * 100

        # Calculate bounds
        z_score = 1.645 if confidence_level_pct == 90 else 1.96  # 90% or 95% CI
        uncertainty_mmbtu = net_savings * (combined_uncertainty / 100) * z_score
        lower_bound = net_savings - uncertainty_mmbtu
        upper_bound = net_savings + uncertainty_mmbtu

        # Calculate cost savings
        energy_cost_savings = net_savings * energy_price_per_mmbtu

        # Calculate emissions reduction (convert kg to metric tons)
        co2e_reduction = (net_savings * co2e_factor_kg_per_mmbtu) / 1000

        # Build methodology description
        calc_methodology = (
            f"Baseline energy ({baseline.total_energy_mmbtu:.2f} MMBtu) adjusted to post-period "
            f"conditions using {len(normalization_factors)} normalization factor(s). "
            f"Adjusted baseline: {baseline_adjusted:.2f} MMBtu. "
            f"Post-period consumption: {post.total_energy_mmbtu:.2f} MMBtu."
        )

        evidence = SavingsEvidence(
            baseline_evidence_id=str(baseline.evidence_id),
            post_evidence_id=str(post.evidence_id),
            methodology=methodology,
            measurement_boundary=measurement_boundary,
            baseline_adjusted_energy_mmbtu=baseline_adjusted,
            post_period_energy_mmbtu=post.total_energy_mmbtu,
            gross_savings_mmbtu=gross_savings,
            gross_savings_pct=gross_savings_pct,
            routine_adjustments_mmbtu=routine_adjustments_mmbtu,
            non_routine_adjustments_mmbtu=non_routine_adjustments_mmbtu,
            net_savings_mmbtu=net_savings,
            net_savings_pct=net_savings_pct,
            savings_uncertainty_pct=combined_uncertainty,
            confidence_level_pct=confidence_level_pct,
            savings_lower_bound_mmbtu=lower_bound,
            savings_upper_bound_mmbtu=upper_bound,
            energy_cost_savings_usd=energy_cost_savings,
            total_cost_savings_usd=energy_cost_savings,  # Add avoided costs if applicable
            co2e_reduction_metric_tons=co2e_reduction,
            normalization_factors=normalization_factors,
            calculation_methodology=calc_methodology,
            baseline_hash=baseline.evidence_hash or baseline.calculate_hash(),
            post_hash=post.evidence_hash or post.calculate_hash(),
        )

        # Calculate and set hash
        evidence_dict = evidence.dict()
        evidence_dict["evidence_hash"] = evidence.calculate_hash()
        evidence = SavingsEvidence(**evidence_dict)

        logger.info(
            f"Savings evidence created: {evidence.evidence_id}",
            extra={
                "net_savings_mmbtu": net_savings,
                "net_savings_pct": net_savings_pct,
                "cost_savings_usd": energy_cost_savings,
            }
        )

        return evidence

    def package_mv_report(
        self,
        savings_evidence: SavingsEvidence,
        baseline_evidence: BaselineEvidence,
        post_evidence: PostEvidence,
        methodology_notes: str,
        project_id: str,
        project_name: str,
        facility_id: str,
        facility_name: str,
        prepared_by: str,
        reviewed_by: Optional[str] = None,
        approved_by: Optional[str] = None,
    ) -> MVReport:
        """
        Package complete M&V report for submission.

        Args:
            savings_evidence: Calculated savings evidence
            baseline_evidence: Baseline period evidence
            post_evidence: Post-implementation evidence
            methodology_notes: Detailed methodology documentation
            project_id: Project identifier
            project_name: Project name
            facility_id: Facility identifier
            facility_name: Facility name
            prepared_by: Preparer name/ID
            reviewed_by: Reviewer name/ID
            approved_by: Approver name/ID

        Returns:
            Complete MVReport
        """
        # Create reporting period from post evidence
        reporting_period = EvidencePeriod(
            start_date=baseline_evidence.baseline_period.start_date,
            end_date=post_evidence.post_period.end_date,
            duration_days=(post_evidence.post_period.end_date - baseline_evidence.baseline_period.start_date).days,
            description="Full M&V reporting period (baseline through post-implementation)",
        )

        # Data quality statement
        avg_completeness = (
            baseline_evidence.data_completeness_pct + post_evidence.data_completeness_pct
        ) / 2
        data_quality_statement = (
            f"Data completeness: Baseline {baseline_evidence.data_completeness_pct:.1f}%, "
            f"Post-implementation {post_evidence.data_completeness_pct:.1f}%. "
            f"Average completeness {avg_completeness:.1f}%. "
            f"All calculations use SHA-256 provenance hashing for integrity verification."
        )

        report = MVReport(
            project_id=project_id,
            project_name=project_name,
            facility_id=facility_id,
            facility_name=facility_name,
            reporting_period=reporting_period,
            methodology=savings_evidence.methodology,
            methodology_notes=methodology_notes,
            measurement_boundary=savings_evidence.measurement_boundary,
            baseline_evidence=baseline_evidence,
            post_evidence=post_evidence,
            savings_evidence=savings_evidence,
            total_savings_mmbtu=savings_evidence.net_savings_mmbtu,
            total_cost_savings_usd=savings_evidence.total_cost_savings_usd,
            total_co2e_reduction_mt=savings_evidence.co2e_reduction_metric_tons,
            data_quality_statement=data_quality_statement,
            prepared_by=prepared_by,
            reviewed_by=reviewed_by,
            approved_by=approved_by,
        )

        # Calculate and set hash
        report_dict = report.dict()
        report_dict["report_hash"] = report.calculate_hash()
        report = MVReport(**report_dict)

        logger.info(
            f"M&V report packaged: {report.report_id}",
            extra={
                "project_id": project_id,
                "total_savings_mmbtu": report.total_savings_mmbtu,
            }
        )

        return report

    def sign_evidence_pack(
        self,
        evidence: Union[BaselineEvidence, PostEvidence, SavingsEvidence, MVReport],
        signer_id: str,
        signer_role: str,
    ) -> SignedEvidence:
        """
        Sign an evidence package with HMAC-SHA256.

        Args:
            evidence: Evidence to sign
            signer_id: Signer identifier
            signer_role: Signer role (PREPARER, REVIEWER, APPROVER)

        Returns:
            SignedEvidence with HMAC signature

        Raises:
            ValueError: If signing key not configured
        """
        if not self._signing_key:
            raise ValueError("Signing key not configured")

        # Get evidence hash
        evidence_hash = evidence.calculate_hash()

        # Create HMAC signature
        message = f"{evidence_hash}:{signer_id}:{signer_role}"
        signature = hmac.new(
            self._signing_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        signed = SignedEvidence(
            evidence_type=evidence.evidence_type,
            evidence_hash=evidence_hash,
            signature=signature,
            signer_id=signer_id,
            signer_role=signer_role,
        )

        logger.info(
            f"Evidence signed: {signed.evidence_id}",
            extra={
                "signer_id": signer_id,
                "signer_role": signer_role,
            }
        )

        return signed

    def verify_signature(
        self,
        signed_evidence: SignedEvidence,
    ) -> bool:
        """
        Verify HMAC signature on signed evidence.

        Args:
            signed_evidence: Signed evidence to verify

        Returns:
            True if signature is valid

        Raises:
            ValueError: If signing key not configured
        """
        if not self._signing_key:
            raise ValueError("Signing key not configured")

        # Recreate message
        message = f"{signed_evidence.evidence_hash}:{signed_evidence.signer_id}:{signed_evidence.signer_role}"

        # Calculate expected signature
        expected = hmac.new(
            self._signing_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signed_evidence.signature, expected)

    def store_evidence(
        self,
        evidence: Union[BaselineEvidence, PostEvidence, SavingsEvidence, MVReport],
        as_zip: bool = False,
    ) -> str:
        """
        Store evidence package to configured storage.

        Args:
            evidence: Evidence to store
            as_zip: Store as ZIP archive

        Returns:
            Storage path

        Raises:
            ValueError: If storage path not configured
        """
        if not self.storage_path:
            raise ValueError("Storage path not configured")

        # Create date-based directory
        date_path = evidence.created_at.strftime("%Y/%m/%d")
        full_path = self.storage_path / date_path
        full_path.mkdir(parents=True, exist_ok=True)

        filename = f"{evidence.evidence_type.lower()}_{evidence.evidence_id}"

        if as_zip:
            file_path = full_path / f"{filename}.zip"
            with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as zf:
                evidence_json = json.dumps(evidence.dict(), indent=2, default=str)
                zf.writestr("evidence.json", evidence_json)
        else:
            file_path = full_path / f"{filename}.json"
            with open(file_path, "w") as f:
                json.dump(evidence.dict(), f, indent=2, default=str)

        logger.info(f"Evidence stored: {file_path}")
        return str(file_path)

    def _create_measurement_summary(
        self,
        measurements: List[MeasurementRecord],
    ) -> Dict[str, Any]:
        """Create summary statistics for measurements."""
        if not measurements:
            return {}

        # Group by tag
        by_tag: Dict[str, List[float]] = {}
        for m in measurements:
            if m.tag_id not in by_tag:
                by_tag[m.tag_id] = []
            by_tag[m.tag_id].append(m.value)

        summary = {
            "total_measurements": len(measurements),
            "unique_tags": len(by_tag),
            "by_tag": {},
        }

        for tag, values in by_tag.items():
            summary["by_tag"][tag] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        return summary
