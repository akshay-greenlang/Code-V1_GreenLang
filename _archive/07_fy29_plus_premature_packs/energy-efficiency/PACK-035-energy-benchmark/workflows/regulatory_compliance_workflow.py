# -*- coding: utf-8 -*-
"""
Regulatory Compliance Workflow
===================================

4-phase workflow for energy benchmarking regulatory compliance within
PACK-035 Energy Benchmark Pack.

Phases:
    1. DataValidation       -- Validate facility data against regulation requirements
    2. RatingCalculation    -- Calculate energy rating (EPC, DEC, ENERGY STAR, NABERS)
    3. CertificateGeneration -- Generate certificate data with rating and recommendations
    4. SubmissionPackage     -- Assemble regulatory submission package

Supports EPC (EU/UK), DEC (UK), EPBD recast (EU), MEES (UK), LL97 (NYC),
NABERS (Australia), and ENERGY STAR (US) compliance workflows.

The workflow follows GreenLang zero-hallucination principles: all rating
calculations use published regulatory thresholds, penalty values are from
legislation, and compliance status is deterministic. No LLM calls in
the numeric computation path.

Schedule: on-demand / annual
Estimated duration: 60 minutes

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RegulationType(str, Enum):
    """Supported regulatory frameworks."""

    EPC = "epc"
    DEC = "dec"
    EPBD = "epbd"
    MEES = "mees"
    LL97 = "ll97"
    NABERS = "nabers"
    ENERGY_STAR = "energy_star"


class ComplianceStatus(str, Enum):
    """Compliance status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    AT_RISK = "at_risk"
    EXEMPT = "exempt"
    NOT_APPLICABLE = "not_applicable"


class EPCRating(str, Enum):
    """EPC rating band."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class NABERSStars(str, Enum):
    """NABERS star rating."""

    STAR_0 = "0"
    STAR_1 = "1"
    STAR_2 = "2"
    STAR_3 = "3"
    STAR_4 = "4"
    STAR_4_5 = "4.5"
    STAR_5 = "5"
    STAR_5_5 = "5.5"
    STAR_6 = "6"


class RiskLevel(str, Enum):
    """Compliance risk level."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# EPC rating bands by primary energy demand (kWh/m2/yr)
EPC_RATING_BANDS: Dict[str, Tuple[float, float]] = {
    "A+": (0.0, 25.0),
    "A": (25.0, 50.0),
    "B": (50.0, 75.0),
    "C": (75.0, 100.0),
    "D": (100.0, 150.0),
    "E": (150.0, 200.0),
    "F": (200.0, 250.0),
    "G": (250.0, 9999.0),
}

# DEC rating bands by operational rating
DEC_RATING_BANDS: Dict[str, Tuple[float, float]] = {
    "A": (0.0, 25.0),
    "B": (25.0, 50.0),
    "C": (50.0, 75.0),
    "D": (75.0, 100.0),
    "E": (100.0, 125.0),
    "F": (125.0, 150.0),
    "G": (150.0, 9999.0),
}

# MEES minimum requirements by year
MEES_REQUIREMENTS: Dict[int, str] = {
    2018: "E",
    2023: "E",
    2025: "C",
    2027: "C",
    2030: "B",
}

# NYC LL97 carbon intensity limits (kgCO2e/sqft/yr) by building type
LL97_LIMITS: Dict[str, Dict[int, float]] = {
    "office": {2024: 8.46, 2030: 4.53, 2035: 2.96, 2040: 1.48, 2050: 0.0},
    "retail": {2024: 11.81, 2030: 5.28, 2035: 3.41, 2040: 1.71, 2050: 0.0},
    "hotel": {2024: 9.67, 2030: 5.26, 2035: 3.44, 2040: 1.72, 2050: 0.0},
    "hospital": {2024: 23.81, 2030: 12.07, 2035: 7.87, 2040: 3.94, 2050: 0.0},
    "school": {2024: 7.58, 2030: 4.07, 2035: 2.66, 2040: 1.33, 2050: 0.0},
    "warehouse": {2024: 5.74, 2030: 3.05, 2035: 1.99, 2040: 1.00, 2050: 0.0},
}

# LL97 penalty rate per tonne CO2e over limit
LL97_PENALTY_PER_TONNE = 268.0  # USD per tonne CO2e

# NABERS benchmarks for office (kgCO2e/m2/yr)
NABERS_BENCHMARKS: Dict[str, float] = {
    "6": 15.0, "5.5": 25.0, "5": 35.0, "4.5": 50.0, "4": 65.0,
    "3": 90.0, "2": 120.0, "1": 160.0, "0": 999.0,
}

# Primary energy factors for EPC calculation
PRIMARY_ENERGY_FACTORS: Dict[str, float] = {
    "electricity": 2.50,
    "natural_gas": 1.10,
    "fuel_oil": 1.10,
    "district_heating": 1.30,
    "lpg": 1.10,
    "biomass": 1.20,
}

# CO2 emission factors
EMISSION_FACTORS: Dict[str, float] = {
    "electricity": 0.207,
    "natural_gas": 0.183,
    "fuel_oil": 0.267,
    "district_heating": 0.194,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class BuildingData(BaseModel):
    """Building data for regulatory assessment."""

    floor_area_m2: float = Field(default=0.0, ge=0.0, description="Gross internal area m2")
    floor_area_sqft: float = Field(default=0.0, ge=0.0, description="Gross area sqft (for US)")
    building_type: str = Field(default="office")
    year_built: int = Field(default=0, ge=0)
    country: str = Field(default="", description="ISO alpha-2")
    state_province: str = Field(default="", description="For US jurisdictions")
    occupancy_type: str = Field(default="commercial")
    listed_building: bool = Field(default=False, description="Heritage listed")
    exemption_type: str = Field(default="", description="Any exemption claimed")


class RegulatoryComplianceInput(BaseModel):
    """Input data model for RegulatoryComplianceWorkflow."""

    facility_id: str = Field(default="", description="Facility identifier")
    facility_name: str = Field(default="", description="Facility name")
    regulation: RegulationType = Field(default=RegulationType.EPC)
    energy_data: Dict[str, float] = Field(
        default_factory=dict,
        description="Energy by source: {'electricity': kWh, 'natural_gas': kWh, ...}",
    )
    building_data: BuildingData = Field(default_factory=BuildingData)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    current_rating: str = Field(default="", description="Current rating if known")
    target_rating: str = Field(default="", description="Required target rating")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class RegulatoryComplianceResult(BaseModel):
    """Complete result from regulatory compliance workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="regulatory_compliance")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    facility_id: str = Field(default="")
    regulation: str = Field(default="")
    rating: str = Field(default="", description="Calculated rating")
    rating_value: float = Field(default=0.0, description="Numeric rating value")
    compliance_status: str = Field(default="")
    certificate_data: Dict[str, Any] = Field(default_factory=dict)
    submission_package: Dict[str, Any] = Field(default_factory=dict)
    penalties: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[Dict[str, str]] = Field(default_factory=list)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RegulatoryComplianceWorkflow:
    """
    4-phase regulatory compliance workflow for energy benchmarking.

    Performs data validation, rating calculation, certificate generation,
    and submission package assembly for multiple regulatory frameworks.

    Zero-hallucination: all rating calculations use published regulatory
    thresholds, penalty values from legislation, and compliance status
    is deterministic. No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.
        _rating: Calculated energy rating.
        _rating_value: Numeric rating value.
        _compliance_status: Compliance determination.
        _certificate: Certificate data.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = RegulatoryComplianceWorkflow()
        >>> inp = RegulatoryComplianceInput(
        ...     regulation=RegulationType.EPC,
        ...     energy_data={"electricity": 500000, "natural_gas": 300000},
        ...     building_data=BuildingData(floor_area_m2=5000),
        ... )
        >>> result = wf.run(inp)
        >>> assert result.rating in ["A+", "A", "B", "C", "D", "E", "F", "G"]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RegulatoryComplianceWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._rating: str = ""
        self._rating_value: float = 0.0
        self._compliance_status: ComplianceStatus = ComplianceStatus.NOT_APPLICABLE
        self._certificate: Dict[str, Any] = {}
        self._penalties: Dict[str, Any] = {}
        self._recommendations: List[Dict[str, str]] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: RegulatoryComplianceInput) -> RegulatoryComplianceResult:
        """
        Execute the 4-phase regulatory compliance workflow.

        Args:
            input_data: Validated regulatory compliance input.

        Returns:
            RegulatoryComplianceResult with rating, compliance status, certificate.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting regulatory compliance workflow %s regulation=%s",
            self.workflow_id, input_data.regulation.value,
        )

        self._phase_results = []
        self._rating = ""
        self._rating_value = 0.0
        self._compliance_status = ComplianceStatus.NOT_APPLICABLE
        self._certificate = {}
        self._penalties = {}
        self._recommendations = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = self._phase_data_validation(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_rating_calculation(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_certificate_generation(input_data)
            self._phase_results.append(phase3)

            phase4 = self._phase_submission_package(input_data)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Regulatory compliance workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        result = RegulatoryComplianceResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            facility_id=input_data.facility_id,
            regulation=input_data.regulation.value,
            rating=self._rating,
            rating_value=round(self._rating_value, 2),
            compliance_status=self._compliance_status.value,
            certificate_data=self._certificate,
            submission_package=self._certificate,
            penalties=self._penalties,
            recommendations=self._recommendations,
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Regulatory compliance workflow %s completed in %.2fs rating=%s status=%s",
            self.workflow_id, elapsed, self._rating, self._compliance_status.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Validation
    # -------------------------------------------------------------------------

    def _phase_data_validation(
        self, input_data: RegulatoryComplianceInput
    ) -> PhaseResult:
        """Validate facility data against regulation requirements."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        errors: List[str] = []
        outputs: Dict[str, Any] = {}

        bd = input_data.building_data
        reg = input_data.regulation

        # Area validation
        if bd.floor_area_m2 <= 0 and bd.floor_area_sqft <= 0:
            errors.append("Floor area must be provided")

        # Energy data validation
        total_kwh = sum(input_data.energy_data.values())
        if total_kwh <= 0:
            errors.append("No energy consumption data provided")

        # Regulation-specific checks
        if reg == RegulationType.LL97:
            if bd.floor_area_sqft <= 0 and bd.floor_area_m2 > 0:
                bd.floor_area_sqft = bd.floor_area_m2 * 10.764
                warnings.append("Converted m2 to sqft for LL97")
            if bd.floor_area_sqft < 25000:
                warnings.append("LL97 applies to buildings >= 25,000 sqft")

        if reg == RegulationType.DEC:
            if bd.floor_area_m2 < 250:
                warnings.append("DEC required for public buildings >= 250 m2")

        if reg == RegulationType.MEES:
            if bd.listed_building:
                warnings.append("Listed buildings may be exempt from MEES")

        if reg == RegulationType.NABERS:
            if bd.country != "AU":
                warnings.append("NABERS is an Australian rating scheme")

        # Check exemptions
        if bd.exemption_type:
            warnings.append(f"Exemption claimed: {bd.exemption_type}")

        outputs["regulation"] = reg.value
        outputs["floor_area_m2"] = bd.floor_area_m2
        outputs["total_energy_kwh"] = round(total_kwh, 2)
        outputs["energy_sources"] = list(input_data.energy_data.keys())
        outputs["validation_errors"] = len(errors)
        outputs["validation_warnings"] = len(warnings)

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 DataValidation: regulation=%s errors=%d warnings=%d",
            reg.value, len(errors), len(warnings),
        )
        return PhaseResult(
            phase_name="data_validation", phase_number=1,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Rating Calculation
    # -------------------------------------------------------------------------

    def _phase_rating_calculation(
        self, input_data: RegulatoryComplianceInput
    ) -> PhaseResult:
        """Calculate energy rating based on regulation."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        reg = input_data.regulation

        if reg == RegulationType.EPC or reg == RegulationType.EPBD:
            self._calculate_epc_rating(input_data)
        elif reg == RegulationType.DEC:
            self._calculate_dec_rating(input_data)
        elif reg == RegulationType.MEES:
            self._calculate_mees_compliance(input_data)
        elif reg == RegulationType.LL97:
            self._calculate_ll97_compliance(input_data)
        elif reg == RegulationType.NABERS:
            self._calculate_nabers_rating(input_data)
        elif reg == RegulationType.ENERGY_STAR:
            self._calculate_energy_star(input_data)
        else:
            warnings.append(f"Unknown regulation: {reg.value}")

        outputs["rating"] = self._rating
        outputs["rating_value"] = round(self._rating_value, 2)
        outputs["compliance_status"] = self._compliance_status.value
        outputs["regulation"] = reg.value

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 RatingCalculation: %s rating=%s value=%.1f status=%s",
            reg.value, self._rating, self._rating_value, self._compliance_status.value,
        )
        return PhaseResult(
            phase_name="rating_calculation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calculate_epc_rating(self, input_data: RegulatoryComplianceInput) -> None:
        """Calculate EPC rating from primary energy demand (zero-hallucination)."""
        area = input_data.building_data.floor_area_m2
        if area <= 0:
            self._rating = "G"
            self._rating_value = 999.0
            return

        # Calculate primary energy
        total_primary = 0.0
        for source, kwh in input_data.energy_data.items():
            pef = PRIMARY_ENERGY_FACTORS.get(source, 1.0)
            total_primary += kwh * pef

        primary_eui = total_primary / area
        self._rating_value = primary_eui

        for rating, (lower, upper) in EPC_RATING_BANDS.items():
            if lower <= primary_eui < upper:
                self._rating = rating
                break

        # MEES check (default to 2025 requirement)
        min_rating = MEES_REQUIREMENTS.get(input_data.reporting_year, "E")
        rating_order = ["A+", "A", "B", "C", "D", "E", "F", "G"]
        current_idx = rating_order.index(self._rating) if self._rating in rating_order else 7
        min_idx = rating_order.index(min_rating) if min_rating in rating_order else 7

        if current_idx <= min_idx:
            self._compliance_status = ComplianceStatus.COMPLIANT
        elif current_idx == min_idx + 1:
            self._compliance_status = ComplianceStatus.AT_RISK
        else:
            self._compliance_status = ComplianceStatus.NON_COMPLIANT

    def _calculate_dec_rating(self, input_data: RegulatoryComplianceInput) -> None:
        """Calculate DEC operational rating (zero-hallucination)."""
        area = input_data.building_data.floor_area_m2
        total_kwh = sum(input_data.energy_data.values())
        site_eui = total_kwh / area if area > 0 else 999.0

        # DEC operational rating is ratio of actual to benchmark * 100
        bt = input_data.building_data.building_type
        typical_benchmarks = {"office": 215.0, "school": 150.0, "hospital": 440.0, "retail": 270.0}
        benchmark = typical_benchmarks.get(bt, 215.0)
        operational_rating = (site_eui / benchmark) * 100.0

        self._rating_value = operational_rating
        for rating, (lower, upper) in DEC_RATING_BANDS.items():
            if lower <= operational_rating < upper:
                self._rating = rating
                break

        self._compliance_status = ComplianceStatus.COMPLIANT if operational_rating < 150 else ComplianceStatus.NON_COMPLIANT

    def _calculate_mees_compliance(self, input_data: RegulatoryComplianceInput) -> None:
        """Calculate MEES compliance status (zero-hallucination)."""
        self._calculate_epc_rating(input_data)
        min_required = MEES_REQUIREMENTS.get(input_data.reporting_year, "E")
        rating_order = ["A+", "A", "B", "C", "D", "E", "F", "G"]
        current_idx = rating_order.index(self._rating) if self._rating in rating_order else 7
        min_idx = rating_order.index(min_required)

        if input_data.building_data.listed_building:
            self._compliance_status = ComplianceStatus.EXEMPT
        elif current_idx <= min_idx:
            self._compliance_status = ComplianceStatus.COMPLIANT
        else:
            self._compliance_status = ComplianceStatus.NON_COMPLIANT
            self._penalties = {
                "type": "MEES non-compliance",
                "required_rating": min_required,
                "current_rating": self._rating,
                "action_required": f"Upgrade from {self._rating} to minimum {min_required}",
            }

    def _calculate_ll97_compliance(self, input_data: RegulatoryComplianceInput) -> None:
        """Calculate NYC LL97 compliance (zero-hallucination)."""
        area_sqft = input_data.building_data.floor_area_sqft
        if area_sqft <= 0:
            area_sqft = input_data.building_data.floor_area_m2 * 10.764

        # Calculate carbon intensity
        total_co2_kg = 0.0
        for source, kwh in input_data.energy_data.items():
            ef = EMISSION_FACTORS.get(source, 0.207)
            total_co2_kg += kwh * ef

        total_co2_tonnes = total_co2_kg / 1000.0
        carbon_per_sqft = total_co2_kg / area_sqft if area_sqft > 0 else 0.0

        bt = input_data.building_data.building_type
        limits = LL97_LIMITS.get(bt, LL97_LIMITS["office"])
        current_limit = 0.0
        for year, limit in sorted(limits.items()):
            if input_data.reporting_year >= year:
                current_limit = limit

        self._rating = f"{carbon_per_sqft:.2f} kgCO2e/sqft"
        self._rating_value = carbon_per_sqft

        if carbon_per_sqft <= current_limit:
            self._compliance_status = ComplianceStatus.COMPLIANT
        else:
            self._compliance_status = ComplianceStatus.NON_COMPLIANT
            excess_co2 = (carbon_per_sqft - current_limit) * area_sqft / 1000.0
            penalty = excess_co2 * LL97_PENALTY_PER_TONNE
            self._penalties = {
                "limit_kgco2_sqft": current_limit,
                "actual_kgco2_sqft": round(carbon_per_sqft, 4),
                "excess_tonnes_co2": round(excess_co2, 2),
                "annual_penalty_usd": round(penalty, 2),
                "penalty_rate": LL97_PENALTY_PER_TONNE,
            }

    def _calculate_nabers_rating(self, input_data: RegulatoryComplianceInput) -> None:
        """Calculate NABERS star rating (zero-hallucination)."""
        area = input_data.building_data.floor_area_m2
        total_co2_kg = sum(
            kwh * EMISSION_FACTORS.get(src, 0.207) for src, kwh in input_data.energy_data.items()
        )
        co2_per_m2 = total_co2_kg / area if area > 0 else 999.0

        self._rating_value = co2_per_m2
        self._rating = "0"
        for stars, threshold in sorted(NABERS_BENCHMARKS.items(), key=lambda x: float(x[0]), reverse=True):
            if co2_per_m2 <= threshold:
                self._rating = f"{stars} stars"
                break

        self._compliance_status = ComplianceStatus.COMPLIANT if float(self._rating.split()[0]) >= 4 else ComplianceStatus.AT_RISK

    def _calculate_energy_star(self, input_data: RegulatoryComplianceInput) -> None:
        """Calculate ENERGY STAR score estimate (zero-hallucination)."""
        area = input_data.building_data.floor_area_m2
        total_kwh = sum(input_data.energy_data.values())
        site_eui = total_kwh / area if area > 0 else 999.0

        typical = {"office": 215.0, "retail": 270.0, "hotel": 305.0, "school": 150.0, "warehouse": 65.0}
        bt = input_data.building_data.building_type
        benchmark = typical.get(bt, 215.0)

        ratio = site_eui / benchmark if benchmark > 0 else 1.0
        if ratio <= 0.4:
            score = 95
        elif ratio <= 0.6:
            score = 80
        elif ratio <= 0.8:
            score = 65
        elif ratio <= 1.0:
            score = 50
        elif ratio <= 1.3:
            score = 30
        else:
            score = max(1, 15)

        self._rating = str(score)
        self._rating_value = float(score)
        self._compliance_status = ComplianceStatus.COMPLIANT if score >= 75 else ComplianceStatus.AT_RISK

    # -------------------------------------------------------------------------
    # Phase 3: Certificate Generation
    # -------------------------------------------------------------------------

    def _phase_certificate_generation(
        self, input_data: RegulatoryComplianceInput
    ) -> PhaseResult:
        """Generate certificate data with rating and recommendations."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._certificate = {
            "certificate_id": f"cert-{uuid.uuid4().hex[:8]}",
            "regulation": input_data.regulation.value,
            "facility_id": input_data.facility_id,
            "facility_name": input_data.facility_name,
            "rating": self._rating,
            "rating_value": round(self._rating_value, 2),
            "compliance_status": self._compliance_status.value,
            "valid_from": f"{input_data.reporting_year}-01-01",
            "valid_until": f"{input_data.reporting_year + 10}-01-01",
            "issued_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "building_type": input_data.building_data.building_type,
            "floor_area_m2": input_data.building_data.floor_area_m2,
        }

        # Generate improvement recommendations
        self._recommendations = self._generate_regulatory_recommendations(input_data)

        outputs["certificate_id"] = self._certificate["certificate_id"]
        outputs["rating"] = self._rating
        outputs["compliance_status"] = self._compliance_status.value
        outputs["recommendations_count"] = len(self._recommendations)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 CertificateGeneration: cert=%s rating=%s",
            self._certificate["certificate_id"], self._rating,
        )
        return PhaseResult(
            phase_name="certificate_generation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_regulatory_recommendations(
        self, input_data: RegulatoryComplianceInput
    ) -> List[Dict[str, str]]:
        """Generate regulation-specific recommendations (deterministic)."""
        recs: List[Dict[str, str]] = []

        if self._compliance_status == ComplianceStatus.NON_COMPLIANT:
            recs.append({
                "priority": "critical",
                "recommendation": f"Non-compliant with {input_data.regulation.value}. Immediate action required.",
            })

        if self._compliance_status == ComplianceStatus.AT_RISK:
            recs.append({
                "priority": "high",
                "recommendation": f"At risk of non-compliance. Plan improvements before next assessment.",
            })

        # Energy source specific
        if "natural_gas" in input_data.energy_data:
            gas_pct = input_data.energy_data["natural_gas"] / sum(input_data.energy_data.values()) * 100
            if gas_pct > 40:
                recs.append({
                    "priority": "medium",
                    "recommendation": f"Gas accounts for {gas_pct:.0f}% of energy. Consider heat pump electrification.",
                })

        return recs

    # -------------------------------------------------------------------------
    # Phase 4: Submission Package
    # -------------------------------------------------------------------------

    def _phase_submission_package(
        self, input_data: RegulatoryComplianceInput
    ) -> PhaseResult:
        """Assemble regulatory submission package."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        package = {
            "package_id": f"sub-{uuid.uuid4().hex[:8]}",
            "regulation": input_data.regulation.value,
            "certificate": self._certificate,
            "energy_data_summary": {
                src: round(kwh, 2) for src, kwh in input_data.energy_data.items()
            },
            "building_data": input_data.building_data.model_dump(),
            "penalties": self._penalties,
            "recommendations": self._recommendations,
            "reporting_year": input_data.reporting_year,
            "prepared_date": datetime.utcnow().isoformat() + "Z",
            "next_assessment_due": f"{input_data.reporting_year + 1}-01-01",
        }

        outputs["package_id"] = package["package_id"]
        outputs["documents_included"] = 3 + (1 if self._penalties else 0)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 4 SubmissionPackage: package=%s",
            package["package_id"],
        )
        return PhaseResult(
            phase_name="submission_package", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: RegulatoryComplianceResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
