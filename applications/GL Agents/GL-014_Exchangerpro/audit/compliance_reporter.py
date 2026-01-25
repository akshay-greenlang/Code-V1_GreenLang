# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Compliance Reporter

Generates regulatory compliance reports for heat exchanger operations.
Supports TEMA compliance verification, energy savings verification,
and CO2e accounting audit trails.

Features:
    - Generate TEMA compliance verification reports
    - Energy savings verification and documentation
    - CO2e accounting audit trail generation
    - Multi-framework compliance reporting
    - Automated evidence collection for audits
    - Report export in multiple formats

Standards:
    - TEMA (Tubular Exchanger Manufacturers Association) Standards
    - ASME PTC 12.5-2000 (Single Phase Heat Exchangers)
    - ISO 50001:2018 (Energy Management Systems)
    - EPA 40 CFR Part 98 (Mandatory GHG Reporting)
    - ISO 14064 (GHG Accounting and Verification)
    - 21 CFR Part 11 (Electronic Records)

Example:
    >>> from audit.compliance_reporter import ComplianceReporter
    >>> reporter = ComplianceReporter()
    >>> report = reporter.generate_tema_compliance_report(
    ...     exchanger_id="HEX-001",
    ...     time_period=("2024-01-01", "2024-12-31")
    ... )
    >>> reporter.export_report(report, format="pdf")
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from .schemas import (
    AuditRecord,
    ProvenanceChain,
    ChangeRecord,
    ComputationType,
    AuditStatistics,
    ComplianceStatus,
    ChainVerificationStatus,
    compute_sha256,
)
from .evidence_generator import (
    EvidenceGenerator,
    EvidenceRecord,
    EvidenceType,
    SealStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COMPLIANCE FRAMEWORK ENUMS
# =============================================================================

class ComplianceFramework(str, Enum):
    """Supported regulatory compliance frameworks."""

    TEMA = "tema"  # Tubular Exchanger Manufacturers Association
    ASME_PTC_12_5 = "asme_ptc_12_5"  # Single Phase Heat Exchangers
    ISO_50001 = "iso_50001"  # Energy Management
    EPA_40_CFR_98 = "epa_40_cfr_98"  # GHG Reporting
    ISO_14064 = "iso_14064"  # GHG Accounting
    CFR_21_PART_11 = "cfr_21_part_11"  # Electronic Records


class ReportType(str, Enum):
    """Types of compliance reports."""

    TEMA_COMPLIANCE = "tema_compliance"
    ENERGY_SAVINGS = "energy_savings"
    CO2E_ACCOUNTING = "co2e_accounting"
    PERFORMANCE_VERIFICATION = "performance_verification"
    FOULING_ANALYSIS = "fouling_analysis"
    MAINTENANCE_COMPLIANCE = "maintenance_compliance"
    REGULATORY_SUBMISSION = "regulatory_submission"
    AUDIT_SUMMARY = "audit_summary"


class ReportStatus(str, Enum):
    """Status of compliance report."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    ARCHIVED = "archived"


# =============================================================================
# TEMA COMPLIANCE CHECKS
# =============================================================================

@dataclass
class TEMACheck:
    """Individual TEMA compliance check result."""

    check_id: str = ""
    check_name: str = ""
    tema_reference: str = ""  # e.g., "RCB-2.6"
    category: str = ""  # e.g., "fouling", "pressure_drop", "thermal_design"

    # Check parameters
    parameter_name: str = ""
    parameter_value: float = 0.0
    parameter_unit: str = ""

    # Limits
    tema_limit: Optional[float] = None
    tema_limit_type: str = "max"  # "max", "min", "range"
    tema_limit_lower: Optional[float] = None
    tema_limit_upper: Optional[float] = None

    # Results
    is_compliant: bool = True
    deviation_percent: float = 0.0
    severity: str = "info"  # "info", "warning", "critical"
    notes: str = ""

    # Traceability
    calculation_id: Optional[str] = None
    evidence_hash: str = ""
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Initialize check ID if not provided."""
        if not self.check_id:
            self.check_id = f"TEMA-CHK-{uuid.uuid4().hex[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_id": self.check_id,
            "check_name": self.check_name,
            "tema_reference": self.tema_reference,
            "category": self.category,
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
            "parameter_unit": self.parameter_unit,
            "tema_limit": self.tema_limit,
            "tema_limit_type": self.tema_limit_type,
            "tema_limit_lower": self.tema_limit_lower,
            "tema_limit_upper": self.tema_limit_upper,
            "is_compliant": self.is_compliant,
            "deviation_percent": self.deviation_percent,
            "severity": self.severity,
            "notes": self.notes,
            "calculation_id": self.calculation_id,
            "evidence_hash": self.evidence_hash,
            "checked_at": self.checked_at.isoformat(),
        }


# =============================================================================
# ENERGY SAVINGS VERIFICATION
# =============================================================================

@dataclass
class EnergySavingsRecord:
    """Record of energy savings from heat exchanger optimization."""

    record_id: str = ""
    exchanger_id: str = ""
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Baseline vs actual
    baseline_energy_mj: float = 0.0
    actual_energy_mj: float = 0.0
    savings_mj: float = 0.0
    savings_percent: float = 0.0

    # Cost savings
    energy_cost_rate: float = 0.0  # $/MJ
    cost_savings_usd: float = 0.0

    # Calculation method
    calculation_method: str = ""
    baseline_period: str = ""
    adjustment_factors: Dict[str, float] = field(default_factory=dict)

    # Verification
    measurement_points: List[str] = field(default_factory=list)
    data_quality_score: float = 1.0
    uncertainty_percent: float = 0.0

    # Provenance
    calculation_hash: str = ""
    evidence_records: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize record ID if not provided."""
        if not self.record_id:
            self.record_id = f"ES-{uuid.uuid4().hex[:12].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "exchanger_id": self.exchanger_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "baseline_energy_mj": self.baseline_energy_mj,
            "actual_energy_mj": self.actual_energy_mj,
            "savings_mj": self.savings_mj,
            "savings_percent": self.savings_percent,
            "energy_cost_rate": self.energy_cost_rate,
            "cost_savings_usd": self.cost_savings_usd,
            "calculation_method": self.calculation_method,
            "baseline_period": self.baseline_period,
            "adjustment_factors": self.adjustment_factors,
            "measurement_points": self.measurement_points,
            "data_quality_score": self.data_quality_score,
            "uncertainty_percent": self.uncertainty_percent,
            "calculation_hash": self.calculation_hash,
            "evidence_records": self.evidence_records,
        }


# =============================================================================
# CO2E ACCOUNTING
# =============================================================================

@dataclass
class CO2eAccountingRecord:
    """Record for CO2 equivalent emissions accounting."""

    record_id: str = ""
    exchanger_id: str = ""
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Energy and emissions
    energy_consumed_mj: float = 0.0
    fuel_type: str = ""
    emission_factor_kg_co2e_per_mj: float = 0.0
    emissions_kg_co2e: float = 0.0

    # Avoided emissions from optimization
    baseline_emissions_kg_co2e: float = 0.0
    actual_emissions_kg_co2e: float = 0.0
    avoided_emissions_kg_co2e: float = 0.0
    avoided_emissions_percent: float = 0.0

    # Scope categorization (GHG Protocol)
    scope: int = 1  # 1, 2, or 3
    scope_category: str = ""  # e.g., "stationary combustion", "purchased heat"

    # Methodology
    calculation_method: str = ""
    emission_factor_source: str = ""
    gwp_values: Dict[str, float] = field(default_factory=dict)

    # Verification
    data_source: str = ""
    verification_status: str = "unverified"
    verifier: Optional[str] = None
    verification_date: Optional[datetime] = None

    # Provenance
    calculation_hash: str = ""
    supporting_evidence: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize record ID if not provided."""
        if not self.record_id:
            self.record_id = f"CO2E-{uuid.uuid4().hex[:12].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "exchanger_id": self.exchanger_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "energy_consumed_mj": self.energy_consumed_mj,
            "fuel_type": self.fuel_type,
            "emission_factor_kg_co2e_per_mj": self.emission_factor_kg_co2e_per_mj,
            "emissions_kg_co2e": self.emissions_kg_co2e,
            "baseline_emissions_kg_co2e": self.baseline_emissions_kg_co2e,
            "actual_emissions_kg_co2e": self.actual_emissions_kg_co2e,
            "avoided_emissions_kg_co2e": self.avoided_emissions_kg_co2e,
            "avoided_emissions_percent": self.avoided_emissions_percent,
            "scope": self.scope,
            "scope_category": self.scope_category,
            "calculation_method": self.calculation_method,
            "emission_factor_source": self.emission_factor_source,
            "gwp_values": self.gwp_values,
            "data_source": self.data_source,
            "verification_status": self.verification_status,
            "verifier": self.verifier,
            "verification_date": self.verification_date.isoformat() if self.verification_date else None,
            "calculation_hash": self.calculation_hash,
            "supporting_evidence": self.supporting_evidence,
        }


# =============================================================================
# COMPLIANCE REPORT
# =============================================================================

@dataclass
class ComplianceReport:
    """
    Complete compliance report with all verification data.

    Comprehensive report that can be submitted to regulators
    or used for internal audits.
    """

    report_id: str = ""
    report_type: ReportType = ReportType.AUDIT_SUMMARY
    report_title: str = ""
    report_status: ReportStatus = ReportStatus.DRAFT

    # Scope
    exchangers_covered: List[str] = field(default_factory=list)
    frameworks: List[ComplianceFramework] = field(default_factory=list)
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Compliance summary
    overall_compliant: bool = True
    compliance_score: float = 1.0
    checks_performed: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    checks_warning: int = 0

    # Detailed results
    tema_checks: List[TEMACheck] = field(default_factory=list)
    energy_savings_records: List[EnergySavingsRecord] = field(default_factory=list)
    co2e_accounting_records: List[CO2eAccountingRecord] = field(default_factory=list)

    # Violations and recommendations
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    corrective_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: str = "GL-014-EXCHANGERPRO"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    # Evidence
    evidence_pack_id: Optional[str] = None
    supporting_documents: List[str] = field(default_factory=list)

    # Provenance
    report_hash: str = ""

    def __post_init__(self) -> None:
        """Initialize report ID if not provided."""
        if not self.report_id:
            self.report_id = f"RPT-{uuid.uuid4().hex[:12].upper()}"

    def calculate_compliance_score(self) -> float:
        """Calculate overall compliance score."""
        if self.checks_performed == 0:
            return 1.0

        # Weight: passed = 1.0, warning = 0.7, failed = 0
        weighted_score = (
            self.checks_passed * 1.0 +
            self.checks_warning * 0.7 +
            self.checks_failed * 0.0
        )
        self.compliance_score = weighted_score / self.checks_performed
        return self.compliance_score

    def update_counts(self) -> None:
        """Update check counts from TEMA checks."""
        self.checks_performed = len(self.tema_checks)
        self.checks_passed = sum(1 for c in self.tema_checks if c.is_compliant and c.severity == "info")
        self.checks_warning = sum(1 for c in self.tema_checks if c.severity == "warning")
        self.checks_failed = sum(1 for c in self.tema_checks if not c.is_compliant or c.severity == "critical")
        self.overall_compliant = self.checks_failed == 0

    def compute_report_hash(self) -> str:
        """Compute SHA-256 hash of the report."""
        hash_data = {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "exchangers_covered": self.exchangers_covered,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "compliance_score": self.compliance_score,
            "tema_checks": [c.to_dict() for c in self.tema_checks],
            "generated_at": self.generated_at.isoformat(),
        }
        self.report_hash = compute_sha256(hash_data)
        return self.report_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "report_title": self.report_title,
            "report_status": self.report_status.value,
            "exchangers_covered": self.exchangers_covered,
            "frameworks": [f.value for f in self.frameworks],
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "overall_compliant": self.overall_compliant,
            "compliance_score": self.compliance_score,
            "checks_performed": self.checks_performed,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_warning": self.checks_warning,
            "tema_checks": [c.to_dict() for c in self.tema_checks],
            "energy_savings_records": [r.to_dict() for r in self.energy_savings_records],
            "co2e_accounting_records": [r.to_dict() for r in self.co2e_accounting_records],
            "violations": self.violations,
            "recommendations": self.recommendations,
            "corrective_actions": self.corrective_actions,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "evidence_pack_id": self.evidence_pack_id,
            "supporting_documents": self.supporting_documents,
            "report_hash": self.report_hash,
        }


# =============================================================================
# COMPLIANCE REPORTER
# =============================================================================

class ComplianceReporter:
    """
    Compliance Reporter for GL-014 Exchangerpro.

    Generates comprehensive regulatory compliance reports including
    TEMA verification, energy savings, and CO2e accounting.

    Usage:
        >>> reporter = ComplianceReporter()
        >>> report = reporter.generate_tema_compliance_report(
        ...     exchanger_id="HEX-001",
        ...     period_start=datetime(2024, 1, 1),
        ...     period_end=datetime(2024, 12, 31)
        ... )
        >>> reporter.add_energy_savings_verification(report, energy_data)
        >>> reporter.add_co2e_accounting(report, emissions_data)
        >>> report = reporter.finalize_report(report)
        >>> exported = reporter.export_report(report, format="pdf")
    """

    VERSION = "1.0.0"

    # TEMA standard fouling resistance limits (m2K/W)
    TEMA_FOULING_LIMITS = {
        "cooling_water_treated": 0.000176,
        "cooling_water_untreated": 0.000352,
        "city_water": 0.000176,
        "river_water": 0.000352,
        "sea_water": 0.000176,
        "boiler_feed_water": 0.000088,
        "condensate": 0.000088,
        "steam": 0.0000,
        "refrigerant": 0.000176,
        "hydrocarbon_light": 0.000176,
        "hydrocarbon_heavy": 0.000352,
        "fuel_oil": 0.000528,
        "crude_oil": 0.000352,
    }

    # Standard emission factors (kg CO2e/MJ)
    EMISSION_FACTORS = {
        "natural_gas": 0.0561,
        "fuel_oil_heavy": 0.0774,
        "fuel_oil_light": 0.0733,
        "coal": 0.0946,
        "electricity_grid_average": 0.1111,  # Varies by region
        "steam_from_natural_gas": 0.0673,
    }

    def __init__(self, evidence_generator: Optional[EvidenceGenerator] = None):
        """
        Initialize compliance reporter.

        Args:
            evidence_generator: Optional evidence generator for creating sealed packs
        """
        self.evidence_generator = evidence_generator or EvidenceGenerator()
        self._reports: Dict[str, ComplianceReport] = {}

        logger.info("ComplianceReporter initialized")

    # =========================================================================
    # TEMA COMPLIANCE
    # =========================================================================

    def generate_tema_compliance_report(
        self,
        exchanger_id: str,
        period_start: datetime,
        period_end: datetime,
        operating_data: Optional[Dict[str, Any]] = None,
        design_data: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReport:
        """
        Generate TEMA compliance verification report.

        Args:
            exchanger_id: Heat exchanger identifier
            period_start: Start of reporting period
            period_end: End of reporting period
            operating_data: Operating performance data
            design_data: Design specifications

        Returns:
            ComplianceReport with TEMA checks
        """
        report = ComplianceReport(
            report_type=ReportType.TEMA_COMPLIANCE,
            report_title=f"TEMA Compliance Report - {exchanger_id}",
            exchangers_covered=[exchanger_id],
            frameworks=[ComplianceFramework.TEMA, ComplianceFramework.ASME_PTC_12_5],
            period_start=period_start,
            period_end=period_end,
        )

        operating_data = operating_data or {}
        design_data = design_data or {}

        # Run TEMA compliance checks
        tema_checks = []

        # Check 1: Fouling resistance (TEMA RCB-2.6)
        if "fouling_resistance" in operating_data:
            tema_checks.append(self._check_fouling_resistance(
                exchanger_id=exchanger_id,
                measured_rf=operating_data["fouling_resistance"],
                fluid_type=operating_data.get("fluid_type", "cooling_water_treated"),
                design_rf=design_data.get("design_fouling_resistance"),
            ))

        # Check 2: Pressure drop (TEMA RCB-4.6)
        if "pressure_drop_shell" in operating_data:
            tema_checks.append(self._check_pressure_drop(
                exchanger_id=exchanger_id,
                side="shell",
                measured_dp=operating_data["pressure_drop_shell"],
                design_dp=design_data.get("design_pressure_drop_shell", 0),
            ))

        if "pressure_drop_tube" in operating_data:
            tema_checks.append(self._check_pressure_drop(
                exchanger_id=exchanger_id,
                side="tube",
                measured_dp=operating_data["pressure_drop_tube"],
                design_dp=design_data.get("design_pressure_drop_tube", 0),
            ))

        # Check 3: Heat transfer coefficient (TEMA RCB-4.7)
        if "overall_htc" in operating_data:
            tema_checks.append(self._check_heat_transfer_coefficient(
                exchanger_id=exchanger_id,
                measured_u=operating_data["overall_htc"],
                design_u=design_data.get("design_u", 0),
                clean_u=design_data.get("clean_u", 0),
            ))

        # Check 4: Approach temperature
        if "approach_temperature" in operating_data:
            tema_checks.append(self._check_approach_temperature(
                exchanger_id=exchanger_id,
                measured_approach=operating_data["approach_temperature"],
                design_approach=design_data.get("design_approach", 10.0),
            ))

        # Check 5: Thermal effectiveness
        if "effectiveness" in operating_data:
            tema_checks.append(self._check_effectiveness(
                exchanger_id=exchanger_id,
                measured_effectiveness=operating_data["effectiveness"],
                design_effectiveness=design_data.get("design_effectiveness", 0.7),
            ))

        report.tema_checks = tema_checks
        report.update_counts()
        report.calculate_compliance_score()

        # Generate recommendations
        report.recommendations = self._generate_tema_recommendations(tema_checks)
        report.violations = [
            {"check_id": c.check_id, "tema_reference": c.tema_reference, "notes": c.notes}
            for c in tema_checks if not c.is_compliant
        ]

        self._reports[report.report_id] = report

        logger.info(
            f"Generated TEMA compliance report {report.report_id}: "
            f"{report.checks_passed}/{report.checks_performed} passed, "
            f"score={report.compliance_score:.2%}"
        )

        return report

    def _check_fouling_resistance(
        self,
        exchanger_id: str,
        measured_rf: float,
        fluid_type: str,
        design_rf: Optional[float] = None,
    ) -> TEMACheck:
        """Check fouling resistance against TEMA limits."""
        tema_limit = self.TEMA_FOULING_LIMITS.get(fluid_type, 0.000352)
        design_limit = design_rf if design_rf else tema_limit

        is_compliant = measured_rf <= design_limit * 1.5  # Allow 50% over design
        deviation = ((measured_rf - design_limit) / design_limit * 100) if design_limit > 0 else 0

        severity = "info"
        if measured_rf > design_limit * 1.5:
            severity = "critical"
        elif measured_rf > design_limit:
            severity = "warning"

        return TEMACheck(
            check_name="Fouling Resistance",
            tema_reference="RCB-2.6",
            category="fouling",
            parameter_name="fouling_resistance",
            parameter_value=measured_rf,
            parameter_unit="m2K/W",
            tema_limit=design_limit,
            tema_limit_type="max",
            is_compliant=is_compliant,
            deviation_percent=deviation,
            severity=severity,
            notes=f"Fluid type: {fluid_type}. TEMA standard limit: {tema_limit:.6f} m2K/W",
            evidence_hash=compute_sha256({
                "exchanger_id": exchanger_id,
                "measured_rf": measured_rf,
                "fluid_type": fluid_type,
            }),
        )

    def _check_pressure_drop(
        self,
        exchanger_id: str,
        side: str,
        measured_dp: float,
        design_dp: float,
    ) -> TEMACheck:
        """Check pressure drop against design limits."""
        is_compliant = measured_dp <= design_dp * 1.2  # Allow 20% over design
        deviation = ((measured_dp - design_dp) / design_dp * 100) if design_dp > 0 else 0

        severity = "info"
        if measured_dp > design_dp * 1.5:
            severity = "critical"
        elif measured_dp > design_dp * 1.2:
            severity = "warning"

        return TEMACheck(
            check_name=f"Pressure Drop ({side.title()} Side)",
            tema_reference="RCB-4.6",
            category="pressure_drop",
            parameter_name=f"pressure_drop_{side}",
            parameter_value=measured_dp,
            parameter_unit="kPa",
            tema_limit=design_dp,
            tema_limit_type="max",
            is_compliant=is_compliant,
            deviation_percent=deviation,
            severity=severity,
            notes=f"Design pressure drop: {design_dp} kPa. Measured: {measured_dp} kPa",
            evidence_hash=compute_sha256({
                "exchanger_id": exchanger_id,
                "side": side,
                "measured_dp": measured_dp,
            }),
        )

    def _check_heat_transfer_coefficient(
        self,
        exchanger_id: str,
        measured_u: float,
        design_u: float,
        clean_u: float,
    ) -> TEMACheck:
        """Check overall heat transfer coefficient."""
        # U should be at least 60% of design for acceptable performance
        min_acceptable_u = design_u * 0.6
        is_compliant = measured_u >= min_acceptable_u

        deviation = ((design_u - measured_u) / design_u * 100) if design_u > 0 else 0
        cleanliness_factor = (measured_u / clean_u * 100) if clean_u > 0 else 100

        severity = "info"
        if measured_u < design_u * 0.5:
            severity = "critical"
        elif measured_u < design_u * 0.7:
            severity = "warning"

        return TEMACheck(
            check_name="Overall Heat Transfer Coefficient",
            tema_reference="RCB-4.7",
            category="thermal_design",
            parameter_name="overall_htc",
            parameter_value=measured_u,
            parameter_unit="W/m2K",
            tema_limit=min_acceptable_u,
            tema_limit_type="min",
            is_compliant=is_compliant,
            deviation_percent=deviation,
            severity=severity,
            notes=f"Design U: {design_u} W/m2K. Cleanliness factor: {cleanliness_factor:.1f}%",
            evidence_hash=compute_sha256({
                "exchanger_id": exchanger_id,
                "measured_u": measured_u,
                "design_u": design_u,
            }),
        )

    def _check_approach_temperature(
        self,
        exchanger_id: str,
        measured_approach: float,
        design_approach: float,
    ) -> TEMACheck:
        """Check approach temperature."""
        is_compliant = measured_approach <= design_approach * 1.5
        deviation = ((measured_approach - design_approach) / design_approach * 100) if design_approach > 0 else 0

        severity = "info"
        if measured_approach > design_approach * 2.0:
            severity = "critical"
        elif measured_approach > design_approach * 1.5:
            severity = "warning"

        return TEMACheck(
            check_name="Approach Temperature",
            tema_reference="General",
            category="thermal_design",
            parameter_name="approach_temperature",
            parameter_value=measured_approach,
            parameter_unit="K",
            tema_limit=design_approach,
            tema_limit_type="max",
            is_compliant=is_compliant,
            deviation_percent=deviation,
            severity=severity,
            notes=f"Design approach: {design_approach} K",
            evidence_hash=compute_sha256({
                "exchanger_id": exchanger_id,
                "measured_approach": measured_approach,
            }),
        )

    def _check_effectiveness(
        self,
        exchanger_id: str,
        measured_effectiveness: float,
        design_effectiveness: float,
    ) -> TEMACheck:
        """Check thermal effectiveness."""
        is_compliant = measured_effectiveness >= design_effectiveness * 0.85
        deviation = ((design_effectiveness - measured_effectiveness) / design_effectiveness * 100) if design_effectiveness > 0 else 0

        severity = "info"
        if measured_effectiveness < design_effectiveness * 0.7:
            severity = "critical"
        elif measured_effectiveness < design_effectiveness * 0.85:
            severity = "warning"

        return TEMACheck(
            check_name="Thermal Effectiveness",
            tema_reference="General",
            category="thermal_design",
            parameter_name="effectiveness",
            parameter_value=measured_effectiveness,
            parameter_unit="dimensionless",
            tema_limit=design_effectiveness * 0.85,
            tema_limit_type="min",
            is_compliant=is_compliant,
            deviation_percent=deviation,
            severity=severity,
            notes=f"Design effectiveness: {design_effectiveness:.2%}",
            evidence_hash=compute_sha256({
                "exchanger_id": exchanger_id,
                "measured_effectiveness": measured_effectiveness,
            }),
        )

    def _generate_tema_recommendations(self, checks: List[TEMACheck]) -> List[str]:
        """Generate recommendations based on TEMA check results."""
        recommendations = []

        for check in checks:
            if not check.is_compliant:
                if check.category == "fouling":
                    recommendations.append(
                        f"Schedule cleaning for {check.parameter_name}. "
                        f"Current value {check.parameter_value:.6f} exceeds limit."
                    )
                elif check.category == "pressure_drop":
                    recommendations.append(
                        f"Investigate pressure drop increase on {check.parameter_name}. "
                        f"May indicate fouling or flow restrictions."
                    )
                elif check.category == "thermal_design":
                    recommendations.append(
                        f"Review thermal performance: {check.check_name}. "
                        f"Deviation: {check.deviation_percent:.1f}%"
                    )

        return recommendations

    # =========================================================================
    # ENERGY SAVINGS VERIFICATION
    # =========================================================================

    def add_energy_savings_verification(
        self,
        report: ComplianceReport,
        exchanger_id: str,
        baseline_energy_mj: float,
        actual_energy_mj: float,
        energy_cost_rate: float,
        calculation_method: str = "IPMVP Option B",
        baseline_period: str = "",
        adjustment_factors: Optional[Dict[str, float]] = None,
    ) -> EnergySavingsRecord:
        """
        Add energy savings verification to a compliance report.

        Args:
            report: Compliance report to add to
            exchanger_id: Heat exchanger identifier
            baseline_energy_mj: Baseline period energy consumption
            actual_energy_mj: Actual period energy consumption
            energy_cost_rate: Energy cost in $/MJ
            calculation_method: M&V calculation method
            baseline_period: Description of baseline period
            adjustment_factors: Adjustment factors applied

        Returns:
            EnergySavingsRecord
        """
        savings_mj = baseline_energy_mj - actual_energy_mj
        savings_percent = (savings_mj / baseline_energy_mj * 100) if baseline_energy_mj > 0 else 0
        cost_savings = savings_mj * energy_cost_rate

        record = EnergySavingsRecord(
            exchanger_id=exchanger_id,
            period_start=report.period_start,
            period_end=report.period_end,
            baseline_energy_mj=baseline_energy_mj,
            actual_energy_mj=actual_energy_mj,
            savings_mj=savings_mj,
            savings_percent=savings_percent,
            energy_cost_rate=energy_cost_rate,
            cost_savings_usd=cost_savings,
            calculation_method=calculation_method,
            baseline_period=baseline_period,
            adjustment_factors=adjustment_factors or {},
        )

        # Compute calculation hash
        record.calculation_hash = compute_sha256(record.to_dict())

        report.energy_savings_records.append(record)

        # Add ISO 50001 framework if not present
        if ComplianceFramework.ISO_50001 not in report.frameworks:
            report.frameworks.append(ComplianceFramework.ISO_50001)

        logger.info(
            f"Added energy savings verification for {exchanger_id}: "
            f"{savings_mj:.2f} MJ ({savings_percent:.1f}%) saved"
        )

        return record

    # =========================================================================
    # CO2E ACCOUNTING
    # =========================================================================

    def add_co2e_accounting(
        self,
        report: ComplianceReport,
        exchanger_id: str,
        energy_consumed_mj: float,
        fuel_type: str,
        baseline_energy_mj: Optional[float] = None,
        scope: int = 1,
        scope_category: str = "stationary combustion",
    ) -> CO2eAccountingRecord:
        """
        Add CO2e emissions accounting to a compliance report.

        Args:
            report: Compliance report to add to
            exchanger_id: Heat exchanger identifier
            energy_consumed_mj: Energy consumed in MJ
            fuel_type: Type of fuel (for emission factor lookup)
            baseline_energy_mj: Baseline energy for avoided emissions calc
            scope: GHG Protocol scope (1, 2, or 3)
            scope_category: Category within scope

        Returns:
            CO2eAccountingRecord
        """
        # Get emission factor
        emission_factor = self.EMISSION_FACTORS.get(fuel_type, 0.0561)

        # Calculate emissions
        emissions_kg_co2e = energy_consumed_mj * emission_factor

        # Calculate avoided emissions
        baseline_emissions = 0.0
        avoided_emissions = 0.0
        avoided_percent = 0.0

        if baseline_energy_mj:
            baseline_emissions = baseline_energy_mj * emission_factor
            avoided_emissions = baseline_emissions - emissions_kg_co2e
            avoided_percent = (avoided_emissions / baseline_emissions * 100) if baseline_emissions > 0 else 0

        record = CO2eAccountingRecord(
            exchanger_id=exchanger_id,
            period_start=report.period_start,
            period_end=report.period_end,
            energy_consumed_mj=energy_consumed_mj,
            fuel_type=fuel_type,
            emission_factor_kg_co2e_per_mj=emission_factor,
            emissions_kg_co2e=emissions_kg_co2e,
            baseline_emissions_kg_co2e=baseline_emissions,
            actual_emissions_kg_co2e=emissions_kg_co2e,
            avoided_emissions_kg_co2e=avoided_emissions,
            avoided_emissions_percent=avoided_percent,
            scope=scope,
            scope_category=scope_category,
            calculation_method="GHG Protocol",
            emission_factor_source="EPA 40 CFR Part 98",
        )

        # Compute calculation hash
        record.calculation_hash = compute_sha256(record.to_dict())

        report.co2e_accounting_records.append(record)

        # Add EPA framework if not present
        if ComplianceFramework.EPA_40_CFR_98 not in report.frameworks:
            report.frameworks.append(ComplianceFramework.EPA_40_CFR_98)
        if ComplianceFramework.ISO_14064 not in report.frameworks:
            report.frameworks.append(ComplianceFramework.ISO_14064)

        logger.info(
            f"Added CO2e accounting for {exchanger_id}: "
            f"{emissions_kg_co2e:.2f} kg CO2e, {avoided_emissions:.2f} kg avoided"
        )

        return record

    # =========================================================================
    # REPORT FINALIZATION
    # =========================================================================

    def finalize_report(
        self,
        report: ComplianceReport,
        create_evidence_pack: bool = True,
    ) -> ComplianceReport:
        """
        Finalize a compliance report with evidence packaging.

        Args:
            report: Report to finalize
            create_evidence_pack: Whether to create sealed evidence pack

        Returns:
            Finalized compliance report
        """
        # Update counts and score
        report.update_counts()
        report.calculate_compliance_score()
        report.compute_report_hash()

        # Create evidence pack
        if create_evidence_pack:
            evidence_records = []

            # Create evidence record for each TEMA check
            for check in report.tema_checks:
                evidence_records.append(
                    self.evidence_generator.create_evidence_record(
                        evidence_type=EvidenceType.TEMA_COMPLIANCE,
                        data=check.to_dict(),
                        inputs={"parameter_value": check.parameter_value},
                        outputs={"is_compliant": check.is_compliant},
                        exchanger_id=check.check_id.split("-")[0] if check.check_id else "",
                    )
                )

            if evidence_records:
                pack = self.evidence_generator.create_evidence_pack(
                    evidence_records,
                    exchangers_covered=report.exchangers_covered,
                )
                sealed_pack = self.evidence_generator.seal_pack(pack)
                import json as json_module
                envelope = json_module.loads(sealed_pack)
                report.evidence_pack_id = envelope.get("pack_id")

        report.report_status = ReportStatus.PENDING_REVIEW

        logger.info(
            f"Finalized compliance report {report.report_id}: "
            f"score={report.compliance_score:.2%}, evidence_pack={report.evidence_pack_id}"
        )

        return report

    def approve_report(
        self,
        report: ComplianceReport,
        approved_by: str,
    ) -> ComplianceReport:
        """
        Approve a compliance report.

        Args:
            report: Report to approve
            approved_by: Approver identifier

        Returns:
            Approved report
        """
        report.approved_by = approved_by
        report.approved_at = datetime.now(timezone.utc)
        report.report_status = ReportStatus.APPROVED
        report.compute_report_hash()

        logger.info(f"Approved compliance report {report.report_id} by {approved_by}")

        return report

    # =========================================================================
    # REPORT EXPORT
    # =========================================================================

    def export_report(
        self,
        report: ComplianceReport,
        format: str = "json",
    ) -> bytes:
        """
        Export compliance report in specified format.

        Args:
            report: Report to export
            format: Export format ("json", "xml", "csv", "pdf")

        Returns:
            Exported report as bytes
        """
        if format == "json":
            return self._export_json(report)
        elif format == "xml":
            return self._export_xml(report)
        elif format == "csv":
            return self._export_csv(report)
        elif format == "pdf":
            return self._export_pdf_placeholder(report)
        else:
            return self._export_json(report)

    def _export_json(self, report: ComplianceReport) -> bytes:
        """Export as JSON."""
        return json.dumps(report.to_dict(), indent=2, default=str).encode('utf-8')

    def _export_xml(self, report: ComplianceReport) -> bytes:
        """Export as XML."""
        import xml.etree.ElementTree as ET

        root = ET.Element("ComplianceReport")
        root.set("id", report.report_id)
        root.set("type", report.report_type.value)

        # Summary
        summary = ET.SubElement(root, "Summary")
        ET.SubElement(summary, "Title").text = report.report_title
        ET.SubElement(summary, "Status").text = report.report_status.value
        ET.SubElement(summary, "OverallCompliant").text = str(report.overall_compliant)
        ET.SubElement(summary, "ComplianceScore").text = f"{report.compliance_score:.2%}"

        # Period
        period = ET.SubElement(root, "ReportingPeriod")
        ET.SubElement(period, "Start").text = report.period_start.isoformat()
        ET.SubElement(period, "End").text = report.period_end.isoformat()

        # Exchangers
        exchangers = ET.SubElement(root, "ExchangersCovered")
        for ex_id in report.exchangers_covered:
            ET.SubElement(exchangers, "Exchanger").text = ex_id

        # TEMA Checks
        tema_section = ET.SubElement(root, "TEMAChecks")
        for check in report.tema_checks:
            check_elem = ET.SubElement(tema_section, "Check")
            check_elem.set("id", check.check_id)
            check_elem.set("reference", check.tema_reference)
            ET.SubElement(check_elem, "Name").text = check.check_name
            ET.SubElement(check_elem, "Value").text = str(check.parameter_value)
            ET.SubElement(check_elem, "Unit").text = check.parameter_unit
            ET.SubElement(check_elem, "Compliant").text = str(check.is_compliant)
            ET.SubElement(check_elem, "Severity").text = check.severity

        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def _export_csv(self, report: ComplianceReport) -> bytes:
        """Export TEMA checks as CSV."""
        lines = ["check_id,check_name,tema_reference,parameter_value,parameter_unit,tema_limit,is_compliant,severity"]

        for check in report.tema_checks:
            lines.append(
                f"{check.check_id},{check.check_name},{check.tema_reference},"
                f"{check.parameter_value},{check.parameter_unit},{check.tema_limit},"
                f"{check.is_compliant},{check.severity}"
            )

        return "\n".join(lines).encode('utf-8')

    def _export_pdf_placeholder(self, report: ComplianceReport) -> bytes:
        """Generate PDF placeholder (actual implementation would use reportlab)."""
        text = f"""
GL-014 EXCHANGERPRO COMPLIANCE REPORT
=====================================

Report ID: {report.report_id}
Report Type: {report.report_type.value}
Title: {report.report_title}
Status: {report.report_status.value}

PERIOD: {report.period_start.date()} to {report.period_end.date()}

COMPLIANCE SUMMARY
------------------
Overall Compliant: {'YES' if report.overall_compliant else 'NO'}
Compliance Score: {report.compliance_score:.1%}
Checks Performed: {report.checks_performed}
Checks Passed: {report.checks_passed}
Checks Failed: {report.checks_failed}
Checks Warning: {report.checks_warning}

EXCHANGERS COVERED
------------------
{chr(10).join('- ' + ex for ex in report.exchangers_covered)}

FRAMEWORKS
----------
{chr(10).join('- ' + f.value for f in report.frameworks)}

TEMA CHECKS SUMMARY
-------------------
"""
        for check in report.tema_checks:
            status = "PASS" if check.is_compliant else "FAIL"
            text += f"[{status}] {check.check_name} ({check.tema_reference}): {check.parameter_value} {check.parameter_unit}\n"

        text += f"""
RECOMMENDATIONS
---------------
{chr(10).join('- ' + r for r in report.recommendations) if report.recommendations else 'None'}

Generated: {report.generated_at.isoformat()}
Generated By: {report.generated_by}
Report Hash: {report.report_hash}
Evidence Pack: {report.evidence_pack_id or 'Not created'}
        """

        return text.encode('utf-8')

    # =========================================================================
    # AUDIT SUMMARY GENERATION
    # =========================================================================

    def generate_audit_summary(
        self,
        exchanger_ids: List[str],
        period_start: datetime,
        period_end: datetime,
        audit_statistics: Optional[AuditStatistics] = None,
    ) -> ComplianceReport:
        """
        Generate comprehensive audit summary for multiple exchangers.

        Args:
            exchanger_ids: List of exchanger identifiers
            period_start: Start of audit period
            period_end: End of audit period
            audit_statistics: Optional pre-computed statistics

        Returns:
            ComplianceReport with audit summary
        """
        report = ComplianceReport(
            report_type=ReportType.AUDIT_SUMMARY,
            report_title=f"Audit Summary - {period_start.date()} to {period_end.date()}",
            exchangers_covered=exchanger_ids,
            frameworks=[
                ComplianceFramework.TEMA,
                ComplianceFramework.ISO_50001,
                ComplianceFramework.CFR_21_PART_11,
            ],
            period_start=period_start,
            period_end=period_end,
        )

        if audit_statistics:
            report.checks_performed = audit_statistics.total_records
            # Additional statistics processing would go here

        self._reports[report.report_id] = report

        logger.info(
            f"Generated audit summary {report.report_id} for {len(exchanger_ids)} exchangers"
        )

        return report
