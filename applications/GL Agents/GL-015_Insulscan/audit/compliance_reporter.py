# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Compliance Reporter

Comprehensive regulatory compliance reporting for insulation scanning
and thermal assessment operations. Supports ISO 50001 Energy Management
and ASHRAE standards compliance with gap analysis and remediation tracking.

Features:
    - ISO 50001 Energy Management compliance
    - ASHRAE 90.1 and 189.1 standards compliance
    - ASTM C1060 (Thermographic Inspection) compliance
    - ASTM C680 (Heat Loss Calculations) compliance
    - Generate compliance reports
    - Gap analysis for non-compliance
    - Remediation tracking
    - Energy savings verification
    - CO2e emissions accounting

Standards:
    - ISO 50001:2018 (Energy Management Systems)
    - ASHRAE 90.1 (Energy Standard for Buildings)
    - ASHRAE 189.1 (High-Performance Buildings)
    - ASTM C1060 (Thermographic Inspection)
    - ASTM C680 (Heat Gain/Loss Calculation)
    - EPA 40 CFR Part 98 (GHG Reporting)
    - 21 CFR Part 11 (Electronic Records)

Example:
    >>> from audit.compliance_reporter import InsulationComplianceReporter
    >>> reporter = InsulationComplianceReporter()
    >>> report = reporter.generate_iso_50001_report(
    ...     asset_ids=["INSUL-001", "INSUL-002"],
    ...     period_start=datetime(2024, 1, 1),
    ...     period_end=datetime(2024, 12, 31)
    ... )
    >>> gap_analysis = reporter.analyze_compliance_gaps(report)
    >>> exported = reporter.export_report(report, format="pdf")
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
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .schemas import (
    ComplianceRecord,
    ComplianceStandard,
    ComplianceRequirementStatus,
    ComputationType,
    compute_sha256,
)
from .evidence_generator import (
    InsulationEvidenceGenerator,
    InsulationEvidenceRecord,
    EvidenceType,
    SealStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COMPLIANCE FRAMEWORK ENUMS
# =============================================================================

class ComplianceFramework(str, Enum):
    """Supported regulatory compliance frameworks."""

    ISO_50001 = "iso_50001"
    ASHRAE_90_1 = "ashrae_90_1"
    ASHRAE_189_1 = "ashrae_189_1"
    ASTM_C1060 = "astm_c1060"
    ASTM_C680 = "astm_c680"
    EPA_40_CFR_98 = "epa_40_cfr_98"
    IECC = "iecc"


class ReportType(str, Enum):
    """Types of compliance reports."""

    ISO_50001_COMPLIANCE = "iso_50001_compliance"
    ASHRAE_COMPLIANCE = "ashrae_compliance"
    ENERGY_PERFORMANCE = "energy_performance"
    INSULATION_ASSESSMENT = "insulation_assessment"
    THERMAL_SURVEY = "thermal_survey"
    HEAT_LOSS_ANALYSIS = "heat_loss_analysis"
    CO2E_ACCOUNTING = "co2e_accounting"
    REGULATORY_SUBMISSION = "regulatory_submission"
    AUDIT_SUMMARY = "audit_summary"
    GAP_ANALYSIS = "gap_analysis"


class ReportStatus(str, Enum):
    """Status of compliance report."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    ARCHIVED = "archived"


class RemediationPriority(str, Enum):
    """Priority for remediation actions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# ISO 50001 COMPLIANCE CHECK
# =============================================================================

@dataclass
class ISO50001Check:
    """Individual ISO 50001 compliance check result."""

    check_id: str = ""
    check_name: str = ""
    clause_reference: str = ""  # e.g., "4.4.3.1", "8.1"
    category: str = ""  # e.g., "energy_planning", "monitoring", "improvement"

    # Check parameters
    requirement: str = ""
    measured_value: Optional[float] = None
    target_value: Optional[float] = None
    unit: str = ""

    # Results
    is_compliant: bool = True
    deviation_percent: float = 0.0
    severity: str = "info"  # "info", "warning", "critical"
    finding: str = ""
    evidence_collected: bool = False

    # Traceability
    evidence_ids: List[str] = field(default_factory=list)
    evidence_hash: str = ""
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Initialize check ID if not provided."""
        if not self.check_id:
            self.check_id = f"ISO50001-CHK-{uuid.uuid4().hex[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_id": self.check_id,
            "check_name": self.check_name,
            "clause_reference": self.clause_reference,
            "category": self.category,
            "requirement": self.requirement,
            "measured_value": self.measured_value,
            "target_value": self.target_value,
            "unit": self.unit,
            "is_compliant": self.is_compliant,
            "deviation_percent": self.deviation_percent,
            "severity": self.severity,
            "finding": self.finding,
            "evidence_collected": self.evidence_collected,
            "evidence_ids": self.evidence_ids,
            "evidence_hash": self.evidence_hash,
            "checked_at": self.checked_at.isoformat(),
        }


# =============================================================================
# ASHRAE COMPLIANCE CHECK
# =============================================================================

@dataclass
class ASHRAECheck:
    """Individual ASHRAE standards compliance check result."""

    check_id: str = ""
    check_name: str = ""
    standard: str = "90.1"  # "90.1", "189.1"
    section: str = ""  # e.g., "5.8.1", "7.4.3"
    category: str = ""  # e.g., "envelope", "insulation", "fenestration"

    # Check parameters
    requirement: str = ""
    climate_zone: str = ""
    building_type: str = ""

    # R-value requirements
    required_r_value: Optional[float] = None
    actual_r_value: Optional[float] = None
    r_value_unit: str = "m2K/W"

    # U-value requirements
    required_u_value: Optional[float] = None
    actual_u_value: Optional[float] = None
    u_value_unit: str = "W/m2K"

    # Insulation specific
    insulation_type: str = ""
    application: str = ""  # e.g., "pipe", "duct", "wall", "roof"

    # Results
    is_compliant: bool = True
    deviation_percent: float = 0.0
    severity: str = "info"
    finding: str = ""
    prescriptive_path: bool = True  # vs performance path

    # Traceability
    calculation_id: Optional[str] = None
    evidence_hash: str = ""
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Initialize check ID if not provided."""
        if not self.check_id:
            self.check_id = f"ASHRAE-CHK-{uuid.uuid4().hex[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_id": self.check_id,
            "check_name": self.check_name,
            "standard": self.standard,
            "section": self.section,
            "category": self.category,
            "requirement": self.requirement,
            "climate_zone": self.climate_zone,
            "building_type": self.building_type,
            "required_r_value": self.required_r_value,
            "actual_r_value": self.actual_r_value,
            "r_value_unit": self.r_value_unit,
            "required_u_value": self.required_u_value,
            "actual_u_value": self.actual_u_value,
            "u_value_unit": self.u_value_unit,
            "insulation_type": self.insulation_type,
            "application": self.application,
            "is_compliant": self.is_compliant,
            "deviation_percent": self.deviation_percent,
            "severity": self.severity,
            "finding": self.finding,
            "prescriptive_path": self.prescriptive_path,
            "calculation_id": self.calculation_id,
            "evidence_hash": self.evidence_hash,
            "checked_at": self.checked_at.isoformat(),
        }


# =============================================================================
# ENERGY PERFORMANCE RECORD
# =============================================================================

@dataclass
class EnergyPerformanceRecord:
    """Record of energy performance from insulation assessment."""

    record_id: str = ""
    asset_id: str = ""
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Heat loss performance
    baseline_heat_loss_w_m2: float = 0.0
    actual_heat_loss_w_m2: float = 0.0
    heat_loss_reduction_percent: float = 0.0

    # Energy savings
    baseline_energy_mj: float = 0.0
    actual_energy_mj: float = 0.0
    savings_mj: float = 0.0
    savings_percent: float = 0.0

    # Cost savings
    energy_cost_rate: float = 0.0  # $/MJ
    cost_savings_usd: float = 0.0

    # Performance metrics
    insulation_efficiency: float = 1.0
    thermal_resistance_factor: float = 1.0
    condition_degradation: float = 0.0

    # Calculation method
    calculation_method: str = "ASTM_C680"
    baseline_period: str = ""
    adjustment_factors: Dict[str, float] = field(default_factory=dict)

    # Measurement quality
    measurement_points: List[str] = field(default_factory=list)
    data_quality_score: float = 1.0
    uncertainty_percent: float = 0.0

    # Provenance
    calculation_hash: str = ""
    evidence_records: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize record ID if not provided."""
        if not self.record_id:
            self.record_id = f"PERF-{uuid.uuid4().hex[:12].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "asset_id": self.asset_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "baseline_heat_loss_w_m2": self.baseline_heat_loss_w_m2,
            "actual_heat_loss_w_m2": self.actual_heat_loss_w_m2,
            "heat_loss_reduction_percent": self.heat_loss_reduction_percent,
            "baseline_energy_mj": self.baseline_energy_mj,
            "actual_energy_mj": self.actual_energy_mj,
            "savings_mj": self.savings_mj,
            "savings_percent": self.savings_percent,
            "energy_cost_rate": self.energy_cost_rate,
            "cost_savings_usd": self.cost_savings_usd,
            "insulation_efficiency": self.insulation_efficiency,
            "thermal_resistance_factor": self.thermal_resistance_factor,
            "condition_degradation": self.condition_degradation,
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
# CO2E EMISSIONS RECORD
# =============================================================================

@dataclass
class CO2eEmissionsRecord:
    """Record for CO2 equivalent emissions from insulation performance."""

    record_id: str = ""
    asset_id: str = ""
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Energy and emissions
    energy_consumed_mj: float = 0.0
    fuel_type: str = ""
    emission_factor_kg_co2e_per_mj: float = 0.0
    emissions_kg_co2e: float = 0.0

    # Avoided emissions from insulation improvement
    baseline_emissions_kg_co2e: float = 0.0
    actual_emissions_kg_co2e: float = 0.0
    avoided_emissions_kg_co2e: float = 0.0
    avoided_emissions_percent: float = 0.0

    # GHG Protocol categorization
    scope: int = 1  # 1, 2, or 3
    scope_category: str = ""

    # Methodology
    calculation_method: str = "GHG_Protocol"
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
            "asset_id": self.asset_id,
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
# GAP ANALYSIS RESULT
# =============================================================================

@dataclass
class ComplianceGap:
    """Individual compliance gap identified during analysis."""

    gap_id: str = ""
    standard: str = ""
    requirement: str = ""
    current_state: str = ""
    required_state: str = ""
    gap_description: str = ""
    severity: str = "medium"
    priority: RemediationPriority = RemediationPriority.MEDIUM

    # Impact assessment
    energy_impact_mj_year: float = 0.0
    cost_impact_usd_year: float = 0.0
    co2e_impact_kg_year: float = 0.0

    # Remediation
    remediation_action: str = ""
    estimated_cost_usd: float = 0.0
    estimated_savings_usd_year: float = 0.0
    payback_period_years: float = 0.0
    implementation_timeline: str = ""

    # Tracking
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    status: str = "identified"  # "identified", "in_progress", "resolved", "deferred"

    def __post_init__(self) -> None:
        """Initialize gap ID if not provided."""
        if not self.gap_id:
            self.gap_id = f"GAP-{uuid.uuid4().hex[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gap_id": self.gap_id,
            "standard": self.standard,
            "requirement": self.requirement,
            "current_state": self.current_state,
            "required_state": self.required_state,
            "gap_description": self.gap_description,
            "severity": self.severity,
            "priority": self.priority.value if isinstance(self.priority, RemediationPriority) else str(self.priority),
            "energy_impact_mj_year": self.energy_impact_mj_year,
            "cost_impact_usd_year": self.cost_impact_usd_year,
            "co2e_impact_kg_year": self.co2e_impact_kg_year,
            "remediation_action": self.remediation_action,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_savings_usd_year": self.estimated_savings_usd_year,
            "payback_period_years": self.payback_period_years,
            "implementation_timeline": self.implementation_timeline,
            "assigned_to": self.assigned_to,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "status": self.status,
        }


# =============================================================================
# COMPLIANCE REPORT
# =============================================================================

@dataclass
class InsulationComplianceReport:
    """
    Complete compliance report for insulation assessment.

    Comprehensive report that supports regulatory submissions,
    internal audits, and continuous improvement tracking.
    """

    report_id: str = ""
    report_type: ReportType = ReportType.AUDIT_SUMMARY
    report_title: str = ""
    report_status: ReportStatus = ReportStatus.DRAFT

    # Scope
    assets_covered: List[str] = field(default_factory=list)
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
    iso_50001_checks: List[ISO50001Check] = field(default_factory=list)
    ashrae_checks: List[ASHRAECheck] = field(default_factory=list)
    energy_performance: List[EnergyPerformanceRecord] = field(default_factory=list)
    co2e_emissions: List[CO2eEmissionsRecord] = field(default_factory=list)

    # Gap analysis
    compliance_gaps: List[ComplianceGap] = field(default_factory=list)
    total_energy_impact_mj_year: float = 0.0
    total_cost_impact_usd_year: float = 0.0
    total_co2e_impact_kg_year: float = 0.0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    remediation_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: str = "GL-015-INSULSCAN"
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

        weighted_score = (
            self.checks_passed * 1.0 +
            self.checks_warning * 0.7 +
            self.checks_failed * 0.0
        )
        self.compliance_score = weighted_score / self.checks_performed
        return self.compliance_score

    def update_counts(self) -> None:
        """Update check counts from all checks."""
        all_checks = []

        for check in self.iso_50001_checks:
            all_checks.append((check.is_compliant, check.severity))

        for check in self.ashrae_checks:
            all_checks.append((check.is_compliant, check.severity))

        self.checks_performed = len(all_checks)
        self.checks_passed = sum(1 for c in all_checks if c[0] and c[1] == "info")
        self.checks_warning = sum(1 for c in all_checks if c[1] == "warning")
        self.checks_failed = sum(1 for c in all_checks if not c[0] or c[1] == "critical")
        self.overall_compliant = self.checks_failed == 0

    def update_gap_impacts(self) -> None:
        """Update total impacts from gaps."""
        self.total_energy_impact_mj_year = sum(g.energy_impact_mj_year for g in self.compliance_gaps)
        self.total_cost_impact_usd_year = sum(g.cost_impact_usd_year for g in self.compliance_gaps)
        self.total_co2e_impact_kg_year = sum(g.co2e_impact_kg_year for g in self.compliance_gaps)

    def compute_report_hash(self) -> str:
        """Compute SHA-256 hash of the report."""
        hash_data = {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "assets_covered": self.assets_covered,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "compliance_score": self.compliance_score,
            "checks_performed": self.checks_performed,
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
            "assets_covered": self.assets_covered,
            "frameworks": [f.value for f in self.frameworks],
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "overall_compliant": self.overall_compliant,
            "compliance_score": self.compliance_score,
            "checks_performed": self.checks_performed,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_warning": self.checks_warning,
            "iso_50001_checks": [c.to_dict() for c in self.iso_50001_checks],
            "ashrae_checks": [c.to_dict() for c in self.ashrae_checks],
            "energy_performance": [e.to_dict() for e in self.energy_performance],
            "co2e_emissions": [e.to_dict() for e in self.co2e_emissions],
            "compliance_gaps": [g.to_dict() for g in self.compliance_gaps],
            "total_energy_impact_mj_year": self.total_energy_impact_mj_year,
            "total_cost_impact_usd_year": self.total_cost_impact_usd_year,
            "total_co2e_impact_kg_year": self.total_co2e_impact_kg_year,
            "recommendations": self.recommendations,
            "remediation_actions": self.remediation_actions,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "evidence_pack_id": self.evidence_pack_id,
            "supporting_documents": self.supporting_documents,
            "report_hash": self.report_hash,
        }


# =============================================================================
# INSULATION COMPLIANCE REPORTER
# =============================================================================

class InsulationComplianceReporter:
    """
    Compliance Reporter for GL-015 Insulscan.

    Generates comprehensive regulatory compliance reports for insulation
    scanning and thermal assessment including ISO 50001, ASHRAE standards,
    and EPA greenhouse gas reporting.

    Usage:
        >>> reporter = InsulationComplianceReporter()
        >>> report = reporter.generate_iso_50001_report(
        ...     asset_ids=["INSUL-001"],
        ...     period_start=datetime(2024, 1, 1),
        ...     period_end=datetime(2024, 12, 31)
        ... )
        >>> reporter.add_ashrae_compliance(report, asset_data)
        >>> gap_analysis = reporter.analyze_compliance_gaps(report)
        >>> finalized = reporter.finalize_report(report)
    """

    VERSION = "1.0.0"

    # ASHRAE 90.1 Minimum R-values for pipe insulation (m2K/W)
    # Based on fluid operating temperature and pipe diameter
    ASHRAE_90_1_PIPE_R_VALUES = {
        # (temp_range_c, pipe_diameter_mm): r_value_m2K_W
        ("40-60", "small"): 0.53,    # 25-50mm diameter
        ("40-60", "medium"): 0.70,   # 50-100mm diameter
        ("40-60", "large"): 0.88,    # >100mm diameter
        ("60-90", "small"): 0.70,
        ("60-90", "medium"): 0.88,
        ("60-90", "large"): 1.06,
        ("90-120", "small"): 0.88,
        ("90-120", "medium"): 1.06,
        ("90-120", "large"): 1.23,
        (">120", "small"): 1.06,
        (">120", "medium"): 1.23,
        (">120", "large"): 1.41,
    }

    # Standard emission factors (kg CO2e/MJ)
    EMISSION_FACTORS = {
        "natural_gas": 0.0561,
        "fuel_oil_heavy": 0.0774,
        "fuel_oil_light": 0.0733,
        "coal": 0.0946,
        "electricity_grid_average": 0.1111,
        "steam_from_natural_gas": 0.0673,
        "district_heating": 0.0650,
    }

    def __init__(self, evidence_generator: Optional[InsulationEvidenceGenerator] = None):
        """
        Initialize compliance reporter.

        Args:
            evidence_generator: Optional evidence generator for creating sealed packs
        """
        self.evidence_generator = evidence_generator or InsulationEvidenceGenerator()
        self._reports: Dict[str, InsulationComplianceReport] = {}

        logger.info("InsulationComplianceReporter initialized")

    # =========================================================================
    # ISO 50001 COMPLIANCE
    # =========================================================================

    def generate_iso_50001_report(
        self,
        asset_ids: List[str],
        period_start: datetime,
        period_end: datetime,
        energy_data: Optional[Dict[str, Any]] = None,
        baseline_data: Optional[Dict[str, Any]] = None,
    ) -> InsulationComplianceReport:
        """
        Generate ISO 50001 Energy Management compliance report.

        Args:
            asset_ids: List of insulation asset identifiers
            period_start: Start of reporting period
            period_end: End of reporting period
            energy_data: Current energy performance data
            baseline_data: Baseline energy data for comparison

        Returns:
            InsulationComplianceReport with ISO 50001 checks
        """
        report = InsulationComplianceReport(
            report_type=ReportType.ISO_50001_COMPLIANCE,
            report_title=f"ISO 50001 Compliance Report - {period_start.date()} to {period_end.date()}",
            assets_covered=asset_ids,
            frameworks=[ComplianceFramework.ISO_50001],
            period_start=period_start,
            period_end=period_end,
        )

        energy_data = energy_data or {}
        baseline_data = baseline_data or {}

        # Run ISO 50001 checks
        iso_checks = []

        # Check 4.4.3.1: Energy review - identification of SEUs
        iso_checks.append(self._check_energy_review(asset_ids, energy_data))

        # Check 4.4.3.2: Energy baseline established
        iso_checks.append(self._check_energy_baseline(baseline_data))

        # Check 4.4.3.3: Energy performance indicators
        iso_checks.append(self._check_energy_performance_indicators(energy_data, baseline_data))

        # Check 4.4.6: Monitoring and measurement
        iso_checks.append(self._check_monitoring_measurement(energy_data))

        # Check 4.6.1: Internal audit
        iso_checks.append(self._check_internal_audit(energy_data))

        # Check 10.2: Continual improvement
        iso_checks.append(self._check_continual_improvement(energy_data, baseline_data))

        report.iso_50001_checks = iso_checks
        report.update_counts()
        report.calculate_compliance_score()

        # Generate recommendations
        report.recommendations = self._generate_iso_50001_recommendations(iso_checks)

        self._reports[report.report_id] = report

        logger.info(
            f"Generated ISO 50001 report {report.report_id}: "
            f"{report.checks_passed}/{report.checks_performed} passed, "
            f"score={report.compliance_score:.2%}"
        )

        return report

    def _check_energy_review(
        self,
        asset_ids: List[str],
        energy_data: Dict[str, Any],
    ) -> ISO50001Check:
        """Check energy review compliance (4.4.3.1)."""
        # Check if significant energy uses have been identified
        seus_identified = energy_data.get("significant_energy_uses", [])
        assets_reviewed = energy_data.get("assets_reviewed", [])

        is_compliant = len(seus_identified) > 0 and len(assets_reviewed) >= len(asset_ids)

        return ISO50001Check(
            check_name="Energy Review - SEU Identification",
            clause_reference="4.4.3.1",
            category="energy_planning",
            requirement="Significant energy uses shall be identified based on appropriate criteria",
            measured_value=len(seus_identified),
            target_value=float(len(asset_ids)),
            unit="assets",
            is_compliant=is_compliant,
            severity="info" if is_compliant else "warning",
            finding=f"Identified {len(seus_identified)} SEUs, reviewed {len(assets_reviewed)} assets",
            evidence_collected=len(energy_data) > 0,
            evidence_hash=compute_sha256(energy_data) if energy_data else "",
        )

    def _check_energy_baseline(self, baseline_data: Dict[str, Any]) -> ISO50001Check:
        """Check energy baseline compliance (4.4.3.2)."""
        baseline_established = baseline_data.get("baseline_established", False)
        baseline_date = baseline_data.get("baseline_date")
        baseline_value = baseline_data.get("baseline_energy_mj", 0.0)

        is_compliant = baseline_established and baseline_value > 0

        return ISO50001Check(
            check_name="Energy Baseline Established",
            clause_reference="4.4.3.2",
            category="energy_planning",
            requirement="Energy baseline shall be established using information from the energy review",
            measured_value=baseline_value,
            target_value=0.0,  # Any positive value is acceptable
            unit="MJ",
            is_compliant=is_compliant,
            severity="info" if is_compliant else "critical",
            finding=f"Baseline {'established' if baseline_established else 'not established'} "
                    f"at {baseline_value} MJ" if baseline_date else "No baseline date",
            evidence_collected=baseline_established,
            evidence_hash=compute_sha256(baseline_data) if baseline_data else "",
        )

    def _check_energy_performance_indicators(
        self,
        energy_data: Dict[str, Any],
        baseline_data: Dict[str, Any],
    ) -> ISO50001Check:
        """Check EnPIs compliance (4.4.3.3)."""
        enpis_defined = energy_data.get("enpis", [])
        current_performance = energy_data.get("current_energy_mj", 0.0)
        baseline_value = baseline_data.get("baseline_energy_mj", 0.0)

        improvement = 0.0
        if baseline_value > 0:
            improvement = (baseline_value - current_performance) / baseline_value * 100

        is_compliant = len(enpis_defined) > 0 and improvement >= 0

        return ISO50001Check(
            check_name="Energy Performance Indicators",
            clause_reference="4.4.3.3",
            category="energy_planning",
            requirement="EnPIs shall be appropriate for monitoring and measuring energy performance",
            measured_value=improvement,
            target_value=0.0,  # Any improvement is positive
            unit="%",
            is_compliant=is_compliant,
            deviation_percent=-improvement if improvement < 0 else 0.0,
            severity="info" if is_compliant else "warning",
            finding=f"Energy performance improvement: {improvement:.1f}%",
            evidence_collected=len(enpis_defined) > 0,
        )

    def _check_monitoring_measurement(self, energy_data: Dict[str, Any]) -> ISO50001Check:
        """Check monitoring and measurement compliance (4.4.6)."""
        monitoring_plan = energy_data.get("monitoring_plan", False)
        measurement_frequency = energy_data.get("measurement_frequency", "")
        calibration_current = energy_data.get("calibration_current", False)

        is_compliant = monitoring_plan and calibration_current

        return ISO50001Check(
            check_name="Monitoring and Measurement",
            clause_reference="4.4.6",
            category="monitoring",
            requirement="Key characteristics of operations affecting energy performance shall be monitored",
            is_compliant=is_compliant,
            severity="info" if is_compliant else "warning",
            finding=f"Monitoring plan: {'Yes' if monitoring_plan else 'No'}, "
                    f"Calibration current: {'Yes' if calibration_current else 'No'}",
            evidence_collected=monitoring_plan,
        )

    def _check_internal_audit(self, energy_data: Dict[str, Any]) -> ISO50001Check:
        """Check internal audit compliance (4.6.1)."""
        last_audit_date = energy_data.get("last_audit_date")
        audit_findings_addressed = energy_data.get("audit_findings_addressed", 0)
        audit_findings_total = energy_data.get("audit_findings_total", 0)

        is_compliant = last_audit_date is not None

        addressed_ratio = 0.0
        if audit_findings_total > 0:
            addressed_ratio = audit_findings_addressed / audit_findings_total

        return ISO50001Check(
            check_name="Internal Audit",
            clause_reference="4.6.1",
            category="audit",
            requirement="Internal audits shall be conducted at planned intervals",
            measured_value=addressed_ratio * 100,
            target_value=100.0,
            unit="%",
            is_compliant=is_compliant and addressed_ratio >= 0.8,
            severity="info" if is_compliant else "warning",
            finding=f"Last audit: {last_audit_date}, "
                    f"Findings addressed: {audit_findings_addressed}/{audit_findings_total}",
            evidence_collected=last_audit_date is not None,
        )

    def _check_continual_improvement(
        self,
        energy_data: Dict[str, Any],
        baseline_data: Dict[str, Any],
    ) -> ISO50001Check:
        """Check continual improvement compliance (10.2)."""
        current = energy_data.get("current_energy_mj", 0.0)
        baseline = baseline_data.get("baseline_energy_mj", 0.0)
        improvement_target = energy_data.get("improvement_target_percent", 5.0)

        actual_improvement = 0.0
        if baseline > 0:
            actual_improvement = (baseline - current) / baseline * 100

        is_compliant = actual_improvement >= improvement_target

        return ISO50001Check(
            check_name="Continual Improvement",
            clause_reference="10.2",
            category="improvement",
            requirement="Organization shall continually improve the suitability, adequacy, and effectiveness of the EnMS",
            measured_value=actual_improvement,
            target_value=improvement_target,
            unit="%",
            is_compliant=is_compliant,
            deviation_percent=improvement_target - actual_improvement if not is_compliant else 0.0,
            severity="info" if is_compliant else "warning",
            finding=f"Actual improvement: {actual_improvement:.1f}% vs target: {improvement_target:.1f}%",
            evidence_collected=True,
        )

    def _generate_iso_50001_recommendations(self, checks: List[ISO50001Check]) -> List[str]:
        """Generate recommendations based on ISO 50001 check results."""
        recommendations = []

        for check in checks:
            if not check.is_compliant:
                if "baseline" in check.check_name.lower():
                    recommendations.append(
                        "Establish energy baseline using historical data from the past 12 months. "
                        "Document the baseline methodology and adjustment factors."
                    )
                elif "seu" in check.check_name.lower() or "review" in check.check_name.lower():
                    recommendations.append(
                        "Conduct comprehensive energy review to identify all significant energy uses (SEUs). "
                        "Document the criteria used to identify SEUs."
                    )
                elif "monitoring" in check.check_name.lower():
                    recommendations.append(
                        "Implement comprehensive monitoring and measurement plan. "
                        "Ensure all measurement equipment is calibrated per schedule."
                    )
                elif "improvement" in check.check_name.lower():
                    recommendations.append(
                        f"Current improvement of {check.measured_value:.1f}% is below target. "
                        "Review energy improvement projects and identify additional opportunities."
                    )

        return recommendations

    # =========================================================================
    # ASHRAE COMPLIANCE
    # =========================================================================

    def add_ashrae_compliance(
        self,
        report: InsulationComplianceReport,
        asset_data: List[Dict[str, Any]],
        climate_zone: str = "4A",
        standard_version: str = "90.1-2019",
    ) -> List[ASHRAECheck]:
        """
        Add ASHRAE compliance checks to a report.

        Args:
            report: Existing compliance report
            asset_data: List of asset data with R-values, temperatures, etc.
            climate_zone: ASHRAE climate zone
            standard_version: ASHRAE standard version

        Returns:
            List of ASHRAE checks added
        """
        ashrae_checks = []

        for asset in asset_data:
            asset_id = asset.get("asset_id", "")
            insulation_type = asset.get("insulation_type", "pipe")
            operating_temp_c = asset.get("operating_temp_c", 60.0)
            pipe_diameter_mm = asset.get("pipe_diameter_mm", 50.0)
            actual_r_value = asset.get("actual_r_value", 0.0)

            # Determine required R-value
            temp_range = self._get_temp_range(operating_temp_c)
            size_class = self._get_pipe_size_class(pipe_diameter_mm)
            required_r = self.ASHRAE_90_1_PIPE_R_VALUES.get((temp_range, size_class), 0.70)

            is_compliant = actual_r_value >= required_r
            deviation = ((required_r - actual_r_value) / required_r * 100) if required_r > 0 else 0

            check = ASHRAECheck(
                check_name=f"Pipe Insulation R-Value - {asset_id}",
                standard=standard_version,
                section="5.8.1",
                category="insulation",
                requirement=f"Pipe insulation R-value >= {required_r} m2K/W for {temp_range}C operating temp",
                climate_zone=climate_zone,
                required_r_value=required_r,
                actual_r_value=actual_r_value,
                r_value_unit="m2K/W",
                insulation_type=insulation_type,
                application="pipe",
                is_compliant=is_compliant,
                deviation_percent=deviation if not is_compliant else 0.0,
                severity="info" if is_compliant else ("critical" if deviation > 50 else "warning"),
                finding=f"Actual R-value {actual_r_value:.2f} vs required {required_r:.2f} m2K/W",
                evidence_hash=compute_sha256(asset),
            )

            ashrae_checks.append(check)

        report.ashrae_checks.extend(ashrae_checks)

        if ComplianceFramework.ASHRAE_90_1 not in report.frameworks:
            report.frameworks.append(ComplianceFramework.ASHRAE_90_1)

        report.update_counts()
        report.calculate_compliance_score()

        logger.info(f"Added {len(ashrae_checks)} ASHRAE checks to report {report.report_id}")

        return ashrae_checks

    def _get_temp_range(self, temp_c: float) -> str:
        """Get temperature range category for ASHRAE lookup."""
        if temp_c < 40:
            return "40-60"  # Use lowest range for cold
        elif temp_c < 60:
            return "40-60"
        elif temp_c < 90:
            return "60-90"
        elif temp_c < 120:
            return "90-120"
        else:
            return ">120"

    def _get_pipe_size_class(self, diameter_mm: float) -> str:
        """Get pipe size class for ASHRAE lookup."""
        if diameter_mm < 50:
            return "small"
        elif diameter_mm < 100:
            return "medium"
        else:
            return "large"

    # =========================================================================
    # ENERGY PERFORMANCE VERIFICATION
    # =========================================================================

    def add_energy_performance(
        self,
        report: InsulationComplianceReport,
        asset_id: str,
        baseline_heat_loss: float,
        actual_heat_loss: float,
        surface_area_m2: float,
        operating_hours: float,
        energy_cost_rate: float = 0.00003,  # $/MJ default
    ) -> EnergyPerformanceRecord:
        """
        Add energy performance verification to a report.

        Args:
            report: Compliance report to add to
            asset_id: Asset identifier
            baseline_heat_loss: Baseline heat loss W/m2
            actual_heat_loss: Current heat loss W/m2
            surface_area_m2: Insulated surface area
            operating_hours: Operating hours per year
            energy_cost_rate: Energy cost in $/MJ

        Returns:
            EnergyPerformanceRecord
        """
        # Calculate energy values
        baseline_energy_mj = (baseline_heat_loss * surface_area_m2 * operating_hours * 3600) / 1e6
        actual_energy_mj = (actual_heat_loss * surface_area_m2 * operating_hours * 3600) / 1e6

        savings_mj = baseline_energy_mj - actual_energy_mj
        savings_percent = (savings_mj / baseline_energy_mj * 100) if baseline_energy_mj > 0 else 0
        cost_savings = savings_mj * energy_cost_rate

        heat_loss_reduction = ((baseline_heat_loss - actual_heat_loss) / baseline_heat_loss * 100) if baseline_heat_loss > 0 else 0

        record = EnergyPerformanceRecord(
            asset_id=asset_id,
            period_start=report.period_start,
            period_end=report.period_end,
            baseline_heat_loss_w_m2=baseline_heat_loss,
            actual_heat_loss_w_m2=actual_heat_loss,
            heat_loss_reduction_percent=heat_loss_reduction,
            baseline_energy_mj=baseline_energy_mj,
            actual_energy_mj=actual_energy_mj,
            savings_mj=savings_mj,
            savings_percent=savings_percent,
            energy_cost_rate=energy_cost_rate,
            cost_savings_usd=cost_savings,
        )

        record.calculation_hash = compute_sha256(record.to_dict())

        report.energy_performance.append(record)

        logger.info(
            f"Added energy performance for {asset_id}: "
            f"{savings_mj:.2f} MJ ({savings_percent:.1f}%) saved"
        )

        return record

    # =========================================================================
    # CO2E ACCOUNTING
    # =========================================================================

    def add_co2e_accounting(
        self,
        report: InsulationComplianceReport,
        asset_id: str,
        energy_consumed_mj: float,
        fuel_type: str,
        baseline_energy_mj: Optional[float] = None,
        scope: int = 1,
        scope_category: str = "stationary combustion",
    ) -> CO2eEmissionsRecord:
        """
        Add CO2e emissions accounting to a report.

        Args:
            report: Compliance report to add to
            asset_id: Asset identifier
            energy_consumed_mj: Energy consumed in MJ
            fuel_type: Type of fuel
            baseline_energy_mj: Baseline energy for avoided emissions calc
            scope: GHG Protocol scope (1, 2, or 3)
            scope_category: Category within scope

        Returns:
            CO2eEmissionsRecord
        """
        emission_factor = self.EMISSION_FACTORS.get(fuel_type, 0.0561)
        emissions_kg_co2e = energy_consumed_mj * emission_factor

        baseline_emissions = 0.0
        avoided_emissions = 0.0
        avoided_percent = 0.0

        if baseline_energy_mj:
            baseline_emissions = baseline_energy_mj * emission_factor
            avoided_emissions = baseline_emissions - emissions_kg_co2e
            avoided_percent = (avoided_emissions / baseline_emissions * 100) if baseline_emissions > 0 else 0

        record = CO2eEmissionsRecord(
            asset_id=asset_id,
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
            calculation_method="GHG_Protocol",
            emission_factor_source="EPA 40 CFR Part 98",
        )

        record.calculation_hash = compute_sha256(record.to_dict())

        report.co2e_emissions.append(record)

        if ComplianceFramework.EPA_40_CFR_98 not in report.frameworks:
            report.frameworks.append(ComplianceFramework.EPA_40_CFR_98)

        logger.info(
            f"Added CO2e accounting for {asset_id}: "
            f"{emissions_kg_co2e:.2f} kg CO2e, {avoided_emissions:.2f} kg avoided"
        )

        return record

    # =========================================================================
    # GAP ANALYSIS
    # =========================================================================

    def analyze_compliance_gaps(
        self,
        report: InsulationComplianceReport,
        energy_cost_rate: float = 0.00003,  # $/MJ
        operating_hours_year: float = 8760.0,
    ) -> List[ComplianceGap]:
        """
        Analyze compliance gaps and generate remediation recommendations.

        Args:
            report: Compliance report to analyze
            energy_cost_rate: Energy cost in $/MJ
            operating_hours_year: Annual operating hours

        Returns:
            List of identified compliance gaps
        """
        gaps = []

        # Analyze ISO 50001 gaps
        for check in report.iso_50001_checks:
            if not check.is_compliant:
                gap = ComplianceGap(
                    standard="ISO 50001",
                    requirement=check.requirement,
                    current_state=check.finding,
                    required_state=f"Target: {check.target_value} {check.unit}" if check.target_value else "Full compliance",
                    gap_description=f"{check.check_name}: {check.finding}",
                    severity=check.severity,
                    priority=RemediationPriority.HIGH if check.severity == "critical" else RemediationPriority.MEDIUM,
                )

                # Generate remediation action
                gap.remediation_action = self._generate_iso_50001_remediation(check)
                gaps.append(gap)

        # Analyze ASHRAE gaps
        for check in report.ashrae_checks:
            if not check.is_compliant:
                r_value_gap = (check.required_r_value or 0) - (check.actual_r_value or 0)

                # Estimate energy impact
                # Simplified: higher R-value gap = more heat loss = more energy waste
                energy_impact = r_value_gap * 1000 * operating_hours_year / 1e6  # Rough MJ/year
                cost_impact = energy_impact * energy_cost_rate
                co2_impact = energy_impact * 0.0561  # Assume natural gas

                gap = ComplianceGap(
                    standard=f"ASHRAE {check.standard}",
                    requirement=check.requirement,
                    current_state=f"R-value: {check.actual_r_value:.2f} m2K/W",
                    required_state=f"R-value: {check.required_r_value:.2f} m2K/W",
                    gap_description=f"Insulation R-value deficiency of {check.deviation_percent:.1f}%",
                    severity=check.severity,
                    priority=RemediationPriority.HIGH if check.deviation_percent > 30 else RemediationPriority.MEDIUM,
                    energy_impact_mj_year=energy_impact,
                    cost_impact_usd_year=cost_impact,
                    co2e_impact_kg_year=co2_impact,
                )

                # Generate remediation
                gap.remediation_action = f"Upgrade insulation to achieve R-{check.required_r_value:.2f} m2K/W"
                gap.estimated_cost_usd = self._estimate_insulation_upgrade_cost(r_value_gap)
                gap.estimated_savings_usd_year = cost_impact

                if gap.estimated_cost_usd > 0:
                    gap.payback_period_years = gap.estimated_cost_usd / gap.estimated_savings_usd_year
                else:
                    gap.payback_period_years = 0.0

                gaps.append(gap)

        report.compliance_gaps = gaps
        report.update_gap_impacts()

        logger.info(
            f"Gap analysis for report {report.report_id}: "
            f"{len(gaps)} gaps, total impact: ${report.total_cost_impact_usd_year:.2f}/year"
        )

        return gaps

    def _generate_iso_50001_remediation(self, check: ISO50001Check) -> str:
        """Generate remediation action for ISO 50001 non-compliance."""
        remediation_map = {
            "4.4.3.1": "Conduct comprehensive energy review and document all significant energy uses",
            "4.4.3.2": "Establish energy baseline using 12 months of historical data",
            "4.4.3.3": "Define and implement energy performance indicators (EnPIs)",
            "4.4.6": "Implement monitoring plan with calibrated measurement equipment",
            "4.6.1": "Schedule and conduct internal audit per documented procedures",
            "10.2": "Identify and implement energy improvement projects to meet targets",
        }
        return remediation_map.get(check.clause_reference, "Review and address compliance finding")

    def _estimate_insulation_upgrade_cost(self, r_value_gap: float) -> float:
        """Estimate cost to upgrade insulation (placeholder calculation)."""
        # Rough estimate: $50 per m2K/W improvement per linear meter
        base_cost_per_r_value = 50.0
        estimated_linear_meters = 10.0  # Placeholder
        return r_value_gap * base_cost_per_r_value * estimated_linear_meters

    # =========================================================================
    # REMEDIATION TRACKING
    # =========================================================================

    def track_remediation(
        self,
        report: InsulationComplianceReport,
        gap_id: str,
        status: str,
        assigned_to: Optional[str] = None,
        due_date: Optional[datetime] = None,
        notes: str = "",
    ) -> Optional[ComplianceGap]:
        """
        Update remediation tracking for a compliance gap.

        Args:
            report: Compliance report
            gap_id: Gap identifier to update
            status: New status
            assigned_to: Person assigned
            due_date: Due date for remediation
            notes: Additional notes

        Returns:
            Updated ComplianceGap or None if not found
        """
        for gap in report.compliance_gaps:
            if gap.gap_id == gap_id:
                gap.status = status
                if assigned_to:
                    gap.assigned_to = assigned_to
                if due_date:
                    gap.due_date = due_date

                logger.info(f"Updated remediation for gap {gap_id}: status={status}")
                return gap

        return None

    # =========================================================================
    # REPORT FINALIZATION
    # =========================================================================

    def finalize_report(
        self,
        report: InsulationComplianceReport,
        create_evidence_pack: bool = True,
    ) -> InsulationComplianceReport:
        """
        Finalize a compliance report with evidence packaging.

        Args:
            report: Report to finalize
            create_evidence_pack: Whether to create sealed evidence pack

        Returns:
            Finalized compliance report
        """
        report.update_counts()
        report.calculate_compliance_score()
        report.update_gap_impacts()
        report.compute_report_hash()

        if create_evidence_pack:
            evidence_records = []

            # Create evidence from checks
            for check in report.iso_50001_checks + report.ashrae_checks:
                record = self.evidence_generator.create_evidence_record(
                    evidence_type=EvidenceType.ISO_50001_COMPLIANCE if isinstance(check, ISO50001Check) else EvidenceType.ASHRAE_COMPLIANCE,
                    data=check.to_dict(),
                    inputs={"requirement": check.requirement if hasattr(check, 'requirement') else ""},
                    outputs={"is_compliant": check.is_compliant},
                )
                evidence_records.append(record)

            if evidence_records:
                pack = self.evidence_generator.create_evidence_pack(
                    evidence_records,
                    assets_covered=report.assets_covered,
                )
                sealed = self.evidence_generator.seal_pack(pack)
                import json as json_mod
                envelope = json_mod.loads(sealed)
                report.evidence_pack_id = envelope.get("pack_id")

        report.report_status = ReportStatus.PENDING_REVIEW

        logger.info(
            f"Finalized report {report.report_id}: "
            f"score={report.compliance_score:.2%}, evidence_pack={report.evidence_pack_id}"
        )

        return report

    def approve_report(
        self,
        report: InsulationComplianceReport,
        approved_by: str,
    ) -> InsulationComplianceReport:
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

        logger.info(f"Approved report {report.report_id} by {approved_by}")

        return report

    # =========================================================================
    # REPORT EXPORT
    # =========================================================================

    def export_report(
        self,
        report: InsulationComplianceReport,
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

    def _export_json(self, report: InsulationComplianceReport) -> bytes:
        """Export as JSON."""
        return json.dumps(report.to_dict(), indent=2, default=str).encode('utf-8')

    def _export_xml(self, report: InsulationComplianceReport) -> bytes:
        """Export as XML."""
        import xml.etree.ElementTree as ET

        root = ET.Element("InsulationComplianceReport")
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

        # Assets
        assets = ET.SubElement(root, "AssetsCovered")
        for asset_id in report.assets_covered:
            ET.SubElement(assets, "Asset").text = asset_id

        # Frameworks
        frameworks = ET.SubElement(root, "Frameworks")
        for fw in report.frameworks:
            ET.SubElement(frameworks, "Framework").text = fw.value

        # Checks summary
        checks = ET.SubElement(root, "ChecksSummary")
        ET.SubElement(checks, "Total").text = str(report.checks_performed)
        ET.SubElement(checks, "Passed").text = str(report.checks_passed)
        ET.SubElement(checks, "Failed").text = str(report.checks_failed)
        ET.SubElement(checks, "Warnings").text = str(report.checks_warning)

        # Gaps summary
        if report.compliance_gaps:
            gaps = ET.SubElement(root, "ComplianceGaps")
            ET.SubElement(gaps, "Count").text = str(len(report.compliance_gaps))
            ET.SubElement(gaps, "EnergyImpactMJYear").text = str(report.total_energy_impact_mj_year)
            ET.SubElement(gaps, "CostImpactUSDYear").text = str(report.total_cost_impact_usd_year)
            ET.SubElement(gaps, "CO2eImpactKgYear").text = str(report.total_co2e_impact_kg_year)

        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def _export_csv(self, report: InsulationComplianceReport) -> bytes:
        """Export checks as CSV."""
        lines = ["check_id,standard,requirement,is_compliant,severity,finding"]

        for check in report.iso_50001_checks:
            lines.append(
                f'"{check.check_id}","ISO 50001 {check.clause_reference}",'
                f'"{check.requirement}",{check.is_compliant},{check.severity},"{check.finding}"'
            )

        for check in report.ashrae_checks:
            lines.append(
                f'"{check.check_id}","ASHRAE {check.standard} {check.section}",'
                f'"{check.requirement}",{check.is_compliant},{check.severity},"{check.finding}"'
            )

        return "\n".join(lines).encode('utf-8')

    def _export_pdf_placeholder(self, report: InsulationComplianceReport) -> bytes:
        """Generate PDF placeholder (actual implementation would use reportlab)."""
        text = f"""
GL-015 INSULSCAN COMPLIANCE REPORT
==================================

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

ASSETS COVERED
--------------
{chr(10).join('- ' + a for a in report.assets_covered)}

FRAMEWORKS
----------
{chr(10).join('- ' + f.value for f in report.frameworks)}

COMPLIANCE GAPS
---------------
Total Gaps: {len(report.compliance_gaps)}
Energy Impact: {report.total_energy_impact_mj_year:.2f} MJ/year
Cost Impact: ${report.total_cost_impact_usd_year:.2f}/year
CO2e Impact: {report.total_co2e_impact_kg_year:.2f} kg CO2e/year

RECOMMENDATIONS
---------------
{chr(10).join('- ' + r for r in report.recommendations) if report.recommendations else 'None'}

Generated: {report.generated_at.isoformat()}
Generated By: {report.generated_by}
Report Hash: {report.report_hash}
Evidence Pack: {report.evidence_pack_id or 'Not created'}
        """

        return text.encode('utf-8')
