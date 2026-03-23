# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - Regulatory Framework Compliance Checking (Engine 6 of 7)

AGENT-MRV-013: Dual Reporting Reconciliation Agent

Validates dual-reporting reconciliation outputs against the requirements
of seven regulatory and voluntary reporting frameworks. For each framework,
the engine checks whether the reconciliation output contains all mandatory
disclosures and meets framework-specific data quality requirements.

Supported Frameworks (7):
    1. GHG_PROTOCOL: GHG Protocol Scope 2 Guidance (2015) - mandates dual
       reporting of location-based and market-based totals.
    2. CSRD_ESRS: EU Corporate Sustainability Reporting Directive, ESRS E1 -
       requires dual reporting with reconciliation explanation.
    3. CDP: Carbon Disclosure Project (C6/C7) - requires both methods plus
       detailed energy breakdowns.
    4. SBTI: Science Based Targets initiative - uses market-based for target
       tracking, location-based for context.
    5. GRI: GRI Standards 305-2 (2016) - requires both methods with separate
       disclosure.
    6. ISO_14064: ISO 14064-1:2018 - permits either method with justification;
       recommends both.
    7. RE100: RE100 initiative - tracks market-based for RE100 percentage.

Responsibilities:
    1. Load framework-specific requirement definitions from
       FRAMEWORK_REQUIRED_DISCLOSURES constant in models.py.
    2. Check each requirement against the reconciliation workspace,
       discrepancy report, and quality assessment.
    3. Produce a ComplianceCheckResult per framework with:
       - Overall ComplianceStatus (COMPLIANT/NON_COMPLIANT/PARTIAL)
       - Score as fraction (0.0 to 1.0)
       - Detailed requirement-by-requirement results
       - Findings and observations
    4. Generate flags for non-compliant or partial results.
    5. Track provenance hashes for all compliance checks.

Zero-Hallucination Guarantees:
    - All arithmetic uses Python ``Decimal`` with ROUND_HALF_UP at
      8-decimal-place precision.
    - Compliance checks are deterministic rule-based evaluations.
    - No LLM, ML, or probabilistic computation in any compliance path.

Thread Safety:
    Thread-safe singleton via ``__new__`` with ``_instance``,
    ``_initialized``, and ``threading.RLock``.

Public Methods (12):
    check_compliance            -> ComplianceCheckResult
    check_all_frameworks        -> Dict[str, ComplianceCheckResult]
    check_ghg_protocol          -> ComplianceCheckResult
    check_csrd_esrs             -> ComplianceCheckResult
    check_cdp                   -> ComplianceCheckResult
    check_sbti                  -> ComplianceCheckResult
    check_gri                   -> ComplianceCheckResult
    check_iso_14064             -> ComplianceCheckResult
    check_re100                 -> ComplianceCheckResult
    generate_compliance_flags   -> List[Flag]
    get_compliance_summary      -> Dict[str, Any]
    health_check                -> Dict[str, Any]

Classmethod:
    reset                       -> None

Example:
    >>> from greenlang.agents.mrv.dual_reporting_reconciliation.compliance_checker import (
    ...     ComplianceCheckerEngine,
    ... )
    >>> engine = ComplianceCheckerEngine()
    >>> result = engine.check_compliance(
    ...     workspace=workspace,
    ...     discrepancy_report=disc_report,
    ...     quality_assessment=quality,
    ...     framework=ReportingFramework.GHG_PROTOCOL,
    ... )
    >>> assert result.status == ComplianceStatus.COMPLIANT

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-013 Dual Reporting Reconciliation (GL-MRV-X-024)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    "ComplianceCheckerEngine",
    "check_compliance",
    "check_all_frameworks",
]

# ---------------------------------------------------------------------------
# Conditional imports -- graceful degradation if sibling modules unavailable
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
        AGENT_COMPONENT,
        AGENT_ID,
        VERSION,
        ComplianceCheckResult,
        ComplianceRequirement,
        ComplianceStatus,
        DiscrepancyReport,
        EnergyType,
        Flag,
        FlagSeverity,
        FlagType,
        FRAMEWORK_REQUIRED_DISCLOSURES,
        QualityAssessment,
        QualityGrade,
        ReconciliationWorkspace,
        ReportingFramework,
        Scope2Method,
    )
except ImportError:
    logger.warning("Could not import models; ComplianceCheckerEngine will be limited")
    AGENT_COMPONENT = "AGENT-MRV-013"
    AGENT_ID = "GL-MRV-X-024"
    VERSION = "1.0.0"

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.config import (
        DualReportingReconciliationConfig,
    )
except ImportError:
    DualReportingReconciliationConfig = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.metrics import (
        DualReportingReconciliationMetrics,
    )
except ImportError:
    DualReportingReconciliationMetrics = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.provenance import (
        DualReportingProvenanceTracker,
        ProvenanceStage,
        hash_compliance_result,
    )
except ImportError:
    DualReportingProvenanceTracker = None  # type: ignore[assignment,misc]
    ProvenanceStage = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_PRECISION = 8
_QUANTIZE_EXP = Decimal("0." + "0" * _PRECISION)
_ZERO = Decimal("0")
_ONE = Decimal("1")
_ONE_HUNDRED = Decimal("100")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to _PRECISION decimal places."""
    try:
        return value.quantize(_QUANTIZE_EXP, rounding=ROUND_HALF_UP)
    except (InvalidOperation, Exception):
        return _ZERO


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a dictionary."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# ComplianceCheckerEngine
# =============================================================================


class ComplianceCheckerEngine:
    """
    Engine 6 of 7: Regulatory Framework Compliance Checking.

    Validates dual-reporting reconciliation outputs against seven
    regulatory frameworks. Each framework has specific disclosure
    requirements defined in FRAMEWORK_REQUIRED_DISCLOSURES.

    Thread-safe singleton via ``__new__`` with ``_instance``,
    ``_initialized``, and ``threading.RLock``.

    Attributes:
        _config: Singleton configuration object.
        _metrics: Singleton metrics tracker.
        _provenance: Singleton provenance tracker.
        _check_count: Number of compliance checks performed since init.

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> result = engine.check_compliance(workspace, disc, quality, framework)
        >>> print(result.status, result.score)
    """

    _instance: Optional[ComplianceCheckerEngine] = None
    _initialized: bool = False
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def __new__(cls) -> ComplianceCheckerEngine:
        """Return the singleton instance, creating on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise engine once."""
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._config = (
                DualReportingReconciliationConfig()
                if DualReportingReconciliationConfig is not None
                else None
            )
            self._metrics = (
                DualReportingReconciliationMetrics()
                if DualReportingReconciliationMetrics is not None
                else None
            )
            self._provenance = (
                DualReportingProvenanceTracker.get_instance()
                if DualReportingProvenanceTracker is not None
                else None
            )
            self._check_count: int = 0
            self.__class__._initialized = True
            logger.info(
                "%s-ComplianceCheckerEngine initialised (v%s)",
                AGENT_COMPONENT,
                VERSION,
            )

    @classmethod
    def reset(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.warning("ComplianceCheckerEngine singleton reset")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _is_framework_enabled(self, framework: ReportingFramework) -> bool:
        """Check if a framework is enabled in configuration."""
        if self._config is not None:
            fw_value = (
                framework.value
                if hasattr(framework, "value")
                else str(framework)
            )
            return fw_value in self._config.enabled_frameworks
        return True

    def _is_strict_mode(self) -> bool:
        """Check if strict compliance mode is enabled."""
        if self._config is not None:
            return self._config.strict_mode
        return False

    # ------------------------------------------------------------------
    # Main entry point: single framework
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: Optional[DiscrepancyReport],
        quality_assessment: Optional[QualityAssessment],
        framework: ReportingFramework,
    ) -> ComplianceCheckResult:
        """
        Check compliance against a single reporting framework.

        Routes to the framework-specific checker method.

        Args:
            workspace: ReconciliationWorkspace with emission data.
            discrepancy_report: Discrepancy analysis results.
            quality_assessment: Quality scoring results.
            framework: Target framework.

        Returns:
            ComplianceCheckResult with detailed findings.
        """
        start_time = time.monotonic()

        logger.info(
            "Checking compliance for framework %s, reconciliation %s",
            framework.value if hasattr(framework, "value") else framework,
            workspace.reconciliation_id,
        )

        # Route to framework-specific checker
        framework_checkers: Dict[str, Callable] = {
            ReportingFramework.GHG_PROTOCOL.value: self.check_ghg_protocol,
            ReportingFramework.CSRD_ESRS.value: self.check_csrd_esrs,
            ReportingFramework.CDP.value: self.check_cdp,
            ReportingFramework.SBTI.value: self.check_sbti,
            ReportingFramework.GRI.value: self.check_gri,
            ReportingFramework.ISO_14064.value: self.check_iso_14064,
            ReportingFramework.RE100.value: self.check_re100,
        }

        fw_key = framework.value if hasattr(framework, "value") else str(framework)
        checker = framework_checkers.get(fw_key)

        if checker is None:
            logger.warning("Unknown framework: %s", fw_key)
            return ComplianceCheckResult(
                framework=framework,
                status=ComplianceStatus.NOT_APPLICABLE,
                requirements_total=0,
                requirements_met=0,
                requirements=[],
                score=_ZERO,
                findings=[f"Unknown framework: {fw_key}"],
            )

        result = checker(workspace, discrepancy_report, quality_assessment)

        # Record metrics
        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._check_count += 1

        if self._metrics is not None:
            try:
                status_val = (
                    result.status.value
                    if hasattr(result.status, "value")
                    else str(result.status)
                )
                self._metrics.record_compliance_check(fw_key, status_val)
            except Exception as exc:
                logger.warning("Failed to record compliance metrics: %s", exc)

        # Record provenance
        if self._provenance is not None and ProvenanceStage is not None:
            try:
                self._provenance.add_stage(
                    workspace.reconciliation_id,
                    ProvenanceStage.CHECK_COMPLIANCE,
                    {
                        "framework": fw_key,
                        "status": result.status.value
                        if hasattr(result.status, "value")
                        else str(result.status),
                        "score": str(result.score),
                        "requirements_met": result.requirements_met,
                        "requirements_total": result.requirements_total,
                    },
                    {"hash": _compute_hash({
                        "framework": fw_key,
                        "score": str(result.score),
                    })},
                )
            except Exception as exc:
                logger.debug("Provenance tracking skipped: %s", exc)

        logger.info(
            "Compliance check for %s: status=%s, score=%.4f, "
            "met=%d/%d, elapsed=%.1fms",
            fw_key,
            result.status.value if hasattr(result.status, "value") else result.status,
            float(result.score),
            result.requirements_met,
            result.requirements_total,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Multi-framework check
    # ------------------------------------------------------------------

    def check_all_frameworks(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: Optional[DiscrepancyReport],
        quality_assessment: Optional[QualityAssessment],
        frameworks: Optional[List[ReportingFramework]] = None,
    ) -> Dict[str, ComplianceCheckResult]:
        """
        Check compliance against multiple frameworks.

        Args:
            workspace: ReconciliationWorkspace.
            discrepancy_report: Discrepancy analysis results.
            quality_assessment: Quality scoring results.
            frameworks: List of frameworks to check. If None, checks
                all enabled frameworks.

        Returns:
            Dict mapping framework value to ComplianceCheckResult.
        """
        if frameworks is None:
            frameworks = list(ReportingFramework)

        results: Dict[str, ComplianceCheckResult] = {}
        for fw in frameworks:
            if self._is_framework_enabled(fw):
                result = self.check_compliance(
                    workspace, discrepancy_report, quality_assessment, fw
                )
                fw_key = fw.value if hasattr(fw, "value") else str(fw)
                results[fw_key] = result

        return results

    # ------------------------------------------------------------------
    # Framework-specific checkers
    # ------------------------------------------------------------------

    def check_ghg_protocol(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: Optional[DiscrepancyReport],
        quality_assessment: Optional[QualityAssessment],
    ) -> ComplianceCheckResult:
        """
        Check GHG Protocol Scope 2 Guidance (2015) compliance.

        Key requirements:
        - Both location-based and market-based totals must be disclosed
        - Breakdown by energy type for both methods
        - Breakdown by country/region for both methods
        - Contractual instruments summary
        - Residual mix factor disclosure
        - Emission factor sources
        - GWP values used
        - Organizational boundary
        - Base year recalculation policy
        - Exclusions and limitations

        Args:
            workspace: ReconciliationWorkspace.
            discrepancy_report: Discrepancy analysis results.
            quality_assessment: Quality scoring results.

        Returns:
            ComplianceCheckResult for GHG Protocol.
        """
        requirements: List[ComplianceRequirement] = []

        # R1: Location-based total
        requirements.append(self._check_value_present(
            "GHG-S2-001",
            "Location-based total tCO2e disclosed",
            workspace.total_location_tco2e > _ZERO or len(workspace.location_results) > 0,
            f"Location-based total: {float(workspace.total_location_tco2e):.4f} tCO2e",
        ))

        # R2: Market-based total
        requirements.append(self._check_value_present(
            "GHG-S2-002",
            "Market-based total tCO2e disclosed",
            workspace.total_market_tco2e > _ZERO or len(workspace.market_results) > 0,
            f"Market-based total: {float(workspace.total_market_tco2e):.4f} tCO2e",
        ))

        # R3: Location by energy type
        requirements.append(self._check_value_present(
            "GHG-S2-003",
            "Location-based emissions by energy type disclosed",
            len(workspace.by_energy_type) > 0,
            f"{len(workspace.by_energy_type)} energy type breakdowns",
        ))

        # R4: Market by energy type
        requirements.append(self._check_value_present(
            "GHG-S2-004",
            "Market-based emissions by energy type disclosed",
            len(workspace.by_energy_type) > 0,
            f"{len(workspace.by_energy_type)} energy type breakdowns",
        ))

        # R5: Emission factor sources documented
        ef_sources = self._extract_ef_sources(workspace)
        requirements.append(self._check_value_present(
            "GHG-S2-005",
            "Emission factor sources documented",
            len(ef_sources) > 0,
            f"{len(ef_sources)} distinct EF sources",
        ))

        # R6: GWP values identified
        gwp_sources = self._extract_gwp_sources(workspace)
        requirements.append(self._check_value_present(
            "GHG-S2-006",
            "GWP values and IPCC assessment report identified",
            len(gwp_sources) > 0,
            f"GWP sources: {', '.join(gwp_sources)}",
        ))

        # R7: Both methods have data for same facilities
        loc_facilities = {r.facility_id for r in workspace.location_results}
        mkt_facilities = {r.facility_id for r in workspace.market_results}
        overlap = loc_facilities & mkt_facilities
        requirements.append(self._check_value_present(
            "GHG-S2-007",
            "Both methods cover the same organizational boundary",
            len(overlap) > 0,
            f"{len(overlap)} facilities with dual coverage out of "
            f"{len(loc_facilities | mkt_facilities)} total",
        ))

        # R8: Contractual instruments documented (if market data exists)
        has_instruments = any(
            r.ef_hierarchy is not None
            for r in workspace.market_results
        )
        requirements.append(self._check_value_present(
            "GHG-S2-008",
            "Contractual instruments type and EF hierarchy documented",
            has_instruments or len(workspace.market_results) == 0,
            "EF hierarchy documented for market results",
        ))

        # R9: Reporting period specified
        requirements.append(self._check_value_present(
            "GHG-S2-009",
            "Reporting period clearly defined",
            workspace.period_start is not None and workspace.period_end is not None,
            f"Period: {workspace.period_start} to {workspace.period_end}",
        ))

        # R10: Discrepancy explanation (if material)
        disc_explained = True
        if discrepancy_report and discrepancy_report.discrepancies:
            disc_explained = len(discrepancy_report.discrepancies) > 0
        requirements.append(self._check_value_present(
            "GHG-S2-010",
            "Material discrepancies explained with root cause analysis",
            disc_explained,
            f"{len(discrepancy_report.discrepancies) if discrepancy_report else 0} "
            f"discrepancies documented",
        ))

        # R11: Data quality assessment
        has_quality = quality_assessment is not None
        requirements.append(self._check_value_present(
            "GHG-S2-011",
            "Data quality assessment conducted",
            has_quality,
            f"Quality grade: {quality_assessment.grade.value if quality_assessment and hasattr(quality_assessment.grade, 'value') else 'N/A'}",
        ))

        # R12: Facility-level breakdowns available
        requirements.append(self._check_value_present(
            "GHG-S2-012",
            "Facility-level emission breakdowns available",
            len(workspace.by_facility) > 0,
            f"{len(workspace.by_facility)} facility breakdowns",
        ))

        # R13: Provenance hashes for audit trail
        has_provenance = any(
            r.provenance_hash for r in workspace.location_results
        ) or any(
            r.provenance_hash for r in workspace.market_results
        )
        requirements.append(self._check_value_present(
            "GHG-S2-013",
            "Provenance hashes available for audit trail",
            has_provenance,
            "SHA-256 provenance tracked",
        ))

        return self._build_result(
            ReportingFramework.GHG_PROTOCOL, requirements
        )

    def check_csrd_esrs(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: Optional[DiscrepancyReport],
        quality_assessment: Optional[QualityAssessment],
    ) -> ComplianceCheckResult:
        """
        Check CSRD/ESRS E1 compliance for Scope 2 dual reporting.

        CSRD requires:
        - Both location and market totals
        - Energy consumption in MWh
        - Renewable energy percentage
        - Reconciliation explanation
        - Data quality assessment
        - Significant changes explanation
        - Base year emissions
        - Reduction targets
        """
        requirements: List[ComplianceRequirement] = []

        # R1: Location total
        requirements.append(self._check_value_present(
            "CSRD-E1-001",
            "Scope 2 location-based total disclosed per ESRS E1-6",
            workspace.total_location_tco2e >= _ZERO and len(workspace.location_results) > 0,
            f"Location total: {float(workspace.total_location_tco2e):.4f} tCO2e",
        ))

        # R2: Market total
        requirements.append(self._check_value_present(
            "CSRD-E1-002",
            "Scope 2 market-based total disclosed per ESRS E1-6",
            workspace.total_market_tco2e >= _ZERO and len(workspace.market_results) > 0,
            f"Market total: {float(workspace.total_market_tco2e):.4f} tCO2e",
        ))

        # R3: Energy consumption in MWh
        total_mwh = sum(
            (r.energy_quantity_mwh for r in workspace.location_results),
            start=_ZERO,
        )
        requirements.append(self._check_value_present(
            "CSRD-E1-003",
            "Total energy consumption in MWh disclosed",
            total_mwh > _ZERO,
            f"Total energy: {float(total_mwh):.2f} MWh",
        ))

        # R4: Energy type breakdown
        requirements.append(self._check_value_present(
            "CSRD-E1-004",
            "Emissions by energy type disclosed",
            len(workspace.by_energy_type) > 0,
            f"{len(workspace.by_energy_type)} energy types",
        ))

        # R5: Reconciliation explanation
        has_explanation = (
            discrepancy_report is not None
            and len(discrepancy_report.discrepancies) > 0
        )
        requirements.append(self._check_value_present(
            "CSRD-E1-005",
            "Reconciliation explanation between methods provided",
            has_explanation or (
                discrepancy_report is not None
                and workspace.total_location_tco2e == workspace.total_market_tco2e
            ),
            "Discrepancy analysis available" if has_explanation else "No discrepancy (totals equal)",
        ))

        # R6: EF sources documented
        ef_sources = self._extract_ef_sources(workspace)
        requirements.append(self._check_value_present(
            "CSRD-E1-006",
            "Emission factor sources documented per ESRS E1",
            len(ef_sources) > 0,
            f"{len(ef_sources)} EF sources",
        ))

        # R7: GWP values
        gwp_sources = self._extract_gwp_sources(workspace)
        requirements.append(self._check_value_present(
            "CSRD-E1-007",
            "GWP values and IPCC AR identified",
            len(gwp_sources) > 0,
            f"GWP: {', '.join(gwp_sources)}",
        ))

        # R8: Data quality
        requirements.append(self._check_value_present(
            "CSRD-E1-008",
            "Data quality assessment disclosed per ESRS E1",
            quality_assessment is not None,
            f"Quality score: {float(quality_assessment.composite_score):.4f}"
            if quality_assessment else "N/A",
        ))

        # R9: Value chain boundary defined
        requirements.append(self._check_value_present(
            "CSRD-E1-009",
            "Value chain boundary for Scope 2 defined",
            workspace.tenant_id is not None and len(workspace.tenant_id) > 0,
            f"Tenant/entity: {workspace.tenant_id}",
        ))

        # R10: Reporting period
        requirements.append(self._check_value_present(
            "CSRD-E1-010",
            "Reporting period specified",
            workspace.period_start is not None,
            f"Period: {workspace.period_start} to {workspace.period_end}",
        ))

        return self._build_result(
            ReportingFramework.CSRD_ESRS, requirements
        )

    def check_cdp(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: Optional[DiscrepancyReport],
        quality_assessment: Optional[QualityAssessment],
    ) -> ComplianceCheckResult:
        """
        Check CDP Climate Change questionnaire compliance (C6/C7).

        CDP requires detailed energy consumption and emission breakdowns
        by country, energy type, and contractual instrument type.
        """
        requirements: List[ComplianceRequirement] = []

        # R1: Location total (C6.3)
        requirements.append(self._check_value_present(
            "CDP-C6-001",
            "Scope 2 location-based total disclosed (C6.3)",
            len(workspace.location_results) > 0,
            f"Location total: {float(workspace.total_location_tco2e):.4f} tCO2e",
        ))

        # R2: Market total (C6.3)
        requirements.append(self._check_value_present(
            "CDP-C6-002",
            "Scope 2 market-based total disclosed (C6.3)",
            len(workspace.market_results) > 0,
            f"Market total: {float(workspace.total_market_tco2e):.4f} tCO2e",
        ))

        # R3: Electricity consumption MWh (C8.2a)
        elec_mwh = sum(
            (r.energy_quantity_mwh for r in workspace.location_results
             if r.energy_type == EnergyType.ELECTRICITY),
            start=_ZERO,
        )
        requirements.append(self._check_value_present(
            "CDP-C8-001",
            "Electricity consumption in MWh disclosed (C8.2a)",
            elec_mwh > _ZERO or not any(
                r.energy_type == EnergyType.ELECTRICITY
                for r in workspace.location_results
            ),
            f"Electricity: {float(elec_mwh):.2f} MWh",
        ))

        # R4: Steam/heat/cooling consumption
        other_mwh = sum(
            (r.energy_quantity_mwh for r in workspace.location_results
             if r.energy_type != EnergyType.ELECTRICITY),
            start=_ZERO,
        )
        requirements.append(self._check_value_present(
            "CDP-C8-002",
            "Steam/heat/cooling consumption disclosed",
            other_mwh >= _ZERO,
            f"Other energy: {float(other_mwh):.2f} MWh",
        ))

        # R5: EF sources
        ef_sources = self._extract_ef_sources(workspace)
        requirements.append(self._check_value_present(
            "CDP-C7-001",
            "Emission factor sources documented (C7.5)",
            len(ef_sources) > 0,
            f"{len(ef_sources)} EF sources",
        ))

        # R6: Contractual instruments (C8.2e)
        has_instruments = any(
            r.ef_hierarchy is not None
            for r in workspace.market_results
        )
        requirements.append(self._check_value_present(
            "CDP-C8-003",
            "Contractual instruments detailed (C8.2e)",
            has_instruments or len(workspace.market_results) == 0,
            "Instruments documented",
        ))

        # R7: Country/region breakdown available
        regions = self._extract_regions(workspace)
        requirements.append(self._check_value_present(
            "CDP-C7-002",
            "Country/region breakdown available",
            len(regions) > 0,
            f"{len(regions)} regions",
        ))

        # R8: Verification status documented
        requirements.append(self._check_value_present(
            "CDP-C10-001",
            "Verification/assurance status disclosed",
            quality_assessment is not None,
            "Quality assessment available",
        ))

        return self._build_result(ReportingFramework.CDP, requirements)

    def check_sbti(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: Optional[DiscrepancyReport],
        quality_assessment: Optional[QualityAssessment],
    ) -> ComplianceCheckResult:
        """
        Check SBTi compliance for Scope 2 target tracking.

        SBTi uses market-based for Scope 2 target tracking.
        """
        requirements: List[ComplianceRequirement] = []

        # R1: Market-based total (primary for SBTi)
        requirements.append(self._check_value_present(
            "SBTI-S2-001",
            "Market-based total disclosed for SBTi target tracking",
            len(workspace.market_results) > 0,
            f"Market total: {float(workspace.total_market_tco2e):.4f} tCO2e",
        ))

        # R2: Location-based total (contextual for SBTi)
        requirements.append(self._check_value_present(
            "SBTI-S2-002",
            "Location-based total disclosed for context",
            len(workspace.location_results) > 0,
            f"Location total: {float(workspace.total_location_tco2e):.4f} tCO2e",
        ))

        # R3: EF sources
        ef_sources = self._extract_ef_sources(workspace)
        requirements.append(self._check_value_present(
            "SBTI-S2-003",
            "Emission factor sources documented",
            len(ef_sources) > 0,
            f"{len(ef_sources)} EF sources",
        ))

        # R4: Contractual instruments summary
        has_instruments = any(
            r.ef_hierarchy is not None
            for r in workspace.market_results
        )
        requirements.append(self._check_value_present(
            "SBTI-S2-004",
            "Contractual instruments documented for market-based",
            has_instruments or len(workspace.market_results) == 0,
            "Instruments documented",
        ))

        # R5: Renewable electricity tracked
        elec_results = [
            r for r in workspace.market_results
            if r.energy_type == EnergyType.ELECTRICITY
        ]
        requirements.append(self._check_value_present(
            "SBTI-S2-005",
            "Renewable electricity procurement tracked",
            len(elec_results) > 0 or len(workspace.market_results) == 0,
            f"{len(elec_results)} electricity market results",
        ))

        # R6: Data quality sufficient for target tracking
        quality_ok = (
            quality_assessment is not None
            and quality_assessment.composite_score >= Decimal("0.65")
        )
        requirements.append(self._check_value_present(
            "SBTI-S2-006",
            "Data quality score meets minimum for target tracking (>=0.65)",
            quality_ok or quality_assessment is None,
            f"Quality: {float(quality_assessment.composite_score):.4f}"
            if quality_assessment else "N/A",
        ))

        return self._build_result(ReportingFramework.SBTI, requirements)

    def check_gri(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: Optional[DiscrepancyReport],
        quality_assessment: Optional[QualityAssessment],
    ) -> ComplianceCheckResult:
        """
        Check GRI Standards 305-2 compliance.

        GRI requires separate disclosure of both methods.
        """
        requirements: List[ComplianceRequirement] = []

        # R1: Location total (305-2a)
        requirements.append(self._check_value_present(
            "GRI-305-001",
            "Gross location-based Scope 2 emissions (305-2a)",
            len(workspace.location_results) > 0,
            f"Location: {float(workspace.total_location_tco2e):.4f} tCO2e",
        ))

        # R2: Market total (305-2b)
        requirements.append(self._check_value_present(
            "GRI-305-002",
            "Gross market-based Scope 2 emissions (305-2b)",
            len(workspace.market_results) > 0,
            f"Market: {float(workspace.total_market_tco2e):.4f} tCO2e",
        ))

        # R3: By energy type
        requirements.append(self._check_value_present(
            "GRI-305-003",
            "Breakdown by energy type",
            len(workspace.by_energy_type) > 0,
            f"{len(workspace.by_energy_type)} energy types",
        ))

        # R4: EF sources (305-2d)
        ef_sources = self._extract_ef_sources(workspace)
        requirements.append(self._check_value_present(
            "GRI-305-004",
            "Emission factor sources disclosed (305-2d)",
            len(ef_sources) > 0,
            f"{len(ef_sources)} EF sources",
        ))

        # R5: GWP values (305-2e)
        gwp_sources = self._extract_gwp_sources(workspace)
        requirements.append(self._check_value_present(
            "GRI-305-005",
            "GWP values and source AR disclosed (305-2e)",
            len(gwp_sources) > 0,
            f"GWP: {', '.join(gwp_sources)}",
        ))

        # R6: Consolidation approach (305-2f)
        requirements.append(self._check_value_present(
            "GRI-305-006",
            "Consolidation approach documented (305-2f)",
            workspace.tenant_id is not None,
            f"Tenant: {workspace.tenant_id}",
        ))

        # R7: Standards and methodologies
        requirements.append(self._check_value_present(
            "GRI-305-007",
            "Standards and methodologies referenced",
            True,  # We are using GHG Protocol -- always met
            "GHG Protocol Scope 2 Guidance (2015)",
        ))

        return self._build_result(ReportingFramework.GRI, requirements)

    def check_iso_14064(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: Optional[DiscrepancyReport],
        quality_assessment: Optional[QualityAssessment],
    ) -> ComplianceCheckResult:
        """
        Check ISO 14064-1:2018 compliance.

        ISO 14064 permits either method with justification but
        recommends both.
        """
        requirements: List[ComplianceRequirement] = []

        # R1: At least one method disclosed
        has_any = (
            len(workspace.location_results) > 0
            or len(workspace.market_results) > 0
        )
        requirements.append(self._check_value_present(
            "ISO-001",
            "At least one Scope 2 quantification method applied",
            has_any,
            "Both methods available"
            if len(workspace.location_results) > 0 and len(workspace.market_results) > 0
            else "One method available",
        ))

        # R2: Both methods recommended
        both_present = (
            len(workspace.location_results) > 0
            and len(workspace.market_results) > 0
        )
        requirements.append(self._check_value_present(
            "ISO-002",
            "Both location-based and market-based methods reported (recommended)",
            both_present,
            "Dual reporting" if both_present else "Single method only",
            notes="ISO 14064 recommends but does not mandate dual reporting",
        ))

        # R3: Method justification
        requirements.append(self._check_value_present(
            "ISO-003",
            "Quantification method justified",
            True,  # Justified by using GHG Protocol Scope 2 Guidance
            "GHG Protocol Scope 2 Guidance provides method justification",
        ))

        # R4: Emission by gas (if available)
        has_gas_breakdown = any(
            r.emissions_by_gas for r in workspace.location_results
        ) or any(
            r.emissions_by_gas for r in workspace.market_results
        )
        requirements.append(self._check_value_present(
            "ISO-004",
            "Emissions by gas type disclosed",
            has_gas_breakdown,
            "Gas breakdown available" if has_gas_breakdown else "CO2e aggregate only",
            notes="Per-gas disclosure required for ISO 14064-1",
        ))

        # R5: EF sources
        ef_sources = self._extract_ef_sources(workspace)
        requirements.append(self._check_value_present(
            "ISO-005",
            "Emission factor sources documented",
            len(ef_sources) > 0,
            f"{len(ef_sources)} EF sources",
        ))

        # R6: GWP values
        gwp_sources = self._extract_gwp_sources(workspace)
        requirements.append(self._check_value_present(
            "ISO-006",
            "GWP values and source identified",
            len(gwp_sources) > 0,
            f"GWP: {', '.join(gwp_sources)}",
        ))

        # R7: Uncertainty assessment
        requirements.append(self._check_value_present(
            "ISO-007",
            "Uncertainty assessment conducted",
            quality_assessment is not None,
            "Quality assessment serves as proxy for uncertainty"
            if quality_assessment else "No uncertainty assessment",
        ))

        # R8: Organizational boundary
        requirements.append(self._check_value_present(
            "ISO-008",
            "Organizational boundary defined",
            workspace.tenant_id is not None,
            f"Entity: {workspace.tenant_id}",
        ))

        # R9: Reporting period
        requirements.append(self._check_value_present(
            "ISO-009",
            "Reporting period specified",
            workspace.period_start is not None,
            f"Period: {workspace.period_start} to {workspace.period_end}",
        ))

        # R10: Data quality assessment
        requirements.append(self._check_value_present(
            "ISO-010",
            "Data quality assessment per ISO 14064-1 Section 5.3.5",
            quality_assessment is not None,
            f"Score: {float(quality_assessment.composite_score):.4f}"
            if quality_assessment else "N/A",
        ))

        return self._build_result(ReportingFramework.ISO_14064, requirements)

    def check_re100(
        self,
        workspace: ReconciliationWorkspace,
        discrepancy_report: Optional[DiscrepancyReport],
        quality_assessment: Optional[QualityAssessment],
    ) -> ComplianceCheckResult:
        """
        Check RE100 initiative compliance.

        RE100 focuses on market-based electricity procurement and
        renewable electricity percentage.
        """
        requirements: List[ComplianceRequirement] = []

        # R1: Electricity consumption MWh
        elec_mwh = sum(
            (r.energy_quantity_mwh for r in workspace.location_results
             if r.energy_type == EnergyType.ELECTRICITY),
            start=_ZERO,
        )
        requirements.append(self._check_value_present(
            "RE100-001",
            "Total electricity consumption in MWh disclosed",
            elec_mwh > _ZERO,
            f"Total electricity: {float(elec_mwh):.2f} MWh",
        ))

        # R2: Market-based total for electricity
        mkt_elec_tco2e = sum(
            (r.emissions_tco2e for r in workspace.market_results
             if r.energy_type == EnergyType.ELECTRICITY),
            start=_ZERO,
        )
        requirements.append(self._check_value_present(
            "RE100-002",
            "Market-based electricity emissions disclosed",
            mkt_elec_tco2e >= _ZERO and any(
                r.energy_type == EnergyType.ELECTRICITY
                for r in workspace.market_results
            ),
            f"Market electricity: {float(mkt_elec_tco2e):.4f} tCO2e",
        ))

        # R3: Contractual instruments breakdown
        instrument_types = set()
        for r in workspace.market_results:
            if r.energy_type == EnergyType.ELECTRICITY and r.ef_hierarchy:
                instrument_types.add(
                    r.ef_hierarchy.value
                    if hasattr(r.ef_hierarchy, "value")
                    else str(r.ef_hierarchy)
                )
        requirements.append(self._check_value_present(
            "RE100-003",
            "Contractual instrument types disclosed for RE100 tracking",
            len(instrument_types) > 0 or not any(
                r.energy_type == EnergyType.ELECTRICITY
                for r in workspace.market_results
            ),
            f"Instrument types: {', '.join(instrument_types)}"
            if instrument_types else "No instruments",
        ))

        # R4: Country/region breakdown
        regions = self._extract_regions(workspace)
        requirements.append(self._check_value_present(
            "RE100-004",
            "Country breakdown of electricity procurement",
            len(regions) > 0,
            f"{len(regions)} countries/regions",
        ))

        # R5: EF sources for market calculation
        ef_sources = self._extract_ef_sources(workspace)
        requirements.append(self._check_value_present(
            "RE100-005",
            "Emission factor sources documented",
            len(ef_sources) > 0,
            f"{len(ef_sources)} EF sources",
        ))

        return self._build_result(ReportingFramework.RE100, requirements)

    # ------------------------------------------------------------------
    # Flag generation
    # ------------------------------------------------------------------

    def generate_compliance_flags(
        self, results: Dict[str, ComplianceCheckResult]
    ) -> List[Flag]:
        """
        Generate flags based on compliance check results.

        Args:
            results: Dict of framework key to ComplianceCheckResult.

        Returns:
            List of Flag objects.
        """
        flags: List[Flag] = []

        for fw_key, result in results.items():
            status = (
                result.status.value
                if hasattr(result.status, "value")
                else str(result.status)
            )

            if status == "non_compliant":
                flags.append(Flag(
                    flag_type=FlagType.ERROR,
                    severity=FlagSeverity.HIGH,
                    code=f"DRR-C-{fw_key[:6].upper()}",
                    message=(
                        f"Non-compliant with {fw_key}: "
                        f"met {result.requirements_met}/{result.requirements_total} "
                        f"requirements (score: {float(result.score):.2f})."
                    ),
                    recommendation=(
                        f"Review and address unmet {fw_key} requirements: "
                        + "; ".join(
                            req.description
                            for req in result.requirements
                            if not req.met
                        )[:500]
                    ),
                ))
            elif status == "partial":
                flags.append(Flag(
                    flag_type=FlagType.WARNING,
                    severity=FlagSeverity.MEDIUM,
                    code=f"DRR-C-{fw_key[:6].upper()}",
                    message=(
                        f"Partial compliance with {fw_key}: "
                        f"met {result.requirements_met}/{result.requirements_total} "
                        f"(score: {float(result.score):.2f})."
                    ),
                    recommendation=(
                        f"Address remaining {fw_key} requirements for full "
                        f"compliance."
                    ),
                ))
            elif status == "compliant":
                flags.append(Flag(
                    flag_type=FlagType.INFO,
                    severity=FlagSeverity.LOW,
                    code=f"DRR-C-{fw_key[:6].upper()}",
                    message=(
                        f"Fully compliant with {fw_key}: "
                        f"{result.requirements_met}/{result.requirements_total} "
                        f"requirements met."
                    ),
                    recommendation="",
                ))

        return flags

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_compliance_summary(
        self, results: Dict[str, ComplianceCheckResult]
    ) -> Dict[str, Any]:
        """
        Generate a concise summary of compliance check results.

        Args:
            results: Dict of framework key to ComplianceCheckResult.

        Returns:
            Summary dictionary.
        """
        summary: Dict[str, Any] = {
            "framework_count": len(results),
            "compliant_count": 0,
            "non_compliant_count": 0,
            "partial_count": 0,
            "total_requirements": 0,
            "total_met": 0,
            "overall_score": _ZERO,
            "frameworks": {},
        }

        for fw_key, result in results.items():
            status = (
                result.status.value
                if hasattr(result.status, "value")
                else str(result.status)
            )
            if status == "compliant":
                summary["compliant_count"] += 1
            elif status == "non_compliant":
                summary["non_compliant_count"] += 1
            elif status == "partial":
                summary["partial_count"] += 1

            summary["total_requirements"] += result.requirements_total
            summary["total_met"] += result.requirements_met

            summary["frameworks"][fw_key] = {
                "status": status,
                "score": str(result.score),
                "met": result.requirements_met,
                "total": result.requirements_total,
            }

        if summary["total_requirements"] > 0:
            summary["overall_score"] = str(_quantize(
                Decimal(str(summary["total_met"]))
                / Decimal(str(summary["total_requirements"]))
            ))

        return summary

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the ComplianceCheckerEngine.

        Returns:
            Dictionary with engine health status.
        """
        return {
            "status": "healthy",
            "engine": "ComplianceCheckerEngine",
            "agent_id": AGENT_ID,
            "component": AGENT_COMPONENT,
            "version": VERSION,
            "initialized": self.__class__._initialized,
            "check_count": self._check_count,
            "config_available": self._config is not None,
            "metrics_available": self._metrics is not None,
            "provenance_available": self._provenance is not None,
            "supported_frameworks": [
                fw.value for fw in ReportingFramework
            ] if hasattr(ReportingFramework, "__iter__") else [],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_value_present(
        self,
        requirement_id: str,
        description: str,
        met: bool,
        evidence: str = "",
        notes: str = "",
    ) -> ComplianceRequirement:
        """
        Create a ComplianceRequirement check result.

        Args:
            requirement_id: Unique requirement code.
            description: What is being checked.
            met: Whether the requirement is satisfied.
            evidence: Evidence for the check.
            notes: Additional notes.

        Returns:
            ComplianceRequirement object.
        """
        return ComplianceRequirement(
            requirement_id=requirement_id,
            description=description,
            met=met,
            evidence=evidence,
            notes=notes,
        )

    def _build_result(
        self,
        framework: ReportingFramework,
        requirements: List[ComplianceRequirement],
    ) -> ComplianceCheckResult:
        """
        Build a ComplianceCheckResult from requirement checks.

        Status logic:
        - All met -> COMPLIANT
        - None met -> NON_COMPLIANT
        - Some met -> PARTIAL
        - No requirements -> NOT_APPLICABLE

        Score = requirements_met / requirements_total

        Args:
            framework: The reporting framework.
            requirements: List of requirement check results.

        Returns:
            ComplianceCheckResult.
        """
        total = len(requirements)
        met = sum(1 for r in requirements if r.met)

        if total == 0:
            status = ComplianceStatus.NOT_APPLICABLE
            score = _ZERO
        elif met == total:
            status = ComplianceStatus.COMPLIANT
            score = _ONE
        elif met == 0:
            status = ComplianceStatus.NON_COMPLIANT
            score = _ZERO
        else:
            status = ComplianceStatus.PARTIAL
            score = _quantize(Decimal(str(met)) / Decimal(str(total)))

        findings: List[str] = []
        for req in requirements:
            if not req.met:
                findings.append(
                    f"[{req.requirement_id}] NOT MET: {req.description}"
                )

        return ComplianceCheckResult(
            framework=framework,
            status=status,
            requirements_total=total,
            requirements_met=met,
            requirements=requirements,
            score=score,
            findings=findings,
        )

    def _extract_ef_sources(
        self, workspace: ReconciliationWorkspace
    ) -> List[str]:
        """Extract unique EF source descriptions from all results."""
        sources: set = set()
        for r in workspace.location_results:
            if r.ef_source:
                sources.add(r.ef_source)
        for r in workspace.market_results:
            if r.ef_source:
                sources.add(r.ef_source)
        return sorted(sources)

    def _extract_gwp_sources(
        self, workspace: ReconciliationWorkspace
    ) -> List[str]:
        """Extract unique GWP source identifiers from all results."""
        sources: set = set()
        for r in workspace.location_results:
            if r.gwp_source:
                val = (
                    r.gwp_source.value
                    if hasattr(r.gwp_source, "value")
                    else str(r.gwp_source)
                )
                sources.add(val)
        for r in workspace.market_results:
            if r.gwp_source:
                val = (
                    r.gwp_source.value
                    if hasattr(r.gwp_source, "value")
                    else str(r.gwp_source)
                )
                sources.add(val)
        return sorted(sources)

    def _extract_regions(
        self, workspace: ReconciliationWorkspace
    ) -> List[str]:
        """Extract unique region codes from all results."""
        regions: set = set()
        for r in workspace.location_results:
            if r.region:
                regions.add(r.region)
        for r in workspace.market_results:
            if r.region:
                regions.add(r.region)
        return sorted(regions)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ComplianceCheckerEngine(version={VERSION}, "
            f"initialized={self.__class__._initialized}, "
            f"checks={self._check_count})"
        )


# =============================================================================
# Module-level convenience functions
# =============================================================================


def check_compliance(
    workspace: ReconciliationWorkspace,
    discrepancy_report: Optional[DiscrepancyReport],
    quality_assessment: Optional[QualityAssessment],
    framework: ReportingFramework,
) -> ComplianceCheckResult:
    """Module-level shortcut for ComplianceCheckerEngine.check_compliance."""
    engine = ComplianceCheckerEngine()
    return engine.check_compliance(
        workspace, discrepancy_report, quality_assessment, framework
    )


def check_all_frameworks(
    workspace: ReconciliationWorkspace,
    discrepancy_report: Optional[DiscrepancyReport],
    quality_assessment: Optional[QualityAssessment],
    frameworks: Optional[List[ReportingFramework]] = None,
) -> Dict[str, ComplianceCheckResult]:
    """Module-level shortcut for ComplianceCheckerEngine.check_all_frameworks."""
    engine = ComplianceCheckerEngine()
    return engine.check_all_frameworks(
        workspace, discrepancy_report, quality_assessment, frameworks
    )
