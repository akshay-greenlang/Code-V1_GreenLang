# -*- coding: utf-8 -*-
"""
TaxonomyBridge - Bridge to GL-Taxonomy-APP for PACK-022 Acceleration
========================================================================

This module bridges the Net Zero Acceleration Pack to GL-Taxonomy-APP
(APP-010) for EU Taxonomy climate mitigation and adaptation alignment
assessment. Provides Technical Screening Criteria (TSC) lookup, substantial
contribution evaluation, DNSH (Do No Significant Harm) assessment, and
taxonomy-aligned KPI calculation.

Functions:
    - check_alignment()                  -- Check EU Taxonomy alignment for activities
    - get_tsc_criteria()                 -- Get TSC criteria for a NACE code
    - evaluate_substantial_contribution()-- Evaluate substantial contribution
    - evaluate_dnsh()                    -- Evaluate DNSH compliance
    - calculate_taxonomy_kpis()          -- Calculate taxonomy-aligned KPIs

EU Taxonomy Objectives:
    1. Climate change mitigation
    2. Climate change adaptation
    3. Sustainable use of water and marine resources
    4. Transition to a circular economy
    5. Pollution prevention and control
    6. Protection of biodiversity and ecosystems

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-022 Net Zero Acceleration Pack
Status: Production Ready
"""

import hashlib
import importlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Agent Stubs
# ---------------------------------------------------------------------------


class _AgentStub:
    """Stub for unavailable taxonomy app modules."""

    def __init__(self, component_name: str) -> None:
        self._component_name = component_name
        self._available = False

    def __getattr__(self, name: str) -> Any:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "component": self._component_name,
                "method": name,
                "status": "degraded",
                "message": f"{self._component_name} not available, using stub",
            }
        return _stub_method


def _try_import_taxonomy_component(component_id: str, module_path: str) -> Any:
    """Try to import a taxonomy component with graceful fallback.

    Args:
        component_id: Component identifier.
        module_path: Python module path.

    Returns:
        Imported module or _AgentStub if unavailable.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("Taxonomy component %s not available, using stub", component_id)
        return _AgentStub(component_id)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaxonomyObjective(str, Enum):
    """EU Taxonomy environmental objectives."""

    CLIMATE_MITIGATION = "climate_mitigation"
    CLIMATE_ADAPTATION = "climate_adaptation"
    WATER_MARINE = "water_marine"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity"


class AlignmentStatus(str, Enum):
    """Taxonomy alignment assessment status."""

    ALIGNED = "aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    NOT_ALIGNED = "not_aligned"
    INSUFFICIENT_DATA = "insufficient_data"


class DNSHStatus(str, Enum):
    """DNSH assessment status per objective."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DATA = "insufficient_data"


class SubstantialContributionLevel(str, Enum):
    """Substantial contribution level."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NONE = "none"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class TaxonomyBridgeConfig(BaseModel):
    """Configuration for the Taxonomy Bridge."""

    pack_id: str = Field(default="PACK-022")
    enable_provenance: bool = Field(default=True)
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    primary_objective: TaxonomyObjective = Field(
        default=TaxonomyObjective.CLIMATE_MITIGATION,
    )
    nace_codes: List[str] = Field(default_factory=list)
    include_adaptation: bool = Field(default=True)
    minimum_social_safeguards: bool = Field(default=True)


class TSCCriteria(BaseModel):
    """Technical Screening Criteria for an activity."""

    nace_code: str = Field(default="")
    activity_name: str = Field(default="")
    objective: TaxonomyObjective = Field(default=TaxonomyObjective.CLIMATE_MITIGATION)
    criteria_description: str = Field(default="")
    thresholds: Dict[str, Any] = Field(default_factory=dict)
    metrics: List[str] = Field(default_factory=list)
    delegated_act_reference: str = Field(default="")


class AlignmentResult(BaseModel):
    """Result of taxonomy alignment assessment."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    nace_code: str = Field(default="")
    activity_name: str = Field(default="")
    alignment_status: AlignmentStatus = Field(default=AlignmentStatus.INSUFFICIENT_DATA)
    substantial_contribution: bool = Field(default=False)
    dnsh_compliant: bool = Field(default=False)
    social_safeguards_met: bool = Field(default=False)
    alignment_score: float = Field(default=0.0, ge=0.0, le=100.0)
    objectives_assessed: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class SubstantialContributionResult(BaseModel):
    """Result of substantial contribution evaluation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    nace_code: str = Field(default="")
    objective: TaxonomyObjective = Field(default=TaxonomyObjective.CLIMATE_MITIGATION)
    contributes: bool = Field(default=False)
    contribution_level: SubstantialContributionLevel = Field(
        default=SubstantialContributionLevel.NONE,
    )
    criteria_met: List[Dict[str, Any]] = Field(default_factory=list)
    criteria_not_met: List[Dict[str, Any]] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class DNSHResult(BaseModel):
    """Result of DNSH (Do No Significant Harm) evaluation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    nace_code: str = Field(default="")
    overall_dnsh_pass: bool = Field(default=False)
    assessments: Dict[str, DNSHStatus] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class TaxonomyKPIResult(BaseModel):
    """Result of taxonomy-aligned KPI calculation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    reporting_year: int = Field(default=2025)
    taxonomy_eligible_turnover_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    taxonomy_aligned_turnover_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    taxonomy_eligible_capex_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    taxonomy_aligned_capex_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    taxonomy_eligible_opex_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    taxonomy_aligned_opex_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    activities_assessed: int = Field(default=0, ge=0)
    activities_aligned: int = Field(default=0, ge=0)
    by_activity: List[Dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# TSC Reference Data
# ---------------------------------------------------------------------------

# Reference TSC thresholds for common climate mitigation activities
TSC_REFERENCE: Dict[str, Dict[str, Any]] = {
    "C28": {
        "activity": "Manufacture of machinery and equipment",
        "objective": TaxonomyObjective.CLIMATE_MITIGATION.value,
        "criteria": "Equipment must meet best-available-technology energy efficiency benchmarks",
        "thresholds": {
            "energy_efficiency_improvement_pct": 30.0,
            "lifecycle_ghg_reduction_pct": 50.0,
        },
        "metrics": ["energy_efficiency_pct", "ghg_intensity_per_unit"],
    },
    "D35.1": {
        "activity": "Electricity generation",
        "objective": TaxonomyObjective.CLIMATE_MITIGATION.value,
        "criteria": "Lifecycle GHG emissions below 100 gCO2e/kWh",
        "thresholds": {
            "lifecycle_ghg_gco2e_per_kwh": 100.0,
        },
        "metrics": ["lifecycle_ghg_gco2e_per_kwh", "capacity_factor_pct"],
    },
    "F41": {
        "activity": "Construction of buildings",
        "objective": TaxonomyObjective.CLIMATE_MITIGATION.value,
        "criteria": "Primary Energy Demand at least 10% below NZEB requirement",
        "thresholds": {
            "ped_below_nzeb_pct": 10.0,
            "airtightness_n50": 3.0,
        },
        "metrics": ["primary_energy_demand_kwh_m2", "airtightness_n50"],
    },
    "H49": {
        "activity": "Land transport and transport via pipelines",
        "objective": TaxonomyObjective.CLIMATE_MITIGATION.value,
        "criteria": "Zero direct (tailpipe) CO2 emissions",
        "thresholds": {
            "direct_co2_g_per_km": 0.0,
        },
        "metrics": ["direct_co2_g_per_km", "energy_consumption_kwh_per_km"],
    },
    "J61": {
        "activity": "Telecommunications",
        "objective": TaxonomyObjective.CLIMATE_MITIGATION.value,
        "criteria": "Implement GHG reduction targets aligned with SBTi",
        "thresholds": {
            "sbti_target_set": True,
            "renewable_energy_pct": 50.0,
        },
        "metrics": ["pue_ratio", "renewable_energy_pct"],
    },
    "L68": {
        "activity": "Real estate activities",
        "objective": TaxonomyObjective.CLIMATE_MITIGATION.value,
        "criteria": "Buildings with EPC class A or top 15% national stock",
        "thresholds": {
            "epc_class": "A",
            "top_stock_pct": 15.0,
        },
        "metrics": ["epc_class", "primary_energy_demand_kwh_m2"],
    },
}

# Taxonomy component routing
TAXONOMY_COMPONENTS: Dict[str, str] = {
    "alignment_engine": "greenlang.apps.taxonomy.alignment_engine",
    "tsc_resolver": "greenlang.apps.taxonomy.tsc_resolver",
    "dnsh_evaluator": "greenlang.apps.taxonomy.dnsh_evaluator",
    "kpi_calculator": "greenlang.apps.taxonomy.kpi_calculator",
    "social_safeguards": "greenlang.apps.taxonomy.social_safeguards",
}


# ---------------------------------------------------------------------------
# TaxonomyBridge
# ---------------------------------------------------------------------------


class TaxonomyBridge:
    """Bridge to GL-Taxonomy-APP for EU Taxonomy alignment assessment.

    Provides TSC criteria lookup, substantial contribution evaluation,
    DNSH assessment, and taxonomy-aligned KPI calculation for
    climate mitigation and adaptation.

    Attributes:
        config: Bridge configuration.
        _components: Dict of loaded taxonomy component modules/stubs.

    Example:
        >>> bridge = TaxonomyBridge(TaxonomyBridgeConfig(reporting_year=2025))
        >>> alignment = bridge.check_alignment("C28", activity_data)
        >>> assert alignment.alignment_status == AlignmentStatus.ALIGNED
    """

    def __init__(self, config: Optional[TaxonomyBridgeConfig] = None) -> None:
        """Initialize TaxonomyBridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or TaxonomyBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._components: Dict[str, Any] = {}
        for comp_id, module_path in TAXONOMY_COMPONENTS.items():
            self._components[comp_id] = _try_import_taxonomy_component(
                comp_id, module_path
            )

        available = sum(
            1 for c in self._components.values() if not isinstance(c, _AgentStub)
        )
        self.logger.info(
            "TaxonomyBridge initialized: %d/%d components, year=%d, objective=%s",
            available, len(self._components),
            self.config.reporting_year, self.config.primary_objective.value,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def check_alignment(
        self,
        nace_code: str,
        activity_data: Optional[Dict[str, Any]] = None,
    ) -> AlignmentResult:
        """Check EU Taxonomy alignment for an economic activity.

        Evaluates substantial contribution, DNSH, and social safeguards
        to determine overall alignment status.

        Args:
            nace_code: NACE code of the activity.
            activity_data: Dict with activity metrics and performance data.

        Returns:
            AlignmentResult with alignment status and score.
        """
        start = time.monotonic()
        activity_data = activity_data or {}
        result = AlignmentResult(nace_code=nace_code)

        try:
            # Resolve activity name from TSC reference
            tsc_ref = TSC_REFERENCE.get(nace_code, {})
            result.activity_name = tsc_ref.get("activity", f"Activity {nace_code}")

            # Step 1: Evaluate substantial contribution
            sc_result = self.evaluate_substantial_contribution(
                nace_code, activity_data
            )
            result.substantial_contribution = sc_result.contributes

            # Step 2: Evaluate DNSH
            dnsh_result = self.evaluate_dnsh(nace_code, activity_data)
            result.dnsh_compliant = dnsh_result.overall_dnsh_pass

            # Step 3: Check social safeguards
            result.social_safeguards_met = activity_data.get(
                "social_safeguards_met", self.config.minimum_social_safeguards
            )

            # Step 4: Build objectives assessment
            objectives_assessed = [
                {
                    "objective": self.config.primary_objective.value,
                    "substantial_contribution": sc_result.contributes,
                    "contribution_level": sc_result.contribution_level.value,
                    "score": sc_result.score,
                },
            ]
            if self.config.include_adaptation:
                objectives_assessed.append({
                    "objective": TaxonomyObjective.CLIMATE_ADAPTATION.value,
                    "substantial_contribution": False,
                    "contribution_level": SubstantialContributionLevel.NONE.value,
                    "score": 0.0,
                })
            result.objectives_assessed = objectives_assessed

            # Step 5: Determine overall alignment
            if (result.substantial_contribution
                    and result.dnsh_compliant
                    and result.social_safeguards_met):
                result.alignment_status = AlignmentStatus.ALIGNED
                result.alignment_score = sc_result.score
            elif result.substantial_contribution:
                result.alignment_status = AlignmentStatus.PARTIALLY_ALIGNED
                result.alignment_score = sc_result.score * 0.6
            else:
                result.alignment_status = AlignmentStatus.NOT_ALIGNED
                result.alignment_score = 0.0

            # Build recommendations
            if not result.substantial_contribution:
                result.recommendations.append(
                    f"Activity does not meet TSC for {self.config.primary_objective.value}"
                )
            if not result.dnsh_compliant:
                for issue in dnsh_result.issues:
                    result.recommendations.append(f"DNSH issue: {issue}")
            if not result.social_safeguards_met:
                result.recommendations.append(
                    "Verify compliance with minimum social safeguards (OECD, UN Guiding Principles)"
                )

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("Alignment check failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_tsc_criteria(
        self,
        nace_code: str,
    ) -> Optional[TSCCriteria]:
        """Get Technical Screening Criteria for a NACE code.

        Args:
            nace_code: NACE code to look up.

        Returns:
            TSCCriteria if found, None otherwise.
        """
        ref = TSC_REFERENCE.get(nace_code)
        if ref is None:
            self.logger.debug("No TSC reference for NACE %s", nace_code)
            return None

        return TSCCriteria(
            nace_code=nace_code,
            activity_name=ref.get("activity", ""),
            objective=TaxonomyObjective(ref.get(
                "objective", TaxonomyObjective.CLIMATE_MITIGATION.value
            )),
            criteria_description=ref.get("criteria", ""),
            thresholds=ref.get("thresholds", {}),
            metrics=ref.get("metrics", []),
            delegated_act_reference=f"EU 2021/2139 Annex I - {nace_code}",
        )

    def evaluate_substantial_contribution(
        self,
        nace_code: str,
        activity_data: Optional[Dict[str, Any]] = None,
    ) -> SubstantialContributionResult:
        """Evaluate substantial contribution to climate mitigation.

        Args:
            nace_code: NACE code of the activity.
            activity_data: Dict with performance metrics for threshold checking.

        Returns:
            SubstantialContributionResult with pass/fail per criterion.
        """
        start = time.monotonic()
        activity_data = activity_data or {}
        result = SubstantialContributionResult(
            nace_code=nace_code,
            objective=self.config.primary_objective,
        )

        try:
            ref = TSC_REFERENCE.get(nace_code, {})
            thresholds = ref.get("thresholds", {})

            if not thresholds:
                result.contributes = False
                result.contribution_level = SubstantialContributionLevel.NONE
                result.score = 0.0
                result.status = "completed"
                result.duration_ms = (time.monotonic() - start) * 1000
                if self.config.enable_provenance:
                    result.provenance_hash = _compute_hash(result)
                return result

            criteria_met: List[Dict[str, Any]] = []
            criteria_not_met: List[Dict[str, Any]] = []

            for metric, threshold in thresholds.items():
                actual_value = activity_data.get(metric)
                criterion = {
                    "metric": metric,
                    "threshold": threshold,
                    "actual": actual_value,
                }

                if actual_value is None:
                    criterion["result"] = "insufficient_data"
                    criteria_not_met.append(criterion)
                elif isinstance(threshold, bool):
                    if bool(actual_value) == threshold:
                        criterion["result"] = "met"
                        criteria_met.append(criterion)
                    else:
                        criterion["result"] = "not_met"
                        criteria_not_met.append(criterion)
                elif isinstance(threshold, (int, float)):
                    # For metrics like ghg_gco2e, lower is better
                    if "ghg" in metric or "co2" in metric or "emission" in metric:
                        if actual_value <= threshold:
                            criterion["result"] = "met"
                            criteria_met.append(criterion)
                        else:
                            criterion["result"] = "not_met"
                            criteria_not_met.append(criterion)
                    else:
                        # For metrics like efficiency, higher is better
                        if actual_value >= threshold:
                            criterion["result"] = "met"
                            criteria_met.append(criterion)
                        else:
                            criterion["result"] = "not_met"
                            criteria_not_met.append(criterion)
                else:
                    # String comparison (e.g., EPC class)
                    if str(actual_value) == str(threshold):
                        criterion["result"] = "met"
                        criteria_met.append(criterion)
                    else:
                        criterion["result"] = "not_met"
                        criteria_not_met.append(criterion)

            result.criteria_met = criteria_met
            result.criteria_not_met = criteria_not_met

            total_criteria = len(criteria_met) + len(criteria_not_met)
            if total_criteria > 0:
                result.score = round(
                    (len(criteria_met) / total_criteria) * 100.0, 1
                )

            # Determine contribution level
            if result.score >= 80.0:
                result.contributes = True
                result.contribution_level = SubstantialContributionLevel.HIGH
            elif result.score >= 50.0:
                result.contributes = True
                result.contribution_level = SubstantialContributionLevel.MODERATE
            elif result.score >= 25.0:
                result.contributes = False
                result.contribution_level = SubstantialContributionLevel.LOW
            else:
                result.contributes = False
                result.contribution_level = SubstantialContributionLevel.NONE

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("Substantial contribution evaluation failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def evaluate_dnsh(
        self,
        nace_code: str,
        activity_data: Optional[Dict[str, Any]] = None,
    ) -> DNSHResult:
        """Evaluate DNSH (Do No Significant Harm) compliance.

        Checks that the activity does not significantly harm any of
        the other five environmental objectives.

        Args:
            nace_code: NACE code of the activity.
            activity_data: Dict with environmental impact data.

        Returns:
            DNSHResult with per-objective pass/fail status.
        """
        start = time.monotonic()
        activity_data = activity_data or {}
        result = DNSHResult(nace_code=nace_code)

        try:
            # For each objective that is not the primary, check DNSH
            other_objectives = [
                obj for obj in TaxonomyObjective
                if obj != self.config.primary_objective
            ]

            assessments: Dict[str, DNSHStatus] = {}
            issues: List[str] = []

            for obj in other_objectives:
                obj_key = obj.value
                dnsh_data = activity_data.get(f"dnsh_{obj_key}")

                if dnsh_data is None:
                    # Default to not applicable if no data provided
                    if obj_key in activity_data.get("applicable_dnsh_objectives", []):
                        assessments[obj_key] = DNSHStatus.INSUFFICIENT_DATA
                        issues.append(
                            f"Missing DNSH data for {obj_key}"
                        )
                    else:
                        assessments[obj_key] = DNSHStatus.NOT_APPLICABLE
                elif isinstance(dnsh_data, bool):
                    if dnsh_data:
                        assessments[obj_key] = DNSHStatus.PASS
                    else:
                        assessments[obj_key] = DNSHStatus.FAIL
                        issues.append(
                            f"DNSH failed for {obj_key}"
                        )
                elif isinstance(dnsh_data, dict):
                    passed = dnsh_data.get("pass", True)
                    assessments[obj_key] = (
                        DNSHStatus.PASS if passed else DNSHStatus.FAIL
                    )
                    if not passed:
                        reason = dnsh_data.get("reason", "Threshold exceeded")
                        issues.append(f"DNSH {obj_key}: {reason}")
                else:
                    assessments[obj_key] = DNSHStatus.PASS

            result.assessments = assessments
            result.issues = issues

            # Overall DNSH passes if no objective has FAIL status
            fail_count = sum(
                1 for s in assessments.values() if s == DNSHStatus.FAIL
            )
            result.overall_dnsh_pass = fail_count == 0

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("DNSH evaluation failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def calculate_taxonomy_kpis(
        self,
        activities: Optional[List[Dict[str, Any]]] = None,
    ) -> TaxonomyKPIResult:
        """Calculate taxonomy-aligned KPIs (turnover, CapEx, OpEx).

        Aggregates alignment results across all economic activities to
        produce company-level taxonomy KPIs for disclosure.

        Args:
            activities: List of activity dicts with financial and alignment data.

        Returns:
            TaxonomyKPIResult with percentage KPIs.
        """
        start = time.monotonic()
        activities = activities or []
        result = TaxonomyKPIResult(reporting_year=self.config.reporting_year)

        try:
            total_turnover = 0.0
            eligible_turnover = 0.0
            aligned_turnover = 0.0
            total_capex = 0.0
            eligible_capex = 0.0
            aligned_capex = 0.0
            total_opex = 0.0
            eligible_opex = 0.0
            aligned_opex = 0.0
            activities_aligned_count = 0
            by_activity: List[Dict[str, Any]] = []

            for activity in activities:
                turnover = activity.get("turnover_eur", 0.0)
                capex = activity.get("capex_eur", 0.0)
                opex = activity.get("opex_eur", 0.0)
                is_eligible = activity.get("taxonomy_eligible", False)
                is_aligned = activity.get("taxonomy_aligned", False)
                nace = activity.get("nace_code", "")

                total_turnover += turnover
                total_capex += capex
                total_opex += opex

                if is_eligible:
                    eligible_turnover += turnover
                    eligible_capex += capex
                    eligible_opex += opex

                if is_aligned:
                    aligned_turnover += turnover
                    aligned_capex += capex
                    aligned_opex += opex
                    activities_aligned_count += 1

                by_activity.append({
                    "nace_code": nace,
                    "activity_name": activity.get("activity_name", ""),
                    "turnover_eur": turnover,
                    "capex_eur": capex,
                    "opex_eur": opex,
                    "eligible": is_eligible,
                    "aligned": is_aligned,
                })

            # Calculate percentages
            if total_turnover > 0:
                result.taxonomy_eligible_turnover_pct = round(
                    (eligible_turnover / total_turnover) * 100.0, 1
                )
                result.taxonomy_aligned_turnover_pct = round(
                    (aligned_turnover / total_turnover) * 100.0, 1
                )
            if total_capex > 0:
                result.taxonomy_eligible_capex_pct = round(
                    (eligible_capex / total_capex) * 100.0, 1
                )
                result.taxonomy_aligned_capex_pct = round(
                    (aligned_capex / total_capex) * 100.0, 1
                )
            if total_opex > 0:
                result.taxonomy_eligible_opex_pct = round(
                    (eligible_opex / total_opex) * 100.0, 1
                )
                result.taxonomy_aligned_opex_pct = round(
                    (aligned_opex / total_opex) * 100.0, 1
                )

            result.activities_assessed = len(activities)
            result.activities_aligned = activities_aligned_count
            result.by_activity = by_activity
            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("Taxonomy KPI calculation failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with component availability information.
        """
        available = sum(
            1 for c in self._components.values() if not isinstance(c, _AgentStub)
        )
        return {
            "pack_id": self.config.pack_id,
            "reporting_year": self.config.reporting_year,
            "primary_objective": self.config.primary_objective.value,
            "total_components": len(self._components),
            "available_components": available,
            "tsc_reference_activities": len(TSC_REFERENCE),
            "taxonomy_objectives_count": len(TaxonomyObjective),
        }
