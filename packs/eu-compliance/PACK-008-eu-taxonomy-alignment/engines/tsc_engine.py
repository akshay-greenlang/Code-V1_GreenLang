# -*- coding: utf-8 -*-
"""
Technical Screening Criteria Engine - PACK-008 EU Taxonomy Alignment

This module implements the Technical Screening Criteria (TSC) lookup and
evaluation engine. It holds a representative criteria database covering 10+
economic activities across multiple environmental objectives, supports
quantitative threshold checking and qualitative assessment, manages Delegated
Act versioning (Climate DA 2021/2139, Environmental DA 2023/2486), and
identifies gaps for non-compliant criteria.

All threshold comparisons are deterministic -- no LLM calls are used for
numeric evaluations.

Example:
    >>> engine = TechnicalScreeningCriteriaEngine()
    >>> criteria = engine.get_criteria("4.1", EnvironmentalObjective.CCM)
    >>> evaluation = engine.evaluate_criteria("4.1", EnvironmentalObjective.CCM, {"gco2e_kwh": 80})
    >>> print(evaluation.overall_pass)
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""

    CCM = "CCM"   # Climate Change Mitigation
    CCA = "CCA"   # Climate Change Adaptation
    WTR = "WTR"   # Water and Marine Resources
    CE = "CE"     # Circular Economy
    PPC = "PPC"   # Pollution Prevention and Control
    BIO = "BIO"   # Biodiversity and Ecosystems


class CriterionType(str, Enum):
    """Type of technical screening criterion."""

    QUANTITATIVE = "QUANTITATIVE"
    QUALITATIVE = "QUALITATIVE"


class ComparisonOperator(str, Enum):
    """Comparison operators for quantitative thresholds."""

    LESS_THAN = "LT"
    LESS_EQUAL = "LE"
    GREATER_THAN = "GT"
    GREATER_EQUAL = "GE"
    EQUAL = "EQ"
    BETWEEN = "BETWEEN"


class DelegatedActId(str, Enum):
    """Identifiers for EU Taxonomy Delegated Acts."""

    CLIMATE_DA_2021 = "EU_2021_2139"
    ENVIRONMENTAL_DA_2023 = "EU_2023_2486"
    COMPLEMENTARY_DA_2022 = "EU_2022_1214"
    DISCLOSURES_DA_2021 = "EU_2021_2178"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class DelegatedActVersion(BaseModel):
    """Delegated Act version metadata."""

    da_id: DelegatedActId = Field(..., description="Delegated Act identifier")
    version: str = Field(..., description="Version string (e.g. '2021-06-04')")
    publication_date: str = Field(..., description="Official Journal publication date")
    effective_date: str = Field(..., description="Date criteria become effective")
    description: str = Field(..., description="Short description")


class TSCCriteria(BaseModel):
    """A single Technical Screening Criterion."""

    criterion_id: str = Field(..., description="Unique criterion identifier")
    activity_id: str = Field(..., description="Economic activity NACE-based id")
    activity_name: str = Field(..., description="Human-readable activity name")
    objective: EnvironmentalObjective = Field(..., description="Environmental objective")
    criterion_type: CriterionType = Field(..., description="Quantitative or qualitative")
    description: str = Field(..., description="Criterion text")
    metric: Optional[str] = Field(None, description="Metric key expected in input data")
    unit: Optional[str] = Field(None, description="Unit of measure")
    operator: Optional[ComparisonOperator] = Field(None, description="Comparison operator")
    threshold: Optional[float] = Field(None, description="Threshold value (single bound)")
    threshold_upper: Optional[float] = Field(None, description="Upper bound for BETWEEN")
    da_id: DelegatedActId = Field(..., description="Source Delegated Act")
    da_article: Optional[str] = Field(None, description="Article / Annex reference")
    effective_from: Optional[str] = Field(None, description="Effective date for this criterion")
    sunset_date: Optional[str] = Field(None, description="Date criterion expires / changes")


class CriterionEvaluation(BaseModel):
    """Evaluation result for a single criterion."""

    criterion_id: str = Field(..., description="Criterion identifier")
    passed: bool = Field(..., description="Whether the criterion is met")
    actual_value: Optional[float] = Field(None, description="Actual value from data")
    threshold_value: Optional[float] = Field(None, description="Required threshold")
    unit: Optional[str] = Field(None, description="Unit of measure")
    gap: Optional[float] = Field(None, description="Gap to threshold (negative = exceeded)")
    message: str = Field(..., description="Human-readable result message")


class TSCEvaluation(BaseModel):
    """Aggregated evaluation of all TSC for an activity + objective."""

    activity_id: str = Field(..., description="Economic activity id")
    objective: EnvironmentalObjective = Field(..., description="Environmental objective")
    overall_pass: bool = Field(..., description="True if ALL criteria are met")
    criteria_results: List[CriterionEvaluation] = Field(
        ..., description="Individual criterion results"
    )
    total_criteria: int = Field(..., description="Total criteria evaluated")
    passed_criteria: int = Field(..., description="Number of criteria passed")
    gaps: List[CriterionEvaluation] = Field(
        default_factory=list,
        description="Non-compliant criteria with gap details"
    )
    da_version: Optional[DelegatedActVersion] = Field(
        None, description="Delegated Act version used"
    )
    evaluation_date: str = Field(..., description="Evaluation timestamp (ISO 8601)")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


# ---------------------------------------------------------------------------
# TSC Database (representative criteria for 10+ activities)
# ---------------------------------------------------------------------------

# Each key is activity_id; value is a list of TSCCriteria dicts.
# This is the authoritative in-engine criteria store.

TSC_DATABASE: Dict[str, List[Dict[str, Any]]] = {
    # ------------------------------------------------------------------
    # 4.1 Electricity generation using solar PV / wind / hydro / etc.
    # ------------------------------------------------------------------
    "4.1": [
        {
            "criterion_id": "TSC-4.1-CCM-01",
            "activity_name": "Electricity generation (low-carbon)",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Life-cycle GHG emissions below 100 gCO2e/kWh",
            "metric": "gco2e_kwh",
            "unit": "gCO2e/kWh",
            "operator": "LT",
            "threshold": 100.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 4.1",
        },
    ],
    # ------------------------------------------------------------------
    # 4.3 Electricity generation from wind power
    # ------------------------------------------------------------------
    "4.3": [
        {
            "criterion_id": "TSC-4.3-CCM-01",
            "activity_name": "Electricity generation from wind power",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Life-cycle GHG emissions below 100 gCO2e/kWh",
            "metric": "gco2e_kwh",
            "unit": "gCO2e/kWh",
            "operator": "LT",
            "threshold": 100.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 4.3",
        },
    ],
    # ------------------------------------------------------------------
    # 4.29 Electricity generation from fossil gaseous fuels (transition)
    # ------------------------------------------------------------------
    "4.29": [
        {
            "criterion_id": "TSC-4.29-CCM-01",
            "activity_name": "Electricity generation from fossil gaseous fuels",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Life-cycle emissions < 270 gCO2e/kWh or annual avg < 550 kgCO2e/kW",
            "metric": "gco2e_kwh",
            "unit": "gCO2e/kWh",
            "operator": "LT",
            "threshold": 270.0,
            "da_id": "EU_2022_1214",
            "da_article": "Annex I, Section 4.29",
        },
        {
            "criterion_id": "TSC-4.29-CCM-02",
            "activity_name": "Electricity generation from fossil gaseous fuels",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Annual direct emissions below 550 kgCO2e/kW of installed capacity",
            "metric": "kgco2e_kw_annual",
            "unit": "kgCO2e/kW/year",
            "operator": "LT",
            "threshold": 550.0,
            "da_id": "EU_2022_1214",
            "da_article": "Annex I, Section 4.29",
        },
    ],
    # ------------------------------------------------------------------
    # 7.1 Construction of new buildings
    # ------------------------------------------------------------------
    "7.1": [
        {
            "criterion_id": "TSC-7.1-CCM-01",
            "activity_name": "Construction of new buildings",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Primary energy demand at least 10% below NZEB threshold",
            "metric": "ped_vs_nzeb_pct",
            "unit": "%",
            "operator": "GE",
            "threshold": 10.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 7.1",
        },
        {
            "criterion_id": "TSC-7.1-CCM-02",
            "activity_name": "Construction of new buildings",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Building airtightness and thermal integrity tested",
            "metric": "airtightness_tested",
            "unit": "boolean",
            "operator": "EQ",
            "threshold": 1.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 7.1",
        },
    ],
    # ------------------------------------------------------------------
    # 7.2 Renovation of existing buildings
    # ------------------------------------------------------------------
    "7.2": [
        {
            "criterion_id": "TSC-7.2-CCM-01",
            "activity_name": "Renovation of existing buildings",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "At least 30% reduction in primary energy demand",
            "metric": "energy_demand_reduction_pct",
            "unit": "%",
            "operator": "GE",
            "threshold": 30.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 7.2",
        },
    ],
    # ------------------------------------------------------------------
    # 6.5 Transport by motorbikes, passenger cars and light commercial
    # ------------------------------------------------------------------
    "6.5": [
        {
            "criterion_id": "TSC-6.5-CCM-01",
            "activity_name": "Transport by motorbikes, passenger cars and LCVs",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Zero direct (tailpipe) CO2 emissions",
            "metric": "tailpipe_co2_gkm",
            "unit": "gCO2/km",
            "operator": "EQ",
            "threshold": 0.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 6.5",
        },
    ],
    # ------------------------------------------------------------------
    # 3.7 Manufacture of cement
    # ------------------------------------------------------------------
    "3.7": [
        {
            "criterion_id": "TSC-3.7-CCM-01",
            "activity_name": "Manufacture of cement",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Specific emissions below 0.469 tCO2e per tonne of clinker",
            "metric": "tco2e_per_t_clinker",
            "unit": "tCO2e/t clinker",
            "operator": "LT",
            "threshold": 0.469,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 3.7",
        },
    ],
    # ------------------------------------------------------------------
    # 3.9 Manufacture of iron and steel
    # ------------------------------------------------------------------
    "3.9": [
        {
            "criterion_id": "TSC-3.9-CCM-01",
            "activity_name": "Manufacture of iron and steel",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "GHG emissions below 1.331 tCO2e per tonne of steel",
            "metric": "tco2e_per_t_steel",
            "unit": "tCO2e/t",
            "operator": "LT",
            "threshold": 1.331,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 3.9",
        },
    ],
    # ------------------------------------------------------------------
    # 3.8 Manufacture of aluminium
    # ------------------------------------------------------------------
    "3.8": [
        {
            "criterion_id": "TSC-3.8-CCM-01",
            "activity_name": "Manufacture of aluminium",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "GHG emissions below 1.484 tCO2e per tonne of aluminium",
            "metric": "tco2e_per_t_aluminium",
            "unit": "tCO2e/t",
            "operator": "LT",
            "threshold": 1.484,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 3.8",
        },
    ],
    # ------------------------------------------------------------------
    # 8.1 Data processing, hosting and related activities
    # ------------------------------------------------------------------
    "8.1": [
        {
            "criterion_id": "TSC-8.1-CCM-01",
            "activity_name": "Data processing, hosting and related activities",
            "objective": "CCM",
            "criterion_type": "QUANTITATIVE",
            "description": "Power Usage Effectiveness (PUE) below 1.5",
            "metric": "pue",
            "unit": "ratio",
            "operator": "LT",
            "threshold": 1.5,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 8.1",
        },
        {
            "criterion_id": "TSC-8.1-CCM-02",
            "activity_name": "Data processing, hosting and related activities",
            "objective": "CCM",
            "criterion_type": "QUALITATIVE",
            "description": "European Code of Conduct for Data Centre Energy Efficiency implemented",
            "metric": "eu_code_of_conduct",
            "unit": "boolean",
            "operator": "EQ",
            "threshold": 1.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 8.1",
        },
    ],
    # ------------------------------------------------------------------
    # 5.1 Construction, extension and operation of water systems
    # ------------------------------------------------------------------
    "5.1": [
        {
            "criterion_id": "TSC-5.1-WTR-01",
            "activity_name": "Water collection, treatment and supply",
            "objective": "WTR",
            "criterion_type": "QUANTITATIVE",
            "description": "Network leakage rate (ILI) at or below national average or ILI <= 1.5",
            "metric": "infrastructure_leakage_index",
            "unit": "ILI ratio",
            "operator": "LE",
            "threshold": 1.5,
            "da_id": "EU_2023_2486",
            "da_article": "Annex II, Section 5.1",
        },
        {
            "criterion_id": "TSC-5.1-WTR-02",
            "activity_name": "Water collection, treatment and supply",
            "objective": "WTR",
            "criterion_type": "QUANTITATIVE",
            "description": "Net energy consumption for water system <= 0.5 kWh/m3",
            "metric": "kwh_per_m3",
            "unit": "kWh/m3",
            "operator": "LE",
            "threshold": 0.5,
            "da_id": "EU_2023_2486",
            "da_article": "Annex II, Section 5.1",
        },
    ],
    # ------------------------------------------------------------------
    # 5.3 Construction, extension and operation of wastewater systems
    # ------------------------------------------------------------------
    "5.3": [
        {
            "criterion_id": "TSC-5.3-WTR-01",
            "activity_name": "Wastewater collection and treatment",
            "objective": "WTR",
            "criterion_type": "QUANTITATIVE",
            "description": "Energy consumption for wastewater treatment <= 35 kWh/p.e./year",
            "metric": "kwh_pe_year",
            "unit": "kWh/p.e./year",
            "operator": "LE",
            "threshold": 35.0,
            "da_id": "EU_2023_2486",
            "da_article": "Annex II, Section 5.3",
        },
    ],
    # ------------------------------------------------------------------
    # 1.1 Afforestation
    # ------------------------------------------------------------------
    "1.1": [
        {
            "criterion_id": "TSC-1.1-CCM-01",
            "activity_name": "Afforestation",
            "objective": "CCM",
            "criterion_type": "QUALITATIVE",
            "description": "Forest management plan with carbon benefits analysis",
            "metric": "forest_management_plan",
            "unit": "boolean",
            "operator": "EQ",
            "threshold": 1.0,
            "da_id": "EU_2021_2139",
            "da_article": "Annex I, Section 1.1",
        },
    ],
}

# Delegated Act version registry
_DA_VERSIONS: Dict[str, DelegatedActVersion] = {
    "EU_2021_2139": DelegatedActVersion(
        da_id=DelegatedActId.CLIMATE_DA_2021,
        version="2021-06-04",
        publication_date="2021-12-09",
        effective_date="2022-01-01",
        description="Climate Delegated Act - CCM and CCA criteria",
    ),
    "EU_2023_2486": DelegatedActVersion(
        da_id=DelegatedActId.ENVIRONMENTAL_DA_2023,
        version="2023-06-27",
        publication_date="2023-11-21",
        effective_date="2024-01-01",
        description="Environmental Delegated Act - WTR, CE, PPC, BIO criteria",
    ),
    "EU_2022_1214": DelegatedActVersion(
        da_id=DelegatedActId.COMPLEMENTARY_DA_2022,
        version="2022-03-09",
        publication_date="2022-07-15",
        effective_date="2023-01-01",
        description="Complementary Climate DA - nuclear and gas activities",
    ),
    "EU_2021_2178": DelegatedActVersion(
        da_id=DelegatedActId.DISCLOSURES_DA_2021,
        version="2021-07-06",
        publication_date="2021-12-10",
        effective_date="2022-01-01",
        description="Disclosures DA - Article 8 reporting templates",
    ),
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TechnicalScreeningCriteriaEngine:
    """
    Technical Screening Criteria Engine for EU Taxonomy alignment assessment.

    Provides criteria lookup per activity and environmental objective, quantitative
    threshold evaluation, qualitative assessment, Delegated Act version management,
    and gap identification for non-compliant criteria.

    Attributes:
        criteria_db: In-memory criteria database keyed by activity_id
        da_versions: Delegated Act version registry

    Example:
        >>> engine = TechnicalScreeningCriteriaEngine()
        >>> criteria = engine.get_criteria("4.1", EnvironmentalObjective.CCM)
        >>> assert len(criteria) >= 1
    """

    def __init__(self) -> None:
        """Initialize the Technical Screening Criteria Engine."""
        self.criteria_db = self._load_criteria_database()
        self.da_versions = _DA_VERSIONS
        logger.info(
            f"TechnicalScreeningCriteriaEngine initialized with "
            f"{sum(len(v) for v in self.criteria_db.values())} criteria "
            f"across {len(self.criteria_db)} activities"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_criteria(
        self,
        activity_id: str,
        objective: EnvironmentalObjective,
    ) -> List[TSCCriteria]:
        """
        Retrieve Technical Screening Criteria for an activity and objective.

        Args:
            activity_id: Economic activity identifier (e.g. '4.1')
            objective: Environmental objective to filter by

        Returns:
            List of matching TSCCriteria objects
        """
        raw_list = self.criteria_db.get(activity_id, [])
        results = [c for c in raw_list if c.objective == objective]
        logger.info(
            f"get_criteria({activity_id}, {objective.value}): "
            f"found {len(results)} criteria"
        )
        return results

    def evaluate_criteria(
        self,
        activity_id: str,
        objective: EnvironmentalObjective,
        data: Dict[str, Any],
    ) -> TSCEvaluation:
        """
        Evaluate all TSC for a given activity + objective against provided data.

        Args:
            activity_id: Economic activity identifier
            objective: Environmental objective
            data: Dictionary of metric values (key = metric name, value = number)

        Returns:
            TSCEvaluation with per-criterion pass/fail and gap details

        Raises:
            ValueError: If activity_id has no criteria for the objective
        """
        start = datetime.utcnow()
        criteria = self.get_criteria(activity_id, objective)

        if not criteria:
            raise ValueError(
                f"No TSC found for activity {activity_id} / {objective.value}"
            )

        results: List[CriterionEvaluation] = []
        gaps: List[CriterionEvaluation] = []

        for criterion in criteria:
            evaluation = self._evaluate_single(criterion, data)
            results.append(evaluation)
            if not evaluation.passed:
                gaps.append(evaluation)

        passed_count = sum(1 for r in results if r.passed)
        overall_pass = passed_count == len(results)

        # Resolve DA version
        da_key = criteria[0].da_id.value if criteria else None
        da_version = self.da_versions.get(da_key) if da_key else None

        provenance = self._provenance({
            "activity_id": activity_id,
            "objective": objective.value,
            "data_keys": sorted(data.keys()),
            "ts": start.isoformat(),
        })

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            f"TSC evaluation for {activity_id}/{objective.value}: "
            f"{passed_count}/{len(results)} passed, overall={'PASS' if overall_pass else 'FAIL'} "
            f"in {elapsed_ms:.1f}ms"
        )

        return TSCEvaluation(
            activity_id=activity_id,
            objective=objective,
            overall_pass=overall_pass,
            criteria_results=results,
            total_criteria=len(results),
            passed_criteria=passed_count,
            gaps=gaps,
            da_version=da_version,
            evaluation_date=start.isoformat(),
            provenance_hash=provenance,
        )

    def get_da_version(self, activity_id: str) -> Optional[DelegatedActVersion]:
        """
        Determine the applicable Delegated Act version for an activity.

        Args:
            activity_id: Economic activity identifier

        Returns:
            DelegatedActVersion or None if activity is unknown
        """
        criteria_list = self.criteria_db.get(activity_id, [])
        if not criteria_list:
            logger.warning(f"No criteria found for activity {activity_id}")
            return None

        da_key = criteria_list[0].da_id.value
        return self.da_versions.get(da_key)

    def get_all_activities(self) -> List[str]:
        """
        Return all activity IDs present in the criteria database.

        Returns:
            Sorted list of activity identifiers
        """
        return sorted(self.criteria_db.keys())

    def get_criteria_changes(
        self,
        activity_id: str,
        from_da: DelegatedActId,
        to_da: DelegatedActId,
    ) -> Dict[str, Any]:
        """
        Track criteria changes between two Delegated Act versions.

        Args:
            activity_id: Activity to check
            from_da: Source DA version
            to_da: Target DA version

        Returns:
            Dictionary with added, removed, and modified criteria summaries
        """
        all_criteria = self.criteria_db.get(activity_id, [])
        from_criteria = [c for c in all_criteria if c.da_id == from_da]
        to_criteria = [c for c in all_criteria if c.da_id == to_da]

        from_ids = {c.criterion_id for c in from_criteria}
        to_ids = {c.criterion_id for c in to_criteria}

        added = to_ids - from_ids
        removed = from_ids - to_ids
        common = from_ids & to_ids

        modified: List[str] = []
        for cid in common:
            c_from = next(c for c in from_criteria if c.criterion_id == cid)
            c_to = next(c for c in to_criteria if c.criterion_id == cid)
            if c_from.threshold != c_to.threshold:
                modified.append(cid)

        logger.info(
            f"Criteria changes for {activity_id}: "
            f"added={len(added)}, removed={len(removed)}, modified={len(modified)}"
        )

        return {
            "activity_id": activity_id,
            "from_da": from_da.value,
            "to_da": to_da.value,
            "added": sorted(added),
            "removed": sorted(removed),
            "modified": sorted(modified),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_criteria_database(self) -> Dict[str, List[TSCCriteria]]:
        """Parse TSC_DATABASE raw dicts into validated TSCCriteria objects."""
        db: Dict[str, List[TSCCriteria]] = {}
        for activity_id, raw_list in TSC_DATABASE.items():
            parsed: List[TSCCriteria] = []
            for raw in raw_list:
                parsed.append(TSCCriteria(
                    criterion_id=raw["criterion_id"],
                    activity_id=activity_id,
                    activity_name=raw["activity_name"],
                    objective=EnvironmentalObjective(raw["objective"]),
                    criterion_type=CriterionType(raw["criterion_type"]),
                    description=raw["description"],
                    metric=raw.get("metric"),
                    unit=raw.get("unit"),
                    operator=(
                        ComparisonOperator(raw["operator"])
                        if raw.get("operator") else None
                    ),
                    threshold=raw.get("threshold"),
                    threshold_upper=raw.get("threshold_upper"),
                    da_id=DelegatedActId(raw["da_id"]),
                    da_article=raw.get("da_article"),
                    effective_from=raw.get("effective_from"),
                    sunset_date=raw.get("sunset_date"),
                ))
            db[activity_id] = parsed
        return db

    def _evaluate_single(
        self, criterion: TSCCriteria, data: Dict[str, Any]
    ) -> CriterionEvaluation:
        """Evaluate a single criterion against provided data."""
        metric_key = criterion.metric
        if not metric_key or metric_key not in data:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                passed=False,
                actual_value=None,
                threshold_value=criterion.threshold,
                unit=criterion.unit,
                gap=None,
                message=f"Missing data for metric '{metric_key}'",
            )

        actual = float(data[metric_key])
        threshold = criterion.threshold
        passed = False
        gap: Optional[float] = None

        if criterion.operator == ComparisonOperator.LESS_THAN:
            passed = actual < threshold
            gap = actual - threshold
        elif criterion.operator == ComparisonOperator.LESS_EQUAL:
            passed = actual <= threshold
            gap = actual - threshold
        elif criterion.operator == ComparisonOperator.GREATER_THAN:
            passed = actual > threshold
            gap = threshold - actual
        elif criterion.operator == ComparisonOperator.GREATER_EQUAL:
            passed = actual >= threshold
            gap = threshold - actual
        elif criterion.operator == ComparisonOperator.EQUAL:
            passed = abs(actual - threshold) < 1e-9
            gap = actual - threshold
        elif criterion.operator == ComparisonOperator.BETWEEN:
            upper = criterion.threshold_upper or threshold
            passed = threshold <= actual <= upper
            if actual < threshold:
                gap = actual - threshold
            elif actual > upper:
                gap = actual - upper
            else:
                gap = 0.0

        status = "PASS" if passed else "FAIL"
        msg = (
            f"{criterion.description}: actual={actual} {criterion.unit or ''} "
            f"vs threshold {criterion.operator.value if criterion.operator else '?'} "
            f"{threshold} -> {status}"
        )

        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            passed=passed,
            actual_value=actual,
            threshold_value=threshold,
            unit=criterion.unit,
            gap=gap if not passed else None,
            message=msg,
        )

    @staticmethod
    def _provenance(data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
