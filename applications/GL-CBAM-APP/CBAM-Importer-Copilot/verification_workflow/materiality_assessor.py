# -*- coding: utf-8 -*-
"""
MaterialityAssessorEngine - 5% materiality threshold analysis for CBAM.

Assesses whether discrepancies in reported embedded emissions exceed the
5% materiality threshold established by Implementing Regulation (EU)
2023/1773 Article 12(3). When emissions for a specific CN code deviate
by more than 5% from verified values, the declaration must be corrected.

This engine evaluates materiality at per-CN-code and aggregate levels,
provides recommended verification scope, and tracks materiality trends
across reporting years.

Example:
    >>> assessor = MaterialityAssessorEngine()
    >>> result = assessor.assess_materiality(
    ...     installation_id="INST-001",
    ...     reporting_year=2026,
    ...     emissions_data={
    ...         "72011000": {"declared": Decimal("125.5"), "verified": Decimal("132.8")},
    ...     },
    ... )
    >>> print(result.overall_materiality_pct)

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import hashlib
import logging
import threading
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATERIALITY_THRESHOLD_PCT = Decimal("5")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CnCodeMateriality(BaseModel):
    """Materiality assessment for a single CN code."""

    cn_code: str
    declared_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    verified_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    discrepancy_tco2e: Decimal = Field(default=Decimal("0"))
    materiality_pct: Decimal = Field(default=Decimal("0"))
    above_threshold: bool = Field(default=False)
    direction: str = Field(default="none", description="over/under/none")
    corrective_action_required: bool = Field(default=False)
    notes: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


class MaterialityResult(BaseModel):
    """Aggregate materiality assessment for an installation-year."""

    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    installation_id: str
    year: int
    cn_code_results: List[CnCodeMateriality] = Field(default_factory=list)
    total_declared_tco2e: Decimal = Field(default=Decimal("0"))
    total_verified_tco2e: Decimal = Field(default=Decimal("0"))
    overall_discrepancy_tco2e: Decimal = Field(default=Decimal("0"))
    overall_materiality_pct: Decimal = Field(default=Decimal("0"))
    threshold_pct: Decimal = Field(default=MATERIALITY_THRESHOLD_PCT)
    above_threshold: bool = Field(default=False)
    cn_codes_above_threshold: int = Field(default=0)
    recommended_actions: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


class MaterialityTrend(BaseModel):
    """Historical materiality trend for an installation."""

    installation_id: str
    years: List[int] = Field(default_factory=list)
    overall_materiality_by_year: Dict[str, Decimal] = Field(default_factory=dict)
    cn_codes_above_threshold_by_year: Dict[str, int] = Field(default_factory=dict)
    improving: bool = Field(default=False)
    trend_description: str = Field(default="")
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


class VerificationScope(BaseModel):
    """Recommended verification scope for an installation."""

    installation_id: str
    year: int
    high_risk_cn_codes: List[str] = Field(
        default_factory=list,
        description="CN codes requiring detailed verification",
    )
    medium_risk_cn_codes: List[str] = Field(
        default_factory=list,
        description="CN codes requiring standard verification",
    )
    low_risk_cn_codes: List[str] = Field(
        default_factory=list,
        description="CN codes eligible for desk review only",
    )
    recommended_sample_size_pct: Decimal = Field(default=Decimal("100"))
    estimated_verification_hours: Decimal = Field(default=Decimal("0"))
    scope_rationale: str = Field(default="")
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Engine Implementation
# ---------------------------------------------------------------------------

class MaterialityAssessorEngine:
    """
    Assesses materiality of emission discrepancies for CBAM verification.

    The 5% materiality threshold applies to each CN code individually.
    If any CN code exceeds 5%, corrective action is required. The overall
    aggregate materiality is also computed for reporting purposes.

    Thread-safe: all state mutations are protected by an RLock.
    """

    def __init__(self) -> None:
        """Initialise the materiality assessor."""
        self._lock = threading.RLock()
        self._results_history: Dict[str, Dict[int, MaterialityResult]] = {}
        logger.info("MaterialityAssessorEngine initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_materiality(
        self,
        installation_id: str,
        reporting_year: int,
        emissions_data: Optional[Dict[str, Dict[str, Decimal]]] = None,
    ) -> MaterialityResult:
        """
        Assess materiality of emission discrepancies for an installation.

        Args:
            installation_id: Installation identifier.
            reporting_year: CBAM reporting year.
            emissions_data: Dict keyed by CN code, each containing:
                - "declared": Declared emissions in tCO2e.
                - "verified": Verified emissions in tCO2e.
                If None, returns an empty result.

        Returns:
            MaterialityResult with per-CN-code and aggregate analysis.
        """
        with self._lock:
            data = emissions_data or {}
            cn_results: List[CnCodeMateriality] = []
            total_declared = Decimal("0")
            total_verified = Decimal("0")
            codes_above = 0

            for cn_code, values in data.items():
                declared = values.get("declared", Decimal("0"))
                verified = values.get("verified", Decimal("0"))
                cn_result = self._assess_cn_code(cn_code, declared, verified)
                cn_results.append(cn_result)
                total_declared += declared
                total_verified += verified
                if cn_result.above_threshold:
                    codes_above += 1

            # Overall materiality
            overall_discrepancy = abs(total_declared - total_verified)
            overall_pct = Decimal("0")
            if total_verified > 0:
                overall_pct = (
                    overall_discrepancy / total_verified * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            above = overall_pct > MATERIALITY_THRESHOLD_PCT or codes_above > 0
            actions = self._recommend_actions(cn_results, overall_pct, codes_above)

            result = MaterialityResult(
                installation_id=installation_id,
                year=reporting_year,
                cn_code_results=cn_results,
                total_declared_tco2e=total_declared,
                total_verified_tco2e=total_verified,
                overall_discrepancy_tco2e=overall_discrepancy,
                overall_materiality_pct=overall_pct,
                above_threshold=above,
                cn_codes_above_threshold=codes_above,
                recommended_actions=actions,
            )
            result.provenance_hash = self._hash_result(result)

            # Store in history
            if installation_id not in self._results_history:
                self._results_history[installation_id] = {}
            self._results_history[installation_id][reporting_year] = result

            logger.info(
                "Materiality assessed: installation=%s year=%d overall=%s%% above=%s codes_above=%d",
                installation_id, reporting_year, overall_pct, above, codes_above,
            )
            return result

    def calculate_materiality_percentage(
        self, declared: Decimal, verified: Decimal
    ) -> Decimal:
        """
        Calculate materiality percentage for a single emission value pair.

        Formula: |declared - verified| / verified * 100

        Args:
            declared: Declared emissions in tCO2e.
            verified: Verified emissions in tCO2e.

        Returns:
            Materiality percentage, rounded to 2 decimal places.
        """
        if verified <= 0:
            return Decimal("0") if declared <= 0 else Decimal("100")
        discrepancy = abs(declared - verified)
        pct = (discrepancy / verified * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return pct

    def get_recommended_scope(
        self,
        installation_id: str,
        year: int,
        cn_codes: Optional[List[str]] = None,
    ) -> VerificationScope:
        """
        Determine the recommended verification scope for an installation.

        CN codes are classified into risk tiers based on historical materiality
        and emission volume:
            - High risk: Above 5% materiality in any prior year, or first verification.
            - Medium risk: Below 5% but above 2% in prior years.
            - Low risk: Consistently below 2%.

        Args:
            installation_id: Installation identifier.
            year: Upcoming verification year.
            cn_codes: CN codes to evaluate. If None, uses historical data.

        Returns:
            VerificationScope with risk-tiered CN codes and recommendations.
        """
        with self._lock:
            high_risk: List[str] = []
            medium_risk: List[str] = []
            low_risk: List[str] = []

            codes = cn_codes or self._get_historical_cn_codes(installation_id)
            if not codes:
                # First verification: all codes are high risk
                return VerificationScope(
                    installation_id=installation_id,
                    year=year,
                    high_risk_cn_codes=codes or [],
                    recommended_sample_size_pct=Decimal("100"),
                    estimated_verification_hours=Decimal("40"),
                    scope_rationale="First verification year. Full scope recommended.",
                )

            for cn in codes:
                max_materiality = self._get_max_historical_materiality(installation_id, cn)
                if max_materiality > MATERIALITY_THRESHOLD_PCT:
                    high_risk.append(cn)
                elif max_materiality > Decimal("2"):
                    medium_risk.append(cn)
                else:
                    low_risk.append(cn)

            total_codes = len(high_risk) + len(medium_risk) + len(low_risk)
            if total_codes == 0:
                sample_pct = Decimal("100")
            else:
                high_weight = len(high_risk) * Decimal("100")
                med_weight = len(medium_risk) * Decimal("60")
                low_weight = len(low_risk) * Decimal("25")
                sample_pct = (
                    (high_weight + med_weight + low_weight) / Decimal(str(total_codes))
                ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
                sample_pct = min(sample_pct, Decimal("100"))

            # Estimate hours: 8h/high + 4h/medium + 1h/low + 8h overhead
            hours = (
                Decimal(str(len(high_risk))) * Decimal("8")
                + Decimal(str(len(medium_risk))) * Decimal("4")
                + Decimal(str(len(low_risk))) * Decimal("1")
                + Decimal("8")
            )

            rationale_parts = []
            if high_risk:
                rationale_parts.append(
                    f"{len(high_risk)} high-risk CN codes (>5% historical materiality) "
                    "require detailed on-site verification."
                )
            if medium_risk:
                rationale_parts.append(
                    f"{len(medium_risk)} medium-risk CN codes (2-5%) require standard verification."
                )
            if low_risk:
                rationale_parts.append(
                    f"{len(low_risk)} low-risk CN codes (<2%) eligible for desk review."
                )

            scope = VerificationScope(
                installation_id=installation_id,
                year=year,
                high_risk_cn_codes=high_risk,
                medium_risk_cn_codes=medium_risk,
                low_risk_cn_codes=low_risk,
                recommended_sample_size_pct=sample_pct,
                estimated_verification_hours=hours,
                scope_rationale=" ".join(rationale_parts),
            )
            scope.provenance_hash = self._hash_scope(scope)
            return scope

    def flag_threshold_breaches(
        self,
        installation_id: str,
        year: int,
    ) -> List[Dict[str, Any]]:
        """
        Return CN codes that exceed the 5% materiality threshold.

        Args:
            installation_id: Installation identifier.
            year: Reporting year.

        Returns:
            List of dicts with CN code, materiality percentage, and direction.
        """
        with self._lock:
            result = self._results_history.get(installation_id, {}).get(year)
            if not result:
                return []

            breaches: List[Dict[str, Any]] = []
            for cn in result.cn_code_results:
                if cn.above_threshold:
                    breaches.append({
                        "cn_code": cn.cn_code,
                        "materiality_pct": float(cn.materiality_pct),
                        "direction": cn.direction,
                        "declared_tco2e": float(cn.declared_emissions_tco2e),
                        "verified_tco2e": float(cn.verified_emissions_tco2e),
                        "discrepancy_tco2e": float(cn.discrepancy_tco2e),
                        "corrective_action_required": cn.corrective_action_required,
                    })

            return breaches

    def get_materiality_trend(
        self, installation_id: str, years: Optional[List[int]] = None
    ) -> MaterialityTrend:
        """
        Analyse materiality trends across multiple reporting years.

        Args:
            installation_id: Installation identifier.
            years: Years to include. If None, uses all available years.

        Returns:
            MaterialityTrend with year-over-year analysis.
        """
        with self._lock:
            history = self._results_history.get(installation_id, {})
            if years:
                year_list = sorted(y for y in years if y in history)
            else:
                year_list = sorted(history.keys())

            materiality_by_year: Dict[str, Decimal] = {}
            codes_above_by_year: Dict[str, int] = {}

            for yr in year_list:
                result = history[yr]
                materiality_by_year[str(yr)] = result.overall_materiality_pct
                codes_above_by_year[str(yr)] = result.cn_codes_above_threshold

            # Determine trend direction
            improving = False
            trend_desc = "Insufficient data for trend analysis."
            if len(year_list) >= 2:
                pct_values = [materiality_by_year[str(y)] for y in year_list]
                improving = all(
                    pct_values[i] >= pct_values[i + 1]
                    for i in range(len(pct_values) - 1)
                )
                if improving:
                    trend_desc = (
                        "Materiality is improving (declining) year-over-year. "
                        "Emission reporting accuracy is increasing."
                    )
                elif all(
                    pct_values[i] <= pct_values[i + 1]
                    for i in range(len(pct_values) - 1)
                ):
                    trend_desc = (
                        "Materiality is worsening (increasing) year-over-year. "
                        "Review emission data collection and reporting processes."
                    )
                else:
                    trend_desc = (
                        "Materiality fluctuates year-over-year. "
                        "Investigate root causes for inconsistency."
                    )

            trend = MaterialityTrend(
                installation_id=installation_id,
                years=year_list,
                overall_materiality_by_year=materiality_by_year,
                cn_codes_above_threshold_by_year=codes_above_by_year,
                improving=improving,
                trend_description=trend_desc,
            )
            trend.provenance_hash = self._hash_trend(trend)
            return trend

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assess_cn_code(
        self, cn_code: str, declared: Decimal, verified: Decimal
    ) -> CnCodeMateriality:
        """Assess materiality for a single CN code."""
        discrepancy = declared - verified
        abs_discrepancy = abs(discrepancy)
        pct = self.calculate_materiality_percentage(declared, verified)
        above = pct > MATERIALITY_THRESHOLD_PCT

        if discrepancy > 0:
            direction = "over"
        elif discrepancy < 0:
            direction = "under"
        else:
            direction = "none"

        notes = ""
        if above and direction == "under":
            notes = (
                "Under-reporting detected. Declared emissions are significantly "
                "lower than verified values. Corrective action required."
            )
        elif above and direction == "over":
            notes = (
                "Over-reporting detected. While conservative, the discrepancy "
                "exceeds the 5% materiality threshold. Review emission factors."
            )

        return CnCodeMateriality(
            cn_code=cn_code,
            declared_emissions_tco2e=declared,
            verified_emissions_tco2e=verified,
            discrepancy_tco2e=abs_discrepancy,
            materiality_pct=pct,
            above_threshold=above,
            direction=direction,
            corrective_action_required=above,
            notes=notes,
        )

    def _recommend_actions(
        self,
        cn_results: List[CnCodeMateriality],
        overall_pct: Decimal,
        codes_above: int,
    ) -> List[str]:
        """Build recommended actions based on materiality assessment."""
        actions: List[str] = []

        if codes_above == 0 and overall_pct <= MATERIALITY_THRESHOLD_PCT:
            actions.append(
                "All CN codes and overall emissions are within the 5% materiality threshold. "
                "No corrective action required."
            )
            return actions

        if codes_above > 0:
            above_codes = [
                cn.cn_code for cn in cn_results if cn.above_threshold
            ]
            actions.append(
                f"{codes_above} CN code(s) exceed the 5% materiality threshold: "
                f"{', '.join(above_codes)}. Corrective declarations required."
            )

            under_reporting = [
                cn for cn in cn_results if cn.above_threshold and cn.direction == "under"
            ]
            if under_reporting:
                actions.append(
                    "Under-reporting detected in: "
                    + ", ".join(cn.cn_code for cn in under_reporting)
                    + ". Review emission factors and data collection for these products."
                )

            over_reporting = [
                cn for cn in cn_results if cn.above_threshold and cn.direction == "over"
            ]
            if over_reporting:
                actions.append(
                    "Over-reporting detected in: "
                    + ", ".join(cn.cn_code for cn in over_reporting)
                    + ". Verify emission factor sources to avoid overpayment of CBAM certificates."
                )

        if overall_pct > MATERIALITY_THRESHOLD_PCT:
            actions.append(
                f"Overall emissions materiality ({overall_pct}%) exceeds threshold. "
                "Comprehensive review of all emission calculations recommended."
            )

        actions.append(
            "Submit corrective declaration within 30 days per Article 12(3) "
            "of Implementing Regulation (EU) 2023/1773."
        )

        return actions

    def _get_historical_cn_codes(self, installation_id: str) -> List[str]:
        """Collect all CN codes from historical assessments."""
        codes: set = set()
        history = self._results_history.get(installation_id, {})
        for result in history.values():
            for cn in result.cn_code_results:
                codes.add(cn.cn_code)
        return sorted(codes)

    def _get_max_historical_materiality(
        self, installation_id: str, cn_code: str
    ) -> Decimal:
        """Get the maximum historical materiality for a CN code."""
        max_pct = Decimal("0")
        history = self._results_history.get(installation_id, {})
        for result in history.values():
            for cn in result.cn_code_results:
                if cn.cn_code == cn_code and cn.materiality_pct > max_pct:
                    max_pct = cn.materiality_pct
        return max_pct

    # ------------------------------------------------------------------
    # Provenance hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_result(result: MaterialityResult) -> str:
        """Compute SHA-256 provenance hash for MaterialityResult."""
        payload = (
            f"{result.installation_id}|{result.year}|"
            f"{result.total_declared_tco2e}|{result.total_verified_tco2e}|"
            f"{result.overall_materiality_pct}|{result.above_threshold}|"
            f"{result.cn_codes_above_threshold}|{result.assessed_at.isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_scope(scope: VerificationScope) -> str:
        """Compute SHA-256 provenance hash for VerificationScope."""
        payload = (
            f"{scope.installation_id}|{scope.year}|"
            f"{scope.high_risk_cn_codes}|{scope.medium_risk_cn_codes}|"
            f"{scope.low_risk_cn_codes}|{scope.recommended_sample_size_pct}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_trend(trend: MaterialityTrend) -> str:
        """Compute SHA-256 provenance hash for MaterialityTrend."""
        payload = (
            f"{trend.installation_id}|{trend.years}|"
            f"{trend.overall_materiality_by_year}|{trend.improving}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
