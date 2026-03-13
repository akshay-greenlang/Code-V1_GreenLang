# -*- coding: utf-8 -*-
"""
Year Comparator Engine - AGENT-EUDR-034

Multi-year data comparison engine for identifying trends, regressions,
and areas of improvement in EUDR compliance metrics across annual
review periods.

Zero-Hallucination:
    - All change calculations are deterministic Decimal arithmetic
    - Significance classification uses configured thresholds only
    - Weighted scoring uses explicit, auditable formula
    - No LLM involvement in numeric comparisons

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (GL-EUDR-ARS-034)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional

from .config import AnnualReviewSchedulerConfig, get_config
from .models import (
    AGENT_ID,
    ActionRecommendation,
    ChangeDirection,
    ChangeSignificance,
    ComparisonDimension,
    ComparisonMetric,
    EUDRCommodity,
    YearComparison as YearComparisonModel,
    YearComparisonRecord,
    YearComparisonStatus,
    YearDataPoint,
    YearDimensionComparison,
    YearMetricSnapshot,
)
from .provenance import ProvenanceTracker
from . import metrics as m

# Re-export for backward compat within this module
YearComparison = YearDimensionComparison

logger = logging.getLogger(__name__)

# Dimensions for comparison
_COMPARISON_DIMENSIONS = [
    "supplier_count",
    "risk_score",
    "compliance_score",
    "deforestation_alerts",
    "due_diligence_statements",
    "total_volume_tonnes",
    "high_risk_suppliers",
]


class YearComparator:
    """Multi-year data comparison and trend analysis engine.

    Compares key EUDR compliance metrics across annual review periods,
    calculates percentage changes, classifies significance, and generates
    trend-based recommendations.

    Example:
        >>> comparator = YearComparator()
        >>> record = await comparator.compare_years(
        ...     operator_id="OP-001",
        ...     data_points=[
        ...         {"year": 2025, "supplier_count": 100, "risk_score": 45},
        ...         {"year": 2026, "supplier_count": 120, "risk_score": 38},
        ...     ],
        ... )
        >>> assert record.overall_trend in ("improved", "stable", "degraded")
    """

    def __init__(
        self,
        config: Optional[AnnualReviewSchedulerConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize YearComparator engine."""
        self.config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        self._comparison_records: Dict[str, YearComparisonRecord] = {}
        self._snapshots: Dict[str, YearMetricSnapshot] = {}
        self._comparisons: Dict[str, YearComparisonModel] = {}
        logger.info("YearComparator engine initialized")

    async def detect_changes(
        self,
        operator_id: str,
        year_a_data: Dict[str, Any],
        year_b_data: Dict[str, Any],
    ) -> List[YearComparison]:
        """Detect changes between two specific years.

        Args:
            operator_id: Operator identifier.
            year_a_data: Earlier year data.
            year_b_data: Later year data.

        Returns:
            List of YearComparison results.
        """
        point_a = self._parse_single_point(year_a_data)
        point_b = self._parse_single_point(year_b_data)
        return self._compare_pair(point_a, point_b)

    async def classify_significance(
        self,
        change_percent: Decimal,
    ) -> ChangeSignificance:
        """Classify the significance of a change percentage.

        Args:
            change_percent: Absolute percentage change.

        Returns:
            ChangeSignificance enum value.
        """
        return self._classify_change_significance(change_percent)

    async def generate_comparison_report(
        self,
        comparison_id: str,
    ) -> Dict[str, Any]:
        """Generate a detailed comparison report.

        Args:
            comparison_id: Comparison record identifier.

        Returns:
            Report dictionary with detailed analysis.
        """
        start_time = time.monotonic()
        record = self._get_record(comparison_id)

        report: Dict[str, Any] = {
            "comparison_id": comparison_id,
            "operator_id": record.operator_id,
            "years_compared": record.years_compared,
            "overall_trend": record.overall_trend.value,
            "overall_significance": record.overall_significance.value,
            "weighted_change_score": str(record.weighted_change_score),
            "critical_changes": record.critical_changes,
            "dimensions": {},
            "recommendations": [
                {"action": r.action, "priority": r.priority}
                for r in record.recommendations
            ],
        }

        # Group comparisons by dimension
        for comp in record.comparisons:
            dim = comp.dimension
            if dim not in report["dimensions"]:
                report["dimensions"][dim] = []
            report["dimensions"][dim].append({
                "year_a": comp.year_a,
                "year_b": comp.year_b,
                "value_a": str(comp.value_a),
                "value_b": str(comp.value_b),
                "absolute_change": str(comp.absolute_change),
                "percent_change": str(comp.percent_change),
                "significance": comp.significance.value,
                "direction": comp.direction.value,
            })

        elapsed = time.monotonic() - start_time
        m.observe_comparison_report_duration(elapsed)

        return report

    async def get_record(
        self, comparison_id: str,
    ) -> Optional[YearComparisonRecord]:
        """Get a specific comparison record by ID."""
        return self._comparison_records.get(comparison_id)

    async def list_records(
        self,
        operator_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[YearComparisonRecord]:
        """List comparison records with optional filters."""
        results = list(self._comparison_records.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        results.sort(key=lambda r: r.compared_at, reverse=True)
        return results[offset: offset + limit]

    # -- Snapshot-based API (used by engine tests) --

    async def register_snapshot(
        self, snapshot: YearMetricSnapshot,
    ) -> bool:
        """Register a year metric snapshot.

        Args:
            snapshot: YearMetricSnapshot instance.

        Returns:
            True on success.
        """
        key = f"{snapshot.operator_id}:{snapshot.year}:{snapshot.commodity.value}"
        self._snapshots[key] = snapshot
        return True

    async def get_snapshot(
        self,
        operator_id: str,
        year: int,
        commodity: EUDRCommodity,
    ) -> YearMetricSnapshot:
        """Get a snapshot by operator, year, and commodity.

        Raises:
            ValueError: If snapshot not found.
        """
        key = f"{operator_id}:{year}:{commodity.value}"
        snapshot = self._snapshots.get(key)
        if snapshot is None:
            raise ValueError(
                f"Snapshot not found for operator={operator_id}, year={year}, "
                f"commodity={commodity.value}"
            )
        return snapshot

    async def list_snapshots(
        self,
        operator_id: str,
        commodity: Optional[EUDRCommodity] = None,
    ) -> List[YearMetricSnapshot]:
        """List snapshots for an operator, optionally filtered by commodity."""
        results = [
            s for s in self._snapshots.values()
            if s.operator_id == operator_id
        ]
        if commodity is not None:
            results = [s for s in results if s.commodity == commodity]
        results.sort(key=lambda s: s.year)
        return results

    async def compare_years(  # type: ignore[override]
        self,
        operator_id: str = "",
        commodity: Optional[EUDRCommodity] = None,
        base_year: int = 0,
        compare_year: int = 0,
        *,
        data_points: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Compare years using either snapshot-based or data-point-based API.

        When commodity, base_year, and compare_year are provided, uses the
        snapshot-based comparison API expected by engine tests.
        When data_points is provided, falls back to the legacy API.
        """
        # Legacy path: data_points list provided
        if data_points is not None:
            return await self._compare_years_legacy(operator_id, data_points)

        # Snapshot-based path
        if base_year == compare_year:
            raise ValueError("Cannot compare the same year to itself")

        base_snap = await self.get_snapshot(operator_id, base_year, commodity)
        compare_snap = await self.get_snapshot(operator_id, compare_year, commodity)

        comparison_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Build comparison metrics
        metrics: List[ComparisonMetric] = []

        # compliance_rate
        cr_change = compare_snap.compliance_rate - base_snap.compliance_rate
        cr_pct = self._safe_percent_change(base_snap.compliance_rate, compare_snap.compliance_rate)
        metrics.append(ComparisonMetric(
            dimension=ComparisonDimension.COMPLIANCE_RATE,
            base_value=base_snap.compliance_rate,
            compare_value=compare_snap.compliance_rate,
            change=cr_change,
            percentage_change=cr_pct,
        ))

        # risk_score
        rs_change = compare_snap.average_risk_score - base_snap.average_risk_score
        rs_pct = self._safe_percent_change(base_snap.average_risk_score, compare_snap.average_risk_score)
        metrics.append(ComparisonMetric(
            dimension=ComparisonDimension.RISK_SCORE,
            base_value=base_snap.average_risk_score,
            compare_value=compare_snap.average_risk_score,
            change=rs_change,
            percentage_change=rs_pct,
        ))

        # supplier_count
        sc_change = Decimal(str(compare_snap.total_suppliers)) - Decimal(str(base_snap.total_suppliers))
        sc_pct = self._safe_percent_change(
            Decimal(str(base_snap.total_suppliers)),
            Decimal(str(compare_snap.total_suppliers)),
        )
        metrics.append(ComparisonMetric(
            dimension=ComparisonDimension.SUPPLIER_COUNT,
            base_value=Decimal(str(base_snap.total_suppliers)),
            compare_value=Decimal(str(compare_snap.total_suppliers)),
            change=sc_change,
            percentage_change=sc_pct,
        ))

        # deforestation_rate
        dr_change = compare_snap.deforestation_free_rate - base_snap.deforestation_free_rate
        dr_pct = self._safe_percent_change(base_snap.deforestation_free_rate, compare_snap.deforestation_free_rate)
        metrics.append(ComparisonMetric(
            dimension=ComparisonDimension.DEFORESTATION_RATE,
            base_value=base_snap.deforestation_free_rate,
            compare_value=compare_snap.deforestation_free_rate,
            change=dr_change,
            percentage_change=dr_pct,
        ))

        # audit_findings
        af_change = Decimal(str(compare_snap.audit_findings)) - Decimal(str(base_snap.audit_findings))
        af_pct = self._safe_percent_change(
            Decimal(str(base_snap.audit_findings)),
            Decimal(str(compare_snap.audit_findings)),
        )
        metrics.append(ComparisonMetric(
            dimension=ComparisonDimension.AUDIT_FINDINGS,
            base_value=Decimal(str(base_snap.audit_findings)),
            compare_value=Decimal(str(compare_snap.audit_findings)),
            change=af_change,
            percentage_change=af_pct,
        ))

        # dds_approval_rate
        base_dds_rate = (
            (Decimal(str(base_snap.dds_approved)) / Decimal(str(base_snap.dds_submitted)) * Decimal("100"))
            if base_snap.dds_submitted > 0 else Decimal("0")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        compare_dds_rate = (
            (Decimal(str(compare_snap.dds_approved)) / Decimal(str(compare_snap.dds_submitted)) * Decimal("100"))
            if compare_snap.dds_submitted > 0 else Decimal("0")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        dds_change = compare_dds_rate - base_dds_rate
        dds_pct = self._safe_percent_change(base_dds_rate, compare_dds_rate)
        metrics.append(ComparisonMetric(
            dimension=ComparisonDimension.DDS_APPROVAL_RATE,
            base_value=base_dds_rate,
            compare_value=compare_dds_rate,
            change=dds_change,
            percentage_change=dds_pct,
        ))

        # Classify overall trend
        improving_count = 0
        degrading_count = 0
        # lower-is-better dimensions
        lower_is_better = {ComparisonDimension.RISK_SCORE, ComparisonDimension.AUDIT_FINDINGS}
        for met in metrics:
            if met.change == Decimal("0"):
                continue
            if met.dimension in lower_is_better:
                if met.change < Decimal("0"):
                    improving_count += 1
                else:
                    degrading_count += 1
            else:
                if met.change > Decimal("0"):
                    improving_count += 1
                else:
                    degrading_count += 1

        if improving_count > degrading_count:
            overall_trend = "improving"
        elif degrading_count > improving_count:
            overall_trend = "declining"
        else:
            overall_trend = "stable"

        # Provenance - deterministic: same inputs always produce same hash
        prov_data = {
            "operator_id": operator_id,
            "commodity": commodity.value if commodity else "",
            "base_year": base_year,
            "compare_year": compare_year,
            "metrics": [
                {"dimension": m.dimension.value, "change": str(m.change)}
                for m in metrics
            ],
        }
        prov_hash = self._provenance.compute_hash(prov_data)

        comparison = YearComparisonModel(
            comparison_id=comparison_id,
            operator_id=operator_id,
            commodity=commodity,
            base_year=base_year,
            compare_year=compare_year,
            base_snapshot=base_snap,
            compare_snapshot=compare_snap,
            status=YearComparisonStatus.COMPLETED,
            metrics=metrics,
            overall_trend=overall_trend,
            provenance_hash=prov_hash,
        )
        self._comparisons[comparison_id] = comparison

        return comparison

    async def _compare_years_legacy(
        self,
        operator_id: str,
        data_points: List[Dict[str, Any]],
    ) -> YearComparisonRecord:
        """Legacy compare_years implementation using data_points list."""
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        comparison_id = str(uuid.uuid4())

        parsed_points = self._parse_data_points(data_points)
        parsed_points.sort(key=lambda dp: dp.year)

        if len(parsed_points) > self.config.yoy_max_comparison_years:
            parsed_points = parsed_points[-self.config.yoy_max_comparison_years:]

        years = [dp.year for dp in parsed_points]

        comparisons: List[YearDimensionComparison] = []
        if len(parsed_points) >= 2:
            for i in range(len(parsed_points) - 1):
                year_a = parsed_points[i]
                year_b = parsed_points[i + 1]
                pair_comparisons = self._compare_pair(year_a, year_b)
                comparisons.extend(pair_comparisons)

        overall_trend = self._classify_overall_trend(comparisons)
        overall_significance = self._classify_overall_significance(comparisons)
        weighted_score = self._compute_weighted_change(comparisons)
        critical_count = sum(
            1 for c in comparisons
            if c.significance == ChangeSignificance.CRITICAL
        )

        recommendations = self._generate_recommendations(comparisons)

        record = YearComparisonRecord(
            comparison_id=comparison_id,
            operator_id=operator_id,
            years_compared=years,
            data_points=parsed_points,
            comparisons=comparisons,
            overall_trend=overall_trend,
            overall_significance=overall_significance,
            weighted_change_score=weighted_score,
            critical_changes=critical_count,
            recommendations=recommendations,
            compared_at=now,
        )

        prov_data = {
            "comparison_id": comparison_id,
            "operator_id": operator_id,
            "years": years,
            "comparisons": len(comparisons),
            "compared_at": now.isoformat(),
        }
        record.provenance_hash = self._provenance.compute_hash(prov_data)
        self._provenance.record(
            "year_comparison", "compare", comparison_id, AGENT_ID,
            metadata={"operator_id": operator_id, "years": years},
        )

        self._comparison_records[comparison_id] = record

        elapsed = time.monotonic() - start_time
        m.observe_year_comparison_duration(elapsed)
        m.record_year_comparison(overall_significance.value)
        m.set_critical_changes_detected(critical_count)

        return record

    async def get_comparison(
        self, comparison_id: str,
    ) -> YearComparisonModel:
        """Get a year comparison by ID.

        Raises:
            ValueError: If comparison not found.
        """
        comp = self._comparisons.get(comparison_id)
        if comp is None:
            raise ValueError(f"Comparison {comparison_id} not found")
        return comp

    async def list_comparisons(
        self,
        operator_id: str,
        commodity: Optional[EUDRCommodity] = None,
    ) -> List[YearComparisonModel]:
        """List comparisons for an operator, optionally filtered by commodity."""
        results = [
            c for c in self._comparisons.values()
            if c.operator_id == operator_id
        ]
        if commodity is not None:
            results = [c for c in results if c.commodity == commodity]
        return results

    async def analyze_multi_year_trend(
        self,
        operator_id: str,
        commodity: EUDRCommodity,
        years: List[int],
    ) -> Dict[str, Any]:
        """Analyze trends across multiple years.

        Args:
            operator_id: Operator identifier.
            commodity: EUDR commodity.
            years: List of years to analyze (must be >= 2).

        Returns:
            Dictionary with trend analysis including yearly_data and overall_direction.

        Raises:
            ValueError: If fewer than two years provided or snapshots missing.
        """
        if len(years) < 2:
            raise ValueError("Trend analysis requires at least two years")

        sorted_years = sorted(years)
        snapshots: List[YearMetricSnapshot] = []
        for yr in sorted_years:
            snap = await self.get_snapshot(operator_id, yr, commodity)
            snapshots.append(snap)

        yearly_data = []
        for snap in snapshots:
            yearly_data.append({
                "year": snap.year,
                "compliance_rate": snap.compliance_rate,
                "average_risk_score": snap.average_risk_score,
                "total_suppliers": snap.total_suppliers,
                "deforestation_free_rate": snap.deforestation_free_rate,
                "audit_findings": snap.audit_findings,
            })

        # Determine overall direction from first to last
        first = snapshots[0]
        last = snapshots[-1]
        improving = 0
        declining = 0

        if last.compliance_rate > first.compliance_rate:
            improving += 1
        elif last.compliance_rate < first.compliance_rate:
            declining += 1

        if last.average_risk_score < first.average_risk_score:
            improving += 1
        elif last.average_risk_score > first.average_risk_score:
            declining += 1

        if last.deforestation_free_rate > first.deforestation_free_rate:
            improving += 1
        elif last.deforestation_free_rate < first.deforestation_free_rate:
            declining += 1

        if last.audit_findings < first.audit_findings:
            improving += 1
        elif last.audit_findings > first.audit_findings:
            declining += 1

        if improving > declining:
            overall_direction = "improving"
        elif declining > improving:
            overall_direction = "declining"
        else:
            overall_direction = "stable"

        return {
            "operator_id": operator_id,
            "commodity": commodity.value,
            "years": sorted_years,
            "yearly_data": yearly_data,
            "overall_direction": overall_direction,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "YearComparator",
            "status": "healthy",
            "total_comparisons": len(self._comparison_records) + len(self._comparisons),
            "max_years": self.config.yoy_max_comparison_years,
        }

    # -- Private helpers --

    def _get_record(self, comparison_id: str) -> YearComparisonRecord:
        """Retrieve a comparison record or raise ValueError."""
        record = self._comparison_records.get(comparison_id)
        if record is None:
            raise ValueError(f"Comparison record {comparison_id} not found")
        return record

    def _parse_data_points(
        self, raw_points: List[Dict[str, Any]],
    ) -> List[YearDataPoint]:
        """Parse raw dictionaries into YearDataPoint models."""
        points: List[YearDataPoint] = []
        for raw in raw_points:
            points.append(self._parse_single_point(raw))
        return points

    def _parse_single_point(self, raw: Dict[str, Any]) -> YearDataPoint:
        """Parse a single raw dictionary into a YearDataPoint."""
        return YearDataPoint(
            year=raw.get("year", 0),
            supplier_count=raw.get("supplier_count", 0),
            risk_score=Decimal(str(raw.get("risk_score", 0))),
            compliance_score=Decimal(str(raw.get("compliance_score", 0))),
            deforestation_alerts=raw.get("deforestation_alerts", 0),
            due_diligence_statements=raw.get("due_diligence_statements", 0),
            total_volume_tonnes=Decimal(str(raw.get("total_volume_tonnes", 0))),
            high_risk_suppliers=raw.get("high_risk_suppliers", 0),
        )

    def _compare_pair(
        self, point_a: YearDataPoint, point_b: YearDataPoint,
    ) -> List[YearComparison]:
        """Compare two year data points across all dimensions."""
        comparisons: List[YearComparison] = []

        for dim in _COMPARISON_DIMENSIONS:
            val_a = Decimal(str(getattr(point_a, dim, 0)))
            val_b = Decimal(str(getattr(point_b, dim, 0)))

            absolute_change = val_b - val_a
            percent_change = self._safe_percent_change(val_a, val_b)
            significance = self._classify_change_significance(percent_change)
            direction = self._classify_direction(dim, absolute_change)

            comparison = YearComparison(
                dimension=dim,
                year_a=point_a.year,
                year_b=point_b.year,
                value_a=val_a,
                value_b=val_b,
                absolute_change=absolute_change,
                percent_change=percent_change,
                significance=significance,
                direction=direction,
            )
            comparisons.append(comparison)

        return comparisons

    def _safe_percent_change(
        self, old_val: Decimal, new_val: Decimal,
    ) -> Decimal:
        """Calculate percentage change with zero-division protection."""
        try:
            if old_val == Decimal("0"):
                if new_val == Decimal("0"):
                    return Decimal("0")
                return Decimal("100")
            return (
                (new_val - old_val) / abs(old_val) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        except (InvalidOperation, ZeroDivisionError):
            return Decimal("0")

    def _classify_change_significance(
        self, percent_change: Decimal,
    ) -> ChangeSignificance:
        """Classify significance using config thresholds."""
        abs_change = abs(percent_change)
        if abs_change >= self.config.yoy_critical_change_threshold:
            return ChangeSignificance.CRITICAL
        elif abs_change >= self.config.yoy_significance_threshold:
            return ChangeSignificance.SIGNIFICANT
        return ChangeSignificance.MINOR

    def _classify_direction(
        self, dimension: str, absolute_change: Decimal,
    ) -> ChangeDirection:
        """Classify direction of change for a given dimension.

        For some dimensions (risk_score, deforestation_alerts,
        high_risk_suppliers), a decrease is an improvement.
        """
        # Dimensions where lower is better
        lower_is_better = {
            "risk_score", "deforestation_alerts", "high_risk_suppliers",
        }

        if absolute_change == Decimal("0"):
            return ChangeDirection.STABLE

        if dimension in lower_is_better:
            return (
                ChangeDirection.IMPROVED
                if absolute_change < Decimal("0")
                else ChangeDirection.DEGRADED
            )
        else:
            return (
                ChangeDirection.IMPROVED
                if absolute_change > Decimal("0")
                else ChangeDirection.DEGRADED
            )

    def _classify_overall_trend(
        self, comparisons: List[YearComparison],
    ) -> ChangeDirection:
        """Classify the overall trend across all comparisons."""
        if not comparisons:
            return ChangeDirection.STABLE

        improved = sum(1 for c in comparisons if c.direction == ChangeDirection.IMPROVED)
        degraded = sum(1 for c in comparisons if c.direction == ChangeDirection.DEGRADED)

        if improved > degraded:
            return ChangeDirection.IMPROVED
        elif degraded > improved:
            return ChangeDirection.DEGRADED
        return ChangeDirection.STABLE

    def _classify_overall_significance(
        self, comparisons: List[YearComparison],
    ) -> ChangeSignificance:
        """Classify the overall significance across all comparisons."""
        if not comparisons:
            return ChangeSignificance.MINOR

        critical = sum(1 for c in comparisons if c.significance == ChangeSignificance.CRITICAL)
        significant = sum(1 for c in comparisons if c.significance == ChangeSignificance.SIGNIFICANT)

        if critical > 0:
            return ChangeSignificance.CRITICAL
        elif significant > 0:
            return ChangeSignificance.SIGNIFICANT
        return ChangeSignificance.MINOR

    def _compute_weighted_change(
        self, comparisons: List[YearComparison],
    ) -> Decimal:
        """Compute weighted change score using configured dimension weights."""
        weights = self.config.get_yoy_comparison_weights()
        weight_map = {
            "supplier_count": weights.get("supplier_count", Decimal("0.25")),
            "risk_score": weights.get("risk_score", Decimal("0.25")),
            "compliance_score": weights.get("compliance_score", Decimal("0.25")),
            "deforestation_alerts": weights.get("deforestation", Decimal("0.25")),
        }

        total = Decimal("0")
        for comp in comparisons:
            weight = weight_map.get(comp.dimension, Decimal("0"))
            total += abs(comp.percent_change) * weight

        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _generate_recommendations(
        self, comparisons: List[YearComparison],
    ) -> List[ActionRecommendation]:
        """Generate recommendations based on comparison results."""
        recommendations: List[ActionRecommendation] = []

        for comp in comparisons:
            if comp.significance == ChangeSignificance.CRITICAL:
                if comp.direction == ChangeDirection.DEGRADED:
                    recommendations.append(ActionRecommendation(
                        action=(
                            f"Critical degradation in {comp.dimension}: "
                            f"{comp.percent_change}% change from {comp.year_a} to {comp.year_b}. "
                            f"Immediate investigation required."
                        ),
                        priority="critical",
                        deadline_days=7,
                        category=comp.dimension,
                    ))
            elif comp.significance == ChangeSignificance.SIGNIFICANT:
                if comp.direction == ChangeDirection.DEGRADED:
                    recommendations.append(ActionRecommendation(
                        action=(
                            f"Significant degradation in {comp.dimension}: "
                            f"{comp.percent_change}% change. Review and address."
                        ),
                        priority="high",
                        deadline_days=30,
                        category=comp.dimension,
                    ))

        return recommendations


# Alias for backward compatibility with tests
YearComparatorEngine = YearComparator
