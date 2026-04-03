# -*- coding: utf-8 -*-
"""
AGENT-EUDR-023: Legal Compliance Verifier - Red Flag Detection Engine

Engine 4 of 7. Detects indicators of illegal activity, corruption, and
non-compliance using 40 deterministic red flag patterns across 6 categories.
All scoring is deterministic -- identical inputs always produce identical
outputs. No LLM is used for risk classification.

Red Flag Categories (6):
    1. Corruption & Bribery     (RF-001 to RF-008): 8 indicators
    2. Illegal Logging          (RF-009 to RF-015): 7 indicators
    3. Land Rights Violations   (RF-016 to RF-021): 6 indicators
    4. Labour Violations        (RF-022 to RF-027): 6 indicators
    5. Tax Evasion              (RF-028 to RF-032): 5 indicators
    6. Document Fraud           (RF-033 to RF-040): 8 indicators

Scoring Methodology (per Architecture Spec Appendix D):
    flag_score_i = base_weight_i * country_multiplier * commodity_multiplier
    aggregate = (SUM(flag_scores) / max_possible) * 100

Risk Classification:
    LOW:      0 <= score < 25
    MODERATE: 25 <= score < 50
    HIGH:     50 <= score < 75
    CRITICAL: 75 <= score <= 100

Zero-Hallucination Approach:
    - Red flag patterns are deterministic rule evaluations
    - No LLM used for risk classification
    - Every triggered flag includes the specific data point and threshold

Performance Targets:
    - Single supplier scan: <3s
    - Batch scan (100 suppliers): <60s

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-LCV-023"

# Risk classification thresholds
_RISK_LOW = Decimal("25")
_RISK_MODERATE = Decimal("50")
_RISK_HIGH = Decimal("75")

# Country multiplier ranges based on CPI
_COUNTRY_MULTIPLIER_RANGES: List[Dict[str, Any]] = [
    {"min_cpi": 0, "max_cpi": 20, "multiplier": Decimal("2.0")},
    {"min_cpi": 20, "max_cpi": 30, "multiplier": Decimal("1.8")},
    {"min_cpi": 30, "max_cpi": 40, "multiplier": Decimal("1.5")},
    {"min_cpi": 40, "max_cpi": 50, "multiplier": Decimal("1.3")},
    {"min_cpi": 50, "max_cpi": 60, "multiplier": Decimal("1.1")},
    {"min_cpi": 60, "max_cpi": 100, "multiplier": Decimal("1.0")},
]

# Commodity multiplier ranges
_COMMODITY_MULTIPLIERS: Dict[str, Decimal] = {
    "wood": Decimal("1.5"),
    "oil_palm": Decimal("1.4"),
    "soya": Decimal("1.3"),
    "cattle": Decimal("1.3"),
    "cocoa": Decimal("1.2"),
    "coffee": Decimal("1.1"),
    "rubber": Decimal("1.2"),
}

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.legal_compliance_verifier.config import get_config
except ImportError:
    get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.provenance import get_tracker
except ImportError:
    get_tracker = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.metrics import (
        record_red_flag_scan,
        record_red_flag_triggered,
        observe_red_flag_scan_duration,
    )
except ImportError:
    record_red_flag_scan = None  # type: ignore[assignment]
    record_red_flag_triggered = None  # type: ignore[assignment]
    observe_red_flag_scan_duration = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.reference_data.red_flag_indicators import (
        RED_FLAG_INDICATORS,
        get_red_flag_definition,
        get_flags_by_category,
        get_max_possible_score,
    )
except ImportError:
    RED_FLAG_INDICATORS = {}  # type: ignore[assignment]
    get_red_flag_definition = None  # type: ignore[assignment]
    get_flags_by_category = None  # type: ignore[assignment]
    get_max_possible_score = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# RedFlagDetectionEngine
# ---------------------------------------------------------------------------


class RedFlagDetectionEngine:
    """Engine 4: Deterministic red flag detection with weighted scoring.

    Evaluates 40 red flag indicators against supplier data using
    deterministic threshold comparisons and pattern matching. Scoring
    uses the formula from Architecture Spec Appendix D.

    Example:
        >>> engine = RedFlagDetectionEngine()
        >>> result = engine.scan_red_flags(
        ...     supplier_data={"country_cpi_score": 25},
        ...     country_code="CD",
        ...     commodity="wood",
        ... )
        >>> assert result["total_flags"] >= 1
    """

    def __init__(self) -> None:
        """Initialize the Red Flag Detection Engine."""
        self._indicators: Dict[str, Dict[str, Any]] = dict(RED_FLAG_INDICATORS)
        self._max_possible = self._compute_max_possible()
        logger.info(
            f"RedFlagDetectionEngine v{_MODULE_VERSION} initialized: "
            f"{len(self._indicators)} indicators, "
            f"max_possible_score={self._max_possible}"
        )

    # -------------------------------------------------------------------
    # Public API: Red flag scanning
    # -------------------------------------------------------------------

    def scan_red_flags(
        self,
        supplier_data: Dict[str, Any],
        country_code: str,
        commodity: str,
        include_categories: Optional[List[str]] = None,
        supplier_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scan supplier data for red flag indicators.

        Evaluates all 40 red flag indicators (or a filtered subset)
        against the provided supplier data using deterministic rules.

        Args:
            supplier_data: Dict of supplier data fields for evaluation.
            country_code: ISO 3166-1 alpha-2 country code.
            commodity: EUDR commodity type.
            include_categories: Optional list of categories to scan.
            supplier_id: Optional supplier identifier.

        Returns:
            Dict with triggered flags, scores, risk level, breakdown.

        Example:
            >>> engine = RedFlagDetectionEngine()
            >>> result = engine.scan_red_flags(
            ...     supplier_data={
            ...         "country_cpi_score": 19,
            ...         "sanctions_list_match": True,
            ...     },
            ...     country_code="CD",
            ...     commodity="wood",
            ... )
            >>> assert result["risk_level"] in ("low", "moderate", "high", "critical")
        """
        start_time = time.monotonic()

        country_multiplier = self._get_country_multiplier(supplier_data)
        commodity_multiplier = self._get_commodity_multiplier(commodity)

        triggered_flags: List[Dict[str, Any]] = []
        category_breakdown: Dict[str, int] = {}

        indicators_to_check = self._filter_indicators(include_categories)

        for flag_code, indicator in indicators_to_check.items():
            trigger_result = self._evaluate_trigger(
                indicator, supplier_data,
            )

            if trigger_result["triggered"]:
                flag_entry = self._build_flag_entry(
                    flag_code=flag_code,
                    indicator=indicator,
                    country_code=country_code,
                    country_multiplier=country_multiplier,
                    commodity_multiplier=commodity_multiplier,
                    triggering_data=trigger_result["data"],
                    supplier_id=supplier_id,
                )
                triggered_flags.append(flag_entry)

                cat = indicator.get("category", "unknown")
                category_breakdown[cat] = category_breakdown.get(cat, 0) + 1

                self._record_flag_metric(cat, indicator.get("severity", "unknown"))

        # Compute aggregate score
        aggregate_score = self._compute_aggregate_score(
            triggered_flags, country_multiplier, commodity_multiplier,
        )

        risk_level = self._classify_risk(aggregate_score)

        provenance_hash = self._compute_provenance_hash(
            "scan_red_flags", country_code, commodity, supplier_id or "unknown",
        )

        self._record_provenance(
            "scan", supplier_id or country_code, provenance_hash,
        )
        self._record_scan_metrics(country_code, commodity, start_time)

        return {
            "supplier_id": supplier_id,
            "country_code": country_code,
            "commodity": commodity,
            "flags_triggered": triggered_flags,
            "total_flags": len(triggered_flags),
            "aggregate_score": str(aggregate_score),
            "risk_level": risk_level,
            "category_breakdown": category_breakdown,
            "country_multiplier": str(country_multiplier),
            "commodity_multiplier": str(commodity_multiplier),
            "max_possible_score": str(self._max_possible),
            "indicators_checked": len(indicators_to_check),
            "provenance_hash": provenance_hash,
        }

    # -------------------------------------------------------------------
    # Public API: Single flag evaluation
    # -------------------------------------------------------------------

    def evaluate_single_flag(
        self,
        flag_code: str,
        supplier_data: Dict[str, Any],
        country_code: str = "XX",
        commodity: str = "wood",
    ) -> Dict[str, Any]:
        """Evaluate a single red flag indicator.

        Args:
            flag_code: Red flag code (e.g. "RF-001").
            supplier_data: Supplier data dict.
            country_code: Country code for multiplier.
            commodity: Commodity for multiplier.

        Returns:
            Dict with evaluation result.

        Example:
            >>> engine = RedFlagDetectionEngine()
            >>> result = engine.evaluate_single_flag(
            ...     "RF-001", {"country_cpi_score": 25},
            ... )
            >>> assert "triggered" in result
        """
        indicator = self._indicators.get(flag_code)
        if indicator is None:
            return {
                "flag_code": flag_code,
                "error": f"Unknown red flag code: {flag_code}",
                "triggered": False,
            }

        trigger_result = self._evaluate_trigger(indicator, supplier_data)
        country_mult = self._get_country_multiplier(supplier_data)
        commodity_mult = self._get_commodity_multiplier(commodity)

        base_weight = Decimal(str(indicator.get("base_weight", 0)))
        weighted_score = base_weight * country_mult * commodity_mult

        return {
            "flag_code": flag_code,
            "category": indicator.get("category", ""),
            "description": indicator.get("description", ""),
            "severity": indicator.get("severity", ""),
            "triggered": trigger_result["triggered"],
            "triggering_data": trigger_result["data"],
            "base_weight": str(base_weight),
            "country_multiplier": str(country_mult),
            "commodity_multiplier": str(commodity_mult),
            "weighted_score": str(weighted_score.quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP,
            )),
        }

    # -------------------------------------------------------------------
    # Public API: Category summary
    # -------------------------------------------------------------------

    def get_category_summary(self) -> Dict[str, Any]:
        """Get summary of red flag indicators by category.

        Returns:
            Dict with category names, counts, weight ranges.
        """
        categories: Dict[str, Dict[str, Any]] = {}

        for flag_code, indicator in self._indicators.items():
            cat = indicator.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {
                    "indicator_count": 0,
                    "indicators": [],
                    "weight_range": {"min": Decimal("1"), "max": Decimal("0")},
                    "severities": {},
                }

            entry = categories[cat]
            entry["indicator_count"] += 1
            entry["indicators"].append(flag_code)

            weight = Decimal(str(indicator.get("base_weight", 0)))
            if weight < entry["weight_range"]["min"]:
                entry["weight_range"]["min"] = weight
            if weight > entry["weight_range"]["max"]:
                entry["weight_range"]["max"] = weight

            sev = indicator.get("severity", "unknown")
            entry["severities"][sev] = entry["severities"].get(sev, 0) + 1

        # Convert Decimals to strings for serialization
        for cat, data in categories.items():
            data["weight_range"]["min"] = str(data["weight_range"]["min"])
            data["weight_range"]["max"] = str(data["weight_range"]["max"])

        return {
            "total_indicators": len(self._indicators),
            "total_categories": len(categories),
            "categories": categories,
        }

    # -------------------------------------------------------------------
    # Internal: Trigger evaluation
    # -------------------------------------------------------------------

    def _evaluate_trigger(
        self,
        indicator: Dict[str, Any],
        supplier_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate whether a red flag indicator is triggered.

        Uses deterministic threshold comparisons based on the
        trigger_condition field in the indicator definition.

        Args:
            indicator: Red flag indicator definition.
            supplier_data: Supplier data for evaluation.

        Returns:
            Dict with triggered (bool) and data (triggering data points).
        """
        condition = indicator.get("trigger_condition", "")
        triggered = False
        data: Dict[str, Any] = {}

        try:
            triggered, data = self._evaluate_condition(condition, supplier_data)
        except Exception as exc:
            logger.debug("Trigger evaluation error: %s", exc)
            triggered = False
            data = {"error": str(exc)}

        return {"triggered": triggered, "data": data}

    def _evaluate_condition(
        self,
        condition: str,
        supplier_data: Dict[str, Any],
    ) -> tuple:
        """Evaluate a trigger condition string against supplier data.

        Supports simple comparison operators: <, >, <=, >=, ==, !=

        Args:
            condition: Trigger condition expression.
            supplier_data: Supplier data dict.

        Returns:
            Tuple of (triggered: bool, data: dict).
        """
        # Boolean equality checks (field == True/False)
        if "==" in condition and "True" in condition:
            field = condition.split("==")[0].strip()
            value = supplier_data.get(field)
            if value is True:
                return True, {field: value, "condition": condition}
            return False, {field: value}

        if "==" in condition and "False" in condition:
            field = condition.split("==")[0].strip()
            value = supplier_data.get(field)
            if value is False:
                return True, {field: value, "condition": condition}
            return False, {field: value}

        # String equality check
        if "==" in condition and "'" in condition:
            parts = condition.split("==")
            field = parts[0].strip()
            expected = parts[1].strip().strip("'\"")
            value = supplier_data.get(field)
            if value is not None and str(value) == expected:
                return True, {field: value, "expected": expected}
            return False, {field: value}

        # Numeric comparisons
        for op in ["<=", ">=", "!=", "<", ">"]:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    field = parts[0].strip()
                    threshold_expr = parts[1].strip()

                    value = supplier_data.get(field)
                    if value is None:
                        return False, {field: None}

                    try:
                        actual = Decimal(str(value))
                        threshold = self._resolve_threshold(
                            threshold_expr, supplier_data,
                        )
                        result = self._compare(actual, op, threshold)
                        return result, {
                            field: str(actual),
                            "threshold": str(threshold),
                            "operator": op,
                            "condition": condition,
                        }
                    except Exception:
                        return False, {field: value}
                break

        return False, {"condition": condition, "note": "unresolvable"}

    def _resolve_threshold(
        self,
        expr: str,
        supplier_data: Dict[str, Any],
    ) -> Decimal:
        """Resolve a threshold expression to a Decimal value.

        Handles simple numeric values and expressions like
        "country_avg_days * 0.5" or "market_price * 0.70".

        Args:
            expr: Threshold expression string.
            supplier_data: Supplier data for variable resolution.

        Returns:
            Decimal threshold value.
        """
        expr = expr.strip()

        # Handle parenthesized multiplication expressions
        if "(" in expr:
            inner = expr.strip("()")
            return self._resolve_threshold(inner, supplier_data)

        # Handle multiplication
        if "*" in expr:
            parts = expr.split("*")
            left = parts[0].strip()
            right = parts[1].strip()

            left_val = supplier_data.get(left)
            if left_val is not None:
                return Decimal(str(left_val)) * Decimal(right)

            right_val = supplier_data.get(right)
            if right_val is not None:
                return Decimal(left) * Decimal(str(right_val))

            return Decimal(left) * Decimal(right)

        # Simple numeric
        try:
            return Decimal(expr)
        except Exception:
            val = supplier_data.get(expr)
            if val is not None:
                return Decimal(str(val))
            return Decimal("0")

    def _compare(
        self, actual: Decimal, op: str, threshold: Decimal,
    ) -> bool:
        """Perform a deterministic comparison.

        Args:
            actual: Actual value.
            op: Comparison operator.
            threshold: Threshold value.

        Returns:
            Boolean comparison result.
        """
        if op == "<":
            return actual < threshold
        elif op == ">":
            return actual > threshold
        elif op == "<=":
            return actual <= threshold
        elif op == ">=":
            return actual >= threshold
        elif op == "!=":
            return actual != threshold
        return False

    # -------------------------------------------------------------------
    # Internal: Multiplier computation
    # -------------------------------------------------------------------

    def _get_country_multiplier(
        self, supplier_data: Dict[str, Any],
    ) -> Decimal:
        """Determine country multiplier based on CPI score.

        Args:
            supplier_data: Must contain 'country_cpi_score'.

        Returns:
            Decimal multiplier (1.0 to 2.0).
        """
        cpi = supplier_data.get("country_cpi_score")
        if cpi is None:
            return Decimal("1.0")

        cpi_val = int(cpi)
        for rng in _COUNTRY_MULTIPLIER_RANGES:
            if rng["min_cpi"] <= cpi_val < rng["max_cpi"]:
                return rng["multiplier"]

        return Decimal("1.0")

    def _get_commodity_multiplier(self, commodity: str) -> Decimal:
        """Get commodity-specific multiplier.

        Args:
            commodity: EUDR commodity type.

        Returns:
            Decimal multiplier (1.0 to 1.5).
        """
        return _COMMODITY_MULTIPLIERS.get(commodity, Decimal("1.0"))

    # -------------------------------------------------------------------
    # Internal: Score computation
    # -------------------------------------------------------------------

    def _compute_aggregate_score(
        self,
        flags: List[Dict[str, Any]],
        country_multiplier: Decimal,
        commodity_multiplier: Decimal,
    ) -> Decimal:
        """Compute normalized aggregate red flag score.

        Formula: aggregate = (SUM(flag_scores) / max_possible) * 100

        Args:
            flags: List of triggered flag entry dicts.
            country_multiplier: Country CPI-based multiplier.
            commodity_multiplier: Commodity risk multiplier.

        Returns:
            Decimal aggregate score (0-100).
        """
        if not flags or self._max_possible == Decimal("0"):
            return Decimal("0")

        total = Decimal("0")
        for flag in flags:
            weight = Decimal(str(flag.get("base_weight", "0")))
            score = weight * country_multiplier * commodity_multiplier
            total += score

        aggregate = (total / self._max_possible) * Decimal("100")
        aggregate = aggregate.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return min(aggregate, Decimal("100"))

    def _compute_max_possible(self) -> Decimal:
        """Calculate the maximum possible aggregate score.

        max_possible = sum(base_weight * 2.0 * 1.5) for all 40 flags.

        Returns:
            Decimal maximum possible score.
        """
        total = Decimal("0")
        for indicator in self._indicators.values():
            weight = Decimal(str(indicator.get("base_weight", 0)))
            total += weight * Decimal("2.0") * Decimal("1.5")
        return total if total > 0 else Decimal("1")

    def _classify_risk(self, score: Decimal) -> str:
        """Classify risk level from aggregate score.

        Args:
            score: Aggregate score (0-100).

        Returns:
            Risk level string (low/moderate/high/critical).
        """
        if score >= _RISK_HIGH:
            return "critical"
        elif score >= _RISK_MODERATE:
            return "high"
        elif score >= _RISK_LOW:
            return "moderate"
        return "low"

    # -------------------------------------------------------------------
    # Internal: Flag entry construction
    # -------------------------------------------------------------------

    def _build_flag_entry(
        self,
        flag_code: str,
        indicator: Dict[str, Any],
        country_code: str,
        country_multiplier: Decimal,
        commodity_multiplier: Decimal,
        triggering_data: Dict[str, Any],
        supplier_id: Optional[str],
    ) -> Dict[str, Any]:
        """Build a triggered flag entry dict.

        Args:
            flag_code: Red flag code.
            indicator: Indicator definition.
            country_code: Country code.
            country_multiplier: Country multiplier.
            commodity_multiplier: Commodity multiplier.
            triggering_data: Data that triggered the flag.
            supplier_id: Optional supplier ID.

        Returns:
            Flag entry dict.
        """
        base_weight = Decimal(str(indicator.get("base_weight", 0)))
        weighted = base_weight * country_multiplier * commodity_multiplier
        weighted = weighted.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        return {
            "flag_code": flag_code,
            "category": indicator.get("category", ""),
            "description": indicator.get("description", ""),
            "severity": indicator.get("severity", ""),
            "base_weight": str(base_weight),
            "country_multiplier": str(country_multiplier),
            "commodity_multiplier": str(commodity_multiplier),
            "weighted_score": str(weighted),
            "triggering_data": triggering_data,
            "data_source": indicator.get("data_source", ""),
            "country_code": country_code,
            "supplier_id": supplier_id,
        }

    # -------------------------------------------------------------------
    # Internal: Indicator filtering
    # -------------------------------------------------------------------

    def _filter_indicators(
        self,
        include_categories: Optional[List[str]],
    ) -> Dict[str, Dict[str, Any]]:
        """Filter indicators by category.

        Args:
            include_categories: Optional list of categories to include.

        Returns:
            Filtered indicators dict.
        """
        if not include_categories:
            return self._indicators

        return {
            code: ind for code, ind in self._indicators.items()
            if ind.get("category") in include_categories
        }

    # -------------------------------------------------------------------
    # Internal: Provenance and metrics
    # -------------------------------------------------------------------

    def _compute_provenance_hash(
        self,
        operation: str,
        country_code: str,
        commodity: str,
        supplier_id: str,
    ) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "agent_id": _AGENT_ID,
            "engine": "red_flag_detection",
            "version": _MODULE_VERSION,
            "operation": operation,
            "country_code": country_code,
            "commodity": commodity,
            "supplier_id": supplier_id,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _record_provenance(
        self, action: str, entity_id: str, provenance_hash: str,
    ) -> None:
        """Record provenance entry."""
        if get_tracker is not None:
            try:
                tracker = get_tracker()
                tracker.record(
                    entity_type="red_flag_alert",
                    action=action,
                    entity_id=entity_id,
                    metadata={"provenance_hash": provenance_hash},
                )
            except Exception as exc:
                logger.warning("Provenance recording failed: %s", exc)

    def _record_scan_metrics(
        self, country_code: str, commodity: str, start_time: float,
    ) -> None:
        """Record scan metrics."""
        elapsed = time.monotonic() - start_time
        if record_red_flag_scan is not None:
            try:
                record_red_flag_scan(country_code, commodity)
            except Exception:
                pass
        if observe_red_flag_scan_duration is not None:
            try:
                observe_red_flag_scan_duration(elapsed)
            except Exception:
                pass

    def _record_flag_metric(
        self, category: str, severity: str,
    ) -> None:
        """Record individual flag trigger metric."""
        if record_red_flag_triggered is not None:
            try:
                record_red_flag_triggered(category, severity)
            except Exception:
                pass
