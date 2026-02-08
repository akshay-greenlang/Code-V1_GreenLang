# -*- coding: utf-8 -*-
"""
Validation Engine - AGENT-DATA-001: PDF & Invoice Extractor

Cross-field validation engine that applies configurable rules to
structured extraction output.  Ships with built-in rules for invoices,
shipping manifests, and utility bills covering total consistency,
date ordering, required fields, and numeric range checks.

Features:
    - Document-type-specific validation rule sets
    - Total consistency checks (subtotal + tax = total)
    - Line-item sum vs subtotal reconciliation
    - Date ordering and future-date detection
    - Required-field presence checks
    - Numeric range validation (non-negative amounts, plausible rates)
    - Custom rule registration
    - Thread-safe statistics

Zero-Hallucination Guarantees:
    - All validations are deterministic arithmetic or string checks
    - No ML/LLM inference in the validation path
    - Results contain expected vs actual for full traceability

Example:
    >>> from greenlang.pdf_extractor.validation_engine import ValidationEngine
    >>> engine = ValidationEngine()
    >>> results = engine.validate_document("invoice", structured_data, fields)
    >>> for r in results:
    ...     print(r.rule_name, r.passed, r.message)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationResult",
    "ValidationEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Result of a single validation check."""

    rule_name: str = Field(..., description="Validation rule identifier")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(default="", description="Human-readable message")
    severity: str = Field(
        default="warning",
        description="Severity level: info, warning, error",
    )
    expected: Optional[str] = Field(None, description="Expected value")
    actual: Optional[str] = Field(None, description="Actual value")
    field_name: Optional[str] = Field(
        None, description="Field this rule applies to",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Built-in rule configurations
# ---------------------------------------------------------------------------

_INVOICE_REQUIRED_FIELDS = [
    "invoice_number", "invoice_date", "total_amount",
]

_MANIFEST_REQUIRED_FIELDS = [
    "manifest_number", "shipment_date", "carrier_name",
]

_UTILITY_BILL_REQUIRED_FIELDS = [
    "account_number", "billing_period_start", "billing_period_end",
    "total_amount",
]

_INVOICE_NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "subtotal": (0.0, 1e9),
    "tax_amount": (0.0, 1e8),
    "total_amount": (0.01, 1e9),
}

_MANIFEST_NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "total_weight": (0.0, 1e8),
    "total_pieces": (0, 1e6),
}

_UTILITY_NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "consumption": (0.0, 1e8),
    "rate": (0.0, 1e4),
    "total_amount": (0.0, 1e7),
    "previous_reading": (0.0, 1e9),
    "current_reading": (0.0, 1e9),
}

# Date field names per document type
_DATE_FIELDS: Dict[str, List[str]] = {
    "invoice": ["invoice_date", "due_date"],
    "manifest": ["shipment_date"],
    "utility_bill": ["billing_period_start", "billing_period_end"],
}

# Ordered date pairs for chronological validation
_DATE_ORDERING: Dict[str, List[Tuple[str, str]]] = {
    "invoice": [("invoice_date", "due_date")],
    "utility_bill": [("billing_period_start", "billing_period_end")],
}


# ---------------------------------------------------------------------------
# ValidationEngine
# ---------------------------------------------------------------------------


class ValidationEngine:
    """Cross-field validation engine with built-in and custom rules.

    Applies document-type-specific validation rules to structured
    extraction data.  Rules cover financial consistency, required fields,
    date ordering, numeric ranges, and custom callable validators.

    Attributes:
        _config: Configuration dictionary.
        _custom_rules: User-registered custom rule functions.
        _lock: Threading lock for statistics.
        _stats: Validation statistics.

    Example:
        >>> engine = ValidationEngine()
        >>> results = engine.validate_document("invoice", data, fields)
        >>> failures = [r for r in results if not r.passed]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise ValidationEngine.

        Args:
            config: Optional configuration dict.  Recognised keys:
                - ``tolerance``: float for total matching (default 0.02)
                - ``future_date_check``: bool (default True)
                - ``max_future_days``: int (default 365)
        """
        self._config = config or {}
        self._tolerance: float = self._config.get("tolerance", 0.02)
        self._future_check: bool = self._config.get("future_date_check", True)
        self._max_future_days: int = self._config.get("max_future_days", 365)
        self._custom_rules: Dict[
            str, Callable[[Dict[str, Any]], List[ValidationResult]]
        ] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "total_validations": 0,
            "rules_evaluated": 0,
            "rules_passed": 0,
            "rules_failed": 0,
            "by_type": {},
            "errors": 0,
        }
        logger.info(
            "ValidationEngine initialised: tolerance=%.4f, "
            "future_check=%s, max_future_days=%d",
            self._tolerance, self._future_check, self._max_future_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_document(
        self,
        document_type: str,
        structured_data: Dict[str, Any],
        extracted_fields: Optional[List[Any]] = None,
    ) -> List[ValidationResult]:
        """Validate structured extraction data for a document type.

        Dispatches to the appropriate type-specific validator and also
        runs any registered custom rules.

        Args:
            document_type: Document type key (e.g. "invoice").
            structured_data: Dictionary of extracted field values.
            extracted_fields: Optional list of ExtractedField objects
                (used for confidence-based filtering).

        Returns:
            List of ValidationResult objects.
        """
        start = time.monotonic()

        dispatch: Dict[str, Callable[[Dict[str, Any]], List[ValidationResult]]] = {
            "invoice": self.validate_invoice,
            "manifest": self.validate_manifest,
            "utility_bill": self.validate_utility_bill,
        }

        handler = dispatch.get(document_type)
        results: List[ValidationResult] = []

        if handler:
            results.extend(handler(structured_data))
        else:
            logger.warning(
                "No built-in rules for document_type '%s'", document_type,
            )

        # Run custom rules
        for rule_name, rule_func in self._custom_rules.items():
            try:
                custom_results = rule_func(structured_data)
                results.extend(custom_results)
            except Exception as exc:
                logger.error(
                    "Custom rule '%s' raised: %s", rule_name, exc,
                )
                with self._lock:
                    self._stats["errors"] += 1
                results.append(ValidationResult(
                    rule_name=f"custom_{rule_name}",
                    passed=False,
                    message=f"Custom rule error: {str(exc)}",
                    severity="error",
                ))

        elapsed_ms = (time.monotonic() - start) * 1000

        # Update stats
        with self._lock:
            self._stats["total_validations"] += 1
            self._stats["rules_evaluated"] += len(results)
            self._stats["rules_passed"] += sum(1 for r in results if r.passed)
            self._stats["rules_failed"] += sum(
                1 for r in results if not r.passed
            )
            type_stats = self._stats["by_type"].setdefault(
                document_type, {"calls": 0, "passed": 0, "failed": 0},
            )
            type_stats["calls"] += 1
            type_stats["passed"] += sum(1 for r in results if r.passed)
            type_stats["failed"] += sum(1 for r in results if not r.passed)

        logger.info(
            "Validated '%s': %d rules, %d passed, %d failed (%.1f ms)",
            document_type,
            len(results),
            sum(1 for r in results if r.passed),
            sum(1 for r in results if not r.passed),
            elapsed_ms,
        )
        return results

    def validate_invoice(
        self,
        data: Dict[str, Any],
    ) -> List[ValidationResult]:
        """Run invoice-specific validations.

        Checks:
        - Required fields (invoice_number, invoice_date, total_amount)
        - Totals consistency (subtotal + tax = total)
        - Line items sum vs subtotal
        - Date ordering (invoice_date <= due_date)
        - Future date check
        - Numeric ranges

        Args:
            data: Dictionary of invoice field values.

        Returns:
            List of ValidationResult objects.
        """
        results: List[ValidationResult] = []

        results.extend(
            self._validate_required_fields(data, _INVOICE_REQUIRED_FIELDS)
        )

        subtotal = self._get_float(data, "subtotal")
        tax_amount = self._get_float(data, "tax_amount")
        total_amount = self._get_float(data, "total_amount")

        if subtotal is not None and tax_amount is not None and total_amount is not None:
            results.extend(
                self._validate_totals(subtotal, tax_amount, total_amount)
            )

        line_items = data.get("line_items", [])
        if line_items and subtotal is not None:
            results.extend(
                self._validate_line_items_sum(line_items, subtotal)
            )

        date_dict = {
            k: data.get(k)
            for k in _DATE_FIELDS.get("invoice", [])
            if data.get(k) is not None
        }
        results.extend(self._validate_dates(date_dict, "invoice"))

        results.extend(
            self._validate_numeric_ranges(data, _INVOICE_NUMERIC_RANGES)
        )

        return results

    def validate_manifest(
        self,
        data: Dict[str, Any],
    ) -> List[ValidationResult]:
        """Run manifest-specific validations.

        Checks:
        - Required fields (manifest_number, shipment_date, carrier_name)
        - Weight total vs cargo items
        - Numeric ranges
        - Date validation

        Args:
            data: Dictionary of manifest field values.

        Returns:
            List of ValidationResult objects.
        """
        results: List[ValidationResult] = []

        results.extend(
            self._validate_required_fields(data, _MANIFEST_REQUIRED_FIELDS)
        )

        total_weight = self._get_float(data, "total_weight")
        cargo_items = data.get("cargo_items", [])
        if cargo_items and total_weight is not None:
            weights = []
            for item in cargo_items:
                w = None
                if isinstance(item, dict):
                    w = self._get_float(item, "weight") or self._get_float(item, "amount")
                elif hasattr(item, "amount"):
                    w = item.amount
                if w is not None:
                    weights.append(w)

            if weights:
                computed = round(sum(weights), 2)
                tolerance = max(total_weight * 0.01, 0.5)
                passed = abs(computed - total_weight) <= tolerance
                results.append(ValidationResult(
                    rule_name="manifest_weight_total",
                    passed=passed,
                    message=(
                        "Cargo weights sum matches declared total"
                        if passed
                        else f"Weight mismatch: items={computed}, "
                             f"declared={total_weight}"
                    ),
                    severity="warning" if not passed else "info",
                    expected=str(total_weight),
                    actual=str(computed),
                    field_name="total_weight",
                ))

        results.extend(
            self._validate_numeric_ranges(data, _MANIFEST_NUMERIC_RANGES)
        )

        date_dict = {
            k: data.get(k)
            for k in _DATE_FIELDS.get("manifest", [])
            if data.get(k) is not None
        }
        results.extend(self._validate_dates(date_dict, "manifest"))

        return results

    def validate_utility_bill(
        self,
        data: Dict[str, Any],
    ) -> List[ValidationResult]:
        """Run utility bill specific validations.

        Checks:
        - Required fields
        - Reading order (current >= previous)
        - Consumption = current - previous reading
        - Date ordering (start <= end)
        - Numeric ranges

        Args:
            data: Dictionary of utility bill field values.

        Returns:
            List of ValidationResult objects.
        """
        results: List[ValidationResult] = []

        results.extend(
            self._validate_required_fields(data, _UTILITY_BILL_REQUIRED_FIELDS)
        )

        # Reading consistency
        prev = self._get_float(data, "previous_reading")
        curr = self._get_float(data, "current_reading")
        consumption = self._get_float(data, "consumption")

        if prev is not None and curr is not None:
            passed = curr >= prev
            results.append(ValidationResult(
                rule_name="reading_order",
                passed=passed,
                message=(
                    "Current reading >= previous reading"
                    if passed
                    else f"Current reading ({curr}) < previous ({prev})"
                ),
                severity="error" if not passed else "info",
                expected=f">= {prev}",
                actual=str(curr),
                field_name="current_reading",
            ))

            # Consumption cross-check
            if consumption is not None:
                expected_consumption = round(curr - prev, 2)
                tolerance = max(expected_consumption * 0.01, 0.5)
                cons_passed = abs(consumption - expected_consumption) <= tolerance
                results.append(ValidationResult(
                    rule_name="consumption_consistency",
                    passed=cons_passed,
                    message=(
                        "Consumption matches meter delta"
                        if cons_passed
                        else f"Consumption ({consumption}) != "
                             f"current - previous ({expected_consumption})"
                    ),
                    severity="warning" if not cons_passed else "info",
                    expected=str(expected_consumption),
                    actual=str(consumption),
                    field_name="consumption",
                ))

        date_dict = {
            k: data.get(k)
            for k in _DATE_FIELDS.get("utility_bill", [])
            if data.get(k) is not None
        }
        results.extend(self._validate_dates(date_dict, "utility_bill"))

        results.extend(
            self._validate_numeric_ranges(data, _UTILITY_NUMERIC_RANGES)
        )

        return results

    def register_custom_rule(
        self,
        name: str,
        validation_func: Callable[[Dict[str, Any]], List[ValidationResult]],
    ) -> None:
        """Register a custom validation rule.

        Args:
            name: Unique rule name.
            validation_func: Callable accepting data dict and returning
                a list of ValidationResult objects.
        """
        self._custom_rules[name] = validation_func
        logger.info("Registered custom validation rule: '%s'", name)

    def get_statistics(self) -> Dict[str, Any]:
        """Return validation statistics.

        Returns:
            Dictionary with total validations, pass/fail counts,
            and per-type breakdowns.
        """
        with self._lock:
            total = self._stats["total_validations"]
            evaluated = self._stats["rules_evaluated"]
            return {
                "total_validations": total,
                "rules_evaluated": evaluated,
                "rules_passed": self._stats["rules_passed"],
                "rules_failed": self._stats["rules_failed"],
                "pass_rate": round(
                    self._stats["rules_passed"] / max(evaluated, 1), 4,
                ),
                "by_type": dict(self._stats["by_type"]),
                "custom_rules_registered": len(self._custom_rules),
                "errors": self._stats["errors"],
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Validation building blocks
    # ------------------------------------------------------------------

    def _validate_totals(
        self,
        subtotal: float,
        tax: float,
        total: float,
    ) -> List[ValidationResult]:
        """Validate total = subtotal + tax.

        Args:
            subtotal: Invoice subtotal.
            tax: Tax amount.
            total: Declared total.

        Returns:
            List with one ValidationResult.
        """
        expected = round(subtotal + tax, 2)
        actual = round(total, 2)
        passed = abs(expected - actual) <= self._tolerance

        return [ValidationResult(
            rule_name="totals_consistency",
            passed=passed,
            message=(
                "Total = subtotal + tax"
                if passed
                else f"Total mismatch: subtotal({subtotal}) + "
                     f"tax({tax}) = {expected}, declared total = {actual}"
            ),
            severity="error" if not passed else "info",
            expected=str(expected),
            actual=str(actual),
        )]

    def _validate_line_items_sum(
        self,
        items: List[Any],
        subtotal: float,
    ) -> List[ValidationResult]:
        """Validate that line item amounts sum to subtotal.

        Args:
            items: List of line item dicts or objects with ``amount``.
            subtotal: Declared subtotal.

        Returns:
            List with one ValidationResult, or empty if no amounts.
        """
        amounts: List[float] = []
        for item in items:
            amt = None
            if isinstance(item, dict):
                amt = self._get_float(item, "amount")
            elif hasattr(item, "amount") and item.amount is not None:
                amt = float(item.amount)
            if amt is not None:
                amounts.append(amt)

        if not amounts:
            return []

        computed = round(sum(amounts), 2)
        tolerance = max(subtotal * 0.01, self._tolerance)
        passed = abs(computed - subtotal) <= tolerance

        return [ValidationResult(
            rule_name="line_items_sum",
            passed=passed,
            message=(
                "Line items sum matches subtotal"
                if passed
                else f"Line items sum ({computed}) != subtotal ({subtotal})"
            ),
            severity="warning" if not passed else "info",
            expected=str(subtotal),
            actual=str(computed),
        )]

    def _validate_dates(
        self,
        dates_dict: Dict[str, Any],
        doc_type: str,
    ) -> List[ValidationResult]:
        """Validate date fields for ordering and future-date checks.

        Args:
            dates_dict: Dict of field_name -> date string (ISO format).
            doc_type: Document type key for ordering rules.

        Returns:
            List of ValidationResult objects.
        """
        results: List[ValidationResult] = []

        # Parse dates
        parsed: Dict[str, datetime] = {}
        for field, value in dates_dict.items():
            if value is None:
                continue
            try:
                dt = datetime.strptime(str(value), "%Y-%m-%d")
                parsed[field] = dt
            except ValueError:
                results.append(ValidationResult(
                    rule_name=f"date_format_{field}",
                    passed=False,
                    message=f"Could not parse date for '{field}': {value}",
                    severity="warning",
                    field_name=field,
                    actual=str(value),
                ))

        # Ordering checks
        orderings = _DATE_ORDERING.get(doc_type, [])
        for earlier_field, later_field in orderings:
            if earlier_field in parsed and later_field in parsed:
                passed = parsed[earlier_field] <= parsed[later_field]
                results.append(ValidationResult(
                    rule_name=f"date_order_{earlier_field}_{later_field}",
                    passed=passed,
                    message=(
                        f"{earlier_field} <= {later_field}"
                        if passed
                        else f"{earlier_field} ({dates_dict[earlier_field]}) "
                             f"> {later_field} ({dates_dict[later_field]})"
                    ),
                    severity="warning" if not passed else "info",
                    expected=f"{earlier_field} <= {later_field}",
                ))

        # Future date checks
        if self._future_check:
            now = _utcnow()
            for field, dt in parsed.items():
                days_ahead = (dt - now.replace(tzinfo=None)).days
                if days_ahead > self._max_future_days:
                    results.append(ValidationResult(
                        rule_name=f"future_date_{field}",
                        passed=False,
                        message=(
                            f"{field} is {days_ahead} days in the future "
                            f"(max {self._max_future_days})"
                        ),
                        severity="warning",
                        field_name=field,
                        actual=str(dates_dict[field]),
                    ))

        return results

    def _validate_required_fields(
        self,
        data: Dict[str, Any],
        required_fields: List[str],
    ) -> List[ValidationResult]:
        """Check that required fields are present and non-None.

        Args:
            data: Structured data dictionary.
            required_fields: List of required field names.

        Returns:
            List of ValidationResult objects.
        """
        results: List[ValidationResult] = []
        for field in required_fields:
            value = data.get(field)
            present = value is not None and str(value).strip() != ""
            results.append(ValidationResult(
                rule_name=f"required_{field}",
                passed=present,
                message=(
                    f"{field} is present"
                    if present
                    else f"Required field '{field}' is missing"
                ),
                severity="error" if not present else "info",
                field_name=field,
            ))
        return results

    def _validate_numeric_ranges(
        self,
        data: Dict[str, Any],
        field_ranges: Dict[str, Tuple[float, float]],
    ) -> List[ValidationResult]:
        """Validate numeric fields fall within expected ranges.

        Args:
            data: Structured data dictionary.
            field_ranges: Dict of field_name -> (min_val, max_val).

        Returns:
            List of ValidationResult objects.
        """
        results: List[ValidationResult] = []
        for field, (min_val, max_val) in field_ranges.items():
            value = self._get_float(data, field)
            if value is None:
                continue

            in_range = min_val <= value <= max_val
            results.append(ValidationResult(
                rule_name=f"range_{field}",
                passed=in_range,
                message=(
                    f"{field} ({value}) is within [{min_val}, {max_val}]"
                    if in_range
                    else f"{field} ({value}) is outside [{min_val}, {max_val}]"
                ),
                severity="warning" if not in_range else "info",
                expected=f"[{min_val}, {max_val}]",
                actual=str(value),
                field_name=field,
            ))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_float(
        self,
        data: Dict[str, Any],
        field: str,
    ) -> Optional[float]:
        """Safely extract a float value from a data dict.

        Args:
            data: Data dictionary.
            field: Field name.

        Returns:
            Float value or None.
        """
        val = data.get(field)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
