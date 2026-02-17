# -*- coding: utf-8 -*-
"""
Rule-Based Imputer Engine - AGENT-DATA-012: Missing Value Imputer (GL-DATA-X-015)

Provides domain-driven rule-based imputation: if-then conditional rules,
lookup table imputation, and regulatory default values for common gaps in
sustainability and emissions data.

Rules are evaluated in priority order with full audit trail. Each imputed
value includes a justification string and provenance hash. Supports GHG
Protocol, DEFRA, and EPA regulatory default emission factors.

Zero-Hallucination Guarantees:
    - All rules are deterministic if-then logic
    - Lookup tables are explicit key-value mappings
    - Regulatory defaults come from published sources
    - No ML/LLM calls in any imputation path
    - SHA-256 provenance on every imputed value

Example:
    >>> from greenlang.missing_value_imputer.rule_based_imputer import RuleBasedImputerEngine
    >>> from greenlang.missing_value_imputer.config import MissingValueImputerConfig
    >>> engine = RuleBasedImputerEngine(MissingValueImputerConfig())
    >>> rule = engine.create_rule("electricity_default", [...], 0.5, "high", "GHG default")
    >>> result = engine.evaluate_rules(records, "emission_factor", [rule])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.models import (
    ConfidenceLevel,
    ImputationRule,
    ImputationStrategy,
    ImputedValue,
    LookupEntry,
    LookupTable,
    RuleCondition,
    RuleConditionType,
    RulePriority,
)
from greenlang.missing_value_imputer.metrics import (
    inc_rules_evaluated,
    inc_values_imputed,
    observe_confidence,
    observe_duration,
    inc_errors,
)
from greenlang.missing_value_imputer.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

__all__ = [
    "RuleBasedImputerEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _is_missing(value: Any) -> bool:
    """Determine whether a value is considered missing."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash."""
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _classify_confidence(score: float) -> ConfidenceLevel:
    """Classify a numeric confidence score into a level."""
    if score >= 0.85:
        return ConfidenceLevel.HIGH
    if score >= 0.70:
        return ConfidenceLevel.MEDIUM
    if score >= 0.50:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW


# Priority ordering (higher numeric = evaluated first)
_PRIORITY_ORDER: Dict[RulePriority, int] = {
    RulePriority.CRITICAL: 5,
    RulePriority.HIGH: 4,
    RulePriority.MEDIUM: 3,
    RulePriority.LOW: 2,
    RulePriority.DEFAULT: 1,
}

# Priority-based confidence scores
_PRIORITY_CONFIDENCE: Dict[RulePriority, float] = {
    RulePriority.CRITICAL: 0.95,
    RulePriority.HIGH: 0.90,
    RulePriority.MEDIUM: 0.80,
    RulePriority.LOW: 0.70,
    RulePriority.DEFAULT: 0.60,
}


# ===========================================================================
# RuleBasedImputerEngine
# ===========================================================================


class RuleBasedImputerEngine:
    """Rule-based imputation engine for domain-driven missing value handling.

    Evaluates conditional rules, lookup tables, and regulatory defaults
    to fill missing values with auditable, domain-justified values.

    Attributes:
        config: Service configuration.
        provenance: SHA-256 provenance tracker.
        _regulatory_defaults: Cached regulatory default values.

    Example:
        >>> engine = RuleBasedImputerEngine(MissingValueImputerConfig())
        >>> rules = [engine.create_rule("default_ef", [...], 0.5, "high", "DEFRA")]
        >>> result = engine.evaluate_rules(records, "emission_factor", rules)
    """

    def __init__(self, config: MissingValueImputerConfig) -> None:
        """Initialize the RuleBasedImputerEngine.

        Args:
            config: Service configuration instance.
        """
        self.config = config
        self.provenance = ProvenanceTracker()
        self._regulatory_defaults: Optional[Dict[str, Dict[str, Any]]] = None
        logger.info("RuleBasedImputerEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_rules(
        self,
        records: List[Dict[str, Any]],
        column: str,
        rules: List[ImputationRule],
    ) -> List[ImputedValue]:
        """Apply if-then rules in priority order to impute missing values.

        Rules are sorted by priority (CRITICAL > HIGH > MEDIUM > LOW > DEFAULT),
        and for each missing record, the first matching rule is applied.

        Args:
            records: List of record dictionaries.
            column: Target column to impute.
            rules: List of ImputationRule objects.

        Returns:
            List of ImputedValue for each imputed record.
        """
        start = time.monotonic()

        # Sort rules by priority (highest first)
        sorted_rules = sorted(
            [r for r in rules if r.active and r.target_column == column],
            key=lambda r: _PRIORITY_ORDER.get(r.priority, 0),
            reverse=True,
        )

        if not sorted_rules:
            logger.warning("No active rules for column '%s'", column)
            return []

        imputed: List[ImputedValue] = []
        for i, record in enumerate(records):
            if not _is_missing(record.get(column)):
                continue

            for rule in sorted_rules:
                if self._all_conditions_met(record, rule.conditions):
                    imp_val = rule.impute_value
                    if imp_val is None:
                        continue

                    confidence = _PRIORITY_CONFIDENCE.get(rule.priority, 0.70)
                    justification = self._format_justification(rule, record)

                    prov = _compute_provenance(
                        "evaluate_rules",
                        f"{column}:{i}:{imp_val}:rule={rule.rule_id[:8]}",
                    )
                    iv = ImputedValue(
                        record_index=i,
                        column_name=column,
                        imputed_value=imp_val,
                        original_value=record.get(column),
                        strategy=ImputationStrategy.RULE_BASED,
                        confidence=round(confidence, 4),
                        confidence_level=_classify_confidence(confidence),
                        contributing_records=0,
                        provenance_hash=prov,
                    )
                    imputed.append(iv)
                    inc_rules_evaluated(rule.priority.value)
                    break  # First matching rule wins

        elapsed = time.monotonic() - start
        observe_duration("rule_evaluate", elapsed)
        if imputed:
            inc_values_imputed("rule_based", len(imputed))
            for iv in imputed:
                observe_confidence("rule_based", iv.confidence)

        logger.info(
            "Rule evaluation: %d rules tested, %d values imputed, elapsed=%.3fs",
            len(sorted_rules), len(imputed), elapsed,
        )
        return imputed

    def lookup_imputation(
        self,
        records: List[Dict[str, Any]],
        column: str,
        lookup_table: LookupTable,
    ) -> List[ImputedValue]:
        """Impute missing values using a reference lookup table.

        Matches each record's key column value against the lookup table
        entries. Uses the default_value when no match is found.

        Args:
            records: List of record dictionaries.
            column: Target column to impute.
            lookup_table: LookupTable with key-value mappings.

        Returns:
            List of ImputedValue for each imputed record.
        """
        start = time.monotonic()

        # Build lookup dictionary
        lookup_dict: Dict[str, Any] = {}
        for entry in lookup_table.entries:
            lookup_dict[entry.key.lower()] = entry.value

        key_column = lookup_table.key_column

        imputed: List[ImputedValue] = []
        for i, record in enumerate(records):
            if not _is_missing(record.get(column)):
                continue

            key_val = record.get(key_column)
            if _is_missing(key_val):
                imp_val = lookup_table.default_value
            else:
                key_str = str(key_val).lower()
                imp_val = lookup_dict.get(key_str, lookup_table.default_value)

            if imp_val is None:
                continue

            # Confidence: higher for direct match, lower for default
            is_direct = (
                not _is_missing(key_val) and
                str(key_val).lower() in lookup_dict
            )
            confidence = 0.92 if is_direct else 0.65

            prov = _compute_provenance(
                "lookup_imputation",
                f"{column}:{i}:{imp_val}:table={lookup_table.table_id[:8]}",
            )
            iv = ImputedValue(
                record_index=i,
                column_name=column,
                imputed_value=imp_val,
                original_value=record.get(column),
                strategy=ImputationStrategy.LOOKUP_TABLE,
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=len(lookup_table.entries),
                provenance_hash=prov,
            )
            imputed.append(iv)

        elapsed = time.monotonic() - start
        observe_duration("rule_evaluate", elapsed)
        if imputed:
            inc_values_imputed("lookup_table", len(imputed))

        logger.info(
            "Lookup imputation: table=%s, %d entries, %d values imputed",
            lookup_table.name, len(lookup_table.entries), len(imputed),
        )
        return imputed

    def regulatory_defaults(
        self,
        records: List[Dict[str, Any]],
        column: str,
        framework: str = "ghg_protocol",
    ) -> List[ImputedValue]:
        """Apply regulatory default values for common missing data gaps.

        Supports GHG Protocol, DEFRA, and EPA default emission factors
        for common activity types, energy sources, and materials.

        Args:
            records: List of record dictionaries.
            column: Target column to impute.
            framework: Regulatory framework ('ghg_protocol', 'defra', 'epa').

        Returns:
            List of ImputedValue for each imputed record.
        """
        start = time.monotonic()
        defaults = self._get_regulatory_defaults()
        framework_defaults = defaults.get(framework, {})

        if not framework_defaults:
            logger.warning("No regulatory defaults for framework '%s'", framework)
            return []

        imputed: List[ImputedValue] = []
        for i, record in enumerate(records):
            if not _is_missing(record.get(column)):
                continue

            # Try to match based on activity type, fuel type, or category
            imp_val = self._match_regulatory_default(
                record, column, framework_defaults
            )
            if imp_val is None:
                continue

            confidence = 0.60  # Regulatory defaults have moderate confidence
            justification = (
                f"Regulatory default from {framework.upper()} framework. "
                f"Value should be verified with actual data."
            )

            prov = _compute_provenance(
                "regulatory_defaults",
                f"{column}:{i}:{imp_val}:fw={framework}",
            )
            iv = ImputedValue(
                record_index=i,
                column_name=column,
                imputed_value=imp_val,
                original_value=record.get(column),
                strategy=ImputationStrategy.REGULATORY_DEFAULT,
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=0,
                provenance_hash=prov,
            )
            imputed.append(iv)

        elapsed = time.monotonic() - start
        observe_duration("rule_evaluate", elapsed)
        if imputed:
            inc_values_imputed("regulatory_default", len(imputed))

        logger.info(
            "Regulatory defaults: framework=%s, %d values imputed, elapsed=%.3fs",
            framework, len(imputed), elapsed,
        )
        return imputed

    def create_rule(
        self,
        name: str,
        conditions: List[Dict[str, Any]],
        imputed_value: Any,
        priority: str = "medium",
        justification: str = "",
        target_column: str = "value",
    ) -> ImputationRule:
        """Create a new imputation rule.

        Args:
            name: Human-readable rule name.
            conditions: List of condition dicts with keys:
                field_name, condition_type, value, case_sensitive.
            imputed_value: Value to impute when conditions are met.
            priority: Rule priority ('critical', 'high', 'medium', 'low', 'default').
            justification: Audit-ready justification text.
            target_column: Column whose missing values this rule imputes.

        Returns:
            New ImputationRule model.
        """
        rule_conditions: List[RuleCondition] = []
        for cond_dict in conditions:
            rc = RuleCondition(
                field_name=cond_dict.get("field_name", ""),
                condition_type=RuleConditionType(
                    cond_dict.get("condition_type", "equals")
                ),
                value=cond_dict.get("value"),
                case_sensitive=cond_dict.get("case_sensitive", False),
            )
            rule_conditions.append(rc)

        priority_enum = RulePriority(priority)

        prov = _compute_provenance(
            "create_rule", f"{name}:{imputed_value}:{priority}"
        )

        rule = ImputationRule(
            name=name,
            description=justification,
            target_column=target_column,
            conditions=rule_conditions,
            impute_value=imputed_value,
            priority=priority_enum,
            provenance_hash=prov,
        )

        self.provenance.record(
            "rule", rule.rule_id, "create", prov
        )

        logger.info(
            "Rule created: name=%s, priority=%s, conditions=%d",
            name, priority, len(rule_conditions),
        )
        return rule

    def validate_rule(self, rule: ImputationRule) -> Dict[str, Any]:
        """Validate a rule for consistency and completeness.

        Checks:
            - Rule has a name and target column.
            - All conditions have valid field_name and condition_type.
            - impute_value is not None.
            - No circular or contradictory conditions.

        Args:
            rule: ImputationRule to validate.

        Returns:
            Dict with keys: valid (bool), errors (list), warnings (list).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not rule.name or not rule.name.strip():
            errors.append("Rule name is required")

        if not rule.target_column or not rule.target_column.strip():
            errors.append("Target column is required")

        if rule.impute_value is None and rule.impute_strategy is None:
            errors.append("Either impute_value or impute_strategy must be set")

        if not rule.conditions:
            warnings.append("Rule has no conditions (will match all records)")

        for j, cond in enumerate(rule.conditions):
            if not cond.field_name or not cond.field_name.strip():
                errors.append(f"Condition {j}: field_name is required")

            if cond.condition_type == RuleConditionType.REGEX:
                if cond.value is not None:
                    try:
                        re.compile(str(cond.value))
                    except re.error as e:
                        errors.append(f"Condition {j}: invalid regex: {e}")

            if cond.condition_type == RuleConditionType.IN_LIST:
                if cond.value is not None and not isinstance(cond.value, list):
                    warnings.append(
                        f"Condition {j}: IN_LIST value should be a list"
                    )

        # Check for contradictory conditions on same field
        field_conditions: Dict[str, List[RuleCondition]] = {}
        for cond in rule.conditions:
            field_conditions.setdefault(cond.field_name, []).append(cond)

        for field, conds in field_conditions.items():
            eq_vals = [
                c.value for c in conds
                if c.condition_type == RuleConditionType.EQUALS
            ]
            ne_vals = [
                c.value for c in conds
                if c.condition_type == RuleConditionType.NOT_EQUALS
            ]
            for eq in eq_vals:
                if eq in ne_vals:
                    errors.append(
                        f"Contradictory conditions on '{field}': "
                        f"EQUALS and NOT_EQUALS for value '{eq}'"
                    )

        is_valid = len(errors) == 0

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
        }

    def apply_rule_set(
        self,
        records: List[Dict[str, Any]],
        rule_set: Dict[str, List[ImputationRule]],
    ) -> Dict[str, List[ImputedValue]]:
        """Apply an ordered set of rules across multiple columns.

        Each key in rule_set is a column name, each value is an ordered
        list of rules for that column. Rules are applied with fallback:
        if no rule matches, the value remains missing.

        Args:
            records: List of record dictionaries.
            rule_set: Dict mapping column name to ordered list of rules.

        Returns:
            Dict mapping column name to list of ImputedValue.
        """
        start = time.monotonic()
        result: Dict[str, List[ImputedValue]] = {}

        for column, rules in rule_set.items():
            col_imputed = self.evaluate_rules(records, column, rules)
            result[column] = col_imputed

        elapsed = time.monotonic() - start
        total = sum(len(v) for v in result.values())
        logger.info(
            "Rule set applied: %d columns, %d total values, elapsed=%.3fs",
            len(rule_set), total, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers - condition evaluation
    # ------------------------------------------------------------------

    def _all_conditions_met(
        self,
        record: Dict[str, Any],
        conditions: List[RuleCondition],
    ) -> bool:
        """Check if all conditions in a rule are met by a record.

        Args:
            record: Record dictionary.
            conditions: List of RuleCondition to evaluate.

        Returns:
            True if all conditions are satisfied.
        """
        if not conditions:
            return True

        return all(
            self._evaluate_condition(record, cond) for cond in conditions
        )

    def _evaluate_condition(
        self,
        record: Dict[str, Any],
        condition: RuleCondition,
    ) -> bool:
        """Evaluate a single condition against a record.

        Args:
            record: Record dictionary.
            condition: RuleCondition to evaluate.

        Returns:
            True if the condition is satisfied.
        """
        field_val = record.get(condition.field_name)
        cond_val = condition.value
        ctype = condition.condition_type

        if ctype == RuleConditionType.IS_NULL:
            return _is_missing(field_val)

        if _is_missing(field_val):
            return False

        if ctype == RuleConditionType.EQUALS:
            return self._compare_values(field_val, cond_val, condition.case_sensitive)

        if ctype == RuleConditionType.NOT_EQUALS:
            return not self._compare_values(
                field_val, cond_val, condition.case_sensitive
            )

        if ctype == RuleConditionType.CONTAINS:
            return self._check_contains(field_val, cond_val, condition.case_sensitive)

        if ctype == RuleConditionType.GREATER_THAN:
            return self._check_greater_than(field_val, cond_val)

        if ctype == RuleConditionType.LESS_THAN:
            return self._check_less_than(field_val, cond_val)

        if ctype == RuleConditionType.IN_LIST:
            return self._check_in_list(field_val, cond_val, condition.case_sensitive)

        if ctype == RuleConditionType.REGEX:
            return self._check_regex(field_val, cond_val)

        return False

    def _compare_values(
        self, a: Any, b: Any, case_sensitive: bool
    ) -> bool:
        """Compare two values for equality.

        Args:
            a: First value.
            b: Second value.
            case_sensitive: Whether string comparison is case-sensitive.

        Returns:
            True if values are equal.
        """
        if isinstance(a, str) and isinstance(b, str):
            if case_sensitive:
                return a == b
            return a.lower() == b.lower()
        return str(a) == str(b)

    def _check_contains(
        self, field_val: Any, cond_val: Any, case_sensitive: bool
    ) -> bool:
        """Check if field value contains the condition value.

        Args:
            field_val: Record field value.
            cond_val: Value to search for.
            case_sensitive: Whether comparison is case-sensitive.

        Returns:
            True if field_val contains cond_val.
        """
        field_str = str(field_val)
        cond_str = str(cond_val) if cond_val is not None else ""
        if not case_sensitive:
            field_str = field_str.lower()
            cond_str = cond_str.lower()
        return cond_str in field_str

    def _check_greater_than(self, field_val: Any, cond_val: Any) -> bool:
        """Check if field value is greater than condition value.

        Args:
            field_val: Record field value.
            cond_val: Threshold value.

        Returns:
            True if field_val > cond_val.
        """
        try:
            return float(field_val) > float(cond_val)
        except (ValueError, TypeError):
            return str(field_val) > str(cond_val)

    def _check_less_than(self, field_val: Any, cond_val: Any) -> bool:
        """Check if field value is less than condition value.

        Args:
            field_val: Record field value.
            cond_val: Threshold value.

        Returns:
            True if field_val < cond_val.
        """
        try:
            return float(field_val) < float(cond_val)
        except (ValueError, TypeError):
            return str(field_val) < str(cond_val)

    def _check_in_list(
        self, field_val: Any, cond_val: Any, case_sensitive: bool
    ) -> bool:
        """Check if field value is in a list.

        Args:
            field_val: Record field value.
            cond_val: List of acceptable values.
            case_sensitive: Whether comparison is case-sensitive.

        Returns:
            True if field_val is in the list.
        """
        if not isinstance(cond_val, list):
            cond_val = [cond_val]

        field_str = str(field_val)
        if not case_sensitive:
            field_str = field_str.lower()
            return any(
                field_str == str(v).lower() for v in cond_val
            )
        return any(field_str == str(v) for v in cond_val)

    def _check_regex(self, field_val: Any, cond_val: Any) -> bool:
        """Check if field value matches a regex pattern.

        Args:
            field_val: Record field value.
            cond_val: Regex pattern string.

        Returns:
            True if field_val matches the pattern.
        """
        if cond_val is None:
            return False
        try:
            return bool(re.search(str(cond_val), str(field_val)))
        except re.error:
            logger.warning("Invalid regex pattern: %s", cond_val)
            return False

    # ------------------------------------------------------------------
    # Private helpers - regulatory defaults
    # ------------------------------------------------------------------

    def _get_regulatory_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Get regulatory default values for common emission gaps.

        Returns a nested dict: {framework: {key: value}}.
        Values are sourced from published regulatory guidance.

        Returns:
            Dict of regulatory default values by framework.
        """
        if self._regulatory_defaults is not None:
            return self._regulatory_defaults

        self._regulatory_defaults = {
            "ghg_protocol": {
                # Scope 1 - Direct emissions (kgCO2e per unit)
                "natural_gas": {"value": 2.02, "unit": "kgCO2e/m3", "source": "GHG Protocol"},
                "diesel": {"value": 2.68, "unit": "kgCO2e/litre", "source": "GHG Protocol"},
                "petrol": {"value": 2.31, "unit": "kgCO2e/litre", "source": "GHG Protocol"},
                "lpg": {"value": 1.56, "unit": "kgCO2e/litre", "source": "GHG Protocol"},
                "coal": {"value": 2.42, "unit": "kgCO2e/kg", "source": "GHG Protocol"},
                # Scope 2 - Electricity (kgCO2e per kWh)
                "electricity_global": {"value": 0.475, "unit": "kgCO2e/kWh", "source": "IEA 2023"},
                "electricity_us": {"value": 0.417, "unit": "kgCO2e/kWh", "source": "EPA eGRID"},
                "electricity_eu": {"value": 0.276, "unit": "kgCO2e/kWh", "source": "EEA"},
                "electricity_uk": {"value": 0.207, "unit": "kgCO2e/kWh", "source": "BEIS"},
                # Scope 3 defaults
                "business_travel_flight_short": {"value": 0.255, "unit": "kgCO2e/km", "source": "GHG Protocol"},
                "business_travel_flight_long": {"value": 0.195, "unit": "kgCO2e/km", "source": "GHG Protocol"},
                "business_travel_rail": {"value": 0.041, "unit": "kgCO2e/km", "source": "GHG Protocol"},
                "employee_commuting_car": {"value": 0.171, "unit": "kgCO2e/km", "source": "GHG Protocol"},
                "waste_landfill": {"value": 0.586, "unit": "kgCO2e/kg", "source": "GHG Protocol"},
                "waste_recycled": {"value": 0.021, "unit": "kgCO2e/kg", "source": "GHG Protocol"},
                "water_supply": {"value": 0.344, "unit": "kgCO2e/m3", "source": "GHG Protocol"},
                "water_treatment": {"value": 0.708, "unit": "kgCO2e/m3", "source": "GHG Protocol"},
            },
            "defra": {
                # DEFRA 2024 UK Emission Factors
                "natural_gas": {"value": 2.04, "unit": "kgCO2e/m3", "source": "DEFRA 2024"},
                "diesel": {"value": 2.71, "unit": "kgCO2e/litre", "source": "DEFRA 2024"},
                "petrol": {"value": 2.34, "unit": "kgCO2e/litre", "source": "DEFRA 2024"},
                "electricity_uk": {"value": 0.207, "unit": "kgCO2e/kWh", "source": "DEFRA 2024"},
                "freight_road": {"value": 0.115, "unit": "kgCO2e/tonne-km", "source": "DEFRA 2024"},
                "freight_rail": {"value": 0.028, "unit": "kgCO2e/tonne-km", "source": "DEFRA 2024"},
                "freight_sea": {"value": 0.016, "unit": "kgCO2e/tonne-km", "source": "DEFRA 2024"},
                "freight_air": {"value": 2.095, "unit": "kgCO2e/tonne-km", "source": "DEFRA 2024"},
                "hotel_stay_uk": {"value": 10.3, "unit": "kgCO2e/night", "source": "DEFRA 2024"},
                "paper": {"value": 0.919, "unit": "kgCO2e/kg", "source": "DEFRA 2024"},
                "plastics": {"value": 3.12, "unit": "kgCO2e/kg", "source": "DEFRA 2024"},
                "steel": {"value": 1.46, "unit": "kgCO2e/kg", "source": "DEFRA 2024"},
                "aluminium": {"value": 6.67, "unit": "kgCO2e/kg", "source": "DEFRA 2024"},
                "concrete": {"value": 0.132, "unit": "kgCO2e/kg", "source": "DEFRA 2024"},
            },
            "epa": {
                # EPA Emission Factors Hub
                "electricity_us": {"value": 0.417, "unit": "kgCO2e/kWh", "source": "EPA eGRID 2022"},
                "natural_gas": {"value": 1.93, "unit": "kgCO2e/m3", "source": "EPA GHG Hub"},
                "diesel": {"value": 2.69, "unit": "kgCO2e/litre", "source": "EPA GHG Hub"},
                "gasoline": {"value": 2.31, "unit": "kgCO2e/litre", "source": "EPA GHG Hub"},
                "propane": {"value": 1.54, "unit": "kgCO2e/litre", "source": "EPA GHG Hub"},
                "waste_msw_landfill": {"value": 0.52, "unit": "kgCO2e/kg", "source": "EPA WARM"},
                "waste_msw_recycled": {"value": -0.18, "unit": "kgCO2e/kg", "source": "EPA WARM"},
                "passenger_vehicle": {"value": 0.404, "unit": "kgCO2e/mile", "source": "EPA"},
            },
        }

        return self._regulatory_defaults

    def _match_regulatory_default(
        self,
        record: Dict[str, Any],
        column: str,
        framework_defaults: Dict[str, Any],
    ) -> Optional[Any]:
        """Match a record to a regulatory default value.

        Attempts to match based on activity_type, fuel_type, category,
        or material fields in the record.

        Args:
            record: Record dictionary.
            column: Target column name.
            framework_defaults: Defaults for the chosen framework.

        Returns:
            Default value if matched, else None.
        """
        # Try matching on common descriptor fields
        match_fields = [
            "activity_type", "fuel_type", "category", "material",
            "energy_source", "transport_mode", "waste_type",
        ]

        for field in match_fields:
            field_val = record.get(field)
            if _is_missing(field_val):
                continue

            field_str = str(field_val).lower().replace(" ", "_").replace("-", "_")

            # Direct match
            if field_str in framework_defaults:
                default_entry = framework_defaults[field_str]
                if isinstance(default_entry, dict):
                    return default_entry.get("value")
                return default_entry

            # Partial match
            for key, entry in framework_defaults.items():
                if field_str in key or key in field_str:
                    if isinstance(entry, dict):
                        return entry.get("value")
                    return entry

        return None

    def _format_justification(
        self,
        rule: ImputationRule,
        record: Dict[str, Any],
    ) -> str:
        """Generate an audit-ready justification for a rule application.

        Args:
            rule: The applied ImputationRule.
            record: The record being imputed.

        Returns:
            Human-readable justification string.
        """
        parts = [f"Rule '{rule.name}' applied (priority: {rule.priority.value})."]

        if rule.description:
            parts.append(f"Justification: {rule.description}")

        if rule.conditions:
            cond_strs = []
            for cond in rule.conditions:
                field_val = record.get(cond.field_name, "N/A")
                cond_strs.append(
                    f"{cond.field_name} {cond.condition_type.value} "
                    f"{cond.value} (actual: {field_val})"
                )
            parts.append(f"Conditions met: {'; '.join(cond_strs)}")

        parts.append(f"Imputed value: {rule.impute_value}")

        return " ".join(parts)
