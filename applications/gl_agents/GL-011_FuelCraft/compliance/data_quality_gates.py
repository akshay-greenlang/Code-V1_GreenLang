# -*- coding: utf-8 -*-
"""
DataQualityGates - Data quality validation for GL-011 FuelCraft.

This module implements critical data quality gates with fail-closed behavior
for inventory, calorific values, emission factors, and other critical feeds.
Implements timeliness validation and completeness checks.

Zero-Hallucination Governance:
- Fail-closed for critical missing data
- No free-form narrative generation
- All outputs traceable to raw records
- Deterministic validation rules

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Callable, Set
from pydantic import BaseModel, Field
from datetime import datetime, timezone, timedelta
from enum import Enum
from decimal import Decimal
import hashlib
import json
import logging
import statistics

logger = logging.getLogger(__name__)


class GateStatus(str, Enum):
    """Status of a quality gate."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class FallbackAction(str, Enum):
    """Action to take when data fails quality gate."""
    BLOCK = "block"           # Block processing entirely
    USE_FALLBACK = "fallback" # Use fallback value
    USE_LAST_GOOD = "last_good"  # Use last known good value
    INTERPOLATE = "interpolate"  # Interpolate from neighbors


class DataCategory(str, Enum):
    """Category of data for quality rules."""
    INVENTORY = "inventory"
    PRICE = "price"
    EMISSION_FACTOR = "emission_factor"
    CALORIFIC_VALUE = "calorific_value"
    CONTRACT = "contract"
    TELEMETRY = "telemetry"
    QUALITY_SPEC = "quality_spec"


class GateViolation(BaseModel):
    """Record of a quality gate violation."""
    violation_id: str = Field(...)
    gate_name: str = Field(...)
    data_category: DataCategory = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    field_name: str = Field(...)
    actual_value: Any = Field(...)
    expected_value: Optional[Any] = Field(None)
    threshold: Optional[float] = Field(None)
    message: str = Field(...)
    action_taken: FallbackAction = Field(...)
    fallback_value: Optional[Any] = Field(None)


class QualityGateResult(BaseModel):
    """Result of a quality gate check."""
    gate_id: str = Field(...)
    gate_name: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: GateStatus = Field(...)
    data_category: DataCategory = Field(...)
    records_checked: int = Field(0)
    records_passed: int = Field(0)
    records_failed: int = Field(0)
    violations: List[GateViolation] = Field(default_factory=list)
    blocked: bool = Field(False)
    message: str = Field(...)
    provenance_hash: str = Field(...)


class CriticalFeedValidator:
    """
    Validator for critical data feeds with fail-closed behavior.

    Critical feeds include:
    - Inventory levels (100% required)
    - Calorific values (100% required)
    - Emission factors (100% required)

    FAIL-CLOSED: Missing critical data blocks processing.
    """

    def __init__(self):
        """Initialize critical feed validator."""
        self._required_fields: Dict[DataCategory, Set[str]] = {
            DataCategory.INVENTORY: {"tank_id", "fuel_type", "volume_m3", "timestamp"},
            DataCategory.CALORIFIC_VALUE: {"fuel_type", "hhv_mj_kg", "lhv_mj_kg"},
            DataCategory.EMISSION_FACTOR: {"fuel_type", "co2_kg_per_mj", "source"},
        }
        self._last_good_values: Dict[str, Any] = {}
        logger.info("CriticalFeedValidator initialized with fail-closed behavior")

    def validate_inventory(
        self,
        inventory_data: List[Dict[str, Any]],
        required_tanks: Optional[List[str]] = None
    ) -> QualityGateResult:
        """
        Validate inventory data completeness.

        FAIL-CLOSED: Blocks if any required tank is missing.
        """
        gate_id = hashlib.sha256(
            f"inventory|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        records_passed = 0
        records_failed = 0

        required_fields = self._required_fields[DataCategory.INVENTORY]

        for record in inventory_data:
            missing_fields = required_fields - set(record.keys())

            if missing_fields:
                violations.append(GateViolation(
                    violation_id=f"INV-{records_failed}",
                    gate_name="Inventory Completeness",
                    data_category=DataCategory.INVENTORY,
                    field_name=",".join(missing_fields),
                    actual_value=None,
                    message=f"Missing required fields: {missing_fields}",
                    action_taken=FallbackAction.BLOCK
                ))
                records_failed += 1
            elif record.get("volume_m3") is None or record.get("volume_m3") < 0:
                violations.append(GateViolation(
                    violation_id=f"INV-{records_failed}",
                    gate_name="Inventory Volume",
                    data_category=DataCategory.INVENTORY,
                    field_name="volume_m3",
                    actual_value=record.get("volume_m3"),
                    message="Invalid volume value",
                    action_taken=FallbackAction.BLOCK
                ))
                records_failed += 1
            else:
                records_passed += 1
                # Cache good value
                self._last_good_values[f"inventory_{record.get('tank_id')}"] = record

        # Check required tanks if specified
        if required_tanks:
            present_tanks = {r.get("tank_id") for r in inventory_data}
            missing_tanks = set(required_tanks) - present_tanks

            for tank in missing_tanks:
                violations.append(GateViolation(
                    violation_id=f"INV-TANK-{tank}",
                    gate_name="Required Tank Missing",
                    data_category=DataCategory.INVENTORY,
                    field_name="tank_id",
                    actual_value=None,
                    expected_value=tank,
                    message=f"Required tank {tank} has no inventory data",
                    action_taken=FallbackAction.BLOCK
                ))
                records_failed += 1

        # FAIL-CLOSED: Any violation blocks
        blocked = len(violations) > 0
        status = GateStatus.FAIL if blocked else GateStatus.PASS

        return QualityGateResult(
            gate_id=gate_id,
            gate_name="Critical Inventory Feed",
            status=status,
            data_category=DataCategory.INVENTORY,
            records_checked=len(inventory_data),
            records_passed=records_passed,
            records_failed=records_failed,
            violations=violations,
            blocked=blocked,
            message=f"Inventory validation: {records_passed}/{len(inventory_data)} passed" if not blocked else f"BLOCKED: {len(violations)} critical violations",
            provenance_hash=hashlib.sha256(
                json.dumps({"gate_id": gate_id, "status": status.value, "blocked": blocked}, sort_keys=True).encode()
            ).hexdigest()
        )

    def validate_calorific_values(
        self,
        cv_data: List[Dict[str, Any]],
        required_fuel_types: Optional[List[str]] = None
    ) -> QualityGateResult:
        """
        Validate calorific value data.

        FAIL-CLOSED: Blocks if calorific values are missing for required fuels.
        """
        gate_id = hashlib.sha256(
            f"cv|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        records_passed = 0
        records_failed = 0

        required_fields = self._required_fields[DataCategory.CALORIFIC_VALUE]

        for record in cv_data:
            missing_fields = required_fields - set(record.keys())

            if missing_fields:
                violations.append(GateViolation(
                    violation_id=f"CV-{records_failed}",
                    gate_name="Calorific Value Completeness",
                    data_category=DataCategory.CALORIFIC_VALUE,
                    field_name=",".join(missing_fields),
                    actual_value=None,
                    message=f"Missing required fields: {missing_fields}",
                    action_taken=FallbackAction.BLOCK
                ))
                records_failed += 1
            else:
                hhv = record.get("hhv_mj_kg", 0)
                lhv = record.get("lhv_mj_kg", 0)

                if hhv <= 0 or lhv <= 0:
                    violations.append(GateViolation(
                        violation_id=f"CV-{records_failed}",
                        gate_name="Calorific Value Range",
                        data_category=DataCategory.CALORIFIC_VALUE,
                        field_name="hhv_mj_kg/lhv_mj_kg",
                        actual_value={"hhv": hhv, "lhv": lhv},
                        message="Calorific values must be positive",
                        action_taken=FallbackAction.BLOCK
                    ))
                    records_failed += 1
                elif lhv > hhv:
                    violations.append(GateViolation(
                        violation_id=f"CV-{records_failed}",
                        gate_name="Calorific Value Consistency",
                        data_category=DataCategory.CALORIFIC_VALUE,
                        field_name="lhv_mj_kg",
                        actual_value={"hhv": hhv, "lhv": lhv},
                        message="LHV cannot exceed HHV",
                        action_taken=FallbackAction.BLOCK
                    ))
                    records_failed += 1
                else:
                    records_passed += 1

        # Check required fuel types
        if required_fuel_types:
            present_fuels = {r.get("fuel_type") for r in cv_data}
            missing_fuels = set(required_fuel_types) - present_fuels

            for fuel in missing_fuels:
                violations.append(GateViolation(
                    violation_id=f"CV-FUEL-{fuel}",
                    gate_name="Required Fuel CV Missing",
                    data_category=DataCategory.CALORIFIC_VALUE,
                    field_name="fuel_type",
                    actual_value=None,
                    expected_value=fuel,
                    message=f"No calorific value data for required fuel: {fuel}",
                    action_taken=FallbackAction.BLOCK
                ))
                records_failed += 1

        blocked = len(violations) > 0
        status = GateStatus.FAIL if blocked else GateStatus.PASS

        return QualityGateResult(
            gate_id=gate_id,
            gate_name="Critical Calorific Value Feed",
            status=status,
            data_category=DataCategory.CALORIFIC_VALUE,
            records_checked=len(cv_data),
            records_passed=records_passed,
            records_failed=records_failed,
            violations=violations,
            blocked=blocked,
            message=f"CV validation: {records_passed} passed" if not blocked else f"BLOCKED: {len(violations)} critical violations",
            provenance_hash=hashlib.sha256(
                json.dumps({"gate_id": gate_id, "status": status.value}, sort_keys=True).encode()
            ).hexdigest()
        )

    def validate_emission_factors(
        self,
        ef_data: List[Dict[str, Any]],
        required_fuel_types: Optional[List[str]] = None
    ) -> QualityGateResult:
        """
        Validate emission factor data.

        FAIL-CLOSED: Blocks if emission factors are missing for required fuels.
        """
        gate_id = hashlib.sha256(
            f"ef|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        records_passed = 0
        records_failed = 0

        required_fields = self._required_fields[DataCategory.EMISSION_FACTOR]

        for record in ef_data:
            missing_fields = required_fields - set(record.keys())

            if missing_fields:
                violations.append(GateViolation(
                    violation_id=f"EF-{records_failed}",
                    gate_name="Emission Factor Completeness",
                    data_category=DataCategory.EMISSION_FACTOR,
                    field_name=",".join(missing_fields),
                    actual_value=None,
                    message=f"Missing required fields: {missing_fields}",
                    action_taken=FallbackAction.BLOCK
                ))
                records_failed += 1
            else:
                ef = record.get("co2_kg_per_mj", 0)
                if ef <= 0:
                    violations.append(GateViolation(
                        violation_id=f"EF-{records_failed}",
                        gate_name="Emission Factor Range",
                        data_category=DataCategory.EMISSION_FACTOR,
                        field_name="co2_kg_per_mj",
                        actual_value=ef,
                        message="Emission factor must be positive",
                        action_taken=FallbackAction.BLOCK
                    ))
                    records_failed += 1
                else:
                    records_passed += 1

        # Check required fuel types
        if required_fuel_types:
            present_fuels = {r.get("fuel_type") for r in ef_data}
            missing_fuels = set(required_fuel_types) - present_fuels

            for fuel in missing_fuels:
                violations.append(GateViolation(
                    violation_id=f"EF-FUEL-{fuel}",
                    gate_name="Required Fuel EF Missing",
                    data_category=DataCategory.EMISSION_FACTOR,
                    field_name="fuel_type",
                    actual_value=None,
                    expected_value=fuel,
                    message=f"No emission factor for required fuel: {fuel}",
                    action_taken=FallbackAction.BLOCK
                ))
                records_failed += 1

        blocked = len(violations) > 0
        status = GateStatus.FAIL if blocked else GateStatus.PASS

        return QualityGateResult(
            gate_id=gate_id,
            gate_name="Critical Emission Factor Feed",
            status=status,
            data_category=DataCategory.EMISSION_FACTOR,
            records_checked=len(ef_data),
            records_passed=records_passed,
            records_failed=records_failed,
            violations=violations,
            blocked=blocked,
            message=f"EF validation: {records_passed} passed" if not blocked else f"BLOCKED: {len(violations)} critical violations",
            provenance_hash=hashlib.sha256(
                json.dumps({"gate_id": gate_id, "status": status.value}, sort_keys=True).encode()
            ).hexdigest()
        )


class TimelinessValidator:
    """
    Validator for data timeliness requirements.

    - Telemetry: 10-minute max delay
    - Prices: 24-hour max delay
    - Inventory: 1-hour max delay
    """

    def __init__(
        self,
        telemetry_max_delay_seconds: int = 600,
        price_max_delay_seconds: int = 86400,
        inventory_max_delay_seconds: int = 3600
    ):
        """Initialize timeliness validator."""
        self._max_delays = {
            DataCategory.TELEMETRY: telemetry_max_delay_seconds,
            DataCategory.PRICE: price_max_delay_seconds,
            DataCategory.INVENTORY: inventory_max_delay_seconds,
        }
        logger.info(f"TimelinessValidator initialized: telemetry={telemetry_max_delay_seconds}s")

    def validate_timeliness(
        self,
        data: List[Dict[str, Any]],
        category: DataCategory,
        timestamp_field: str = "timestamp"
    ) -> QualityGateResult:
        """Validate data timeliness."""
        gate_id = hashlib.sha256(
            f"timeliness|{category.value}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        records_passed = 0
        records_failed = 0

        now = datetime.now(timezone.utc)
        max_delay = self._max_delays.get(category, 3600)

        for idx, record in enumerate(data):
            ts_value = record.get(timestamp_field)

            if ts_value is None:
                violations.append(GateViolation(
                    violation_id=f"TIME-{idx}",
                    gate_name="Timeliness",
                    data_category=category,
                    field_name=timestamp_field,
                    actual_value=None,
                    message="Missing timestamp",
                    action_taken=FallbackAction.BLOCK if category == DataCategory.TELEMETRY else FallbackAction.USE_LAST_GOOD
                ))
                records_failed += 1
                continue

            if isinstance(ts_value, str):
                try:
                    ts = datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
                except ValueError:
                    violations.append(GateViolation(
                        violation_id=f"TIME-{idx}",
                        gate_name="Timeliness",
                        data_category=category,
                        field_name=timestamp_field,
                        actual_value=ts_value,
                        message="Invalid timestamp format",
                        action_taken=FallbackAction.BLOCK
                    ))
                    records_failed += 1
                    continue
            else:
                ts = ts_value

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            delay_seconds = (now - ts).total_seconds()

            if delay_seconds > max_delay:
                violations.append(GateViolation(
                    violation_id=f"TIME-{idx}",
                    gate_name="Timeliness",
                    data_category=category,
                    field_name=timestamp_field,
                    actual_value=delay_seconds,
                    threshold=float(max_delay),
                    message=f"Data is {delay_seconds:.0f}s old (max: {max_delay}s)",
                    action_taken=FallbackAction.BLOCK if category == DataCategory.TELEMETRY else FallbackAction.WARNING
                ))
                records_failed += 1
            else:
                records_passed += 1

        telemetry_violations = [v for v in violations if v.action_taken == FallbackAction.BLOCK]
        blocked = len(telemetry_violations) > 0

        status = GateStatus.FAIL if blocked else (GateStatus.WARNING if violations else GateStatus.PASS)

        return QualityGateResult(
            gate_id=gate_id,
            gate_name=f"Timeliness ({category.value})",
            status=status,
            data_category=category,
            records_checked=len(data),
            records_passed=records_passed,
            records_failed=records_failed,
            violations=violations,
            blocked=blocked,
            message=f"Timeliness: {records_passed}/{len(data)} within limit" if not blocked else f"BLOCKED: Stale {category.value} data",
            provenance_hash=hashlib.sha256(
                json.dumps({"gate_id": gate_id, "status": status.value}, sort_keys=True).encode()
            ).hexdigest()
        )


class CompletenessValidator:
    """
    Validator for data completeness requirements.

    - Prices: 95% minimum completeness
    - Inventory: 100% completeness
    - Contracts: 100% completeness
    """

    def __init__(
        self,
        price_min_completeness_pct: float = 95.0,
        inventory_min_completeness_pct: float = 100.0,
        contract_min_completeness_pct: float = 100.0
    ):
        """Initialize completeness validator."""
        self._min_completeness = {
            DataCategory.PRICE: price_min_completeness_pct,
            DataCategory.INVENTORY: inventory_min_completeness_pct,
            DataCategory.CONTRACT: contract_min_completeness_pct,
        }
        logger.info(f"CompletenessValidator initialized: inventory={inventory_min_completeness_pct}%")

    def validate_completeness(
        self,
        data: List[Dict[str, Any]],
        category: DataCategory,
        required_fields: List[str],
        identifier_field: str = "id"
    ) -> QualityGateResult:
        """Validate data completeness."""
        gate_id = hashlib.sha256(
            f"completeness|{category.value}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        records_passed = 0
        records_failed = 0

        min_completeness = self._min_completeness.get(category, 95.0)

        for idx, record in enumerate(data):
            record_id = record.get(identifier_field, f"record_{idx}")
            missing_fields = []

            for field in required_fields:
                value = record.get(field)
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    missing_fields.append(field)

            if missing_fields:
                violations.append(GateViolation(
                    violation_id=f"COMP-{record_id}",
                    gate_name="Completeness",
                    data_category=category,
                    field_name=",".join(missing_fields),
                    actual_value=None,
                    message=f"Record {record_id} missing fields: {missing_fields}",
                    action_taken=FallbackAction.BLOCK if min_completeness == 100.0 else FallbackAction.USE_FALLBACK
                ))
                records_failed += 1
            else:
                records_passed += 1

        total_records = len(data)
        completeness_pct = (records_passed / total_records * 100.0) if total_records > 0 else 0.0

        blocked = completeness_pct < min_completeness

        status = GateStatus.FAIL if blocked else (GateStatus.WARNING if violations else GateStatus.PASS)

        return QualityGateResult(
            gate_id=gate_id,
            gate_name=f"Completeness ({category.value})",
            status=status,
            data_category=category,
            records_checked=total_records,
            records_passed=records_passed,
            records_failed=records_failed,
            violations=violations,
            blocked=blocked,
            message=f"Completeness: {completeness_pct:.1f}% (min: {min_completeness}%)" if not blocked else f"BLOCKED: {completeness_pct:.1f}% < {min_completeness}%",
            provenance_hash=hashlib.sha256(
                json.dumps({"gate_id": gate_id, "completeness": completeness_pct}, sort_keys=True).encode()
            ).hexdigest()
        )


class OutlierDetector:
    """
    Detector for statistical outliers with fallback policy.

    Uses IQR method for outlier detection with configurable fallback.
    """

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        fallback_action: FallbackAction = FallbackAction.USE_LAST_GOOD
    ):
        """Initialize outlier detector."""
        self._iqr_multiplier = iqr_multiplier
        self._fallback_action = fallback_action
        self._historical_values: Dict[str, List[float]] = {}
        logger.info(f"OutlierDetector initialized: IQR multiplier={iqr_multiplier}")

    def detect_outliers(
        self,
        data: List[Dict[str, Any]],
        value_field: str,
        category: DataCategory,
        identifier_field: str = "id"
    ) -> QualityGateResult:
        """Detect statistical outliers."""
        gate_id = hashlib.sha256(
            f"outlier|{category.value}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        records_passed = 0
        records_failed = 0

        values = [r.get(value_field) for r in data if r.get(value_field) is not None]

        if len(values) < 4:
            return QualityGateResult(
                gate_id=gate_id,
                gate_name=f"Outlier Detection ({category.value})",
                status=GateStatus.SKIPPED,
                data_category=category,
                records_checked=len(data),
                records_passed=len(data),
                records_failed=0,
                violations=[],
                blocked=False,
                message="Insufficient data for outlier detection (need 4+ records)",
                provenance_hash=hashlib.sha256(
                    json.dumps({"gate_id": gate_id, "status": "skipped"}, sort_keys=True).encode()
                ).hexdigest()
            )

        sorted_values = sorted(values)
        q1_idx = len(sorted_values) // 4
        q3_idx = (3 * len(sorted_values)) // 4
        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1

        lower_bound = q1 - (self._iqr_multiplier * iqr)
        upper_bound = q3 + (self._iqr_multiplier * iqr)

        for idx, record in enumerate(data):
            value = record.get(value_field)
            record_id = record.get(identifier_field, f"record_{idx}")

            if value is None:
                continue

            if value < lower_bound or value > upper_bound:
                # Determine fallback value
                hist_key = f"{category.value}_{value_field}"
                hist_values = self._historical_values.get(hist_key, [])
                fallback_value = statistics.mean(hist_values) if hist_values else None

                violations.append(GateViolation(
                    violation_id=f"OUT-{record_id}",
                    gate_name="Outlier Detection",
                    data_category=category,
                    field_name=value_field,
                    actual_value=value,
                    threshold=upper_bound if value > upper_bound else lower_bound,
                    message=f"Outlier detected: {value} outside [{lower_bound:.2f}, {upper_bound:.2f}]",
                    action_taken=self._fallback_action,
                    fallback_value=fallback_value
                ))
                records_failed += 1
            else:
                records_passed += 1
                # Store good value for history
                hist_key = f"{category.value}_{value_field}"
                if hist_key not in self._historical_values:
                    self._historical_values[hist_key] = []
                self._historical_values[hist_key].append(value)
                if len(self._historical_values[hist_key]) > 100:
                    self._historical_values[hist_key] = self._historical_values[hist_key][-100:]

        blocked = self._fallback_action == FallbackAction.BLOCK and len(violations) > 0

        status = GateStatus.FAIL if blocked else (GateStatus.WARNING if violations else GateStatus.PASS)

        return QualityGateResult(
            gate_id=gate_id,
            gate_name=f"Outlier Detection ({category.value})",
            status=status,
            data_category=category,
            records_checked=len(data),
            records_passed=records_passed,
            records_failed=records_failed,
            violations=violations,
            blocked=blocked,
            message=f"Outliers: {len(violations)} detected ({self._fallback_action.value})",
            provenance_hash=hashlib.sha256(
                json.dumps({"gate_id": gate_id, "outliers": len(violations)}, sort_keys=True).encode()
            ).hexdigest()
        )


class DataQualityGateRunner:
    """
    Unified runner for all data quality gates.

    Executes all quality gates and produces consolidated results
    with fail-closed behavior for critical feeds.
    """

    def __init__(self):
        """Initialize quality gate runner."""
        self.critical_feed_validator = CriticalFeedValidator()
        self.timeliness_validator = TimelinessValidator()
        self.completeness_validator = CompletenessValidator()
        self.outlier_detector = OutlierDetector()
        logger.info("DataQualityGateRunner initialized")

    def run_all_gates(
        self,
        inventory_data: Optional[List[Dict[str, Any]]] = None,
        price_data: Optional[List[Dict[str, Any]]] = None,
        cv_data: Optional[List[Dict[str, Any]]] = None,
        ef_data: Optional[List[Dict[str, Any]]] = None,
        required_fuel_types: Optional[List[str]] = None,
        required_tanks: Optional[List[str]] = None
    ) -> Dict[str, QualityGateResult]:
        """
        Run all quality gates.

        Returns dict of gate results. Processing should be blocked
        if any result has blocked=True.
        """
        results = {}

        if inventory_data is not None:
            results["inventory_critical"] = self.critical_feed_validator.validate_inventory(
                inventory_data, required_tanks
            )
            results["inventory_timeliness"] = self.timeliness_validator.validate_timeliness(
                inventory_data, DataCategory.INVENTORY
            )

        if cv_data is not None:
            results["cv_critical"] = self.critical_feed_validator.validate_calorific_values(
                cv_data, required_fuel_types
            )

        if ef_data is not None:
            results["ef_critical"] = self.critical_feed_validator.validate_emission_factors(
                ef_data, required_fuel_types
            )

        if price_data is not None:
            results["price_completeness"] = self.completeness_validator.validate_completeness(
                price_data, DataCategory.PRICE,
                required_fields=["fuel_type", "price_per_mt", "timestamp"]
            )
            results["price_timeliness"] = self.timeliness_validator.validate_timeliness(
                price_data, DataCategory.PRICE
            )
            results["price_outliers"] = self.outlier_detector.detect_outliers(
                price_data, "price_per_mt", DataCategory.PRICE
            )

        return results

    def is_blocked(self, results: Dict[str, QualityGateResult]) -> tuple[bool, List[str]]:
        """
        Check if any gate is blocking.

        Returns (is_blocked, list of blocking gate names).
        """
        blocking_gates = [
            name for name, result in results.items()
            if result.blocked
        ]
        return len(blocking_gates) > 0, blocking_gates
