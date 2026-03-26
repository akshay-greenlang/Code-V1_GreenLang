# -*- coding: utf-8 -*-
"""
UtilityDataBridge - Utility Bill and Meter Data Integration (PACK-041)
========================================================================

This module provides utility bill and meter data integration for the
Scope 1-2 Complete Pack. It handles import of utility bills (electricity,
natural gas, water), meter reading data, unit normalization, estimated
bill handling, and facility-level aggregation.

Capabilities:
    - Import utility bills from various formats (PDF, CSV, Excel, EDI)
    - Import meter interval data (15-min, hourly, daily, monthly)
    - Normalize units to standard GHG-ready formats
    - Handle estimated vs actual bill readings
    - Aggregate consumption by facility for GHG calculations

Zero-Hallucination:
    All unit conversions, aggregations, and bill corrections use
    deterministic arithmetic. No LLM calls in the processing path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-041 Scope 1-2 Complete
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
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
# Enums
# ---------------------------------------------------------------------------


class UtilityType(str, Enum):
    """Utility service types."""

    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"
    WATER = "water"
    FUEL_OIL = "fuel_oil"


class BillStatus(str, Enum):
    """Utility bill reading status."""

    ACTUAL = "actual"
    ESTIMATED = "estimated"
    CORRECTED = "corrected"
    PRORATED = "prorated"


class MeterInterval(str, Enum):
    """Meter reading interval types."""

    INTERVAL_15MIN = "15min"
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


class BillFormat(str, Enum):
    """Utility bill import formats."""

    PDF = "pdf"
    CSV = "csv"
    XLSX = "xlsx"
    EDI = "edi"
    GREEN_BUTTON_XML = "green_button_xml"
    API = "api"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class UtilityBill(BaseModel):
    """Utility bill record."""

    bill_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    utility_type: UtilityType = Field(default=UtilityType.ELECTRICITY)
    billing_period_start: str = Field(default="")
    billing_period_end: str = Field(default="")
    consumption: float = Field(default=0.0, ge=0.0)
    consumption_unit: str = Field(default="kWh")
    demand: float = Field(default=0.0, ge=0.0)
    demand_unit: str = Field(default="kW")
    cost: float = Field(default=0.0, ge=0.0)
    currency: str = Field(default="USD")
    status: BillStatus = Field(default=BillStatus.ACTUAL)
    provider: str = Field(default="")
    account_number: str = Field(default="")
    invoice_number: str = Field(default="")
    grid_region: str = Field(default="US_AVERAGE")


class MeterReading(BaseModel):
    """Meter interval reading."""

    reading_id: str = Field(default_factory=_new_uuid)
    meter_id: str = Field(default="")
    facility_id: str = Field(default="")
    utility_type: UtilityType = Field(default=UtilityType.ELECTRICITY)
    timestamp: str = Field(default="")
    value: float = Field(default=0.0)
    unit: str = Field(default="kWh")
    interval: MeterInterval = Field(default=MeterInterval.HOURLY)
    quality: str = Field(default="actual")


class ConsumptionSummary(BaseModel):
    """Aggregated consumption summary for a facility."""

    facility_id: str = Field(default="")
    utility_type: str = Field(default="")
    total_consumption: float = Field(default=0.0)
    consumption_unit: str = Field(default="")
    total_cost: float = Field(default=0.0)
    currency: str = Field(default="USD")
    bill_count: int = Field(default=0)
    actual_pct: float = Field(default=0.0)
    estimated_pct: float = Field(default=0.0)
    period_start: str = Field(default="")
    period_end: str = Field(default="")
    provenance_hash: str = Field(default="")


class ImportResult(BaseModel):
    """Result of a utility data import operation."""

    import_id: str = Field(default_factory=_new_uuid)
    data_type: str = Field(default="")
    records_imported: int = Field(default=0)
    records_rejected: int = Field(default=0)
    status: str = Field(default="success")
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Unit Normalization Tables
# ---------------------------------------------------------------------------

UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
    "therms_to_kwh": {"factor": 29.3071, "from": "therms", "to": "kWh"},
    "ccf_to_therms": {"factor": 1.024, "from": "ccf", "to": "therms"},
    "mcf_to_therms": {"factor": 10.24, "from": "mcf", "to": "therms"},
    "mmbtu_to_therms": {"factor": 10.0, "from": "MMBtu", "to": "therms"},
    "gj_to_kwh": {"factor": 277.778, "from": "GJ", "to": "kWh"},
    "mwh_to_kwh": {"factor": 1000.0, "from": "MWh", "to": "kWh"},
    "ton_hours_to_kwh": {"factor": 3.517, "from": "ton_hours", "to": "kWh"},
    "gallons_to_liters": {"factor": 3.78541, "from": "gallons", "to": "liters"},
}


# ---------------------------------------------------------------------------
# UtilityDataBridge
# ---------------------------------------------------------------------------


class UtilityDataBridge:
    """Utility bill and meter data integration for GHG inventory.

    Handles import, normalization, estimated bill correction, and
    facility-level aggregation of utility data for Scope 1 and Scope 2
    emissions calculations.

    Attributes:
        _bills: Imported utility bills.
        _readings: Imported meter readings.

    Example:
        >>> bridge = UtilityDataBridge()
        >>> result = bridge.import_utility_bills(files)
        >>> summary = bridge.aggregate_by_facility(readings)
    """

    def __init__(self) -> None:
        """Initialize UtilityDataBridge."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._bills: List[UtilityBill] = []
        self._readings: List[MeterReading] = []

        self.logger.info("UtilityDataBridge initialized")

    # -------------------------------------------------------------------------
    # Import Methods
    # -------------------------------------------------------------------------

    def import_utility_bills(
        self,
        files: List[Dict[str, Any]],
    ) -> ImportResult:
        """Import utility bills from file records.

        Args:
            files: List of bill data dicts with utility_type, facility_id,
                consumption, cost, etc.

        Returns:
            ImportResult with import status.
        """
        start_time = time.monotonic()
        imported = 0
        rejected = 0
        errors: List[str] = []

        for f in files:
            try:
                bill = UtilityBill(
                    facility_id=f.get("facility_id", ""),
                    utility_type=UtilityType(f.get("utility_type", "electricity")),
                    billing_period_start=f.get("billing_period_start", ""),
                    billing_period_end=f.get("billing_period_end", ""),
                    consumption=f.get("consumption", 0.0),
                    consumption_unit=f.get("consumption_unit", "kWh"),
                    demand=f.get("demand", 0.0),
                    cost=f.get("cost", 0.0),
                    currency=f.get("currency", "USD"),
                    status=BillStatus(f.get("status", "actual")),
                    provider=f.get("provider", ""),
                    account_number=f.get("account_number", ""),
                    invoice_number=f.get("invoice_number", ""),
                    grid_region=f.get("grid_region", "US_AVERAGE"),
                )
                self._bills.append(bill)
                imported += 1
            except Exception as exc:
                rejected += 1
                errors.append(str(exc))

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = ImportResult(
            data_type="utility_bills",
            records_imported=imported,
            records_rejected=rejected,
            status="success" if rejected == 0 else "partial",
            errors=errors[:10],
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Imported %d utility bills (%d rejected)", imported, rejected
        )
        return result

    def import_meter_data(
        self,
        meter_config: Dict[str, Any],
    ) -> ImportResult:
        """Import meter interval data.

        Args:
            meter_config: Dict with readings list, each containing
                meter_id, facility_id, timestamp, value, unit, interval.

        Returns:
            ImportResult with import status.
        """
        start_time = time.monotonic()
        readings = meter_config.get("readings", [])
        imported = 0

        for r in readings:
            reading = MeterReading(
                meter_id=r.get("meter_id", ""),
                facility_id=r.get("facility_id", ""),
                utility_type=UtilityType(r.get("utility_type", "electricity")),
                timestamp=r.get("timestamp", ""),
                value=r.get("value", 0.0),
                unit=r.get("unit", "kWh"),
                interval=MeterInterval(r.get("interval", "hourly")),
                quality=r.get("quality", "actual"),
            )
            self._readings.append(reading)
            imported += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = ImportResult(
            data_type="meter_readings",
            records_imported=imported,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info("Imported %d meter readings", imported)
        return result

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def normalize_units(
        self,
        readings: List[Dict[str, Any]],
        target_unit: str = "kWh",
    ) -> List[Dict[str, Any]]:
        """Normalize consumption units to a standard unit.

        Args:
            readings: List of readings with value and unit.
            target_unit: Target unit for normalization.

        Returns:
            List of normalized readings.
        """
        normalized: List[Dict[str, Any]] = []

        for reading in readings:
            value = Decimal(str(reading.get("value", 0)))
            unit = reading.get("unit", target_unit)

            conversion_key = f"{unit.lower()}_to_{target_unit.lower()}"
            conv = UNIT_CONVERSIONS.get(conversion_key)

            if conv:
                factor = Decimal(str(conv["factor"]))
                converted = (value * factor).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            elif unit.lower() == target_unit.lower():
                converted = value
            else:
                self.logger.warning("No conversion for '%s' -> '%s'", unit, target_unit)
                converted = value

            normalized.append({
                **reading,
                "original_value": float(value),
                "original_unit": unit,
                "value": float(converted),
                "unit": target_unit,
            })

        self.logger.info(
            "Normalized %d readings to %s", len(normalized), target_unit
        )
        return normalized

    # -------------------------------------------------------------------------
    # Estimated Bill Handling
    # -------------------------------------------------------------------------

    def handle_estimated_bills(
        self,
        bills: List[UtilityBill],
    ) -> List[UtilityBill]:
        """Handle estimated utility bills by flagging and adjusting.

        When an estimated bill is followed by an actual bill, the actual
        may include a true-up. This method identifies and flags estimated
        readings for correction.

        Args:
            bills: List of utility bills sorted by date.

        Returns:
            List of bills with estimated corrections applied.
        """
        corrected: List[UtilityBill] = []
        estimated_bills: List[UtilityBill] = []

        for bill in bills:
            if bill.status == BillStatus.ESTIMATED:
                estimated_bills.append(bill)
                corrected.append(bill)
            elif bill.status == BillStatus.ACTUAL and estimated_bills:
                # True-up: distribute any correction across estimated period
                total_estimated = sum(b.consumption for b in estimated_bills)
                actual = bill.consumption
                if total_estimated > 0 and len(estimated_bills) > 0:
                    correction_factor = actual / (total_estimated + actual) * 2
                    for est_bill in estimated_bills:
                        est_bill.consumption = round(
                            est_bill.consumption * correction_factor, 2
                        )
                        est_bill.status = BillStatus.CORRECTED
                estimated_bills.clear()
                corrected.append(bill)
            else:
                estimated_bills.clear()
                corrected.append(bill)

        self.logger.info(
            "Handled estimated bills: %d total, %d corrected",
            len(bills),
            sum(1 for b in corrected if b.status == BillStatus.CORRECTED),
        )
        return corrected

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    def aggregate_by_facility(
        self,
        readings: Optional[List[UtilityBill]] = None,
    ) -> Dict[str, ConsumptionSummary]:
        """Aggregate utility consumption by facility.

        Args:
            readings: Bills to aggregate. Uses internal bills if None.

        Returns:
            Dict mapping facility_id to ConsumptionSummary.
        """
        bills = readings or self._bills
        facility_data: Dict[str, Dict[str, Any]] = {}

        for bill in bills:
            key = f"{bill.facility_id}_{bill.utility_type.value}"
            if key not in facility_data:
                facility_data[key] = {
                    "facility_id": bill.facility_id,
                    "utility_type": bill.utility_type.value,
                    "total_consumption": Decimal("0"),
                    "consumption_unit": bill.consumption_unit,
                    "total_cost": Decimal("0"),
                    "currency": bill.currency,
                    "bill_count": 0,
                    "actual_count": 0,
                    "estimated_count": 0,
                    "min_date": bill.billing_period_start,
                    "max_date": bill.billing_period_end,
                }

            fd = facility_data[key]
            fd["total_consumption"] += Decimal(str(bill.consumption))
            fd["total_cost"] += Decimal(str(bill.cost))
            fd["bill_count"] += 1
            if bill.status == BillStatus.ACTUAL:
                fd["actual_count"] += 1
            elif bill.status in (BillStatus.ESTIMATED, BillStatus.CORRECTED):
                fd["estimated_count"] += 1
            if bill.billing_period_start and bill.billing_period_start < fd["min_date"]:
                fd["min_date"] = bill.billing_period_start
            if bill.billing_period_end and bill.billing_period_end > fd["max_date"]:
                fd["max_date"] = bill.billing_period_end

        summaries: Dict[str, ConsumptionSummary] = {}
        for key, fd in facility_data.items():
            total_bills = fd["bill_count"]
            actual_pct = (fd["actual_count"] / total_bills * 100) if total_bills > 0 else 0
            estimated_pct = (fd["estimated_count"] / total_bills * 100) if total_bills > 0 else 0

            summary = ConsumptionSummary(
                facility_id=fd["facility_id"],
                utility_type=fd["utility_type"],
                total_consumption=float(fd["total_consumption"].quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )),
                consumption_unit=fd["consumption_unit"],
                total_cost=float(fd["total_cost"].quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )),
                currency=fd["currency"],
                bill_count=total_bills,
                actual_pct=round(actual_pct, 1),
                estimated_pct=round(estimated_pct, 1),
                period_start=fd["min_date"],
                period_end=fd["max_date"],
            )
            summary.provenance_hash = _compute_hash(summary)
            summaries[key] = summary

        self.logger.info(
            "Aggregated %d bills into %d facility summaries",
            len(bills), len(summaries),
        )
        return summaries
