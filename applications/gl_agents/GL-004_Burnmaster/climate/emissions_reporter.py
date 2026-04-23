"""
GL-004 BURNMASTER - Emissions Reporter

Core emissions aggregation and reporting engine for climate intelligence.
Provides unified emissions tracking across multiple reporting frameworks.

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ReportingStandard(str, Enum):
    """Supported reporting standards."""
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    EPA_CFR_98 = "epa_cfr_98"
    EU_ETS = "eu_ets"
    TCFD = "tcfd"
    UK_SECR = "uk_secr"
    CDP = "cdp"


class PollutantType(str, Enum):
    """Types of emissions tracked."""
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    CO2E = "co2e"  # CO2 equivalent
    NOX = "nox"
    CO = "co"
    SO2 = "so2"
    PM = "pm"


class FuelType(str, Enum):
    """Fuel types for emissions calculations."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    DIESEL = "diesel"
    PROPANE = "propane"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    REFINERY_GAS = "refinery_gas"
    HYDROGEN = "hydrogen"
    BIOGAS = "biogas"
    BIOMETHANE = "biomethane"


class UncertaintyLevel(str, Enum):
    """Uncertainty levels for emissions data."""
    LOW = "low"        # < 5%
    MEDIUM = "medium"  # 5-15%
    HIGH = "high"      # > 15%


class VerificationStatus(str, Enum):
    """Verification status for emissions reports."""
    UNVERIFIED = "unverified"
    SELF_VERIFIED = "self_verified"
    THIRD_PARTY_VERIFIED = "third_party_verified"
    AUDITED = "audited"


# GWP values (100-year, AR5)
GWP_AR5: Dict[str, int] = {
    "co2": 1,
    "ch4": 28,
    "n2o": 265,
}


@dataclass
class ReportingPeriod:
    """Represents a reporting period for emissions data."""
    start: datetime
    end: datetime
    period_type: str = "calendar_year"  # calendar_year, fiscal_year, quarter, month

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError("start must be before end")

    @property
    def duration_days(self) -> int:
        return (self.end - self.start).days

    @classmethod
    def calendar_year(cls, year: int) -> "ReportingPeriod":
        return cls(
            start=datetime(year, 1, 1, tzinfo=timezone.utc),
            end=datetime(year + 1, 1, 1, tzinfo=timezone.utc),
            period_type="calendar_year"
        )

    @classmethod
    def quarter(cls, year: int, quarter: int) -> "ReportingPeriod":
        if quarter not in (1, 2, 3, 4):
            raise ValueError("quarter must be 1-4")
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 3
        end_year = year if end_month <= 12 else year + 1
        end_month = end_month if end_month <= 12 else end_month - 12
        return cls(
            start=datetime(year, start_month, 1, tzinfo=timezone.utc),
            end=datetime(end_year, end_month, 1, tzinfo=timezone.utc),
            period_type="quarter"
        )


class EmissionRecord(BaseModel):
    """Single emission measurement or calculation record."""
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    unit_id: str = Field(..., description="Combustion unit identifier")
    fuel_type: FuelType = Field(..., description="Type of fuel burned")
    fuel_quantity: Decimal = Field(..., ge=0, description="Fuel quantity consumed")
    fuel_unit: str = Field(default="kg", description="Unit of fuel quantity")
    co2_kg: Decimal = Field(..., ge=0, description="CO2 emissions in kg")
    ch4_kg: Decimal = Field(default=Decimal("0"), ge=0, description="CH4 emissions in kg")
    n2o_kg: Decimal = Field(default=Decimal("0"), ge=0, description="N2O emissions in kg")
    co2e_kg: Decimal = Field(..., ge=0, description="CO2 equivalent in kg")
    calculation_method: str = Field(default="tier_1", description="Calculation tier/method")
    data_quality: UncertaintyLevel = Field(default=UncertaintyLevel.MEDIUM)
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")

    @field_validator("co2e_kg", mode="before")
    @classmethod
    def compute_co2e(cls, v, info):
        if v is not None:
            return v
        values = info.data
        co2 = values.get("co2_kg", Decimal("0"))
        ch4 = values.get("ch4_kg", Decimal("0"))
        n2o = values.get("n2o_kg", Decimal("0"))
        return co2 + ch4 * GWP_AR5["ch4"] + n2o * GWP_AR5["n2o"]


class EmissionsReport(BaseModel):
    """Comprehensive emissions report for a reporting period."""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    period_start: datetime = Field(..., description="Reporting period start")
    period_end: datetime = Field(..., description="Reporting period end")
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(default="", description="Facility name")

    # Total emissions by pollutant
    total_co2_tonnes: Decimal = Field(default=Decimal("0"), description="Total CO2 (tonnes)")
    total_ch4_tonnes: Decimal = Field(default=Decimal("0"), description="Total CH4 (tonnes)")
    total_n2o_tonnes: Decimal = Field(default=Decimal("0"), description="Total N2O (tonnes)")
    total_co2e_tonnes: Decimal = Field(default=Decimal("0"), description="Total CO2e (tonnes)")

    # Breakdown by fuel type
    emissions_by_fuel: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict)

    # Breakdown by unit
    emissions_by_unit: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict)

    # Activity data
    fuel_consumption: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict)

    # Methodology and quality
    calculation_methodology: str = Field(default="tier_1")
    data_quality_score: float = Field(default=0.0, ge=0, le=1)
    uncertainty_percent: float = Field(default=0.0, ge=0)
    verification_status: VerificationStatus = Field(default=VerificationStatus.UNVERIFIED)

    # Intensity metrics
    intensity_per_unit_output: Optional[Decimal] = Field(None, description="CO2e per unit output")
    output_metric: Optional[str] = Field(None, description="Output metric (e.g., MWh, tonnes)")

    # Audit trail
    provenance_hash: str = Field(default="", description="SHA-256 hash of inputs")
    records_count: int = Field(default=0, description="Number of underlying records")

    standards_applied: List[str] = Field(default_factory=list)


class EmissionsReporter:
    """
    Core emissions aggregation and reporting engine.

    Provides unified emissions tracking across multiple reporting frameworks
    with full audit trail and provenance tracking.

    Example:
        >>> reporter = EmissionsReporter()
        >>> record = reporter.record_emission(
        ...     unit_id="BOILER-001",
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     fuel_quantity=Decimal("1000"),
        ...     fuel_unit="kg"
        ... )
        >>> report = reporter.generate_report(
        ...     ReportingPeriod.calendar_year(2024),
        ...     facility_id="FACILITY-001"
        ... )
    """

    # Default emission factors (kg CO2 per kg fuel) - simplified
    DEFAULT_EMISSION_FACTORS: Dict[FuelType, Dict[str, float]] = {
        FuelType.NATURAL_GAS: {"co2": 2.75, "ch4": 0.001, "n2o": 0.0001},
        FuelType.FUEL_OIL_NO2: {"co2": 3.16, "ch4": 0.0003, "n2o": 0.0006},
        FuelType.FUEL_OIL_NO6: {"co2": 3.11, "ch4": 0.0003, "n2o": 0.0006},
        FuelType.DIESEL: {"co2": 3.16, "ch4": 0.0003, "n2o": 0.0006},
        FuelType.PROPANE: {"co2": 2.99, "ch4": 0.001, "n2o": 0.0001},
        FuelType.COAL_BITUMINOUS: {"co2": 2.42, "ch4": 0.011, "n2o": 0.0016},
        FuelType.COAL_SUBBITUMINOUS: {"co2": 1.88, "ch4": 0.011, "n2o": 0.0016},
        FuelType.REFINERY_GAS: {"co2": 2.50, "ch4": 0.003, "n2o": 0.0001},
        FuelType.HYDROGEN: {"co2": 0.0, "ch4": 0.0, "n2o": 0.0},
        FuelType.BIOGAS: {"co2": 0.0, "ch4": 0.001, "n2o": 0.0001},  # Biogenic CO2
        FuelType.BIOMETHANE: {"co2": 0.0, "ch4": 0.001, "n2o": 0.0001},
    }

    def __init__(self, precision: int = 4):
        """
        Initialize emissions reporter.

        Args:
            precision: Decimal places for calculations
        """
        self.precision = precision
        self._quantize_str = "0." + "0" * precision
        self._records: List[EmissionRecord] = []
        self._custom_factors: Dict[str, Dict[str, float]] = {}
        logger.info("EmissionsReporter initialized")

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def set_custom_emission_factor(
        self,
        fuel_type: FuelType,
        factors: Dict[str, float]
    ) -> None:
        """
        Set custom emission factors for a fuel type.

        Args:
            fuel_type: The fuel type
            factors: Dict with 'co2', 'ch4', 'n2o' factors (kg/kg fuel)
        """
        self._custom_factors[fuel_type.value] = factors
        logger.info(f"Custom emission factors set for {fuel_type.value}")

    def get_emission_factors(self, fuel_type: FuelType) -> Dict[str, float]:
        """Get emission factors for a fuel type (custom or default)."""
        if fuel_type.value in self._custom_factors:
            return self._custom_factors[fuel_type.value]
        return self.DEFAULT_EMISSION_FACTORS.get(
            fuel_type,
            {"co2": 2.5, "ch4": 0.001, "n2o": 0.0001}
        )

    def record_emission(
        self,
        unit_id: str,
        fuel_type: FuelType,
        fuel_quantity: Decimal,
        fuel_unit: str = "kg",
        timestamp: Optional[datetime] = None,
        calculation_method: str = "tier_1",
        data_quality: UncertaintyLevel = UncertaintyLevel.MEDIUM,
    ) -> EmissionRecord:
        """
        Record a single emission measurement.

        Args:
            unit_id: Combustion unit identifier
            fuel_type: Type of fuel burned
            fuel_quantity: Amount of fuel consumed
            fuel_unit: Unit of fuel quantity (kg, m3, mmbtu)
            timestamp: Timestamp of emission (default: now)
            calculation_method: Calculation tier (tier_1, tier_2, tier_3)
            data_quality: Uncertainty level of the data

        Returns:
            EmissionRecord with calculated emissions
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        # Get emission factors
        factors = self.get_emission_factors(fuel_type)

        # Convert fuel quantity to kg if needed
        fuel_kg = self._convert_to_kg(fuel_quantity, fuel_unit, fuel_type)

        # Calculate emissions (DETERMINISTIC)
        co2_kg = self._quantize(fuel_kg * Decimal(str(factors["co2"])))
        ch4_kg = self._quantize(fuel_kg * Decimal(str(factors["ch4"])))
        n2o_kg = self._quantize(fuel_kg * Decimal(str(factors["n2o"])))

        # Calculate CO2e (using GWP AR5)
        co2e_kg = self._quantize(
            co2_kg +
            ch4_kg * GWP_AR5["ch4"] +
            n2o_kg * GWP_AR5["n2o"]
        )

        # Compute provenance hash
        provenance = self._compute_hash({
            "unit_id": unit_id,
            "fuel_type": fuel_type.value,
            "fuel_quantity": str(fuel_quantity),
            "fuel_unit": fuel_unit,
            "timestamp": timestamp.isoformat(),
            "factors": factors
        })

        record = EmissionRecord(
            timestamp=timestamp,
            unit_id=unit_id,
            fuel_type=fuel_type,
            fuel_quantity=fuel_quantity,
            fuel_unit=fuel_unit,
            co2_kg=co2_kg,
            ch4_kg=ch4_kg,
            n2o_kg=n2o_kg,
            co2e_kg=co2e_kg,
            calculation_method=calculation_method,
            data_quality=data_quality,
            provenance_hash=provenance
        )

        self._records.append(record)
        logger.debug(f"Recorded emission: {record.record_id}")
        return record

    def _convert_to_kg(
        self,
        quantity: Decimal,
        unit: str,
        fuel_type: FuelType
    ) -> Decimal:
        """Convert fuel quantity to kg."""
        # Density approximations (kg/m3)
        densities = {
            FuelType.NATURAL_GAS: 0.76,  # kg/Nm3
            FuelType.PROPANE: 1.88,
            FuelType.FUEL_OIL_NO2: 850.0,
            FuelType.FUEL_OIL_NO6: 970.0,
            FuelType.DIESEL: 850.0,
        }

        unit_lower = unit.lower()

        if unit_lower == "kg":
            return quantity
        elif unit_lower in ("m3", "nm3"):
            density = densities.get(fuel_type, 1.0)
            return self._quantize(quantity * Decimal(str(density)))
        elif unit_lower == "tonnes":
            return self._quantize(quantity * Decimal("1000"))
        elif unit_lower == "mmbtu":
            # Approximate conversion using heating values
            hhv_mj_per_kg = {
                FuelType.NATURAL_GAS: 52.0,
                FuelType.FUEL_OIL_NO2: 45.5,
                FuelType.DIESEL: 45.5,
                FuelType.COAL_BITUMINOUS: 27.0,
            }
            hhv = hhv_mj_per_kg.get(fuel_type, 45.0)
            mj = quantity * Decimal("1055.06")  # MMBtu to MJ
            return self._quantize(mj / Decimal(str(hhv)))
        else:
            logger.warning(f"Unknown unit {unit}, assuming kg")
            return quantity

    def generate_report(
        self,
        period: ReportingPeriod,
        facility_id: str,
        facility_name: str = "",
        unit_ids: Optional[List[str]] = None,
        standards: Optional[List[ReportingStandard]] = None,
    ) -> EmissionsReport:
        """
        Generate comprehensive emissions report for a period.

        Args:
            period: Reporting period
            facility_id: Facility identifier
            facility_name: Facility display name
            unit_ids: Filter to specific units (optional)
            standards: Reporting standards to apply

        Returns:
            EmissionsReport with aggregated emissions data
        """
        # Filter records by period and units
        filtered = [
            r for r in self._records
            if period.start <= r.timestamp < period.end
            and (unit_ids is None or r.unit_id in unit_ids)
        ]

        if not filtered:
            logger.warning(f"No emission records found for period {period.start} to {period.end}")

        # Aggregate totals
        total_co2 = sum(r.co2_kg for r in filtered)
        total_ch4 = sum(r.ch4_kg for r in filtered)
        total_n2o = sum(r.n2o_kg for r in filtered)
        total_co2e = sum(r.co2e_kg for r in filtered)

        # Convert to tonnes
        total_co2_tonnes = self._quantize(total_co2 / Decimal("1000"))
        total_ch4_tonnes = self._quantize(total_ch4 / Decimal("1000"))
        total_n2o_tonnes = self._quantize(total_n2o / Decimal("1000"))
        total_co2e_tonnes = self._quantize(total_co2e / Decimal("1000"))

        # Breakdown by fuel
        emissions_by_fuel: Dict[str, Dict[str, Decimal]] = {}
        fuel_consumption: Dict[str, Dict[str, Decimal]] = {}

        for r in filtered:
            fuel = r.fuel_type.value
            if fuel not in emissions_by_fuel:
                emissions_by_fuel[fuel] = {
                    "co2_kg": Decimal("0"),
                    "ch4_kg": Decimal("0"),
                    "n2o_kg": Decimal("0"),
                    "co2e_kg": Decimal("0"),
                }
                fuel_consumption[fuel] = {"quantity_kg": Decimal("0")}

            emissions_by_fuel[fuel]["co2_kg"] += r.co2_kg
            emissions_by_fuel[fuel]["ch4_kg"] += r.ch4_kg
            emissions_by_fuel[fuel]["n2o_kg"] += r.n2o_kg
            emissions_by_fuel[fuel]["co2e_kg"] += r.co2e_kg
            fuel_consumption[fuel]["quantity_kg"] += self._convert_to_kg(
                r.fuel_quantity, r.fuel_unit, r.fuel_type
            )

        # Breakdown by unit
        emissions_by_unit: Dict[str, Dict[str, Decimal]] = {}

        for r in filtered:
            unit = r.unit_id
            if unit not in emissions_by_unit:
                emissions_by_unit[unit] = {
                    "co2_kg": Decimal("0"),
                    "ch4_kg": Decimal("0"),
                    "n2o_kg": Decimal("0"),
                    "co2e_kg": Decimal("0"),
                }

            emissions_by_unit[unit]["co2_kg"] += r.co2_kg
            emissions_by_unit[unit]["ch4_kg"] += r.ch4_kg
            emissions_by_unit[unit]["n2o_kg"] += r.n2o_kg
            emissions_by_unit[unit]["co2e_kg"] += r.co2e_kg

        # Calculate data quality score
        quality_scores = {
            UncertaintyLevel.LOW: 1.0,
            UncertaintyLevel.MEDIUM: 0.7,
            UncertaintyLevel.HIGH: 0.4,
        }
        if filtered:
            data_quality_score = sum(
                quality_scores[r.data_quality] for r in filtered
            ) / len(filtered)
        else:
            data_quality_score = 0.0

        # Compute provenance hash
        provenance = self._compute_hash({
            "facility_id": facility_id,
            "period_start": period.start.isoformat(),
            "period_end": period.end.isoformat(),
            "records_count": len(filtered),
            "total_co2e_kg": str(total_co2e),
        })

        return EmissionsReport(
            period_start=period.start,
            period_end=period.end,
            facility_id=facility_id,
            facility_name=facility_name,
            total_co2_tonnes=total_co2_tonnes,
            total_ch4_tonnes=total_ch4_tonnes,
            total_n2o_tonnes=total_n2o_tonnes,
            total_co2e_tonnes=total_co2e_tonnes,
            emissions_by_fuel=emissions_by_fuel,
            emissions_by_unit=emissions_by_unit,
            fuel_consumption=fuel_consumption,
            data_quality_score=round(data_quality_score, 2),
            provenance_hash=provenance,
            records_count=len(filtered),
            standards_applied=[s.value for s in (standards or [])],
        )

    def get_records(
        self,
        period: Optional[ReportingPeriod] = None,
        unit_id: Optional[str] = None,
        fuel_type: Optional[FuelType] = None,
    ) -> List[EmissionRecord]:
        """Get emission records with optional filtering."""
        records = self._records

        if period:
            records = [r for r in records if period.start <= r.timestamp < period.end]
        if unit_id:
            records = [r for r in records if r.unit_id == unit_id]
        if fuel_type:
            records = [r for r in records if r.fuel_type == fuel_type]

        return records

    def clear_records(self) -> int:
        """Clear all emission records. Returns count of cleared records."""
        count = len(self._records)
        self._records = []
        logger.info(f"Cleared {count} emission records")
        return count
