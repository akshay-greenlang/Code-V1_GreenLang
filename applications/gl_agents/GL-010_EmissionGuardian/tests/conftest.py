"""
GL-010 EmissionsGuardian Test Configuration

Pytest fixtures for comprehensive testing of the EmissionsGuardian compliance agent.
Includes generators for CEMS data, RATA tests, permit configurations, fugitive sensors,
and trading price data.

Author: GreenLang Test Engineering
Version: 1.0.0
"""

import pytest
import hashlib
import json
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Generator, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from unittest.mock import Mock, MagicMock, AsyncMock

import numpy as np
from hypothesis import strategies as st, settings


# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class PollutantType(Enum):
    """Supported pollutant types for CEMS monitoring."""
    NOX = "NOx"
    SO2 = "SO2"
    CO = "CO"
    CO2 = "CO2"
    PM = "PM"
    VOC = "VOC"
    HCL = "HCl"
    HG = "Hg"
    O2 = "O2"


class DataQualityFlag(Enum):
    """Data quality status flags."""
    VALID = "VALID"
    MISSING = "MISSING"
    SUBSTITUTED = "SUBSTITUTED"
    CALIBRATION = "CALIBRATION"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    RATE_OF_CHANGE_EXCEEDED = "RATE_OF_CHANGE_EXCEEDED"


class ComplianceStatus(Enum):
    """Compliance evaluation status."""
    COMPLIANT = "COMPLIANT"
    EXCEEDANCE = "EXCEEDANCE"
    EXEMPTION_APPLIED = "EXEMPTION_APPLIED"
    PENDING_REVIEW = "PENDING_REVIEW"


# EPA Reference Values (40 CFR Part 75)
EPA_REFERENCE_O2_DRY = Decimal("7.0")  # % O2 dry basis for correction
EPA_F_FACTOR_COAL = Decimal("9780")  # dscf/MMBtu
EPA_F_FACTOR_OIL = Decimal("9190")  # dscf/MMBtu
EPA_F_FACTOR_GAS = Decimal("8710")  # dscf/MMBtu


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CEMSReading:
    """Single CEMS reading with all required parameters."""
    timestamp: datetime
    unit_id: str
    pollutant: PollutantType
    concentration_ppm: Decimal
    flow_rate_scfh: Decimal
    o2_percent: Decimal
    moisture_percent: Decimal
    temperature_fahrenheit: Decimal
    pressure_in_hg: Decimal
    quality_flag: DataQualityFlag = DataQualityFlag.VALID
    calibration_factor: Decimal = Decimal("1.0")
    provenance_hash: Optional[str] = None

    def __post_init__(self):
        """Calculate provenance hash after initialization."""
        if self.provenance_hash is None:
            self.provenance_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate deterministic hash of reading data."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "unit_id": self.unit_id,
            "pollutant": self.pollutant.value,
            "concentration_ppm": str(self.concentration_ppm),
            "flow_rate_scfh": str(self.flow_rate_scfh),
            "o2_percent": str(self.o2_percent),
            "moisture_percent": str(self.moisture_percent),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class HourlyAverage:
    """Hourly average data for compliance reporting."""
    hour_start: datetime
    hour_end: datetime
    unit_id: str
    pollutant: PollutantType
    average_concentration_ppm: Decimal
    average_emission_rate_lb_per_mmbtu: Decimal
    average_mass_rate_lb_per_hour: Decimal
    data_completeness_percent: Decimal
    quality_assured: bool
    readings_count: int
    substitution_applied: bool = False
    provenance_hash: Optional[str] = None


@dataclass
class RATATestRun:
    """RATA test run data per EPA 40 CFR Part 75."""
    test_id: str
    unit_id: str
    pollutant: PollutantType
    test_date: datetime
    reference_method: str
    operating_level: str
    run_number: int
    reference_value: Decimal
    cems_value: Decimal
    difference: Decimal
    percent_difference: Decimal
    reference_method_uncertainty: Optional[Decimal] = None
    provenance_hash: Optional[str] = None


@dataclass
class RATATestResult:
    """Complete RATA test result with all statistical calculations."""
    test_id: str
    unit_id: str
    pollutant: PollutantType
    test_date: datetime
    operating_level: str
    runs: List[RATATestRun] = field(default_factory=list)
    mean_difference: Optional[Decimal] = None
    standard_deviation: Optional[Decimal] = None
    confidence_coefficient: Optional[Decimal] = None
    relative_accuracy: Optional[Decimal] = None
    bias_adjusted_relative_accuracy: Optional[Decimal] = None
    passed: Optional[bool] = None
    passing_threshold: Decimal = Decimal("10.0")
    provenance_hash: Optional[str] = None


@dataclass
class PermitLimit:
    """Emission permit limit specification."""
    permit_id: str
    unit_id: str
    pollutant: PollutantType
    limit_value: Decimal
    limit_units: str
    averaging_period: str
    effective_date: datetime
    expiration_date: Optional[datetime] = None
    version: int = 1
    startup_exemption_hours: int = 0
    shutdown_exemption_hours: int = 0
    malfunction_provisions: bool = False


@dataclass
class PermitConfiguration:
    """Complete permit configuration for a facility."""
    facility_id: str
    facility_name: str
    permits: List[PermitLimit] = field(default_factory=list)
    regulatory_agency: str = "EPA"
    state_agency: Optional[str] = None
    version: int = 1
    effective_date: datetime = field(default_factory=datetime.now)
    provenance_hash: Optional[str] = None


@dataclass
class FugitiveSensorReading:
    """Fugitive emissions sensor reading."""
    sensor_id: str
    timestamp: datetime
    location_id: str
    component_type: str
    concentration_ppm: Decimal
    wind_speed_mph: Decimal
    wind_direction_degrees: Decimal
    ambient_temperature_fahrenheit: Decimal
    leak_detected: bool
    quality_flag: DataQualityFlag = DataQualityFlag.VALID
    provenance_hash: Optional[str] = None


@dataclass
class TradingPriceData:
    """Emissions allowance trading price data."""
    trading_date: datetime
    market: str
    vintage_year: int
    allowance_type: str
    settlement_price_usd: Decimal
    volume_traded: int
    open_price_usd: Optional[Decimal] = None
    high_price_usd: Optional[Decimal] = None
    low_price_usd: Optional[Decimal] = None
    provenance_hash: Optional[str] = None


# =============================================================================
# CEMS DATA GENERATORS
# =============================================================================

class CEMSDataGenerator:
    """Generator for realistic CEMS monitoring data."""

    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def generate_reading(
        self,
        timestamp: datetime,
        unit_id: str = "UNIT-001",
        pollutant: PollutantType = PollutantType.NOX,
        base_concentration: float = 100.0,
        noise_std: float = 5.0,
    ) -> CEMSReading:
        """Generate a single CEMS reading with realistic variation."""
        concentration = Decimal(str(max(0, self.rng.normal(base_concentration, noise_std))))
        flow_rate = Decimal(str(max(1000, self.rng.normal(50000, 2000))))
        o2_percent = Decimal(str(max(0, min(21, self.rng.normal(5.0, 0.5)))))
        moisture = Decimal(str(max(0, min(30, self.rng.normal(8.0, 1.0)))))
        temperature = Decimal(str(self.rng.normal(350, 20)))
        pressure = Decimal(str(self.rng.normal(29.92, 0.5)))

        return CEMSReading(
            timestamp=timestamp,
            unit_id=unit_id,
            pollutant=pollutant,
            concentration_ppm=concentration.quantize(Decimal("0.01")),
            flow_rate_scfh=flow_rate.quantize(Decimal("1")),
            o2_percent=o2_percent.quantize(Decimal("0.01")),
            moisture_percent=moisture.quantize(Decimal("0.01")),
            temperature_fahrenheit=temperature.quantize(Decimal("0.1")),
            pressure_in_hg=pressure.quantize(Decimal("0.01")),
            quality_flag=DataQualityFlag.VALID,
        )

    def generate_hourly_readings(
        self,
        hour_start: datetime,
        unit_id: str = "UNIT-001",
        pollutant: PollutantType = PollutantType.NOX,
        readings_per_hour: int = 4,
        base_concentration: float = 100.0,
        completeness: float = 1.0,
    ) -> List[CEMSReading]:
        """Generate readings for one hour at specified intervals."""
        readings = []
        interval_minutes = 60 // readings_per_hour

        for i in range(readings_per_hour):
            if self.rng.random() <= completeness:
                timestamp = hour_start + timedelta(minutes=i * interval_minutes)
                reading = self.generate_reading(
                    timestamp=timestamp,
                    unit_id=unit_id,
                    pollutant=pollutant,
                    base_concentration=base_concentration,
                )
                readings.append(reading)

        return readings

    def generate_daily_readings(
        self,
        day_start: datetime,
        unit_id: str = "UNIT-001",
        pollutant: PollutantType = PollutantType.NOX,
        readings_per_hour: int = 4,
        base_concentration: float = 100.0,
        completeness: float = 1.0,
    ) -> List[CEMSReading]:
        """Generate readings for a full day."""
        readings = []
        for hour in range(24):
            hour_start = day_start + timedelta(hours=hour)
            hourly_readings = self.generate_hourly_readings(
                hour_start=hour_start,
                unit_id=unit_id,
                pollutant=pollutant,
                readings_per_hour=readings_per_hour,
                base_concentration=base_concentration,
                completeness=completeness,
            )
            readings.extend(hourly_readings)
        return readings

    def generate_exceedance_event(
        self,
        hour_start: datetime,
        unit_id: str = "UNIT-001",
        pollutant: PollutantType = PollutantType.NOX,
        normal_concentration: float = 100.0,
        exceedance_concentration: float = 200.0,
        readings_per_hour: int = 4,
    ) -> List[CEMSReading]:
        """Generate readings that represent an exceedance event."""
        readings = []
        interval_minutes = 60 // readings_per_hour

        for i in range(readings_per_hour):
            timestamp = hour_start + timedelta(minutes=i * interval_minutes)
            concentration = normal_concentration + (exceedance_concentration - normal_concentration) * (i / readings_per_hour)
            reading = CEMSReading(
                timestamp=timestamp,
                unit_id=unit_id,
                pollutant=pollutant,
                concentration_ppm=Decimal(str(concentration)).quantize(Decimal("0.01")),
                flow_rate_scfh=Decimal("50000"),
                o2_percent=Decimal("5.00"),
                moisture_percent=Decimal("8.00"),
                temperature_fahrenheit=Decimal("350.0"),
                pressure_in_hg=Decimal("29.92"),
                quality_flag=DataQualityFlag.VALID,
            )
            readings.append(reading)

        return readings
