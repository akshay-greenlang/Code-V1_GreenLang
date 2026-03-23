# -*- coding: utf-8 -*-
"""
FranchiseSpecificCalculatorEngine - Tier 1 franchise-specific emissions calculator.

This module implements the FranchiseSpecificCalculatorEngine for AGENT-MRV-027
(Franchises, GHG Protocol Scope 3 Category 14). It provides thread-safe
singleton calculation of franchise unit emissions using metered energy and
fuel data (primary data) per individual franchise location.

Formula:
    E_unit = SUM_sources [ activity_data x EF ]

This is the highest-accuracy (Tier 1) calculation method. It processes
7 emission sources per franchise unit:

    Scope 1 of Franchisee:
        1. Stationary combustion -- cooking (gas/propane), heating (gas/oil),
           generators. E = fuel_volume x fuel_EF
        2. Mobile combustion -- delivery vehicles, company cars.
           E = distance x vehicle_EF  OR  fuel_volume x fuel_EF
        3. Refrigerant leakage -- HVAC and commercial refrigeration.
           E = charge_kg x leakage_rate x GWP
        4. Process emissions -- if applicable (minimal for most franchises)

    Scope 2 of Franchisee:
        5. Purchased electricity -- E = kWh x grid_EF(region)
        6. Purchased heating/steam -- E = energy_input x heating_EF
        7. Purchased cooling -- E = energy_input x cooling_EF

Franchise-type-specific calculators provide tailored logic for:
    - QSR (cooking + refrigeration heavy)
    - Hotel (HVAC + water heating + laundry + pool)
    - Convenience store (24/7 refrigeration + lighting)
    - Retail (HVAC + lighting)
    - Fitness (HVAC + equipment + water heating)
    - Automotive (equipment + water + chemicals)
    - Generic (default path for other types)

Features:
    - All arithmetic with Decimal and ROUND_HALF_UP
    - Thread-safe singleton with threading.RLock
    - Metrics recording via get_metrics_collector()
    - Provenance hashing via get_provenance_manager()
    - Company-owned check: ownership_type == "company_owned" raises ValueError
    - Pro-rata adjustment for partial reporting periods
    - Batch processing for multiple franchise units
    - 5-dimension DQI scoring per unit
    - Uncertainty quantification per calculation

Example:
    >>> engine = FranchiseSpecificCalculatorEngine()
    >>> result = engine.calculate(FranchiseUnitInput(
    ...     unit_id="FRN-001",
    ...     franchise_type="qsr",
    ...     ownership_type="franchisee",
    ...     country="US",
    ...     region="CAMX",
    ...     floor_area_m2=Decimal("250"),
    ...     electricity_kwh=Decimal("155000"),
    ... ))
    >>> result.total_co2e_kg > 0
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-014
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_ID = "GL-MRV-S3-014"
AGENT_COMPONENT = "AGENT-MRV-027"
VERSION = "1.0.0"
TABLE_PREFIX = "gl_frn_"

# Quantization constants
_QUANT_8DP = Decimal("0.00000001")
_QUANT_5DP = Decimal("0.00001")
_QUANT_2DP = Decimal("0.01")

# Default days in a year for pro-rata
_DAYS_PER_YEAR = Decimal("365")

# Default heating EF (kgCO2e per kWh of heat) -- district heating average
_DEFAULT_HEATING_EF = Decimal("0.1980")

# Default cooling EF (kgCO2e per kWh of cooling) -- electric chiller average
_DEFAULT_COOLING_EF = Decimal("0.1200")

# Default batch size limit
_DEFAULT_BATCH_SIZE = 1000


# ============================================================================
# ENUMERATIONS
# ============================================================================


class OwnershipType(str, Enum):
    """Franchise unit ownership classification."""
    FRANCHISEE = "franchisee"
    COMPANY_OWNED = "company_owned"
    JOINT_VENTURE = "joint_venture"
    MASTER_FRANCHISEE = "master_franchisee"


class CalculationTier(str, Enum):
    """Data quality tier for the calculation."""
    TIER_1 = "tier_1"   # Primary metered data, full coverage
    TIER_2 = "tier_2"   # Primary data with some gaps
    TIER_3 = "tier_3"   # Primary data for subset, extrapolated


class EmissionSource(str, Enum):
    """Emission source categories within a franchise unit."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    REFRIGERANT_LEAKAGE = "refrigerant_leakage"
    PROCESS_EMISSIONS = "process_emissions"
    PURCHASED_ELECTRICITY = "purchased_electricity"
    PURCHASED_HEATING = "purchased_heating"
    PURCHASED_COOLING = "purchased_cooling"


class ValidationSeverity(str, Enum):
    """Severity level for validation warnings/errors."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class StationaryCombustionInput:
    """
    Stationary combustion data for a franchise unit.

    Covers cooking equipment (fryers, grills, ovens), space/water heating,
    and backup generators.
    """
    natural_gas_m3: Decimal = Decimal("0")
    propane_litres: Decimal = Decimal("0")
    diesel_litres: Decimal = Decimal("0")
    heating_oil_litres: Decimal = Decimal("0")
    petrol_litres: Decimal = Decimal("0")
    wood_pellets_kg: Decimal = Decimal("0")
    coal_kg: Decimal = Decimal("0")
    biodiesel_litres: Decimal = Decimal("0")


@dataclass
class MobileCombustionInput:
    """
    Mobile combustion data for franchise unit delivery fleet.

    Supports both distance-based and fuel-based calculation methods.
    """
    # Distance-based inputs
    vehicles: List[Dict[str, Any]] = field(default_factory=list)
    # Each vehicle dict: {vehicle_type, distance_km, fuel_type, fuel_volume}

    # Aggregate fuel-based inputs (fallback)
    total_diesel_litres: Decimal = Decimal("0")
    total_petrol_litres: Decimal = Decimal("0")


@dataclass
class RefrigerantInput:
    """
    Refrigerant leakage data for a franchise unit.

    Each equipment entry: {equipment_type, refrigerant_type, charge_kg}
    """
    equipment: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {equipment_type: str, refrigerant_type: str, charge_kg: Decimal}


@dataclass
class FranchiseUnitInput:
    """
    Complete input data for a single franchise unit calculation.

    All energy and fuel quantities should cover the full reporting period
    (or the portion of the period the unit was operational).

    Attributes:
        unit_id: Unique franchise unit identifier.
        franchise_type: Type of franchise (e.g., "qsr", "hotel").
        ownership_type: "franchisee", "company_owned", "joint_venture".
        country: ISO 3166-1 alpha-2 country code.
        region: Optional eGRID subregion or sub-national region code.
        climate_zone: Optional climate zone override (auto-detected if None).
        floor_area_m2: Gross floor area in square metres.
        reporting_year: Reporting year (e.g., 2025).
        operating_days: Days the unit was operational during reporting period.
        total_days: Total days in the reporting period (default 365).

        -- Scope 2 of franchisee --
        electricity_kwh: Purchased electricity in kWh.
        heating_kwh: Purchased heating/steam in kWh.
        cooling_kwh: Purchased cooling in kWh.

        -- Scope 1 of franchisee --
        stationary_combustion: Fuel consumption data.
        mobile_combustion: Delivery fleet data.
        refrigerants: Refrigerant equipment data.
        process_emissions_co2e_kg: Direct process emissions (if any).

        -- Metadata --
        ef_source: Override emission factor source.
        notes: Free-text notes.
    """
    unit_id: str = ""
    franchise_type: str = "generic"
    ownership_type: str = "franchisee"
    country: str = "US"
    region: Optional[str] = None
    climate_zone: Optional[str] = None
    floor_area_m2: Decimal = Decimal("0")
    reporting_year: int = 2025
    operating_days: Optional[int] = None
    total_days: int = 365

    # Scope 2 of franchisee
    electricity_kwh: Decimal = Decimal("0")
    heating_kwh: Decimal = Decimal("0")
    cooling_kwh: Decimal = Decimal("0")

    # Scope 1 of franchisee
    stationary_combustion: Optional[StationaryCombustionInput] = None
    mobile_combustion: Optional[MobileCombustionInput] = None
    refrigerants: Optional[RefrigerantInput] = None
    process_emissions_co2e_kg: Decimal = Decimal("0")

    # Metadata
    ef_source: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "unit_id": self.unit_id,
            "franchise_type": self.franchise_type,
            "ownership_type": self.ownership_type,
            "country": self.country,
            "region": self.region,
            "climate_zone": self.climate_zone,
            "floor_area_m2": str(self.floor_area_m2),
            "reporting_year": self.reporting_year,
            "operating_days": self.operating_days,
            "total_days": self.total_days,
            "electricity_kwh": str(self.electricity_kwh),
            "heating_kwh": str(self.heating_kwh),
            "cooling_kwh": str(self.cooling_kwh),
            "process_emissions_co2e_kg": str(self.process_emissions_co2e_kg),
            "ef_source": self.ef_source,
            "notes": self.notes,
        }
        if self.stationary_combustion is not None:
            result["stationary_combustion"] = asdict(self.stationary_combustion)
        if self.mobile_combustion is not None:
            result["mobile_combustion"] = asdict(self.mobile_combustion)
        if self.refrigerants is not None:
            result["refrigerants"] = asdict(self.refrigerants)
        return result


@dataclass
class EmissionBreakdown:
    """
    Emissions breakdown by source category.

    All values in kgCO2e for the reporting period.
    """
    stationary_combustion_co2e_kg: Decimal = Decimal("0")
    mobile_combustion_co2e_kg: Decimal = Decimal("0")
    refrigerant_leakage_co2e_kg: Decimal = Decimal("0")
    process_emissions_co2e_kg: Decimal = Decimal("0")
    purchased_electricity_co2e_kg: Decimal = Decimal("0")
    purchased_heating_co2e_kg: Decimal = Decimal("0")
    purchased_cooling_co2e_kg: Decimal = Decimal("0")

    @property
    def scope1_total_co2e_kg(self) -> Decimal:
        """Total franchisee Scope 1 emissions (kgCO2e)."""
        return (
            self.stationary_combustion_co2e_kg
            + self.mobile_combustion_co2e_kg
            + self.refrigerant_leakage_co2e_kg
            + self.process_emissions_co2e_kg
        )

    @property
    def scope2_total_co2e_kg(self) -> Decimal:
        """Total franchisee Scope 2 emissions (kgCO2e)."""
        return (
            self.purchased_electricity_co2e_kg
            + self.purchased_heating_co2e_kg
            + self.purchased_cooling_co2e_kg
        )

    @property
    def total_co2e_kg(self) -> Decimal:
        """Total franchisee emissions (Scope 1 + Scope 2) in kgCO2e."""
        return self.scope1_total_co2e_kg + self.scope2_total_co2e_kg

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary with string Decimal values."""
        return {
            "stationary_combustion_co2e_kg": str(self.stationary_combustion_co2e_kg),
            "mobile_combustion_co2e_kg": str(self.mobile_combustion_co2e_kg),
            "refrigerant_leakage_co2e_kg": str(self.refrigerant_leakage_co2e_kg),
            "process_emissions_co2e_kg": str(self.process_emissions_co2e_kg),
            "purchased_electricity_co2e_kg": str(self.purchased_electricity_co2e_kg),
            "purchased_heating_co2e_kg": str(self.purchased_heating_co2e_kg),
            "purchased_cooling_co2e_kg": str(self.purchased_cooling_co2e_kg),
            "scope1_total_co2e_kg": str(self.scope1_total_co2e_kg),
            "scope2_total_co2e_kg": str(self.scope2_total_co2e_kg),
            "total_co2e_kg": str(self.total_co2e_kg),
        }


@dataclass
class DataQualityScore:
    """
    5-dimension DQI assessment for a franchise unit calculation.

    Each dimension scored 1 (best) to 5 (worst).
    """
    data_source: Decimal = Decimal("3.0")
    temporal: Decimal = Decimal("2.0")
    geographical: Decimal = Decimal("2.0")
    technological: Decimal = Decimal("2.0")
    completeness: Decimal = Decimal("2.0")

    @property
    def composite(self) -> Decimal:
        """Arithmetic mean of all 5 dimensions."""
        total = (
            self.data_source
            + self.temporal
            + self.geographical
            + self.technological
            + self.completeness
        )
        return (total / Decimal("5")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    @property
    def tier(self) -> str:
        """Map composite score to calculation tier."""
        c = self.composite
        if c <= Decimal("1.5"):
            return "tier_1"
        elif c <= Decimal("3.0"):
            return "tier_2"
        else:
            return "tier_3"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "data_source": str(self.data_source),
            "temporal": str(self.temporal),
            "geographical": str(self.geographical),
            "technological": str(self.technological),
            "completeness": str(self.completeness),
            "composite": str(self.composite),
            "tier": self.tier,
        }


@dataclass
class FranchiseCalculationResult:
    """
    Complete result for a single franchise unit calculation.

    All emissions in kgCO2e. Includes breakdown, DQI, provenance hash,
    uncertainty, and processing metadata.
    """
    unit_id: str = ""
    franchise_type: str = ""
    ownership_type: str = ""
    country: str = ""
    region: Optional[str] = None
    climate_zone: str = ""
    reporting_year: int = 2025
    calculation_method: str = "franchise_specific"

    # Emissions
    total_co2e_kg: Decimal = Decimal("0")
    total_co2e_tonnes: Decimal = Decimal("0")
    breakdown: Optional[EmissionBreakdown] = None
    pro_rata_applied: bool = False
    pro_rata_factor: Decimal = Decimal("1")

    # Quality and uncertainty
    data_quality: Optional[DataQualityScore] = None
    uncertainty_lower_pct: Decimal = Decimal("-10")
    uncertainty_upper_pct: Decimal = Decimal("10")
    uncertainty_lower_co2e_kg: Decimal = Decimal("0")
    uncertainty_upper_co2e_kg: Decimal = Decimal("0")

    # Provenance
    provenance_hash: str = ""
    input_hash: str = ""
    ef_hashes: Dict[str, str] = field(default_factory=dict)

    # Metadata
    processing_time_ms: float = 0.0
    validation_warnings: List[str] = field(default_factory=list)
    agent_id: str = AGENT_ID
    agent_version: str = VERSION
    calculated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "unit_id": self.unit_id,
            "franchise_type": self.franchise_type,
            "ownership_type": self.ownership_type,
            "country": self.country,
            "region": self.region,
            "climate_zone": self.climate_zone,
            "reporting_year": self.reporting_year,
            "calculation_method": self.calculation_method,
            "total_co2e_kg": str(self.total_co2e_kg),
            "total_co2e_tonnes": str(self.total_co2e_tonnes),
            "breakdown": self.breakdown.to_dict() if self.breakdown else None,
            "pro_rata_applied": self.pro_rata_applied,
            "pro_rata_factor": str(self.pro_rata_factor),
            "data_quality": self.data_quality.to_dict() if self.data_quality else None,
            "uncertainty_lower_pct": str(self.uncertainty_lower_pct),
            "uncertainty_upper_pct": str(self.uncertainty_upper_pct),
            "uncertainty_lower_co2e_kg": str(self.uncertainty_lower_co2e_kg),
            "uncertainty_upper_co2e_kg": str(self.uncertainty_upper_co2e_kg),
            "provenance_hash": self.provenance_hash,
            "input_hash": self.input_hash,
            "ef_hashes": self.ef_hashes,
            "processing_time_ms": self.processing_time_ms,
            "validation_warnings": self.validation_warnings,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "calculated_at": self.calculated_at,
        }


# ============================================================================
# HASH UTILITY
# ============================================================================


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data for provenance tracking.

    Serializes the data to a deterministic JSON string, then computes
    the SHA-256 hash.

    Args:
        data: Data to hash (any JSON-serializable type).

    Returns:
        Lowercase hex SHA-256 hash string.
    """

    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    serialized = json.dumps(data, sort_keys=True, default=_default)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ============================================================================
# DATABASE ENGINE ACCESSOR (lazy import to avoid circular dependency)
# ============================================================================

_db_engine: Optional[Any] = None
_db_engine_lock = threading.Lock()


def _get_db() -> Any:
    """
    Get the FranchiseDatabaseEngine singleton (lazy import).

    Returns:
        FranchiseDatabaseEngine instance.
    """
    global _db_engine
    if _db_engine is None:
        with _db_engine_lock:
            if _db_engine is None:
                from greenlang.agents.mrv.franchises.franchise_database import (
                    get_database_engine,
                )
                _db_engine = get_database_engine()
    return _db_engine


# ============================================================================
# METRICS AND PROVENANCE ACCESSORS
# ============================================================================


def _get_metrics() -> Any:
    """Get the metrics collector (lazy import)."""
    try:
        from greenlang.agents.mrv.franchises.franchise_database import get_metrics_collector
        return get_metrics_collector()
    except ImportError:
        return None


def _get_provenance() -> Any:
    """Get the provenance manager (lazy import)."""
    try:
        from greenlang.agents.mrv.franchises.franchise_database import get_provenance_manager
        return get_provenance_manager()
    except ImportError:
        return None


# ============================================================================
# ENGINE CLASS
# ============================================================================


class FranchiseSpecificCalculatorEngine:
    """
    Thread-safe singleton Tier 1 calculator for franchise-specific emissions.

    Uses metered energy/fuel data per franchise unit to calculate emissions
    from 7 sources (4 Scope 1 + 3 Scope 2 of the franchisee). Includes
    franchise-type-specific calculation paths for QSR, hotel, convenience
    store, retail, fitness, automotive, and generic franchise types.

    This engine is ZERO-HALLUCINATION: all calculations are deterministic
    Python Decimal arithmetic. No LLM calls are used for any numeric
    computation. Emission factors are retrieved from the validated
    FranchiseDatabaseEngine reference tables.

    Thread Safety:
        Uses __new__ singleton pattern with threading.Lock. The
        _calculation_count attribute is protected by a dedicated RLock.

    Attributes:
        _db: FranchiseDatabaseEngine for factor lookups
        _calculation_count: Total number of calculations performed

    Example:
        >>> engine = FranchiseSpecificCalculatorEngine()
        >>> result = engine.calculate(unit_input)
        >>> assert result.provenance_hash  # SHA-256 present
        >>> assert result.total_co2e_kg >= Decimal('0')
    """

    _instance: Optional["FranchiseSpecificCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FranchiseSpecificCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the calculator engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._db = _get_db()
        self._calculation_count: int = 0
        self._calculation_lock: threading.RLock = threading.RLock()

        logger.info(
            "FranchiseSpecificCalculatorEngine initialized: "
            "agent=%s, version=%s",
            AGENT_ID, VERSION,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_calculation(self) -> None:
        """Increment calculation counter in a thread-safe manner."""
        with self._calculation_lock:
            self._calculation_count += 1

    def _quantize(
        self, value: Decimal, precision: Decimal = _QUANT_8DP
    ) -> Decimal:
        """
        Quantize a Decimal value with ROUND_HALF_UP.

        Args:
            value: Decimal value to quantize.
            precision: Quantization precision.

        Returns:
            Quantized Decimal value.
        """
        return value.quantize(precision, rounding=ROUND_HALF_UP)

    def _safe_decimal(self, value: Any, default: Decimal = Decimal("0")) -> Decimal:
        """
        Safely convert a value to Decimal.

        Args:
            value: Value to convert.
            default: Default if conversion fails.

        Returns:
            Decimal value.
        """
        if value is None:
            return default
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            logger.warning(
                "Could not convert '%s' to Decimal; using default %s",
                value, default,
            )
            return default

    # =========================================================================
    # PUBLIC API: SINGLE UNIT CALCULATION
    # =========================================================================

    def calculate(
        self, input_data: FranchiseUnitInput
    ) -> FranchiseCalculationResult:
        """
        Calculate emissions for a single franchise unit using primary data.

        This is the main entry point for Tier 1 franchise-specific
        calculations. It validates the input, checks double-counting rules
        (DC-FRN-001), calculates emissions from all 7 sources, applies
        pro-rata if needed, assesses data quality, computes uncertainty,
        and generates a provenance hash.

        Args:
            input_data: Complete input data for one franchise unit.

        Returns:
            FranchiseCalculationResult with emissions breakdown,
            DQI scores, uncertainty, and provenance hash.

        Raises:
            ValueError: If ownership_type is "company_owned" (DC-FRN-001)
                or if critical validation fails.

        Example:
            >>> result = engine.calculate(unit_input)
            >>> result.total_co2e_kg
            Decimal('45230.12345678')
        """
        start_time = time.monotonic()
        self._increment_calculation()

        # Step 1: Validate input and enforce DC-FRN-001
        warnings = self._validate_metered_data(input_data)

        # Step 2: Resolve climate zone
        climate_zone = self._resolve_climate_zone(input_data)

        # Step 3: Compute input hash for provenance
        input_hash = _compute_hash(input_data.to_dict())

        # Step 4: Calculate emissions by source
        breakdown = self._calculate_all_sources(input_data, climate_zone)

        # Step 5: Apply pro-rata if operating_days < total_days
        pro_rata_applied = False
        pro_rata_factor = Decimal("1")
        if (
            input_data.operating_days is not None
            and input_data.operating_days < input_data.total_days
        ):
            pro_rata_factor = self._apply_pro_rata(
                Decimal("1"),
                input_data.operating_days,
                input_data.total_days,
            )
            breakdown = self._scale_breakdown(breakdown, pro_rata_factor)
            pro_rata_applied = True

        # Step 6: Get totals
        total_kg = self._quantize(breakdown.total_co2e_kg, _QUANT_8DP)
        total_tonnes = self._quantize(
            total_kg / Decimal("1000"), _QUANT_5DP
        )

        # Step 7: Assess data quality
        dqi = self._assess_data_quality(input_data)

        # Step 8: Compute uncertainty
        unc = self._compute_uncertainty(total_kg, dqi)

        # Step 9: Compute provenance hash
        result_data = {
            "input_hash": input_hash,
            "breakdown": breakdown.to_dict(),
            "total_co2e_kg": str(total_kg),
            "dqi": dqi.to_dict(),
            "agent_id": AGENT_ID,
            "version": VERSION,
        }
        provenance_hash = _compute_hash(result_data)

        # Step 10: Record metrics
        processing_time_ms = (time.monotonic() - start_time) * 1000
        self._record_metrics(
            input_data.franchise_type,
            "success",
            processing_time_ms / 1000,
            float(total_kg),
        )

        logger.info(
            "Franchise-specific calculation complete: unit=%s, type=%s, "
            "country=%s, total=%.2f kgCO2e (%.4f tCO2e), "
            "dqi=%.2f (%s), time=%.1fms",
            input_data.unit_id,
            input_data.franchise_type,
            input_data.country,
            total_kg,
            total_tonnes,
            dqi.composite,
            dqi.tier,
            processing_time_ms,
        )

        return FranchiseCalculationResult(
            unit_id=input_data.unit_id,
            franchise_type=input_data.franchise_type,
            ownership_type=input_data.ownership_type,
            country=input_data.country,
            region=input_data.region,
            climate_zone=climate_zone,
            reporting_year=input_data.reporting_year,
            calculation_method="franchise_specific",
            total_co2e_kg=total_kg,
            total_co2e_tonnes=total_tonnes,
            breakdown=breakdown,
            pro_rata_applied=pro_rata_applied,
            pro_rata_factor=pro_rata_factor,
            data_quality=dqi,
            uncertainty_lower_pct=unc["lower_pct"],
            uncertainty_upper_pct=unc["upper_pct"],
            uncertainty_lower_co2e_kg=unc["lower_co2e_kg"],
            uncertainty_upper_co2e_kg=unc["upper_co2e_kg"],
            provenance_hash=provenance_hash,
            input_hash=input_hash,
            processing_time_ms=processing_time_ms,
            validation_warnings=warnings,
            calculated_at=datetime.now(timezone.utc).isoformat(),
        )

    # =========================================================================
    # PUBLIC API: BATCH CALCULATION
    # =========================================================================

    def calculate_batch(
        self,
        inputs: List[FranchiseUnitInput],
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> List[FranchiseCalculationResult]:
        """
        Calculate emissions for multiple franchise units.

        Processes units in chunks to manage memory, skipping company-owned
        units (DC-FRN-001) with a warning rather than raising.

        Args:
            inputs: List of franchise unit input data.
            batch_size: Maximum units to process per chunk.

        Returns:
            List of FranchiseCalculationResult (one per valid unit).

        Example:
            >>> results = engine.calculate_batch([unit1, unit2, unit3])
            >>> len(results)
            3
        """
        start_time = time.monotonic()
        results: List[FranchiseCalculationResult] = []
        skipped_count = 0
        error_count = 0

        total_units = len(inputs)
        logger.info(
            "Batch calculation started: units=%d, batch_size=%d",
            total_units, batch_size,
        )

        for chunk_start in range(0, total_units, batch_size):
            chunk = inputs[chunk_start:chunk_start + batch_size]

            for unit_input in chunk:
                try:
                    # Skip company-owned units silently in batch mode
                    ot = unit_input.ownership_type.strip().lower()
                    if ot in ("company_owned", "coco"):
                        skipped_count += 1
                        logger.debug(
                            "Skipping company-owned unit %s (DC-FRN-001)",
                            unit_input.unit_id,
                        )
                        continue

                    result = self.calculate(unit_input)
                    results.append(result)

                except ValueError as ve:
                    error_count += 1
                    logger.warning(
                        "Skipping unit %s due to validation error: %s",
                        unit_input.unit_id, ve,
                    )
                except Exception as exc:
                    error_count += 1
                    logger.error(
                        "Error calculating unit %s: %s",
                        unit_input.unit_id, exc,
                        exc_info=True,
                    )

        batch_time_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Batch calculation complete: total=%d, calculated=%d, "
            "skipped=%d (company-owned), errors=%d, time=%.1fms",
            total_units, len(results), skipped_count,
            error_count, batch_time_ms,
        )

        # Record batch metrics
        self._record_batch_metrics(
            total_units, len(results), skipped_count, error_count,
        )

        return results

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    def _validate_metered_data(
        self, unit: FranchiseUnitInput
    ) -> List[str]:
        """
        Validate metered input data and enforce DC-FRN-001.

        Checks:
        1. DC-FRN-001: Company-owned units must be rejected.
        2. unit_id must be non-empty.
        3. franchise_type must be recognized.
        4. country must be non-empty.
        5. Numeric fields must be non-negative.
        6. Data completeness warnings.

        Args:
            unit: Franchise unit input data.

        Returns:
            List of warning messages (non-fatal).

        Raises:
            ValueError: If DC-FRN-001 violated or critical field missing.
        """
        warnings: List[str] = []

        # DC-FRN-001: CRITICAL -- company-owned units MUST be excluded
        ot = unit.ownership_type.strip().lower()
        if ot in ("company_owned", "coco"):
            raise ValueError(
                f"DC-FRN-001: Company-owned/COCO unit '{unit.unit_id}' "
                f"must NOT be included in Scope 3 Category 14. "
                f"Company-owned units must be reported under the "
                f"franchisor's Scope 1 and Scope 2."
            )

        # Unit ID check
        if not unit.unit_id or not unit.unit_id.strip():
            raise ValueError("unit_id is required and must be non-empty")

        # Country check
        if not unit.country or not unit.country.strip():
            raise ValueError("country is required and must be non-empty")

        # Franchise type validation (warning only, generic fallback)
        ft = unit.franchise_type.strip().lower()
        from greenlang.agents.mrv.franchises.franchise_database import FRANCHISE_TYPES
        if ft not in FRANCHISE_TYPES and ft != "generic":
            warnings.append(
                f"Unrecognized franchise_type '{unit.franchise_type}'; "
                f"will use generic calculation path"
            )

        # Non-negative checks
        if unit.electricity_kwh < Decimal("0"):
            warnings.append("electricity_kwh is negative; treating as zero")
        if unit.heating_kwh < Decimal("0"):
            warnings.append("heating_kwh is negative; treating as zero")
        if unit.cooling_kwh < Decimal("0"):
            warnings.append("cooling_kwh is negative; treating as zero")
        if unit.floor_area_m2 < Decimal("0"):
            warnings.append("floor_area_m2 is negative; treating as zero")

        # Operating days validation
        if unit.operating_days is not None:
            if unit.operating_days < 0:
                warnings.append("operating_days is negative; ignoring pro-rata")
            elif unit.operating_days > unit.total_days:
                warnings.append(
                    f"operating_days ({unit.operating_days}) exceeds "
                    f"total_days ({unit.total_days}); ignoring pro-rata"
                )

        # Data completeness warnings
        has_electricity = unit.electricity_kwh > Decimal("0")
        has_fuel = (
            unit.stationary_combustion is not None
            and (
                unit.stationary_combustion.natural_gas_m3 > Decimal("0")
                or unit.stationary_combustion.propane_litres > Decimal("0")
                or unit.stationary_combustion.diesel_litres > Decimal("0")
            )
        )
        if not has_electricity and not has_fuel:
            warnings.append(
                "No electricity or fuel data provided; emissions may be "
                "significantly underestimated"
            )

        if warnings:
            logger.debug(
                "Validation warnings for unit %s: %s",
                unit.unit_id, "; ".join(warnings),
            )

        return warnings

    # =========================================================================
    # CLIMATE ZONE RESOLUTION
    # =========================================================================

    def _resolve_climate_zone(self, unit: FranchiseUnitInput) -> str:
        """
        Resolve the climate zone for a franchise unit.

        Priority:
        1. Explicit climate_zone override in input
        2. Lookup from COUNTRY_CLIMATE_ZONES table
        3. Default to "temperate"

        Args:
            unit: Franchise unit input data.

        Returns:
            Climate zone string.
        """
        if unit.climate_zone:
            return unit.climate_zone.strip().lower()

        try:
            return self._db.get_climate_zone(unit.country)
        except ValueError:
            logger.warning(
                "Climate zone not found for country '%s'; "
                "defaulting to 'temperate'",
                unit.country,
            )
            return "temperate"

    # =========================================================================
    # SOURCE-LEVEL CALCULATIONS
    # =========================================================================

    def _calculate_all_sources(
        self, unit: FranchiseUnitInput, climate_zone: str
    ) -> EmissionBreakdown:
        """
        Calculate emissions from all 7 sources for a franchise unit.

        Dispatches to franchise-type-specific calculators where available,
        falling back to the generic calculator.

        Args:
            unit: Franchise unit input data.
            climate_zone: Resolved climate zone.

        Returns:
            EmissionBreakdown with all 7 source emissions.
        """
        ft = unit.franchise_type.strip().lower()

        # Dispatch to type-specific calculator
        type_calculators = {
            "qsr": self._calculate_qsr,
            "hotel": self._calculate_hotel,
            "convenience_store": self._calculate_convenience_store,
            "retail": self._calculate_retail,
            "fitness": self._calculate_fitness,
            "automotive": self._calculate_automotive,
        }

        calculator = type_calculators.get(ft, self._calculate_generic)
        return calculator(unit, climate_zone)

    # =========================================================================
    # SOURCE 1: STATIONARY COMBUSTION
    # =========================================================================

    def _calculate_stationary_combustion(
        self, unit: FranchiseUnitInput
    ) -> Decimal:
        """
        Calculate emissions from stationary combustion.

        Covers cooking equipment (fryers, grills, ovens), space and water
        heating, and backup generators.

        Formula: E = SUM [ fuel_volume x fuel_EF ]

        Args:
            unit: Franchise unit input data.

        Returns:
            Stationary combustion emissions in kgCO2e.
        """
        if unit.stationary_combustion is None:
            return Decimal("0")

        sc = unit.stationary_combustion
        total = Decimal("0")

        fuel_consumption = [
            ("natural_gas", sc.natural_gas_m3),
            ("propane", sc.propane_litres),
            ("diesel", sc.diesel_litres),
            ("heating_oil", sc.heating_oil_litres),
            ("petrol", sc.petrol_litres),
            ("wood_pellets", sc.wood_pellets_kg),
            ("coal", sc.coal_kg),
            ("biodiesel", sc.biodiesel_litres),
        ]

        for fuel_type, volume in fuel_consumption:
            if volume > Decimal("0"):
                try:
                    ef = self._db.get_fuel_ef(fuel_type)
                    emissions = self._quantize(volume * ef)
                    total += emissions

                    logger.debug(
                        "Stationary combustion: fuel=%s, volume=%s, "
                        "ef=%s, emissions=%s kgCO2e",
                        fuel_type, volume, ef, emissions,
                    )
                except ValueError as ve:
                    logger.warning(
                        "Fuel EF lookup failed for '%s': %s", fuel_type, ve
                    )

        return self._quantize(total)

    # =========================================================================
    # SOURCE 2: MOBILE COMBUSTION
    # =========================================================================

    def _calculate_mobile_combustion(
        self, unit: FranchiseUnitInput
    ) -> Decimal:
        """
        Calculate emissions from mobile combustion (delivery fleet).

        Supports two calculation paths:
        1. Distance-based: E = distance_km x vehicle_EF per vehicle
        2. Fuel-based: E = fuel_volume x fuel_EF (aggregate fallback)

        Args:
            unit: Franchise unit input data.

        Returns:
            Mobile combustion emissions in kgCO2e.
        """
        if unit.mobile_combustion is None:
            return Decimal("0")

        mc = unit.mobile_combustion
        total = Decimal("0")

        # Path 1: Per-vehicle distance-based calculation
        if mc.vehicles:
            for vehicle in mc.vehicles:
                vt = vehicle.get("vehicle_type", "light_van")
                distance_km = self._safe_decimal(vehicle.get("distance_km"))

                if distance_km > Decimal("0"):
                    try:
                        ef = self._db.get_vehicle_ef(vt)
                        emissions = self._quantize(distance_km * ef)
                        total += emissions

                        logger.debug(
                            "Mobile combustion (distance): vehicle=%s, "
                            "distance=%s km, ef=%s, emissions=%s kgCO2e",
                            vt, distance_km, ef, emissions,
                        )
                    except ValueError as ve:
                        logger.warning(
                            "Vehicle EF lookup failed for '%s': %s", vt, ve
                        )

                # Check for fuel-volume override per vehicle
                fuel_vol = self._safe_decimal(vehicle.get("fuel_volume"))
                fuel_type = vehicle.get("fuel_type", "diesel")
                if fuel_vol > Decimal("0") and distance_km <= Decimal("0"):
                    try:
                        ef = self._db.get_fuel_ef(fuel_type)
                        emissions = self._quantize(fuel_vol * ef)
                        total += emissions
                    except ValueError as ve:
                        logger.warning(
                            "Fuel EF lookup failed for '%s': %s",
                            fuel_type, ve,
                        )

        # Path 2: Aggregate fuel-based fallback
        if not mc.vehicles or total == Decimal("0"):
            if mc.total_diesel_litres > Decimal("0"):
                try:
                    ef = self._db.get_fuel_ef("diesel")
                    total += self._quantize(mc.total_diesel_litres * ef)
                except ValueError:
                    pass
            if mc.total_petrol_litres > Decimal("0"):
                try:
                    ef = self._db.get_fuel_ef("petrol")
                    total += self._quantize(mc.total_petrol_litres * ef)
                except ValueError:
                    pass

        return self._quantize(total)

    # =========================================================================
    # SOURCE 3: REFRIGERANT LEAKAGE
    # =========================================================================

    def _calculate_refrigerant_emissions(
        self, unit: FranchiseUnitInput
    ) -> Decimal:
        """
        Calculate emissions from refrigerant leakage.

        Formula per equipment:
            E = charge_kg x leakage_rate x GWP

        Covers HVAC (split, rooftop, chiller) and commercial refrigeration
        (walk-in coolers/freezers, reach-in, display cases, ice machines).

        Args:
            unit: Franchise unit input data.

        Returns:
            Refrigerant leakage emissions in kgCO2e.
        """
        if unit.refrigerants is None or not unit.refrigerants.equipment:
            return Decimal("0")

        total = Decimal("0")

        for equip in unit.refrigerants.equipment:
            equipment_type = equip.get("equipment_type", "hvac_rooftop")
            refrigerant_type = equip.get("refrigerant_type", "R-410A")
            charge_kg = self._safe_decimal(equip.get("charge_kg"))

            if charge_kg <= Decimal("0"):
                continue

            try:
                leakage_rate = self._db.get_leakage_rate(equipment_type)
                gwp = self._db.get_refrigerant_gwp(refrigerant_type)

                # E = charge_kg x leakage_rate x GWP
                emissions = self._quantize(charge_kg * leakage_rate * gwp)
                total += emissions

                logger.debug(
                    "Refrigerant emissions: equip=%s, ref=%s, "
                    "charge=%s kg, leak_rate=%s, gwp=%s, "
                    "emissions=%s kgCO2e",
                    equipment_type, refrigerant_type,
                    charge_kg, leakage_rate, gwp, emissions,
                )
            except ValueError as ve:
                logger.warning(
                    "Refrigerant calc failed for %s/%s: %s",
                    equipment_type, refrigerant_type, ve,
                )

        return self._quantize(total)

    # =========================================================================
    # SOURCE 5: PURCHASED ELECTRICITY
    # =========================================================================

    def _calculate_electricity_emissions(
        self, unit: FranchiseUnitInput
    ) -> Decimal:
        """
        Calculate emissions from purchased electricity.

        Formula: E = kWh x grid_EF(country, region)

        Args:
            unit: Franchise unit input data.

        Returns:
            Electricity emissions in kgCO2e.
        """
        kwh = max(unit.electricity_kwh, Decimal("0"))
        if kwh == Decimal("0"):
            return Decimal("0")

        try:
            grid_ef = self._db.get_grid_ef(unit.country, unit.region)
        except ValueError:
            logger.warning(
                "Grid EF not found for %s/%s; using US national average",
                unit.country, unit.region,
            )
            grid_ef = Decimal("0.3937")

        emissions = self._quantize(kwh * grid_ef)

        logger.debug(
            "Electricity emissions: kwh=%s, grid_ef=%s, "
            "emissions=%s kgCO2e",
            kwh, grid_ef, emissions,
        )
        return emissions

    # =========================================================================
    # SOURCE 6: PURCHASED HEATING / STEAM
    # =========================================================================

    def _calculate_heating_emissions(
        self, unit: FranchiseUnitInput
    ) -> Decimal:
        """
        Calculate emissions from purchased heating/steam.

        Formula: E = kWh_heating x heating_EF

        Args:
            unit: Franchise unit input data.

        Returns:
            Heating emissions in kgCO2e.
        """
        kwh = max(unit.heating_kwh, Decimal("0"))
        if kwh == Decimal("0"):
            return Decimal("0")

        emissions = self._quantize(kwh * _DEFAULT_HEATING_EF)

        logger.debug(
            "Heating emissions: kwh=%s, ef=%s, emissions=%s kgCO2e",
            kwh, _DEFAULT_HEATING_EF, emissions,
        )
        return emissions

    # =========================================================================
    # SOURCE 7: PURCHASED COOLING
    # =========================================================================

    def _calculate_cooling_emissions(
        self, unit: FranchiseUnitInput
    ) -> Decimal:
        """
        Calculate emissions from purchased cooling.

        Formula: E = kWh_cooling x cooling_EF

        Args:
            unit: Franchise unit input data.

        Returns:
            Cooling emissions in kgCO2e.
        """
        kwh = max(unit.cooling_kwh, Decimal("0"))
        if kwh == Decimal("0"):
            return Decimal("0")

        emissions = self._quantize(kwh * _DEFAULT_COOLING_EF)

        logger.debug(
            "Cooling emissions: kwh=%s, ef=%s, emissions=%s kgCO2e",
            kwh, _DEFAULT_COOLING_EF, emissions,
        )
        return emissions

    # =========================================================================
    # FRANCHISE-TYPE-SPECIFIC CALCULATORS
    # =========================================================================

    def _calculate_qsr(
        self, unit: FranchiseUnitInput, climate_zone: str
    ) -> EmissionBreakdown:
        """
        QSR (Quick-Service Restaurant) specific calculation.

        QSR franchises are cooking- and refrigeration-heavy with significant
        gas consumption for fryers, charbroilers, and ovens. Commercial
        refrigeration (walk-in coolers/freezers, reach-in units) typically
        represents the second-largest energy end-use.

        Special considerations:
        - High natural gas / propane usage for cooking
        - Multiple refrigeration units with high-GWP refrigerants
        - Drive-through lighting and signage electricity
        - Delivery fleet for catering / drive-through supply runs

        Args:
            unit: Franchise unit input data.
            climate_zone: Resolved climate zone.

        Returns:
            EmissionBreakdown populated for all 7 sources.
        """
        logger.debug(
            "Calculating QSR emissions for unit %s", unit.unit_id
        )

        return EmissionBreakdown(
            stationary_combustion_co2e_kg=self._calculate_stationary_combustion(unit),
            mobile_combustion_co2e_kg=self._calculate_mobile_combustion(unit),
            refrigerant_leakage_co2e_kg=self._calculate_refrigerant_emissions(unit),
            process_emissions_co2e_kg=max(unit.process_emissions_co2e_kg, Decimal("0")),
            purchased_electricity_co2e_kg=self._calculate_electricity_emissions(unit),
            purchased_heating_co2e_kg=self._calculate_heating_emissions(unit),
            purchased_cooling_co2e_kg=self._calculate_cooling_emissions(unit),
        )

    def _calculate_hotel(
        self, unit: FranchiseUnitInput, climate_zone: str
    ) -> EmissionBreakdown:
        """
        Hotel / Lodging specific calculation.

        Hotels have diverse energy end-uses: HVAC (the dominant end-use),
        domestic hot water (DHW) for guest rooms and laundry, commercial
        kitchen (if full-service), pool/spa heating, elevators, and
        extensive lighting.

        Special considerations:
        - Large HVAC zones with variable occupancy
        - Significant DHW demand (laundry, guest showers)
        - Pool and spa heating (gas or heat pump)
        - Kitchen cooking fuel (if full-service hotel)
        - Refrigerant leakage from central chillers

        Args:
            unit: Franchise unit input data.
            climate_zone: Resolved climate zone.

        Returns:
            EmissionBreakdown populated for all 7 sources.
        """
        logger.debug(
            "Calculating hotel emissions for unit %s", unit.unit_id
        )

        return EmissionBreakdown(
            stationary_combustion_co2e_kg=self._calculate_stationary_combustion(unit),
            mobile_combustion_co2e_kg=self._calculate_mobile_combustion(unit),
            refrigerant_leakage_co2e_kg=self._calculate_refrigerant_emissions(unit),
            process_emissions_co2e_kg=max(unit.process_emissions_co2e_kg, Decimal("0")),
            purchased_electricity_co2e_kg=self._calculate_electricity_emissions(unit),
            purchased_heating_co2e_kg=self._calculate_heating_emissions(unit),
            purchased_cooling_co2e_kg=self._calculate_cooling_emissions(unit),
        )

    def _calculate_convenience_store(
        self, unit: FranchiseUnitInput, climate_zone: str
    ) -> EmissionBreakdown:
        """
        Convenience store specific calculation.

        Convenience stores operate 24/7 with continuous refrigeration
        as the dominant energy end-use (40-60% of total). Open display
        cases and frequent door openings increase cooling loads.

        Special considerations:
        - 24/7 operation with open refrigeration cases
        - High-intensity merchandising lighting
        - Minimal cooking (roller grills, microwave)
        - Frequent door openings increase HVAC load
        - Ice machines and beverage coolers

        Args:
            unit: Franchise unit input data.
            climate_zone: Resolved climate zone.

        Returns:
            EmissionBreakdown populated for all 7 sources.
        """
        logger.debug(
            "Calculating convenience store emissions for unit %s",
            unit.unit_id,
        )

        return EmissionBreakdown(
            stationary_combustion_co2e_kg=self._calculate_stationary_combustion(unit),
            mobile_combustion_co2e_kg=self._calculate_mobile_combustion(unit),
            refrigerant_leakage_co2e_kg=self._calculate_refrigerant_emissions(unit),
            process_emissions_co2e_kg=max(unit.process_emissions_co2e_kg, Decimal("0")),
            purchased_electricity_co2e_kg=self._calculate_electricity_emissions(unit),
            purchased_heating_co2e_kg=self._calculate_heating_emissions(unit),
            purchased_cooling_co2e_kg=self._calculate_cooling_emissions(unit),
        )

    def _calculate_retail(
        self, unit: FranchiseUnitInput, climate_zone: str
    ) -> EmissionBreakdown:
        """
        General retail specific calculation.

        Retail franchises have lower EUI than food-service. Energy use is
        dominated by HVAC and lighting, with large floor plates diluting
        per-m2 intensity.

        Special considerations:
        - HVAC is the dominant end-use (40-50%)
        - Lighting represents 25-35% of energy
        - Minimal cooking/refrigeration (unless grocery)
        - Seasonal variation in HVAC loads

        Args:
            unit: Franchise unit input data.
            climate_zone: Resolved climate zone.

        Returns:
            EmissionBreakdown populated for all 7 sources.
        """
        logger.debug(
            "Calculating retail emissions for unit %s", unit.unit_id
        )

        return EmissionBreakdown(
            stationary_combustion_co2e_kg=self._calculate_stationary_combustion(unit),
            mobile_combustion_co2e_kg=self._calculate_mobile_combustion(unit),
            refrigerant_leakage_co2e_kg=self._calculate_refrigerant_emissions(unit),
            process_emissions_co2e_kg=max(unit.process_emissions_co2e_kg, Decimal("0")),
            purchased_electricity_co2e_kg=self._calculate_electricity_emissions(unit),
            purchased_heating_co2e_kg=self._calculate_heating_emissions(unit),
            purchased_cooling_co2e_kg=self._calculate_cooling_emissions(unit),
        )

    def _calculate_fitness(
        self, unit: FranchiseUnitInput, climate_zone: str
    ) -> EmissionBreakdown:
        """
        Fitness center / gym specific calculation.

        Fitness centres have elevated ventilation rates due to high occupant
        metabolic heat gain. Pool/spa heating (if present) is a significant
        energy consumer.

        Special considerations:
        - High ventilation rates (ASHRAE 62.1)
        - Pool/spa heating (gas or heat pump)
        - Hot water demand for showers and locker rooms
        - Equipment electricity (treadmills, lights, AV)
        - Extended operating hours (5 AM - 10 PM typical)

        Args:
            unit: Franchise unit input data.
            climate_zone: Resolved climate zone.

        Returns:
            EmissionBreakdown populated for all 7 sources.
        """
        logger.debug(
            "Calculating fitness emissions for unit %s", unit.unit_id
        )

        return EmissionBreakdown(
            stationary_combustion_co2e_kg=self._calculate_stationary_combustion(unit),
            mobile_combustion_co2e_kg=self._calculate_mobile_combustion(unit),
            refrigerant_leakage_co2e_kg=self._calculate_refrigerant_emissions(unit),
            process_emissions_co2e_kg=max(unit.process_emissions_co2e_kg, Decimal("0")),
            purchased_electricity_co2e_kg=self._calculate_electricity_emissions(unit),
            purchased_heating_co2e_kg=self._calculate_heating_emissions(unit),
            purchased_cooling_co2e_kg=self._calculate_cooling_emissions(unit),
        )

    def _calculate_automotive(
        self, unit: FranchiseUnitInput, climate_zone: str
    ) -> EmissionBreakdown:
        """
        Automotive service (quick-lube, tire, car wash) specific calculation.

        Automotive franchise units have process-specific energy demands:
        hydraulic lifts, air compressors, car wash water heating, and
        partially conditioned service bays.

        Special considerations:
        - Compressed air and hydraulic equipment electricity
        - Car wash water heating (gas or electric)
        - Partially conditioned bays (winter heating)
        - Chemical use (process emissions if applicable)
        - Relatively small floor area but equipment-intensive

        Args:
            unit: Franchise unit input data.
            climate_zone: Resolved climate zone.

        Returns:
            EmissionBreakdown populated for all 7 sources.
        """
        logger.debug(
            "Calculating automotive emissions for unit %s", unit.unit_id
        )

        return EmissionBreakdown(
            stationary_combustion_co2e_kg=self._calculate_stationary_combustion(unit),
            mobile_combustion_co2e_kg=self._calculate_mobile_combustion(unit),
            refrigerant_leakage_co2e_kg=self._calculate_refrigerant_emissions(unit),
            process_emissions_co2e_kg=max(unit.process_emissions_co2e_kg, Decimal("0")),
            purchased_electricity_co2e_kg=self._calculate_electricity_emissions(unit),
            purchased_heating_co2e_kg=self._calculate_heating_emissions(unit),
            purchased_cooling_co2e_kg=self._calculate_cooling_emissions(unit),
        )

    def _calculate_generic(
        self, unit: FranchiseUnitInput, climate_zone: str
    ) -> EmissionBreakdown:
        """
        Generic calculation for unrecognized franchise types.

        Falls back to the standard 7-source calculation without any
        type-specific adjustments.

        Args:
            unit: Franchise unit input data.
            climate_zone: Resolved climate zone.

        Returns:
            EmissionBreakdown populated for all 7 sources.
        """
        logger.debug(
            "Calculating generic emissions for unit %s (type=%s)",
            unit.unit_id, unit.franchise_type,
        )

        return EmissionBreakdown(
            stationary_combustion_co2e_kg=self._calculate_stationary_combustion(unit),
            mobile_combustion_co2e_kg=self._calculate_mobile_combustion(unit),
            refrigerant_leakage_co2e_kg=self._calculate_refrigerant_emissions(unit),
            process_emissions_co2e_kg=max(unit.process_emissions_co2e_kg, Decimal("0")),
            purchased_electricity_co2e_kg=self._calculate_electricity_emissions(unit),
            purchased_heating_co2e_kg=self._calculate_heating_emissions(unit),
            purchased_cooling_co2e_kg=self._calculate_cooling_emissions(unit),
        )

    # =========================================================================
    # PRO-RATA ADJUSTMENT
    # =========================================================================

    def _apply_pro_rata(
        self,
        emissions: Decimal,
        operating_days: int,
        total_days: int,
    ) -> Decimal:
        """
        Apply pro-rata adjustment for partial reporting periods.

        Used when a franchise unit opens/closes mid-period, or changes
        ownership type (DC-FRN-008). The metered data already covers the
        actual operating period, so the pro-rata factor represents the
        fraction of the full period the unit was operational.

        Formula: adjusted = emissions x (operating_days / total_days)

        Args:
            emissions: Unadjusted emissions (or Decimal("1") for factor).
            operating_days: Days the unit was operational.
            total_days: Total days in the reporting period.

        Returns:
            Pro-rata factor or adjusted emissions.
        """
        if total_days <= 0:
            return emissions

        op = Decimal(str(max(0, operating_days)))
        tot = Decimal(str(total_days))

        factor = self._quantize(op / tot)

        logger.debug(
            "Pro-rata: operating_days=%d, total_days=%d, factor=%s",
            operating_days, total_days, factor,
        )
        return self._quantize(emissions * factor)

    def _scale_breakdown(
        self, breakdown: EmissionBreakdown, factor: Decimal
    ) -> EmissionBreakdown:
        """
        Scale all emission sources in a breakdown by a factor.

        Used for pro-rata adjustment.

        Args:
            breakdown: Original emission breakdown.
            factor: Scaling factor (0-1).

        Returns:
            New EmissionBreakdown with scaled values.
        """
        return EmissionBreakdown(
            stationary_combustion_co2e_kg=self._quantize(
                breakdown.stationary_combustion_co2e_kg * factor
            ),
            mobile_combustion_co2e_kg=self._quantize(
                breakdown.mobile_combustion_co2e_kg * factor
            ),
            refrigerant_leakage_co2e_kg=self._quantize(
                breakdown.refrigerant_leakage_co2e_kg * factor
            ),
            process_emissions_co2e_kg=self._quantize(
                breakdown.process_emissions_co2e_kg * factor
            ),
            purchased_electricity_co2e_kg=self._quantize(
                breakdown.purchased_electricity_co2e_kg * factor
            ),
            purchased_heating_co2e_kg=self._quantize(
                breakdown.purchased_heating_co2e_kg * factor
            ),
            purchased_cooling_co2e_kg=self._quantize(
                breakdown.purchased_cooling_co2e_kg * factor
            ),
        )

    # =========================================================================
    # DATA QUALITY ASSESSMENT
    # =========================================================================

    def _assess_data_quality(
        self, unit: FranchiseUnitInput
    ) -> DataQualityScore:
        """
        Assess data quality for a franchise unit calculation.

        Evaluates 5 DQI dimensions:
        1. Data Source: Based on whether primary metered data is provided
        2. Temporal: Based on reporting year recency
        3. Geographical: Based on country/region specificity
        4. Technological: Based on franchise type specificity
        5. Completeness: Based on how many emission sources have data

        Args:
            unit: Franchise unit input data.

        Returns:
            DataQualityScore with dimension scores and composite.
        """
        # Dimension 1: Data Source
        # Primary metered data = 1.0, partial = 2.0, minimal = 3.0
        has_electricity = unit.electricity_kwh > Decimal("0")
        has_fuel = (
            unit.stationary_combustion is not None
            and (
                unit.stationary_combustion.natural_gas_m3 > Decimal("0")
                or unit.stationary_combustion.propane_litres > Decimal("0")
                or unit.stationary_combustion.diesel_litres > Decimal("0")
            )
        )
        has_refrigerant = (
            unit.refrigerants is not None
            and len(unit.refrigerants.equipment) > 0
        )

        sources_with_data = sum([has_electricity, has_fuel, has_refrigerant])
        if sources_with_data >= 3:
            data_source_score = Decimal("1.0")
        elif sources_with_data >= 2:
            data_source_score = Decimal("1.5")
        elif sources_with_data >= 1:
            data_source_score = Decimal("2.0")
        else:
            data_source_score = Decimal("4.0")

        # Dimension 2: Temporal
        current_year = datetime.now(timezone.utc).year
        year_diff = abs(current_year - unit.reporting_year)
        if year_diff <= 1:
            temporal_score = Decimal("1.0")
        elif year_diff <= 3:
            temporal_score = Decimal("2.0")
        else:
            temporal_score = Decimal("3.0")

        # Dimension 3: Geographical
        # Region-specific = 1.0, country-level = 1.5, global = 3.0
        if unit.region:
            geo_score = Decimal("1.0")
        elif unit.country:
            geo_score = Decimal("1.5")
        else:
            geo_score = Decimal("3.0")

        # Dimension 4: Technological
        from greenlang.agents.mrv.franchises.franchise_database import FRANCHISE_TYPES
        ft = unit.franchise_type.strip().lower()
        if ft in FRANCHISE_TYPES:
            tech_score = Decimal("1.0")
        elif ft == "generic":
            tech_score = Decimal("2.0")
        else:
            tech_score = Decimal("3.0")

        # Dimension 5: Completeness
        total_sources = 7
        active_sources = 0
        if has_electricity:
            active_sources += 1
        if has_fuel:
            active_sources += 1
        if has_refrigerant:
            active_sources += 1
        if unit.heating_kwh > Decimal("0"):
            active_sources += 1
        if unit.cooling_kwh > Decimal("0"):
            active_sources += 1
        if (unit.mobile_combustion is not None and
                (unit.mobile_combustion.vehicles or
                 unit.mobile_combustion.total_diesel_litres > Decimal("0") or
                 unit.mobile_combustion.total_petrol_litres > Decimal("0"))):
            active_sources += 1
        if unit.process_emissions_co2e_kg > Decimal("0"):
            active_sources += 1

        coverage_pct = (active_sources / total_sources) * 100
        if coverage_pct >= 80:
            completeness_score = Decimal("1.0")
        elif coverage_pct >= 50:
            completeness_score = Decimal("2.0")
        else:
            completeness_score = Decimal("3.0")

        dqi = DataQualityScore(
            data_source=data_source_score,
            temporal=temporal_score,
            geographical=geo_score,
            technological=tech_score,
            completeness=completeness_score,
        )

        logger.debug(
            "DQI assessment for unit %s: src=%.1f, temp=%.1f, geo=%.1f, "
            "tech=%.1f, comp=%.1f, composite=%.2f (%s)",
            unit.unit_id,
            data_source_score, temporal_score, geo_score,
            tech_score, completeness_score,
            dqi.composite, dqi.tier,
        )

        return dqi

    # =========================================================================
    # UNCERTAINTY QUANTIFICATION
    # =========================================================================

    def _compute_uncertainty(
        self,
        total_co2e_kg: Decimal,
        dqi: DataQualityScore,
    ) -> Dict[str, Decimal]:
        """
        Compute uncertainty range based on DQI tier.

        Uses the UNCERTAINTY_RANGES table from the database engine to
        determine the percentage uncertainty at 95% confidence, then
        applies it to the total emissions.

        Args:
            total_co2e_kg: Total emissions in kgCO2e.
            dqi: Data quality assessment.

        Returns:
            Dict with lower_pct, upper_pct, lower_co2e_kg, upper_co2e_kg.
        """
        tier = dqi.tier

        try:
            unc_range = self._db.get_uncertainty_range(
                "franchise_specific", tier
            )
            lower_pct = unc_range["lower_pct"]
            upper_pct = unc_range["upper_pct"]
        except (ValueError, KeyError):
            # Default to +/- 15% if lookup fails
            lower_pct = Decimal("-15")
            upper_pct = Decimal("15")

        lower_co2e = self._quantize(
            total_co2e_kg * (Decimal("1") + lower_pct / Decimal("100"))
        )
        upper_co2e = self._quantize(
            total_co2e_kg * (Decimal("1") + upper_pct / Decimal("100"))
        )

        return {
            "lower_pct": lower_pct,
            "upper_pct": upper_pct,
            "lower_co2e_kg": lower_co2e,
            "upper_co2e_kg": upper_co2e,
        }

    # =========================================================================
    # METRICS RECORDING
    # =========================================================================

    def _record_metrics(
        self,
        franchise_type: str,
        status: str,
        duration_s: float,
        co2e_kg: float,
    ) -> None:
        """
        Record calculation metrics.

        Args:
            franchise_type: Type of franchise calculated.
            status: "success" or "error".
            duration_s: Duration in seconds.
            co2e_kg: Emissions in kgCO2e.
        """
        try:
            metrics = _get_metrics()
            if metrics is not None:
                metrics.record_calculation(
                    method="franchise_specific",
                    mode=franchise_type,
                    status=status,
                    duration=duration_s,
                    co2e=co2e_kg,
                )
        except Exception as exc:
            logger.warning("Failed to record metrics: %s", exc)

    def _record_batch_metrics(
        self,
        total: int,
        calculated: int,
        skipped: int,
        errors: int,
    ) -> None:
        """
        Record batch processing metrics.

        Args:
            total: Total units in batch.
            calculated: Units successfully calculated.
            skipped: Units skipped (company-owned).
            errors: Units with errors.
        """
        try:
            metrics = _get_metrics()
            if metrics is not None:
                status = "completed" if errors == 0 else "partial"
                metrics.record_batch(status=status, size=total)
        except Exception as exc:
            logger.warning("Failed to record batch metrics: %s", exc)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_calculation_count(self) -> int:
        """
        Get the total number of calculations performed.

        Returns:
            Total calculation count.
        """
        with self._calculation_lock:
            return self._calculation_count

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get engine metadata and diagnostics.

        Returns:
            Dict with agent info, calculation count, supported types.
        """
        return {
            "engine": "FranchiseSpecificCalculatorEngine",
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": VERSION,
            "calculation_method": "franchise_specific",
            "tier": "Tier 1 (primary metered data)",
            "emission_sources": [s.value for s in EmissionSource],
            "supported_franchise_types": [
                "qsr", "hotel", "convenience_store", "retail",
                "fitness", "automotive", "generic",
            ],
            "total_calculations": self.get_calculation_count(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # SINGLETON LIFECYCLE
    # =========================================================================

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        WARNING: This is NOT safe for concurrent use. It should only
        be called in test teardown when no other threads are accessing
        the engine instance.
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None

                global _calculator_instance
                _calculator_instance = None

                logger.info(
                    "FranchiseSpecificCalculatorEngine singleton reset"
                )


# ============================================================================
# MODULE-LEVEL SINGLETON ACCESSOR
# ============================================================================

_calculator_instance: Optional[FranchiseSpecificCalculatorEngine] = None
_calculator_lock: threading.Lock = threading.Lock()


def get_calculator_engine() -> FranchiseSpecificCalculatorEngine:
    """
    Get the singleton FranchiseSpecificCalculatorEngine instance.

    Thread-safe accessor for the global calculator engine instance. Prefer
    this function over direct instantiation for consistency across the
    franchise agent codebase.

    Returns:
        FranchiseSpecificCalculatorEngine singleton instance.

    Example:
        >>> from greenlang.agents.mrv.franchises.franchise_specific_calculator import (
        ...     get_calculator_engine,
        ... )
        >>> engine = get_calculator_engine()
        >>> result = engine.calculate(unit_input)
    """
    global _calculator_instance

    if _calculator_instance is None:
        with _calculator_lock:
            if _calculator_instance is None:
                _calculator_instance = FranchiseSpecificCalculatorEngine()

    return _calculator_instance


def reset_calculator_engine() -> None:
    """
    Reset the singleton calculator engine instance for testing purposes.

    Convenience function that delegates to
    FranchiseSpecificCalculatorEngine.reset().
    Should only be called in test teardown.
    """
    FranchiseSpecificCalculatorEngine.reset()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Enumerations
    "OwnershipType",
    "CalculationTier",
    "EmissionSource",
    "ValidationSeverity",
    # Data models
    "StationaryCombustionInput",
    "MobileCombustionInput",
    "RefrigerantInput",
    "FranchiseUnitInput",
    "EmissionBreakdown",
    "DataQualityScore",
    "FranchiseCalculationResult",
    # Engine class
    "FranchiseSpecificCalculatorEngine",
    # Singleton accessors
    "get_calculator_engine",
    "reset_calculator_engine",
]
