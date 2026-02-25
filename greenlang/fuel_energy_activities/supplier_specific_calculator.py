# -*- coding: utf-8 -*-
"""
SupplierSpecificCalculatorEngine - Engine 5: Fuel & Energy Activities Agent (AGENT-MRV-016)

Supplier-specific upstream emission calculations using direct supplier data
(EPDs, LCAs, MiQ certificates, OGMP 2.0 reports, PPAs, green tariffs) to
calculate upstream emissions with higher accuracy than average-data methods.

Applies to all three sub-activities of GHG Protocol Scope 3 Category 3:
  3a - Upstream emissions of purchased fuels (well-to-tank / WTT)
  3b - Upstream emissions of purchased electricity (generation lifecycle)
  3c - Transmission & distribution losses (T&D line losses)

Core Capabilities:
  - Supplier-specific WTT factor application for fuel upstream emissions
  - Supplier-specific upstream electricity EF from EPDs / generation mix
  - MiQ methane intensity grading (Grade A through F) with upstream adjustments
  - OGMP 2.0 reporting level validation (Level 1-5) with quality scoring
  - PPA / green tariff upstream EF calculation (construction/maintenance only)
  - Multi-product allocation (revenue, production, energy, mass, economic)
  - Blending of supplier-specific and average-data for partial coverage
  - Verification level scoring (third_party_verified > certified > self_declared)
  - Supplier coverage analysis (percentage of fuel/energy with supplier data)
  - DQI scoring based on supplier data quality characteristics
  - Uncertainty quantification with narrower ranges for verified supplier data
  - Comparison of supplier-specific vs average-data results

Zero-Hallucination Guarantees:
  - All calculations use Python Decimal (8 decimal places)
  - No LLM calls in the calculation path
  - Every step uses deterministic arithmetic from supplier-provided factors
  - SHA-256 provenance hash for every result

Example:
    >>> from greenlang.fuel_energy_activities.supplier_specific_calculator import (
    ...     SupplierSpecificCalculatorEngine,
    ... )
    >>> from greenlang.fuel_energy_activities.models import (
    ...     FuelConsumptionRecord, SupplierFuelData, FuelType,
    ... )
    >>> from decimal import Decimal
    >>> from datetime import date
    >>> engine = SupplierSpecificCalculatorEngine()
    >>> fuel_rec = FuelConsumptionRecord(
    ...     fuel_type=FuelType.NATURAL_GAS,
    ...     quantity=Decimal("10000"),
    ...     unit="kWh",
    ...     quantity_kwh=Decimal("10000"),
    ...     period_start=date(2025, 1, 1),
    ...     period_end=date(2025, 12, 31),
    ...     reporting_year=2025,
    ...     supplier_id="SUP-001",
    ... )
    >>> supplier = SupplierFuelData(
    ...     supplier_id="SUP-001",
    ...     supplier_name="Acme Gas Ltd",
    ...     fuel_type=FuelType.NATURAL_GAS,
    ...     upstream_ef=Decimal("0.01850"),
    ...     verification_level="third_party_verified",
    ...     miq_grade="A",
    ... )
    >>> result = engine.calculate_fuel_upstream(fuel_rec, supplier, "AR6")
    >>> assert result.emissions_total > Decimal("0")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-016 Fuel & Energy Activities (GL-MRV-S3-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.fuel_energy_activities.models import (
    AccountingMethod,
    Activity3aResult,
    Activity3bResult,
    Activity3cResult,
    ActivityType,
    AllocationMethod,
    CalculationMethod,
    CalculationResult,
    DQIAssessment,
    DQIScore,
    DQI_QUALITY_TIERS,
    DQI_SCORE_VALUES,
    DECIMAL_PLACES,
    EF_HIERARCHY_PRIORITY,
    ElectricityConsumptionRecord,
    EmissionGas,
    EnergyType,
    FuelCategory,
    FuelConsumptionRecord,
    FuelType,
    GWPSource,
    GWP_VALUES,
    GasBreakdown,
    ONE,
    ONE_HUNDRED,
    ONE_THOUSAND,
    SupplierDataSource,
    SupplierFuelData,
    UNCERTAINTY_RANGES,
    UncertaintyMethod,
    UncertaintyResult,
    WTTFactorSource,
    WTT_FUEL_EMISSION_FACTORS,
    UPSTREAM_ELECTRICITY_FACTORS,
    ZERO,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decimal precision constant (8 decimal places)
# ---------------------------------------------------------------------------
_PRECISION = Decimal("0.00000001")

# ---------------------------------------------------------------------------
# Five-element Decimal constant for DQI arithmetic mean
# ---------------------------------------------------------------------------
_FIVE = Decimal("5")

# ---------------------------------------------------------------------------
# Two-element Decimal constant
# ---------------------------------------------------------------------------
_TWO = Decimal("2")

# ---------------------------------------------------------------------------
# MiQ Methane Intensity Grade Thresholds
# Grade A: < 0.05%, B: 0.05-0.2%, C: 0.2-0.6%, D: 0.6-1.5%,
# E: 1.5-3.0%, F: > 3.0%
# Source: MiQ Standard v2.0 (2024)
# ---------------------------------------------------------------------------
MIQ_GRADE_THRESHOLDS: Dict[str, Tuple[Decimal, Decimal]] = {
    "A": (Decimal("0"), Decimal("0.05")),
    "B": (Decimal("0.05"), Decimal("0.2")),
    "C": (Decimal("0.2"), Decimal("0.6")),
    "D": (Decimal("0.6"), Decimal("1.5")),
    "E": (Decimal("1.5"), Decimal("3.0")),
    "F": (Decimal("3.0"), Decimal("100.0")),
}

# ---------------------------------------------------------------------------
# MiQ upstream adjustment multipliers by grade
# These represent the ratio of the grade's midpoint methane intensity
# to the industry-average methane intensity (~2.3% per IEA 2023).
# Grade A supply chains have very low methane leakage, so the
# upstream factor is substantially reduced.
# ---------------------------------------------------------------------------
MIQ_UPSTREAM_ADJUSTMENTS: Dict[str, Decimal] = {
    "A": Decimal("0.15"),   # ~0.025% vs 2.3% average
    "B": Decimal("0.35"),   # ~0.125% vs 2.3% average
    "C": Decimal("0.55"),   # ~0.4% vs 2.3% average
    "D": Decimal("0.78"),   # ~1.05% vs 2.3% average
    "E": Decimal("0.95"),   # ~2.25% vs 2.3% average
    "F": Decimal("1.30"),   # >3.0% -- worse than average
}

# ---------------------------------------------------------------------------
# OGMP 2.0 Reporting Level definitions
# Level 1: Default emission factors (generic)
# Level 2: Source-level emission factors (engineering estimates)
# Level 3: Source-level direct measurements (periodic)
# Level 4: Site-level continuous measurement
# Level 5: Reconciled, externally verified measurements
# ---------------------------------------------------------------------------
OGMP_LEVEL_DQI_SCORES: Dict[int, Decimal] = {
    1: Decimal("4.5"),   # Very Low quality
    2: Decimal("3.5"),   # Low quality
    3: Decimal("2.5"),   # Medium quality
    4: Decimal("1.5"),   # High quality
    5: Decimal("1.0"),   # Very High quality
}

# ---------------------------------------------------------------------------
# Verification level scoring (0.0 to 1.0)
# Higher score = more trustworthy
# ---------------------------------------------------------------------------
VERIFICATION_SCORES: Dict[str, Decimal] = {
    "third_party_verified": Decimal("1.0"),
    "certified": Decimal("0.85"),
    "second_party_verified": Decimal("0.70"),
    "self_declared": Decimal("0.50"),
    "estimated": Decimal("0.30"),
    "unverified": Decimal("0.15"),
}

# ---------------------------------------------------------------------------
# PPA / Green Tariff upstream emission factors by technology
# These represent lifecycle upstream emissions from construction,
# manufacturing, and maintenance of renewable generation assets.
# Units: kgCO2e per kWh
# Source: IPCC WG3 AR5 Annex III, ecoinvent 3.11
# ---------------------------------------------------------------------------
PPA_UPSTREAM_FACTORS: Dict[str, Decimal] = {
    "solar_pv": Decimal("0.00350"),
    "onshore_wind": Decimal("0.00100"),
    "offshore_wind": Decimal("0.00120"),
    "hydropower": Decimal("0.00200"),
    "geothermal": Decimal("0.00280"),
    "biomass": Decimal("0.00850"),
    "nuclear": Decimal("0.00160"),
    "concentrated_solar": Decimal("0.00400"),
    "tidal": Decimal("0.00250"),
    "wave": Decimal("0.00300"),
}

# ---------------------------------------------------------------------------
# Default PPA upstream factor for unspecified technology
# ---------------------------------------------------------------------------
DEFAULT_PPA_UPSTREAM_EF: Decimal = Decimal("0.00250")

# ---------------------------------------------------------------------------
# Uncertainty reduction factors for verified supplier data
# Verified supplier data narrows uncertainty ranges compared to defaults.
# ---------------------------------------------------------------------------
VERIFICATION_UNCERTAINTY_REDUCTION: Dict[str, Decimal] = {
    "third_party_verified": Decimal("0.30"),   # 70% reduction
    "certified": Decimal("0.45"),              # 55% reduction
    "second_party_verified": Decimal("0.60"),  # 40% reduction
    "self_declared": Decimal("0.75"),          # 25% reduction
    "estimated": Decimal("0.90"),              # 10% reduction
    "unverified": Decimal("1.00"),             # no reduction
}


class SupplierSpecificCalculatorEngine:
    """Engine 5: Supplier-specific upstream emission calculations.

    Uses direct supplier data (EPDs, LCAs, MiQ certificates, OGMP 2.0
    data, PPAs) to calculate upstream emissions with higher accuracy
    than average-data methods. Supports fuel upstream (3a), electricity
    upstream (3b), and T&D losses (3c) sub-activities.

    Thread-safe. All arithmetic uses Python Decimal with 8 decimal
    places (ROUND_HALF_UP). SHA-256 provenance hashing on every result.

    Attributes:
        _config: Engine configuration dictionary.
        _lock: Thread lock for shared mutable state.
        _stats: Runtime statistics counters.

    Example:
        >>> engine = SupplierSpecificCalculatorEngine()
        >>> grade = engine.assess_miq_grade(Decimal("0.03"))
        >>> assert grade == "A"
        >>> multiplier = engine.get_miq_upstream_adjustment("A")
        >>> assert multiplier == Decimal("0.15")
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SupplierSpecificCalculatorEngine.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - decimal_places (int): Decimal precision (default 8).
                - default_gwp_source (str): Default GWP (default "AR6").
                - default_verification (str): Default verification level.
                - enable_provenance (bool): Enable provenance hashing.
                - enable_metrics (bool): Enable metrics recording.
        """
        self._config: Dict[str, Any] = config or {}
        self._lock = threading.Lock()
        self._decimal_places: int = self._config.get(
            "decimal_places", DECIMAL_PLACES
        )
        self._precision = Decimal(10) ** (-self._decimal_places)
        self._default_gwp_source: str = self._config.get(
            "default_gwp_source", "AR6"
        )
        self._default_verification: str = self._config.get(
            "default_verification", "unverified"
        )
        self._enable_provenance: bool = self._config.get(
            "enable_provenance", True
        )
        self._enable_metrics: bool = self._config.get(
            "enable_metrics", True
        )
        self._stats: Dict[str, int] = self._init_stats()
        logger.info(
            "SupplierSpecificCalculatorEngine initialized "
            "(decimal_places=%d, gwp=%s)",
            self._decimal_places,
            self._default_gwp_source,
        )

    def _init_stats(self) -> Dict[str, int]:
        """Initialize runtime statistics counters.

        Returns:
            Dictionary with zeroed statistic counters.
        """
        return {
            "fuel_calculations": 0,
            "electricity_calculations": 0,
            "batch_calculations": 0,
            "epd_validations": 0,
            "miq_assessments": 0,
            "ogmp_assessments": 0,
            "ppa_calculations": 0,
            "allocations": 0,
            "blends": 0,
            "coverage_assessments": 0,
            "dqi_assessments": 0,
            "uncertainty_quantifications": 0,
            "comparisons": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, config: Dict[str, Any]) -> None:
        """Update engine configuration at runtime.

        Thread-safe. Merges provided keys into the existing config.

        Args:
            config: Dictionary of configuration keys to update.
                Supported keys: decimal_places, default_gwp_source,
                default_verification, enable_provenance, enable_metrics.
        """
        with self._lock:
            self._config.update(config)
            if "decimal_places" in config:
                self._decimal_places = int(config["decimal_places"])
                self._precision = Decimal(10) ** (-self._decimal_places)
            if "default_gwp_source" in config:
                self._default_gwp_source = str(config["default_gwp_source"])
            if "default_verification" in config:
                self._default_verification = str(config["default_verification"])
            if "enable_provenance" in config:
                self._enable_provenance = bool(config["enable_provenance"])
            if "enable_metrics" in config:
                self._enable_metrics = bool(config["enable_metrics"])
        logger.info(
            "SupplierSpecificCalculatorEngine reconfigured: %s",
            list(config.keys()),
        )

    def reset(self) -> None:
        """Reset engine state and statistics.

        Thread-safe. Clears all runtime statistics counters.
        Does not reset configuration.
        """
        with self._lock:
            self._stats = self._init_stats()
        logger.info("SupplierSpecificCalculatorEngine state reset")

    def get_statistics(self) -> Dict[str, int]:
        """Return a snapshot of runtime statistics.

        Thread-safe. Returns a copy of internal counters.

        Returns:
            Dictionary of statistic name to count value.
        """
        with self._lock:
            return dict(self._stats)

    # ------------------------------------------------------------------
    # Decimal helpers
    # ------------------------------------------------------------------

    def _q(self, value: Decimal) -> Decimal:
        """Quantize a Decimal to the configured precision.

        Args:
            value: Decimal value to quantize.

        Returns:
            Quantized Decimal value.
        """
        return value.quantize(self._precision, rounding=ROUND_HALF_UP)

    def _safe_decimal(self, value: Any) -> Decimal:
        """Convert a value to Decimal safely.

        Args:
            value: Value to convert (str, int, float, Decimal).

        Returns:
            Decimal representation of the value.

        Raises:
            ValueError: If value cannot be converted.
        """
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(
                f"Cannot convert {value!r} to Decimal: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Provenance hashing
    # ------------------------------------------------------------------

    def _compute_provenance_hash(self, *parts: Any) -> str:
        """Compute SHA-256 provenance hash over arbitrary inputs.

        Serializes each part to a canonical string representation
        and hashes the concatenated result.

        Args:
            *parts: Values to include in the hash computation.

        Returns:
            64-character hex SHA-256 digest string.
        """
        if not self._enable_provenance:
            return ""
        hasher = hashlib.sha256()
        for part in parts:
            serialized = self._serialize_for_hash(part)
            hasher.update(serialized.encode("utf-8"))
        return hasher.hexdigest()

    def _serialize_for_hash(self, obj: Any) -> str:
        """Serialize an object to a canonical string for hashing.

        Handles Decimal, datetime, Enum, Pydantic models, dicts,
        lists, and primitive types.

        Args:
            obj: Object to serialize.

        Returns:
            Canonical string representation.
        """
        if obj is None:
            return "null"
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (int, float, bool)):
            return str(obj)
        if isinstance(obj, str):
            return obj
        if hasattr(obj, "model_dump"):
            return json.dumps(
                obj.model_dump(mode="json"), sort_keys=True, default=str
            )
        if isinstance(obj, dict):
            return json.dumps(obj, sort_keys=True, default=str)
        if isinstance(obj, (list, tuple)):
            return json.dumps(
                [self._serialize_for_hash(x) for x in obj],
                sort_keys=True,
                default=str,
            )
        return str(obj)

    # ------------------------------------------------------------------
    # Statistics incrementer
    # ------------------------------------------------------------------

    def _increment_stat(self, key: str, amount: int = 1) -> None:
        """Thread-safe increment of a statistics counter.

        Args:
            key: Name of the statistic to increment.
            amount: Amount to increment by (default 1).
        """
        with self._lock:
            self._stats[key] = self._stats.get(key, 0) + amount

    # ------------------------------------------------------------------
    # Engine 5 - Method 1: calculate_fuel_upstream
    # ------------------------------------------------------------------

    def calculate_fuel_upstream(
        self,
        fuel_record: FuelConsumptionRecord,
        supplier_data: SupplierFuelData,
        gwp_source: str = "AR6",
    ) -> Activity3aResult:
        """Calculate upstream emissions for a fuel record using supplier data.

        Applies the supplier-specific WTT emission factor (from EPD, LCA,
        MiQ certificate, etc.) to the fuel consumption record to compute
        upstream emissions. If the supplier has a MiQ grade, the upstream
        factor is adjusted accordingly.

        Formula:
            effective_ef = supplier.upstream_ef * allocation_factor
            if miq_grade:
                effective_ef = effective_ef * miq_adjustment
            emissions_total = fuel_consumed_kwh * effective_ef

        Args:
            fuel_record: Fuel consumption record with quantity in kWh.
            supplier_data: Supplier-specific emission data.
            gwp_source: IPCC Assessment Report source (default "AR6").

        Returns:
            Activity3aResult with supplier-specific upstream emissions.

        Raises:
            ValueError: If fuel_record.quantity_kwh is None or negative,
                or if supplier fuel type does not match record fuel type.
        """
        start_time = time.monotonic()
        try:
            # Validate fuel type match
            if fuel_record.fuel_type != supplier_data.fuel_type:
                raise ValueError(
                    f"Fuel type mismatch: record={fuel_record.fuel_type.value} "
                    f"vs supplier={supplier_data.fuel_type.value}"
                )

            # Resolve fuel quantity in kWh
            fuel_kwh = self._resolve_fuel_kwh(fuel_record)

            # Compute effective emission factor
            effective_ef = self._compute_effective_fuel_ef(supplier_data)

            # Calculate total emissions
            emissions_total = self._q(fuel_kwh * effective_ef)

            # Estimate per-gas breakdown using WTT proportions
            gas_breakdown = self._estimate_gas_breakdown(
                emissions_total, fuel_record.fuel_type, gwp_source
            )

            # DQI score based on supplier data quality
            dqi_score = self._compute_supplier_dqi_score(supplier_data)

            # Uncertainty percentage (narrower for verified data)
            uncertainty_pct = self._compute_supplier_uncertainty(
                supplier_data
            )

            # Determine biogenic flag from fuel category
            is_biogenic = self._is_biogenic_fuel(fuel_record.fuel_type)

            # Determine EF source label
            ef_source = self._get_supplier_ef_source_label(supplier_data)

            # Provenance hash
            provenance_hash = self._compute_provenance_hash(
                fuel_record, supplier_data, gwp_source,
                emissions_total, effective_ef,
            )

            result = Activity3aResult(
                fuel_record_id=fuel_record.record_id,
                fuel_type=fuel_record.fuel_type,
                fuel_category=fuel_record.fuel_category,
                fuel_consumed_kwh=fuel_kwh,
                wtt_ef_total=effective_ef,
                wtt_ef_source=ef_source,
                emissions_co2=gas_breakdown["co2"],
                emissions_ch4=gas_breakdown["ch4"],
                emissions_n2o=gas_breakdown["n2o"],
                emissions_total=emissions_total,
                is_biogenic=is_biogenic,
                dqi_score=dqi_score,
                uncertainty_pct=uncertainty_pct,
                provenance_hash=provenance_hash,
            )

            self._increment_stat("fuel_calculations")
            elapsed = (time.monotonic() - start_time) * 1000
            logger.info(
                "Supplier-specific fuel upstream calculated: "
                "supplier=%s fuel=%s emissions=%.4f kgCO2e "
                "(%.1f ms)",
                supplier_data.supplier_id,
                fuel_record.fuel_type.value,
                float(emissions_total),
                elapsed,
            )
            return result

        except Exception:
            self._increment_stat("errors")
            logger.error(
                "Supplier-specific fuel upstream calculation failed: "
                "record=%s supplier=%s",
                fuel_record.record_id,
                supplier_data.supplier_id,
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Engine 5 - Method 2: calculate_electricity_upstream
    # ------------------------------------------------------------------

    def calculate_electricity_upstream(
        self,
        elec_record: ElectricityConsumptionRecord,
        supplier_data: SupplierFuelData,
        gwp_source: str = "AR6",
    ) -> Activity3bResult:
        """Calculate upstream emissions for electricity using supplier data.

        Uses supplier-specific upstream emission factors for electricity
        purchases. For PPA/green tariff suppliers, applies lifecycle
        upstream factors (construction/maintenance only). For conventional
        suppliers with EPDs, applies the supplier-provided upstream EF.

        Formula:
            if PPA/green_tariff:
                upstream_ef = PPA_UPSTREAM_FACTORS[technology]
            else:
                upstream_ef = supplier.upstream_ef * allocation_factor
            emissions_total = electricity_kwh * upstream_ef

        Args:
            elec_record: Electricity consumption record with quantity_kwh.
            supplier_data: Supplier-specific emission data.
            gwp_source: IPCC Assessment Report source (default "AR6").

        Returns:
            Activity3bResult with supplier-specific upstream emissions.

        Raises:
            ValueError: If quantity_kwh is not positive.
        """
        start_time = time.monotonic()
        try:
            energy_kwh = self._safe_decimal(elec_record.quantity_kwh)
            if energy_kwh <= ZERO:
                raise ValueError(
                    f"Electricity quantity must be positive, got {energy_kwh}"
                )

            # Determine if this is a PPA/green tariff supplier
            is_ppa = supplier_data.data_source in (
                SupplierDataSource.PPA,
                SupplierDataSource.GREEN_TARIFF,
            )

            if is_ppa:
                # Use PPA lifecycle upstream factor
                technology = supplier_data.metadata.get(
                    "technology", "solar_pv"
                )
                upstream_ef = self.calculate_ppa_upstream(
                    supplier_data, technology
                )
            else:
                # Use supplier-provided upstream EF
                upstream_ef = self._q(
                    supplier_data.upstream_ef * supplier_data.allocation_factor
                )

            # Calculate total upstream emissions
            emissions_total = self._q(energy_kwh * upstream_ef)

            # DQI score
            dqi_score = self._compute_supplier_dqi_score(supplier_data)

            # Uncertainty
            uncertainty_pct = self._compute_supplier_uncertainty(
                supplier_data
            )

            # EF source label
            ef_source = self._get_supplier_ef_source_label(supplier_data)

            # Accounting method
            accounting = elec_record.accounting_method

            # Provenance hash
            provenance_hash = self._compute_provenance_hash(
                elec_record, supplier_data, gwp_source,
                emissions_total, upstream_ef,
            )

            result = Activity3bResult(
                electricity_record_id=elec_record.record_id,
                energy_type=elec_record.energy_type,
                energy_consumed_kwh=energy_kwh,
                upstream_ef=upstream_ef,
                upstream_ef_source=ef_source,
                accounting_method=accounting,
                grid_region=elec_record.grid_region,
                emissions_total=emissions_total,
                is_renewable=is_ppa or elec_record.is_renewable,
                dqi_score=dqi_score,
                uncertainty_pct=uncertainty_pct,
                provenance_hash=provenance_hash,
            )

            self._increment_stat("electricity_calculations")
            elapsed = (time.monotonic() - start_time) * 1000
            logger.info(
                "Supplier-specific electricity upstream calculated: "
                "supplier=%s region=%s emissions=%.4f kgCO2e "
                "(%.1f ms)",
                supplier_data.supplier_id,
                elec_record.grid_region,
                float(emissions_total),
                elapsed,
            )
            return result

        except Exception:
            self._increment_stat("errors")
            logger.error(
                "Supplier-specific electricity upstream failed: "
                "record=%s supplier=%s",
                elec_record.record_id,
                supplier_data.supplier_id,
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Engine 5 - Method 3: calculate_batch
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        records: List[Union[FuelConsumptionRecord, ElectricityConsumptionRecord]],
        supplier_data_map: Dict[str, SupplierFuelData],
        gwp_source: str = "AR6",
    ) -> List[Union[Activity3aResult, Activity3bResult]]:
        """Calculate supplier-specific upstream emissions for a batch of records.

        Iterates over a mixed list of fuel and electricity records, matching
        each to the supplier_data_map by supplier_id. Records without a
        matching supplier entry are skipped with a warning.

        Args:
            records: List of FuelConsumptionRecord and/or
                ElectricityConsumptionRecord instances.
            supplier_data_map: Mapping of supplier_id to SupplierFuelData.
            gwp_source: IPCC Assessment Report source (default "AR6").

        Returns:
            List of Activity3aResult and/or Activity3bResult for records
            that had matching supplier data.
        """
        start_time = time.monotonic()
        results: List[Union[Activity3aResult, Activity3bResult]] = []
        skipped = 0

        for record in records:
            supplier_id = getattr(record, "supplier_id", None)
            if supplier_id is None or supplier_id not in supplier_data_map:
                skipped += 1
                logger.debug(
                    "Skipping record %s: no supplier data for supplier_id=%s",
                    record.record_id,
                    supplier_id,
                )
                continue

            supplier = supplier_data_map[supplier_id]

            try:
                if isinstance(record, FuelConsumptionRecord):
                    result = self.calculate_fuel_upstream(
                        record, supplier, gwp_source
                    )
                    results.append(result)
                elif isinstance(record, ElectricityConsumptionRecord):
                    result = self.calculate_electricity_upstream(
                        record, supplier, gwp_source
                    )
                    results.append(result)
                else:
                    logger.warning(
                        "Unknown record type for record %s: %s",
                        record.record_id,
                        type(record).__name__,
                    )
                    skipped += 1
            except Exception as exc:
                logger.warning(
                    "Batch calculation failed for record %s: %s",
                    record.record_id,
                    str(exc),
                )
                skipped += 1

        self._increment_stat("batch_calculations")
        elapsed = (time.monotonic() - start_time) * 1000
        logger.info(
            "Batch supplier-specific calculation complete: "
            "total=%d processed=%d skipped=%d (%.1f ms)",
            len(records),
            len(results),
            skipped,
            elapsed,
        )
        return results

    # ------------------------------------------------------------------
    # Engine 5 - Method 4: validate_supplier_data
    # ------------------------------------------------------------------

    def validate_supplier_data(
        self, supplier_data: SupplierFuelData
    ) -> Tuple[bool, List[str]]:
        """Validate supplier-specific data quality and completeness.

        Checks required fields, value ranges, verification level,
        MiQ grade validity, OGMP 2.0 level validity, and reporting
        year recency.

        Args:
            supplier_data: SupplierFuelData instance to validate.

        Returns:
            Tuple of (is_valid, list_of_issues). is_valid is True if
            no critical issues were found. The issues list contains
            warning and error messages.
        """
        issues: List[str] = []
        is_valid = True

        # Check supplier_id is non-empty
        if not supplier_data.supplier_id or not supplier_data.supplier_id.strip():
            issues.append("ERROR: supplier_id is empty or blank")
            is_valid = False

        # Check supplier_name
        if not supplier_data.supplier_name or not supplier_data.supplier_name.strip():
            issues.append("ERROR: supplier_name is empty or blank")
            is_valid = False

        # Check upstream_ef is non-negative
        if supplier_data.upstream_ef < ZERO:
            issues.append(
                f"ERROR: upstream_ef is negative ({supplier_data.upstream_ef})"
            )
            is_valid = False

        # Check upstream_ef is not unreasonably high (> 1.0 kgCO2e/kWh)
        if supplier_data.upstream_ef > Decimal("1.0"):
            issues.append(
                f"WARNING: upstream_ef is unusually high "
                f"({supplier_data.upstream_ef} kgCO2e/kWh); "
                f"WTT factors are typically < 0.1"
            )

        # Validate verification level
        valid_levels = set(VERIFICATION_SCORES.keys())
        if supplier_data.verification_level not in valid_levels:
            issues.append(
                f"WARNING: verification_level "
                f"'{supplier_data.verification_level}' is not "
                f"recognized; valid levels: {sorted(valid_levels)}"
            )

        # Validate MiQ grade if provided
        if supplier_data.miq_grade is not None:
            valid_grades = set(MIQ_GRADE_THRESHOLDS.keys())
            grade_upper = supplier_data.miq_grade.upper()
            if grade_upper not in valid_grades:
                issues.append(
                    f"ERROR: miq_grade '{supplier_data.miq_grade}' "
                    f"is not valid; must be one of {sorted(valid_grades)}"
                )
                is_valid = False

        # Validate OGMP 2.0 level if provided
        if supplier_data.ogmp2_level is not None:
            if supplier_data.ogmp2_level < 1 or supplier_data.ogmp2_level > 5:
                issues.append(
                    f"ERROR: ogmp2_level must be 1-5, "
                    f"got {supplier_data.ogmp2_level}"
                )
                is_valid = False

        # Check reporting year recency
        current_year = datetime.now(timezone.utc).year
        if supplier_data.reporting_year is not None:
            age = current_year - supplier_data.reporting_year
            if age > 3:
                issues.append(
                    f"WARNING: reporting_year {supplier_data.reporting_year} "
                    f"is {age} years old; supplier data older than 3 years "
                    f"may be outdated"
                )
            if supplier_data.reporting_year > current_year:
                issues.append(
                    f"WARNING: reporting_year {supplier_data.reporting_year} "
                    f"is in the future"
                )

        # Check allocation factor bounds
        if supplier_data.allocation_factor <= ZERO:
            issues.append(
                f"WARNING: allocation_factor is zero or negative "
                f"({supplier_data.allocation_factor}); "
                f"this will result in zero emissions"
            )

        return is_valid, issues

    # ------------------------------------------------------------------
    # Engine 5 - Method 5: validate_epd
    # ------------------------------------------------------------------

    def validate_epd(
        self, epd_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate Environmental Product Declaration (EPD) data.

        Checks the presence and validity of required EPD fields per
        ISO 14025 / EN 15804 standards. Validates the EPD number
        format, product category, declared unit, and data quality.

        Required EPD fields:
            - epd_number: Registration number
            - programme_operator: EPD programme name
            - product_category: Product category rules (PCR)
            - declared_unit: Functional unit for the EPD
            - upstream_ef: Upstream emission factor (kgCO2e/unit)
            - valid_from: EPD issuance date (ISO 8601)
            - valid_until: EPD expiry date (ISO 8601)
            - verification_body: Third-party verifier

        Args:
            epd_data: Dictionary containing EPD fields.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        self._increment_stat("epd_validations")
        issues: List[str] = []
        is_valid = True

        # Required fields
        required_fields = [
            "epd_number",
            "programme_operator",
            "product_category",
            "declared_unit",
            "upstream_ef",
        ]
        for field_name in required_fields:
            if field_name not in epd_data or epd_data[field_name] is None:
                issues.append(f"ERROR: required EPD field '{field_name}' is missing")
                is_valid = False

        # Validate upstream_ef if present
        if "upstream_ef" in epd_data and epd_data["upstream_ef"] is not None:
            try:
                ef_val = self._safe_decimal(epd_data["upstream_ef"])
                if ef_val < ZERO:
                    issues.append(
                        f"ERROR: upstream_ef must be non-negative, "
                        f"got {ef_val}"
                    )
                    is_valid = False
                if ef_val > Decimal("1.0"):
                    issues.append(
                        f"WARNING: upstream_ef value {ef_val} is "
                        f"unusually high for WTT factors"
                    )
            except ValueError:
                issues.append(
                    f"ERROR: upstream_ef '{epd_data['upstream_ef']}' "
                    f"is not a valid number"
                )
                is_valid = False

        # Validate date range if both dates present
        if "valid_from" in epd_data and "valid_until" in epd_data:
            try:
                valid_from = epd_data["valid_from"]
                valid_until = epd_data["valid_until"]
                if valid_from and valid_until and valid_until < valid_from:
                    issues.append(
                        f"ERROR: valid_until ({valid_until}) is before "
                        f"valid_from ({valid_from})"
                    )
                    is_valid = False
            except (TypeError, ValueError):
                issues.append(
                    "WARNING: could not validate EPD date range"
                )

        # Check for verification body
        if "verification_body" not in epd_data or not epd_data.get(
            "verification_body"
        ):
            issues.append(
                "WARNING: verification_body not provided; "
                "EPDs should be third-party verified"
            )

        # Validate EPD number format (basic check for non-empty string)
        if "epd_number" in epd_data and epd_data["epd_number"]:
            epd_num = str(epd_data["epd_number"]).strip()
            if len(epd_num) < 3:
                issues.append(
                    f"WARNING: epd_number '{epd_num}' appears too short"
                )

        return is_valid, issues

    # ------------------------------------------------------------------
    # Engine 5 - Method 6: assess_miq_grade
    # ------------------------------------------------------------------

    def assess_miq_grade(self, methane_intensity_pct: Decimal) -> str:
        """Assess MiQ methane intensity grade from percentage.

        MiQ (Methane Intelligence Quotient) certifies natural gas
        supply chains based on their methane intensity (methane
        emitted as a percentage of gas produced).

        Grade thresholds:
            A: < 0.05%
            B: 0.05% - 0.2%
            C: 0.2%  - 0.6%
            D: 0.6%  - 1.5%
            E: 1.5%  - 3.0%
            F: > 3.0%

        Args:
            methane_intensity_pct: Methane intensity as a percentage
                (e.g., 0.03 for 0.03%).

        Returns:
            MiQ grade string: "A", "B", "C", "D", "E", or "F".

        Raises:
            ValueError: If methane_intensity_pct is negative.
        """
        self._increment_stat("miq_assessments")
        intensity = self._safe_decimal(methane_intensity_pct)

        if intensity < ZERO:
            raise ValueError(
                f"Methane intensity cannot be negative: {intensity}"
            )

        for grade in ("A", "B", "C", "D", "E", "F"):
            lower, upper = MIQ_GRADE_THRESHOLDS[grade]
            if intensity < upper:
                logger.debug(
                    "MiQ grade assessed: intensity=%.4f%% -> Grade %s",
                    float(intensity),
                    grade,
                )
                return grade

        # Fallback for edge case (should not be reached)
        return "F"

    # ------------------------------------------------------------------
    # Engine 5 - Method 7: get_miq_upstream_adjustment
    # ------------------------------------------------------------------

    def get_miq_upstream_adjustment(self, grade: str) -> Decimal:
        """Get the upstream emission factor adjustment multiplier for a MiQ grade.

        The multiplier adjusts the average WTT emission factor based
        on the certified methane intensity of the supply chain.
        Lower grades (A/B) substantially reduce the upstream factor;
        higher grades (E/F) may increase it above average.

        Args:
            grade: MiQ grade string ("A" through "F").

        Returns:
            Decimal multiplier (e.g., 0.15 for Grade A).

        Raises:
            ValueError: If grade is not a valid MiQ grade (A-F).
        """
        grade_upper = grade.upper().strip()
        if grade_upper not in MIQ_UPSTREAM_ADJUSTMENTS:
            raise ValueError(
                f"Invalid MiQ grade '{grade}'; "
                f"must be one of {sorted(MIQ_UPSTREAM_ADJUSTMENTS.keys())}"
            )
        adjustment = MIQ_UPSTREAM_ADJUSTMENTS[grade_upper]
        logger.debug(
            "MiQ upstream adjustment: Grade %s -> multiplier %s",
            grade_upper,
            adjustment,
        )
        return adjustment

    # ------------------------------------------------------------------
    # Engine 5 - Method 8: assess_ogmp_level
    # ------------------------------------------------------------------

    def assess_ogmp_level(
        self, reporting_data: Dict[str, Any]
    ) -> int:
        """Assess OGMP 2.0 reporting level from supplier reporting data.

        The Oil & Gas Methane Partnership (OGMP) 2.0 Framework defines
        five levels of methane reporting quality, from Level 1 (generic
        emission factors) through Level 5 (reconciled, externally
        verified site-level measurements).

        Assessment criteria:
            Level 5: has 'external_verification' AND 'reconciliation' AND
                     'site_measurement'
            Level 4: has 'site_measurement' or 'continuous_monitoring'
            Level 3: has 'source_measurement' or 'direct_measurement'
            Level 2: has 'engineering_estimates' or 'source_level_factors'
            Level 1: default (generic emission factors)

        Args:
            reporting_data: Dictionary with OGMP reporting characteristics.
                Expected keys: 'external_verification', 'reconciliation',
                'site_measurement', 'continuous_monitoring',
                'source_measurement', 'direct_measurement',
                'engineering_estimates', 'source_level_factors'.

        Returns:
            OGMP 2.0 level (1 through 5).
        """
        self._increment_stat("ogmp_assessments")

        has_external_verification = bool(
            reporting_data.get("external_verification", False)
        )
        has_reconciliation = bool(
            reporting_data.get("reconciliation", False)
        )
        has_site_measurement = bool(
            reporting_data.get("site_measurement", False)
        )
        has_continuous_monitoring = bool(
            reporting_data.get("continuous_monitoring", False)
        )
        has_source_measurement = bool(
            reporting_data.get("source_measurement", False)
        )
        has_direct_measurement = bool(
            reporting_data.get("direct_measurement", False)
        )
        has_engineering_estimates = bool(
            reporting_data.get("engineering_estimates", False)
        )
        has_source_level_factors = bool(
            reporting_data.get("source_level_factors", False)
        )

        # Level 5: All three required
        if (
            has_external_verification
            and has_reconciliation
            and has_site_measurement
        ):
            level = 5
        # Level 4: Site-level or continuous
        elif has_site_measurement or has_continuous_monitoring:
            level = 4
        # Level 3: Source-level measurement
        elif has_source_measurement or has_direct_measurement:
            level = 3
        # Level 2: Engineering estimates
        elif has_engineering_estimates or has_source_level_factors:
            level = 2
        # Level 1: Default
        else:
            level = 1

        logger.debug(
            "OGMP 2.0 level assessed: Level %d (data keys: %s)",
            level,
            list(reporting_data.keys()),
        )
        return level

    # ------------------------------------------------------------------
    # Engine 5 - Method 9: calculate_ppa_upstream
    # ------------------------------------------------------------------

    def calculate_ppa_upstream(
        self,
        ppa_data: SupplierFuelData,
        technology: str = "solar_pv",
    ) -> Decimal:
        """Calculate upstream emission factor for PPA or green tariff electricity.

        Renewable energy under PPAs or green tariffs has near-zero
        operational emissions but non-zero upstream emissions from
        manufacturing, construction, and maintenance of generation
        assets. This method returns the lifecycle upstream factor.

        Args:
            ppa_data: Supplier data for the PPA/green tariff.
            technology: Renewable technology type. Valid values:
                solar_pv, onshore_wind, offshore_wind, hydropower,
                geothermal, biomass, nuclear, concentrated_solar,
                tidal, wave.

        Returns:
            Upstream emission factor in kgCO2e per kWh.
        """
        self._increment_stat("ppa_calculations")

        tech_key = technology.lower().strip()
        upstream_ef = PPA_UPSTREAM_FACTORS.get(tech_key, DEFAULT_PPA_UPSTREAM_EF)

        # Apply allocation factor if multi-product supplier
        effective_ef = self._q(upstream_ef * ppa_data.allocation_factor)

        logger.debug(
            "PPA upstream EF: technology=%s base_ef=%s "
            "allocation=%.4f effective_ef=%s",
            tech_key,
            upstream_ef,
            float(ppa_data.allocation_factor),
            effective_ef,
        )
        return effective_ef

    # ------------------------------------------------------------------
    # Engine 5 - Method 10: allocate_supplier_emissions
    # ------------------------------------------------------------------

    def allocate_supplier_emissions(
        self,
        total_emissions: Decimal,
        allocation_method: AllocationMethod,
        allocation_data: Dict[str, Decimal],
    ) -> Decimal:
        """Allocate a supplier's total upstream emissions to the reporting entity.

        When a supplier reports aggregate upstream emissions across
        multiple products or customers, this method applies an allocation
        factor to determine the portion attributable to the reporting
        company.

        Allocation factor = entity_share / total

        Args:
            total_emissions: Total supplier upstream emissions (kgCO2e).
            allocation_method: Method of allocation (revenue, production,
                energy_content, mass, economic).
            allocation_data: Dictionary with keys:
                - 'entity_share': Reporting entity's share amount.
                - 'total': Total supplier amount.
                Both must be positive.

        Returns:
            Allocated emissions in kgCO2e (Decimal).

        Raises:
            ValueError: If entity_share or total is missing, zero, or
                negative; or if entity_share exceeds total.
        """
        self._increment_stat("allocations")

        entity_share = allocation_data.get("entity_share")
        total = allocation_data.get("total")

        if entity_share is None or total is None:
            raise ValueError(
                "allocation_data must contain 'entity_share' and 'total'"
            )

        entity_share = self._safe_decimal(entity_share)
        total = self._safe_decimal(total)

        if total <= ZERO:
            raise ValueError(
                f"Allocation total must be positive, got {total}"
            )
        if entity_share < ZERO:
            raise ValueError(
                f"entity_share must be non-negative, got {entity_share}"
            )
        if entity_share > total:
            raise ValueError(
                f"entity_share ({entity_share}) exceeds total ({total})"
            )

        total_emissions = self._safe_decimal(total_emissions)
        allocation_factor = self._q(entity_share / total)
        allocated = self._q(total_emissions * allocation_factor)

        logger.debug(
            "Emission allocation: method=%s entity_share=%s total=%s "
            "factor=%.8f allocated=%s kgCO2e",
            allocation_method.value,
            entity_share,
            total,
            float(allocation_factor),
            allocated,
        )
        return allocated

    # ------------------------------------------------------------------
    # Engine 5 - Method 11: blend_with_average
    # ------------------------------------------------------------------

    def blend_with_average(
        self,
        supplier_results: List[Union[Activity3aResult, Activity3bResult]],
        average_results: List[Union[Activity3aResult, Activity3bResult]],
        coverage_pct: Decimal,
    ) -> Dict[str, Any]:
        """Blend supplier-specific and average-data results for partial coverage.

        When a reporting entity has supplier-specific data for a portion
        of its fuel or electricity consumption, this method blends the
        supplier-specific results with average-data results for the
        uncovered portion.

        Formula:
            blended_total = supplier_total * (coverage / 100)
                          + average_total * (1 - coverage / 100)

        Args:
            supplier_results: Results calculated with supplier-specific data.
            average_results: Results calculated with average-data method.
            coverage_pct: Percentage of consumption covered by supplier
                data (0-100).

        Returns:
            Dictionary with blended emissions, method, and breakdown.
        """
        self._increment_stat("blends")

        coverage_frac = self._q(
            self._safe_decimal(coverage_pct) / ONE_HUNDRED
        )
        average_frac = self._q(ONE - coverage_frac)

        # Sum supplier emissions
        supplier_total = ZERO
        for r in supplier_results:
            supplier_total += r.emissions_total
        supplier_total = self._q(supplier_total)

        # Sum average emissions
        average_total = ZERO
        for r in average_results:
            average_total += r.emissions_total
        average_total = self._q(average_total)

        # Blend
        blended_total = self._q(
            supplier_total * coverage_frac
            + average_total * average_frac
        )

        # Blended DQI: weighted average
        supplier_dqi_sum = ZERO
        supplier_count = len(supplier_results)
        for r in supplier_results:
            supplier_dqi_sum += r.dqi_score

        average_dqi_sum = ZERO
        average_count = len(average_results)
        for r in average_results:
            average_dqi_sum += r.dqi_score

        supplier_avg_dqi = (
            self._q(supplier_dqi_sum / Decimal(str(supplier_count)))
            if supplier_count > 0
            else Decimal("3.0")
        )
        average_avg_dqi = (
            self._q(average_dqi_sum / Decimal(str(average_count)))
            if average_count > 0
            else Decimal("3.0")
        )
        blended_dqi = self._q(
            supplier_avg_dqi * coverage_frac
            + average_avg_dqi * average_frac
        )

        # Blended uncertainty
        supplier_unc = ZERO
        for r in supplier_results:
            supplier_unc += r.uncertainty_pct
        supplier_avg_unc = (
            self._q(supplier_unc / Decimal(str(supplier_count)))
            if supplier_count > 0
            else Decimal("10.0")
        )

        average_unc = ZERO
        for r in average_results:
            average_unc += r.uncertainty_pct
        average_avg_unc = (
            self._q(average_unc / Decimal(str(average_count)))
            if average_count > 0
            else Decimal("25.0")
        )
        blended_uncertainty = self._q(
            supplier_avg_unc * coverage_frac
            + average_avg_unc * average_frac
        )

        result = {
            "blended_total_kgco2e": blended_total,
            "supplier_total_kgco2e": supplier_total,
            "average_total_kgco2e": average_total,
            "coverage_pct": coverage_pct,
            "supplier_record_count": supplier_count,
            "average_record_count": average_count,
            "method": CalculationMethod.HYBRID.value,
            "blended_dqi_score": blended_dqi,
            "blended_uncertainty_pct": blended_uncertainty,
            "provenance_hash": self._compute_provenance_hash(
                blended_total, supplier_total, average_total,
                coverage_pct,
            ),
        }

        logger.info(
            "Blended results: supplier=%.4f avg=%.4f "
            "blended=%.4f kgCO2e (coverage=%.1f%%)",
            float(supplier_total),
            float(average_total),
            float(blended_total),
            float(coverage_pct),
        )
        return result

    # ------------------------------------------------------------------
    # Engine 5 - Method 12: assess_coverage
    # ------------------------------------------------------------------

    def assess_coverage(
        self,
        supplier_records: List[Union[FuelConsumptionRecord, ElectricityConsumptionRecord]],
        total_records: List[Union[FuelConsumptionRecord, ElectricityConsumptionRecord]],
    ) -> Decimal:
        """Assess supplier-specific data coverage as a percentage.

        Computes the percentage of total energy consumption (in kWh)
        that is covered by supplier-specific data. This determines
        whether the supplier-specific method, hybrid method, or
        average-data method should be used.

        Args:
            supplier_records: Records with supplier-specific data.
            total_records: All records (supplier + non-supplier).

        Returns:
            Coverage percentage (0-100) as Decimal.
        """
        self._increment_stat("coverage_assessments")

        if not total_records:
            logger.debug("No records provided for coverage assessment")
            return ZERO

        # Sum energy content for supplier records
        supplier_kwh = ZERO
        for rec in supplier_records:
            kwh = self._extract_kwh(rec)
            supplier_kwh += kwh

        # Sum energy content for all records
        total_kwh = ZERO
        for rec in total_records:
            kwh = self._extract_kwh(rec)
            total_kwh += kwh

        if total_kwh <= ZERO:
            return ZERO

        coverage_pct = self._q(
            (supplier_kwh / total_kwh) * ONE_HUNDRED
        )

        # Cap at 100%
        if coverage_pct > ONE_HUNDRED:
            coverage_pct = ONE_HUNDRED

        logger.info(
            "Supplier coverage: %.4f kWh / %.4f kWh = %.2f%%",
            float(supplier_kwh),
            float(total_kwh),
            float(coverage_pct),
        )
        return coverage_pct

    # ------------------------------------------------------------------
    # Engine 5 - Method 13: assess_verification_level
    # ------------------------------------------------------------------

    def assess_verification_level(
        self, supplier_data: SupplierFuelData
    ) -> Decimal:
        """Assess the verification level score for supplier data.

        Returns a score from 0.0 to 1.0 based on the declared
        verification level of the supplier data. Higher scores
        indicate more trustworthy data.

        Scoring:
            third_party_verified: 1.00
            certified:            0.85
            second_party_verified: 0.70
            self_declared:        0.50
            estimated:            0.30
            unverified:           0.15

        Args:
            supplier_data: SupplierFuelData instance.

        Returns:
            Verification score (Decimal 0.0 to 1.0).
        """
        level = supplier_data.verification_level.lower().strip()
        score = VERIFICATION_SCORES.get(
            level,
            VERIFICATION_SCORES.get(
                self._default_verification, Decimal("0.15")
            ),
        )
        logger.debug(
            "Verification level assessed: '%s' -> score %s",
            level,
            score,
        )
        return score

    # ------------------------------------------------------------------
    # Engine 5 - Method 14: compare_with_average
    # ------------------------------------------------------------------

    def compare_with_average(
        self,
        supplier_result: Union[Activity3aResult, Activity3bResult],
        average_result: Union[Activity3aResult, Activity3bResult],
    ) -> Dict[str, Any]:
        """Compare supplier-specific result with average-data result.

        Calculates the absolute and relative difference between the
        two methods to help assess the impact of using supplier data.

        Args:
            supplier_result: Result from supplier-specific calculation.
            average_result: Result from average-data calculation.

        Returns:
            Dictionary with comparison metrics:
                - supplier_emissions_kgco2e
                - average_emissions_kgco2e
                - absolute_difference_kgco2e
                - relative_difference_pct (positive = supplier higher)
                - supplier_dqi_score
                - average_dqi_score
                - dqi_improvement (lower = better)
                - recommendation (use_supplier / use_average / equivalent)
        """
        self._increment_stat("comparisons")

        sup_em = supplier_result.emissions_total
        avg_em = average_result.emissions_total

        abs_diff = self._q(sup_em - avg_em)

        # Relative difference as percentage of average
        if avg_em > ZERO:
            rel_diff_pct = self._q(
                (abs_diff / avg_em) * ONE_HUNDRED
            )
        else:
            rel_diff_pct = ZERO if sup_em == ZERO else ONE_HUNDRED

        # DQI comparison
        sup_dqi = supplier_result.dqi_score
        avg_dqi = average_result.dqi_score
        dqi_improvement = self._q(avg_dqi - sup_dqi)

        # Recommendation logic
        if dqi_improvement > Decimal("0.5"):
            recommendation = "use_supplier"
        elif dqi_improvement < Decimal("-0.5"):
            recommendation = "use_average"
        else:
            # Similar DQI: prefer the one with lower emissions
            if abs_diff < ZERO:
                recommendation = "use_supplier"
            elif abs_diff > ZERO:
                recommendation = "use_average"
            else:
                recommendation = "equivalent"

        result = {
            "supplier_emissions_kgco2e": sup_em,
            "average_emissions_kgco2e": avg_em,
            "absolute_difference_kgco2e": abs_diff,
            "relative_difference_pct": rel_diff_pct,
            "supplier_dqi_score": sup_dqi,
            "average_dqi_score": avg_dqi,
            "dqi_improvement": dqi_improvement,
            "recommendation": recommendation,
            "provenance_hash": self._compute_provenance_hash(
                sup_em, avg_em, abs_diff, rel_diff_pct,
            ),
        }

        logger.info(
            "Comparison: supplier=%.4f avg=%.4f diff=%.4f kgCO2e "
            "(%.2f%%) recommendation=%s",
            float(sup_em),
            float(avg_em),
            float(abs_diff),
            float(rel_diff_pct),
            recommendation,
        )
        return result

    # ------------------------------------------------------------------
    # Engine 5 - Method 15: assess_dqi
    # ------------------------------------------------------------------

    def assess_dqi(
        self, supplier_data: SupplierFuelData
    ) -> DQIAssessment:
        """Assess data quality indicators for supplier data.

        Evaluates the five GHG Protocol DQI dimensions (temporal,
        geographical, technological, completeness, reliability) based
        on the supplier data characteristics and returns a composite
        DQI assessment.

        Scoring logic:
            - Temporal: Based on reporting_year recency
            - Geographical: Based on data_source specificity
            - Technological: Based on data_source type (EPD > LCA > CDP)
            - Completeness: Based on presence of key fields
            - Reliability: Based on verification_level

        Args:
            supplier_data: SupplierFuelData instance.

        Returns:
            DQIAssessment with per-dimension and composite scores.
        """
        self._increment_stat("dqi_assessments")
        findings: List[str] = []

        # 1. Temporal score
        temporal = self._assess_temporal_dqi(supplier_data, findings)

        # 2. Geographical score
        geographical = self._assess_geographical_dqi(supplier_data, findings)

        # 3. Technological score
        technological = self._assess_technological_dqi(supplier_data, findings)

        # 4. Completeness score
        completeness = self._assess_completeness_dqi(supplier_data, findings)

        # 5. Reliability score
        reliability = self._assess_reliability_dqi(supplier_data, findings)

        # Composite: arithmetic mean of 5 dimensions
        composite = self._q(
            (temporal + geographical + technological + completeness + reliability)
            / _FIVE
        )

        # Determine quality tier
        tier = self._determine_quality_tier(composite)

        # Determine activity type based on data source
        if supplier_data.data_source in (
            SupplierDataSource.PPA,
            SupplierDataSource.GREEN_TARIFF,
        ):
            activity_type = ActivityType.ACTIVITY_3B
        else:
            activity_type = ActivityType.ACTIVITY_3A

        assessment = DQIAssessment(
            record_id=supplier_data.supplier_id,
            activity_type=activity_type,
            temporal=temporal,
            geographical=geographical,
            technological=technological,
            completeness=completeness,
            reliability=reliability,
            composite=composite,
            tier=tier,
            findings=findings,
        )

        logger.info(
            "DQI assessed for supplier %s: composite=%.2f tier=%s",
            supplier_data.supplier_id,
            float(composite),
            tier,
        )
        return assessment

    # ------------------------------------------------------------------
    # Engine 5 - Method 16: quantify_uncertainty
    # ------------------------------------------------------------------

    def quantify_uncertainty(
        self,
        supplier_data: SupplierFuelData,
        method: str = "analytical",
    ) -> UncertaintyResult:
        """Quantify uncertainty for supplier-specific calculations.

        Supplier-specific data typically has narrower uncertainty ranges
        than average-data methods. The uncertainty is derived from:
        1. Base range for supplier-specific method (+/- 5-10%)
        2. Verification level reduction factor
        3. MiQ grade (if applicable) further narrows uncertainty

        Args:
            supplier_data: SupplierFuelData instance.
            method: Uncertainty quantification method. Valid values:
                "analytical", "monte_carlo", "ipcc_default".

        Returns:
            UncertaintyResult with confidence interval bounds.
        """
        self._increment_stat("uncertainty_quantifications")

        # Base uncertainty range for supplier-specific method
        base_min, base_max = UNCERTAINTY_RANGES[CalculationMethod.SUPPLIER_SPECIFIC]

        # Verification reduction factor
        reduction_factor = VERIFICATION_UNCERTAINTY_REDUCTION.get(
            supplier_data.verification_level.lower().strip(),
            Decimal("1.0"),
        )

        # MiQ grade bonus (tighter for higher grades)
        miq_bonus = ZERO
        if supplier_data.miq_grade is not None:
            grade = supplier_data.miq_grade.upper().strip()
            if grade in MIQ_UPSTREAM_ADJUSTMENTS:
                adjustment = MIQ_UPSTREAM_ADJUSTMENTS[grade]
                # Higher grades (A/B) further tighten uncertainty
                miq_bonus = self._q(
                    (ONE - adjustment) * Decimal("2.0")
                )

        # Calculate effective uncertainty percentage
        effective_min = self._q(base_min * reduction_factor)
        effective_max = self._q(base_max * reduction_factor)

        # Apply MiQ bonus reduction
        if miq_bonus > ZERO:
            effective_min = self._q(max(ZERO, effective_min - miq_bonus))
            effective_max = self._q(max(effective_min, effective_max - miq_bonus))

        # Mean and std_dev from the range
        mean_pct = self._q((effective_min + effective_max) / _TWO)
        std_dev_pct = self._q((effective_max - effective_min) / Decimal("4"))

        # Use supplier upstream_ef as the central value
        central_value = supplier_data.upstream_ef
        mean_value = central_value
        std_dev_value = self._q(central_value * std_dev_pct / ONE_HUNDRED)

        # Coefficient of variation
        cv = self._q(std_dev_value / mean_value) if mean_value > ZERO else ZERO

        # 95% confidence interval (1.96 sigma for normal distribution)
        z_score = Decimal("1.96")
        ci_half_width = self._q(std_dev_value * z_score)
        ci_lower = self._q(max(ZERO, mean_value - ci_half_width))
        ci_upper = self._q(mean_value + ci_half_width)

        # Map method string to enum
        method_enum = UncertaintyMethod.ANALYTICAL
        method_lower = method.lower().strip()
        if method_lower == "monte_carlo":
            method_enum = UncertaintyMethod.MONTE_CARLO
        elif method_lower == "ipcc_default":
            method_enum = UncertaintyMethod.IPCC_DEFAULT

        result = UncertaintyResult(
            mean=mean_value,
            std_dev=std_dev_value,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=Decimal("95.0"),
            method=method_enum,
        )

        logger.info(
            "Uncertainty quantified for supplier %s: "
            "mean=%s std_dev=%s CI=[%s, %s] (method=%s)",
            supplier_data.supplier_id,
            mean_value,
            std_dev_value,
            ci_lower,
            ci_upper,
            method_enum.value,
        )
        return result

    # ------------------------------------------------------------------
    # Engine 5 - Method 17: aggregate_by_supplier
    # ------------------------------------------------------------------

    def aggregate_by_supplier(
        self,
        results: List[Union[Activity3aResult, Activity3bResult]],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by supplier across multiple results.

        Groups results by their provenance (wtt_ef_source for 3a,
        upstream_ef_source for 3b) and sums emissions per supplier.

        Args:
            results: List of Activity3aResult and/or Activity3bResult.

        Returns:
            Dictionary mapping supplier/source label to total
            emissions in kgCO2e.
        """
        aggregation: Dict[str, Decimal] = {}

        for result in results:
            # Determine the source label
            if isinstance(result, Activity3aResult):
                label = result.wtt_ef_source
            elif isinstance(result, Activity3bResult):
                label = result.upstream_ef_source
            else:
                label = "unknown"

            current = aggregation.get(label, ZERO)
            aggregation[label] = self._q(current + result.emissions_total)

        logger.info(
            "Aggregated by supplier: %d suppliers, total sources=%d",
            len(aggregation),
            len(results),
        )
        return aggregation

    # ==================================================================
    # Internal helper methods
    # ==================================================================

    # ------------------------------------------------------------------
    # Fuel energy resolution
    # ------------------------------------------------------------------

    def _resolve_fuel_kwh(
        self, fuel_record: FuelConsumptionRecord
    ) -> Decimal:
        """Resolve fuel energy content in kWh from a fuel record.

        Uses quantity_kwh if provided, otherwise raises ValueError.

        Args:
            fuel_record: FuelConsumptionRecord with quantity and unit.

        Returns:
            Fuel energy content in kWh (Decimal).

        Raises:
            ValueError: If quantity_kwh is None or not positive.
        """
        if fuel_record.quantity_kwh is not None:
            kwh = self._safe_decimal(fuel_record.quantity_kwh)
            if kwh <= ZERO:
                raise ValueError(
                    f"quantity_kwh must be positive, got {kwh}"
                )
            return kwh

        # If unit is kWh, use quantity directly
        if fuel_record.unit.lower() in ("kwh", "kilowatt_hour", "kilowatt-hour"):
            return self._safe_decimal(fuel_record.quantity)

        raise ValueError(
            f"Cannot resolve fuel kWh: quantity_kwh is not provided "
            f"and unit '{fuel_record.unit}' cannot be auto-converted. "
            f"Provide quantity_kwh for supplier-specific calculations."
        )

    # ------------------------------------------------------------------
    # Effective emission factor computation
    # ------------------------------------------------------------------

    def _compute_effective_fuel_ef(
        self, supplier_data: SupplierFuelData
    ) -> Decimal:
        """Compute effective WTT emission factor from supplier data.

        Applies allocation factor and MiQ grade adjustment (if any)
        to the supplier's upstream EF.

        Args:
            supplier_data: SupplierFuelData with upstream_ef and metadata.

        Returns:
            Effective emission factor in kgCO2e per kWh (Decimal).
        """
        base_ef = supplier_data.upstream_ef
        allocation_factor = supplier_data.allocation_factor

        # Apply allocation
        effective_ef = self._q(base_ef * allocation_factor)

        # Apply MiQ grade adjustment if present
        if supplier_data.miq_grade is not None:
            grade = supplier_data.miq_grade.upper().strip()
            if grade in MIQ_UPSTREAM_ADJUSTMENTS:
                adjustment = MIQ_UPSTREAM_ADJUSTMENTS[grade]
                effective_ef = self._q(effective_ef * adjustment)
                logger.debug(
                    "MiQ adjustment applied: grade=%s "
                    "multiplier=%s effective_ef=%s",
                    grade,
                    adjustment,
                    effective_ef,
                )

        return effective_ef

    # ------------------------------------------------------------------
    # Gas breakdown estimation
    # ------------------------------------------------------------------

    def _estimate_gas_breakdown(
        self,
        total_emissions: Decimal,
        fuel_type: FuelType,
        gwp_source: str,
    ) -> Dict[str, Decimal]:
        """Estimate per-gas breakdown from total WTT emissions.

        Uses the WTT_FUEL_EMISSION_FACTORS proportions to split the
        total emissions into CO2, CH4, and N2O components.

        Args:
            total_emissions: Total upstream emissions in kgCO2e.
            fuel_type: Fuel type for proportion lookup.
            gwp_source: IPCC AR source (not used for proportioning but
                recorded for traceability).

        Returns:
            Dictionary with keys 'co2', 'ch4', 'n2o' (all Decimal).
        """
        wtt_factors = WTT_FUEL_EMISSION_FACTORS.get(fuel_type)
        if wtt_factors is None or wtt_factors.get("total", ZERO) <= ZERO:
            # Cannot determine proportions; assign all to CO2
            return {
                "co2": total_emissions,
                "ch4": ZERO,
                "n2o": ZERO,
            }

        wtt_total = wtt_factors["total"]
        co2_frac = self._q(wtt_factors["co2"] / wtt_total)
        ch4_frac = self._q(wtt_factors["ch4"] / wtt_total)
        n2o_frac = self._q(wtt_factors["n2o"] / wtt_total)

        co2 = self._q(total_emissions * co2_frac)
        ch4 = self._q(total_emissions * ch4_frac)
        n2o = self._q(total_emissions * n2o_frac)

        # Ensure rounding does not lose mass: assign residual to CO2
        residual = self._q(total_emissions - co2 - ch4 - n2o)
        co2 = self._q(co2 + residual)

        return {"co2": co2, "ch4": ch4, "n2o": n2o}

    # ------------------------------------------------------------------
    # Biogenic fuel detection
    # ------------------------------------------------------------------

    def _is_biogenic_fuel(self, fuel_type: FuelType) -> bool:
        """Determine if a fuel type is classified as biogenic.

        Args:
            fuel_type: FuelType enum member.

        Returns:
            True if the fuel is biogenic (biomass, biofuel, landfill gas).
        """
        biogenic_types = {
            FuelType.ETHANOL,
            FuelType.BIODIESEL,
            FuelType.BIOGAS,
            FuelType.HVO,
            FuelType.WOOD_PELLETS,
            FuelType.BIOMASS_SOLID,
            FuelType.BIOMASS_LIQUID,
            FuelType.LANDFILL_GAS,
        }
        return fuel_type in biogenic_types

    # ------------------------------------------------------------------
    # EF source label generation
    # ------------------------------------------------------------------

    def _get_supplier_ef_source_label(
        self, supplier_data: SupplierFuelData
    ) -> str:
        """Generate a human-readable EF source label for a supplier.

        Combines the data source, supplier name, and verification level
        into a descriptive label for audit trail purposes.

        Args:
            supplier_data: SupplierFuelData instance.

        Returns:
            Descriptive source label string (max 100 chars).
        """
        parts = [
            f"Supplier:{supplier_data.supplier_name}",
            f"({supplier_data.data_source.value}",
        ]
        if supplier_data.miq_grade:
            parts.append(f"MiQ-{supplier_data.miq_grade.upper()}")
        parts.append(f"{supplier_data.verification_level})")

        label = " ".join(parts)
        # Truncate to 100 chars to match model constraint
        return label[:100]

    # ------------------------------------------------------------------
    # Supplier DQI score computation
    # ------------------------------------------------------------------

    def _compute_supplier_dqi_score(
        self, supplier_data: SupplierFuelData
    ) -> Decimal:
        """Compute a composite DQI score from supplier data characteristics.

        Quick composite score based on:
        - Verification level (40% weight)
        - Data source type (30% weight)
        - Reporting year recency (30% weight)

        Args:
            supplier_data: SupplierFuelData instance.

        Returns:
            DQI score (1.0 to 5.0, lower is better).
        """
        # Verification component (1.0 = best, 5.0 = worst)
        verification_score = VERIFICATION_SCORES.get(
            supplier_data.verification_level.lower().strip(),
            Decimal("0.15"),
        )
        # Invert to DQI scale: 1.0 score -> 1.0 DQI, 0.15 -> ~4.5
        verification_dqi = self._q(
            ONE + (ONE - verification_score) * Decimal("4")
        )

        # Data source component
        source_dqi_map: Dict[SupplierDataSource, Decimal] = {
            SupplierDataSource.EPD: Decimal("1.0"),
            SupplierDataSource.PCF: Decimal("1.2"),
            SupplierDataSource.LCA: Decimal("1.5"),
            SupplierDataSource.MIQ_CERTIFICATE: Decimal("1.0"),
            SupplierDataSource.OGMP2: Decimal("1.5"),
            SupplierDataSource.DIRECT_MEASUREMENT: Decimal("1.8"),
            SupplierDataSource.CDP: Decimal("2.0"),
            SupplierDataSource.PPA: Decimal("1.5"),
            SupplierDataSource.GREEN_TARIFF: Decimal("2.0"),
            SupplierDataSource.CUSTOM: Decimal("3.0"),
        }
        source_dqi = source_dqi_map.get(
            supplier_data.data_source, Decimal("3.0")
        )

        # Temporal component
        current_year = datetime.now(timezone.utc).year
        if supplier_data.reporting_year is not None:
            age = current_year - supplier_data.reporting_year
            if age <= 0:
                temporal_dqi = Decimal("1.0")
            elif age == 1:
                temporal_dqi = Decimal("1.5")
            elif age == 2:
                temporal_dqi = Decimal("2.0")
            elif age == 3:
                temporal_dqi = Decimal("3.0")
            else:
                temporal_dqi = Decimal("4.0")
        else:
            temporal_dqi = Decimal("3.5")

        # Weighted composite (40/30/30)
        composite = self._q(
            verification_dqi * Decimal("0.40")
            + source_dqi * Decimal("0.30")
            + temporal_dqi * Decimal("0.30")
        )

        # Clamp to 1.0-5.0 range
        composite = max(Decimal("1.0"), min(Decimal("5.0"), composite))

        return composite

    # ------------------------------------------------------------------
    # Supplier uncertainty computation
    # ------------------------------------------------------------------

    def _compute_supplier_uncertainty(
        self, supplier_data: SupplierFuelData
    ) -> Decimal:
        """Compute uncertainty percentage for supplier-specific data.

        Uses the supplier-specific base range (5-10%) and applies
        verification-level and MiQ-grade reductions.

        Args:
            supplier_data: SupplierFuelData instance.

        Returns:
            Uncertainty percentage (Decimal, e.g. 7.5 for +/- 7.5%).
        """
        base_min, base_max = UNCERTAINTY_RANGES[CalculationMethod.SUPPLIER_SPECIFIC]
        base_pct = self._q((base_min + base_max) / _TWO)

        # Apply verification reduction
        reduction = VERIFICATION_UNCERTAINTY_REDUCTION.get(
            supplier_data.verification_level.lower().strip(),
            Decimal("1.0"),
        )
        adjusted = self._q(base_pct * reduction)

        # Apply MiQ grade bonus
        if supplier_data.miq_grade is not None:
            grade = supplier_data.miq_grade.upper().strip()
            if grade in MIQ_UPSTREAM_ADJUSTMENTS:
                adjustment = MIQ_UPSTREAM_ADJUSTMENTS[grade]
                miq_reduction = self._q(
                    (ONE - adjustment) * Decimal("1.5")
                )
                adjusted = self._q(max(Decimal("1.0"), adjusted - miq_reduction))

        return adjusted

    # ------------------------------------------------------------------
    # kWh extraction from records
    # ------------------------------------------------------------------

    def _extract_kwh(
        self, record: Union[FuelConsumptionRecord, ElectricityConsumptionRecord]
    ) -> Decimal:
        """Extract energy content in kWh from a record.

        Args:
            record: Fuel or electricity consumption record.

        Returns:
            Energy content in kWh (Decimal).
        """
        if isinstance(record, ElectricityConsumptionRecord):
            return self._safe_decimal(record.quantity_kwh)
        if isinstance(record, FuelConsumptionRecord):
            if record.quantity_kwh is not None:
                return self._safe_decimal(record.quantity_kwh)
            # Fallback to quantity if unit is kWh
            if record.unit.lower() in ("kwh", "kilowatt_hour", "kilowatt-hour"):
                return self._safe_decimal(record.quantity)
            return ZERO
        return ZERO

    # ------------------------------------------------------------------
    # DQI dimension assessors
    # ------------------------------------------------------------------

    def _assess_temporal_dqi(
        self,
        supplier_data: SupplierFuelData,
        findings: List[str],
    ) -> Decimal:
        """Assess temporal DQI dimension for supplier data.

        Args:
            supplier_data: SupplierFuelData instance.
            findings: List to append findings to.

        Returns:
            Temporal score (1.0 to 5.0).
        """
        current_year = datetime.now(timezone.utc).year

        if supplier_data.reporting_year is None:
            findings.append(
                "Temporal: reporting_year not provided; "
                "cannot assess data recency"
            )
            return Decimal("4.0")

        age = current_year - supplier_data.reporting_year

        if age <= 0:
            findings.append("Temporal: current or future year data")
            return Decimal("1.0")
        elif age == 1:
            findings.append("Temporal: data is 1 year old")
            return Decimal("1.5")
        elif age == 2:
            findings.append("Temporal: data is 2 years old")
            return Decimal("2.5")
        elif age <= 4:
            findings.append(f"Temporal: data is {age} years old")
            return Decimal("3.5")
        else:
            findings.append(
                f"Temporal: data is {age} years old; "
                f"consider requesting updated supplier data"
            )
            return Decimal("5.0")

    def _assess_geographical_dqi(
        self,
        supplier_data: SupplierFuelData,
        findings: List[str],
    ) -> Decimal:
        """Assess geographical DQI dimension for supplier data.

        Supplier-specific data is inherently geographically
        representative since it comes from the actual supplier.

        Args:
            supplier_data: SupplierFuelData instance.
            findings: List to append findings to.

        Returns:
            Geographical score (1.0 to 5.0).
        """
        # Supplier-specific data is always geographically representative
        # of that specific supply chain
        source = supplier_data.data_source

        if source in (
            SupplierDataSource.EPD,
            SupplierDataSource.PCF,
            SupplierDataSource.LCA,
            SupplierDataSource.DIRECT_MEASUREMENT,
        ):
            findings.append(
                "Geographical: supplier-specific primary data; "
                "geographically representative"
            )
            return Decimal("1.0")
        elif source in (
            SupplierDataSource.MIQ_CERTIFICATE,
            SupplierDataSource.OGMP2,
            SupplierDataSource.PPA,
            SupplierDataSource.GREEN_TARIFF,
        ):
            findings.append(
                "Geographical: certified/contracted data; "
                "region-specific"
            )
            return Decimal("1.5")
        elif source == SupplierDataSource.CDP:
            findings.append(
                "Geographical: CDP disclosure data; "
                "corporate-level (may span regions)"
            )
            return Decimal("2.5")
        else:
            findings.append(
                "Geographical: custom or unspecified data source"
            )
            return Decimal("3.0")

    def _assess_technological_dqi(
        self,
        supplier_data: SupplierFuelData,
        findings: List[str],
    ) -> Decimal:
        """Assess technological DQI dimension for supplier data.

        Evaluates how well the data represents the actual technology
        used by the supplier.

        Args:
            supplier_data: SupplierFuelData instance.
            findings: List to append findings to.

        Returns:
            Technological score (1.0 to 5.0).
        """
        source = supplier_data.data_source

        if source in (
            SupplierDataSource.EPD,
            SupplierDataSource.PCF,
            SupplierDataSource.DIRECT_MEASUREMENT,
        ):
            findings.append(
                "Technological: product-specific data; "
                "technologically representative"
            )
            return Decimal("1.0")
        elif source in (
            SupplierDataSource.LCA,
            SupplierDataSource.MIQ_CERTIFICATE,
        ):
            findings.append(
                "Technological: full LCA or certified data; "
                "good technological match"
            )
            return Decimal("1.5")
        elif source in (
            SupplierDataSource.OGMP2,
            SupplierDataSource.PPA,
            SupplierDataSource.GREEN_TARIFF,
        ):
            findings.append(
                "Technological: sector-level data; "
                "moderate technological match"
            )
            return Decimal("2.0")
        elif source == SupplierDataSource.CDP:
            findings.append(
                "Technological: corporate disclosure; "
                "may not reflect specific product technology"
            )
            return Decimal("3.0")
        else:
            findings.append(
                "Technological: custom/unknown data source"
            )
            return Decimal("3.5")

    def _assess_completeness_dqi(
        self,
        supplier_data: SupplierFuelData,
        findings: List[str],
    ) -> Decimal:
        """Assess completeness DQI dimension for supplier data.

        Checks presence of key data fields that indicate complete
        supplier reporting.

        Args:
            supplier_data: SupplierFuelData instance.
            findings: List to append findings to.

        Returns:
            Completeness score (1.0 to 5.0).
        """
        completeness_items = 0
        total_items = 7

        # Check presence of key fields
        if supplier_data.upstream_ef > ZERO:
            completeness_items += 1
        if supplier_data.reporting_year is not None:
            completeness_items += 1
        if supplier_data.verification_level != "unverified":
            completeness_items += 1
        if supplier_data.epd_number is not None:
            completeness_items += 1
        if supplier_data.miq_grade is not None:
            completeness_items += 1
        if supplier_data.ogmp2_level is not None:
            completeness_items += 1
        if supplier_data.metadata:
            completeness_items += 1

        completeness_ratio = Decimal(str(completeness_items)) / Decimal(
            str(total_items)
        )

        if completeness_ratio >= Decimal("0.85"):
            score = Decimal("1.0")
        elif completeness_ratio >= Decimal("0.70"):
            score = Decimal("1.5")
        elif completeness_ratio >= Decimal("0.55"):
            score = Decimal("2.0")
        elif completeness_ratio >= Decimal("0.40"):
            score = Decimal("3.0")
        elif completeness_ratio >= Decimal("0.25"):
            score = Decimal("4.0")
        else:
            score = Decimal("5.0")

        findings.append(
            f"Completeness: {completeness_items}/{total_items} fields "
            f"populated ({float(completeness_ratio) * 100:.0f}%)"
        )
        return score

    def _assess_reliability_dqi(
        self,
        supplier_data: SupplierFuelData,
        findings: List[str],
    ) -> Decimal:
        """Assess reliability DQI dimension for supplier data.

        Based primarily on verification level.

        Args:
            supplier_data: SupplierFuelData instance.
            findings: List to append findings to.

        Returns:
            Reliability score (1.0 to 5.0).
        """
        verification_score = VERIFICATION_SCORES.get(
            supplier_data.verification_level.lower().strip(),
            Decimal("0.15"),
        )

        # Map verification score to DQI (1.0 = highest V score, 5.0 = lowest)
        reliability_dqi = self._q(
            ONE + (ONE - verification_score) * Decimal("4")
        )
        reliability_dqi = max(Decimal("1.0"), min(Decimal("5.0"), reliability_dqi))

        if verification_score >= Decimal("0.85"):
            findings.append(
                f"Reliability: {supplier_data.verification_level} -- "
                f"highly reliable"
            )
        elif verification_score >= Decimal("0.50"):
            findings.append(
                f"Reliability: {supplier_data.verification_level} -- "
                f"moderate reliability"
            )
        else:
            findings.append(
                f"Reliability: {supplier_data.verification_level} -- "
                f"low reliability; consider third-party verification"
            )

        return reliability_dqi

    # ------------------------------------------------------------------
    # Quality tier determination
    # ------------------------------------------------------------------

    def _determine_quality_tier(self, composite: Decimal) -> str:
        """Determine quality tier label from composite DQI score.

        Args:
            composite: Composite DQI score (1.0 to 5.0).

        Returns:
            Quality tier string ("Very High", "High", "Medium", "Low",
            "Very Low").
        """
        for tier_label, (tier_min, tier_max) in DQI_QUALITY_TIERS.items():
            if tier_min <= composite < tier_max:
                return tier_label
        return "Very Low"
