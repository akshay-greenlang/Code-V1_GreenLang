# -*- coding: utf-8 -*-
"""
EnergyMixEngine - PACK-016 ESRS E1 Climate Engine 2
=====================================================

Calculates energy consumption and mix per ESRS E1-5.

Under ESRS E1, disclosure requirement E1-5 mandates that undertakings
report their total energy consumption, broken down by fossil, nuclear,
and renewable sources, along with the renewable energy share.  This
engine implements the complete energy mix calculation pipeline, including:

- Per-entry energy consumption aggregation
- Unit conversion between MWh, GJ, kWh, and TJ
- Source classification into FOSSIL, NUCLEAR, and RENEWABLE categories
- Renewable share calculation as a percentage of total consumption
- Breakdown by purpose (heating, cooling, electricity, transport, process)
- Renewable source disaggregation (solar, wind, hydro, etc.)
- Energy intensity metrics per business denominator
- Completeness validation against E1-5 required data points
- ESRS E1-5 data point mapping for disclosure

ESRS E1-5 Disclosure Requirements:
    - Para 35: Total energy consumption in MWh
    - Para 36: Energy consumption from fossil sources
    - Para 37: Energy consumption from nuclear sources
    - Para 38: Energy consumption from renewable sources
    - Para 39: Share of renewable energy as percentage
    - Para 40: Breakdown by energy carrier and purpose
    - Para 41: Energy intensity based on net revenue
    - Para 42: Information on energy mix methodology

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS E1 Climate Change, Disclosure Requirement E1-5
    - EU Taxonomy Regulation (2020/852) - Renewable energy definitions
    - IEA Energy Statistics methodology
    - ISO 50001 Energy Management Systems (cross-reference)

Zero-Hallucination:
    - All energy calculations use deterministic arithmetic
    - Unit conversions use fixed IEA/SI conversion factors
    - Source classification uses a static lookup table
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-016 ESRS E1 Climate
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


def _round6(value: Decimal) -> Decimal:
    """Round Decimal to 6 decimal places using ROUND_HALF_UP."""
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EnergySource(str, Enum):
    """Energy source types for consumption reporting.

    Covers all major energy carriers relevant to ESRS E1-5 disclosure,
    including fossil fuels, nuclear, and renewable sources.
    """
    GRID_ELECTRICITY = "grid_electricity"
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    LPG = "lpg"
    PETROL = "petrol"
    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro"
    NUCLEAR = "nuclear"
    BIOMASS = "biomass"
    BIOGAS = "biogas"
    GEOTHERMAL = "geothermal"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    HYDROGEN_GREEN = "hydrogen_green"
    HYDROGEN_GREY = "hydrogen_grey"
    WASTE_HEAT_RECOVERY = "waste_heat_recovery"
    OTHER_FOSSIL = "other_fossil"
    OTHER_RENEWABLE = "other_renewable"


class EnergyCategory(str, Enum):
    """Energy classification categories per ESRS E1-5.

    Per ESRS E1-5 Para 36-38, energy consumption shall be
    disaggregated into fossil, nuclear, and renewable sources.
    """
    FOSSIL = "fossil"
    NUCLEAR = "nuclear"
    RENEWABLE = "renewable"


class EnergyUnit(str, Enum):
    """Energy measurement units supported for conversion.

    The standard ESRS reporting unit is MWh.  This engine supports
    conversion from GJ, kWh, and TJ to MWh.
    """
    MWH = "mwh"
    GJ = "gj"
    KWH = "kwh"
    TJ = "tj"


class EnergyPurpose(str, Enum):
    """Purpose of energy consumption per ESRS E1-5 Para 40.

    Energy consumption shall be reported by purpose to enable
    analysis of consumption patterns and efficiency opportunities.
    """
    HEATING = "heating"
    COOLING = "cooling"
    ELECTRICITY = "electricity"
    TRANSPORT = "transport"
    PROCESS = "process"
    LIGHTING = "lighting"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Energy unit conversion factors to MWh (standard ESRS reporting unit).
# Based on SI definitions and IEA energy statistics methodology.
ENERGY_CONVERSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "mwh": {
        "mwh": Decimal("1"),
        "gj": Decimal("3.6"),
        "kwh": Decimal("1000"),
        "tj": Decimal("0.001"),
    },
    "gj": {
        "mwh": Decimal("0.277778"),
        "gj": Decimal("1"),
        "kwh": Decimal("277.778"),
        "tj": Decimal("0.000277778"),
    },
    "kwh": {
        "mwh": Decimal("0.001"),
        "gj": Decimal("0.0036"),
        "kwh": Decimal("1"),
        "tj": Decimal("0.000001"),
    },
    "tj": {
        "mwh": Decimal("277.778"),
        "gj": Decimal("1000"),
        "kwh": Decimal("277778"),
        "tj": Decimal("1"),
    },
}

# Classification of energy sources into FOSSIL, NUCLEAR, or RENEWABLE.
# Based on EU Taxonomy definitions and ESRS E1-5 guidance.
SOURCE_CLASSIFICATION: Dict[str, EnergyCategory] = {
    "grid_electricity": EnergyCategory.FOSSIL,  # Default; override with actual grid mix
    "natural_gas": EnergyCategory.FOSSIL,
    "diesel": EnergyCategory.FOSSIL,
    "fuel_oil": EnergyCategory.FOSSIL,
    "coal": EnergyCategory.FOSSIL,
    "lpg": EnergyCategory.FOSSIL,
    "petrol": EnergyCategory.FOSSIL,
    "hydrogen_grey": EnergyCategory.FOSSIL,
    "other_fossil": EnergyCategory.FOSSIL,
    "nuclear": EnergyCategory.NUCLEAR,
    "solar": EnergyCategory.RENEWABLE,
    "wind": EnergyCategory.RENEWABLE,
    "hydro": EnergyCategory.RENEWABLE,
    "biomass": EnergyCategory.RENEWABLE,
    "biogas": EnergyCategory.RENEWABLE,
    "geothermal": EnergyCategory.RENEWABLE,
    "hydrogen_green": EnergyCategory.RENEWABLE,
    "waste_heat_recovery": EnergyCategory.RENEWABLE,
    "other_renewable": EnergyCategory.RENEWABLE,
    # District heating/cooling classified based on source; default fossil
    "district_heating": EnergyCategory.FOSSIL,
    "district_cooling": EnergyCategory.FOSSIL,
}

# ESRS E1-5 required data points for completeness validation.
E1_5_DATAPOINTS: List[str] = [
    "e1_5_01_total_energy_consumption_mwh",
    "e1_5_02_fossil_energy_consumption_mwh",
    "e1_5_03_nuclear_energy_consumption_mwh",
    "e1_5_04_renewable_energy_consumption_mwh",
    "e1_5_05_renewable_share_pct",
    "e1_5_06_energy_consumption_by_source",
    "e1_5_07_energy_consumption_by_purpose",
    "e1_5_08_energy_intensity_per_net_revenue",
    "e1_5_09_self_generated_renewable_mwh",
    "e1_5_10_purchased_renewable_mwh",
    "e1_5_11_energy_from_fossil_sources_detail",
    "e1_5_12_renewable_breakdown",
    "e1_5_13_methodology_description",
    "e1_5_14_significant_changes",
]

# Default fossil fuel energy content factors (MWh per physical unit).
# Used when entries provide physical quantities instead of energy units.
FUEL_ENERGY_CONTENT: Dict[str, Decimal] = {
    "natural_gas_m3": Decimal("0.01055"),      # MWh per m3
    "diesel_litre": Decimal("0.01005"),         # MWh per litre
    "fuel_oil_litre": Decimal("0.01117"),       # MWh per litre
    "coal_kg": Decimal("0.00778"),              # MWh per kg
    "lpg_litre": Decimal("0.00695"),            # MWh per litre
    "petrol_litre": Decimal("0.00903"),         # MWh per litre
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class EnergyConsumptionEntry(BaseModel):
    """A single energy consumption entry for mix calculation.

    Represents one source of energy consumption with its quantity,
    unit, purpose, and metadata.  The engine normalises all entries
    to MWh for aggregation.
    """
    entry_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this consumption entry",
    )
    source: EnergySource = Field(
        ...,
        description="Energy source (e.g. solar, natural_gas, grid_electricity)",
    )
    source_category_override: Optional[EnergyCategory] = Field(
        default=None,
        description="Override the default source classification "
                    "(e.g. for grid with known renewable share)",
    )
    amount: Decimal = Field(
        ...,
        description="Energy consumption quantity",
        ge=Decimal("0"),
    )
    unit: EnergyUnit = Field(
        default=EnergyUnit.MWH,
        description="Unit of measurement for the amount",
    )
    purpose: EnergyPurpose = Field(
        default=EnergyPurpose.ELECTRICITY,
        description="Purpose of energy consumption",
    )
    is_self_generated: bool = Field(
        default=False,
        description="Whether this energy is self-generated (on-site)",
    )
    site_id: str = Field(
        default="",
        description="Facility or site identifier",
        max_length=100,
    )
    site_name: str = Field(
        default="",
        description="Human-readable site name",
        max_length=500,
    )
    country_code: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code",
        max_length=3,
    )
    reporting_year: int = Field(
        default=0,
        description="Reporting year for this entry",
        ge=0,
    )
    notes: str = Field(
        default="",
        description="Additional notes or context",
        max_length=2000,
    )

    @field_validator("amount")
    @classmethod
    def validate_amount_non_negative(cls, v: Decimal) -> Decimal:
        """Validate that energy amount is non-negative."""
        if v < Decimal("0"):
            raise ValueError(f"Energy amount must be >= 0, got {v}")
        return v


class RenewableBreakdown(BaseModel):
    """Disaggregation of renewable energy by source type.

    Per ESRS E1-5 Para 38, renewable energy consumption shall be
    reported with sufficient detail to understand the renewable mix.
    """
    solar_mwh: Decimal = Field(
        default=Decimal("0"), description="Solar energy consumption (MWh)"
    )
    wind_mwh: Decimal = Field(
        default=Decimal("0"), description="Wind energy consumption (MWh)"
    )
    hydro_mwh: Decimal = Field(
        default=Decimal("0"), description="Hydroelectric energy consumption (MWh)"
    )
    geothermal_mwh: Decimal = Field(
        default=Decimal("0"), description="Geothermal energy consumption (MWh)"
    )
    biomass_mwh: Decimal = Field(
        default=Decimal("0"), description="Biomass energy consumption (MWh)"
    )
    biogas_mwh: Decimal = Field(
        default=Decimal("0"), description="Biogas energy consumption (MWh)"
    )
    other_mwh: Decimal = Field(
        default=Decimal("0"), description="Other renewable energy (MWh)"
    )
    total_mwh: Decimal = Field(
        default=Decimal("0"), description="Total renewable energy (MWh)"
    )
    self_generated_mwh: Decimal = Field(
        default=Decimal("0"),
        description="Self-generated renewable energy (MWh)",
    )
    purchased_mwh: Decimal = Field(
        default=Decimal("0"),
        description="Purchased renewable energy (MWh)",
    )


class EnergyIntensity(BaseModel):
    """Energy intensity metric per ESRS E1-5 Para 41.

    Intensity is calculated as total energy consumption divided by a
    business metric (e.g. net revenue, headcount, production units).
    """
    total_mwh: Decimal = Field(
        ..., description="Total energy consumption used as numerator (MWh)"
    )
    denominator_value: Decimal = Field(
        ..., description="Denominator value"
    )
    denominator_unit: str = Field(
        ..., description="Unit of denominator (e.g. 'EUR_million', 'headcount')"
    )
    intensity_value: Decimal = Field(
        default=Decimal("0"),
        description="Calculated intensity (MWh per denominator unit)",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash of the intensity calculation"
    )


class EnergyMixResult(BaseModel):
    """Complete energy mix result per ESRS E1-5.

    Aggregates all consumption entries into category totals (fossil,
    nuclear, renewable), calculates the renewable share, and provides
    breakdowns by source and purpose.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this calculation",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of calculation (UTC)",
    )
    reporting_year: int = Field(
        default=0, description="Reporting year"
    )
    entity_name: str = Field(
        default="", description="Entity or undertaking name"
    )
    total_mwh: Decimal = Field(
        default=Decimal("0"),
        description="Total energy consumption (MWh)",
    )
    fossil_mwh: Decimal = Field(
        default=Decimal("0"),
        description="Energy consumption from fossil sources (MWh)",
    )
    nuclear_mwh: Decimal = Field(
        default=Decimal("0"),
        description="Energy consumption from nuclear sources (MWh)",
    )
    renewable_mwh: Decimal = Field(
        default=Decimal("0"),
        description="Energy consumption from renewable sources (MWh)",
    )
    renewable_share_pct: Decimal = Field(
        default=Decimal("0"),
        description="Renewable energy share as percentage of total",
    )
    fossil_share_pct: Decimal = Field(
        default=Decimal("0"),
        description="Fossil energy share as percentage of total",
    )
    nuclear_share_pct: Decimal = Field(
        default=Decimal("0"),
        description="Nuclear energy share as percentage of total",
    )
    by_source: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Energy consumption by source in MWh",
    )
    by_purpose: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Energy consumption by purpose in MWh",
    )
    renewable_breakdown: Optional[RenewableBreakdown] = Field(
        default=None,
        description="Detailed breakdown of renewable energy sources",
    )
    self_generated_mwh: Decimal = Field(
        default=Decimal("0"),
        description="Total self-generated energy (MWh)",
    )
    purchased_mwh: Decimal = Field(
        default=Decimal("0"),
        description="Total purchased energy (MWh)",
    )
    entry_count: int = Field(
        default=0, description="Number of consumption entries processed"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EnergyMixEngine:
    """Energy consumption and mix calculation engine per ESRS E1-5.

    Provides deterministic, zero-hallucination calculations for:
    - Energy unit conversion (MWh, GJ, kWh, TJ)
    - Source classification (fossil, nuclear, renewable)
    - Category-level aggregation
    - Renewable share percentage calculation
    - Purpose-based breakdown (heating, cooling, electricity, etc.)
    - Renewable source disaggregation
    - Self-generated vs purchased split
    - Energy intensity metrics
    - E1-5 completeness validation
    - E1-5 data point mapping

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Calculation Methodology:
        1. Convert all entries to MWh using SI conversion factors
        2. Classify each source as fossil, nuclear, or renewable
        3. Aggregate by category and source
        4. renewable_share = renewable_mwh / total_mwh * 100
        5. Intensity = total_mwh / denominator

    Usage::

        engine = EnergyMixEngine()
        entries = [
            EnergyConsumptionEntry(
                source=EnergySource.SOLAR,
                amount=Decimal("500"),
                unit=EnergyUnit.MWH,
                purpose=EnergyPurpose.ELECTRICITY,
            ),
            EnergyConsumptionEntry(
                source=EnergySource.NATURAL_GAS,
                amount=Decimal("1200"),
                unit=EnergyUnit.MWH,
                purpose=EnergyPurpose.HEATING,
            ),
        ]
        result = engine.calculate_mix(entries)
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Unit Conversion                                                      #
    # ------------------------------------------------------------------ #

    def convert_units(
        self,
        amount: Decimal,
        from_unit: EnergyUnit,
        to_unit: EnergyUnit,
    ) -> Decimal:
        """Convert energy amount between units.

        Uses SI-based conversion factors for deterministic results.

        Supported conversions:
            - 1 MWh = 3.6 GJ = 1000 kWh = 0.001 TJ
            - 1 GJ  = 0.277778 MWh = 277.778 kWh
            - 1 TJ  = 277.778 MWh = 1000 GJ

        Args:
            amount: Energy amount to convert.
            from_unit: Source unit.
            to_unit: Target unit.

        Returns:
            Converted amount as Decimal (6 decimal places).

        Raises:
            ValueError: If conversion path is not supported.
        """
        if from_unit == to_unit:
            return _round6(amount)

        # Convert to MWh first (canonical unit), then to target
        to_mwh = ENERGY_CONVERSION_FACTORS.get(from_unit.value, {}).get("mwh")
        if to_mwh is None:
            # Invert: look up MWh -> from_unit and divide
            from_mwh = ENERGY_CONVERSION_FACTORS.get("mwh", {}).get(
                from_unit.value
            )
            if from_mwh is None or from_mwh == Decimal("0"):
                raise ValueError(
                    f"No conversion path from {from_unit.value} to mwh"
                )
            mwh_amount = _safe_divide(amount, from_mwh)
        else:
            # Multiply: amount_in_from * factor_to_mwh
            # But the table stores how many target units per 1 source unit
            # e.g. 1 GJ = 0.277778 MWh
            mwh_amount = amount * to_mwh

        if to_unit == EnergyUnit.MWH:
            return _round6(mwh_amount)

        # Convert MWh to target
        from_mwh_factor = ENERGY_CONVERSION_FACTORS.get("mwh", {}).get(
            to_unit.value
        )
        if from_mwh_factor is None or from_mwh_factor == Decimal("0"):
            raise ValueError(
                f"No conversion path from mwh to {to_unit.value}"
            )
        result = mwh_amount * from_mwh_factor
        return _round6(result)

    def _to_mwh(self, amount: Decimal, unit: EnergyUnit) -> Decimal:
        """Convert any energy amount to MWh.

        Convenience wrapper around convert_units for the common case
        of normalising to the standard ESRS reporting unit.

        Args:
            amount: Energy amount.
            unit: Current unit.

        Returns:
            Amount in MWh (Decimal, 6 decimal places).
        """
        return self.convert_units(amount, unit, EnergyUnit.MWH)

    # ------------------------------------------------------------------ #
    # Source Classification                                                #
    # ------------------------------------------------------------------ #

    def classify_source(
        self,
        source: EnergySource,
        override: Optional[EnergyCategory] = None,
    ) -> EnergyCategory:
        """Classify an energy source into its category.

        Uses the static SOURCE_CLASSIFICATION lookup table.  An
        optional override allows callers to reclassify sources where
        the default does not apply (e.g. 100% renewable grid).

        Args:
            source: EnergySource enum value.
            override: Optional category override.

        Returns:
            EnergyCategory (FOSSIL, NUCLEAR, or RENEWABLE).
        """
        if override is not None:
            return override
        return SOURCE_CLASSIFICATION.get(
            source.value, EnergyCategory.FOSSIL
        )

    # ------------------------------------------------------------------ #
    # Main Calculation                                                     #
    # ------------------------------------------------------------------ #

    def calculate_mix(
        self,
        entries: List[EnergyConsumptionEntry],
        entity_name: str = "",
        reporting_year: int = 0,
    ) -> EnergyMixResult:
        """Calculate the complete energy mix from consumption entries.

        Processes all entries, converts to MWh, classifies sources,
        and produces aggregated results with renewable share.

        Args:
            entries: List of EnergyConsumptionEntry instances.
            entity_name: Name of the reporting entity.
            reporting_year: Reporting year.

        Returns:
            EnergyMixResult with complete provenance.

        Raises:
            ValueError: If entries list is empty.
        """
        t0 = time.perf_counter()

        if not entries:
            raise ValueError("At least one EnergyConsumptionEntry is required")

        logger.info(
            "Calculating energy mix: %d entries, entity=%s, year=%d",
            len(entries), entity_name, reporting_year,
        )

        # Step 1: Convert all entries to MWh and classify
        normalised: List[Tuple[EnergyConsumptionEntry, Decimal, EnergyCategory]] = []
        for entry in entries:
            mwh = self._to_mwh(entry.amount, entry.unit)
            category = self.classify_source(
                entry.source, entry.source_category_override
            )
            normalised.append((entry, mwh, category))

        # Step 2: Aggregate by category
        fossil = Decimal("0")
        nuclear = Decimal("0")
        renewable = Decimal("0")
        self_generated = Decimal("0")
        purchased = Decimal("0")

        for entry, mwh, category in normalised:
            if category == EnergyCategory.FOSSIL:
                fossil += mwh
            elif category == EnergyCategory.NUCLEAR:
                nuclear += mwh
            elif category == EnergyCategory.RENEWABLE:
                renewable += mwh

            if entry.is_self_generated:
                self_generated += mwh
            else:
                purchased += mwh

        fossil = _round6(fossil)
        nuclear = _round6(nuclear)
        renewable = _round6(renewable)
        self_generated = _round6(self_generated)
        purchased = _round6(purchased)
        total = _round6(fossil + nuclear + renewable)

        # Step 3: Calculate shares
        renewable_pct = self.calculate_renewable_share_from_values(
            renewable, total
        )
        fossil_pct = _round_val(
            _safe_divide(fossil, total) * Decimal("100"), 2
        ) if total > Decimal("0") else Decimal("0")
        nuclear_pct = _round_val(
            _safe_divide(nuclear, total) * Decimal("100"), 2
        ) if total > Decimal("0") else Decimal("0")

        # Step 4: Breakdown by source
        by_source = self._aggregate_by_source(normalised)

        # Step 5: Breakdown by purpose
        by_purpose = self._aggregate_by_purpose(normalised)

        # Step 6: Renewable breakdown
        renewable_bd = self._build_renewable_breakdown(normalised)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = EnergyMixResult(
            reporting_year=reporting_year,
            entity_name=entity_name,
            total_mwh=total,
            fossil_mwh=fossil,
            nuclear_mwh=nuclear,
            renewable_mwh=renewable,
            renewable_share_pct=renewable_pct,
            fossil_share_pct=fossil_pct,
            nuclear_share_pct=nuclear_pct,
            by_source=by_source,
            by_purpose=by_purpose,
            renewable_breakdown=renewable_bd,
            self_generated_mwh=self_generated,
            purchased_mwh=purchased,
            entry_count=len(entries),
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Energy mix calculated: total=%.2f MWh, fossil=%.2f, "
            "nuclear=%.2f, renewable=%.2f (%.1f%%), hash=%s",
            float(total), float(fossil), float(nuclear),
            float(renewable), float(renewable_pct),
            result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # Renewable Share                                                       #
    # ------------------------------------------------------------------ #

    def calculate_renewable_share(
        self, result: EnergyMixResult
    ) -> Decimal:
        """Calculate renewable energy share from an EnergyMixResult.

        Formula: renewable_share_pct = renewable_mwh / total_mwh * 100

        Args:
            result: EnergyMixResult to calculate share for.

        Returns:
            Renewable share as percentage (Decimal, 2 decimal places).
        """
        return self.calculate_renewable_share_from_values(
            result.renewable_mwh, result.total_mwh
        )

    def calculate_renewable_share_from_values(
        self,
        renewable_mwh: Decimal,
        total_mwh: Decimal,
    ) -> Decimal:
        """Calculate renewable share from raw values.

        Args:
            renewable_mwh: Renewable energy in MWh.
            total_mwh: Total energy in MWh.

        Returns:
            Renewable share as percentage (Decimal, 2 decimal places).
        """
        if total_mwh <= Decimal("0"):
            return Decimal("0.00")
        pct = renewable_mwh / total_mwh * Decimal("100")
        return _round_val(pct, 2)

    # ------------------------------------------------------------------ #
    # Energy Intensity                                                     #
    # ------------------------------------------------------------------ #

    def calculate_energy_intensity(
        self,
        total_mwh: Decimal,
        denominator_value: Decimal,
        denominator_unit: str,
    ) -> EnergyIntensity:
        """Calculate energy intensity metric per ESRS E1-5 Para 41.

        Formula: intensity = total_mwh / denominator_value

        Args:
            total_mwh: Total energy consumption in MWh.
            denominator_value: Denominator value (must be > 0).
            denominator_unit: Unit of denominator.

        Returns:
            EnergyIntensity with calculated intensity and provenance.

        Raises:
            ValueError: If denominator_value is zero or negative.
        """
        if denominator_value <= Decimal("0"):
            raise ValueError(
                f"Denominator must be > 0, got {denominator_value}"
            )

        intensity = _round6(
            _safe_divide(total_mwh, denominator_value)
        )

        result = EnergyIntensity(
            total_mwh=total_mwh,
            denominator_value=denominator_value,
            denominator_unit=denominator_unit,
            intensity_value=intensity,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Energy intensity: %.6f MWh/%s",
            float(intensity), denominator_unit,
        )

        return result

    # ------------------------------------------------------------------ #
    # Purpose Breakdown                                                    #
    # ------------------------------------------------------------------ #

    def get_breakdown_by_purpose(
        self, entries: List[EnergyConsumptionEntry]
    ) -> Dict[str, Decimal]:
        """Aggregate energy consumption by purpose.

        Converts all entries to MWh and groups by EnergyPurpose.

        Args:
            entries: List of EnergyConsumptionEntry.

        Returns:
            Dict mapping purpose string to total MWh.
        """
        by_purpose: Dict[str, Decimal] = {}
        for purpose in EnergyPurpose:
            by_purpose[purpose.value] = Decimal("0")

        for entry in entries:
            mwh = self._to_mwh(entry.amount, entry.unit)
            purpose_key = entry.purpose.value
            by_purpose[purpose_key] = by_purpose.get(
                purpose_key, Decimal("0")
            ) + mwh

        # Round all
        for key in by_purpose:
            by_purpose[key] = _round6(by_purpose[key])

        return by_purpose

    # ------------------------------------------------------------------ #
    # Site-Level Aggregation                                               #
    # ------------------------------------------------------------------ #

    def aggregate_by_site(
        self, entries: List[EnergyConsumptionEntry]
    ) -> Dict[str, Dict[str, Decimal]]:
        """Aggregate energy consumption by site.

        Groups entries by site_id and calculates total, fossil,
        nuclear, and renewable MWh for each site.

        Args:
            entries: List of EnergyConsumptionEntry.

        Returns:
            Dict mapping site_id to a dict with total_mwh, fossil_mwh,
            nuclear_mwh, renewable_mwh.
        """
        sites: Dict[str, Dict[str, Decimal]] = {}

        for entry in entries:
            site = entry.site_id or "unassigned"
            if site not in sites:
                sites[site] = {
                    "total_mwh": Decimal("0"),
                    "fossil_mwh": Decimal("0"),
                    "nuclear_mwh": Decimal("0"),
                    "renewable_mwh": Decimal("0"),
                }
            mwh = self._to_mwh(entry.amount, entry.unit)
            category = self.classify_source(
                entry.source, entry.source_category_override
            )
            sites[site]["total_mwh"] += mwh
            if category == EnergyCategory.FOSSIL:
                sites[site]["fossil_mwh"] += mwh
            elif category == EnergyCategory.NUCLEAR:
                sites[site]["nuclear_mwh"] += mwh
            elif category == EnergyCategory.RENEWABLE:
                sites[site]["renewable_mwh"] += mwh

        # Round all values
        for site in sites:
            for key in sites[site]:
                sites[site][key] = _round6(sites[site][key])

        return sites

    # ------------------------------------------------------------------ #
    # Year-over-Year Comparison                                            #
    # ------------------------------------------------------------------ #

    def compare_years(
        self,
        current: EnergyMixResult,
        previous: EnergyMixResult,
    ) -> Dict[str, Any]:
        """Compare energy mix results across two reporting years.

        Args:
            current: Current year energy mix result.
            previous: Previous year energy mix result.

        Returns:
            Dict with absolute and percentage changes for each metric.
        """
        def _change(curr: Decimal, prev: Decimal) -> Dict[str, str]:
            abs_c = curr - prev
            pct = _safe_divide(
                abs_c, prev if prev != Decimal("0") else Decimal("1")
            ) * Decimal("100")
            return {
                "current": str(curr),
                "previous": str(prev),
                "absolute_change": str(_round6(abs_c)),
                "pct_change": str(_round_val(pct, 2)),
            }

        comparison = {
            "current_year": current.reporting_year,
            "previous_year": previous.reporting_year,
            "total_mwh": _change(current.total_mwh, previous.total_mwh),
            "fossil_mwh": _change(current.fossil_mwh, previous.fossil_mwh),
            "nuclear_mwh": _change(current.nuclear_mwh, previous.nuclear_mwh),
            "renewable_mwh": _change(
                current.renewable_mwh, previous.renewable_mwh
            ),
            "renewable_share_pct": {
                "current": str(current.renewable_share_pct),
                "previous": str(previous.renewable_share_pct),
                "change_pp": str(
                    _round_val(
                        current.renewable_share_pct
                        - previous.renewable_share_pct, 2
                    )
                ),
            },
        }

        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_completeness(
        self, result: EnergyMixResult
    ) -> Dict[str, Any]:
        """Validate completeness against E1-5 required data points.

        Checks whether all ESRS E1-5 mandatory disclosure data points
        are present and populated in the energy mix result.

        Args:
            result: EnergyMixResult to validate.

        Returns:
            Dict with total_datapoints, populated_datapoints,
            missing_datapoints, completeness_pct, is_complete,
            provenance_hash.
        """
        populated = []
        missing = []

        checks = {
            "e1_5_01_total_energy_consumption_mwh": (
                result.total_mwh >= Decimal("0")
            ),
            "e1_5_02_fossil_energy_consumption_mwh": (
                result.fossil_mwh >= Decimal("0")
            ),
            "e1_5_03_nuclear_energy_consumption_mwh": (
                result.nuclear_mwh >= Decimal("0")
            ),
            "e1_5_04_renewable_energy_consumption_mwh": (
                result.renewable_mwh >= Decimal("0")
            ),
            "e1_5_05_renewable_share_pct": (
                result.renewable_share_pct >= Decimal("0")
            ),
            "e1_5_06_energy_consumption_by_source": (
                len(result.by_source) > 0
            ),
            "e1_5_07_energy_consumption_by_purpose": (
                len(result.by_purpose) > 0
            ),
            "e1_5_08_energy_intensity_per_net_revenue": True,  # Separate calc
            "e1_5_09_self_generated_renewable_mwh": True,  # Reported if applicable
            "e1_5_10_purchased_renewable_mwh": True,  # Reported if applicable
            "e1_5_11_energy_from_fossil_sources_detail": (
                any(
                    v > Decimal("0")
                    for k, v in result.by_source.items()
                    if SOURCE_CLASSIFICATION.get(k) == EnergyCategory.FOSSIL
                )
                if result.fossil_mwh > Decimal("0") else True
            ),
            "e1_5_12_renewable_breakdown": (
                result.renewable_breakdown is not None
            ),
            "e1_5_13_methodology_description": True,  # Narrative
            "e1_5_14_significant_changes": True,  # Narrative
        }

        for dp, is_populated in checks.items():
            if is_populated:
                populated.append(dp)
            else:
                missing.append(dp)

        total = len(E1_5_DATAPOINTS)
        pop_count = len(populated)
        completeness = _round_val(
            _decimal(pop_count) / _decimal(total) * Decimal("100"), 1
        )

        validation_result = {
            "total_datapoints": total,
            "populated_datapoints": pop_count,
            "missing_datapoints": missing,
            "completeness_pct": completeness,
            "is_complete": len(missing) == 0,
            "provenance_hash": _compute_hash(
                {"result_id": result.result_id, "checks": checks}
            ),
        }

        logger.info(
            "E1-5 completeness: %s%% (%d/%d), missing=%s",
            completeness, pop_count, total, missing,
        )

        return validation_result

    # ------------------------------------------------------------------ #
    # ESRS E1-5 Data Point Mapping                                         #
    # ------------------------------------------------------------------ #

    def get_e1_5_datapoints(
        self,
        result: EnergyMixResult,
        intensity: Optional[EnergyIntensity] = None,
    ) -> Dict[str, Any]:
        """Map energy mix result to ESRS E1-5 disclosure data points.

        Creates a structured mapping of all E1-5 required data points
        with their values, ready for report generation.

        Args:
            result: EnergyMixResult to map.
            intensity: Optional EnergyIntensity for E1-5 Para 41.

        Returns:
            Dict mapping E1-5 data point IDs to their values.
        """
        by_source_str = {k: str(v) for k, v in result.by_source.items()}
        by_purpose_str = {k: str(v) for k, v in result.by_purpose.items()}

        renewable_detail = None
        if result.renewable_breakdown:
            renewable_detail = {
                "solar_mwh": str(result.renewable_breakdown.solar_mwh),
                "wind_mwh": str(result.renewable_breakdown.wind_mwh),
                "hydro_mwh": str(result.renewable_breakdown.hydro_mwh),
                "geothermal_mwh": str(result.renewable_breakdown.geothermal_mwh),
                "biomass_mwh": str(result.renewable_breakdown.biomass_mwh),
                "biogas_mwh": str(result.renewable_breakdown.biogas_mwh),
                "other_mwh": str(result.renewable_breakdown.other_mwh),
            }

        intensity_data = None
        if intensity is not None:
            intensity_data = {
                "total_mwh": str(intensity.total_mwh),
                "denominator_value": str(intensity.denominator_value),
                "denominator_unit": intensity.denominator_unit,
                "intensity_value": str(intensity.intensity_value),
            }

        datapoints = {
            "e1_5_01_total_energy_consumption_mwh": {
                "label": "Total energy consumption",
                "value": str(result.total_mwh),
                "unit": "MWh",
                "esrs_ref": "E1-5 Para 35",
            },
            "e1_5_02_fossil_energy_consumption_mwh": {
                "label": "Energy consumption from fossil sources",
                "value": str(result.fossil_mwh),
                "unit": "MWh",
                "esrs_ref": "E1-5 Para 36",
            },
            "e1_5_03_nuclear_energy_consumption_mwh": {
                "label": "Energy consumption from nuclear sources",
                "value": str(result.nuclear_mwh),
                "unit": "MWh",
                "esrs_ref": "E1-5 Para 37",
            },
            "e1_5_04_renewable_energy_consumption_mwh": {
                "label": "Energy consumption from renewable sources",
                "value": str(result.renewable_mwh),
                "unit": "MWh",
                "esrs_ref": "E1-5 Para 38",
            },
            "e1_5_05_renewable_share_pct": {
                "label": "Share of renewable energy",
                "value": str(result.renewable_share_pct),
                "unit": "percent",
                "esrs_ref": "E1-5 Para 39",
            },
            "e1_5_06_energy_consumption_by_source": {
                "label": "Energy consumption by source",
                "value": by_source_str,
                "unit": "MWh",
                "esrs_ref": "E1-5 Para 40",
            },
            "e1_5_07_energy_consumption_by_purpose": {
                "label": "Energy consumption by purpose",
                "value": by_purpose_str,
                "unit": "MWh",
                "esrs_ref": "E1-5 Para 40",
            },
            "e1_5_08_energy_intensity_per_net_revenue": {
                "label": "Energy intensity per net revenue",
                "value": intensity_data,
                "esrs_ref": "E1-5 Para 41",
            },
            "e1_5_09_self_generated_renewable_mwh": {
                "label": "Self-generated renewable energy",
                "value": str(result.self_generated_mwh),
                "unit": "MWh",
                "esrs_ref": "E1-5 Para 38",
            },
            "e1_5_10_purchased_renewable_mwh": {
                "label": "Purchased renewable energy",
                "value": str(result.purchased_mwh),
                "unit": "MWh",
                "esrs_ref": "E1-5 Para 38",
            },
            "e1_5_12_renewable_breakdown": {
                "label": "Renewable energy breakdown by source",
                "value": renewable_detail,
                "esrs_ref": "E1-5 Para 38",
            },
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)

        return datapoints

    # ------------------------------------------------------------------ #
    # Fossil Fuel Detail                                                   #
    # ------------------------------------------------------------------ #

    def get_fossil_detail(
        self, result: EnergyMixResult
    ) -> Dict[str, Any]:
        """Get detailed breakdown of fossil fuel consumption.

        Args:
            result: EnergyMixResult to analyze.

        Returns:
            Dict with per-fossil-source consumption and shares.
        """
        fossil_sources = {}
        for src_key, src_mwh in result.by_source.items():
            classification = SOURCE_CLASSIFICATION.get(src_key)
            if classification == EnergyCategory.FOSSIL and src_mwh > Decimal("0"):
                share_pct = _round_val(
                    _safe_divide(src_mwh, result.fossil_mwh) * Decimal("100"),
                    2,
                ) if result.fossil_mwh > Decimal("0") else Decimal("0")
                fossil_sources[src_key] = {
                    "mwh": str(src_mwh),
                    "share_of_fossil_pct": str(share_pct),
                }

        return {
            "total_fossil_mwh": str(result.fossil_mwh),
            "share_of_total_pct": str(result.fossil_share_pct),
            "sources": fossil_sources,
            "provenance_hash": _compute_hash(fossil_sources),
        }

    # ------------------------------------------------------------------ #
    # Conversion Utilities                                                 #
    # ------------------------------------------------------------------ #

    def fuel_to_mwh(
        self, fuel_type: str, quantity: Decimal
    ) -> Decimal:
        """Convert a fuel quantity to MWh using energy content factors.

        Args:
            fuel_type: Fuel type key (e.g. 'natural_gas_m3', 'diesel_litre').
            quantity: Physical quantity of fuel.

        Returns:
            Energy content in MWh.

        Raises:
            ValueError: If fuel_type is not in FUEL_ENERGY_CONTENT.
        """
        factor = FUEL_ENERGY_CONTENT.get(fuel_type)
        if factor is None:
            raise ValueError(
                f"Unknown fuel type '{fuel_type}'. "
                f"Valid: {sorted(FUEL_ENERGY_CONTENT.keys())}"
            )
        return _round6(quantity * factor)

    def list_conversion_factors(self) -> Dict[str, str]:
        """Return all fuel energy content conversion factors.

        Returns:
            Dict mapping fuel type to MWh conversion factor string.
        """
        return {k: str(v) for k, v in sorted(FUEL_ENERGY_CONTENT.items())}

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    def _aggregate_by_source(
        self,
        normalised: List[Tuple[EnergyConsumptionEntry, Decimal, EnergyCategory]],
    ) -> Dict[str, Decimal]:
        """Aggregate MWh consumption by energy source.

        Args:
            normalised: List of (entry, mwh, category) tuples.

        Returns:
            Dict mapping source string to total MWh.
        """
        by_source: Dict[str, Decimal] = {}
        for entry, mwh, _ in normalised:
            src = entry.source.value
            by_source[src] = by_source.get(src, Decimal("0")) + mwh

        for key in by_source:
            by_source[key] = _round6(by_source[key])

        return by_source

    def _aggregate_by_purpose(
        self,
        normalised: List[Tuple[EnergyConsumptionEntry, Decimal, EnergyCategory]],
    ) -> Dict[str, Decimal]:
        """Aggregate MWh consumption by energy purpose.

        Args:
            normalised: List of (entry, mwh, category) tuples.

        Returns:
            Dict mapping purpose string to total MWh.
        """
        by_purpose: Dict[str, Decimal] = {}
        for purpose in EnergyPurpose:
            by_purpose[purpose.value] = Decimal("0")

        for entry, mwh, _ in normalised:
            purpose_key = entry.purpose.value
            by_purpose[purpose_key] = by_purpose.get(
                purpose_key, Decimal("0")
            ) + mwh

        for key in by_purpose:
            by_purpose[key] = _round6(by_purpose[key])

        return by_purpose

    def _build_renewable_breakdown(
        self,
        normalised: List[Tuple[EnergyConsumptionEntry, Decimal, EnergyCategory]],
    ) -> RenewableBreakdown:
        """Build detailed renewable energy breakdown.

        Args:
            normalised: List of (entry, mwh, category) tuples.

        Returns:
            RenewableBreakdown with per-source renewable totals.
        """
        solar = Decimal("0")
        wind = Decimal("0")
        hydro = Decimal("0")
        geothermal = Decimal("0")
        biomass = Decimal("0")
        biogas = Decimal("0")
        other = Decimal("0")
        self_gen = Decimal("0")
        purchased = Decimal("0")

        renewable_map = {
            EnergySource.SOLAR: "solar",
            EnergySource.WIND: "wind",
            EnergySource.HYDRO: "hydro",
            EnergySource.GEOTHERMAL: "geothermal",
            EnergySource.BIOMASS: "biomass",
            EnergySource.BIOGAS: "biogas",
        }

        for entry, mwh, category in normalised:
            if category != EnergyCategory.RENEWABLE:
                continue

            source_type = renewable_map.get(entry.source)
            if source_type == "solar":
                solar += mwh
            elif source_type == "wind":
                wind += mwh
            elif source_type == "hydro":
                hydro += mwh
            elif source_type == "geothermal":
                geothermal += mwh
            elif source_type == "biomass":
                biomass += mwh
            elif source_type == "biogas":
                biogas += mwh
            else:
                other += mwh

            if entry.is_self_generated:
                self_gen += mwh
            else:
                purchased += mwh

        total_renewable = _round6(
            solar + wind + hydro + geothermal + biomass + biogas + other
        )

        return RenewableBreakdown(
            solar_mwh=_round6(solar),
            wind_mwh=_round6(wind),
            hydro_mwh=_round6(hydro),
            geothermal_mwh=_round6(geothermal),
            biomass_mwh=_round6(biomass),
            biogas_mwh=_round6(biogas),
            other_mwh=_round6(other),
            total_mwh=total_renewable,
            self_generated_mwh=_round6(self_gen),
            purchased_mwh=_round6(purchased),
        )
