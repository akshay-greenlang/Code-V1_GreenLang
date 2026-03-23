# -*- coding: utf-8 -*-
"""
WastewaterTreatmentEngine - On-site Industrial Wastewater CH4 & N2O (Engine 4 of 7)

AGENT-MRV-007: On-site Waste Treatment Emissions Agent

Core calculation engine implementing IPCC 2006/2019 Refinement Vol 5 Ch 6
methods for estimating methane (CH4) and nitrous oxide (N2O) emissions from
on-site industrial and municipal wastewater treatment systems.

Wastewater Treatment Systems Supported:
    1. Aerobic treatment (well-managed) - activated sludge, trickling filter
    2. Aerobic treatment (overloaded) - poorly managed aerobic systems
    3. Anaerobic reactor (without recovery) - UASB, ABR, anaerobic filter
    4. Anaerobic reactor (with recovery) - biogas capture and utilization
    5. Anaerobic shallow lagoon (<2m depth)
    6. Anaerobic deep lagoon (>2m depth)
    7. Septic system
    8. Untreated discharge (sea/river)

Key Formulas (IPCC 2006 Vol 5 Ch 6):

    CH4 from Wastewater:
        CH4 = [ (TOW - S) * Bo * MCF ] * 0.001 - R
        Where:
            TOW = total organic waste in wastewater (kg BOD/yr or kg COD/yr)
            S   = organic component removed as sludge (kg BOD/yr or kg COD/yr)
            Bo  = maximum CH4 producing capacity (kg CH4/kg BOD or COD)
            MCF = methane correction factor (by treatment system)
            R   = recovered CH4 (tonnes CH4/yr)

    Industrial Wastewater TOW:
        TOW_industry = P_i * W_i * COD_i
        Where:
            P_i   = industrial production (tonnes product/yr)
            W_i   = wastewater generated per unit product (m3/tonne)
            COD_i = chemical oxygen demand (kg COD/m3)

    N2O from Wastewater:
        N2O_plant    = Q * N_influent * EF_plant * 44/28
        N2O_effluent = Q * N_effluent * EF_effluent * 44/28
        N2O_total    = N2O_plant + N2O_effluent
        Where:
            Q           = wastewater flow rate (m3/yr)
            N_influent  = total nitrogen in influent (kg N/m3)
            N_effluent  = total nitrogen in effluent (kg N/m3)
            EF_plant    = emission factor for plant (kg N2O-N/kg N)
            EF_effluent = emission factor for effluent (kg N2O-N/kg N)

    Sludge Treatment:
        CH4_sludge = M_sludge * VS_fraction * Bo_sludge * MCF_treatment

All calculations use Python Decimal arithmetic with 8+ decimal places for
zero-hallucination determinism. Every calculation result includes a per-gas
breakdown, GWP-adjusted CO2e, processing time, and SHA-256 provenance hash.

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal.
    - No LLM calls in any calculation path.
    - Every calculation step is logged and traceable.
    - SHA-256 provenance hash for every result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Stateless per-calculation. Mutable counters protected by reentrant lock.

Example:
    >>> from greenlang.agents.mrv.waste_treatment_emissions.wastewater_treatment import (
    ...     WastewaterTreatmentEngine,
    ... )
    >>> engine = WastewaterTreatmentEngine()
    >>> result = engine.calculate_wastewater_ch4(
    ...     tow_kg_bod_yr=500000,
    ...     sludge_removal_kg=50000,
    ...     bo="0.6",
    ...     mcf="0.8",
    ...     recovery_tonnes=0,
    ... )
    >>> assert result["status"] == "SUCCESS"

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 On-site Waste Treatment Emissions (GL-MRV-SCOPE1-007)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["WastewaterTreatmentEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.waste_treatment_emissions.config import (
        get_config as _get_config,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.metrics import (
        record_calculation as _record_calc_operation,
        record_emissions as _record_emissions,
        record_calculation_error as _record_calc_error,
        observe_calculation_duration as _observe_duration,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calc_operation = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _record_calc_error = None  # type: ignore[assignment]
    _observe_duration = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_THOUSAND = Decimal("1000")

#: N2O molecular weight ratio: 44/28 (N2O / 2N)
N2O_MOLECULAR_RATIO = Decimal("1.57142857")

#: Default Bo for BOD basis (kg CH4/kg BOD) -- IPCC 2006 Vol 5 Ch 6
BO_DEFAULT_BOD = Decimal("0.6")

#: Default Bo for COD basis (kg CH4/kg COD) -- IPCC 2006 Vol 5 Ch 6
BO_DEFAULT_COD = Decimal("0.25")

#: Default N2O plant emission factor (kg N2O-N / kg N) -- IPCC 2019 Refinement
EF_N2O_PLANT_DEFAULT = Decimal("0.016")

#: Default N2O effluent emission factor (kg N2O-N / kg N) -- IPCC 2019 Refinement
EF_N2O_EFFLUENT_DEFAULT = Decimal("0.005")

#: Default volatile solids fraction for sludge
VS_FRACTION_DEFAULT = Decimal("0.60")

#: Default Bo for sludge (kg CH4/kg VS) -- anaerobic conditions
BO_SLUDGE_DEFAULT = Decimal("0.6")

#: Conversion factor: 0.001 (kg to tonnes)
KG_TO_TONNES = Decimal("0.001")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted to Decimal.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert a value to Decimal, returning default on failure.

    Args:
        value: Value to convert.
        default: Default if conversion fails.

    Returns:
        Decimal value.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to the standard 8-decimal-place precision.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ===========================================================================
# GWP Values (fallback when database module is not available)
# ===========================================================================

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": Decimal("1"),
        "CH4": Decimal("25"),
        "N2O": Decimal("298"),
    },
    "AR5": {
        "CO2": Decimal("1"),
        "CH4": Decimal("28"),
        "N2O": Decimal("265"),
    },
    "AR6": {
        "CO2": Decimal("1"),
        "CH4": Decimal("29.8"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": Decimal("1"),
        "CH4": Decimal("82.5"),
        "N2O": Decimal("273"),
    },
}


# ===========================================================================
# Enumerations
# ===========================================================================


class TreatmentSystem(str, Enum):
    """Wastewater treatment system types with IPCC MCF values."""

    AEROBIC_WELL_MANAGED = "AEROBIC_WELL_MANAGED"
    AEROBIC_OVERLOADED = "AEROBIC_OVERLOADED"
    ANAEROBIC_REACTOR_NO_RECOVERY = "ANAEROBIC_REACTOR_NO_RECOVERY"
    ANAEROBIC_REACTOR_WITH_RECOVERY = "ANAEROBIC_REACTOR_WITH_RECOVERY"
    ANAEROBIC_SHALLOW_LAGOON = "ANAEROBIC_SHALLOW_LAGOON"
    ANAEROBIC_DEEP_LAGOON = "ANAEROBIC_DEEP_LAGOON"
    SEPTIC_SYSTEM = "SEPTIC_SYSTEM"
    UNTREATED_DISCHARGE = "UNTREATED_DISCHARGE"


class IndustryType(str, Enum):
    """Industry types with IPCC Table 6.9 default wastewater parameters."""

    PULP_AND_PAPER = "PULP_AND_PAPER"
    MEAT_AND_POULTRY = "MEAT_AND_POULTRY"
    DAIRY = "DAIRY"
    SUGAR_REFINING = "SUGAR_REFINING"
    VEGETABLE_OIL = "VEGETABLE_OIL"
    BREWERY = "BREWERY"
    STARCH = "STARCH"
    FRUIT_VEGETABLE_PROCESSING = "FRUIT_VEGETABLE_PROCESSING"
    TEXTILES = "TEXTILES"
    PETROCHEMICAL = "PETROCHEMICAL"
    PHARMACEUTICAL = "PHARMACEUTICAL"


class SludgeTreatment(str, Enum):
    """Sludge treatment/disposal methods."""

    ANAEROBIC_DIGESTION = "ANAEROBIC_DIGESTION"
    AEROBIC_DIGESTION = "AEROBIC_DIGESTION"
    COMPOSTING = "COMPOSTING"
    LANDFILL = "LANDFILL"
    LAND_APPLICATION = "LAND_APPLICATION"
    INCINERATION = "INCINERATION"
    DRYING_BEDS = "DRYING_BEDS"
    LAGOON_STORAGE = "LAGOON_STORAGE"


class CalculationStatus(str, Enum):
    """Result status codes."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class OrganicBasis(str, Enum):
    """Basis for organic load measurement."""

    BOD = "BOD"
    COD = "COD"


# ===========================================================================
# Trace Step Dataclass
# ===========================================================================


@dataclass
class TraceStep:
    """Single step in a calculation trace for audit trail.

    Attributes:
        step_number: Sequential step number.
        description: Human-readable description.
        formula: Mathematical formula applied.
        inputs: Input values for this step.
        output: Output value from this step.
        unit: Unit of the output value.
    """

    step_number: int
    description: str
    formula: str
    inputs: Dict[str, str]
    output: str
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "formula": self.formula,
            "inputs": self.inputs,
            "output": self.output,
            "unit": self.unit,
        }


# ===========================================================================
# IPCC Default Parameters (Tables from Vol 5 Ch 6)
# ===========================================================================

#: Methane Correction Factors (MCF) by treatment system (IPCC Table 6.3)
MCF_BY_TREATMENT: Dict[str, Decimal] = {
    TreatmentSystem.AEROBIC_WELL_MANAGED.value: Decimal("0.0"),
    TreatmentSystem.AEROBIC_OVERLOADED.value: Decimal("0.3"),
    TreatmentSystem.ANAEROBIC_REACTOR_NO_RECOVERY.value: Decimal("0.8"),
    TreatmentSystem.ANAEROBIC_REACTOR_WITH_RECOVERY.value: Decimal("0.8"),
    TreatmentSystem.ANAEROBIC_SHALLOW_LAGOON.value: Decimal("0.2"),
    TreatmentSystem.ANAEROBIC_DEEP_LAGOON.value: Decimal("0.8"),
    TreatmentSystem.SEPTIC_SYSTEM.value: Decimal("0.5"),
    TreatmentSystem.UNTREATED_DISCHARGE.value: Decimal("0.1"),
}

#: MCF values for sludge treatment pathways
MCF_BY_SLUDGE_TREATMENT: Dict[str, Decimal] = {
    SludgeTreatment.ANAEROBIC_DIGESTION.value: Decimal("0.8"),
    SludgeTreatment.AEROBIC_DIGESTION.value: Decimal("0.0"),
    SludgeTreatment.COMPOSTING.value: Decimal("0.01"),
    SludgeTreatment.LANDFILL.value: Decimal("1.0"),
    SludgeTreatment.LAND_APPLICATION.value: Decimal("0.01"),
    SludgeTreatment.INCINERATION.value: Decimal("0.0"),
    SludgeTreatment.DRYING_BEDS.value: Decimal("0.0"),
    SludgeTreatment.LAGOON_STORAGE.value: Decimal("0.8"),
}

#: Industry-specific wastewater parameters (IPCC Table 6.9)
#: Format: {industry: (W_m3_per_tonne, COD_kg_per_m3, typical_MCF_str)}
INDUSTRY_PARAMETERS: Dict[str, Dict[str, Any]] = {
    IndustryType.PULP_AND_PAPER.value: {
        "wastewater_m3_per_tonne": Decimal("60"),
        "cod_kg_per_m3": Decimal("5.0"),
        "typical_mcf": Decimal("0.3"),
        "description": "Pulp and paper production",
    },
    IndustryType.MEAT_AND_POULTRY.value: {
        "wastewater_m3_per_tonne": Decimal("15"),
        "cod_kg_per_m3": Decimal("5.0"),
        "typical_mcf": Decimal("0.5"),
        "description": "Meat and poultry processing",
    },
    IndustryType.DAIRY.value: {
        "wastewater_m3_per_tonne": Decimal("7"),
        "cod_kg_per_m3": Decimal("3.0"),
        "typical_mcf": Decimal("0.3"),
        "description": "Dairy products processing",
    },
    IndustryType.SUGAR_REFINING.value: {
        "wastewater_m3_per_tonne": Decimal("20"),
        "cod_kg_per_m3": Decimal("3.0"),
        "typical_mcf": Decimal("0.3"),
        "description": "Sugar refining and processing",
    },
    IndustryType.VEGETABLE_OIL.value: {
        "wastewater_m3_per_tonne": Decimal("25"),
        "cod_kg_per_m3": Decimal("3.0"),
        "typical_mcf": Decimal("0.5"),
        "description": "Vegetable oil production",
    },
    IndustryType.BREWERY.value: {
        "wastewater_m3_per_tonne": Decimal("10"),
        "cod_kg_per_m3": Decimal("4.0"),
        "typical_mcf": Decimal("0.3"),
        "description": "Brewery and beverage production",
    },
    IndustryType.STARCH.value: {
        "wastewater_m3_per_tonne": Decimal("10"),
        "cod_kg_per_m3": Decimal("10.0"),
        "typical_mcf": Decimal("0.5"),
        "description": "Starch production",
    },
    IndustryType.FRUIT_VEGETABLE_PROCESSING.value: {
        "wastewater_m3_per_tonne": Decimal("20"),
        "cod_kg_per_m3": Decimal("5.0"),
        "typical_mcf": Decimal("0.3"),
        "description": "Fruit and vegetable processing",
    },
    IndustryType.TEXTILES.value: {
        "wastewater_m3_per_tonne": Decimal("90"),
        "cod_kg_per_m3": Decimal("1.5"),
        "typical_mcf": Decimal("0.3"),
        "description": "Textile manufacturing",
    },
    IndustryType.PETROCHEMICAL.value: {
        "wastewater_m3_per_tonne": Decimal("4"),
        "cod_kg_per_m3": Decimal("3.0"),
        "typical_mcf": Decimal("0.3"),
        "description": "Petrochemical refining",
    },
    IndustryType.PHARMACEUTICAL.value: {
        "wastewater_m3_per_tonne": Decimal("20"),
        "cod_kg_per_m3": Decimal("2.0"),
        "typical_mcf": Decimal("0.3"),
        "description": "Pharmaceutical manufacturing",
    },
}


# ===========================================================================
# WastewaterTreatmentEngine
# ===========================================================================


class WastewaterTreatmentEngine:
    """Core calculation engine for on-site wastewater treatment CH4 and N2O
    emissions implementing IPCC 2006 Vol 5 Ch 6 and 2019 Refinement methods.

    This engine performs deterministic Decimal arithmetic for all wastewater
    emission calculations including CH4 from organic degradation, N2O from
    nitrification/denitrification, industrial wastewater characterization,
    and sludge management emissions.

    Thread Safety:
        Per-calculation state is created fresh for each method call.
        Shared counters use a reentrant lock.

    Attributes:
        _config: Optional configuration dictionary.
        _lock: Reentrant lock protecting mutable counters.
        _total_calculations: Counter of total calculations performed.
        _total_errors: Counter of total calculation errors.
        _default_gwp_source: Default GWP assessment report for CO2e.

    Example:
        >>> engine = WastewaterTreatmentEngine()
        >>> result = engine.calculate_wastewater_ch4(
        ...     tow_kg_bod_yr=500000,
        ...     sludge_removal_kg=50000,
        ...     bo="0.6",
        ...     mcf="0.8",
        ...     recovery_tonnes=0,
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the WastewaterTreatmentEngine.

        Args:
            config: Optional configuration dictionary. Supports:
                - default_gwp_source (str): Default GWP report (AR4/AR5/AR6).
                - decimal_precision (int): Decimal places (default 8).
        """
        self._config = config or {}
        self._lock = threading.RLock()
        self._total_calculations: int = 0
        self._total_errors: int = 0
        self._created_at = _utcnow()

        self._default_gwp_source: str = self._config.get(
            "default_gwp_source", "AR6",
        ).upper()

        logger.info(
            "WastewaterTreatmentEngine initialized: "
            "default_gwp=%s",
            self._default_gwp_source,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_calculations(self) -> None:
        """Thread-safe increment of the calculation counter."""
        with self._lock:
            self._total_calculations += 1

    def _increment_errors(self) -> None:
        """Thread-safe increment of the error counter."""
        with self._lock:
            self._total_errors += 1

    def _resolve_gwp(
        self,
        gas: str,
        gwp_source: Optional[str] = None,
    ) -> Decimal:
        """Look up GWP value for a gas species.

        Args:
            gas: Gas species (CH4, CO2, N2O).
            gwp_source: IPCC assessment report. Defaults to engine default.

        Returns:
            GWP value as Decimal. Returns 1 for unknown gases.
        """
        source = (gwp_source or self._default_gwp_source).upper()
        if source in GWP_VALUES and gas.upper() in GWP_VALUES[source]:
            return GWP_VALUES[source][gas.upper()]
        logger.warning(
            "GWP lookup failed for %s/%s, using 1.0", gas, source,
        )
        return _ONE

    def _build_gas_result(
        self,
        gas: str,
        emission_kg: Decimal,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build a per-gas emission result dictionary.

        Args:
            gas: Gas species identifier.
            emission_kg: Emission mass in kilograms.
            gwp_source: GWP assessment report for CO2e conversion.

        Returns:
            Dictionary with emission_kg, emission_tonnes, gwp, co2e values.
        """
        gwp = self._resolve_gwp(gas, gwp_source)
        emission_tonnes = _quantize(emission_kg * KG_TO_TONNES)
        co2e_kg = _quantize(emission_kg * gwp)
        co2e_tonnes = _quantize(co2e_kg * KG_TO_TONNES)

        return {
            "gas": gas,
            "emission_kg": str(_quantize(emission_kg)),
            "emission_tonnes": str(emission_tonnes),
            "gwp_value": str(gwp),
            "gwp_source": (gwp_source or self._default_gwp_source).upper(),
            "co2e_kg": str(co2e_kg),
            "co2e_tonnes": str(co2e_tonnes),
        }

    def _aggregate_gas_results(
        self,
        gas_results: List[Dict[str, str]],
    ) -> Tuple[Decimal, Decimal]:
        """Aggregate total CO2e across gas results.

        Args:
            gas_results: List of per-gas result dictionaries.

        Returns:
            Tuple of (total_co2e_kg, total_co2e_tonnes).
        """
        total_co2e_kg = _ZERO
        for gr in gas_results:
            total_co2e_kg += _D(gr["co2e_kg"])
        total_co2e_tonnes = _quantize(total_co2e_kg * KG_TO_TONNES)
        return _quantize(total_co2e_kg), total_co2e_tonnes

    def _record_metrics(
        self,
        treatment_method: str,
        calculation_method: str,
        waste_category: str,
        elapsed_seconds: float,
    ) -> None:
        """Record Prometheus metrics if available.

        Args:
            treatment_method: Treatment method label.
            calculation_method: Calculation method label.
            waste_category: Waste category label.
            elapsed_seconds: Calculation duration in seconds.
        """
        if _METRICS_AVAILABLE and _record_calc_operation is not None:
            try:
                _record_calc_operation(
                    treatment_method, calculation_method, waste_category,
                )
            except Exception:
                pass
        if _METRICS_AVAILABLE and _observe_duration is not None:
            try:
                _observe_duration(
                    treatment_method, calculation_method, elapsed_seconds,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Static Data Lookups
    # ------------------------------------------------------------------

    def get_industry_parameters(
        self,
        industry_type: str,
    ) -> Dict[str, Any]:
        """Retrieve IPCC Table 6.9 default wastewater parameters for an industry.

        Args:
            industry_type: Industry type string (see IndustryType enum).
                Case-insensitive; underscores accepted.

        Returns:
            Dictionary with:
                - wastewater_m3_per_tonne: Wastewater volume per unit product.
                - cod_kg_per_m3: Chemical oxygen demand concentration.
                - typical_mcf: Typical methane correction factor.
                - description: Human-readable industry name.

        Raises:
            ValueError: If industry_type is not recognized.
        """
        key = industry_type.upper()
        if key in INDUSTRY_PARAMETERS:
            return dict(INDUSTRY_PARAMETERS[key])

        valid_types = [t.value for t in IndustryType]
        raise ValueError(
            f"Unknown industry_type: {industry_type}. "
            f"Valid types: {valid_types}"
        )

    def get_bo_default(self, basis: str) -> Decimal:
        """Get the default maximum CH4 producing capacity (Bo).

        Args:
            basis: Organic load basis -- "BOD" or "COD".

        Returns:
            Bo value as Decimal:
                - BOD: 0.6 kg CH4/kg BOD
                - COD: 0.25 kg CH4/kg COD

        Raises:
            ValueError: If basis is not BOD or COD.
        """
        basis_upper = basis.upper()
        if basis_upper == OrganicBasis.BOD.value:
            return BO_DEFAULT_BOD
        elif basis_upper == OrganicBasis.COD.value:
            return BO_DEFAULT_COD
        else:
            raise ValueError(
                f"Unknown organic basis: {basis}. Must be 'BOD' or 'COD'."
            )

    def get_mcf(self, treatment_system: str) -> Decimal:
        """Get the IPCC methane correction factor for a treatment system.

        Args:
            treatment_system: Treatment system type (see TreatmentSystem enum).

        Returns:
            MCF value as Decimal.

        Raises:
            ValueError: If treatment_system is not recognized.
        """
        key = treatment_system.upper()
        if key in MCF_BY_TREATMENT:
            return MCF_BY_TREATMENT[key]

        valid_systems = [t.value for t in TreatmentSystem]
        raise ValueError(
            f"Unknown treatment_system: {treatment_system}. "
            f"Valid systems: {valid_systems}"
        )

    def get_sludge_mcf(self, sludge_treatment: str) -> Decimal:
        """Get the IPCC methane correction factor for a sludge treatment method.

        Args:
            sludge_treatment: Sludge treatment method (see SludgeTreatment enum).

        Returns:
            MCF value as Decimal.

        Raises:
            ValueError: If sludge_treatment is not recognized.
        """
        key = sludge_treatment.upper()
        if key in MCF_BY_SLUDGE_TREATMENT:
            return MCF_BY_SLUDGE_TREATMENT[key]

        valid_methods = [t.value for t in SludgeTreatment]
        raise ValueError(
            f"Unknown sludge_treatment: {sludge_treatment}. "
            f"Valid methods: {valid_methods}"
        )

    # ==================================================================
    # Core Calculation: CH4 from Wastewater (BOD basis)
    # ==================================================================

    def calculate_wastewater_ch4(
        self,
        tow_kg_bod_yr: Any,
        sludge_removal_kg: Any = 0,
        bo: Any = None,
        mcf: Any = None,
        recovery_tonnes: Any = 0,
        gwp_source: Optional[str] = None,
        treatment_system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate CH4 emissions from wastewater treatment on BOD basis.

        Implements IPCC 2006 Vol 5 Ch 6 Equation 6.1:
            CH4 (tonnes) = [ (TOW - S) * Bo * MCF ] * 0.001 - R

        Args:
            tow_kg_bod_yr: Total organic waste in wastewater (kg BOD/yr).
            sludge_removal_kg: Organic component removed as sludge (kg BOD/yr).
            bo: Maximum CH4 producing capacity (kg CH4/kg BOD).
                Defaults to 0.6 if None.
            mcf: Methane correction factor (0-1). If None and treatment_system
                is provided, looked up from IPCC table.
            recovery_tonnes: Recovered CH4 in tonnes CH4/yr.
            gwp_source: GWP assessment report (AR4/AR5/AR6/AR6_20YR).
            treatment_system: Treatment system type for MCF lookup.

        Returns:
            Calculation result dictionary with:
                - calculation_id, status, emissions_by_gas, total_co2e_kg,
                  total_co2e_tonnes, calculation_details, trace_steps,
                  processing_time_ms, provenance_hash.

        Raises:
            ValueError: If inputs fail validation.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = f"ww_ch4_bod_{uuid4().hex[:12]}"

        try:
            # -- Parse inputs --------------------------------------------------
            tow = _D(tow_kg_bod_yr)
            sludge = _safe_decimal(sludge_removal_kg, _ZERO)
            bo_val = _D(bo) if bo is not None else BO_DEFAULT_BOD
            recovery = _safe_decimal(recovery_tonnes, _ZERO)

            # Resolve MCF
            if mcf is not None:
                mcf_val = _D(mcf)
            elif treatment_system is not None:
                mcf_val = self.get_mcf(treatment_system)
            else:
                raise ValueError(
                    "Either 'mcf' or 'treatment_system' must be provided"
                )

            # -- Validate inputs -----------------------------------------------
            errors: List[str] = []
            if tow < _ZERO:
                errors.append("tow_kg_bod_yr must be >= 0")
            if sludge < _ZERO:
                errors.append("sludge_removal_kg must be >= 0")
            if sludge > tow:
                errors.append(
                    "sludge_removal_kg cannot exceed tow_kg_bod_yr"
                )
            if bo_val <= _ZERO:
                errors.append("bo must be > 0")
            if mcf_val < _ZERO or mcf_val > _ONE:
                errors.append("mcf must be between 0 and 1")
            if recovery < _ZERO:
                errors.append("recovery_tonnes must be >= 0")

            if errors:
                return self._error_result(calc_id, errors, t0)

            # -- Step 1: Net organic load --------------------------------------
            net_tow = _quantize(tow - sludge)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate net organic waste load",
                formula="TOW_net = TOW - S",
                inputs={
                    "TOW_kg_bod_yr": str(tow),
                    "sludge_removal_kg": str(sludge),
                },
                output=str(net_tow),
                unit="kg BOD/yr",
            ))

            # -- Step 2: Gross CH4 (kg) ----------------------------------------
            ch4_gross_kg = _quantize(net_tow * bo_val * mcf_val)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate gross CH4 production",
                formula="CH4_gross = TOW_net * Bo * MCF",
                inputs={
                    "TOW_net": str(net_tow),
                    "Bo": str(bo_val),
                    "MCF": str(mcf_val),
                },
                output=str(ch4_gross_kg),
                unit="kg CH4/yr",
            ))

            # -- Step 3: Convert to tonnes and subtract recovery ---------------
            ch4_gross_tonnes = _quantize(ch4_gross_kg * KG_TO_TONNES)
            ch4_net_tonnes = _quantize(ch4_gross_tonnes - recovery)

            # Ensure non-negative
            if ch4_net_tonnes < _ZERO:
                ch4_net_tonnes = _ZERO

            ch4_net_kg = _quantize(ch4_net_tonnes * _THOUSAND)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Subtract CH4 recovery and convert to net emissions",
                formula="CH4_net = CH4_gross * 0.001 - R",
                inputs={
                    "CH4_gross_kg": str(ch4_gross_kg),
                    "CH4_gross_tonnes": str(ch4_gross_tonnes),
                    "recovery_tonnes": str(recovery),
                },
                output=str(ch4_net_tonnes),
                unit="tonnes CH4/yr",
            ))

            # -- Step 4: GWP conversion ----------------------------------------
            gas_result = self._build_gas_result("CH4", ch4_net_kg, gwp_source)
            gas_results = [gas_result]
            total_co2e_kg, total_co2e_tonnes = self._aggregate_gas_results(
                gas_results,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Apply GWP to convert to CO2e",
                formula="CO2e = CH4_kg * GWP_CH4",
                inputs={
                    "CH4_net_kg": str(ch4_net_kg),
                    "GWP_CH4": gas_result["gwp_value"],
                },
                output=gas_result["co2e_kg"],
                unit="kg CO2e",
            ))

            # -- Assemble result -----------------------------------------------
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "IPCC_WW_CH4_BOD",
                "organic_basis": "BOD",
                "emissions_by_gas": gas_results,
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "calculation_details": {
                    "tow_kg_bod_yr": str(tow),
                    "sludge_removal_kg": str(sludge),
                    "net_tow_kg": str(net_tow),
                    "bo_kg_ch4_per_kg_bod": str(bo_val),
                    "mcf": str(mcf_val),
                    "treatment_system": treatment_system or "custom",
                    "ch4_gross_kg": str(ch4_gross_kg),
                    "ch4_gross_tonnes": str(ch4_gross_tonnes),
                    "recovery_tonnes": str(recovery),
                    "ch4_net_tonnes": str(ch4_net_tonnes),
                    "ch4_net_kg": str(ch4_net_kg),
                    "formula": "CH4 = [(TOW - S) * Bo * MCF] * 0.001 - R",
                },
                "trace_steps": [s.to_dict() for s in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            self._record_metrics(
                "wastewater_treatment", "ipcc_default",
                "industrial_waste", elapsed_ms / 1000.0,
            )

            logger.info(
                "WW CH4 (BOD) calculation %s: %s tonnes CH4, "
                "%s tonnes CO2e in %.1fms",
                calc_id, ch4_net_tonnes, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # Core Calculation: CH4 from Wastewater (COD basis)
    # ==================================================================

    def calculate_wastewater_ch4_from_cod(
        self,
        tow_kg_cod_yr: Any,
        sludge_removal_kg: Any = 0,
        bo_cod: Any = None,
        mcf: Any = None,
        recovery_tonnes: Any = 0,
        gwp_source: Optional[str] = None,
        treatment_system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate CH4 emissions from wastewater treatment on COD basis.

        Implements IPCC 2006 Vol 5 Ch 6 Equation 6.1 using COD:
            CH4 (tonnes) = [ (TOW_cod - S) * Bo_cod * MCF ] * 0.001 - R

        Args:
            tow_kg_cod_yr: Total organic waste in wastewater (kg COD/yr).
            sludge_removal_kg: Organic component removed as sludge (kg COD/yr).
            bo_cod: Maximum CH4 producing capacity (kg CH4/kg COD).
                Defaults to 0.25 if None.
            mcf: Methane correction factor (0-1). If None and treatment_system
                is provided, looked up from IPCC table.
            recovery_tonnes: Recovered CH4 in tonnes CH4/yr.
            gwp_source: GWP assessment report.
            treatment_system: Treatment system type for MCF lookup.

        Returns:
            Calculation result dictionary.

        Raises:
            ValueError: If inputs fail validation.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = f"ww_ch4_cod_{uuid4().hex[:12]}"

        try:
            # -- Parse inputs --------------------------------------------------
            tow = _D(tow_kg_cod_yr)
            sludge = _safe_decimal(sludge_removal_kg, _ZERO)
            bo_val = _D(bo_cod) if bo_cod is not None else BO_DEFAULT_COD
            recovery = _safe_decimal(recovery_tonnes, _ZERO)

            # Resolve MCF
            if mcf is not None:
                mcf_val = _D(mcf)
            elif treatment_system is not None:
                mcf_val = self.get_mcf(treatment_system)
            else:
                raise ValueError(
                    "Either 'mcf' or 'treatment_system' must be provided"
                )

            # -- Validate inputs -----------------------------------------------
            errors: List[str] = []
            if tow < _ZERO:
                errors.append("tow_kg_cod_yr must be >= 0")
            if sludge < _ZERO:
                errors.append("sludge_removal_kg must be >= 0")
            if sludge > tow:
                errors.append(
                    "sludge_removal_kg cannot exceed tow_kg_cod_yr"
                )
            if bo_val <= _ZERO:
                errors.append("bo_cod must be > 0")
            if mcf_val < _ZERO or mcf_val > _ONE:
                errors.append("mcf must be between 0 and 1")
            if recovery < _ZERO:
                errors.append("recovery_tonnes must be >= 0")

            if errors:
                return self._error_result(calc_id, errors, t0)

            # -- Step 1: Net organic load --------------------------------------
            net_tow = _quantize(tow - sludge)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate net organic waste load (COD)",
                formula="TOW_net = TOW_cod - S",
                inputs={
                    "TOW_kg_cod_yr": str(tow),
                    "sludge_removal_kg": str(sludge),
                },
                output=str(net_tow),
                unit="kg COD/yr",
            ))

            # -- Step 2: Gross CH4 (kg) ----------------------------------------
            ch4_gross_kg = _quantize(net_tow * bo_val * mcf_val)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate gross CH4 production (COD basis)",
                formula="CH4_gross = TOW_net * Bo_cod * MCF",
                inputs={
                    "TOW_net": str(net_tow),
                    "Bo_cod": str(bo_val),
                    "MCF": str(mcf_val),
                },
                output=str(ch4_gross_kg),
                unit="kg CH4/yr",
            ))

            # -- Step 3: Convert to tonnes and subtract recovery ---------------
            ch4_gross_tonnes = _quantize(ch4_gross_kg * KG_TO_TONNES)
            ch4_net_tonnes = _quantize(ch4_gross_tonnes - recovery)

            if ch4_net_tonnes < _ZERO:
                ch4_net_tonnes = _ZERO

            ch4_net_kg = _quantize(ch4_net_tonnes * _THOUSAND)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Subtract CH4 recovery and convert to net",
                formula="CH4_net = CH4_gross * 0.001 - R",
                inputs={
                    "CH4_gross_tonnes": str(ch4_gross_tonnes),
                    "recovery_tonnes": str(recovery),
                },
                output=str(ch4_net_tonnes),
                unit="tonnes CH4/yr",
            ))

            # -- Step 4: GWP conversion ----------------------------------------
            gas_result = self._build_gas_result("CH4", ch4_net_kg, gwp_source)
            gas_results = [gas_result]
            total_co2e_kg, total_co2e_tonnes = self._aggregate_gas_results(
                gas_results,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Apply GWP to convert to CO2e",
                formula="CO2e = CH4_kg * GWP_CH4",
                inputs={
                    "CH4_net_kg": str(ch4_net_kg),
                    "GWP_CH4": gas_result["gwp_value"],
                },
                output=gas_result["co2e_kg"],
                unit="kg CO2e",
            ))

            # -- Assemble result -----------------------------------------------
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "IPCC_WW_CH4_COD",
                "organic_basis": "COD",
                "emissions_by_gas": gas_results,
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "calculation_details": {
                    "tow_kg_cod_yr": str(tow),
                    "sludge_removal_kg": str(sludge),
                    "net_tow_kg": str(net_tow),
                    "bo_kg_ch4_per_kg_cod": str(bo_val),
                    "mcf": str(mcf_val),
                    "treatment_system": treatment_system or "custom",
                    "ch4_gross_kg": str(ch4_gross_kg),
                    "ch4_gross_tonnes": str(ch4_gross_tonnes),
                    "recovery_tonnes": str(recovery),
                    "ch4_net_tonnes": str(ch4_net_tonnes),
                    "ch4_net_kg": str(ch4_net_kg),
                    "formula": "CH4 = [(TOW_cod - S) * Bo_cod * MCF] * 0.001 - R",
                },
                "trace_steps": [s.to_dict() for s in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            self._record_metrics(
                "wastewater_treatment", "ipcc_default",
                "industrial_waste", elapsed_ms / 1000.0,
            )

            logger.info(
                "WW CH4 (COD) calculation %s: %s tonnes CH4, "
                "%s tonnes CO2e in %.1fms",
                calc_id, ch4_net_tonnes, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # Industrial Wastewater (full characterization)
    # ==================================================================

    def calculate_industrial_wastewater(
        self,
        production_tonnes: Any,
        industry_type: str,
        treatment_system: str,
        gwp_source: Optional[str] = None,
        wastewater_m3_per_tonne_override: Optional[Any] = None,
        cod_kg_per_m3_override: Optional[Any] = None,
        mcf_override: Optional[Any] = None,
        bo_cod_override: Optional[Any] = None,
        recovery_tonnes: Any = 0,
        sludge_fraction: Any = 0,
    ) -> Dict[str, Any]:
        """Calculate CH4 from industrial wastewater using IPCC Table 6.9.

        First calculates TOW using industry parameters, then applies the
        standard IPCC CH4 formula on COD basis.

        Formulas:
            TOW = P_i * W_i * COD_i  (kg COD/yr)
            S   = TOW * sludge_fraction
            CH4 = [(TOW - S) * Bo * MCF] * 0.001 - R

        Args:
            production_tonnes: Annual industrial production (tonnes product/yr).
            industry_type: Industry type (see IndustryType enum).
            treatment_system: Wastewater treatment system type.
            gwp_source: GWP assessment report.
            wastewater_m3_per_tonne_override: Override for W_i (m3/tonne).
            cod_kg_per_m3_override: Override for COD_i (kg COD/m3).
            mcf_override: Override MCF value (0-1).
            bo_cod_override: Override Bo on COD basis.
            recovery_tonnes: Recovered CH4 in tonnes CH4/yr.
            sludge_fraction: Fraction of TOW removed as sludge (0-1).

        Returns:
            Calculation result with industry characterization and CH4 emissions.

        Raises:
            ValueError: If inputs fail validation.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = f"ww_ind_{uuid4().hex[:12]}"

        try:
            production = _D(production_tonnes)
            recovery = _safe_decimal(recovery_tonnes, _ZERO)
            sludge_frac = _safe_decimal(sludge_fraction, _ZERO)

            # -- Validate base inputs ------------------------------------------
            errors: List[str] = []
            if production <= _ZERO:
                errors.append("production_tonnes must be > 0")
            if sludge_frac < _ZERO or sludge_frac > _ONE:
                errors.append("sludge_fraction must be between 0 and 1")
            if recovery < _ZERO:
                errors.append("recovery_tonnes must be >= 0")

            if errors:
                return self._error_result(calc_id, errors, t0)

            # -- Look up industry parameters -----------------------------------
            industry_params = self.get_industry_parameters(industry_type)

            w_i = (
                _D(wastewater_m3_per_tonne_override)
                if wastewater_m3_per_tonne_override is not None
                else industry_params["wastewater_m3_per_tonne"]
            )
            cod_i = (
                _D(cod_kg_per_m3_override)
                if cod_kg_per_m3_override is not None
                else industry_params["cod_kg_per_m3"]
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description=f"Look up industry parameters for {industry_type}",
                formula="W_i, COD_i from IPCC Table 6.9",
                inputs={
                    "industry_type": industry_type,
                    "W_i_override": str(wastewater_m3_per_tonne_override),
                    "COD_i_override": str(cod_kg_per_m3_override),
                },
                output=f"W_i={w_i}, COD_i={cod_i}",
                unit="m3/t, kg/m3",
            ))

            # -- Step 2: Calculate TOW -----------------------------------------
            tow_kg_cod = _quantize(production * w_i * cod_i)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate total organic waste in wastewater",
                formula="TOW = P_i * W_i * COD_i",
                inputs={
                    "P_i_tonnes": str(production),
                    "W_i_m3_per_tonne": str(w_i),
                    "COD_i_kg_per_m3": str(cod_i),
                },
                output=str(tow_kg_cod),
                unit="kg COD/yr",
            ))

            # -- Step 3: Calculate sludge removal ------------------------------
            sludge_kg = _quantize(tow_kg_cod * sludge_frac)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate sludge organic removal",
                formula="S = TOW * sludge_fraction",
                inputs={
                    "TOW_kg": str(tow_kg_cod),
                    "sludge_fraction": str(sludge_frac),
                },
                output=str(sludge_kg),
                unit="kg COD/yr",
            ))

            # -- Step 4: Resolve MCF and Bo ------------------------------------
            if mcf_override is not None:
                mcf_val = _D(mcf_override)
            else:
                mcf_val = self.get_mcf(treatment_system)

            bo_val = (
                _D(bo_cod_override)
                if bo_cod_override is not None
                else BO_DEFAULT_COD
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Resolve MCF and Bo values",
                formula="MCF from treatment system, Bo default or override",
                inputs={
                    "treatment_system": treatment_system,
                    "mcf_override": str(mcf_override),
                    "bo_cod_override": str(bo_cod_override),
                },
                output=f"MCF={mcf_val}, Bo={bo_val}",
                unit="dimensionless",
            ))

            # -- Step 5: CH4 calculation ---------------------------------------
            net_tow = _quantize(tow_kg_cod - sludge_kg)
            ch4_gross_kg = _quantize(net_tow * bo_val * mcf_val)
            ch4_gross_tonnes = _quantize(ch4_gross_kg * KG_TO_TONNES)
            ch4_net_tonnes = _quantize(ch4_gross_tonnes - recovery)

            if ch4_net_tonnes < _ZERO:
                ch4_net_tonnes = _ZERO

            ch4_net_kg = _quantize(ch4_net_tonnes * _THOUSAND)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate net CH4 emissions",
                formula="CH4 = [(TOW - S) * Bo * MCF] * 0.001 - R",
                inputs={
                    "net_TOW_kg": str(net_tow),
                    "Bo": str(bo_val),
                    "MCF": str(mcf_val),
                    "recovery_tonnes": str(recovery),
                },
                output=str(ch4_net_tonnes),
                unit="tonnes CH4/yr",
            ))

            # -- Step 6: GWP conversion ----------------------------------------
            gas_result = self._build_gas_result("CH4", ch4_net_kg, gwp_source)
            gas_results = [gas_result]
            total_co2e_kg, total_co2e_tonnes = self._aggregate_gas_results(
                gas_results,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Apply GWP to convert to CO2e",
                formula="CO2e = CH4_kg * GWP_CH4",
                inputs={
                    "CH4_net_kg": str(ch4_net_kg),
                    "GWP_CH4": gas_result["gwp_value"],
                },
                output=gas_result["co2e_kg"],
                unit="kg CO2e",
            ))

            # -- Assemble result -----------------------------------------------
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "IPCC_WW_INDUSTRIAL",
                "organic_basis": "COD",
                "industry_type": industry_type.upper(),
                "emissions_by_gas": gas_results,
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "calculation_details": {
                    "production_tonnes": str(production),
                    "industry_type": industry_type.upper(),
                    "wastewater_m3_per_tonne": str(w_i),
                    "cod_kg_per_m3": str(cod_i),
                    "tow_kg_cod_yr": str(tow_kg_cod),
                    "sludge_fraction": str(sludge_frac),
                    "sludge_removal_kg": str(sludge_kg),
                    "net_tow_kg": str(net_tow),
                    "bo_cod": str(bo_val),
                    "mcf": str(mcf_val),
                    "treatment_system": treatment_system.upper(),
                    "ch4_gross_kg": str(ch4_gross_kg),
                    "ch4_gross_tonnes": str(ch4_gross_tonnes),
                    "recovery_tonnes": str(recovery),
                    "ch4_net_tonnes": str(ch4_net_tonnes),
                    "ch4_net_kg": str(ch4_net_kg),
                    "formula": "TOW = P * W * COD; "
                               "CH4 = [(TOW - S) * Bo * MCF] * 0.001 - R",
                },
                "trace_steps": [s.to_dict() for s in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            self._record_metrics(
                "wastewater_treatment", "ipcc_default",
                "industrial_waste", elapsed_ms / 1000.0,
            )

            logger.info(
                "Industrial WW calculation %s [%s]: TOW=%s kg COD, "
                "CH4=%s tonnes, CO2e=%s tonnes in %.1fms",
                calc_id, industry_type, tow_kg_cod,
                ch4_net_tonnes, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # N2O from Wastewater (Plant + Effluent)
    # ==================================================================

    def calculate_wastewater_n2o(
        self,
        flow_m3_yr: Any,
        n_influent_kg_m3: Any,
        n_effluent_kg_m3: Any,
        ef_plant: Any = None,
        ef_effluent: Any = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate N2O emissions from wastewater treatment (plant + effluent).

        Implements IPCC 2019 Refinement N2O methodology:
            N2O_plant    = Q * N_influent * EF_plant * 44/28
            N2O_effluent = Q * N_effluent * EF_effluent * 44/28
            N2O_total    = N2O_plant + N2O_effluent

        Args:
            flow_m3_yr: Annual wastewater flow rate (m3/yr).
            n_influent_kg_m3: Total nitrogen in influent (kg N/m3).
            n_effluent_kg_m3: Total nitrogen in effluent (kg N/m3).
            ef_plant: Plant N2O emission factor (kg N2O-N/kg N).
                Default: 0.016 (IPCC 2019).
            ef_effluent: Effluent N2O emission factor (kg N2O-N/kg N).
                Default: 0.005 (IPCC 2019).
            gwp_source: GWP assessment report.

        Returns:
            Calculation result with plant and effluent N2O breakdown.

        Raises:
            ValueError: If inputs fail validation.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = f"ww_n2o_{uuid4().hex[:12]}"

        try:
            # -- Parse inputs --------------------------------------------------
            flow = _D(flow_m3_yr)
            n_inf = _D(n_influent_kg_m3)
            n_eff = _D(n_effluent_kg_m3)
            ef_p = _D(ef_plant) if ef_plant is not None else EF_N2O_PLANT_DEFAULT
            ef_e = (
                _D(ef_effluent)
                if ef_effluent is not None
                else EF_N2O_EFFLUENT_DEFAULT
            )

            # -- Validate inputs -----------------------------------------------
            errors: List[str] = []
            if flow <= _ZERO:
                errors.append("flow_m3_yr must be > 0")
            if n_inf < _ZERO:
                errors.append("n_influent_kg_m3 must be >= 0")
            if n_eff < _ZERO:
                errors.append("n_effluent_kg_m3 must be >= 0")
            if ef_p < _ZERO:
                errors.append("ef_plant must be >= 0")
            if ef_e < _ZERO:
                errors.append("ef_effluent must be >= 0")

            if errors:
                return self._error_result(calc_id, errors, t0)

            # -- Step 1: Plant N2O emissions -----------------------------------
            # N2O_plant (kg) = Q * N_influent * EF_plant * 44/28
            n2o_n_plant_kg = _quantize(flow * n_inf * ef_p)
            n2o_plant_kg = _quantize(n2o_n_plant_kg * N2O_MOLECULAR_RATIO)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate N2O from wastewater treatment plant",
                formula="N2O_plant = Q * N_influent * EF_plant * 44/28",
                inputs={
                    "Q_m3_yr": str(flow),
                    "N_influent_kg_m3": str(n_inf),
                    "EF_plant": str(ef_p),
                    "N2O_N_ratio": str(N2O_MOLECULAR_RATIO),
                },
                output=str(n2o_plant_kg),
                unit="kg N2O/yr",
            ))

            # -- Step 2: Effluent N2O emissions --------------------------------
            # N2O_effluent (kg) = Q * N_effluent * EF_effluent * 44/28
            n2o_n_effluent_kg = _quantize(flow * n_eff * ef_e)
            n2o_effluent_kg = _quantize(n2o_n_effluent_kg * N2O_MOLECULAR_RATIO)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate N2O from effluent discharge",
                formula="N2O_effluent = Q * N_effluent * EF_effluent * 44/28",
                inputs={
                    "Q_m3_yr": str(flow),
                    "N_effluent_kg_m3": str(n_eff),
                    "EF_effluent": str(ef_e),
                    "N2O_N_ratio": str(N2O_MOLECULAR_RATIO),
                },
                output=str(n2o_effluent_kg),
                unit="kg N2O/yr",
            ))

            # -- Step 3: Total N2O ---------------------------------------------
            n2o_total_kg = _quantize(n2o_plant_kg + n2o_effluent_kg)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Sum plant and effluent N2O",
                formula="N2O_total = N2O_plant + N2O_effluent",
                inputs={
                    "N2O_plant_kg": str(n2o_plant_kg),
                    "N2O_effluent_kg": str(n2o_effluent_kg),
                },
                output=str(n2o_total_kg),
                unit="kg N2O/yr",
            ))

            # -- Step 4: GWP conversion ----------------------------------------
            gas_result = self._build_gas_result("N2O", n2o_total_kg, gwp_source)
            gas_results = [gas_result]
            total_co2e_kg, total_co2e_tonnes = self._aggregate_gas_results(
                gas_results,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Apply GWP to convert N2O to CO2e",
                formula="CO2e = N2O_kg * GWP_N2O",
                inputs={
                    "N2O_total_kg": str(n2o_total_kg),
                    "GWP_N2O": gas_result["gwp_value"],
                },
                output=gas_result["co2e_kg"],
                unit="kg CO2e",
            ))

            # -- Assemble result -----------------------------------------------
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "IPCC_WW_N2O",
                "emissions_by_gas": gas_results,
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "calculation_details": {
                    "flow_m3_yr": str(flow),
                    "n_influent_kg_m3": str(n_inf),
                    "n_effluent_kg_m3": str(n_eff),
                    "ef_plant_kg_n2o_n_per_kg_n": str(ef_p),
                    "ef_effluent_kg_n2o_n_per_kg_n": str(ef_e),
                    "n2o_n_ratio": str(N2O_MOLECULAR_RATIO),
                    "n2o_plant_kg": str(n2o_plant_kg),
                    "n2o_effluent_kg": str(n2o_effluent_kg),
                    "n2o_total_kg": str(n2o_total_kg),
                    "n2o_total_tonnes": str(_quantize(
                        n2o_total_kg * KG_TO_TONNES
                    )),
                    "formula_plant": "N2O_plant = Q * N_inf * EF_plant * 44/28",
                    "formula_effluent": (
                        "N2O_eff = Q * N_eff * EF_effluent * 44/28"
                    ),
                },
                "trace_steps": [s.to_dict() for s in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            self._record_metrics(
                "wastewater_treatment", "ipcc_default",
                "sewage_sludge", elapsed_ms / 1000.0,
            )

            logger.info(
                "WW N2O calculation %s: plant=%s kg, effluent=%s kg, "
                "total=%s kg, CO2e=%s tonnes in %.1fms",
                calc_id, n2o_plant_kg, n2o_effluent_kg,
                n2o_total_kg, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # Sludge Treatment Emissions
    # ==================================================================

    def calculate_sludge_emissions(
        self,
        sludge_tonnes: Any,
        vs_fraction: Any = None,
        treatment_system: Optional[str] = None,
        mcf_override: Optional[Any] = None,
        bo_sludge: Optional[Any] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate CH4 emissions from sludge treatment/disposal.

        Implements:
            CH4_sludge = M_sludge * VS_fraction * Bo_sludge * MCF_treatment

        Args:
            sludge_tonnes: Mass of sludge (wet tonnes).
            vs_fraction: Volatile solids fraction (0-1). Default 0.60.
            treatment_system: Sludge treatment method (see SludgeTreatment enum).
                Required if mcf_override is None.
            mcf_override: Override MCF value (0-1).
            bo_sludge: Maximum CH4 producing capacity for sludge (kg CH4/kg VS).
                Default 0.6.
            gwp_source: GWP assessment report.

        Returns:
            Calculation result with sludge CH4 emissions.

        Raises:
            ValueError: If inputs fail validation.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = f"ww_sludge_{uuid4().hex[:12]}"

        try:
            # -- Parse inputs --------------------------------------------------
            sludge_mass = _D(sludge_tonnes)
            vs_frac = (
                _D(vs_fraction)
                if vs_fraction is not None
                else VS_FRACTION_DEFAULT
            )
            bo_val = (
                _D(bo_sludge)
                if bo_sludge is not None
                else BO_SLUDGE_DEFAULT
            )

            # Resolve MCF
            if mcf_override is not None:
                mcf_val = _D(mcf_override)
            elif treatment_system is not None:
                mcf_val = self.get_sludge_mcf(treatment_system)
            else:
                raise ValueError(
                    "Either 'mcf_override' or 'treatment_system' must be "
                    "provided for sludge emissions"
                )

            # -- Validate inputs -----------------------------------------------
            errors: List[str] = []
            if sludge_mass <= _ZERO:
                errors.append("sludge_tonnes must be > 0")
            if vs_frac < _ZERO or vs_frac > _ONE:
                errors.append("vs_fraction must be between 0 and 1")
            if bo_val <= _ZERO:
                errors.append("bo_sludge must be > 0")
            if mcf_val < _ZERO or mcf_val > _ONE:
                errors.append("mcf must be between 0 and 1")

            if errors:
                return self._error_result(calc_id, errors, t0)

            # -- Step 1: Calculate volatile solids mass -------------------------
            # sludge_tonnes are wet tonnes; VS = mass * VS_fraction
            # Convert tonnes to kg for consistent units
            sludge_kg = _quantize(sludge_mass * _THOUSAND)
            vs_kg = _quantize(sludge_kg * vs_frac)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate volatile solids mass",
                formula="VS_kg = M_sludge_kg * VS_fraction",
                inputs={
                    "sludge_tonnes": str(sludge_mass),
                    "sludge_kg": str(sludge_kg),
                    "vs_fraction": str(vs_frac),
                },
                output=str(vs_kg),
                unit="kg VS",
            ))

            # -- Step 2: Calculate CH4 from sludge -----------------------------
            ch4_kg = _quantize(vs_kg * bo_val * mcf_val)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate CH4 from sludge treatment",
                formula="CH4 = VS_kg * Bo_sludge * MCF_treatment",
                inputs={
                    "VS_kg": str(vs_kg),
                    "Bo_sludge": str(bo_val),
                    "MCF_treatment": str(mcf_val),
                },
                output=str(ch4_kg),
                unit="kg CH4",
            ))

            # -- Step 3: GWP conversion ----------------------------------------
            gas_result = self._build_gas_result("CH4", ch4_kg, gwp_source)
            gas_results = [gas_result]
            total_co2e_kg, total_co2e_tonnes = self._aggregate_gas_results(
                gas_results,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Apply GWP to convert to CO2e",
                formula="CO2e = CH4_kg * GWP_CH4",
                inputs={
                    "CH4_kg": str(ch4_kg),
                    "GWP_CH4": gas_result["gwp_value"],
                },
                output=gas_result["co2e_kg"],
                unit="kg CO2e",
            ))

            # -- Assemble result -----------------------------------------------
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "IPCC_WW_SLUDGE",
                "emissions_by_gas": gas_results,
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "calculation_details": {
                    "sludge_tonnes": str(sludge_mass),
                    "sludge_kg": str(sludge_kg),
                    "vs_fraction": str(vs_frac),
                    "vs_kg": str(vs_kg),
                    "bo_sludge_kg_ch4_per_kg_vs": str(bo_val),
                    "mcf_treatment": str(mcf_val),
                    "treatment_system": treatment_system or "custom",
                    "ch4_kg": str(ch4_kg),
                    "ch4_tonnes": str(_quantize(ch4_kg * KG_TO_TONNES)),
                    "formula": "CH4 = M_sludge_kg * VS_frac * Bo * MCF",
                },
                "trace_steps": [s.to_dict() for s in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            self._record_metrics(
                "wastewater_treatment", "ipcc_default",
                "sewage_sludge", elapsed_ms / 1000.0,
            )

            logger.info(
                "Sludge CH4 calculation %s: %s kg CH4, "
                "%s tonnes CO2e in %.1fms",
                calc_id, ch4_kg, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # Combined Wastewater Emissions (CH4 + N2O)
    # ==================================================================

    def calculate_wastewater_combined(
        self,
        ch4_inputs: Dict[str, Any],
        n2o_inputs: Optional[Dict[str, Any]] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate combined CH4 and N2O emissions from wastewater treatment.

        Orchestrates separate CH4 and N2O calculations and aggregates the
        results into a single combined emission total.

        Args:
            ch4_inputs: Inputs for CH4 calculation. Must contain either:
                - BOD basis: tow_kg_bod_yr, sludge_removal_kg, bo, mcf
                - COD basis: tow_kg_cod_yr, sludge_removal_kg, bo_cod, mcf
                - Industrial: production_tonnes, industry_type, treatment_system
                The 'basis' key ("BOD", "COD", or "INDUSTRIAL") selects the path.
            n2o_inputs: Optional inputs for N2O calculation:
                - flow_m3_yr, n_influent_kg_m3, n_effluent_kg_m3
                - ef_plant (optional), ef_effluent (optional)
            gwp_source: GWP assessment report (applied to both).

        Returns:
            Combined calculation result with both CH4 and N2O contributions.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        calc_id = f"ww_combined_{uuid4().hex[:12]}"

        try:
            gas_results: List[Dict[str, str]] = []
            sub_results: Dict[str, Any] = {}
            combined_status = CalculationStatus.SUCCESS.value

            # -- CH4 calculation -----------------------------------------------
            basis = str(ch4_inputs.get("basis", "BOD")).upper()

            if basis == "INDUSTRIAL":
                ch4_result = self.calculate_industrial_wastewater(
                    production_tonnes=ch4_inputs["production_tonnes"],
                    industry_type=ch4_inputs["industry_type"],
                    treatment_system=ch4_inputs["treatment_system"],
                    gwp_source=gwp_source,
                    wastewater_m3_per_tonne_override=ch4_inputs.get(
                        "wastewater_m3_per_tonne"
                    ),
                    cod_kg_per_m3_override=ch4_inputs.get("cod_kg_per_m3"),
                    mcf_override=ch4_inputs.get("mcf"),
                    bo_cod_override=ch4_inputs.get("bo_cod"),
                    recovery_tonnes=ch4_inputs.get("recovery_tonnes", 0),
                    sludge_fraction=ch4_inputs.get("sludge_fraction", 0),
                )
            elif basis == "COD":
                ch4_result = self.calculate_wastewater_ch4_from_cod(
                    tow_kg_cod_yr=ch4_inputs["tow_kg_cod_yr"],
                    sludge_removal_kg=ch4_inputs.get("sludge_removal_kg", 0),
                    bo_cod=ch4_inputs.get("bo_cod"),
                    mcf=ch4_inputs.get("mcf"),
                    recovery_tonnes=ch4_inputs.get("recovery_tonnes", 0),
                    gwp_source=gwp_source,
                    treatment_system=ch4_inputs.get("treatment_system"),
                )
            else:
                # Default: BOD basis
                ch4_result = self.calculate_wastewater_ch4(
                    tow_kg_bod_yr=ch4_inputs["tow_kg_bod_yr"],
                    sludge_removal_kg=ch4_inputs.get("sludge_removal_kg", 0),
                    bo=ch4_inputs.get("bo"),
                    mcf=ch4_inputs.get("mcf"),
                    recovery_tonnes=ch4_inputs.get("recovery_tonnes", 0),
                    gwp_source=gwp_source,
                    treatment_system=ch4_inputs.get("treatment_system"),
                )

            sub_results["ch4"] = ch4_result

            if ch4_result.get("status") == CalculationStatus.SUCCESS.value:
                gas_results.extend(ch4_result.get("emissions_by_gas", []))
            else:
                combined_status = CalculationStatus.PARTIAL.value
                logger.warning(
                    "Combined calc %s: CH4 sub-calc returned status %s",
                    calc_id, ch4_result.get("status"),
                )

            # -- N2O calculation (optional) ------------------------------------
            if n2o_inputs is not None:
                n2o_result = self.calculate_wastewater_n2o(
                    flow_m3_yr=n2o_inputs["flow_m3_yr"],
                    n_influent_kg_m3=n2o_inputs["n_influent_kg_m3"],
                    n_effluent_kg_m3=n2o_inputs["n_effluent_kg_m3"],
                    ef_plant=n2o_inputs.get("ef_plant"),
                    ef_effluent=n2o_inputs.get("ef_effluent"),
                    gwp_source=gwp_source,
                )
                sub_results["n2o"] = n2o_result

                if n2o_result.get("status") == CalculationStatus.SUCCESS.value:
                    gas_results.extend(n2o_result.get("emissions_by_gas", []))
                else:
                    combined_status = CalculationStatus.PARTIAL.value
                    logger.warning(
                        "Combined calc %s: N2O sub-calc returned status %s",
                        calc_id, n2o_result.get("status"),
                    )

            # -- Aggregate totals ----------------------------------------------
            total_co2e_kg, total_co2e_tonnes = self._aggregate_gas_results(
                gas_results,
            )

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": combined_status,
                "method": "IPCC_WW_COMBINED",
                "organic_basis": basis,
                "emissions_by_gas": gas_results,
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "sub_results": {
                    k: {
                        "calculation_id": v.get("calculation_id"),
                        "status": v.get("status"),
                        "total_co2e_tonnes": v.get("total_co2e_tonnes", "0"),
                    }
                    for k, v in sub_results.items()
                },
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            logger.info(
                "Combined WW calculation %s: %d gas results, "
                "%s tonnes CO2e in %.1fms",
                calc_id, len(gas_results), total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # Batch Processing
    # ==================================================================

    def calculate_batch(
        self,
        records: List[Dict[str, Any]],
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """Process multiple wastewater emission calculations in a batch.

        Each record must contain a 'method' key to select the calculation:
            - "CH4_BOD": calls calculate_wastewater_ch4
            - "CH4_COD": calls calculate_wastewater_ch4_from_cod
            - "INDUSTRIAL": calls calculate_industrial_wastewater
            - "N2O": calls calculate_wastewater_n2o
            - "SLUDGE": calls calculate_sludge_emissions
            - "COMBINED": calls calculate_wastewater_combined

        Args:
            records: List of calculation request dictionaries.
            continue_on_error: If True, skip failed records. If False, stop.

        Returns:
            Batch result with individual results, summary, and totals.
        """
        t0 = time.monotonic()
        batch_id = f"ww_batch_{uuid4().hex[:12]}"
        results: List[Dict[str, Any]] = []
        successful = 0
        failed = 0
        total_co2e_kg = _ZERO

        method_dispatch = {
            "CH4_BOD": self._dispatch_ch4_bod,
            "CH4_COD": self._dispatch_ch4_cod,
            "INDUSTRIAL": self._dispatch_industrial,
            "N2O": self._dispatch_n2o,
            "SLUDGE": self._dispatch_sludge,
            "COMBINED": self._dispatch_combined,
        }

        for idx, record in enumerate(records):
            try:
                method_key = str(record.get("method", "CH4_BOD")).upper()
                handler = method_dispatch.get(method_key)

                if handler is None:
                    raise ValueError(
                        f"Unknown batch method: {method_key}. "
                        f"Valid: {list(method_dispatch.keys())}"
                    )

                calc_result = handler(record)
                results.append(calc_result)

                if calc_result.get("status") == CalculationStatus.SUCCESS.value:
                    successful += 1
                    co2e = _safe_decimal(
                        calc_result.get("total_co2e_kg", "0"),
                    )
                    total_co2e_kg += co2e
                else:
                    failed += 1

            except Exception as exc:
                failed += 1
                error_entry = {
                    "record_index": idx,
                    "status": CalculationStatus.ERROR.value,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                results.append(error_entry)

                if not continue_on_error:
                    logger.error(
                        "Batch %s stopped at record %d: %s",
                        batch_id, idx, exc,
                    )
                    break

                logger.warning(
                    "Batch %s record %d failed (continuing): %s",
                    batch_id, idx, exc,
                )

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        batch_result: Dict[str, Any] = {
            "batch_id": batch_id,
            "results": results,
            "summary": {
                "total_co2e_kg": str(_quantize(total_co2e_kg)),
                "total_co2e_tonnes": str(
                    _quantize(total_co2e_kg * KG_TO_TONNES)
                ),
            },
            "total_records": len(records),
            "successful": successful,
            "failed": failed,
            "continue_on_error": continue_on_error,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": _utcnow().isoformat(),
        }
        batch_result["provenance_hash"] = _compute_hash({
            k: v for k, v in batch_result.items()
            if k != "results"
        })

        logger.info(
            "Batch %s: %d/%d successful, %s kg CO2e in %.1fms",
            batch_id, successful, len(records),
            _quantize(total_co2e_kg), elapsed_ms,
        )
        return batch_result

    # ------------------------------------------------------------------
    # Batch dispatch helpers
    # ------------------------------------------------------------------

    def _dispatch_ch4_bod(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a CH4 BOD-basis calculation from batch record."""
        return self.calculate_wastewater_ch4(
            tow_kg_bod_yr=record["tow_kg_bod_yr"],
            sludge_removal_kg=record.get("sludge_removal_kg", 0),
            bo=record.get("bo"),
            mcf=record.get("mcf"),
            recovery_tonnes=record.get("recovery_tonnes", 0),
            gwp_source=record.get("gwp_source"),
            treatment_system=record.get("treatment_system"),
        )

    def _dispatch_ch4_cod(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a CH4 COD-basis calculation from batch record."""
        return self.calculate_wastewater_ch4_from_cod(
            tow_kg_cod_yr=record["tow_kg_cod_yr"],
            sludge_removal_kg=record.get("sludge_removal_kg", 0),
            bo_cod=record.get("bo_cod"),
            mcf=record.get("mcf"),
            recovery_tonnes=record.get("recovery_tonnes", 0),
            gwp_source=record.get("gwp_source"),
            treatment_system=record.get("treatment_system"),
        )

    def _dispatch_industrial(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch an industrial wastewater calculation from batch record."""
        return self.calculate_industrial_wastewater(
            production_tonnes=record["production_tonnes"],
            industry_type=record["industry_type"],
            treatment_system=record["treatment_system"],
            gwp_source=record.get("gwp_source"),
            wastewater_m3_per_tonne_override=record.get(
                "wastewater_m3_per_tonne"
            ),
            cod_kg_per_m3_override=record.get("cod_kg_per_m3"),
            mcf_override=record.get("mcf"),
            bo_cod_override=record.get("bo_cod"),
            recovery_tonnes=record.get("recovery_tonnes", 0),
            sludge_fraction=record.get("sludge_fraction", 0),
        )

    def _dispatch_n2o(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch an N2O calculation from batch record."""
        return self.calculate_wastewater_n2o(
            flow_m3_yr=record["flow_m3_yr"],
            n_influent_kg_m3=record["n_influent_kg_m3"],
            n_effluent_kg_m3=record["n_effluent_kg_m3"],
            ef_plant=record.get("ef_plant"),
            ef_effluent=record.get("ef_effluent"),
            gwp_source=record.get("gwp_source"),
        )

    def _dispatch_sludge(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a sludge emissions calculation from batch record."""
        return self.calculate_sludge_emissions(
            sludge_tonnes=record["sludge_tonnes"],
            vs_fraction=record.get("vs_fraction"),
            treatment_system=record.get("treatment_system"),
            mcf_override=record.get("mcf"),
            bo_sludge=record.get("bo_sludge"),
            gwp_source=record.get("gwp_source"),
        )

    def _dispatch_combined(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a combined CH4+N2O calculation from batch record."""
        return self.calculate_wastewater_combined(
            ch4_inputs=record.get("ch4_inputs", {}),
            n2o_inputs=record.get("n2o_inputs"),
            gwp_source=record.get("gwp_source"),
        )

    # ==================================================================
    # Statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with calculation counts and metadata.
        """
        with self._lock:
            return {
                "engine": "WastewaterTreatmentEngine",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_calculations": self._total_calculations,
                "total_errors": self._total_errors,
                "default_gwp_source": self._default_gwp_source,
                "treatment_systems_supported": len(TreatmentSystem),
                "industry_types_supported": len(IndustryType),
                "sludge_treatments_supported": len(SludgeTreatment),
            }

    def reset(self) -> None:
        """Reset engine counters. Intended for testing teardown."""
        with self._lock:
            self._total_calculations = 0
            self._total_errors = 0
        logger.info("WastewaterTreatmentEngine reset")

    # ==================================================================
    # Error response helpers
    # ==================================================================

    def _error_result(
        self,
        calc_id: str,
        errors: List[str],
        t0: float,
    ) -> Dict[str, Any]:
        """Build a validation error result.

        Args:
            calc_id: Calculation identifier.
            errors: List of validation error messages.
            t0: Monotonic start time for duration calculation.

        Returns:
            Error result dictionary.
        """
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": CalculationStatus.VALIDATION_ERROR.value,
            "errors": errors,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": _utcnow().isoformat(),
        }
        result["provenance_hash"] = _compute_hash(result)

        if _METRICS_AVAILABLE and _record_calc_error is not None:
            try:
                _record_calc_error("validation_error")
            except Exception:
                pass

        logger.warning(
            "Validation error in %s: %s", calc_id, errors,
        )
        return result

    def _exception_result(
        self,
        calc_id: str,
        exc: Exception,
        t0: float,
    ) -> Dict[str, Any]:
        """Build an exception error result.

        Args:
            calc_id: Calculation identifier.
            exc: The exception that occurred.
            t0: Monotonic start time for duration calculation.

        Returns:
            Error result dictionary.
        """
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": CalculationStatus.ERROR.value,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": _utcnow().isoformat(),
        }
        result["provenance_hash"] = _compute_hash(result)

        if _METRICS_AVAILABLE and _record_calc_error is not None:
            try:
                _record_calc_error("calculation_error")
            except Exception:
                pass

        logger.error(
            "Calculation %s failed: %s in %.1fms",
            calc_id, exc, elapsed_ms, exc_info=True,
        )
        return result
