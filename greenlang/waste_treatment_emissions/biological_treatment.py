# -*- coding: utf-8 -*-
"""
BiologicalTreatmentEngine - Composting, AD & MBT Emissions (Engine 2 of 7)

AGENT-MRV-007: On-site Waste Treatment Emissions Agent

Core calculation engine implementing IPCC 2006/2019 Refined methodologies for
biological waste treatment emissions from three treatment pathways:

    1. **Composting** (IPCC 2019 Vol 5 Ch 4):
       CH4_emitted = M_waste * EF_CH4 * (1 - R_biofilter) * (1/1000)
       N2O_emitted = M_waste * EF_N2O * (1/1000)
       CO2e = CH4_emitted * GWP_CH4 + N2O_emitted * GWP_N2O
       Five composting types: windrow, in-vessel, aerated static pile,
       vermicomposting, home composting.

    2. **Anaerobic Digestion** (AD):
       V_biogas = M_vs * BMP * eta_digestion
       V_CH4 = V_biogas * X_CH4
       CH4_generated = V_CH4 * rho_CH4 (0.0007168 t/m3 STP)
       CH4_captured = CH4_generated * eta_capture
       CH4_flared = CH4_captured * f_flare * (1 - eta_destruction)
       CH4_utilized = CH4_captured * f_utilize * (1 - eta_conversion)
       CH4_vented = CH4_captured * f_vent
       CH4_fugitive = CH4_generated * (1 - eta_capture)
       CH4_emitted = CH4_fugitive + CH4_flared + CH4_vented
       Wet/dry process, mesophilic/thermophilic, single/two-stage.

    3. **Mechanical-Biological Treatment** (MBT):
       CH4_mbt = M_bio_fraction * EF_CH4_mbt
       N2O_mbt = M_bio_fraction * EF_N2O_mbt
       Combined mechanical sorting + biological stabilization.

All calculations use Python Decimal arithmetic with 8+ decimal places for
zero-hallucination determinism.  Every calculation result includes a per-gas
breakdown, GWP-adjusted CO2e, full calculation trace, processing time, and
SHA-256 provenance hash.

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal.
    - No LLM calls in any calculation path.
    - Every calculation step is logged and traceable via TraceStep.
    - SHA-256 provenance hash for every result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Stateless per-calculation.  Mutable counters protected by reentrant lock.

Example:
    >>> from greenlang.waste_treatment_emissions.biological_treatment import (
    ...     BiologicalTreatmentEngine,
    ... )
    >>> engine = BiologicalTreatmentEngine()
    >>> result = engine.calculate_composting(
    ...     waste_tonnes=1000,
    ...     waste_category="FOOD_WASTE",
    ...     composting_type="WINDROW",
    ... )
    >>> assert result["status"] == "SUCCESS"
    >>> assert result["total_co2e_tonnes"]

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

__all__ = ["BiologicalTreatmentEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.waste_treatment_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.waste_treatment_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.waste_treatment_emissions.metrics import (
        record_calculation as _record_calculation,
        observe_calculation_duration as _observe_calculation_duration,
        record_emissions as _record_emissions,
        record_biological_treatment as _record_biological_treatment,
        record_waste_processed as _record_waste_processed,
        record_methane_recovery as _record_methane_recovery,
        record_calculation_error as _record_calculation_error,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calculation = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _record_biological_treatment = None  # type: ignore[assignment]
    _record_waste_processed = None  # type: ignore[assignment]
    _record_methane_recovery = None  # type: ignore[assignment]
    _record_calculation_error = None  # type: ignore[assignment]


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
_KG_TO_TONNES = Decimal("0.001")
_TONNES_TO_KG = Decimal("1000")

#: CH4 density at STP in tonnes per cubic metre (0.0007168 t/m3)
_CH4_DENSITY_T_PER_M3 = Decimal("0.0007168")


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
        default: Fallback Decimal value.

    Returns:
        Converted Decimal or default.
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
    try:
        return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        logger.warning("Failed to quantize value: %s", value)
        return value


# ===========================================================================
# Enumerations
# ===========================================================================


class CompostingType(str, Enum):
    """Supported composting system types."""

    WINDROW = "WINDROW"
    IN_VESSEL = "IN_VESSEL"
    AERATED_STATIC_PILE = "AERATED_STATIC_PILE"
    VERMICOMPOSTING = "VERMICOMPOSTING"
    HOME_COMPOSTING = "HOME_COMPOSTING"


class ADProcessType(str, Enum):
    """Anaerobic digestion process configurations."""

    WET_MESOPHILIC = "WET_MESOPHILIC"
    WET_THERMOPHILIC = "WET_THERMOPHILIC"
    DRY_MESOPHILIC = "DRY_MESOPHILIC"
    DRY_THERMOPHILIC = "DRY_THERMOPHILIC"
    TWO_STAGE = "TWO_STAGE"


class MBTType(str, Enum):
    """MBT configuration types."""

    MBT_AEROBIC = "MBT_AEROBIC"
    MBT_ANAEROBIC = "MBT_ANAEROBIC"
    MBT_BIODRYING = "MBT_BIODRYING"


class WasteCategory(str, Enum):
    """Waste feedstock categories for biological treatment."""

    FOOD_WASTE = "FOOD_WASTE"
    GARDEN_WASTE = "GARDEN_WASTE"
    MIXED_FOOD_GARDEN = "MIXED_FOOD_GARDEN"
    PAPER_CARDBOARD = "PAPER_CARDBOARD"
    WOOD_WASTE = "WOOD_WASTE"
    SEWAGE_SLUDGE = "SEWAGE_SLUDGE"
    AGRICULTURAL_RESIDUES = "AGRICULTURAL_RESIDUES"
    ANIMAL_MANURE = "ANIMAL_MANURE"
    MSW_ORGANIC_FRACTION = "MSW_ORGANIC_FRACTION"
    INDUSTRIAL_ORGANIC = "INDUSTRIAL_ORGANIC"


class BiofilterType(str, Enum):
    """Biofilter system types for methane oxidation."""

    NONE = "NONE"
    OPEN_BIOFILTER = "OPEN_BIOFILTER"
    ENCLOSED_BIOFILTER = "ENCLOSED_BIOFILTER"
    BIOCOVER = "BIOCOVER"
    COMPOST_COVER = "COMPOST_COVER"


class ManagementQuality(str, Enum):
    """Composting facility management quality levels."""

    WELL_MANAGED = "WELL_MANAGED"
    NOT_WELL_MANAGED = "NOT_WELL_MANAGED"
    UNMANAGED = "UNMANAGED"


class CalculationStatus(str, Enum):
    """Result status codes."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"


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
# GWP Lookup Tables (built-in fallback)
# ===========================================================================

_GWP_TABLE: Dict[str, Dict[str, Decimal]] = {
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
        "CH4": Decimal("27.9"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": Decimal("1"),
        "CH4": Decimal("82.5"),
        "N2O": Decimal("273"),
    },
}


# ===========================================================================
# Composting Emission Factor Tables (IPCC 2019 Refined, Vol 5, Ch 4, T4.1)
# ===========================================================================

#: CH4 emission factors for composting in g CH4 / kg waste treated (wet basis)
_COMPOSTING_EF_CH4: Dict[str, Dict[str, Decimal]] = {
    "WELL_MANAGED": {
        "FOOD_WASTE": Decimal("0.08"),
        "GARDEN_WASTE": Decimal("0.08"),
        "MIXED_FOOD_GARDEN": Decimal("0.08"),
        "PAPER_CARDBOARD": Decimal("0.06"),
        "WOOD_WASTE": Decimal("0.03"),
        "SEWAGE_SLUDGE": Decimal("0.08"),
        "AGRICULTURAL_RESIDUES": Decimal("0.05"),
        "ANIMAL_MANURE": Decimal("0.08"),
        "MSW_ORGANIC_FRACTION": Decimal("0.08"),
        "INDUSTRIAL_ORGANIC": Decimal("0.06"),
    },
    "NOT_WELL_MANAGED": {
        "FOOD_WASTE": Decimal("4.0"),
        "GARDEN_WASTE": Decimal("4.0"),
        "MIXED_FOOD_GARDEN": Decimal("4.0"),
        "PAPER_CARDBOARD": Decimal("3.0"),
        "WOOD_WASTE": Decimal("1.5"),
        "SEWAGE_SLUDGE": Decimal("4.0"),
        "AGRICULTURAL_RESIDUES": Decimal("2.5"),
        "ANIMAL_MANURE": Decimal("4.0"),
        "MSW_ORGANIC_FRACTION": Decimal("4.0"),
        "INDUSTRIAL_ORGANIC": Decimal("3.0"),
    },
    "UNMANAGED": {
        "FOOD_WASTE": Decimal("10.0"),
        "GARDEN_WASTE": Decimal("10.0"),
        "MIXED_FOOD_GARDEN": Decimal("10.0"),
        "PAPER_CARDBOARD": Decimal("7.5"),
        "WOOD_WASTE": Decimal("4.0"),
        "SEWAGE_SLUDGE": Decimal("10.0"),
        "AGRICULTURAL_RESIDUES": Decimal("6.0"),
        "ANIMAL_MANURE": Decimal("10.0"),
        "MSW_ORGANIC_FRACTION": Decimal("10.0"),
        "INDUSTRIAL_ORGANIC": Decimal("7.5"),
    },
}

#: N2O emission factors for composting in g N2O / kg waste treated (wet basis)
_COMPOSTING_EF_N2O: Dict[str, Dict[str, Decimal]] = {
    "WELL_MANAGED": {
        "FOOD_WASTE": Decimal("0.24"),
        "GARDEN_WASTE": Decimal("0.24"),
        "MIXED_FOOD_GARDEN": Decimal("0.24"),
        "PAPER_CARDBOARD": Decimal("0.12"),
        "WOOD_WASTE": Decimal("0.06"),
        "SEWAGE_SLUDGE": Decimal("0.24"),
        "AGRICULTURAL_RESIDUES": Decimal("0.18"),
        "ANIMAL_MANURE": Decimal("0.30"),
        "MSW_ORGANIC_FRACTION": Decimal("0.24"),
        "INDUSTRIAL_ORGANIC": Decimal("0.18"),
    },
    "NOT_WELL_MANAGED": {
        "FOOD_WASTE": Decimal("0.60"),
        "GARDEN_WASTE": Decimal("0.60"),
        "MIXED_FOOD_GARDEN": Decimal("0.60"),
        "PAPER_CARDBOARD": Decimal("0.30"),
        "WOOD_WASTE": Decimal("0.15"),
        "SEWAGE_SLUDGE": Decimal("0.60"),
        "AGRICULTURAL_RESIDUES": Decimal("0.45"),
        "ANIMAL_MANURE": Decimal("0.75"),
        "MSW_ORGANIC_FRACTION": Decimal("0.60"),
        "INDUSTRIAL_ORGANIC": Decimal("0.45"),
    },
    "UNMANAGED": {
        "FOOD_WASTE": Decimal("0.60"),
        "GARDEN_WASTE": Decimal("0.60"),
        "MIXED_FOOD_GARDEN": Decimal("0.60"),
        "PAPER_CARDBOARD": Decimal("0.30"),
        "WOOD_WASTE": Decimal("0.15"),
        "SEWAGE_SLUDGE": Decimal("0.60"),
        "AGRICULTURAL_RESIDUES": Decimal("0.45"),
        "ANIMAL_MANURE": Decimal("0.75"),
        "MSW_ORGANIC_FRACTION": Decimal("0.60"),
        "INDUSTRIAL_ORGANIC": Decimal("0.45"),
    },
}


# ===========================================================================
# Composting-type specific management-quality mapping
# ===========================================================================

#: Default management quality by composting type
_COMPOSTING_TYPE_QUALITY_MAP: Dict[str, str] = {
    "WINDROW": "NOT_WELL_MANAGED",
    "IN_VESSEL": "WELL_MANAGED",
    "AERATED_STATIC_PILE": "WELL_MANAGED",
    "VERMICOMPOSTING": "WELL_MANAGED",
    "HOME_COMPOSTING": "NOT_WELL_MANAGED",
}


# ===========================================================================
# Biofilter Efficiency Lookup
# ===========================================================================

#: Default biofilter CH4 oxidation efficiency by type (fraction 0-1)
_BIOFILTER_EFFICIENCY: Dict[str, Decimal] = {
    "NONE": Decimal("0.0"),
    "OPEN_BIOFILTER": Decimal("0.10"),
    "ENCLOSED_BIOFILTER": Decimal("0.65"),
    "BIOCOVER": Decimal("0.35"),
    "COMPOST_COVER": Decimal("0.50"),
}

# ===========================================================================
# Biochemical Methane Potential (BMP) Defaults (m3 CH4 / tonne VS at STP)
# ===========================================================================

_BMP_DEFAULTS: Dict[str, Decimal] = {
    "FOOD_WASTE": Decimal("400"),
    "GARDEN_WASTE": Decimal("200"),
    "MIXED_FOOD_GARDEN": Decimal("300"),
    "PAPER_CARDBOARD": Decimal("270"),
    "WOOD_WASTE": Decimal("150"),
    "SEWAGE_SLUDGE": Decimal("280"),
    "AGRICULTURAL_RESIDUES": Decimal("250"),
    "ANIMAL_MANURE": Decimal("220"),
    "MSW_ORGANIC_FRACTION": Decimal("350"),
    "INDUSTRIAL_ORGANIC": Decimal("320"),
}

# ===========================================================================
# Volatile Solids Fraction Defaults (fraction of total solids)
# ===========================================================================

_VS_FRACTION_DEFAULTS: Dict[str, Decimal] = {
    "FOOD_WASTE": Decimal("0.85"),
    "GARDEN_WASTE": Decimal("0.65"),
    "MIXED_FOOD_GARDEN": Decimal("0.75"),
    "PAPER_CARDBOARD": Decimal("0.82"),
    "WOOD_WASTE": Decimal("0.90"),
    "SEWAGE_SLUDGE": Decimal("0.65"),
    "AGRICULTURAL_RESIDUES": Decimal("0.70"),
    "ANIMAL_MANURE": Decimal("0.80"),
    "MSW_ORGANIC_FRACTION": Decimal("0.70"),
    "INDUSTRIAL_ORGANIC": Decimal("0.75"),
}

# ===========================================================================
# Total Solids Fraction Defaults (fraction of wet weight)
# ===========================================================================

_TS_FRACTION_DEFAULTS: Dict[str, Decimal] = {
    "FOOD_WASTE": Decimal("0.25"),
    "GARDEN_WASTE": Decimal("0.45"),
    "MIXED_FOOD_GARDEN": Decimal("0.35"),
    "PAPER_CARDBOARD": Decimal("0.85"),
    "WOOD_WASTE": Decimal("0.80"),
    "SEWAGE_SLUDGE": Decimal("0.05"),
    "AGRICULTURAL_RESIDUES": Decimal("0.40"),
    "ANIMAL_MANURE": Decimal("0.15"),
    "MSW_ORGANIC_FRACTION": Decimal("0.30"),
    "INDUSTRIAL_ORGANIC": Decimal("0.35"),
}

# ===========================================================================
# AD Process Parameters
# ===========================================================================

#: Default digestion efficiency by AD process type
_AD_DIGESTION_EFFICIENCY: Dict[str, Decimal] = {
    "WET_MESOPHILIC": Decimal("0.60"),
    "WET_THERMOPHILIC": Decimal("0.70"),
    "DRY_MESOPHILIC": Decimal("0.55"),
    "DRY_THERMOPHILIC": Decimal("0.65"),
    "TWO_STAGE": Decimal("0.75"),
}

#: Default methane fraction in biogas (X_CH4) by AD process type
_AD_METHANE_FRACTION: Dict[str, Decimal] = {
    "WET_MESOPHILIC": Decimal("0.60"),
    "WET_THERMOPHILIC": Decimal("0.62"),
    "DRY_MESOPHILIC": Decimal("0.58"),
    "DRY_THERMOPHILIC": Decimal("0.60"),
    "TWO_STAGE": Decimal("0.65"),
}

#: Default capture efficiency by AD process type
_AD_CAPTURE_EFFICIENCY: Dict[str, Decimal] = {
    "WET_MESOPHILIC": Decimal("0.98"),
    "WET_THERMOPHILIC": Decimal("0.98"),
    "DRY_MESOPHILIC": Decimal("0.95"),
    "DRY_THERMOPHILIC": Decimal("0.96"),
    "TWO_STAGE": Decimal("0.98"),
}

# ===========================================================================
# MBT Emission Factors (IPCC 2019 Refined, Vol 5 Ch 4 Table 4.3)
# ===========================================================================

#: CH4 emission factors for MBT in g CH4 / kg waste (wet basis)
_MBT_EF_CH4: Dict[str, Decimal] = {
    "MBT_AEROBIC": Decimal("2.0"),
    "MBT_ANAEROBIC": Decimal("0.50"),
    "MBT_BIODRYING": Decimal("3.5"),
}

#: N2O emission factors for MBT in g N2O / kg waste (wet basis)
_MBT_EF_N2O: Dict[str, Decimal] = {
    "MBT_AEROBIC": Decimal("0.30"),
    "MBT_ANAEROBIC": Decimal("0.10"),
    "MBT_BIODRYING": Decimal("0.50"),
}

#: Default biological fraction of input waste by MBT type
_MBT_BIO_FRACTION: Dict[str, Decimal] = {
    "MBT_AEROBIC": Decimal("0.55"),
    "MBT_ANAEROBIC": Decimal("0.55"),
    "MBT_BIODRYING": Decimal("0.60"),
}


# ===========================================================================
# BiologicalTreatmentEngine
# ===========================================================================


class BiologicalTreatmentEngine:
    """Core calculation engine for biological waste treatment emissions
    implementing IPCC 2006/2019 methodologies for composting, anaerobic
    digestion, and mechanical-biological treatment.

    Uses deterministic Decimal arithmetic throughout.  All numeric lookups
    are performed from built-in IPCC default tables.  External database
    engines may be injected to override defaults when site-specific data
    is available.

    Thread Safety:
        Per-calculation state is created fresh for each method call.
        Shared counters use a reentrant lock.

    Attributes:
        _config: Optional configuration dictionary.
        _lock: Reentrant lock protecting mutable counters.
        _total_calculations: Counter of total calculations performed.
        _total_batches: Counter of total batch operations.
        _total_errors: Counter of total errors encountered.
        _default_gwp_source: Default GWP source for CO2e conversion.
        _created_at: Engine initialization timestamp.

    Example:
        >>> engine = BiologicalTreatmentEngine()
        >>> result = engine.calculate_composting(
        ...     waste_tonnes=500,
        ...     waste_category="FOOD_WASTE",
        ...     composting_type="IN_VESSEL",
        ...     biofilter_type="ENCLOSED_BIOFILTER",
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(
        self,
        waste_database: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the BiologicalTreatmentEngine.

        Args:
            waste_database: Optional waste treatment database engine for
                site-specific emission factor lookups.  If None, built-in
                IPCC default tables are used.
            config: Optional configuration dictionary.  Supports:
                - default_gwp_source (str): Default GWP report (AR4/AR5/AR6).
                - enable_provenance (bool): Enable provenance tracking.
                - decimal_precision (int): Decimal places (default 8).
        """
        self._waste_db = waste_database
        self._config = config or {}
        self._lock = threading.RLock()

        # Configuration
        self._default_gwp_source: str = self._config.get(
            "default_gwp_source", "AR6",
        )
        self._enable_provenance: bool = self._config.get(
            "enable_provenance", True,
        )

        # Provenance tracker
        if self._enable_provenance and _PROVENANCE_AVAILABLE:
            self._provenance = _get_provenance_tracker()
        else:
            self._provenance = None

        # Statistics
        self._total_calculations: int = 0
        self._total_batches: int = 0
        self._total_errors: int = 0
        self._created_at: datetime = _utcnow()

        logger.info(
            "BiologicalTreatmentEngine initialized: default_gwp=%s, "
            "provenance=%s, waste_db=%s",
            self._default_gwp_source,
            "enabled" if self._provenance else "disabled",
            "connected" if self._waste_db else "built-in",
        )

    # ------------------------------------------------------------------
    # Thread-safe counter helpers
    # ------------------------------------------------------------------

    def _increment_calculations(self) -> None:
        """Thread-safe increment of the calculation counter."""
        with self._lock:
            self._total_calculations += 1

    def _increment_batches(self) -> None:
        """Thread-safe increment of the batch counter."""
        with self._lock:
            self._total_batches += 1

    def _increment_errors(self) -> None:
        """Thread-safe increment of the error counter."""
        with self._lock:
            self._total_errors += 1

    # ==================================================================
    # PUBLIC API: Composting
    # ==================================================================

    def calculate_composting(
        self,
        waste_tonnes: Any,
        waste_category: str = "FOOD_WASTE",
        composting_type: str = "WINDROW",
        management_quality: Optional[str] = None,
        biofilter_type: str = "NONE",
        biofilter_efficiency: Optional[Any] = None,
        gwp_source: Optional[str] = None,
        ef_ch4_override: Optional[Any] = None,
        ef_n2o_override: Optional[Any] = None,
        calculation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate GHG emissions from composting of organic waste.

        Implements IPCC 2019 Refined Guidelines Vol 5 Ch 4 methodology:
            CH4_emitted = M_waste * EF_CH4 * (1 - R_biofilter) * (1/1000)
            N2O_emitted = M_waste * EF_N2O * (1/1000)
            CO2e = CH4_emitted * GWP_CH4 + N2O_emitted * GWP_N2O

        Biogenic CO2 from composting is considered carbon-neutral under
        IPCC/GHG Protocol and is not counted toward total CO2e.

        Args:
            waste_tonnes: Mass of waste composted in tonnes (wet basis).
            waste_category: Waste feedstock category.  Must be a valid
                WasteCategory value.  Default ``"FOOD_WASTE"``.
            composting_type: Composting system type.  Must be a valid
                CompostingType value.  Default ``"WINDROW"``.
            management_quality: Override management quality level.
                If None, inferred from composting_type.
            biofilter_type: Type of biofilter installed for CH4 oxidation.
                Default ``"NONE"``.
            biofilter_efficiency: Override biofilter CH4 oxidation efficiency
                (0.0 to 1.0).  If None, looked up from biofilter_type.
            gwp_source: GWP report edition override.  Default uses engine
                configuration (``"AR6"``).
            ef_ch4_override: Override CH4 emission factor in g CH4 / kg waste.
            ef_n2o_override: Override N2O emission factor in g N2O / kg waste.
            calculation_id: Optional external calculation identifier.

        Returns:
            Dictionary with keys:
                - calculation_id (str)
                - status (str): "SUCCESS" or "ERROR"
                - treatment_type (str): "COMPOSTING"
                - composting_type (str)
                - waste_category (str)
                - management_quality (str)
                - waste_tonnes (str): Input waste mass
                - emissions_by_gas (List[Dict]): Per-gas breakdown
                - total_co2e_kg (str)
                - total_co2e_tonnes (str)
                - biofilter_details (Dict)
                - calculation_trace (List[Dict])
                - provenance_hash (str)
                - processing_time_ms (float)
                - calculated_at (str)

        Raises:
            ValueError: If required fields are missing or invalid.

        Example:
            >>> result = engine.calculate_composting(
            ...     waste_tonnes=1000,
            ...     waste_category="GARDEN_WASTE",
            ...     composting_type="AERATED_STATIC_PILE",
            ...     biofilter_type="ENCLOSED_BIOFILTER",
            ... )
            >>> result["status"]
            'SUCCESS'
        """
        self._increment_calculations()
        start_time = time.monotonic()
        calc_id = calculation_id or f"wt_comp_{uuid4().hex[:12]}"
        trace_steps: List[TraceStep] = []
        step_num = 0

        try:
            # -- Parse and validate inputs ------------------------------------
            m_waste = _D(waste_tonnes)
            cat_key = waste_category.upper()
            comp_key = composting_type.upper()
            gwp = (gwp_source or self._default_gwp_source).upper()

            errors = self._validate_composting_inputs(
                m_waste, cat_key, comp_key,
            )
            if errors:
                raise ValueError(
                    f"Composting input validation failed: {'; '.join(errors)}"
                )

            # -- Resolve management quality -----------------------------------
            if management_quality is not None:
                mgmt_key = management_quality.upper()
            else:
                mgmt_key = _COMPOSTING_TYPE_QUALITY_MAP.get(
                    comp_key, "NOT_WELL_MANAGED",
                )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Input parameters validated",
                formula="N/A",
                inputs={
                    "waste_tonnes": str(m_waste),
                    "waste_category": cat_key,
                    "composting_type": comp_key,
                    "management_quality": mgmt_key,
                    "gwp_source": gwp,
                },
                output="VALID",
                unit="N/A",
            ))

            # -- Resolve emission factors -------------------------------------
            ef_ch4, ef_n2o = self._resolve_composting_efs(
                cat_key, mgmt_key, ef_ch4_override, ef_n2o_override,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Emission factors resolved",
                formula="EF lookup from IPCC 2019 Table 4.1 or user override",
                inputs={
                    "waste_category": cat_key,
                    "management_quality": mgmt_key,
                },
                output=f"EF_CH4={ef_ch4} g/kg, EF_N2O={ef_n2o} g/kg",
                unit="g/kg waste",
            ))

            # -- Resolve biofilter efficiency ---------------------------------
            bf_eff = self._resolve_biofilter_efficiency(
                biofilter_type.upper(), biofilter_efficiency,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Biofilter efficiency resolved",
                formula="R_biofilter lookup or user override",
                inputs={
                    "biofilter_type": biofilter_type.upper(),
                    "user_override": str(biofilter_efficiency),
                },
                output=str(bf_eff),
                unit="fraction (0-1)",
            ))

            # -- Calculate CH4 emissions (g -> tonnes) ------------------------
            # CH4_gross = M_waste * EF_CH4 * (1/1000)
            # Convert: M_waste (tonnes) * EF_CH4 (g/kg) * 1000 kg/t / 1e6 g/t
            # Simplification: M_waste * EF_CH4 / 1000 = tonnes CH4
            ch4_gross_tonnes = _quantize(
                m_waste * ef_ch4 / _THOUSAND
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="CH4 gross emissions before biofilter",
                formula="CH4_gross = M_waste * EF_CH4 / 1000",
                inputs={
                    "M_waste": str(m_waste),
                    "EF_CH4": str(ef_ch4),
                },
                output=str(ch4_gross_tonnes),
                unit="tonnes CH4",
            ))

            # Apply biofilter oxidation
            # CH4_emitted = CH4_gross * (1 - R_biofilter)
            ch4_emitted_tonnes = _quantize(
                ch4_gross_tonnes * (_ONE - bf_eff)
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="CH4 net emissions after biofilter oxidation",
                formula="CH4_emitted = CH4_gross * (1 - R_biofilter)",
                inputs={
                    "CH4_gross": str(ch4_gross_tonnes),
                    "R_biofilter": str(bf_eff),
                },
                output=str(ch4_emitted_tonnes),
                unit="tonnes CH4",
            ))

            ch4_oxidized_tonnes = _quantize(
                ch4_gross_tonnes * bf_eff
            )

            # -- Calculate N2O emissions (g -> tonnes) ------------------------
            # N2O_emitted = M_waste * EF_N2O / 1000
            n2o_emitted_tonnes = _quantize(
                m_waste * ef_n2o / _THOUSAND
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="N2O emissions from composting",
                formula="N2O_emitted = M_waste * EF_N2O / 1000",
                inputs={
                    "M_waste": str(m_waste),
                    "EF_N2O": str(ef_n2o),
                },
                output=str(n2o_emitted_tonnes),
                unit="tonnes N2O",
            ))

            # -- GWP conversion -----------------------------------------------
            gwp_ch4 = self._resolve_gwp("CH4", gwp)
            gwp_n2o = self._resolve_gwp("N2O", gwp)

            ch4_co2e_tonnes = _quantize(ch4_emitted_tonnes * gwp_ch4)
            n2o_co2e_tonnes = _quantize(n2o_emitted_tonnes * gwp_n2o)
            total_co2e_tonnes = _quantize(ch4_co2e_tonnes + n2o_co2e_tonnes)
            total_co2e_kg = _quantize(total_co2e_tonnes * _TONNES_TO_KG)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="GWP conversion to CO2e",
                formula="CO2e = CH4 * GWP_CH4 + N2O * GWP_N2O",
                inputs={
                    "CH4_emitted": str(ch4_emitted_tonnes),
                    "GWP_CH4": str(gwp_ch4),
                    "N2O_emitted": str(n2o_emitted_tonnes),
                    "GWP_N2O": str(gwp_n2o),
                },
                output=str(total_co2e_tonnes),
                unit="tonnes CO2e",
            ))

            # -- Build per-gas breakdown --------------------------------------
            emissions_by_gas = [
                {
                    "gas": "CH4",
                    "gross_emission_tonnes": str(ch4_gross_tonnes),
                    "net_emission_tonnes": str(ch4_emitted_tonnes),
                    "oxidized_tonnes": str(ch4_oxidized_tonnes),
                    "gwp_value": str(gwp_ch4),
                    "gwp_source": gwp,
                    "co2e_tonnes": str(ch4_co2e_tonnes),
                    "co2e_kg": str(_quantize(ch4_co2e_tonnes * _TONNES_TO_KG)),
                },
                {
                    "gas": "N2O",
                    "gross_emission_tonnes": str(n2o_emitted_tonnes),
                    "net_emission_tonnes": str(n2o_emitted_tonnes),
                    "oxidized_tonnes": "0",
                    "gwp_value": str(gwp_n2o),
                    "gwp_source": gwp,
                    "co2e_tonnes": str(n2o_co2e_tonnes),
                    "co2e_kg": str(_quantize(n2o_co2e_tonnes * _TONNES_TO_KG)),
                },
            ]

            # -- Build result -------------------------------------------------
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "treatment_type": "COMPOSTING",
                "composting_type": comp_key,
                "waste_category": cat_key,
                "management_quality": mgmt_key,
                "waste_tonnes": str(m_waste),
                "ef_ch4_g_per_kg": str(ef_ch4),
                "ef_n2o_g_per_kg": str(ef_n2o),
                "emissions_by_gas": emissions_by_gas,
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "biofilter_details": {
                    "biofilter_type": biofilter_type.upper(),
                    "efficiency": str(bf_eff),
                    "ch4_oxidized_tonnes": str(ch4_oxidized_tonnes),
                },
                "gwp_source": gwp,
                "calculation_trace": [ts.to_dict() for ts in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            # -- Metrics ------------------------------------------------------
            self._record_metrics_composting(
                comp_key, cat_key, total_co2e_tonnes,
                ch4_emitted_tonnes, n2o_emitted_tonnes, m_waste, elapsed_ms,
            )

            # -- Provenance ---------------------------------------------------
            self._record_provenance(
                "calculate_composting", calc_id,
                {
                    "composting_type": comp_key,
                    "waste_category": cat_key,
                    "total_co2e_tonnes": str(total_co2e_tonnes),
                    "hash": result["provenance_hash"],
                },
            )

            logger.info(
                "Composting %s: %s t waste -> %s t CO2e "
                "(%s CH4, %s N2O) in %.1fms [%s]",
                comp_key, m_waste, total_co2e_tonnes,
                ch4_emitted_tonnes, n2o_emitted_tonnes, elapsed_ms, calc_id,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            if _METRICS_AVAILABLE and _record_calculation_error is not None:
                _record_calculation_error("calculation_error")

            error_result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.ERROR.value,
                "treatment_type": "COMPOSTING",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            error_result["provenance_hash"] = _compute_hash(error_result)

            logger.error(
                "Composting calculation failed [%s]: %s in %.1fms",
                calc_id, exc, elapsed_ms, exc_info=True,
            )
            return error_result

    # ==================================================================
    # PUBLIC API: Anaerobic Digestion
    # ==================================================================

    def calculate_anaerobic_digestion(
        self,
        waste_tonnes: Any,
        waste_category: str = "FOOD_WASTE",
        ad_process_type: str = "WET_MESOPHILIC",
        volatile_solids_fraction: Optional[Any] = None,
        total_solids_fraction: Optional[Any] = None,
        bmp: Optional[Any] = None,
        digestion_efficiency: Optional[Any] = None,
        methane_fraction: Optional[Any] = None,
        capture_efficiency: Optional[Any] = None,
        flare_fraction: Optional[Any] = None,
        utilize_fraction: Optional[Any] = None,
        vent_fraction: Optional[Any] = None,
        flare_destruction_efficiency: Optional[Any] = None,
        utilization_conversion_loss: Optional[Any] = None,
        digestate_n2o_ef: Optional[Any] = None,
        gwp_source: Optional[str] = None,
        calculation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate GHG emissions from anaerobic digestion of organic waste.

        Implements a complete AD biogas mass balance:
            V_biogas = M_vs * BMP * eta_digestion
            V_CH4 = V_biogas * X_CH4
            CH4_generated = V_CH4 * rho_CH4
            CH4_captured = CH4_generated * eta_capture
            CH4_flared = CH4_captured * f_flare * (1 - eta_destruction)
            CH4_utilized = CH4_captured * f_utilize * (1 - eta_conversion)
            CH4_vented = CH4_captured * f_vent
            CH4_fugitive = CH4_generated * (1 - eta_capture)
            CH4_emitted = CH4_fugitive + CH4_flared + CH4_vented

        Digestate N2O is optionally calculated using a user-supplied
        emission factor (kg N2O / tonne waste).

        Args:
            waste_tonnes: Mass of waste fed to digester (wet basis, tonnes).
            waste_category: Waste feedstock category for default lookups.
            ad_process_type: AD process configuration.  Default
                ``"WET_MESOPHILIC"``.
            volatile_solids_fraction: VS fraction of total solids (0-1).
                If None, looked up from waste_category.
            total_solids_fraction: TS fraction of wet weight (0-1).
                If None, looked up from waste_category.
            bmp: Biochemical methane potential (m3 CH4 / tonne VS at STP).
                If None, looked up from waste_category.
            digestion_efficiency: VS destruction efficiency (0-1).
                If None, default for ad_process_type.
            methane_fraction: CH4 volume fraction in biogas (0.5-0.7).
                If None, default for ad_process_type.
            capture_efficiency: Biogas capture efficiency (0-1).
                If None, default for ad_process_type.
            flare_fraction: Fraction of captured biogas sent to flare (0-1).
            utilize_fraction: Fraction of captured biogas utilized (0-1).
            vent_fraction: Fraction of captured biogas intentionally vented
                (0-1).  Sum of flare + utilize + vent must equal 1.0 (or
                vent is computed as the remainder).
            flare_destruction_efficiency: Flare DRE (0-1).  Default 0.98.
            utilization_conversion_loss: Fraction of utilized CH4 that
                escapes uncombusted (0-1).  Default 0.02.
            digestate_n2o_ef: Optional N2O emission factor for digestate
                handling in kg N2O / tonne waste.  Default 0.
            gwp_source: GWP report edition override.
            calculation_id: Optional external calculation identifier.

        Returns:
            Dictionary with keys:
                - calculation_id, status, treatment_type ("ANAEROBIC_DIGESTION")
                - ad_process_type, waste_category, waste_tonnes
                - biogas_details (Dict): Biogas generation parameters
                - methane_balance (Dict): Full CH4 mass balance
                - emissions_by_gas (List[Dict])
                - total_co2e_kg, total_co2e_tonnes
                - calculation_trace, provenance_hash, processing_time_ms

        Raises:
            ValueError: If input validation fails.

        Example:
            >>> result = engine.calculate_anaerobic_digestion(
            ...     waste_tonnes=2000,
            ...     waste_category="FOOD_WASTE",
            ...     ad_process_type="WET_THERMOPHILIC",
            ...     flare_fraction=0.3,
            ...     utilize_fraction=0.7,
            ... )
            >>> result["methane_balance"]["ch4_emitted_tonnes"]
        """
        self._increment_calculations()
        start_time = time.monotonic()
        calc_id = calculation_id or f"wt_ad_{uuid4().hex[:12]}"
        trace_steps: List[TraceStep] = []
        step_num = 0

        try:
            # -- Parse and validate inputs ------------------------------------
            m_waste = _D(waste_tonnes)
            cat_key = waste_category.upper()
            ad_key = ad_process_type.upper()
            gwp = (gwp_source or self._default_gwp_source).upper()

            errors = self._validate_ad_inputs(m_waste, cat_key, ad_key)
            if errors:
                raise ValueError(
                    f"AD input validation failed: {'; '.join(errors)}"
                )

            # -- Resolve parameters -------------------------------------------
            ts_frac = self._resolve_param(
                total_solids_fraction,
                _TS_FRACTION_DEFAULTS.get(cat_key, Decimal("0.30")),
            )
            vs_frac = self._resolve_param(
                volatile_solids_fraction,
                _VS_FRACTION_DEFAULTS.get(cat_key, Decimal("0.70")),
            )
            bmp_val = self._resolve_param(
                bmp,
                _BMP_DEFAULTS.get(cat_key, Decimal("300")),
            )
            eta_dig = self._resolve_param(
                digestion_efficiency,
                _AD_DIGESTION_EFFICIENCY.get(ad_key, Decimal("0.60")),
            )
            x_ch4 = self._resolve_param(
                methane_fraction,
                _AD_METHANE_FRACTION.get(ad_key, Decimal("0.60")),
            )
            eta_cap = self._resolve_param(
                capture_efficiency,
                _AD_CAPTURE_EFFICIENCY.get(ad_key, Decimal("0.98")),
            )
            eta_flare_dre = self._resolve_param(
                flare_destruction_efficiency, Decimal("0.98"),
            )
            eta_util_loss = self._resolve_param(
                utilization_conversion_loss, Decimal("0.02"),
            )
            n2o_ef = self._resolve_param(
                digestate_n2o_ef, _ZERO,
            )

            # Resolve biogas destination fractions
            f_flare, f_utilize, f_vent = self._resolve_biogas_fractions(
                flare_fraction, utilize_fraction, vent_fraction,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="AD parameters resolved",
                formula="Parameter lookup / user override",
                inputs={
                    "waste_tonnes": str(m_waste),
                    "TS_fraction": str(ts_frac),
                    "VS_fraction": str(vs_frac),
                    "BMP": str(bmp_val),
                    "eta_digestion": str(eta_dig),
                    "X_CH4": str(x_ch4),
                    "eta_capture": str(eta_cap),
                    "f_flare": str(f_flare),
                    "f_utilize": str(f_utilize),
                    "f_vent": str(f_vent),
                },
                output="PARAMETERS_RESOLVED",
                unit="N/A",
            ))

            # -- Step 1: Calculate volatile solids mass -----------------------
            m_ts = _quantize(m_waste * ts_frac)
            m_vs = _quantize(m_ts * vs_frac)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Volatile solids mass calculation",
                formula="M_VS = M_waste * TS_fraction * VS_fraction",
                inputs={
                    "M_waste": str(m_waste),
                    "TS_fraction": str(ts_frac),
                    "VS_fraction": str(vs_frac),
                },
                output=str(m_vs),
                unit="tonnes VS",
            ))

            # -- Step 2: Biogas and methane volume ----------------------------
            # V_biogas = M_vs * BMP * eta_digestion (m3 biogas at STP)
            # BMP is in m3 CH4 / tonne VS, so V_CH4 = M_vs * BMP * eta_dig
            # V_biogas = V_CH4 / X_CH4
            v_ch4 = _quantize(m_vs * bmp_val * eta_dig)
            v_biogas = _quantize(
                v_ch4 / x_ch4 if x_ch4 > _ZERO else _ZERO
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Biogas and methane volume generation",
                formula="V_CH4 = M_VS * BMP * eta_digestion; "
                        "V_biogas = V_CH4 / X_CH4",
                inputs={
                    "M_VS": str(m_vs),
                    "BMP": str(bmp_val),
                    "eta_digestion": str(eta_dig),
                    "X_CH4": str(x_ch4),
                },
                output=f"V_CH4={v_ch4}, V_biogas={v_biogas}",
                unit="m3 at STP",
            ))

            # -- Step 3: CH4 mass generated -----------------------------------
            # CH4_generated = V_CH4 * rho_CH4 (tonnes)
            ch4_generated = _quantize(v_ch4 * _CH4_DENSITY_T_PER_M3)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="CH4 mass generated from biogas",
                formula="CH4_generated = V_CH4 * rho_CH4",
                inputs={
                    "V_CH4": str(v_ch4),
                    "rho_CH4": str(_CH4_DENSITY_T_PER_M3),
                },
                output=str(ch4_generated),
                unit="tonnes CH4",
            ))

            # -- Step 4: Capture and fugitive ---------------------------------
            ch4_captured = _quantize(ch4_generated * eta_cap)
            ch4_fugitive = _quantize(ch4_generated * (_ONE - eta_cap))

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="CH4 capture and fugitive split",
                formula="CH4_captured = CH4_gen * eta_capture; "
                        "CH4_fugitive = CH4_gen * (1 - eta_capture)",
                inputs={
                    "CH4_generated": str(ch4_generated),
                    "eta_capture": str(eta_cap),
                },
                output=f"captured={ch4_captured}, fugitive={ch4_fugitive}",
                unit="tonnes CH4",
            ))

            # -- Step 5: Biogas destination pathways --------------------------
            # Flared CH4 emission = captured * f_flare * (1 - DRE)
            ch4_to_flare = _quantize(ch4_captured * f_flare)
            ch4_flared_emission = _quantize(
                ch4_to_flare * (_ONE - eta_flare_dre)
            )
            ch4_flare_destroyed = _quantize(ch4_to_flare * eta_flare_dre)

            # Utilized CH4 emission = captured * f_utilize * conversion_loss
            ch4_to_utilize = _quantize(ch4_captured * f_utilize)
            ch4_utilized_emission = _quantize(
                ch4_to_utilize * eta_util_loss
            )
            ch4_utilized_effective = _quantize(
                ch4_to_utilize * (_ONE - eta_util_loss)
            )

            # Vented CH4 = captured * f_vent (all emitted)
            ch4_vented = _quantize(ch4_captured * f_vent)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Biogas destination pathway CH4 emissions",
                formula="CH4_flared_em = cap * f_flare * (1-DRE); "
                        "CH4_util_em = cap * f_util * loss; "
                        "CH4_vented = cap * f_vent",
                inputs={
                    "CH4_captured": str(ch4_captured),
                    "f_flare": str(f_flare),
                    "eta_flare_DRE": str(eta_flare_dre),
                    "f_utilize": str(f_utilize),
                    "eta_util_loss": str(eta_util_loss),
                    "f_vent": str(f_vent),
                },
                output=(
                    f"flared_em={ch4_flared_emission}, "
                    f"util_em={ch4_utilized_emission}, "
                    f"vented={ch4_vented}"
                ),
                unit="tonnes CH4",
            ))

            # -- Step 6: Total CH4 emitted ------------------------------------
            ch4_emitted = _quantize(
                ch4_fugitive + ch4_flared_emission + ch4_vented
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Total CH4 emitted to atmosphere",
                formula="CH4_emitted = CH4_fugitive + CH4_flared_em + "
                        "CH4_vented",
                inputs={
                    "CH4_fugitive": str(ch4_fugitive),
                    "CH4_flared_emission": str(ch4_flared_emission),
                    "CH4_vented": str(ch4_vented),
                },
                output=str(ch4_emitted),
                unit="tonnes CH4",
            ))

            # -- Step 7: Digestate N2O (optional) -----------------------------
            n2o_emitted = _ZERO
            if n2o_ef > _ZERO:
                # n2o_ef is in kg N2O / tonne waste
                n2o_emitted = _quantize(m_waste * n2o_ef * _KG_TO_TONNES)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Digestate N2O emissions",
                formula="N2O = M_waste * EF_N2O * (1/1000)",
                inputs={
                    "M_waste": str(m_waste),
                    "EF_N2O_kg_per_t": str(n2o_ef),
                },
                output=str(n2o_emitted),
                unit="tonnes N2O",
            ))

            # -- Step 8: GWP conversion ---------------------------------------
            gwp_ch4 = self._resolve_gwp("CH4", gwp)
            gwp_n2o = self._resolve_gwp("N2O", gwp)

            ch4_co2e_tonnes = _quantize(ch4_emitted * gwp_ch4)
            n2o_co2e_tonnes = _quantize(n2o_emitted * gwp_n2o)
            total_co2e_tonnes = _quantize(ch4_co2e_tonnes + n2o_co2e_tonnes)
            total_co2e_kg = _quantize(total_co2e_tonnes * _TONNES_TO_KG)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="GWP conversion to CO2e",
                formula="CO2e = CH4_em * GWP_CH4 + N2O_em * GWP_N2O",
                inputs={
                    "CH4_emitted": str(ch4_emitted),
                    "GWP_CH4": str(gwp_ch4),
                    "N2O_emitted": str(n2o_emitted),
                    "GWP_N2O": str(gwp_n2o),
                },
                output=str(total_co2e_tonnes),
                unit="tonnes CO2e",
            ))

            # -- Build per-gas breakdown --------------------------------------
            emissions_by_gas = self._build_ad_gas_breakdown(
                ch4_emitted, n2o_emitted,
                gwp_ch4, gwp_n2o, gwp,
                ch4_co2e_tonnes, n2o_co2e_tonnes,
            )

            # -- Build methane balance ----------------------------------------
            methane_balance = {
                "ch4_generated_tonnes": str(ch4_generated),
                "ch4_captured_tonnes": str(ch4_captured),
                "ch4_fugitive_tonnes": str(ch4_fugitive),
                "ch4_to_flare_tonnes": str(ch4_to_flare),
                "ch4_flare_destroyed_tonnes": str(ch4_flare_destroyed),
                "ch4_flared_emission_tonnes": str(ch4_flared_emission),
                "ch4_to_utilize_tonnes": str(ch4_to_utilize),
                "ch4_utilized_effective_tonnes": str(ch4_utilized_effective),
                "ch4_utilized_emission_tonnes": str(ch4_utilized_emission),
                "ch4_vented_tonnes": str(ch4_vented),
                "ch4_emitted_tonnes": str(ch4_emitted),
            }

            # -- Build biogas details -----------------------------------------
            biogas_details = {
                "volatile_solids_tonnes": str(m_vs),
                "total_solids_tonnes": str(m_ts),
                "bmp_m3_ch4_per_t_vs": str(bmp_val),
                "digestion_efficiency": str(eta_dig),
                "biogas_volume_m3": str(v_biogas),
                "methane_volume_m3": str(v_ch4),
                "methane_fraction": str(x_ch4),
                "capture_efficiency": str(eta_cap),
            }

            # -- Build result -------------------------------------------------
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "treatment_type": "ANAEROBIC_DIGESTION",
                "ad_process_type": ad_key,
                "waste_category": cat_key,
                "waste_tonnes": str(m_waste),
                "biogas_details": biogas_details,
                "methane_balance": methane_balance,
                "emissions_by_gas": emissions_by_gas,
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "gwp_source": gwp,
                "calculation_trace": [ts.to_dict() for ts in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            # -- Metrics ------------------------------------------------------
            self._record_metrics_ad(
                ad_key, cat_key, total_co2e_tonnes,
                ch4_emitted, n2o_emitted, m_waste, elapsed_ms,
                ch4_captured, ch4_fugitive, ch4_vented,
            )

            # -- Provenance ---------------------------------------------------
            self._record_provenance(
                "calculate_anaerobic_digestion", calc_id,
                {
                    "ad_process_type": ad_key,
                    "waste_category": cat_key,
                    "total_co2e_tonnes": str(total_co2e_tonnes),
                    "ch4_emitted_tonnes": str(ch4_emitted),
                    "hash": result["provenance_hash"],
                },
            )

            logger.info(
                "AD %s: %s t waste -> %s t CO2e "
                "(CH4_gen=%s, CH4_em=%s, N2O=%s) in %.1fms [%s]",
                ad_key, m_waste, total_co2e_tonnes,
                ch4_generated, ch4_emitted, n2o_emitted, elapsed_ms, calc_id,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            if _METRICS_AVAILABLE and _record_calculation_error is not None:
                _record_calculation_error("calculation_error")

            error_result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.ERROR.value,
                "treatment_type": "ANAEROBIC_DIGESTION",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            error_result["provenance_hash"] = _compute_hash(error_result)

            logger.error(
                "AD calculation failed [%s]: %s in %.1fms",
                calc_id, exc, elapsed_ms, exc_info=True,
            )
            return error_result

    # ==================================================================
    # PUBLIC API: Mechanical-Biological Treatment (MBT)
    # ==================================================================

    def calculate_mbt(
        self,
        waste_tonnes: Any,
        waste_category: str = "MSW_ORGANIC_FRACTION",
        mbt_type: str = "MBT_AEROBIC",
        bio_fraction: Optional[Any] = None,
        ef_ch4_override: Optional[Any] = None,
        ef_n2o_override: Optional[Any] = None,
        gwp_source: Optional[str] = None,
        calculation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate GHG emissions from mechanical-biological treatment.

        MBT combines mechanical sorting (to separate recyclables and RDF)
        with biological stabilization of the organic fraction.

        Formulas (IPCC 2019 Refined, Vol 5 Ch 4):
            M_bio = M_waste * bio_fraction
            CH4_mbt = M_bio * EF_CH4_mbt / 1000  (tonnes)
            N2O_mbt = M_bio * EF_N2O_mbt / 1000  (tonnes)
            CO2e = CH4_mbt * GWP_CH4 + N2O_mbt * GWP_N2O

        Three MBT configurations are supported:
            - MBT_AEROBIC: Aerobic stabilization (composting-like)
            - MBT_ANAEROBIC: Anaerobic component (AD + mechanical)
            - MBT_BIODRYING: Biodrying for RDF production

        Args:
            waste_tonnes: Mass of waste entering MBT (wet basis, tonnes).
            waste_category: Waste feedstock category.  Default
                ``"MSW_ORGANIC_FRACTION"``.
            mbt_type: MBT configuration type.  Default ``"MBT_AEROBIC"``.
            bio_fraction: Override biological fraction of input waste (0-1).
                If None, looked up from mbt_type.
            ef_ch4_override: Override CH4 EF in g CH4 / kg waste.
            ef_n2o_override: Override N2O EF in g N2O / kg waste.
            gwp_source: GWP report edition override.
            calculation_id: Optional external calculation identifier.

        Returns:
            Dictionary with keys:
                - calculation_id, status, treatment_type ("MBT")
                - mbt_type, waste_category, waste_tonnes
                - bio_fraction_used, bio_fraction_tonnes
                - emissions_by_gas, total_co2e_kg, total_co2e_tonnes
                - calculation_trace, provenance_hash, processing_time_ms

        Raises:
            ValueError: If input validation fails.

        Example:
            >>> result = engine.calculate_mbt(
            ...     waste_tonnes=5000,
            ...     mbt_type="MBT_ANAEROBIC",
            ... )
            >>> result["total_co2e_tonnes"]
        """
        self._increment_calculations()
        start_time = time.monotonic()
        calc_id = calculation_id or f"wt_mbt_{uuid4().hex[:12]}"
        trace_steps: List[TraceStep] = []
        step_num = 0

        try:
            # -- Parse and validate inputs ------------------------------------
            m_waste = _D(waste_tonnes)
            cat_key = waste_category.upper()
            mbt_key = mbt_type.upper()
            gwp = (gwp_source or self._default_gwp_source).upper()

            errors = self._validate_mbt_inputs(m_waste, cat_key, mbt_key)
            if errors:
                raise ValueError(
                    f"MBT input validation failed: {'; '.join(errors)}"
                )

            # -- Resolve parameters -------------------------------------------
            bf = self._resolve_param(
                bio_fraction,
                _MBT_BIO_FRACTION.get(mbt_key, Decimal("0.55")),
            )

            ef_ch4 = self._resolve_param(
                ef_ch4_override,
                _MBT_EF_CH4.get(mbt_key, Decimal("2.0")),
            )
            ef_n2o = self._resolve_param(
                ef_n2o_override,
                _MBT_EF_N2O.get(mbt_key, Decimal("0.30")),
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="MBT parameters resolved",
                formula="Parameter lookup / user override",
                inputs={
                    "waste_tonnes": str(m_waste),
                    "mbt_type": mbt_key,
                    "bio_fraction": str(bf),
                    "EF_CH4": str(ef_ch4),
                    "EF_N2O": str(ef_n2o),
                },
                output="PARAMETERS_RESOLVED",
                unit="N/A",
            ))

            # -- Step 1: Calculate biological fraction mass -------------------
            m_bio = _quantize(m_waste * bf)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Biological fraction mass",
                formula="M_bio = M_waste * bio_fraction",
                inputs={
                    "M_waste": str(m_waste),
                    "bio_fraction": str(bf),
                },
                output=str(m_bio),
                unit="tonnes",
            ))

            # -- Step 2: CH4 emissions ----------------------------------------
            # EF is in g CH4 / kg waste; M_bio in tonnes
            # CH4 = M_bio * EF_CH4 / 1000
            ch4_emitted = _quantize(m_bio * ef_ch4 / _THOUSAND)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="CH4 emissions from MBT biological stabilization",
                formula="CH4_mbt = M_bio * EF_CH4 / 1000",
                inputs={
                    "M_bio": str(m_bio),
                    "EF_CH4": str(ef_ch4),
                },
                output=str(ch4_emitted),
                unit="tonnes CH4",
            ))

            # -- Step 3: N2O emissions ----------------------------------------
            n2o_emitted = _quantize(m_bio * ef_n2o / _THOUSAND)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="N2O emissions from MBT biological stabilization",
                formula="N2O_mbt = M_bio * EF_N2O / 1000",
                inputs={
                    "M_bio": str(m_bio),
                    "EF_N2O": str(ef_n2o),
                },
                output=str(n2o_emitted),
                unit="tonnes N2O",
            ))

            # -- Step 4: GWP conversion ---------------------------------------
            gwp_ch4 = self._resolve_gwp("CH4", gwp)
            gwp_n2o = self._resolve_gwp("N2O", gwp)

            ch4_co2e_tonnes = _quantize(ch4_emitted * gwp_ch4)
            n2o_co2e_tonnes = _quantize(n2o_emitted * gwp_n2o)
            total_co2e_tonnes = _quantize(ch4_co2e_tonnes + n2o_co2e_tonnes)
            total_co2e_kg = _quantize(total_co2e_tonnes * _TONNES_TO_KG)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="GWP conversion to CO2e",
                formula="CO2e = CH4 * GWP_CH4 + N2O * GWP_N2O",
                inputs={
                    "CH4_emitted": str(ch4_emitted),
                    "GWP_CH4": str(gwp_ch4),
                    "N2O_emitted": str(n2o_emitted),
                    "GWP_N2O": str(gwp_n2o),
                },
                output=str(total_co2e_tonnes),
                unit="tonnes CO2e",
            ))

            # -- Build per-gas breakdown --------------------------------------
            emissions_by_gas = [
                {
                    "gas": "CH4",
                    "net_emission_tonnes": str(ch4_emitted),
                    "gwp_value": str(gwp_ch4),
                    "gwp_source": gwp,
                    "co2e_tonnes": str(ch4_co2e_tonnes),
                    "co2e_kg": str(_quantize(ch4_co2e_tonnes * _TONNES_TO_KG)),
                },
                {
                    "gas": "N2O",
                    "net_emission_tonnes": str(n2o_emitted),
                    "gwp_value": str(gwp_n2o),
                    "gwp_source": gwp,
                    "co2e_tonnes": str(n2o_co2e_tonnes),
                    "co2e_kg": str(_quantize(n2o_co2e_tonnes * _TONNES_TO_KG)),
                },
            ]

            # -- Build result -------------------------------------------------
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "treatment_type": "MBT",
                "mbt_type": mbt_key,
                "waste_category": cat_key,
                "waste_tonnes": str(m_waste),
                "bio_fraction_used": str(bf),
                "bio_fraction_tonnes": str(m_bio),
                "ef_ch4_g_per_kg": str(ef_ch4),
                "ef_n2o_g_per_kg": str(ef_n2o),
                "emissions_by_gas": emissions_by_gas,
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "gwp_source": gwp,
                "calculation_trace": [ts.to_dict() for ts in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            # -- Metrics ------------------------------------------------------
            self._record_metrics_mbt(
                mbt_key, cat_key, total_co2e_tonnes,
                ch4_emitted, n2o_emitted, m_waste, elapsed_ms,
            )

            # -- Provenance ---------------------------------------------------
            self._record_provenance(
                "calculate_mbt", calc_id,
                {
                    "mbt_type": mbt_key,
                    "waste_category": cat_key,
                    "total_co2e_tonnes": str(total_co2e_tonnes),
                    "hash": result["provenance_hash"],
                },
            )

            logger.info(
                "MBT %s: %s t waste (bio=%s t) -> %s t CO2e "
                "(%s CH4, %s N2O) in %.1fms [%s]",
                mbt_key, m_waste, m_bio, total_co2e_tonnes,
                ch4_emitted, n2o_emitted, elapsed_ms, calc_id,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            if _METRICS_AVAILABLE and _record_calculation_error is not None:
                _record_calculation_error("calculation_error")

            error_result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.ERROR.value,
                "treatment_type": "MBT",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": _utcnow().isoformat(),
            }
            error_result["provenance_hash"] = _compute_hash(error_result)

            logger.error(
                "MBT calculation failed [%s]: %s in %.1fms",
                calc_id, exc, elapsed_ms, exc_info=True,
            )
            return error_result

    # ==================================================================
    # PUBLIC API: Batch Processing
    # ==================================================================

    def calculate_biological_batch(
        self,
        treatments: List[Dict[str, Any]],
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process multiple biological treatment calculations in batch.

        Each element in ``treatments`` must include a ``treatment_type``
        field (``"COMPOSTING"``, ``"ANAEROBIC_DIGESTION"``, or ``"MBT"``)
        plus the parameters for the corresponding calculation method.

        Args:
            treatments: List of treatment calculation parameter dictionaries.
                Each must have ``treatment_type`` and the parameters accepted
                by the corresponding ``calculate_*`` method.
            gwp_source: Optional GWP override applied to all calculations.

        Returns:
            Dictionary with keys:
                - batch_id (str)
                - results (List[Dict]): Individual calculation results
                - total_co2e_kg (str)
                - total_co2e_tonnes (str)
                - success_count (int)
                - failure_count (int)
                - total_count (int)
                - summary_by_type (Dict): Aggregation by treatment type
                - processing_time_ms (float)

        Example:
            >>> batch = engine.calculate_biological_batch([
            ...     {
            ...         "treatment_type": "COMPOSTING",
            ...         "waste_tonnes": 500,
            ...         "waste_category": "FOOD_WASTE",
            ...     },
            ...     {
            ...         "treatment_type": "ANAEROBIC_DIGESTION",
            ...         "waste_tonnes": 2000,
            ...         "waste_category": "FOOD_WASTE",
            ...     },
            ...     {
            ...         "treatment_type": "MBT",
            ...         "waste_tonnes": 3000,
            ...     },
            ... ])
            >>> batch["success_count"]
            3
        """
        self._increment_batches()
        start_time = time.monotonic()
        batch_id = f"wt_bio_batch_{uuid4().hex[:12]}"

        results: List[Dict[str, Any]] = []
        total_co2e_tonnes = _ZERO
        total_co2e_kg = _ZERO
        success_count = 0
        failure_count = 0

        # Aggregation by treatment type
        summary: Dict[str, Dict[str, Decimal]] = {}

        for idx, params in enumerate(treatments):
            treatment_type = str(params.get("treatment_type", "")).upper()

            # Apply GWP override
            if gwp_source and "gwp_source" not in params:
                params["gwp_source"] = gwp_source

            # Dispatch
            if treatment_type == "COMPOSTING":
                result = self._dispatch_composting(params)
            elif treatment_type in ("ANAEROBIC_DIGESTION", "AD"):
                result = self._dispatch_ad(params)
            elif treatment_type == "MBT":
                result = self._dispatch_mbt(params)
            else:
                result = {
                    "calculation_id": f"wt_batch_{batch_id}_{idx}",
                    "status": CalculationStatus.ERROR.value,
                    "treatment_type": treatment_type or "UNKNOWN",
                    "error": (
                        f"Unknown treatment_type: '{treatment_type}'. "
                        "Valid: COMPOSTING, ANAEROBIC_DIGESTION, MBT"
                    ),
                    "error_type": "ValueError",
                    "processing_time_ms": 0.0,
                    "calculated_at": _utcnow().isoformat(),
                }
                result["provenance_hash"] = _compute_hash(result)

            results.append(result)

            if result.get("status") == CalculationStatus.SUCCESS.value:
                success_count += 1
                r_co2e_t = _safe_decimal(result.get("total_co2e_tonnes"))
                r_co2e_kg = _safe_decimal(result.get("total_co2e_kg"))
                total_co2e_tonnes += r_co2e_t
                total_co2e_kg += r_co2e_kg

                # Update summary
                tt = result.get("treatment_type", "UNKNOWN")
                if tt not in summary:
                    summary[tt] = {
                        "count": _ZERO,
                        "co2e_tonnes": _ZERO,
                        "waste_tonnes": _ZERO,
                    }
                summary[tt]["count"] += _ONE
                summary[tt]["co2e_tonnes"] += r_co2e_t
                summary[tt]["waste_tonnes"] += _safe_decimal(
                    result.get("waste_tonnes")
                )
            else:
                failure_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        # Serialize summary
        summary_serialized: Dict[str, Dict[str, str]] = {}
        for tt, data in summary.items():
            summary_serialized[tt] = {
                "count": str(data["count"]),
                "co2e_tonnes": str(_quantize(data["co2e_tonnes"])),
                "waste_tonnes": str(_quantize(data["waste_tonnes"])),
            }

        batch_result: Dict[str, Any] = {
            "batch_id": batch_id,
            "results": results,
            "total_co2e_kg": str(_quantize(total_co2e_kg)),
            "total_co2e_tonnes": str(_quantize(total_co2e_tonnes)),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": len(treatments),
            "summary_by_type": summary_serialized,
            "processing_time_ms": round(elapsed_ms, 3),
        }

        logger.info(
            "Biological batch %s: %d/%d succeeded, total=%s t CO2e in %.1fms",
            batch_id, success_count, len(treatments),
            _quantize(total_co2e_tonnes), elapsed_ms,
        )
        return batch_result

    # ==================================================================
    # PUBLIC API: Default Parameter Lookups
    # ==================================================================

    def get_bmp_default(self, waste_category: str) -> Decimal:
        """Return the default Biochemical Methane Potential for a waste type.

        BMP values represent the maximum volume of CH4 that can be produced
        per tonne of volatile solids under ideal anaerobic conditions (STP).

        Args:
            waste_category: Waste feedstock category (e.g. ``"FOOD_WASTE"``).

        Returns:
            BMP in m3 CH4 / tonne VS at STP.

        Raises:
            KeyError: If waste_category is not recognized.

        Example:
            >>> engine.get_bmp_default("FOOD_WASTE")
            Decimal('400')
        """
        key = waste_category.upper()
        if key in _BMP_DEFAULTS:
            return _BMP_DEFAULTS[key]
        raise KeyError(
            f"No default BMP for waste category '{waste_category}'. "
            f"Available: {sorted(_BMP_DEFAULTS.keys())}"
        )

    def get_biofilter_efficiency(self, biofilter_type: str) -> Decimal:
        """Return the default biofilter CH4 oxidation efficiency.

        Args:
            biofilter_type: Biofilter system type (e.g.
                ``"ENCLOSED_BIOFILTER"``).

        Returns:
            Oxidation efficiency as a fraction (0-1).

        Raises:
            KeyError: If biofilter_type is not recognized.

        Example:
            >>> engine.get_biofilter_efficiency("ENCLOSED_BIOFILTER")
            Decimal('0.65')
        """
        key = biofilter_type.upper()
        if key in _BIOFILTER_EFFICIENCY:
            return _BIOFILTER_EFFICIENCY[key]
        raise KeyError(
            f"No default efficiency for biofilter type '{biofilter_type}'. "
            f"Available: {sorted(_BIOFILTER_EFFICIENCY.keys())}"
        )

    def estimate_volatile_solids(self, waste_category: str) -> Decimal:
        """Estimate the volatile solids fraction for a waste category.

        Returns the fraction of volatile solids relative to total solids
        (not relative to wet weight).  To compute VS mass:
            M_vs = M_waste * TS_fraction * VS_fraction

        Args:
            waste_category: Waste feedstock category.

        Returns:
            VS fraction of total solids (0-1).

        Raises:
            KeyError: If waste_category is not recognized.

        Example:
            >>> engine.estimate_volatile_solids("FOOD_WASTE")
            Decimal('0.85')
        """
        key = waste_category.upper()
        if key in _VS_FRACTION_DEFAULTS:
            return _VS_FRACTION_DEFAULTS[key]
        raise KeyError(
            f"No default VS fraction for waste category '{waste_category}'. "
            f"Available: {sorted(_VS_FRACTION_DEFAULTS.keys())}"
        )

    def get_composting_ef(
        self,
        waste_category: str,
        management_quality: str = "WELL_MANAGED",
    ) -> Dict[str, Decimal]:
        """Return default composting emission factors.

        Args:
            waste_category: Waste feedstock category.
            management_quality: Management quality level.

        Returns:
            Dictionary with ``ef_ch4`` and ``ef_n2o`` in g/kg waste.

        Raises:
            KeyError: If parameters are not recognized.

        Example:
            >>> engine.get_composting_ef("FOOD_WASTE", "WELL_MANAGED")
            {'ef_ch4': Decimal('0.08'), 'ef_n2o': Decimal('0.24')}
        """
        cat = waste_category.upper()
        mgmt = management_quality.upper()

        ch4_table = _COMPOSTING_EF_CH4.get(mgmt)
        if ch4_table is None:
            raise KeyError(
                f"Unknown management_quality '{management_quality}'. "
                f"Available: {sorted(_COMPOSTING_EF_CH4.keys())}"
            )

        n2o_table = _COMPOSTING_EF_N2O.get(mgmt, {})

        ef_ch4 = ch4_table.get(cat)
        if ef_ch4 is None:
            raise KeyError(
                f"No CH4 EF for waste_category '{waste_category}' in "
                f"management quality '{management_quality}'. "
                f"Available: {sorted(ch4_table.keys())}"
            )

        ef_n2o = n2o_table.get(cat, Decimal("0.24"))

        return {"ef_ch4": ef_ch4, "ef_n2o": ef_n2o}

    def get_mbt_ef(self, mbt_type: str) -> Dict[str, Decimal]:
        """Return default MBT emission factors.

        Args:
            mbt_type: MBT configuration type.

        Returns:
            Dictionary with ``ef_ch4`` and ``ef_n2o`` in g/kg waste.

        Raises:
            KeyError: If mbt_type is not recognized.

        Example:
            >>> engine.get_mbt_ef("MBT_AEROBIC")
            {'ef_ch4': Decimal('2.0'), 'ef_n2o': Decimal('0.30')}
        """
        key = mbt_type.upper()
        if key not in _MBT_EF_CH4:
            raise KeyError(
                f"Unknown mbt_type '{mbt_type}'. "
                f"Available: {sorted(_MBT_EF_CH4.keys())}"
            )
        return {
            "ef_ch4": _MBT_EF_CH4[key],
            "ef_n2o": _MBT_EF_N2O.get(key, Decimal("0.30")),
        }

    def get_ad_defaults(self, ad_process_type: str) -> Dict[str, Decimal]:
        """Return default AD process parameters.

        Args:
            ad_process_type: AD process configuration type.

        Returns:
            Dictionary of default parameter values.

        Raises:
            KeyError: If ad_process_type is not recognized.

        Example:
            >>> engine.get_ad_defaults("WET_THERMOPHILIC")
        """
        key = ad_process_type.upper()
        if key not in _AD_DIGESTION_EFFICIENCY:
            raise KeyError(
                f"Unknown ad_process_type '{ad_process_type}'. "
                f"Available: {sorted(_AD_DIGESTION_EFFICIENCY.keys())}"
            )
        return {
            "digestion_efficiency": _AD_DIGESTION_EFFICIENCY[key],
            "methane_fraction": _AD_METHANE_FRACTION.get(key, Decimal("0.60")),
            "capture_efficiency": _AD_CAPTURE_EFFICIENCY.get(
                key, Decimal("0.98")
            ),
        }

    # ==================================================================
    # PUBLIC API: Statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine runtime statistics.

        Returns:
            Dictionary with total_calculations, total_batches,
            total_errors, created_at, uptime_seconds.
        """
        with self._lock:
            now = _utcnow()
            uptime = (now - self._created_at).total_seconds()
            return {
                "total_calculations": self._total_calculations,
                "total_batches": self._total_batches,
                "total_errors": self._total_errors,
                "created_at": self._created_at.isoformat(),
                "uptime_seconds": round(uptime, 1),
                "default_gwp_source": self._default_gwp_source,
                "available_composting_types": [
                    t.value for t in CompostingType
                ],
                "available_ad_types": [t.value for t in ADProcessType],
                "available_mbt_types": [t.value for t in MBTType],
                "available_waste_categories": [
                    c.value for c in WasteCategory
                ],
                "available_gwp_sources": sorted(_GWP_TABLE.keys()),
            }

    # ==================================================================
    # PRIVATE: Validation helpers
    # ==================================================================

    def _validate_composting_inputs(
        self,
        m_waste: Decimal,
        waste_category: str,
        composting_type: str,
    ) -> List[str]:
        """Validate composting calculation inputs.

        Args:
            m_waste: Waste mass in tonnes.
            waste_category: Waste category key.
            composting_type: Composting type key.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if m_waste <= _ZERO:
            errors.append("waste_tonnes must be > 0")

        valid_categories = {c.value for c in WasteCategory}
        if waste_category not in valid_categories:
            errors.append(
                f"Invalid waste_category '{waste_category}'. "
                f"Valid: {sorted(valid_categories)}"
            )

        valid_types = {t.value for t in CompostingType}
        if composting_type not in valid_types:
            errors.append(
                f"Invalid composting_type '{composting_type}'. "
                f"Valid: {sorted(valid_types)}"
            )

        return errors

    def _validate_ad_inputs(
        self,
        m_waste: Decimal,
        waste_category: str,
        ad_process_type: str,
    ) -> List[str]:
        """Validate anaerobic digestion calculation inputs.

        Args:
            m_waste: Waste mass in tonnes.
            waste_category: Waste category key.
            ad_process_type: AD process type key.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if m_waste <= _ZERO:
            errors.append("waste_tonnes must be > 0")

        valid_categories = {c.value for c in WasteCategory}
        if waste_category not in valid_categories:
            errors.append(
                f"Invalid waste_category '{waste_category}'. "
                f"Valid: {sorted(valid_categories)}"
            )

        valid_types = {t.value for t in ADProcessType}
        if ad_process_type not in valid_types:
            errors.append(
                f"Invalid ad_process_type '{ad_process_type}'. "
                f"Valid: {sorted(valid_types)}"
            )

        return errors

    def _validate_mbt_inputs(
        self,
        m_waste: Decimal,
        waste_category: str,
        mbt_type: str,
    ) -> List[str]:
        """Validate MBT calculation inputs.

        Args:
            m_waste: Waste mass in tonnes.
            waste_category: Waste category key.
            mbt_type: MBT type key.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if m_waste <= _ZERO:
            errors.append("waste_tonnes must be > 0")

        valid_categories = {c.value for c in WasteCategory}
        if waste_category not in valid_categories:
            errors.append(
                f"Invalid waste_category '{waste_category}'. "
                f"Valid: {sorted(valid_categories)}"
            )

        valid_types = {t.value for t in MBTType}
        if mbt_type not in valid_types:
            errors.append(
                f"Invalid mbt_type '{mbt_type}'. "
                f"Valid: {sorted(valid_types)}"
            )

        return errors

    # ==================================================================
    # PRIVATE: Parameter resolution helpers
    # ==================================================================

    def _resolve_param(
        self,
        user_value: Optional[Any],
        default: Decimal,
    ) -> Decimal:
        """Resolve a parameter, using user override or default.

        Args:
            user_value: User-supplied value (may be None).
            default: Default Decimal value.

        Returns:
            Resolved Decimal value.
        """
        if user_value is not None:
            return _D(user_value)
        return default

    def _resolve_composting_efs(
        self,
        waste_category: str,
        management_quality: str,
        ef_ch4_override: Optional[Any],
        ef_n2o_override: Optional[Any],
    ) -> Tuple[Decimal, Decimal]:
        """Resolve composting emission factors from tables or overrides.

        Args:
            waste_category: Waste category key.
            management_quality: Management quality key.
            ef_ch4_override: User-supplied CH4 EF override.
            ef_n2o_override: User-supplied N2O EF override.

        Returns:
            Tuple of (ef_ch4, ef_n2o) in g/kg waste.
        """
        # CH4
        if ef_ch4_override is not None:
            ef_ch4 = _D(ef_ch4_override)
        else:
            ch4_table = _COMPOSTING_EF_CH4.get(management_quality, {})
            ef_ch4 = ch4_table.get(waste_category, Decimal("4.0"))

        # N2O
        if ef_n2o_override is not None:
            ef_n2o = _D(ef_n2o_override)
        else:
            n2o_table = _COMPOSTING_EF_N2O.get(management_quality, {})
            ef_n2o = n2o_table.get(waste_category, Decimal("0.24"))

        return ef_ch4, ef_n2o

    def _resolve_biofilter_efficiency(
        self,
        biofilter_type: str,
        user_override: Optional[Any],
    ) -> Decimal:
        """Resolve biofilter efficiency from type lookup or user override.

        Args:
            biofilter_type: Biofilter type key.
            user_override: User-supplied efficiency value.

        Returns:
            Biofilter efficiency as Decimal (0-1).

        Raises:
            ValueError: If efficiency is out of range.
        """
        if user_override is not None:
            eff = _D(user_override)
            if eff < _ZERO or eff > _ONE:
                raise ValueError(
                    f"Biofilter efficiency must be 0.0-1.0, got {eff}"
                )
            return eff

        return _BIOFILTER_EFFICIENCY.get(biofilter_type, _ZERO)

    def _resolve_biogas_fractions(
        self,
        flare_fraction: Optional[Any],
        utilize_fraction: Optional[Any],
        vent_fraction: Optional[Any],
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Resolve and validate biogas destination fractions.

        The three fractions must sum to 1.0.  If vent_fraction is not
        specified, it is calculated as the remainder.

        Args:
            flare_fraction: Fraction sent to flare.
            utilize_fraction: Fraction utilized for energy.
            vent_fraction: Fraction intentionally vented.

        Returns:
            Tuple of (f_flare, f_utilize, f_vent).

        Raises:
            ValueError: If fractions are invalid or sum exceeds 1.0.
        """
        f_flare = _D(flare_fraction) if flare_fraction is not None else _ZERO
        f_utilize = (
            _D(utilize_fraction) if utilize_fraction is not None else _ZERO
        )

        if vent_fraction is not None:
            f_vent = _D(vent_fraction)
        else:
            # Calculate remainder
            f_vent = _quantize(_ONE - f_flare - f_utilize)

        # Validate
        for name, val in [
            ("flare_fraction", f_flare),
            ("utilize_fraction", f_utilize),
            ("vent_fraction", f_vent),
        ]:
            if val < _ZERO or val > _ONE:
                raise ValueError(
                    f"{name} must be 0.0-1.0, got {val}"
                )

        total = f_flare + f_utilize + f_vent
        if abs(total - _ONE) > Decimal("0.001"):
            raise ValueError(
                f"Biogas fractions must sum to 1.0, got {total} "
                f"(flare={f_flare}, utilize={f_utilize}, vent={f_vent})"
            )

        return f_flare, f_utilize, f_vent

    def _resolve_gwp(self, gas: str, gwp_source: str) -> Decimal:
        """Resolve GWP value from built-in table or database.

        Args:
            gas: Gas identifier (CH4, N2O, CO2).
            gwp_source: GWP report edition (AR4, AR5, AR6, AR6_20YR).

        Returns:
            GWP value as Decimal.

        Raises:
            KeyError: If GWP not found.
        """
        gas_key = gas.upper()
        source_key = gwp_source.upper()

        # Try external database first
        if self._waste_db is not None:
            try:
                return self._waste_db.get_gwp(gas_key, source_key)
            except (AttributeError, KeyError):
                pass

        # Built-in fallback
        gwp_table = _GWP_TABLE.get(source_key)
        if gwp_table is None:
            raise KeyError(
                f"Unknown GWP source '{gwp_source}'. "
                f"Available: {sorted(_GWP_TABLE.keys())}"
            )

        gwp_value = gwp_table.get(gas_key)
        if gwp_value is None:
            raise KeyError(
                f"No GWP for gas '{gas}' in source '{gwp_source}'. "
                f"Available gases: {sorted(gwp_table.keys())}"
            )

        return gwp_value

    # ==================================================================
    # PRIVATE: AD gas breakdown builder
    # ==================================================================

    def _build_ad_gas_breakdown(
        self,
        ch4_emitted: Decimal,
        n2o_emitted: Decimal,
        gwp_ch4: Decimal,
        gwp_n2o: Decimal,
        gwp_source: str,
        ch4_co2e: Decimal,
        n2o_co2e: Decimal,
    ) -> List[Dict[str, str]]:
        """Build the per-gas emissions breakdown for AD results.

        Args:
            ch4_emitted: Net CH4 emitted (tonnes).
            n2o_emitted: Net N2O emitted (tonnes).
            gwp_ch4: GWP value for CH4.
            gwp_n2o: GWP value for N2O.
            gwp_source: GWP source string.
            ch4_co2e: CH4 CO2e (tonnes).
            n2o_co2e: N2O CO2e (tonnes).

        Returns:
            List of per-gas emission dictionaries.
        """
        breakdown: List[Dict[str, str]] = [
            {
                "gas": "CH4",
                "net_emission_tonnes": str(ch4_emitted),
                "gwp_value": str(gwp_ch4),
                "gwp_source": gwp_source,
                "co2e_tonnes": str(ch4_co2e),
                "co2e_kg": str(_quantize(ch4_co2e * _TONNES_TO_KG)),
            },
        ]

        if n2o_emitted > _ZERO:
            breakdown.append({
                "gas": "N2O",
                "net_emission_tonnes": str(n2o_emitted),
                "gwp_value": str(gwp_n2o),
                "gwp_source": gwp_source,
                "co2e_tonnes": str(n2o_co2e),
                "co2e_kg": str(_quantize(n2o_co2e * _TONNES_TO_KG)),
            })

        return breakdown

    # ==================================================================
    # PRIVATE: Batch dispatch helpers
    # ==================================================================

    def _dispatch_composting(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract parameters and call calculate_composting.

        Args:
            params: Raw parameter dictionary from batch.

        Returns:
            Composting calculation result.
        """
        return self.calculate_composting(
            waste_tonnes=params.get("waste_tonnes", 0),
            waste_category=params.get("waste_category", "FOOD_WASTE"),
            composting_type=params.get("composting_type", "WINDROW"),
            management_quality=params.get("management_quality"),
            biofilter_type=params.get("biofilter_type", "NONE"),
            biofilter_efficiency=params.get("biofilter_efficiency"),
            gwp_source=params.get("gwp_source"),
            ef_ch4_override=params.get("ef_ch4_override"),
            ef_n2o_override=params.get("ef_n2o_override"),
            calculation_id=params.get("calculation_id"),
        )

    def _dispatch_ad(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract parameters and call calculate_anaerobic_digestion.

        Args:
            params: Raw parameter dictionary from batch.

        Returns:
            AD calculation result.
        """
        return self.calculate_anaerobic_digestion(
            waste_tonnes=params.get("waste_tonnes", 0),
            waste_category=params.get("waste_category", "FOOD_WASTE"),
            ad_process_type=params.get("ad_process_type", "WET_MESOPHILIC"),
            volatile_solids_fraction=params.get("volatile_solids_fraction"),
            total_solids_fraction=params.get("total_solids_fraction"),
            bmp=params.get("bmp"),
            digestion_efficiency=params.get("digestion_efficiency"),
            methane_fraction=params.get("methane_fraction"),
            capture_efficiency=params.get("capture_efficiency"),
            flare_fraction=params.get("flare_fraction"),
            utilize_fraction=params.get("utilize_fraction"),
            vent_fraction=params.get("vent_fraction"),
            flare_destruction_efficiency=params.get(
                "flare_destruction_efficiency"
            ),
            utilization_conversion_loss=params.get(
                "utilization_conversion_loss"
            ),
            digestate_n2o_ef=params.get("digestate_n2o_ef"),
            gwp_source=params.get("gwp_source"),
            calculation_id=params.get("calculation_id"),
        )

    def _dispatch_mbt(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract parameters and call calculate_mbt.

        Args:
            params: Raw parameter dictionary from batch.

        Returns:
            MBT calculation result.
        """
        return self.calculate_mbt(
            waste_tonnes=params.get("waste_tonnes", 0),
            waste_category=params.get(
                "waste_category", "MSW_ORGANIC_FRACTION"
            ),
            mbt_type=params.get("mbt_type", "MBT_AEROBIC"),
            bio_fraction=params.get("bio_fraction"),
            ef_ch4_override=params.get("ef_ch4_override"),
            ef_n2o_override=params.get("ef_n2o_override"),
            gwp_source=params.get("gwp_source"),
            calculation_id=params.get("calculation_id"),
        )

    # ==================================================================
    # PRIVATE: Metrics recording helpers
    # ==================================================================

    def _record_metrics_composting(
        self,
        composting_type: str,
        waste_category: str,
        total_co2e: Decimal,
        ch4_tonnes: Decimal,
        n2o_tonnes: Decimal,
        waste_tonnes: Decimal,
        elapsed_ms: float,
    ) -> None:
        """Record Prometheus metrics for a composting calculation.

        Args:
            composting_type: Composting type used.
            waste_category: Waste category processed.
            total_co2e: Total CO2e in tonnes.
            ch4_tonnes: CH4 emitted in tonnes.
            n2o_tonnes: N2O emitted in tonnes.
            waste_tonnes: Waste processed in tonnes.
            elapsed_ms: Processing time in milliseconds.
        """
        if not _METRICS_AVAILABLE:
            return

        cat_lower = waste_category.lower()
        if _record_calculation is not None:
            _record_calculation("composting", "emission_factor", cat_lower)
        if _observe_calculation_duration is not None:
            _observe_calculation_duration(
                "composting", "emission_factor", elapsed_ms / 1000.0,
            )
        if _record_emissions is not None:
            _record_emissions("CH4", "composting", cat_lower, float(ch4_tonnes))
            _record_emissions("N2O", "composting", cat_lower, float(n2o_tonnes))
        if _record_waste_processed is not None:
            _record_waste_processed("composting", cat_lower, float(waste_tonnes))
        if _record_biological_treatment is not None:
            _record_biological_treatment("composting")

    def _record_metrics_ad(
        self,
        ad_type: str,
        waste_category: str,
        total_co2e: Decimal,
        ch4_tonnes: Decimal,
        n2o_tonnes: Decimal,
        waste_tonnes: Decimal,
        elapsed_ms: float,
        ch4_captured: Decimal,
        ch4_fugitive: Decimal,
        ch4_vented: Decimal,
    ) -> None:
        """Record Prometheus metrics for an AD calculation.

        Args:
            ad_type: AD process type.
            waste_category: Waste category processed.
            total_co2e: Total CO2e in tonnes.
            ch4_tonnes: CH4 emitted in tonnes.
            n2o_tonnes: N2O emitted in tonnes.
            waste_tonnes: Waste processed in tonnes.
            elapsed_ms: Processing time in milliseconds.
            ch4_captured: CH4 captured in tonnes.
            ch4_fugitive: CH4 fugitive in tonnes.
            ch4_vented: CH4 vented in tonnes.
        """
        if not _METRICS_AVAILABLE:
            return

        cat_lower = waste_category.lower()
        if _record_calculation is not None:
            _record_calculation(
                "anaerobic_digestion", "emission_factor", cat_lower,
            )
        if _observe_calculation_duration is not None:
            _observe_calculation_duration(
                "anaerobic_digestion", "emission_factor", elapsed_ms / 1000.0,
            )
        if _record_emissions is not None:
            _record_emissions(
                "CH4", "anaerobic_digestion", cat_lower, float(ch4_tonnes),
            )
            if n2o_tonnes > _ZERO:
                _record_emissions(
                    "N2O", "anaerobic_digestion", cat_lower,
                    float(n2o_tonnes),
                )
        if _record_waste_processed is not None:
            _record_waste_processed(
                "anaerobic_digestion", cat_lower, float(waste_tonnes),
            )
        if _record_methane_recovery is not None:
            _record_methane_recovery("captured", float(ch4_captured))
            _record_methane_recovery("vented", float(ch4_vented))
        if _record_biological_treatment is not None:
            _record_biological_treatment("ad")

    def _record_metrics_mbt(
        self,
        mbt_type: str,
        waste_category: str,
        total_co2e: Decimal,
        ch4_tonnes: Decimal,
        n2o_tonnes: Decimal,
        waste_tonnes: Decimal,
        elapsed_ms: float,
    ) -> None:
        """Record Prometheus metrics for an MBT calculation.

        Args:
            mbt_type: MBT type used.
            waste_category: Waste category processed.
            total_co2e: Total CO2e in tonnes.
            ch4_tonnes: CH4 emitted in tonnes.
            n2o_tonnes: N2O emitted in tonnes.
            waste_tonnes: Waste processed in tonnes.
            elapsed_ms: Processing time in milliseconds.
        """
        if not _METRICS_AVAILABLE:
            return

        cat_lower = waste_category.lower()
        if _record_calculation is not None:
            _record_calculation(
                "mechanical_biological_treatment", "emission_factor", cat_lower,
            )
        if _observe_calculation_duration is not None:
            _observe_calculation_duration(
                "mechanical_biological_treatment", "emission_factor",
                elapsed_ms / 1000.0,
            )
        if _record_emissions is not None:
            _record_emissions(
                "CH4", "mechanical_biological_treatment", cat_lower,
                float(ch4_tonnes),
            )
            _record_emissions(
                "N2O", "mechanical_biological_treatment", cat_lower,
                float(n2o_tonnes),
            )
        if _record_waste_processed is not None:
            _record_waste_processed(
                "mechanical_biological_treatment", cat_lower,
                float(waste_tonnes),
            )
        if _record_biological_treatment is not None:
            _record_biological_treatment("mbt")

    # ==================================================================
    # PRIVATE: Provenance recording
    # ==================================================================

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an operation in the provenance tracker if available.

        Args:
            action: Action name (e.g. ``"calculate_composting"``).
            entity_id: Entity identifier (calculation ID).
            data: Optional data dictionary.
        """
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="biological_treatment",
                    action=action,
                    entity_id=entity_id,
                    data=data or {},
                )
            except Exception as exc:
                logger.debug(
                    "Provenance recording failed (non-critical): %s", exc,
                )
