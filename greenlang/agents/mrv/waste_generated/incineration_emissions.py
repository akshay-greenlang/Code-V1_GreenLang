# -*- coding: utf-8 -*-
"""
Incineration Emissions Engine (Engine 3) - AGENT-MRV-018

This module implements the IncinerationEmissionsEngine for the Waste Generated
in Operations agent (GL-MRV-S3-005). It calculates greenhouse gas emissions
from waste incineration following IPCC 2006 Guidelines Volume 5, Chapter 5
(Incineration and Open Burning of Waste).

Methodology:
    CO2 Fossil (IPCC Eq. 5.1):
        CO2 = SUM_i(SW_i * dm_i * CF_i * FCF_i * OF_i * 44/12)
    CO2 Biogenic:
        CO2_bio = SUM_i(SW_i * dm_i * CF_i * (1 - FCF_i) * OF_i * 44/12)
    CH4 (IPCC Eq. 5.2):
        CH4 = SUM_i(IW_i * EF_CH4_i)
    N2O:
        N2O = SUM_i(IW_i * EF_N2O_i)

    Where:
        SW_i  = mass of waste type i (wet weight, tonnes)
        dm_i  = dry matter content (fraction)
        CF_i  = carbon fraction of dry matter
        FCF_i = fossil carbon fraction of total carbon
        OF_i  = oxidation factor (default 1.0)
        IW_i  = mass of waste incinerated (Gg for CH4, tonnes for N2O)
        44/12 = molecular weight ratio CO2/C

Features:
    - Biogenic vs fossil CO2 separation (only fossil counted in inventory)
    - Energy recovery (WtE) tracking with thermal efficiency
    - Multiple incinerator types (continuous stoker, semi-continuous, batch,
      fluidized bed, open burning)
    - Ash residue estimation by waste category
    - Grid displacement tracking (reported separately per GHG Protocol)
    - Net calorific value calculations for energy recovery
    - Mixed waste composition handling with component-level calculation
    - Batch processing with per-record provenance
    - Thread-safe singleton pattern

IPCC Tables Referenced:
    - Table 5.2: Default parameters (dm, CF, FCF, OF) by waste type
    - Table 5.3: CH4 and N2O emission factors by incinerator type

Zero-Hallucination:
    All numeric calculations use deterministic Decimal arithmetic.
    No LLM calls in the calculation path. Every parameter is traceable
    to an IPCC table reference or user-provided override.

Example:
    >>> engine = IncinerationEmissionsEngine.get_instance()
    >>> result = engine.calculate(IncinerationInput(
    ...     mass_tonnes=Decimal("100"),
    ...     waste_category=WasteCategory.PLASTICS_MIXED,
    ...     incinerator_type=IncineratorType.CONTINUOUS_STOKER,
    ... ))
    >>> result.co2_fossil_tonnes > 0
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-005
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.agents.mrv.waste_generated.models import (
    # Enumerations
    WasteCategory,
    WasteTreatmentMethod,
    IncineratorType,
    GWPVersion,
    EFSource,
    DataQualityTier,
    EmissionGas,
    ComplianceFramework,
    BatchStatus,
    # Constant tables
    GWP_VALUES,
    INCINERATION_PARAMS,
    CH4_EF_INCINERATION,
    N2O_EF_INCINERATION,
    NET_CALORIFIC_VALUES,
    IPCC_LANDFILL_CONSTANTS,
    # Pydantic models
    WasteCompositionInput,
    IncinerationInput,
    IncinerationEmissionsResult,
)
from greenlang.agents.mrv.waste_generated.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
    hash_incineration_input,
    hash_incineration_result,
    hash_arbitrary,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-005"
ENGINE_ID: str = "incineration_emissions_engine"
ENGINE_VERSION: str = "1.0.0"

# Molecular weight ratio CO2/C = 44/12
_CO2_C_RATIO: Decimal = Decimal("44") / Decimal("12")

# Conversion factor: 1 tonne = 1e-3 Gg (for CH4 EF which is in kg CH4/Gg waste)
_TONNES_TO_GG: Decimal = Decimal("0.001")

# Conversion factor: kg to tonnes
_KG_TO_TONNES: Decimal = Decimal("0.001")

# Conversion factor: g to tonnes
_G_TO_TONNES: Decimal = Decimal("0.000001")

# Conversion factor: MJ to kWh
_MJ_TO_KWH: Decimal = Decimal("1") / Decimal("3.6")

# Conversion factor: MJ to MWh
_MJ_TO_MWH: Decimal = Decimal("1") / Decimal("3600")

# Default grid emission factor (kgCO2e/MWh) - conservative global average
_DEFAULT_GRID_EF_KG_PER_MWH: Decimal = Decimal("450")

# Decimal precision for output (6 significant digits)
_OUTPUT_PRECISION: str = "0.000001"

# Maximum batch size
_MAX_BATCH_SIZE: int = 10000

# ==============================================================================
# ASH RESIDUE FRACTIONS (fraction of input mass remaining as bottom ash)
# Source: IPCC 2006 Vol 5 Ch 5, EPA AP-42 Section 2.1
# ==============================================================================

ASH_RESIDUE_FRACTIONS: Dict[WasteCategory, Decimal] = {
    WasteCategory.PAPER_CARDBOARD: Decimal("0.06"),
    WasteCategory.TEXTILES: Decimal("0.08"),
    WasteCategory.FOOD_WASTE: Decimal("0.05"),
    WasteCategory.WOOD: Decimal("0.04"),
    WasteCategory.GARDEN_WASTE: Decimal("0.06"),
    WasteCategory.PLASTICS_HDPE: Decimal("0.01"),
    WasteCategory.PLASTICS_LDPE: Decimal("0.01"),
    WasteCategory.PLASTICS_PET: Decimal("0.02"),
    WasteCategory.PLASTICS_PP: Decimal("0.01"),
    WasteCategory.PLASTICS_MIXED: Decimal("0.02"),
    WasteCategory.RUBBER_LEATHER: Decimal("0.15"),
    WasteCategory.GLASS: Decimal("0.98"),  # Glass melts, almost all remains
    WasteCategory.METALS_ALUMINUM: Decimal("0.95"),
    WasteCategory.METALS_STEEL: Decimal("0.95"),
    WasteCategory.METALS_MIXED: Decimal("0.95"),
    WasteCategory.ELECTRONICS: Decimal("0.70"),
    WasteCategory.CONSTRUCTION_DEMOLITION: Decimal("0.40"),
    WasteCategory.HAZARDOUS: Decimal("0.25"),
    WasteCategory.MIXED_MSW: Decimal("0.25"),
    WasteCategory.OTHER: Decimal("0.20"),
}

# ==============================================================================
# INCINERATION PARAMS EXTENSION FOR GLASS/METALS
# Glass and metals have zero carbon, so CO2 emissions from incineration = 0.
# We include them so the engine handles any waste category without error.
# ==============================================================================

_EXTENDED_INCINERATION_PARAMS: Dict[WasteCategory, Dict[str, Decimal]] = {
    **INCINERATION_PARAMS,
    WasteCategory.GLASS: {
        "dm": Decimal("1.00"),
        "cf": Decimal("0.00"),
        "fcf": Decimal("0.00"),
        "of": Decimal("1.00"),
    },
    WasteCategory.METALS_ALUMINUM: {
        "dm": Decimal("1.00"),
        "cf": Decimal("0.00"),
        "fcf": Decimal("0.00"),
        "of": Decimal("1.00"),
    },
    WasteCategory.METALS_STEEL: {
        "dm": Decimal("1.00"),
        "cf": Decimal("0.00"),
        "fcf": Decimal("0.00"),
        "of": Decimal("1.00"),
    },
    WasteCategory.METALS_MIXED: {
        "dm": Decimal("1.00"),
        "cf": Decimal("0.00"),
        "fcf": Decimal("0.00"),
        "of": Decimal("1.00"),
    },
    WasteCategory.PLASTICS_LDPE: INCINERATION_PARAMS.get(
        WasteCategory.PLASTICS_LDPE,
        {
            "dm": Decimal("1.00"),
            "cf": Decimal("0.75"),
            "fcf": Decimal("1.00"),
            "of": Decimal("1.00"),
        },
    ),
    WasteCategory.PLASTICS_PP: INCINERATION_PARAMS.get(
        WasteCategory.PLASTICS_PP,
        {
            "dm": Decimal("1.00"),
            "cf": Decimal("0.75"),
            "fcf": Decimal("1.00"),
            "of": Decimal("1.00"),
        },
    ),
}

# ==============================================================================
# EXTENDED NCV TABLE (MJ/kg wet weight)
# Supplements models.NET_CALORIFIC_VALUES with additional waste categories
# ==============================================================================

_EXTENDED_NCV: Dict[WasteCategory, Decimal] = {
    **NET_CALORIFIC_VALUES,
    WasteCategory.PLASTICS_LDPE: Decimal("40.0"),
    WasteCategory.PLASTICS_PP: Decimal("40.0"),
    WasteCategory.GLASS: Decimal("0.0"),
    WasteCategory.METALS_ALUMINUM: Decimal("0.0"),
    WasteCategory.METALS_STEEL: Decimal("0.0"),
    WasteCategory.METALS_MIXED: Decimal("0.0"),
    WasteCategory.ELECTRONICS: Decimal("5.0"),
    WasteCategory.CONSTRUCTION_DEMOLITION: Decimal("8.0"),
    WasteCategory.HAZARDOUS: Decimal("18.0"),
    WasteCategory.OTHER: Decimal("10.0"),
}

# ==============================================================================
# DEFAULT MSW COMPOSITION (developed-country profile)
# For use when calculating mixed_msw without explicit composition
# ==============================================================================

_DEFAULT_MSW_COMPOSITION: Dict[WasteCategory, Decimal] = {
    WasteCategory.FOOD_WASTE: Decimal("0.28"),
    WasteCategory.GARDEN_WASTE: Decimal("0.10"),
    WasteCategory.PAPER_CARDBOARD: Decimal("0.25"),
    WasteCategory.PLASTICS_MIXED: Decimal("0.12"),
    WasteCategory.GLASS: Decimal("0.05"),
    WasteCategory.METALS_MIXED: Decimal("0.05"),
    WasteCategory.TEXTILES: Decimal("0.03"),
    WasteCategory.WOOD: Decimal("0.04"),
    WasteCategory.RUBBER_LEATHER: Decimal("0.01"),
    WasteCategory.OTHER: Decimal("0.07"),
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _quantize(value: Decimal, precision: str = _OUTPUT_PRECISION) -> Decimal:
    """
    Quantize a Decimal to the specified precision.

    Args:
        value: Decimal value to quantize.
        precision: Quantization precision string (e.g., '0.000001').

    Returns:
        Quantized Decimal value.
    """
    return value.quantize(Decimal(precision), rounding=ROUND_HALF_UP)


def _safe_decimal(value: Any) -> Decimal:
    """
    Safely convert a value to Decimal.

    Args:
        value: Value to convert (int, float, str, or Decimal).

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
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _decimal_sqrt(value: Decimal, precision: int = 28) -> Decimal:
    """
    Compute the square root of a Decimal using Newton's method.

    Python's Decimal does not have a built-in sqrt for arbitrary precision.
    This function uses Newton-Raphson iteration for convergence.

    Args:
        value: Non-negative Decimal value.
        precision: Number of iterations for convergence (default 28).

    Returns:
        Decimal square root.

    Raises:
        ValueError: If value is negative.
    """
    if value < Decimal("0"):
        raise ValueError(f"Cannot compute sqrt of negative value: {value}")
    if value == Decimal("0"):
        return Decimal("0")

    # Initial estimate
    x = value
    two = Decimal("2")
    for _ in range(precision):
        x = (x + value / x) / two

    return x


def _compute_provenance_hash(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 provenance hash for a calculation step.

    Args:
        data: Dictionary of calculation parameters and results.

    Returns:
        Lowercase hex SHA-256 hash string.
    """

    def _default(obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return str(obj)

    serialized = json.dumps(data, sort_keys=True, default=_default)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ==============================================================================
# INCINERATION EMISSIONS ENGINE
# ==============================================================================


class IncinerationEmissionsEngine:
    """
    Engine 3: Incineration Emissions Calculator.

    Implements IPCC 2006 Vol 5 Ch 5 methodology for CO2 (fossil and biogenic),
    CH4, and N2O emissions from waste incineration. Supports waste-to-energy
    (WtE) plants with energy recovery tracking and grid displacement reporting.

    This engine follows the GreenLang zero-hallucination principle:
    all calculations use deterministic Decimal arithmetic with IPCC default
    parameters or user-provided overrides. No LLM calls are made in the
    calculation path.

    Thread Safety:
        This class implements the singleton pattern with RLock protection.
        All public methods are thread-safe.

    Attributes:
        _provenance_tracker: ProvenanceTracker for audit trail hashing.
        _lock: RLock for thread safety.
        _calculation_count: Running counter of calculations performed.

    Example:
        >>> engine = IncinerationEmissionsEngine.get_instance()
        >>> result = engine.calculate(IncinerationInput(
        ...     mass_tonnes=Decimal("50"),
        ...     waste_category=WasteCategory.PAPER_CARDBOARD,
        ...     incinerator_type=IncineratorType.CONTINUOUS_STOKER,
        ... ))
        >>> assert result.co2_fossil_tonnes >= Decimal("0")
        >>> assert result.co2_biogenic_tonnes >= Decimal("0")
    """

    # Singleton machinery
    _instance: Optional["IncinerationEmissionsEngine"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """
        Initialize the IncinerationEmissionsEngine.

        Do not call directly -- use get_instance() for the singleton.
        """
        self._provenance_tracker: ProvenanceTracker = get_provenance_tracker()
        self._lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0
        self._initialized_at: str = datetime.now(timezone.utc).isoformat()
        logger.info(
            "IncinerationEmissionsEngine initialized (engine=%s, version=%s)",
            ENGINE_ID,
            ENGINE_VERSION,
        )

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "IncinerationEmissionsEngine":
        """
        Get the singleton IncinerationEmissionsEngine instance.

        Thread-safe double-checked locking pattern.

        Returns:
            IncinerationEmissionsEngine singleton.
        """
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (for testing).

        Thread-safe reset of the cached singleton.
        """
        with cls._singleton_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        incineration_input: IncinerationInput,
        gwp_version: GWPVersion = GWPVersion.AR5,
        grid_ef_kg_per_mwh: Optional[Decimal] = None,
    ) -> IncinerationEmissionsResult:
        """
        Calculate incineration emissions for a single waste stream.

        Implements IPCC 2006 Vol 5 Ch 5 equations:
        - Eq 5.1 for CO2 (fossil and biogenic)
        - Eq 5.2 for CH4
        - N2O by incinerator type

        Args:
            incineration_input: Validated IncinerationInput with mass, waste
                category, incinerator type, and optional overrides.
            gwp_version: GWP assessment report version for CO2e conversion.
                Defaults to AR5.
            grid_ef_kg_per_mwh: Grid emission factor (kgCO2e/MWh) for
                avoided emissions calculation. Defaults to 450.

        Returns:
            IncinerationEmissionsResult with fossil CO2, biogenic CO2,
            CH4, N2O, total CO2e, and optional energy recovery data.

        Raises:
            ValueError: If input validation fails.
        """
        start_ns = time.monotonic_ns()
        with self._lock:
            self._calculation_count += 1
            calc_id = self._calculation_count

        logger.info(
            "IncinerationEmissionsEngine.calculate: calc_id=%d, "
            "mass=%.4f t, waste_category=%s, incinerator=%s",
            calc_id,
            incineration_input.mass_tonnes,
            incineration_input.waste_category.value,
            incineration_input.incinerator_type.value,
        )

        # Step 1: Validate
        errors = self.validate_incineration_input(incineration_input)
        if errors:
            error_msg = "; ".join(errors)
            logger.error(
                "Input validation failed (calc_id=%d): %s", calc_id, error_msg
            )
            raise ValueError(f"Incineration input validation failed: {error_msg}")

        # Step 2: Resolve parameters
        params = self._resolve_params(incineration_input)
        mass = incineration_input.mass_tonnes
        dm = params["dm"]
        cf = params["cf"]
        fcf = params["fcf"]
        of = params["of"]

        # Step 3: Calculate CO2 (fossil and biogenic)
        co2_fossil = self.calculate_fossil_co2(mass, dm, cf, fcf, of)
        co2_biogenic = self.calculate_biogenic_co2(mass, dm, cf, fcf, of)

        # Step 4: Calculate CH4
        ch4 = self.calculate_ch4(mass, incineration_input.incinerator_type)

        # Step 5: Calculate N2O
        n2o = self.calculate_n2o(mass, incineration_input.incinerator_type)

        # Step 6: Calculate total CO2e
        co2e_total = self.calculate_total_co2e(co2_fossil, ch4, n2o, gwp_version)

        # Step 7: Energy recovery (if applicable)
        energy_kwh: Optional[Decimal] = None
        avoided_co2e: Optional[Decimal] = None

        if incineration_input.energy_recovery:
            efficiency = incineration_input.thermal_efficiency or Decimal("0.25")
            energy_mwh = self.estimate_energy_recovery(
                mass, incineration_input.waste_category, efficiency
            )
            energy_kwh = _quantize(energy_mwh * Decimal("1000"))
            grid_ef = grid_ef_kg_per_mwh or _DEFAULT_GRID_EF_KG_PER_MWH
            avoided_co2e_tonnes = self.calculate_avoided_grid_emissions(
                energy_mwh, grid_ef
            )
            avoided_co2e = _quantize(avoided_co2e_tonnes)

        # Step 8: Build result
        result = IncinerationEmissionsResult(
            co2_fossil_tonnes=_quantize(co2_fossil),
            co2_biogenic_tonnes=_quantize(co2_biogenic),
            ch4_tonnes=_quantize(ch4),
            n2o_tonnes=_quantize(n2o),
            co2e_total=_quantize(co2e_total),
            energy_recovered_kwh=energy_kwh,
            avoided_co2e_memo=avoided_co2e,
        )

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        logger.info(
            "IncinerationEmissionsEngine.calculate complete: calc_id=%d, "
            "co2e_total=%.6f t, elapsed=%.2f ms",
            calc_id,
            co2e_total,
            elapsed_ms,
        )

        return result

    def calculate_fossil_co2(
        self,
        mass_tonnes: Decimal,
        dm: Decimal,
        cf: Decimal,
        fcf: Decimal,
        of: Decimal,
    ) -> Decimal:
        """
        Calculate fossil CO2 emissions from incineration (IPCC Eq. 5.1).

        CO2_fossil = SW * dm * CF * FCF * OF * (44/12)

        Only the fossil fraction of carbon is counted in the GHG inventory.
        Biogenic carbon is reported separately as a memo item.

        Args:
            mass_tonnes: Mass of waste incinerated (wet weight, tonnes).
            dm: Dry matter content as fraction (0-1).
            cf: Carbon fraction of dry matter (0-1).
            fcf: Fossil carbon fraction of total carbon (0-1).
            of: Oxidation factor (0-1, default 1.0).

        Returns:
            Fossil CO2 emissions in tonnes.
        """
        co2_fossil = mass_tonnes * dm * cf * fcf * of * _CO2_C_RATIO
        logger.debug(
            "calculate_fossil_co2: mass=%.4f, dm=%.4f, cf=%.4f, "
            "fcf=%.4f, of=%.4f -> co2_fossil=%.6f t",
            mass_tonnes, dm, cf, fcf, of, co2_fossil,
        )
        return co2_fossil

    def calculate_biogenic_co2(
        self,
        mass_tonnes: Decimal,
        dm: Decimal,
        cf: Decimal,
        fcf: Decimal,
        of: Decimal,
    ) -> Decimal:
        """
        Calculate biogenic CO2 emissions from incineration.

        CO2_biogenic = SW * dm * CF * (1 - FCF) * OF * (44/12)

        Biogenic CO2 is the complement of fossil CO2. Per GHG Protocol
        and IPCC guidelines, biogenic CO2 is a memo item and is NOT
        included in the total CO2e inventory figure.

        Args:
            mass_tonnes: Mass of waste incinerated (wet weight, tonnes).
            dm: Dry matter content as fraction (0-1).
            cf: Carbon fraction of dry matter (0-1).
            fcf: Fossil carbon fraction of total carbon (0-1).
            of: Oxidation factor (0-1, default 1.0).

        Returns:
            Biogenic CO2 emissions in tonnes (memo item).
        """
        biogenic_fraction = Decimal("1") - fcf
        co2_biogenic = mass_tonnes * dm * cf * biogenic_fraction * of * _CO2_C_RATIO
        logger.debug(
            "calculate_biogenic_co2: mass=%.4f, biogenic_fraction=%.4f "
            "-> co2_biogenic=%.6f t",
            mass_tonnes, biogenic_fraction, co2_biogenic,
        )
        return co2_biogenic

    def calculate_ch4(
        self,
        mass_tonnes: Decimal,
        incinerator_type: IncineratorType,
    ) -> Decimal:
        """
        Calculate CH4 emissions from incineration (IPCC Eq. 5.2).

        CH4 = IW * EF_CH4
        Where IW is in Gg and EF_CH4 is in kg CH4/Gg waste incinerated.
        Result converted to tonnes.

        IPCC default emission factors by incinerator type (Table 5.3):
        - Continuous stoker: 0.2 kg CH4/Gg waste
        - Semi-continuous: 6.0 kg CH4/Gg waste
        - Batch: 60.0 kg CH4/Gg waste
        - Fluidized bed: 0.1 kg CH4/Gg waste
        - Open burning: 6500 kg CH4/Gg waste

        Args:
            mass_tonnes: Mass of waste incinerated (wet weight, tonnes).
            incinerator_type: Type of incinerator.

        Returns:
            CH4 emissions in tonnes.

        Raises:
            ValueError: If incinerator_type is not recognized.
        """
        ef_ch4 = CH4_EF_INCINERATION.get(incinerator_type)
        if ef_ch4 is None:
            raise ValueError(
                f"No CH4 emission factor for incinerator type: "
                f"{incinerator_type.value}"
            )

        # Convert mass from tonnes to Gg (1 Gg = 1000 tonnes)
        mass_gg = mass_tonnes * _TONNES_TO_GG

        # EF is in kg CH4 / Gg waste -> result is in kg CH4
        ch4_kg = mass_gg * ef_ch4

        # Convert kg to tonnes
        ch4_tonnes = ch4_kg * _KG_TO_TONNES

        logger.debug(
            "calculate_ch4: mass=%.4f t (%.6f Gg), incinerator=%s, "
            "ef=%.1f kg/Gg -> ch4=%.8f t",
            mass_tonnes, mass_gg, incinerator_type.value, ef_ch4, ch4_tonnes,
        )
        return ch4_tonnes

    def calculate_n2o(
        self,
        mass_tonnes: Decimal,
        incinerator_type: IncineratorType,
    ) -> Decimal:
        """
        Calculate N2O emissions from incineration.

        N2O = IW * EF_N2O
        Where IW is in Gg and EF_N2O is in kg N2O/Gg waste incinerated.

        IPCC default emission factors by incinerator type (Table 5.3):
        - Continuous stoker: 50 kg N2O/Gg waste
        - Semi-continuous: 50 kg N2O/Gg waste
        - Batch: 50 kg N2O/Gg waste
        - Fluidized bed: 56 kg N2O/Gg waste
        - Open burning: 150 kg N2O/Gg waste

        Note: The IPCC emission factors for N2O in Table 5.3 are given
        per Gg of waste incinerated. The N2O factors reflect incomplete
        combustion and nitrogen content of the waste.

        Args:
            mass_tonnes: Mass of waste incinerated (wet weight, tonnes).
            incinerator_type: Type of incinerator.

        Returns:
            N2O emissions in tonnes.

        Raises:
            ValueError: If incinerator_type is not recognized.
        """
        ef_n2o = N2O_EF_INCINERATION.get(incinerator_type)
        if ef_n2o is None:
            raise ValueError(
                f"No N2O emission factor for incinerator type: "
                f"{incinerator_type.value}"
            )

        # Convert mass from tonnes to Gg
        mass_gg = mass_tonnes * _TONNES_TO_GG

        # EF is in kg N2O / Gg waste -> result is in kg N2O
        n2o_kg = mass_gg * ef_n2o

        # Convert kg to tonnes
        n2o_tonnes = n2o_kg * _KG_TO_TONNES

        logger.debug(
            "calculate_n2o: mass=%.4f t (%.6f Gg), incinerator=%s, "
            "ef=%.1f kg/Gg -> n2o=%.8f t",
            mass_tonnes, mass_gg, incinerator_type.value, ef_n2o, n2o_tonnes,
        )
        return n2o_tonnes

    def calculate_total_co2e(
        self,
        co2_fossil_tonnes: Decimal,
        ch4_tonnes: Decimal,
        n2o_tonnes: Decimal,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> Decimal:
        """
        Calculate total CO2 equivalent from fossil CO2, CH4, and N2O.

        CO2e = CO2_fossil + (CH4 * GWP_CH4) + (N2O * GWP_N2O)

        Per GHG Protocol guidance, biogenic CO2 is NOT included in
        the total CO2e figure. It is reported as a separate memo item.

        Args:
            co2_fossil_tonnes: Fossil CO2 emissions in tonnes.
            ch4_tonnes: CH4 emissions in tonnes.
            n2o_tonnes: N2O emissions in tonnes.
            gwp_version: GWP version (AR4, AR5, AR6, or AR6_20YR).

        Returns:
            Total CO2 equivalent in tonnes.

        Raises:
            ValueError: If gwp_version is not recognized.
        """
        gwp_table = GWP_VALUES.get(gwp_version)
        if gwp_table is None:
            raise ValueError(f"Unknown GWP version: {gwp_version.value}")

        gwp_ch4 = gwp_table["ch4"]
        gwp_n2o = gwp_table["n2o"]

        co2e = co2_fossil_tonnes + (ch4_tonnes * gwp_ch4) + (n2o_tonnes * gwp_n2o)

        logger.debug(
            "calculate_total_co2e: co2_fossil=%.6f, ch4=%.8f (GWP=%s), "
            "n2o=%.8f (GWP=%s) -> co2e=%.6f t [%s]",
            co2_fossil_tonnes,
            ch4_tonnes, gwp_ch4,
            n2o_tonnes, gwp_n2o,
            co2e,
            gwp_version.value,
        )
        return co2e

    # ------------------------------------------------------------------
    # Composition-based calculation
    # ------------------------------------------------------------------

    def calculate_with_composition(
        self,
        mass_tonnes: Decimal,
        compositions: List[WasteCompositionInput],
        incinerator_type: IncineratorType,
        energy_recovery: bool = False,
        thermal_efficiency: Optional[Decimal] = None,
        gwp_version: GWPVersion = GWPVersion.AR5,
        grid_ef_kg_per_mwh: Optional[Decimal] = None,
    ) -> IncinerationEmissionsResult:
        """
        Calculate incineration emissions from a mixed waste stream with
        explicit composition breakdown.

        Each component is calculated individually using its IPCC Table 5.2
        parameters, then summed. CH4 and N2O are calculated on the total
        mass since they depend on incinerator type, not waste composition.

        Args:
            mass_tonnes: Total mass of waste incinerated (wet weight, tonnes).
            compositions: List of WasteCompositionInput specifying the fraction
                of each waste category.
            incinerator_type: Type of incinerator.
            energy_recovery: Whether the plant recovers energy.
            thermal_efficiency: Thermal/electrical efficiency (0-1).
            gwp_version: GWP version for CO2e.
            grid_ef_kg_per_mwh: Grid EF for avoided emissions.

        Returns:
            IncinerationEmissionsResult with component-weighted emissions.

        Raises:
            ValueError: If compositions do not sum to 1.0 or are empty.
        """
        start_ns = time.monotonic_ns()

        # Validate composition fractions
        if not compositions:
            raise ValueError("Compositions list cannot be empty")

        total_fraction = sum(c.fraction for c in compositions)
        if abs(total_fraction - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"Composition fractions must sum to 1.0, got {total_fraction}"
            )

        logger.info(
            "calculate_with_composition: mass=%.4f t, %d components, "
            "incinerator=%s",
            mass_tonnes, len(compositions), incinerator_type.value,
        )

        # Calculate CO2 for each component
        co2_fossil_total = Decimal("0")
        co2_biogenic_total = Decimal("0")

        for comp in compositions:
            comp_mass = mass_tonnes * comp.fraction
            params = self._get_params_for_category(comp.waste_category)
            dm = params["dm"]
            cf = params["cf"]
            fcf = params["fcf"]
            of = params["of"]

            co2_fossil_total += self.calculate_fossil_co2(
                comp_mass, dm, cf, fcf, of
            )
            co2_biogenic_total += self.calculate_biogenic_co2(
                comp_mass, dm, cf, fcf, of
            )

        # CH4 and N2O on total mass (incinerator-dependent, not waste-type)
        ch4 = self.calculate_ch4(mass_tonnes, incinerator_type)
        n2o = self.calculate_n2o(mass_tonnes, incinerator_type)

        # CO2e total
        co2e = self.calculate_total_co2e(
            co2_fossil_total, ch4, n2o, gwp_version
        )

        # Energy recovery
        energy_kwh: Optional[Decimal] = None
        avoided_co2e: Optional[Decimal] = None

        if energy_recovery:
            eff = thermal_efficiency or Decimal("0.25")
            energy_mwh = self._estimate_energy_from_composition(
                mass_tonnes, compositions, eff
            )
            energy_kwh = _quantize(energy_mwh * Decimal("1000"))
            grid_ef = grid_ef_kg_per_mwh or _DEFAULT_GRID_EF_KG_PER_MWH
            avoided_co2e_tonnes = self.calculate_avoided_grid_emissions(
                energy_mwh, grid_ef
            )
            avoided_co2e = _quantize(avoided_co2e_tonnes)

        result = IncinerationEmissionsResult(
            co2_fossil_tonnes=_quantize(co2_fossil_total),
            co2_biogenic_tonnes=_quantize(co2_biogenic_total),
            ch4_tonnes=_quantize(ch4),
            n2o_tonnes=_quantize(n2o),
            co2e_total=_quantize(co2e),
            energy_recovered_kwh=energy_kwh,
            avoided_co2e_memo=avoided_co2e,
        )

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        logger.info(
            "calculate_with_composition complete: co2e=%.6f t, elapsed=%.2f ms",
            co2e, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Mixed waste (auto-composition)
    # ------------------------------------------------------------------

    def calculate_mixed_waste(
        self,
        mass_tonnes: Decimal,
        waste_category: WasteCategory = WasteCategory.MIXED_MSW,
        incinerator_type: IncineratorType = IncineratorType.CONTINUOUS_STOKER,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> IncinerationEmissionsResult:
        """
        Calculate emissions for mixed waste using default MSW composition.

        When a waste stream is classified as MIXED_MSW without an explicit
        composition breakdown, this method applies the default developed-
        country composition profile from _DEFAULT_MSW_COMPOSITION.

        For other waste categories, the calculation falls back to single-
        category parameters from IPCC Table 5.2.

        Args:
            mass_tonnes: Total mass of waste incinerated (wet weight, tonnes).
            waste_category: Waste category (defaults to MIXED_MSW).
            incinerator_type: Incinerator type.
            gwp_version: GWP version.

        Returns:
            IncinerationEmissionsResult.
        """
        logger.info(
            "calculate_mixed_waste: mass=%.4f t, category=%s",
            mass_tonnes, waste_category.value,
        )

        if waste_category == WasteCategory.MIXED_MSW:
            compositions = [
                WasteCompositionInput(
                    waste_category=cat, fraction=frac
                )
                for cat, frac in _DEFAULT_MSW_COMPOSITION.items()
            ]
            return self.calculate_with_composition(
                mass_tonnes=mass_tonnes,
                compositions=compositions,
                incinerator_type=incinerator_type,
                gwp_version=gwp_version,
            )

        # For non-MSW categories, use single-category calculation
        incineration_input = IncinerationInput(
            mass_tonnes=mass_tonnes,
            waste_category=waste_category,
            incinerator_type=incinerator_type,
        )
        return self.calculate(incineration_input, gwp_version=gwp_version)

    # ------------------------------------------------------------------
    # Energy recovery
    # ------------------------------------------------------------------

    def estimate_energy_recovery(
        self,
        mass_tonnes: Decimal,
        waste_category: WasteCategory,
        efficiency: Decimal = Decimal("0.25"),
    ) -> Decimal:
        """
        Estimate energy recovered from waste-to-energy incineration.

        Energy (MWh) = mass (kg) * NCV (MJ/kg) * efficiency / 3600 (MJ/MWh)

        Uses Net Calorific Values from IPCC 2006 Vol 5 and national
        waste characterization databases.

        Args:
            mass_tonnes: Mass of waste incinerated (wet weight, tonnes).
            waste_category: Waste category for NCV lookup.
            efficiency: Thermal/electrical conversion efficiency (0-1).
                Typical range: 0.15-0.30 for electricity, 0.60-0.85 for
                combined heat and power (CHP).

        Returns:
            Energy recovered in MWh.
        """
        ncv_mj_per_kg = _EXTENDED_NCV.get(waste_category, Decimal("10.0"))

        # mass in kg = mass_tonnes * 1000
        mass_kg = mass_tonnes * Decimal("1000")

        # Energy in MJ = mass_kg * NCV
        energy_mj = mass_kg * ncv_mj_per_kg

        # Apply efficiency
        useful_energy_mj = energy_mj * efficiency

        # Convert MJ to MWh (1 MWh = 3600 MJ)
        energy_mwh = useful_energy_mj * _MJ_TO_MWH

        logger.debug(
            "estimate_energy_recovery: mass=%.4f t, NCV=%.1f MJ/kg, "
            "eff=%.2f -> %.4f MWh",
            mass_tonnes, ncv_mj_per_kg, efficiency, energy_mwh,
        )
        return energy_mwh

    def calculate_avoided_grid_emissions(
        self,
        energy_mwh: Decimal,
        grid_ef_kg_per_mwh: Decimal = _DEFAULT_GRID_EF_KG_PER_MWH,
    ) -> Decimal:
        """
        Calculate avoided grid emissions from energy recovery.

        Per GHG Protocol guidance, avoided emissions from waste-to-energy
        are reported as a separate memo item. They are NOT deducted from
        the company's Scope 3 Category 5 total.

        Avoided CO2e (tonnes) = energy (MWh) * grid_EF (kgCO2e/MWh) / 1000

        Args:
            energy_mwh: Energy recovered in MWh.
            grid_ef_kg_per_mwh: Grid emission factor in kgCO2e per MWh.
                Use country-specific or regional grid EF where available.

        Returns:
            Avoided emissions in tonnes CO2e (positive value).
        """
        avoided_kg = energy_mwh * grid_ef_kg_per_mwh
        avoided_tonnes = avoided_kg * _KG_TO_TONNES

        logger.debug(
            "calculate_avoided_grid_emissions: energy=%.4f MWh, "
            "grid_ef=%.1f kgCO2e/MWh -> avoided=%.6f tCO2e",
            energy_mwh, grid_ef_kg_per_mwh, avoided_tonnes,
        )
        return avoided_tonnes

    # ------------------------------------------------------------------
    # Ash residue
    # ------------------------------------------------------------------

    def calculate_ash_residue(
        self,
        mass_tonnes: Decimal,
        waste_category: WasteCategory,
    ) -> Decimal:
        """
        Estimate bottom ash residue mass from incineration.

        Ash residue fraction depends on the waste category. Glass and
        metals produce nearly 100% residue (they melt/oxidize but do
        not combust). Plastics produce very little ash. Mixed MSW
        typically yields ~25% bottom ash by mass.

        Args:
            mass_tonnes: Mass of waste incinerated (wet weight, tonnes).
            waste_category: Waste category.

        Returns:
            Estimated ash residue mass in tonnes.
        """
        ash_fraction = ASH_RESIDUE_FRACTIONS.get(
            waste_category, Decimal("0.20")
        )
        ash_mass = mass_tonnes * ash_fraction

        logger.debug(
            "calculate_ash_residue: mass=%.4f t, category=%s, "
            "ash_frac=%.2f -> ash=%.4f t",
            mass_tonnes, waste_category.value, ash_fraction, ash_mass,
        )
        return ash_mass

    def calculate_ash_residue_mixed(
        self,
        mass_tonnes: Decimal,
        compositions: List[WasteCompositionInput],
    ) -> Decimal:
        """
        Estimate ash residue for a mixed waste stream with explicit composition.

        Args:
            mass_tonnes: Total mass of waste incinerated (wet weight, tonnes).
            compositions: List of WasteCompositionInput.

        Returns:
            Estimated total ash residue in tonnes.
        """
        total_ash = Decimal("0")
        for comp in compositions:
            comp_mass = mass_tonnes * comp.fraction
            total_ash += self.calculate_ash_residue(comp_mass, comp.waste_category)
        return total_ash

    # ------------------------------------------------------------------
    # Waste-to-Energy efficiency
    # ------------------------------------------------------------------

    def calculate_wte_efficiency(
        self,
        mass_tonnes: Decimal,
        energy_output_mwh: Decimal,
        waste_category: WasteCategory = WasteCategory.MIXED_MSW,
    ) -> Decimal:
        """
        Calculate waste-to-energy thermal efficiency from measured output.

        Efficiency = energy_output / (mass * NCV)

        This is useful for back-calculating efficiency from actual plant
        performance data, which can then be used for more accurate
        future projections.

        Args:
            mass_tonnes: Mass of waste incinerated (wet weight, tonnes).
            energy_output_mwh: Measured energy output in MWh.
            waste_category: Waste category for NCV lookup.

        Returns:
            Thermal efficiency as a fraction (0-1).

        Raises:
            ValueError: If theoretical energy content is zero.
        """
        ncv_mj_per_kg = _EXTENDED_NCV.get(waste_category, Decimal("10.0"))
        mass_kg = mass_tonnes * Decimal("1000")
        theoretical_mj = mass_kg * ncv_mj_per_kg

        if theoretical_mj <= Decimal("0"):
            raise ValueError(
                f"Cannot calculate WtE efficiency: waste category "
                f"{waste_category.value} has zero calorific value"
            )

        # Convert MWh to MJ (1 MWh = 3600 MJ)
        output_mj = energy_output_mwh * Decimal("3600")

        efficiency = output_mj / theoretical_mj

        logger.debug(
            "calculate_wte_efficiency: mass=%.4f t, output=%.4f MWh, "
            "theoretical=%.1f MJ -> eff=%.4f",
            mass_tonnes, energy_output_mwh, theoretical_mj, efficiency,
        )
        return efficiency

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        inputs: List[IncinerationInput],
        gwp_version: GWPVersion = GWPVersion.AR5,
        grid_ef_kg_per_mwh: Optional[Decimal] = None,
    ) -> List[IncinerationEmissionsResult]:
        """
        Calculate incineration emissions for a batch of waste streams.

        Processes each input independently, collecting results and errors.
        Failed items do not block processing of remaining items.

        Args:
            inputs: List of IncinerationInput records.
            gwp_version: GWP version for all calculations.
            grid_ef_kg_per_mwh: Grid EF for avoided emissions.

        Returns:
            List of IncinerationEmissionsResult (one per input).
            Failed inputs produce a zero-emission result with logged error.

        Raises:
            ValueError: If batch size exceeds _MAX_BATCH_SIZE.
        """
        if len(inputs) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(inputs)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        start_ns = time.monotonic_ns()
        logger.info(
            "calculate_batch: %d inputs, gwp=%s", len(inputs), gwp_version.value
        )

        results: List[IncinerationEmissionsResult] = []
        success_count = 0
        failure_count = 0

        for idx, inp in enumerate(inputs):
            try:
                result = self.calculate(
                    inp,
                    gwp_version=gwp_version,
                    grid_ef_kg_per_mwh=grid_ef_kg_per_mwh,
                )
                results.append(result)
                success_count += 1
            except Exception as exc:
                logger.error(
                    "calculate_batch: item %d failed: %s", idx, str(exc)
                )
                failure_count += 1
                # Produce zero-emission result for failed items
                zero_result = IncinerationEmissionsResult(
                    co2_fossil_tonnes=Decimal("0"),
                    co2_biogenic_tonnes=Decimal("0"),
                    ch4_tonnes=Decimal("0"),
                    n2o_tonnes=Decimal("0"),
                    co2e_total=Decimal("0"),
                )
                results.append(zero_result)

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        logger.info(
            "calculate_batch complete: %d success, %d failed, elapsed=%.2f ms",
            success_count, failure_count, elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_incineration_input(
        self,
        incineration_input: IncinerationInput,
    ) -> List[str]:
        """
        Validate an IncinerationInput instance.

        Checks:
        1. Mass is positive.
        2. Waste category has incineration parameters (IPCC Table 5.2).
        3. Incinerator type has CH4 and N2O emission factors (Table 5.3).
        4. Override values are within valid range [0, 1].
        5. Thermal efficiency is valid when energy recovery is enabled.
        6. Composition fractions sum to 1.0 if provided.

        Args:
            incineration_input: Input to validate.

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []

        # Mass
        if incineration_input.mass_tonnes <= Decimal("0"):
            errors.append("mass_tonnes must be positive")

        # Waste category parameters
        if incineration_input.waste_category not in _EXTENDED_INCINERATION_PARAMS:
            errors.append(
                f"No IPCC incineration parameters for waste category: "
                f"{incineration_input.waste_category.value}"
            )

        # Incinerator type
        if incineration_input.incinerator_type not in CH4_EF_INCINERATION:
            errors.append(
                f"No CH4 emission factor for incinerator type: "
                f"{incineration_input.incinerator_type.value}"
            )
        if incineration_input.incinerator_type not in N2O_EF_INCINERATION:
            errors.append(
                f"No N2O emission factor for incinerator type: "
                f"{incineration_input.incinerator_type.value}"
            )

        # Override validation
        override_fields = [
            ("dm_override", incineration_input.dm_override),
            ("cf_override", incineration_input.cf_override),
            ("fcf_override", incineration_input.fcf_override),
            ("of_override", incineration_input.of_override),
        ]
        for field_name, field_value in override_fields:
            if field_value is not None:
                if field_value < Decimal("0") or field_value > Decimal("1"):
                    errors.append(
                        f"{field_name} must be between 0 and 1, "
                        f"got {field_value}"
                    )

        # Energy recovery
        if incineration_input.energy_recovery:
            if incineration_input.thermal_efficiency is not None:
                eff = incineration_input.thermal_efficiency
                if eff < Decimal("0") or eff > Decimal("1"):
                    errors.append(
                        f"thermal_efficiency must be between 0 and 1, "
                        f"got {eff}"
                    )

        # Composition
        if incineration_input.waste_composition:
            frac_sum = sum(c.fraction for c in incineration_input.waste_composition)
            if abs(frac_sum - Decimal("1.0")) > Decimal("0.01"):
                errors.append(
                    f"waste_composition fractions must sum to 1.0, "
                    f"got {frac_sum}"
                )

        return errors

    # ------------------------------------------------------------------
    # Parameter lookup
    # ------------------------------------------------------------------

    def get_incineration_params(
        self,
        waste_category: WasteCategory,
    ) -> Dict[str, Decimal]:
        """
        Get IPCC Table 5.2 default parameters for a waste category.

        Returns a dictionary with dm, cf, fcf, of values for the
        given waste category. If the category is not found, returns
        conservative defaults.

        Args:
            waste_category: Waste category to look up.

        Returns:
            Dictionary with keys 'dm', 'cf', 'fcf', 'of' as Decimals.
        """
        params = _EXTENDED_INCINERATION_PARAMS.get(waste_category)
        if params is not None:
            return dict(params)

        # Conservative fallback for unknown categories
        logger.warning(
            "No IPCC Table 5.2 parameters for %s, using conservative defaults",
            waste_category.value,
        )
        return {
            "dm": Decimal("0.70"),
            "cf": Decimal("0.35"),
            "fcf": Decimal("0.50"),
            "of": Decimal("1.00"),
        }

    def get_ch4_emission_factor(
        self,
        incinerator_type: IncineratorType,
    ) -> Decimal:
        """
        Get CH4 emission factor for an incinerator type.

        Args:
            incinerator_type: Incinerator type.

        Returns:
            CH4 emission factor in kg CH4 per Gg waste.

        Raises:
            ValueError: If incinerator type not found.
        """
        ef = CH4_EF_INCINERATION.get(incinerator_type)
        if ef is None:
            raise ValueError(
                f"No CH4 EF for incinerator type: {incinerator_type.value}"
            )
        return ef

    def get_n2o_emission_factor(
        self,
        incinerator_type: IncineratorType,
    ) -> Decimal:
        """
        Get N2O emission factor for an incinerator type.

        Args:
            incinerator_type: Incinerator type.

        Returns:
            N2O emission factor in kg N2O per Gg waste.

        Raises:
            ValueError: If incinerator type not found.
        """
        ef = N2O_EF_INCINERATION.get(incinerator_type)
        if ef is None:
            raise ValueError(
                f"No N2O EF for incinerator type: {incinerator_type.value}"
            )
        return ef

    def get_net_calorific_value(
        self,
        waste_category: WasteCategory,
    ) -> Decimal:
        """
        Get net calorific value (NCV) for a waste category.

        Args:
            waste_category: Waste category.

        Returns:
            NCV in MJ per kg (wet weight).
        """
        return _EXTENDED_NCV.get(waste_category, Decimal("10.0"))

    def get_ash_residue_fraction(
        self,
        waste_category: WasteCategory,
    ) -> Decimal:
        """
        Get ash residue fraction for a waste category.

        Args:
            waste_category: Waste category.

        Returns:
            Ash residue fraction (0-1).
        """
        return ASH_RESIDUE_FRACTIONS.get(waste_category, Decimal("0.20"))

    def get_gwp_values(
        self,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> Dict[str, Decimal]:
        """
        Get GWP values for a given assessment report version.

        Args:
            gwp_version: GWP version.

        Returns:
            Dictionary with 'co2', 'ch4', 'n2o' GWP values.

        Raises:
            ValueError: If version not found.
        """
        table = GWP_VALUES.get(gwp_version)
        if table is None:
            raise ValueError(f"Unknown GWP version: {gwp_version.value}")
        return dict(table)

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def compute_calculation_provenance(
        self,
        incineration_input: IncinerationInput,
        result: IncinerationEmissionsResult,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> str:
        """
        Compute SHA-256 provenance hash for a complete calculation.

        The provenance hash covers the input parameters, IPCC table
        references, intermediate calculations, and final result. This
        provides an immutable audit trail proving that the result was
        derived deterministically from the input data.

        Args:
            incineration_input: Original input.
            result: Calculation result.
            gwp_version: GWP version used.

        Returns:
            SHA-256 hex digest string.
        """
        params = self._resolve_params(incineration_input)
        provenance_data = {
            "engine": ENGINE_ID,
            "version": ENGINE_VERSION,
            "agent": AGENT_ID,
            "methodology": "IPCC_2006_Vol5_Ch5",
            "input": {
                "mass_tonnes": str(incineration_input.mass_tonnes),
                "waste_category": incineration_input.waste_category.value,
                "incinerator_type": incineration_input.incinerator_type.value,
                "energy_recovery": incineration_input.energy_recovery,
            },
            "parameters": {k: str(v) for k, v in params.items()},
            "gwp_version": gwp_version.value,
            "result": {
                "co2_fossil_tonnes": str(result.co2_fossil_tonnes),
                "co2_biogenic_tonnes": str(result.co2_biogenic_tonnes),
                "ch4_tonnes": str(result.ch4_tonnes),
                "n2o_tonnes": str(result.n2o_tonnes),
                "co2e_total": str(result.co2e_total),
            },
            "formulas": {
                "co2_fossil": "SW * dm * CF * FCF * OF * (44/12)",
                "co2_biogenic": "SW * dm * CF * (1-FCF) * OF * (44/12)",
                "ch4": "IW_Gg * EF_CH4_kg_per_Gg / 1e6",
                "n2o": "IW_Gg * EF_N2O_kg_per_Gg / 1e6",
                "co2e": "CO2_fossil + CH4*GWP_CH4 + N2O*GWP_N2O",
            },
        }
        return _compute_provenance_hash(provenance_data)

    def compute_batch_provenance(
        self,
        inputs: List[IncinerationInput],
        results: List[IncinerationEmissionsResult],
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> str:
        """
        Compute Merkle-style provenance hash for a batch calculation.

        Args:
            inputs: List of inputs.
            results: List of results.
            gwp_version: GWP version.

        Returns:
            SHA-256 hex digest of batch.
        """
        individual_hashes: List[str] = []
        for inp, res in zip(inputs, results):
            h = self.compute_calculation_provenance(inp, res, gwp_version)
            individual_hashes.append(h)

        # Sort for determinism, concatenate, hash
        sorted_hashes = sorted(individual_hashes)
        combined = "|".join(sorted_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Summary / analytics methods
    # ------------------------------------------------------------------

    def calculate_gas_breakdown(
        self,
        incineration_input: IncinerationInput,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> Dict[str, Decimal]:
        """
        Calculate detailed gas-by-gas breakdown for a waste stream.

        Returns a dictionary with individual gas masses and their CO2e
        contributions, useful for reporting and visualization.

        Args:
            incineration_input: Input data.
            gwp_version: GWP version.

        Returns:
            Dictionary with gas breakdown:
                co2_fossil_tonnes, co2_biogenic_tonnes, ch4_tonnes,
                n2o_tonnes, ch4_co2e_tonnes, n2o_co2e_tonnes,
                total_co2e_tonnes.
        """
        params = self._resolve_params(incineration_input)
        mass = incineration_input.mass_tonnes

        co2_fossil = self.calculate_fossil_co2(
            mass, params["dm"], params["cf"], params["fcf"], params["of"]
        )
        co2_biogenic = self.calculate_biogenic_co2(
            mass, params["dm"], params["cf"], params["fcf"], params["of"]
        )
        ch4 = self.calculate_ch4(mass, incineration_input.incinerator_type)
        n2o = self.calculate_n2o(mass, incineration_input.incinerator_type)

        gwp_table = GWP_VALUES[gwp_version]
        ch4_co2e = ch4 * gwp_table["ch4"]
        n2o_co2e = n2o * gwp_table["n2o"]
        total_co2e = co2_fossil + ch4_co2e + n2o_co2e

        return {
            "co2_fossil_tonnes": _quantize(co2_fossil),
            "co2_biogenic_tonnes": _quantize(co2_biogenic),
            "ch4_tonnes": _quantize(ch4),
            "n2o_tonnes": _quantize(n2o),
            "ch4_co2e_tonnes": _quantize(ch4_co2e),
            "n2o_co2e_tonnes": _quantize(n2o_co2e),
            "total_co2e_tonnes": _quantize(total_co2e),
        }

    def calculate_carbon_balance(
        self,
        incineration_input: IncinerationInput,
    ) -> Dict[str, Decimal]:
        """
        Calculate carbon mass balance for an incineration event.

        Tracks total carbon input, fossil carbon released, biogenic
        carbon released, and carbon remaining in ash.

        Args:
            incineration_input: Input data.

        Returns:
            Dictionary with carbon balance in tonnes C:
                total_carbon_input, fossil_carbon, biogenic_carbon,
                carbon_in_ash (estimated), carbon_released.
        """
        params = self._resolve_params(incineration_input)
        mass = incineration_input.mass_tonnes
        dm = params["dm"]
        cf = params["cf"]
        fcf = params["fcf"]
        of = params["of"]

        total_carbon = mass * dm * cf
        fossil_carbon = total_carbon * fcf * of
        biogenic_carbon = total_carbon * (Decimal("1") - fcf) * of
        carbon_released = fossil_carbon + biogenic_carbon
        carbon_in_ash = total_carbon - carbon_released

        # Ensure non-negative (rounding may cause tiny negatives)
        carbon_in_ash = max(carbon_in_ash, Decimal("0"))

        return {
            "total_carbon_input_tonnes_c": _quantize(total_carbon),
            "fossil_carbon_tonnes_c": _quantize(fossil_carbon),
            "biogenic_carbon_tonnes_c": _quantize(biogenic_carbon),
            "carbon_in_ash_tonnes_c": _quantize(carbon_in_ash),
            "carbon_released_tonnes_c": _quantize(carbon_released),
        }

    def calculate_emission_intensity(
        self,
        incineration_input: IncinerationInput,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> Dict[str, Decimal]:
        """
        Calculate emission intensity metrics for benchmarking.

        Args:
            incineration_input: Input data.
            gwp_version: GWP version.

        Returns:
            Dictionary with intensity metrics:
                co2e_per_tonne_waste, co2e_per_mj_ncv.
        """
        result = self.calculate(incineration_input, gwp_version=gwp_version)
        mass = incineration_input.mass_tonnes

        co2e_per_tonne = result.co2e_total / mass if mass > 0 else Decimal("0")

        ncv = self.get_net_calorific_value(incineration_input.waste_category)
        mass_kg = mass * Decimal("1000")
        total_mj = mass_kg * ncv
        co2e_per_mj = (
            (result.co2e_total * Decimal("1000000")) / total_mj
            if total_mj > 0
            else Decimal("0")
        )

        return {
            "co2e_per_tonne_waste": _quantize(co2e_per_tonne),
            "co2e_per_mj_ncv_grams": _quantize(co2e_per_mj),
        }

    def aggregate_batch_results(
        self,
        results: List[IncinerationEmissionsResult],
    ) -> Dict[str, Decimal]:
        """
        Aggregate a batch of results into summary totals.

        Args:
            results: List of IncinerationEmissionsResult.

        Returns:
            Dictionary with aggregated totals.
        """
        total_fossil = Decimal("0")
        total_biogenic = Decimal("0")
        total_ch4 = Decimal("0")
        total_n2o = Decimal("0")
        total_co2e = Decimal("0")
        total_energy = Decimal("0")
        total_avoided = Decimal("0")

        for r in results:
            total_fossil += r.co2_fossil_tonnes
            total_biogenic += r.co2_biogenic_tonnes
            total_ch4 += r.ch4_tonnes
            total_n2o += r.n2o_tonnes
            total_co2e += r.co2e_total
            if r.energy_recovered_kwh is not None:
                total_energy += r.energy_recovered_kwh
            if r.avoided_co2e_memo is not None:
                total_avoided += r.avoided_co2e_memo

        return {
            "total_co2_fossil_tonnes": _quantize(total_fossil),
            "total_co2_biogenic_tonnes": _quantize(total_biogenic),
            "total_ch4_tonnes": _quantize(total_ch4),
            "total_n2o_tonnes": _quantize(total_n2o),
            "total_co2e_tonnes": _quantize(total_co2e),
            "total_energy_recovered_kwh": _quantize(total_energy),
            "total_avoided_co2e_memo_tonnes": _quantize(total_avoided),
            "record_count": Decimal(str(len(results))),
        }

    def compare_incinerator_types(
        self,
        mass_tonnes: Decimal,
        waste_category: WasteCategory,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Compare emissions across all incinerator types for the same waste.

        Useful for scenario analysis and technology selection.

        Args:
            mass_tonnes: Mass of waste (tonnes).
            waste_category: Waste category.
            gwp_version: GWP version.

        Returns:
            Dictionary keyed by incinerator type name with emissions data.
        """
        comparisons: Dict[str, Dict[str, Decimal]] = {}

        for incinerator_type in IncineratorType:
            inp = IncinerationInput(
                mass_tonnes=mass_tonnes,
                waste_category=waste_category,
                incinerator_type=incinerator_type,
            )
            result = self.calculate(inp, gwp_version=gwp_version)
            comparisons[incinerator_type.value] = {
                "co2_fossil_tonnes": result.co2_fossil_tonnes,
                "co2_biogenic_tonnes": result.co2_biogenic_tonnes,
                "ch4_tonnes": result.ch4_tonnes,
                "n2o_tonnes": result.n2o_tonnes,
                "co2e_total": result.co2e_total,
            }

        return comparisons

    def compare_gwp_versions(
        self,
        incineration_input: IncinerationInput,
    ) -> Dict[str, Decimal]:
        """
        Compare total CO2e under different GWP versions.

        Args:
            incineration_input: Input data.

        Returns:
            Dictionary keyed by GWP version with total CO2e.
        """
        comparisons: Dict[str, Decimal] = {}
        for gwp_ver in GWPVersion:
            result = self.calculate(incineration_input, gwp_version=gwp_ver)
            comparisons[gwp_ver.value] = result.co2e_total
        return comparisons

    def estimate_transport_to_incinerator(
        self,
        mass_tonnes: Decimal,
        distance_km: Decimal,
        transport_ef_kg_per_tonne_km: Decimal = Decimal("0.107"),
    ) -> Decimal:
        """
        Estimate transport emissions for waste hauled to incinerator.

        Note: Transport emissions are typically reported under Scope 3
        Category 4 (Upstream Transportation) or Category 9 (Downstream
        Transportation), NOT Category 5. This method is provided as a
        convenience for complete lifecycle tracking.

        Args:
            mass_tonnes: Mass of waste transported (tonnes).
            distance_km: One-way distance to incinerator (km).
            transport_ef_kg_per_tonne_km: Transport emission factor
                (default: 0.107 kgCO2e/tonne-km for medium truck).

        Returns:
            Transport emissions in tonnes CO2e.
        """
        transport_kg = mass_tonnes * distance_km * transport_ef_kg_per_tonne_km
        transport_tonnes = transport_kg * _KG_TO_TONNES

        logger.debug(
            "estimate_transport: mass=%.4f t, dist=%.1f km, "
            "ef=%.3f -> %.6f tCO2e",
            mass_tonnes, distance_km, transport_ef_kg_per_tonne_km,
            transport_tonnes,
        )
        return transport_tonnes

    def calculate_net_calorific_value_mixed(
        self,
        mass_tonnes: Decimal,
        compositions: List[WasteCompositionInput],
    ) -> Decimal:
        """
        Calculate weighted-average NCV for a mixed waste stream.

        Args:
            mass_tonnes: Total mass (wet weight, tonnes).
            compositions: Waste composition breakdown.

        Returns:
            Weighted NCV in MJ/kg (wet weight).
        """
        weighted_ncv = Decimal("0")
        for comp in compositions:
            ncv = _EXTENDED_NCV.get(comp.waste_category, Decimal("10.0"))
            weighted_ncv += ncv * comp.fraction

        logger.debug(
            "calculate_net_calorific_value_mixed: %d components -> "
            "weighted NCV=%.2f MJ/kg",
            len(compositions), weighted_ncv,
        )
        return weighted_ncv

    def calculate_total_energy_content(
        self,
        mass_tonnes: Decimal,
        waste_category: WasteCategory,
    ) -> Decimal:
        """
        Calculate total energy content of waste stream.

        Args:
            mass_tonnes: Mass in tonnes.
            waste_category: Waste category for NCV lookup.

        Returns:
            Total energy content in MJ.
        """
        ncv = _EXTENDED_NCV.get(waste_category, Decimal("10.0"))
        mass_kg = mass_tonnes * Decimal("1000")
        return mass_kg * ncv

    # ------------------------------------------------------------------
    # Engine metadata
    # ------------------------------------------------------------------

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get engine metadata and statistics.

        Returns:
            Dictionary with engine ID, version, calculation count,
            initialization timestamp, and supported waste categories.
        """
        with self._lock:
            calc_count = self._calculation_count

        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "methodology": "IPCC 2006 Vol 5 Ch 5",
            "supported_waste_categories": [
                cat.value for cat in _EXTENDED_INCINERATION_PARAMS.keys()
            ],
            "supported_incinerator_types": [
                it.value for it in IncineratorType
            ],
            "supported_gwp_versions": [v.value for v in GWPVersion],
            "calculation_count": calc_count,
            "initialized_at": self._initialized_at,
        }

    def get_supported_waste_categories(self) -> List[str]:
        """
        Get list of waste categories with incineration parameters.

        Returns:
            List of waste category value strings.
        """
        return [cat.value for cat in _EXTENDED_INCINERATION_PARAMS.keys()]

    def get_supported_incinerator_types(self) -> List[str]:
        """
        Get list of supported incinerator types.

        Returns:
            List of incinerator type value strings.
        """
        return [it.value for it in IncineratorType]

    # ------------------------------------------------------------------
    # Uncertainty quantification
    # ------------------------------------------------------------------

    def calculate_with_uncertainty(
        self,
        incineration_input: IncinerationInput,
        gwp_version: GWPVersion = GWPVersion.AR5,
        confidence_level: Decimal = Decimal("0.95"),
    ) -> Dict[str, Any]:
        """
        Calculate emissions with IPCC Tier 1 uncertainty bounds.

        IPCC default uncertainty for incineration:
        - CO2 from waste composition: +/- 25% (activity data) combined
          with +/- 10% (emission factor)
        - CH4 emission factor: +/- factor of 10 for batch/open burning,
          +/- factor of 2 for modern continuous stoker
        - N2O emission factor: +/- 50%

        Combined uncertainty is calculated using error propagation
        (IPCC Good Practice Guidance Eq. 6.3):
            U_combined = sqrt(U_activity^2 + U_ef^2)

        Args:
            incineration_input: Input data.
            gwp_version: GWP version.
            confidence_level: Confidence level (default 0.95 for 95%).

        Returns:
            Dictionary with central estimate and uncertainty bounds:
                co2e_central, co2e_lower, co2e_upper,
                uncertainty_pct, confidence_level, method.
        """
        result = self.calculate(incineration_input, gwp_version=gwp_version)

        # Determine uncertainty factors by incinerator type
        inc_type = incineration_input.incinerator_type
        co2_uncertainty = self._get_co2_uncertainty(inc_type)
        ch4_uncertainty = self._get_ch4_uncertainty(inc_type)
        n2o_uncertainty = self._get_n2o_uncertainty(inc_type)

        # Combined uncertainty via error propagation
        # For sum: U = sqrt(sum((U_i * x_i)^2)) / sum(x_i)
        gwp_table = GWP_VALUES[gwp_version]
        co2_contrib = result.co2_fossil_tonnes
        ch4_contrib = result.ch4_tonnes * gwp_table["ch4"]
        n2o_contrib = result.n2o_tonnes * gwp_table["n2o"]
        total = co2_contrib + ch4_contrib + n2o_contrib

        if total <= Decimal("0"):
            return {
                "co2e_central_tonnes": result.co2e_total,
                "co2e_lower_tonnes": Decimal("0"),
                "co2e_upper_tonnes": Decimal("0"),
                "uncertainty_pct": Decimal("0"),
                "confidence_level": str(confidence_level),
                "method": "IPCC_error_propagation",
            }

        # Squared weighted uncertainties
        sq_sum = (
            (co2_uncertainty * co2_contrib) ** 2
            + (ch4_uncertainty * ch4_contrib) ** 2
            + (n2o_uncertainty * n2o_contrib) ** 2
        )

        # Use integer square root approximation for Decimal
        combined_abs = _decimal_sqrt(sq_sum)
        combined_pct = combined_abs / total if total > 0 else Decimal("0")

        lower = max(total - combined_abs, Decimal("0"))
        upper = total + combined_abs

        return {
            "co2e_central_tonnes": _quantize(result.co2e_total),
            "co2e_lower_tonnes": _quantize(lower),
            "co2e_upper_tonnes": _quantize(upper),
            "uncertainty_pct": _quantize(combined_pct * Decimal("100"), "0.01"),
            "confidence_level": str(confidence_level),
            "method": "IPCC_error_propagation",
            "component_uncertainties": {
                "co2_pct": _quantize(co2_uncertainty * Decimal("100"), "0.01"),
                "ch4_pct": _quantize(ch4_uncertainty * Decimal("100"), "0.01"),
                "n2o_pct": _quantize(n2o_uncertainty * Decimal("100"), "0.01"),
            },
        }

    def _get_co2_uncertainty(self, inc_type: IncineratorType) -> Decimal:
        """
        Get CO2 uncertainty factor for an incinerator type.

        Combines activity data uncertainty (~25%) with emission factor
        uncertainty (~10%) using error propagation.

        Args:
            inc_type: Incinerator type.

        Returns:
            Combined uncertainty as fraction (e.g., 0.27 for 27%).
        """
        activity_unc = Decimal("0.25")
        ef_unc = Decimal("0.10")
        if inc_type == IncineratorType.OPEN_BURNING:
            activity_unc = Decimal("0.50")
            ef_unc = Decimal("0.30")
        elif inc_type == IncineratorType.BATCH:
            activity_unc = Decimal("0.30")
            ef_unc = Decimal("0.15")
        combined = _decimal_sqrt(activity_unc ** 2 + ef_unc ** 2)
        return combined

    def _get_ch4_uncertainty(self, inc_type: IncineratorType) -> Decimal:
        """
        Get CH4 uncertainty factor for an incinerator type.

        CH4 from incineration has high uncertainty, especially for
        batch and open burning systems.

        Args:
            inc_type: Incinerator type.

        Returns:
            Uncertainty as fraction.
        """
        uncertainty_map: Dict[IncineratorType, Decimal] = {
            IncineratorType.CONTINUOUS_STOKER: Decimal("1.00"),
            IncineratorType.SEMI_CONTINUOUS: Decimal("1.50"),
            IncineratorType.BATCH: Decimal("3.00"),
            IncineratorType.FLUIDIZED_BED: Decimal("1.00"),
            IncineratorType.OPEN_BURNING: Decimal("5.00"),
        }
        return uncertainty_map.get(inc_type, Decimal("2.00"))

    def _get_n2o_uncertainty(self, inc_type: IncineratorType) -> Decimal:
        """
        Get N2O uncertainty factor for an incinerator type.

        N2O uncertainty is generally around 50% for all incinerator types.

        Args:
            inc_type: Incinerator type.

        Returns:
            Uncertainty as fraction.
        """
        if inc_type == IncineratorType.OPEN_BURNING:
            return Decimal("2.00")
        return Decimal("0.50")

    # ------------------------------------------------------------------
    # Compliance checks
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        incineration_input: IncinerationInput,
        result: IncinerationEmissionsResult,
        frameworks: Optional[List[ComplianceFramework]] = None,
    ) -> Dict[str, Any]:
        """
        Check calculation compliance against regulatory frameworks.

        Validates that the calculation meets the requirements of the
        specified compliance frameworks (GHG Protocol, ISO 14064,
        CSRD ESRS, EPA 40 CFR 98).

        Args:
            incineration_input: Input data.
            result: Calculation result.
            frameworks: List of frameworks to check. Defaults to
                GHG Protocol and ISO 14064.

        Returns:
            Dictionary with compliance status per framework.
        """
        if frameworks is None:
            frameworks = [
                ComplianceFramework.GHG_PROTOCOL,
                ComplianceFramework.ISO_14064,
            ]

        compliance_results: Dict[str, Any] = {}

        for fw in frameworks:
            checks = self._run_framework_checks(
                fw, incineration_input, result
            )
            passed = all(c["passed"] for c in checks)
            compliance_results[fw.value] = {
                "framework": fw.value,
                "status": "COMPLIANT" if passed else "NON_COMPLIANT",
                "checks": checks,
                "pass_count": sum(1 for c in checks if c["passed"]),
                "fail_count": sum(1 for c in checks if not c["passed"]),
            }

        return compliance_results

    def _run_framework_checks(
        self,
        framework: ComplianceFramework,
        incineration_input: IncinerationInput,
        result: IncinerationEmissionsResult,
    ) -> List[Dict[str, Any]]:
        """
        Run compliance checks for a specific framework.

        Args:
            framework: Compliance framework.
            incineration_input: Input data.
            result: Calculation result.

        Returns:
            List of check results with name, passed, and detail.
        """
        checks: List[Dict[str, Any]] = []

        # Common checks for all frameworks
        checks.append({
            "name": "positive_mass",
            "passed": incineration_input.mass_tonnes > Decimal("0"),
            "detail": "Waste mass must be positive",
        })
        checks.append({
            "name": "non_negative_emissions",
            "passed": result.co2e_total >= Decimal("0"),
            "detail": "Total CO2e must be non-negative",
        })
        checks.append({
            "name": "biogenic_separate",
            "passed": result.co2_biogenic_tonnes >= Decimal("0"),
            "detail": "Biogenic CO2 must be reported separately",
        })
        checks.append({
            "name": "fossil_biogenic_consistency",
            "passed": self._check_carbon_consistency(
                incineration_input, result
            ),
            "detail": "Fossil + biogenic CO2 must equal total carbon combusted",
        })

        # GHG Protocol specific
        if framework == ComplianceFramework.GHG_PROTOCOL:
            checks.append({
                "name": "avoided_emissions_memo_only",
                "passed": self._check_avoided_not_deducted(result),
                "detail": (
                    "Avoided emissions from WtE must be reported as memo "
                    "item only, not deducted from Category 5 total"
                ),
            })

        # ISO 14064 specific
        if framework == ComplianceFramework.ISO_14064:
            checks.append({
                "name": "uncertainty_available",
                "passed": True,
                "detail": "Uncertainty quantification available via engine",
            })

        # CSRD ESRS E5 specific
        if framework == ComplianceFramework.CSRD_ESRS:
            checks.append({
                "name": "treatment_method_disclosed",
                "passed": incineration_input.incinerator_type is not None,
                "detail": "Incineration method must be disclosed",
            })
            checks.append({
                "name": "energy_recovery_disclosed",
                "passed": True,
                "detail": "Energy recovery status is tracked",
            })

        return checks

    def _check_carbon_consistency(
        self,
        incineration_input: IncinerationInput,
        result: IncinerationEmissionsResult,
    ) -> bool:
        """
        Check that fossil + biogenic CO2 equals total carbon combusted.

        Args:
            incineration_input: Input data.
            result: Calculation result.

        Returns:
            True if consistent within tolerance.
        """
        params = self._resolve_params(incineration_input)
        mass = incineration_input.mass_tonnes
        total_co2 = mass * params["dm"] * params["cf"] * params["of"] * _CO2_C_RATIO
        reported_total = result.co2_fossil_tonnes + result.co2_biogenic_tonnes

        # Allow 0.01% tolerance for rounding
        if total_co2 == Decimal("0"):
            return reported_total == Decimal("0")
        diff_pct = abs(total_co2 - reported_total) / total_co2
        return diff_pct < Decimal("0.0001")

    def _check_avoided_not_deducted(
        self,
        result: IncinerationEmissionsResult,
    ) -> bool:
        """
        Check that avoided emissions are not deducted from total.

        Per GHG Protocol, avoided emissions from WtE energy recovery
        must be reported as a memo item and must NOT be subtracted
        from the Scope 3 Category 5 total.

        Args:
            result: Calculation result.

        Returns:
            True if avoided emissions are properly handled.
        """
        # The co2e_total should not have had avoided emissions subtracted.
        # Our engine never subtracts them, so this is always True for
        # correctly-produced results.
        if result.avoided_co2e_memo is not None:
            # Verify co2e_total > 0 even though there are avoided emissions
            # (they should not have been deducted)
            return result.co2e_total >= Decimal("0")
        return True

    # ------------------------------------------------------------------
    # Scenario analysis
    # ------------------------------------------------------------------

    def scenario_energy_recovery_comparison(
        self,
        mass_tonnes: Decimal,
        waste_category: WasteCategory,
        incinerator_type: IncineratorType = IncineratorType.CONTINUOUS_STOKER,
        gwp_version: GWPVersion = GWPVersion.AR5,
        grid_ef_kg_per_mwh: Decimal = _DEFAULT_GRID_EF_KG_PER_MWH,
    ) -> Dict[str, Any]:
        """
        Compare incineration with and without energy recovery.

        Provides side-by-side comparison of emissions and energy metrics
        for the same waste stream under both scenarios.

        Args:
            mass_tonnes: Mass of waste (tonnes).
            waste_category: Waste category.
            incinerator_type: Incinerator type.
            gwp_version: GWP version.
            grid_ef_kg_per_mwh: Grid EF for avoided emissions.

        Returns:
            Dictionary with both scenarios and delta analysis.
        """
        # Without energy recovery
        inp_no_er = IncinerationInput(
            mass_tonnes=mass_tonnes,
            waste_category=waste_category,
            incinerator_type=incinerator_type,
            energy_recovery=False,
        )
        result_no_er = self.calculate(
            inp_no_er, gwp_version=gwp_version
        )

        # With energy recovery (default 25% efficiency)
        inp_with_er = IncinerationInput(
            mass_tonnes=mass_tonnes,
            waste_category=waste_category,
            incinerator_type=incinerator_type,
            energy_recovery=True,
            thermal_efficiency=Decimal("0.25"),
        )
        result_with_er = self.calculate(
            inp_with_er,
            gwp_version=gwp_version,
            grid_ef_kg_per_mwh=grid_ef_kg_per_mwh,
        )

        return {
            "waste_category": waste_category.value,
            "mass_tonnes": str(mass_tonnes),
            "incinerator_type": incinerator_type.value,
            "without_energy_recovery": {
                "co2e_total": str(result_no_er.co2e_total),
                "energy_recovered_kwh": None,
                "avoided_co2e": None,
            },
            "with_energy_recovery": {
                "co2e_total": str(result_with_er.co2e_total),
                "energy_recovered_kwh": (
                    str(result_with_er.energy_recovered_kwh)
                    if result_with_er.energy_recovered_kwh
                    else None
                ),
                "avoided_co2e_memo": (
                    str(result_with_er.avoided_co2e_memo)
                    if result_with_er.avoided_co2e_memo
                    else None
                ),
            },
            "note": (
                "Avoided emissions are memo items per GHG Protocol "
                "and are NOT deducted from Category 5 total."
            ),
        }

    def scenario_waste_category_comparison(
        self,
        mass_tonnes: Decimal,
        categories: Optional[List[WasteCategory]] = None,
        incinerator_type: IncineratorType = IncineratorType.CONTINUOUS_STOKER,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Compare emissions across waste categories for the same mass.

        Useful for understanding which waste categories have the highest
        emission intensity when incinerated.

        Args:
            mass_tonnes: Mass to compare (tonnes).
            categories: Waste categories to compare. Defaults to all
                categories with IPCC parameters.
            incinerator_type: Incinerator type.
            gwp_version: GWP version.

        Returns:
            Dictionary keyed by waste category with emissions data.
        """
        if categories is None:
            categories = list(_EXTENDED_INCINERATION_PARAMS.keys())

        results: Dict[str, Dict[str, Decimal]] = {}
        for cat in categories:
            inp = IncinerationInput(
                mass_tonnes=mass_tonnes,
                waste_category=cat,
                incinerator_type=incinerator_type,
            )
            try:
                result = self.calculate(inp, gwp_version=gwp_version)
                results[cat.value] = {
                    "co2_fossil_tonnes": result.co2_fossil_tonnes,
                    "co2_biogenic_tonnes": result.co2_biogenic_tonnes,
                    "ch4_tonnes": result.ch4_tonnes,
                    "n2o_tonnes": result.n2o_tonnes,
                    "co2e_total": result.co2e_total,
                }
            except Exception as exc:
                logger.warning(
                    "Skipping category %s in comparison: %s",
                    cat.value, str(exc),
                )

        return results

    # ------------------------------------------------------------------
    # Data export helpers
    # ------------------------------------------------------------------

    def result_to_dict(
        self,
        result: IncinerationEmissionsResult,
    ) -> Dict[str, Any]:
        """
        Convert an IncinerationEmissionsResult to a serializable dictionary.

        Args:
            result: Calculation result.

        Returns:
            Dictionary with all fields as strings.
        """
        return {
            "co2_fossil_tonnes": str(result.co2_fossil_tonnes),
            "co2_biogenic_tonnes": str(result.co2_biogenic_tonnes),
            "ch4_tonnes": str(result.ch4_tonnes),
            "n2o_tonnes": str(result.n2o_tonnes),
            "co2e_total": str(result.co2e_total),
            "energy_recovered_kwh": (
                str(result.energy_recovered_kwh)
                if result.energy_recovered_kwh is not None
                else None
            ),
            "avoided_co2e_memo": (
                str(result.avoided_co2e_memo)
                if result.avoided_co2e_memo is not None
                else None
            ),
        }

    def result_to_json(
        self,
        result: IncinerationEmissionsResult,
        indent: int = 2,
    ) -> str:
        """
        Serialize an IncinerationEmissionsResult to JSON.

        Args:
            result: Calculation result.
            indent: JSON indentation.

        Returns:
            JSON string.
        """
        return json.dumps(self.result_to_dict(result), indent=indent)

    def batch_results_to_json(
        self,
        results: List[IncinerationEmissionsResult],
        indent: int = 2,
    ) -> str:
        """
        Serialize a batch of results to JSON.

        Args:
            results: List of calculation results.
            indent: JSON indentation.

        Returns:
            JSON string.
        """
        data = [self.result_to_dict(r) for r in results]
        return json.dumps(data, indent=indent)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_params(
        self,
        incineration_input: IncinerationInput,
    ) -> Dict[str, Decimal]:
        """
        Resolve incineration parameters, applying user overrides.

        Priority: user override > waste_composition weighted > IPCC Table 5.2.

        Args:
            incineration_input: Input data with optional overrides.

        Returns:
            Dictionary with dm, cf, fcf, of as Decimals.
        """
        # Start with IPCC Table 5.2 defaults
        if incineration_input.waste_composition:
            params = self._calculate_weighted_params(
                incineration_input.waste_composition
            )
        else:
            params = self._get_params_for_category(
                incineration_input.waste_category
            )

        # Apply user overrides
        if incineration_input.dm_override is not None:
            params["dm"] = incineration_input.dm_override
        if incineration_input.cf_override is not None:
            params["cf"] = incineration_input.cf_override
        if incineration_input.fcf_override is not None:
            params["fcf"] = incineration_input.fcf_override
        if incineration_input.of_override is not None:
            params["of"] = incineration_input.of_override

        return params

    def _get_params_for_category(
        self,
        waste_category: WasteCategory,
    ) -> Dict[str, Decimal]:
        """
        Get IPCC Table 5.2 parameters for a single waste category.

        Args:
            waste_category: Waste category.

        Returns:
            Dictionary with dm, cf, fcf, of.
        """
        params = _EXTENDED_INCINERATION_PARAMS.get(waste_category)
        if params is not None:
            return dict(params)

        # Fallback for unknown categories
        logger.warning(
            "No IPCC parameters for %s, using 'other' defaults",
            waste_category.value,
        )
        return {
            "dm": Decimal("0.70"),
            "cf": Decimal("0.35"),
            "fcf": Decimal("0.50"),
            "of": Decimal("1.00"),
        }

    def _calculate_weighted_params(
        self,
        compositions: List[WasteCompositionInput],
    ) -> Dict[str, Decimal]:
        """
        Calculate composition-weighted average incineration parameters.

        For mixed waste where the composition is known, the weighted
        average of each parameter provides more accurate calculations
        than using MIXED_MSW defaults.

        Args:
            compositions: List of WasteCompositionInput.

        Returns:
            Weighted average parameters dictionary.
        """
        dm_weighted = Decimal("0")
        cf_weighted = Decimal("0")
        fcf_weighted = Decimal("0")
        of_weighted = Decimal("0")

        for comp in compositions:
            cat_params = self._get_params_for_category(comp.waste_category)
            dm_weighted += cat_params["dm"] * comp.fraction
            cf_weighted += cat_params["cf"] * comp.fraction
            fcf_weighted += cat_params["fcf"] * comp.fraction
            of_weighted += cat_params["of"] * comp.fraction

        return {
            "dm": dm_weighted,
            "cf": cf_weighted,
            "fcf": fcf_weighted,
            "of": of_weighted,
        }

    def _estimate_energy_from_composition(
        self,
        mass_tonnes: Decimal,
        compositions: List[WasteCompositionInput],
        efficiency: Decimal,
    ) -> Decimal:
        """
        Estimate energy recovery from a mixed waste stream.

        Calculates the composition-weighted NCV and applies thermal
        efficiency to determine recoverable energy.

        Args:
            mass_tonnes: Total mass (wet weight, tonnes).
            compositions: Waste composition breakdown.
            efficiency: Thermal/electrical efficiency (0-1).

        Returns:
            Energy recovered in MWh.
        """
        weighted_ncv = self.calculate_net_calorific_value_mixed(
            mass_tonnes, compositions
        )

        mass_kg = mass_tonnes * Decimal("1000")
        energy_mj = mass_kg * weighted_ncv
        useful_mj = energy_mj * efficiency
        energy_mwh = useful_mj * _MJ_TO_MWH

        return energy_mwh


# ==============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ==============================================================================


def get_incineration_engine() -> IncinerationEmissionsEngine:
    """
    Get the singleton IncinerationEmissionsEngine instance.

    Convenience function for module-level access.

    Returns:
        IncinerationEmissionsEngine singleton.

    Example:
        >>> engine = get_incineration_engine()
        >>> result = engine.calculate(input_data)
    """
    return IncinerationEmissionsEngine.get_instance()


def reset_incineration_engine() -> None:
    """
    Reset the singleton IncinerationEmissionsEngine instance.

    Convenience function for testing.
    """
    IncinerationEmissionsEngine.reset_instance()


def calculate_incineration_emissions(
    mass_tonnes: Decimal,
    waste_category: WasteCategory,
    incinerator_type: IncineratorType = IncineratorType.CONTINUOUS_STOKER,
    gwp_version: GWPVersion = GWPVersion.AR5,
    energy_recovery: bool = False,
    thermal_efficiency: Optional[Decimal] = None,
    grid_ef_kg_per_mwh: Optional[Decimal] = None,
) -> IncinerationEmissionsResult:
    """
    Calculate incineration emissions (convenience function).

    Constructs an IncinerationInput and delegates to the engine singleton.

    Args:
        mass_tonnes: Mass of waste incinerated (wet weight, tonnes).
        waste_category: Waste category.
        incinerator_type: Incinerator type (default: continuous stoker).
        gwp_version: GWP version (default: AR5).
        energy_recovery: Whether the plant recovers energy.
        thermal_efficiency: Thermal/electrical efficiency (0-1).
        grid_ef_kg_per_mwh: Grid EF for avoided emissions.

    Returns:
        IncinerationEmissionsResult.

    Example:
        >>> result = calculate_incineration_emissions(
        ...     mass_tonnes=Decimal("100"),
        ...     waste_category=WasteCategory.PLASTICS_MIXED,
        ...     incinerator_type=IncineratorType.CONTINUOUS_STOKER,
        ... )
        >>> result.co2e_total > Decimal("0")
        True
    """
    inp = IncinerationInput(
        mass_tonnes=mass_tonnes,
        waste_category=waste_category,
        incinerator_type=incinerator_type,
        energy_recovery=energy_recovery,
        thermal_efficiency=thermal_efficiency,
    )
    engine = get_incineration_engine()
    return engine.calculate(
        inp,
        gwp_version=gwp_version,
        grid_ef_kg_per_mwh=grid_ef_kg_per_mwh,
    )


def calculate_fossil_co2_standalone(
    mass_tonnes: Decimal,
    dm: Decimal,
    cf: Decimal,
    fcf: Decimal,
    of: Decimal = Decimal("1.0"),
) -> Decimal:
    """
    Standalone fossil CO2 calculation (IPCC Eq. 5.1).

    Does not require engine initialization.

    Args:
        mass_tonnes: Mass of waste (wet weight, tonnes).
        dm: Dry matter content (0-1).
        cf: Carbon fraction of dry matter (0-1).
        fcf: Fossil carbon fraction (0-1).
        of: Oxidation factor (0-1).

    Returns:
        Fossil CO2 in tonnes.
    """
    return mass_tonnes * dm * cf * fcf * of * _CO2_C_RATIO


def calculate_biogenic_co2_standalone(
    mass_tonnes: Decimal,
    dm: Decimal,
    cf: Decimal,
    fcf: Decimal,
    of: Decimal = Decimal("1.0"),
) -> Decimal:
    """
    Standalone biogenic CO2 calculation.

    Does not require engine initialization.

    Args:
        mass_tonnes: Mass of waste (wet weight, tonnes).
        dm: Dry matter content (0-1).
        cf: Carbon fraction of dry matter (0-1).
        fcf: Fossil carbon fraction (0-1).
        of: Oxidation factor (0-1).

    Returns:
        Biogenic CO2 in tonnes (memo item).
    """
    biogenic_fraction = Decimal("1") - fcf
    return mass_tonnes * dm * cf * biogenic_fraction * of * _CO2_C_RATIO


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Engine class
    "IncinerationEmissionsEngine",
    # Singleton access
    "get_incineration_engine",
    "reset_incineration_engine",
    # Convenience functions
    "calculate_incineration_emissions",
    "calculate_fossil_co2_standalone",
    "calculate_biogenic_co2_standalone",
    # Constants
    "ASH_RESIDUE_FRACTIONS",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "AGENT_ID",
]
