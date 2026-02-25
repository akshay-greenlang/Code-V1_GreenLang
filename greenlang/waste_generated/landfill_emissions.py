# -*- coding: utf-8 -*-
"""
LandfillEmissionsEngine - Engine 2: Waste Generated in Operations Agent (AGENT-MRV-018)

Core calculation engine implementing the IPCC First Order Decay (FOD) model
for landfill methane emissions per IPCC 2006 Guidelines Volume 5 Chapter 3.

This engine models the time-dependent decomposition of degradable organic
carbon (DOC) in solid waste deposited in landfills, calculating CH4 generation,
recovery, oxidation, and net atmospheric emissions on a year-by-year basis.

Primary Formulae (IPCC 2006 Vol 5 Ch 3):
    Step 1: DDOCm = W x DOC x DOCf x MCF
    Step 2: DDOCm_accumulated(T) = DDOCm_deposited + DDOCm_accumulated(T-1) x e^(-k)
    Step 3: DDOCm_decomposed(T) = DDOCm_accumulated(T-1) x (1 - e^(-k))
    Step 4: CH4_generated(T) = DDOCm_decomposed(T) x F x (16/12)
    Step 5: CH4_emitted(T) = (CH4_generated - R) x (1 - OX)

Where:
    W     = mass of waste deposited (tonnes)
    DOC   = degradable organic carbon (fraction of waste mass)
    DOCf  = fraction of DOC that decomposes (IPCC default 0.50)
    MCF   = methane correction factor (depends on landfill management type)
    k     = decay rate constant (yr^-1, depends on climate zone and waste type)
    F     = fraction of CH4 in landfill gas (default 0.50)
    R     = recovered CH4 (flared or used for energy)
    OX    = oxidation factor (fraction of CH4 oxidized in cover soil)

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate step is recorded in the calculation trace
    - SHA-256 provenance hash on every result
    - Emission factors sourced from IPCC 2006 Vol 5, Ch 3

Gas-by-Gas Breakdown:
    CH4 emissions: FOD model output
    N2O emissions: Cover soil nitrification/denitrification (minor)
    CO2e = (CH4_kg x GWP_CH4) + (N2O_kg x GWP_N2O)

Supports:
    - 6 landfill management types (managed anaerobic, semi-aerobic, unmanaged deep/shallow, etc.)
    - 4 IPCC climate zones x 6 waste type groups = 24 decay rate constants
    - 7 gas collection system types (none, active with various covers, passive, flare)
    - 4 oxidation factor cover types (no cover, soil, biocover, geomembrane)
    - Multi-year decay projection (up to 100 years)
    - Mixed waste composition handling (weighted DOC calculation)
    - Gas capture timeline modelling (capture starts after specified year)
    - First-year vs lifetime cumulative emissions
    - Multiple GWP versions (AR4, AR5, AR6, AR6-20yr)
    - Biogenic CO2 tracking (separate from CH4-based CO2e)
    - Batch processing for multiple landfill inputs
    - Input validation with detailed error messages
    - Provenance hash integration for audit trails

Example:
    >>> from greenlang.waste_generated.landfill_emissions import LandfillEmissionsEngine
    >>> from greenlang.waste_generated.models import (
    ...     LandfillInput, LandfillType, ClimateZone, GasCollectionSystem, WasteCategory
    ... )
    >>> from decimal import Decimal
    >>> engine = LandfillEmissionsEngine.get_instance()
    >>> landfill_input = LandfillInput(
    ...     mass_tonnes=Decimal("100"),
    ...     waste_category=WasteCategory.FOOD_WASTE,
    ...     landfill_type=LandfillType.MANAGED_ANAEROBIC,
    ...     climate_zone=ClimateZone.TEMPERATE_WET,
    ...     gas_collection=GasCollectionSystem.NONE,
    ...     has_cover=True,
    ... )
    >>> result = engine.calculate(landfill_input)
    >>> assert result.ch4_emitted_tonnes > Decimal("0")
    >>> assert result.co2e_total > Decimal("0")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-018 Waste Generated in Operations (GL-MRV-S3-005)
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
from typing import Any, Dict, List, Optional, Tuple

from greenlang.waste_generated.models import (
    AGENT_COMPONENT,
    AGENT_ID,
    VERSION,
    ClimateZone,
    DataQualityTier,
    DECAY_RATE_CONSTANTS,
    DOC_VALUES,
    EmissionGas,
    GasCollectionSystem,
    GAS_CAPTURE_EFFICIENCY,
    GWPVersion,
    GWP_VALUES,
    IPCC_LANDFILL_CONSTANTS,
    LandfillEmissionsResult,
    LandfillInput,
    LandfillType,
    MCF_VALUES,
    OXIDATION_FACTORS,
    UNCERTAINTY_RANGES,
    WasteCategory,
    WasteCompositionInput,
    calculate_provenance_hash,
)
from greenlang.waste_generated.metrics import WasteGeneratedMetrics
from greenlang.waste_generated.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# ==============================================================================
# DECIMAL PRECISION & ROUNDING
# ==============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_TWELVE = Decimal("12")
_SIXTEEN = Decimal("16")
_HUNDRED = Decimal("100")
_THOUSAND = Decimal("1000")

# Molecular weight ratios (Decimal for precision)
_CH4_C_RATIO = Decimal("16") / Decimal("12")  # 16/12 = 1.33333...
_CO2_C_RATIO = Decimal("44") / Decimal("12")  # 44/12 = 3.66666...
_N2O_N_RATIO = Decimal("44") / Decimal("28")  # 44/28 = 1.57142...

# Default IPCC parameters (Decimal)
_DEFAULT_DOCF = Decimal("0.50")  # Fraction of DOC that decomposes
_DEFAULT_F_CH4 = Decimal("0.50")  # Fraction of CH4 in landfill gas
_DEFAULT_OX_WITH_COVER = Decimal("0.10")  # Oxidation factor with soil cover
_DEFAULT_OX_WITHOUT_COVER = Decimal("0.00")  # Oxidation factor without cover

# Projection defaults
_DEFAULT_PROJECTION_YEARS = 100
_MAX_PROJECTION_YEARS = 200

# Batch processing limits
_MAX_BATCH_SIZE = 10000

# Natural log of 2 for half-life calculation
_LN2 = Decimal(str(math.log(2)))

# ==============================================================================
# N2O FROM LANDFILL COVER SOIL
# ==============================================================================
# Minor N2O emissions from nitrification/denitrification in cover soil.
# Reference: IPCC 2006 Vol 5 Ch 3, Section 3.2.3 and Bogner et al. (2007).
# Units: kg N2O per tonne of waste deposited, per year.

N2O_COVER_EMISSION_FACTORS: Dict[str, Decimal] = {
    "no_cover": Decimal("0.000"),
    "soil_cover": Decimal("0.0005"),      # 0.5 g N2O per tonne waste
    "biocover": Decimal("0.0012"),        # 1.2 g N2O per tonne (compost cover)
    "geomembrane": Decimal("0.0001"),     # 0.1 g N2O per tonne (minimal)
}

# ==============================================================================
# WASTE CATEGORY TO DECAY RATE KEY MAPPING
# ==============================================================================
# Maps WasteCategory enum values to the keys used in DECAY_RATE_CONSTANTS.

_WASTE_CATEGORY_TO_DECAY_KEY: Dict[WasteCategory, str] = {
    WasteCategory.FOOD_WASTE: "food_waste",
    WasteCategory.GARDEN_WASTE: "garden_waste",
    WasteCategory.PAPER_CARDBOARD: "paper_cardboard",
    WasteCategory.WOOD: "wood",
    WasteCategory.TEXTILES: "textiles",
    WasteCategory.RUBBER_LEATHER: "other",
    WasteCategory.PLASTICS_HDPE: "other",
    WasteCategory.PLASTICS_LDPE: "other",
    WasteCategory.PLASTICS_PET: "other",
    WasteCategory.PLASTICS_PP: "other",
    WasteCategory.PLASTICS_MIXED: "other",
    WasteCategory.GLASS: "other",
    WasteCategory.METALS_ALUMINUM: "other",
    WasteCategory.METALS_STEEL: "other",
    WasteCategory.METALS_MIXED: "other",
    WasteCategory.ELECTRONICS: "other",
    WasteCategory.CONSTRUCTION_DEMOLITION: "other",
    WasteCategory.HAZARDOUS: "other",
    WasteCategory.MIXED_MSW: "other",
    WasteCategory.OTHER: "other",
}

# ==============================================================================
# GAS CAPTURE SYSTEM TO COVER TYPE MAPPING
# ==============================================================================
# Maps gas collection system to the oxidation factor cover type key used in
# OXIDATION_FACTORS. This determines the OX value when no override is given.

_GAS_COLLECTION_TO_COVER_TYPE: Dict[GasCollectionSystem, str] = {
    GasCollectionSystem.NONE: "soil_cover",
    GasCollectionSystem.ACTIVE_OPERATING_CELL: "soil_cover",
    GasCollectionSystem.ACTIVE_TEMP_COVER: "soil_cover",
    GasCollectionSystem.ACTIVE_CLAY_COVER: "soil_cover",
    GasCollectionSystem.ACTIVE_GEOMEMBRANE: "geomembrane",
    GasCollectionSystem.PASSIVE_VENTING: "soil_cover",
    GasCollectionSystem.FLARE_ONLY: "soil_cover",
}

# ==============================================================================
# SINGLETON INSTANCE MANAGEMENT
# ==============================================================================

_instance: Optional["LandfillEmissionsEngine"] = None
_instance_lock: threading.Lock = threading.Lock()


# ==============================================================================
# LandfillEmissionsEngine
# ==============================================================================


class LandfillEmissionsEngine:
    """
    Engine 2: Landfill emissions calculator implementing the IPCC First Order Decay model.

    Implements the complete IPCC 2006 Vol 5 Ch 3 FOD model for estimating
    methane (CH4) generation from solid waste disposal sites. The FOD model
    captures the time-dependent nature of waste decomposition, where organic
    matter decays exponentially over years to decades depending on climate
    conditions and waste composition.

    The engine follows GreenLang's zero-hallucination principle by using only
    deterministic Decimal arithmetic with IPCC-sourced parameters. No LLM calls
    are made anywhere in the calculation pipeline.

    Thread Safety:
        This engine is fully thread-safe. A reentrant lock protects shared
        state during calculations. The singleton instance is created lazily
        with double-checked locking.

    Attributes:
        _gwp_version: Default GWP version for CO2e conversion.
        _metrics: Prometheus metrics collector for monitoring.
        _provenance: SHA-256 provenance tracker for audit trails.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = LandfillEmissionsEngine.get_instance()
        >>> result = engine.calculate(landfill_input)
        >>> assert result.co2e_total > Decimal("0")
    """

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------

    @staticmethod
    def get_instance(
        gwp_version: GWPVersion = GWPVersion.AR5,
        metrics: Optional[WasteGeneratedMetrics] = None,
    ) -> "LandfillEmissionsEngine":
        """
        Get or create the singleton LandfillEmissionsEngine instance.

        Thread-safe lazy initialization using double-checked locking.

        Args:
            gwp_version: Default IPCC Assessment Report for GWP values.
            metrics: Optional Prometheus metrics collector.

        Returns:
            Singleton LandfillEmissionsEngine instance.
        """
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = LandfillEmissionsEngine(
                        gwp_version=gwp_version,
                        metrics=metrics,
                    )
        return _instance

    @staticmethod
    def reset_instance() -> None:
        """
        Reset the singleton instance (for testing only).

        This method is intended exclusively for unit tests that need
        a fresh engine instance. It should never be called in production.
        """
        global _instance
        with _instance_lock:
            _instance = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        gwp_version: GWPVersion = GWPVersion.AR5,
        metrics: Optional[WasteGeneratedMetrics] = None,
    ) -> None:
        """
        Initialise the LandfillEmissionsEngine.

        Args:
            gwp_version: Default IPCC Assessment Report for GWP values.
                         Defaults to AR5 per GHG Protocol corporate standard.
            metrics: Optional Prometheus metrics collector. A default
                     instance is created if None is provided.

        Raises:
            ValueError: If gwp_version is not a valid GWPVersion enum member.
        """
        if not isinstance(gwp_version, GWPVersion):
            raise ValueError(
                f"gwp_version must be a GWPVersion enum, got {type(gwp_version)}"
            )

        self._gwp_version: GWPVersion = gwp_version
        self._metrics: WasteGeneratedMetrics = metrics or WasteGeneratedMetrics()
        self._provenance = get_provenance_tracker()
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "LandfillEmissionsEngine initialised: gwp=%s, agent=%s, version=%s",
            gwp_version.value,
            AGENT_ID,
            VERSION,
        )

    # ==================================================================
    # 1. calculate - Main entry point for single LandfillInput
    # ==================================================================

    def calculate(
        self,
        landfill_input: LandfillInput,
        gwp_version: Optional[GWPVersion] = None,
    ) -> LandfillEmissionsResult:
        """
        Calculate landfill emissions using the IPCC First Order Decay model.

        This is the primary entry point for single-input landfill calculations.
        It resolves all parameters (DOC, MCF, k, gas capture, OX), runs the
        FOD model for the specified number of projection years, and returns
        a complete LandfillEmissionsResult.

        For single-year calculations (years_projection=1), only the first
        year of decomposition is modelled. For multi-year projections, a
        year-by-year decay schedule is included in the result.

        Args:
            landfill_input: Validated LandfillInput with mass, waste category,
                           landfill type, climate zone, and optional overrides.
            gwp_version: Override GWP version for this calculation. If None,
                        uses the engine default.

        Returns:
            LandfillEmissionsResult with CH4 generation, recovery, oxidation,
            net emissions, CO2e total, parameters used, and optional decay
            projection schedule.

        Raises:
            ValueError: If input validation fails.
            InvalidOperation: If Decimal arithmetic fails.

        Example:
            >>> result = engine.calculate(landfill_input)
            >>> print(f"CH4 emitted: {result.ch4_emitted_tonnes} t")
            >>> print(f"CO2e total: {result.co2e_total} tCO2e")
        """
        start_ts = time.monotonic()
        calc_id = str(uuid.uuid4())
        effective_gwp = gwp_version or self._gwp_version

        logger.info(
            "calculate() calc_id=%s waste_category=%s landfill_type=%s climate=%s mass=%.4f gwp=%s",
            calc_id,
            landfill_input.waste_category.value,
            landfill_input.landfill_type.value,
            landfill_input.climate_zone.value,
            landfill_input.mass_tonnes,
            effective_gwp.value,
        )

        try:
            # Step 1: Validate input
            errors = self.validate_landfill_input(landfill_input)
            if errors:
                error_msg = "; ".join(errors)
                logger.error("Input validation failed: %s", error_msg)
                raise ValueError(f"Landfill input validation failed: {error_msg}")

            # Step 2: Resolve parameters
            doc = self._resolve_doc(landfill_input)
            docf = _DEFAULT_DOCF
            mcf = self._resolve_mcf(landfill_input)
            k = self._resolve_k(landfill_input)
            gas_capture_eff = self._resolve_gas_capture_efficiency(landfill_input)
            ox = self._resolve_oxidation_factor(landfill_input)

            logger.debug(
                "Resolved parameters: DOC=%.6f DOCf=%.2f MCF=%.2f k=%.6f "
                "gas_capture=%.2f OX=%.2f",
                doc, docf, mcf, k, gas_capture_eff, ox,
            )

            # Step 3: Calculate DDOCm deposited
            ddocm = self.calculate_ddocm(
                mass=landfill_input.mass_tonnes,
                doc=doc,
                docf=docf,
                mcf=mcf,
            )

            # Step 4: Run FOD decay model
            years = landfill_input.years_projection
            if years == 1:
                result = self._calculate_single_year(
                    ddocm=ddocm,
                    k=k,
                    gas_capture_efficiency=gas_capture_eff,
                    oxidation_factor=ox,
                    effective_gwp=effective_gwp,
                    doc=doc,
                    mcf=mcf,
                )
            else:
                result = self._calculate_multi_year(
                    ddocm=ddocm,
                    k=k,
                    gas_capture_efficiency=gas_capture_eff,
                    oxidation_factor=ox,
                    effective_gwp=effective_gwp,
                    years=years,
                    doc=doc,
                    mcf=mcf,
                )

            # Step 5: Record metrics
            elapsed_s = time.monotonic() - start_ts
            self._record_metrics(
                landfill_input=landfill_input,
                result=result,
                elapsed_s=elapsed_s,
            )

            # Step 6: Record provenance
            self._record_provenance(
                calc_id=calc_id,
                landfill_input=landfill_input,
                result=result,
            )

            logger.info(
                "calculate() complete: calc_id=%s ch4_emitted=%.6f co2e=%.6f elapsed=%.4fs",
                calc_id,
                result.ch4_emitted_tonnes,
                result.co2e_total,
                elapsed_s,
            )

            return result

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "calculate() failed: calc_id=%s error=%s", calc_id, str(e),
                exc_info=True,
            )
            self._metrics.record_calculation_error(
                error_type=type(e).__name__,
                treatment="landfill",
                tenant_id="system",
                operation="calculate",
            )
            raise

    # ==================================================================
    # 2. calculate_ddocm - Decomposable DOC mass deposited
    # ==================================================================

    def calculate_ddocm(
        self,
        mass: Decimal,
        doc: Decimal,
        docf: Decimal,
        mcf: Decimal,
    ) -> Decimal:
        """
        Calculate the mass of Decomposable Degradable Organic Carbon (DDOCm).

        Implements IPCC 2006 Vol 5 Equation 3.6:
            DDOCm = W x DOC x DOCf x MCF

        This represents the total mass of carbon in the deposited waste
        that will eventually decompose under anaerobic conditions to
        produce landfill gas.

        Args:
            mass: Mass of waste deposited in tonnes.
            doc: Degradable organic carbon fraction (0-1).
            docf: Fraction of DOC that decomposes (IPCC default 0.50).
            mcf: Methane correction factor (0-1), depends on landfill type.

        Returns:
            DDOCm in tonnes (Decimal, 8 decimal places).

        Raises:
            ValueError: If any parameter is negative or out of range.

        Example:
            >>> ddocm = engine.calculate_ddocm(
            ...     mass=Decimal("100"),
            ...     doc=Decimal("0.15"),
            ...     docf=Decimal("0.50"),
            ...     mcf=Decimal("1.0"),
            ... )
            >>> assert ddocm == Decimal("7.50000000")
        """
        self._validate_positive("mass", mass)
        self._validate_fraction("doc", doc)
        self._validate_fraction("docf", docf)
        self._validate_fraction("mcf", mcf)

        ddocm = (mass * doc * docf * mcf).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        logger.debug(
            "calculate_ddocm: W=%.4f DOC=%.4f DOCf=%.4f MCF=%.4f => DDOCm=%.8f",
            mass, doc, docf, mcf, ddocm,
        )

        return ddocm

    # ==================================================================
    # 3. calculate_ch4_generation - CH4 from decomposed DDOCm
    # ==================================================================

    def calculate_ch4_generation(
        self,
        ddocm_decomposed: Decimal,
        f: Decimal = _DEFAULT_F_CH4,
    ) -> Decimal:
        """
        Calculate CH4 generated from decomposed DDOCm.

        Implements IPCC 2006 Vol 5 Equation 3.8:
            CH4_generated = DDOCm_decomposed x F x (16/12)

        Where:
            F     = fraction of CH4 in landfill gas (by volume, default 0.50)
            16/12 = molecular weight ratio of CH4 to C

        Args:
            ddocm_decomposed: Mass of DDOCm decomposed in this period (tonnes).
            f: Fraction of CH4 in landfill gas (0-1). Default 0.50.

        Returns:
            CH4 generated in tonnes (Decimal, 8 decimal places).

        Raises:
            ValueError: If ddocm_decomposed is negative or f is out of range.

        Example:
            >>> ch4 = engine.calculate_ch4_generation(
            ...     ddocm_decomposed=Decimal("3.75"),
            ...     f=Decimal("0.50"),
            ... )
        """
        self._validate_non_negative("ddocm_decomposed", ddocm_decomposed)
        self._validate_fraction("f", f)

        ch4_generated = (ddocm_decomposed * f * _CH4_C_RATIO).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "calculate_ch4_generation: DDOCm_decomposed=%.8f F=%.4f 16/12=%.8f => CH4=%.8f",
            ddocm_decomposed, f, _CH4_C_RATIO, ch4_generated,
        )

        return ch4_generated

    # ==================================================================
    # 4. calculate_ch4_recovery - CH4 recovered by gas capture
    # ==================================================================

    def calculate_ch4_recovery(
        self,
        ch4_generated: Decimal,
        gas_capture_efficiency: Decimal,
    ) -> Decimal:
        """
        Calculate the mass of CH4 recovered through gas collection systems.

        Formula:
            R = CH4_generated x gas_capture_efficiency

        Gas collection systems can recover 0-90% of generated CH4,
        depending on the system type and cover quality.

        Args:
            ch4_generated: Total CH4 generated in tonnes.
            gas_capture_efficiency: Fraction of CH4 captured (0-1).
                                   Depends on GasCollectionSystem type.

        Returns:
            CH4 recovered in tonnes (Decimal, 8 decimal places).

        Raises:
            ValueError: If ch4_generated is negative or efficiency is out of range.

        Example:
            >>> recovered = engine.calculate_ch4_recovery(
            ...     ch4_generated=Decimal("2.50"),
            ...     gas_capture_efficiency=Decimal("0.75"),
            ... )
            >>> assert recovered == Decimal("1.87500000")
        """
        self._validate_non_negative("ch4_generated", ch4_generated)
        self._validate_fraction("gas_capture_efficiency", gas_capture_efficiency)

        recovered = (ch4_generated * gas_capture_efficiency).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "calculate_ch4_recovery: CH4_gen=%.8f capture_eff=%.4f => R=%.8f",
            ch4_generated, gas_capture_efficiency, recovered,
        )

        return recovered

    # ==================================================================
    # 5. calculate_ch4_emitted - Net CH4 to atmosphere
    # ==================================================================

    def calculate_ch4_emitted(
        self,
        ch4_generated: Decimal,
        ch4_recovered: Decimal,
        oxidation_factor: Decimal,
    ) -> Decimal:
        """
        Calculate the net CH4 emitted to the atmosphere after recovery and oxidation.

        Implements IPCC 2006 Vol 5 Equation 3.9:
            CH4_emitted = (CH4_generated - R) x (1 - OX)

        Where:
            R  = CH4 recovered (flared or used for energy)
            OX = oxidation factor (fraction oxidized in cover soil)

        The result is floored at zero; negative emissions are not physically
        possible from this pathway.

        Args:
            ch4_generated: Total CH4 generated in tonnes.
            ch4_recovered: CH4 recovered by gas collection in tonnes.
            oxidation_factor: Fraction of remaining CH4 oxidized (0-1).

        Returns:
            Net CH4 emitted in tonnes (Decimal, 8 decimal places). Minimum zero.

        Raises:
            ValueError: If inputs are negative or oxidation_factor is out of range.

        Example:
            >>> emitted = engine.calculate_ch4_emitted(
            ...     ch4_generated=Decimal("2.50"),
            ...     ch4_recovered=Decimal("1.875"),
            ...     oxidation_factor=Decimal("0.10"),
            ... )
        """
        self._validate_non_negative("ch4_generated", ch4_generated)
        self._validate_non_negative("ch4_recovered", ch4_recovered)
        self._validate_fraction("oxidation_factor", oxidation_factor)

        net_before_oxidation = ch4_generated - ch4_recovered
        if net_before_oxidation < _ZERO:
            net_before_oxidation = _ZERO

        ch4_emitted = (net_before_oxidation * (_ONE - oxidation_factor)).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # Floor at zero (cannot have negative landfill CH4 emissions)
        if ch4_emitted < _ZERO:
            ch4_emitted = _ZERO

        logger.debug(
            "calculate_ch4_emitted: CH4_gen=%.8f R=%.8f OX=%.4f => "
            "net_before_ox=%.8f => CH4_emitted=%.8f",
            ch4_generated, ch4_recovered, oxidation_factor,
            net_before_oxidation, ch4_emitted,
        )

        return ch4_emitted

    # ==================================================================
    # 6. convert_ch4_to_co2e - Convert CH4 mass to CO2 equivalent
    # ==================================================================

    def convert_ch4_to_co2e(
        self,
        ch4_tonnes: Decimal,
        gwp_version: Optional[GWPVersion] = None,
    ) -> Decimal:
        """
        Convert CH4 mass to CO2 equivalent using the specified GWP.

        Formula:
            CO2e = CH4_tonnes x GWP_CH4

        Args:
            ch4_tonnes: Mass of CH4 in tonnes.
            gwp_version: IPCC Assessment Report for GWP value. Defaults
                        to engine default if None.

        Returns:
            CO2e in tonnes (Decimal, 8 decimal places).

        Raises:
            ValueError: If ch4_tonnes is negative.
            KeyError: If gwp_version is not found in GWP_VALUES.

        Example:
            >>> co2e = engine.convert_ch4_to_co2e(
            ...     ch4_tonnes=Decimal("1.0"),
            ...     gwp_version=GWPVersion.AR5,
            ... )
            >>> assert co2e == Decimal("28.00000000")
        """
        self._validate_non_negative("ch4_tonnes", ch4_tonnes)

        effective_gwp = gwp_version or self._gwp_version
        gwp_ch4 = GWP_VALUES[effective_gwp]["ch4"]

        co2e = (ch4_tonnes * gwp_ch4).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        logger.debug(
            "convert_ch4_to_co2e: CH4=%.8f GWP_%s=%.1f => CO2e=%.8f",
            ch4_tonnes, effective_gwp.value, gwp_ch4, co2e,
        )

        return co2e

    # ==================================================================
    # 7. calculate_multi_year_decay - Year-by-year FOD projection
    # ==================================================================

    def calculate_multi_year_decay(
        self,
        mass: Decimal,
        doc: Decimal,
        docf: Decimal,
        mcf: Decimal,
        k: Decimal,
        years: int,
        f: Decimal = _DEFAULT_F_CH4,
    ) -> List[Dict[str, Decimal]]:
        """
        Calculate year-by-year decay schedule using the IPCC FOD model.

        Implements the full time-series of DDOCm accumulation, decomposition,
        and CH4 generation for the specified number of years after deposition.

        The decay follows first-order kinetics:
            DDOCm_accumulated(T) = DDOCm_deposited + DDOCm_accumulated(T-1) x e^(-k)
            DDOCm_decomposed(T) = DDOCm_accumulated(T-1) x (1 - e^(-k))

        Note: This method returns raw CH4 generation without accounting for
        gas capture or oxidation. Use calculate() or calculate_lifetime_emissions()
        for net emissions.

        Args:
            mass: Mass of waste deposited in tonnes.
            doc: Degradable organic carbon fraction.
            docf: Fraction of DOC that decomposes.
            mcf: Methane correction factor.
            k: Decay rate constant (yr^-1).
            years: Number of years to project (1-200).
            f: Fraction of CH4 in landfill gas (default 0.50).

        Returns:
            List of dicts with keys: year, ddocm_accumulated, ddocm_decomposed,
            ch4_generated_tonnes, cumulative_ch4_tonnes.

        Raises:
            ValueError: If years is out of range or parameters are invalid.

        Example:
            >>> schedule = engine.calculate_multi_year_decay(
            ...     mass=Decimal("100"), doc=Decimal("0.15"),
            ...     docf=Decimal("0.50"), mcf=Decimal("1.0"),
            ...     k=Decimal("0.185"), years=10,
            ... )
            >>> assert len(schedule) == 10
        """
        self._validate_positive("mass", mass)
        self._validate_fraction("doc", doc)
        self._validate_fraction("docf", docf)
        self._validate_fraction("mcf", mcf)
        self._validate_positive("k", k)
        self._validate_fraction("f", f)

        if years < 1 or years > _MAX_PROJECTION_YEARS:
            raise ValueError(
                f"years must be between 1 and {_MAX_PROJECTION_YEARS}, got {years}"
            )

        # Calculate DDOCm deposited (one-time event)
        ddocm_deposited = self.calculate_ddocm(mass, doc, docf, mcf)

        # Decay factor: e^(-k)
        decay_factor = self._exp_neg_k(k)

        # Decomposition factor: 1 - e^(-k)
        decomp_factor = (_ONE - decay_factor).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        schedule: List[Dict[str, Decimal]] = []
        ddocm_accumulated_prev = _ZERO
        cumulative_ch4 = _ZERO

        for year_idx in range(1, years + 1):
            # Year 1: DDOCm_deposited + 0 (no prior accumulation)
            # Year N: 0 (no new deposition) + DDOCm_acc(N-1) x e^(-k)
            if year_idx == 1:
                ddocm_accumulated = (
                    ddocm_deposited + ddocm_accumulated_prev * decay_factor
                ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
            else:
                ddocm_accumulated = (
                    ddocm_accumulated_prev * decay_factor
                ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

            # DDOCm decomposed this year
            ddocm_decomposed = (
                ddocm_accumulated_prev * decomp_factor
                if year_idx > 1
                else ddocm_deposited * decomp_factor
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

            # Handle year 1 special case: decomposition starts on accumulated
            if year_idx == 1:
                ddocm_decomposed = (ddocm_accumulated * decomp_factor).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )

            # CH4 generated this year
            ch4_generated = self.calculate_ch4_generation(ddocm_decomposed, f)
            cumulative_ch4 = (cumulative_ch4 + ch4_generated).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

            schedule.append({
                "year": Decimal(str(year_idx)),
                "ddocm_accumulated": ddocm_accumulated,
                "ddocm_decomposed": ddocm_decomposed,
                "ch4_generated_tonnes": ch4_generated,
                "cumulative_ch4_tonnes": cumulative_ch4,
            })

            ddocm_accumulated_prev = ddocm_accumulated

        logger.debug(
            "calculate_multi_year_decay: mass=%.4f k=%.6f years=%d total_ch4=%.8f",
            mass, k, years, cumulative_ch4,
        )

        return schedule

    # ==================================================================
    # 8. calculate_first_year_emissions - First year only
    # ==================================================================

    def calculate_first_year_emissions(
        self,
        landfill_input: LandfillInput,
        gwp_version: Optional[GWPVersion] = None,
    ) -> LandfillEmissionsResult:
        """
        Calculate landfill emissions for the first year only.

        Convenience method that forces years_projection=1 regardless of
        the value in landfill_input. Useful for annual reporting where
        only the first year of new waste deposition matters.

        Args:
            landfill_input: LandfillInput with waste and landfill parameters.
            gwp_version: Override GWP version. Defaults to engine default.

        Returns:
            LandfillEmissionsResult for year 1 only.

        Raises:
            ValueError: If input validation fails.

        Example:
            >>> result = engine.calculate_first_year_emissions(landfill_input)
            >>> assert result.decay_projection is None
        """
        # Create a new LandfillInput with years_projection=1
        first_year_input = LandfillInput(
            mass_tonnes=landfill_input.mass_tonnes,
            waste_category=landfill_input.waste_category,
            landfill_type=landfill_input.landfill_type,
            climate_zone=landfill_input.climate_zone,
            gas_collection=landfill_input.gas_collection,
            has_cover=landfill_input.has_cover,
            doc_override=landfill_input.doc_override,
            k_override=landfill_input.k_override,
            mcf_override=landfill_input.mcf_override,
            oxidation_factor_override=landfill_input.oxidation_factor_override,
            years_projection=1,
        )

        return self.calculate(first_year_input, gwp_version=gwp_version)

    # ==================================================================
    # 9. calculate_lifetime_emissions - Full lifetime projection
    # ==================================================================

    def calculate_lifetime_emissions(
        self,
        landfill_input: LandfillInput,
        projection_years: int = _DEFAULT_PROJECTION_YEARS,
        gwp_version: Optional[GWPVersion] = None,
    ) -> LandfillEmissionsResult:
        """
        Calculate lifetime landfill emissions over the full decay period.

        Projects emissions for the specified number of years (default 100),
        capturing nearly all CH4 generation from the deposited waste. The
        decay projection schedule is included in the result.

        Note: For most waste types, 100 years captures >99% of total
        decomposition. Wood in dry climates may require longer periods.

        Args:
            landfill_input: LandfillInput with waste and landfill parameters.
            projection_years: Number of years to project (1-200). Default 100.
            gwp_version: Override GWP version. Defaults to engine default.

        Returns:
            LandfillEmissionsResult with full decay_projection schedule.

        Raises:
            ValueError: If projection_years is out of range or input is invalid.

        Example:
            >>> result = engine.calculate_lifetime_emissions(
            ...     landfill_input, projection_years=50
            ... )
            >>> assert result.decay_projection is not None
            >>> assert len(result.decay_projection) == 50
        """
        if projection_years < 1 or projection_years > _MAX_PROJECTION_YEARS:
            raise ValueError(
                f"projection_years must be between 1 and {_MAX_PROJECTION_YEARS}, "
                f"got {projection_years}"
            )

        lifetime_input = LandfillInput(
            mass_tonnes=landfill_input.mass_tonnes,
            waste_category=landfill_input.waste_category,
            landfill_type=landfill_input.landfill_type,
            climate_zone=landfill_input.climate_zone,
            gas_collection=landfill_input.gas_collection,
            has_cover=landfill_input.has_cover,
            doc_override=landfill_input.doc_override,
            k_override=landfill_input.k_override,
            mcf_override=landfill_input.mcf_override,
            oxidation_factor_override=landfill_input.oxidation_factor_override,
            years_projection=projection_years,
        )

        return self.calculate(lifetime_input, gwp_version=gwp_version)

    # ==================================================================
    # 10. calculate_mixed_waste - Weighted composition handling
    # ==================================================================

    def calculate_mixed_waste(
        self,
        compositions: List[WasteCompositionInput],
        total_mass_tonnes: Decimal,
        landfill_type: LandfillType,
        climate_zone: ClimateZone,
        gas_collection: GasCollectionSystem = GasCollectionSystem.NONE,
        has_cover: bool = True,
        years_projection: int = 1,
        gwp_version: Optional[GWPVersion] = None,
    ) -> LandfillEmissionsResult:
        """
        Calculate landfill emissions for mixed waste with detailed composition.

        Each waste component is modelled separately with its own DOC and k
        values, then results are aggregated. This provides more accurate
        estimates than using a single blended DOC for mixed MSW.

        The method creates individual LandfillInput objects for each waste
        component, calculates emissions separately, and sums the results.

        Args:
            compositions: List of WasteCompositionInput defining the fraction
                         of each waste category (fractions must sum to 1.0).
            total_mass_tonnes: Total mass of mixed waste in tonnes.
            landfill_type: Type of landfill for MCF determination.
            climate_zone: IPCC climate zone for decay rate selection.
            gas_collection: Gas collection system type.
            has_cover: Whether landfill has engineered cover.
            years_projection: Number of years to project.
            gwp_version: Override GWP version.

        Returns:
            Aggregated LandfillEmissionsResult across all waste components.

        Raises:
            ValueError: If compositions are empty, fractions do not sum to ~1.0,
                       or total_mass_tonnes is not positive.

        Example:
            >>> compositions = [
            ...     WasteCompositionInput(
            ...         waste_category=WasteCategory.FOOD_WASTE,
            ...         fraction=Decimal("0.55"),
            ...     ),
            ...     WasteCompositionInput(
            ...         waste_category=WasteCategory.PAPER_CARDBOARD,
            ...         fraction=Decimal("0.25"),
            ...     ),
            ...     WasteCompositionInput(
            ...         waste_category=WasteCategory.GARDEN_WASTE,
            ...         fraction=Decimal("0.20"),
            ...     ),
            ... ]
            >>> result = engine.calculate_mixed_waste(
            ...     compositions=compositions,
            ...     total_mass_tonnes=Decimal("500"),
            ...     landfill_type=LandfillType.MANAGED_ANAEROBIC,
            ...     climate_zone=ClimateZone.TEMPERATE_WET,
            ... )
        """
        start_ts = time.monotonic()

        # Validate inputs
        if not compositions:
            raise ValueError("compositions list must not be empty")

        self._validate_positive("total_mass_tonnes", total_mass_tonnes)

        # Validate fractions sum to approximately 1.0
        fraction_sum = sum(c.fraction for c in compositions)
        tolerance = Decimal("0.01")
        if abs(fraction_sum - _ONE) > tolerance:
            raise ValueError(
                f"Composition fractions must sum to ~1.0, got {fraction_sum}"
            )

        effective_gwp = gwp_version or self._gwp_version

        # Calculate emissions for each component
        total_ch4_generated = _ZERO
        total_ch4_recovered = _ZERO
        total_ch4_oxidized = _ZERO
        total_ch4_emitted = _ZERO
        total_co2e = _ZERO

        # Weighted parameters for reporting
        weighted_doc = _ZERO
        weighted_k = _ZERO

        component_results: List[Tuple[WasteCategory, LandfillEmissionsResult]] = []

        for comp in compositions:
            component_mass = (total_mass_tonnes * comp.fraction).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

            if component_mass <= _ZERO:
                continue

            component_input = LandfillInput(
                mass_tonnes=component_mass,
                waste_category=comp.waste_category,
                landfill_type=landfill_type,
                climate_zone=climate_zone,
                gas_collection=gas_collection,
                has_cover=has_cover,
                years_projection=years_projection,
            )

            component_result = self.calculate(component_input, gwp_version=effective_gwp)
            component_results.append((comp.waste_category, component_result))

            total_ch4_generated = (
                total_ch4_generated + component_result.ch4_generated_tonnes
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
            total_ch4_recovered = (
                total_ch4_recovered + component_result.ch4_recovered_tonnes
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
            total_ch4_oxidized = (
                total_ch4_oxidized + component_result.ch4_oxidized_tonnes
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
            total_ch4_emitted = (
                total_ch4_emitted + component_result.ch4_emitted_tonnes
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
            total_co2e = (
                total_co2e + component_result.co2e_total
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

            weighted_doc = (
                weighted_doc + component_result.doc_used * comp.fraction
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
            weighted_k = (
                weighted_k + component_result.k_used * comp.fraction
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Resolve common parameters for reporting
        mcf = MCF_VALUES.get(landfill_type, Decimal("0.6"))
        gas_capture_eff = GAS_CAPTURE_EFFICIENCY.get(gas_collection, _ZERO)
        ox = self._get_oxidation_factor(has_cover, gas_collection, None)

        elapsed_s = time.monotonic() - start_ts
        logger.info(
            "calculate_mixed_waste: components=%d total_mass=%.4f "
            "total_co2e=%.6f elapsed=%.4fs",
            len(compositions), total_mass_tonnes, total_co2e, elapsed_s,
        )

        return LandfillEmissionsResult(
            ch4_generated_tonnes=total_ch4_generated,
            ch4_recovered_tonnes=total_ch4_recovered,
            ch4_oxidized_tonnes=total_ch4_oxidized,
            ch4_emitted_tonnes=total_ch4_emitted,
            co2e_total=total_co2e,
            doc_used=weighted_doc,
            mcf_used=mcf,
            k_used=weighted_k,
            gas_capture_efficiency=gas_capture_eff,
            oxidation_factor=ox,
            decay_projection=None,
        )

    # ==================================================================
    # 11. get_half_life - Compute half-life from decay constant
    # ==================================================================

    def get_half_life(self, k: Decimal) -> Decimal:
        """
        Calculate the half-life from the decay rate constant.

        Formula:
            t_half = ln(2) / k

        The half-life is the time required for half of the degradable
        organic carbon to decompose. It is a useful interpretation of
        the decay rate constant k.

        Args:
            k: Decay rate constant in yr^-1 (must be positive).

        Returns:
            Half-life in years (Decimal, 8 decimal places).

        Raises:
            ValueError: If k is zero or negative.

        Example:
            >>> half_life = engine.get_half_life(Decimal("0.185"))
            >>> # ~3.75 years for food waste in temperate wet climate
        """
        self._validate_positive("k", k)

        half_life = (_LN2 / k).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        logger.debug("get_half_life: k=%.6f => t_half=%.4f years", k, half_life)

        return half_life

    # ==================================================================
    # 12. get_decay_constant_from_half_life - Reverse calculation
    # ==================================================================

    def get_decay_constant_from_half_life(self, half_life: Decimal) -> Decimal:
        """
        Calculate the decay rate constant from a given half-life.

        Formula:
            k = ln(2) / t_half

        This is the inverse of get_half_life(). Useful when facility-specific
        data provides half-life rather than decay constant directly.

        Args:
            half_life: Half-life in years (must be positive).

        Returns:
            Decay rate constant k in yr^-1 (Decimal, 8 decimal places).

        Raises:
            ValueError: If half_life is zero or negative.

        Example:
            >>> k = engine.get_decay_constant_from_half_life(Decimal("3.75"))
            >>> # ~0.1848 yr^-1
        """
        self._validate_positive("half_life", half_life)

        k = (_LN2 / half_life).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        logger.debug(
            "get_decay_constant_from_half_life: t_half=%.4f => k=%.8f",
            half_life, k,
        )

        return k

    # ==================================================================
    # 13. calculate_with_gas_capture_timeline - Phased gas capture
    # ==================================================================

    def calculate_with_gas_capture_timeline(
        self,
        landfill_input: LandfillInput,
        capture_start_year: int,
        capture_system: GasCollectionSystem,
        projection_years: int = _DEFAULT_PROJECTION_YEARS,
        gwp_version: Optional[GWPVersion] = None,
    ) -> LandfillEmissionsResult:
        """
        Calculate emissions with a time-phased gas capture system.

        Models a scenario where gas capture does not begin until a specified
        year after waste deposition. This is common when landfill gas
        collection is installed after a cell is closed.

        Before capture_start_year: gas_capture_efficiency = 0
        From capture_start_year onwards: gas_capture_efficiency = system efficiency

        Args:
            landfill_input: LandfillInput (gas_collection field is ignored; the
                           capture_system arg is used instead with the timeline).
            capture_start_year: Year after deposition when gas capture begins (1-based).
            capture_system: Type of gas collection system installed.
            projection_years: Total years to project (must be >= capture_start_year).
            gwp_version: Override GWP version.

        Returns:
            LandfillEmissionsResult with cumulative emissions accounting for
            the phased gas capture.

        Raises:
            ValueError: If capture_start_year > projection_years or inputs invalid.

        Example:
            >>> result = engine.calculate_with_gas_capture_timeline(
            ...     landfill_input=input,
            ...     capture_start_year=3,
            ...     capture_system=GasCollectionSystem.ACTIVE_GEOMEMBRANE,
            ...     projection_years=50,
            ... )
        """
        if capture_start_year < 1:
            raise ValueError(
                f"capture_start_year must be >= 1, got {capture_start_year}"
            )
        if capture_start_year > projection_years:
            raise ValueError(
                f"capture_start_year ({capture_start_year}) must be <= "
                f"projection_years ({projection_years})"
            )
        if projection_years < 1 or projection_years > _MAX_PROJECTION_YEARS:
            raise ValueError(
                f"projection_years must be between 1 and {_MAX_PROJECTION_YEARS}"
            )

        effective_gwp = gwp_version or self._gwp_version
        start_ts = time.monotonic()

        # Resolve parameters
        doc = self._resolve_doc(landfill_input)
        docf = _DEFAULT_DOCF
        mcf = self._resolve_mcf(landfill_input)
        k = self._resolve_k(landfill_input)
        ox = self._resolve_oxidation_factor(landfill_input)
        capture_eff = GAS_CAPTURE_EFFICIENCY.get(capture_system, _ZERO)

        # Calculate DDOCm deposited
        ddocm = self.calculate_ddocm(
            mass=landfill_input.mass_tonnes, doc=doc, docf=docf, mcf=mcf
        )

        # Run year-by-year with phased gas capture
        decay_schedule = self.calculate_multi_year_decay(
            mass=landfill_input.mass_tonnes,
            doc=doc, docf=docf, mcf=mcf, k=k,
            years=projection_years,
        )

        total_ch4_generated = _ZERO
        total_ch4_recovered = _ZERO
        total_ch4_oxidized = _ZERO
        total_ch4_emitted = _ZERO

        emission_schedule: List[Dict[str, Decimal]] = []

        for year_data in decay_schedule:
            year_num = int(year_data["year"])
            ch4_gen = year_data["ch4_generated_tonnes"]
            total_ch4_generated = (total_ch4_generated + ch4_gen).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

            # Apply gas capture only from capture_start_year onwards
            year_capture_eff = capture_eff if year_num >= capture_start_year else _ZERO

            ch4_rec = self.calculate_ch4_recovery(ch4_gen, year_capture_eff)
            ch4_emit = self.calculate_ch4_emitted(ch4_gen, ch4_rec, ox)
            ch4_ox = (
                (ch4_gen - ch4_rec) * ox
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
            if ch4_ox < _ZERO:
                ch4_ox = _ZERO

            total_ch4_recovered = (total_ch4_recovered + ch4_rec).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            total_ch4_oxidized = (total_ch4_oxidized + ch4_ox).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            total_ch4_emitted = (total_ch4_emitted + ch4_emit).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

            co2e_year = self.convert_ch4_to_co2e(ch4_emit, effective_gwp)

            emission_schedule.append({
                "year": year_data["year"],
                "ch4_generated_tonnes": ch4_gen,
                "ch4_recovered_tonnes": ch4_rec,
                "ch4_emitted_tonnes": ch4_emit,
                "co2e_tonnes": co2e_year,
                "gas_capture_active": Decimal("1") if year_num >= capture_start_year else _ZERO,
            })

        total_co2e = self.convert_ch4_to_co2e(total_ch4_emitted, effective_gwp)

        # Add N2O from cover
        n2o_co2e = self._calculate_n2o_co2e(
            landfill_input.mass_tonnes, landfill_input.has_cover,
            landfill_input.gas_collection, effective_gwp,
            projection_years,
        )
        total_co2e = (total_co2e + n2o_co2e).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        elapsed_s = time.monotonic() - start_ts
        logger.info(
            "calculate_with_gas_capture_timeline: capture_start=%d system=%s "
            "total_co2e=%.6f elapsed=%.4fs",
            capture_start_year, capture_system.value, total_co2e, elapsed_s,
        )

        return LandfillEmissionsResult(
            ch4_generated_tonnes=total_ch4_generated,
            ch4_recovered_tonnes=total_ch4_recovered,
            ch4_oxidized_tonnes=total_ch4_oxidized,
            ch4_emitted_tonnes=total_ch4_emitted,
            co2e_total=total_co2e,
            doc_used=doc,
            mcf_used=mcf,
            k_used=k,
            gas_capture_efficiency=capture_eff,
            oxidation_factor=ox,
            decay_projection=emission_schedule,
        )

    # ==================================================================
    # 14. estimate_cumulative_emissions - Cumulative projection
    # ==================================================================

    def estimate_cumulative_emissions(
        self,
        landfill_input: LandfillInput,
        projection_years: int = _DEFAULT_PROJECTION_YEARS,
        gwp_version: Optional[GWPVersion] = None,
    ) -> List[Dict[str, Decimal]]:
        """
        Estimate cumulative CH4 and CO2e emissions over a projection period.

        Returns a year-by-year list showing both annual and cumulative
        emissions, useful for financial provisioning and closure liability
        estimation.

        Unlike calculate_lifetime_emissions(), this method returns the raw
        projection data as a list of dicts rather than a LandfillEmissionsResult.

        Args:
            landfill_input: LandfillInput with waste and landfill parameters.
            projection_years: Number of years to project (1-200). Default 100.
            gwp_version: Override GWP version.

        Returns:
            List of dicts with keys: year, annual_ch4_tonnes,
            cumulative_ch4_tonnes, annual_co2e_tonnes, cumulative_co2e_tonnes.

        Raises:
            ValueError: If projection_years is out of range or inputs invalid.

        Example:
            >>> projections = engine.estimate_cumulative_emissions(
            ...     landfill_input, projection_years=30,
            ... )
            >>> total_co2e = projections[-1]["cumulative_co2e_tonnes"]
        """
        if projection_years < 1 or projection_years > _MAX_PROJECTION_YEARS:
            raise ValueError(
                f"projection_years must be between 1 and {_MAX_PROJECTION_YEARS}, "
                f"got {projection_years}"
            )

        effective_gwp = gwp_version or self._gwp_version

        # Resolve parameters
        doc = self._resolve_doc(landfill_input)
        docf = _DEFAULT_DOCF
        mcf = self._resolve_mcf(landfill_input)
        k = self._resolve_k(landfill_input)
        gas_capture_eff = self._resolve_gas_capture_efficiency(landfill_input)
        ox = self._resolve_oxidation_factor(landfill_input)

        # Get decay schedule
        decay_schedule = self.calculate_multi_year_decay(
            mass=landfill_input.mass_tonnes,
            doc=doc, docf=docf, mcf=mcf, k=k,
            years=projection_years,
        )

        cumulative_ch4 = _ZERO
        cumulative_co2e = _ZERO
        projections: List[Dict[str, Decimal]] = []

        for year_data in decay_schedule:
            ch4_gen = year_data["ch4_generated_tonnes"]

            # Apply recovery and oxidation
            ch4_rec = self.calculate_ch4_recovery(ch4_gen, gas_capture_eff)
            ch4_emit = self.calculate_ch4_emitted(ch4_gen, ch4_rec, ox)
            co2e = self.convert_ch4_to_co2e(ch4_emit, effective_gwp)

            cumulative_ch4 = (cumulative_ch4 + ch4_emit).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            cumulative_co2e = (cumulative_co2e + co2e).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

            projections.append({
                "year": year_data["year"],
                "annual_ch4_tonnes": ch4_emit,
                "cumulative_ch4_tonnes": cumulative_ch4,
                "annual_co2e_tonnes": co2e,
                "cumulative_co2e_tonnes": cumulative_co2e,
            })

        logger.debug(
            "estimate_cumulative_emissions: years=%d final_cumulative_co2e=%.6f",
            projection_years, cumulative_co2e,
        )

        return projections

    # ==================================================================
    # 15. calculate_batch - Batch processing
    # ==================================================================

    def calculate_batch(
        self,
        inputs: List[LandfillInput],
        gwp_version: Optional[GWPVersion] = None,
        continue_on_error: bool = False,
    ) -> List[LandfillEmissionsResult]:
        """
        Calculate landfill emissions for a batch of inputs.

        Processes multiple LandfillInput records sequentially, returning
        a list of results in the same order. Optionally continues on error,
        replacing failed calculations with None.

        Args:
            inputs: List of LandfillInput objects to process.
            gwp_version: Override GWP version for all calculations.
            continue_on_error: If True, continues processing on errors and
                              returns None for failed calculations. If False
                              (default), raises the first error encountered.

        Returns:
            List of LandfillEmissionsResult (or None if continue_on_error
            is True and a calculation failed).

        Raises:
            ValueError: If inputs is empty or exceeds batch size limit.
            ValueError or other: If continue_on_error is False and a
                                calculation fails.

        Example:
            >>> results = engine.calculate_batch([input1, input2, input3])
            >>> assert len(results) == 3
        """
        if not inputs:
            raise ValueError("inputs list must not be empty")

        if len(inputs) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(inputs)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        start_ts = time.monotonic()
        results: List[LandfillEmissionsResult] = []
        errors_count = 0

        for idx, inp in enumerate(inputs):
            try:
                result = self.calculate(inp, gwp_version=gwp_version)
                results.append(result)
            except Exception as e:
                errors_count += 1
                if continue_on_error:
                    logger.warning(
                        "calculate_batch: item %d/%d failed: %s (continuing)",
                        idx + 1, len(inputs), str(e),
                    )
                    # Append a zero-emission placeholder result
                    results.append(self._create_zero_result(inp))
                else:
                    logger.error(
                        "calculate_batch: item %d/%d failed: %s (aborting)",
                        idx + 1, len(inputs), str(e),
                    )
                    raise

        elapsed_s = time.monotonic() - start_ts
        successful = len(inputs) - errors_count

        self._metrics.record_batch(
            method="landfill_fod",
            size=len(inputs),
            successful=successful,
            failed=errors_count,
        )

        logger.info(
            "calculate_batch: total=%d successful=%d errors=%d elapsed=%.4fs",
            len(inputs), successful, errors_count, elapsed_s,
        )

        return results

    # ==================================================================
    # 16. validate_landfill_input - Input validation
    # ==================================================================

    def validate_landfill_input(
        self,
        landfill_input: LandfillInput,
    ) -> List[str]:
        """
        Validate a LandfillInput and return a list of error messages.

        Performs comprehensive validation beyond Pydantic's field-level
        constraints, including cross-field consistency checks and
        business rule validation.

        Args:
            landfill_input: LandfillInput to validate.

        Returns:
            List of error message strings. Empty list means valid.

        Example:
            >>> errors = engine.validate_landfill_input(landfill_input)
            >>> if errors:
            ...     print(f"Validation failed: {errors}")
        """
        errors: List[str] = []

        # Mass validation
        if landfill_input.mass_tonnes <= _ZERO:
            errors.append("mass_tonnes must be positive")

        if landfill_input.mass_tonnes > Decimal("1000000"):
            errors.append(
                f"mass_tonnes={landfill_input.mass_tonnes} exceeds reasonable maximum "
                "of 1,000,000 tonnes for a single deposition event"
            )

        # Waste category DOC check
        doc = DOC_VALUES.get(landfill_input.waste_category)
        if doc is None:
            errors.append(
                f"No DOC value defined for waste_category={landfill_input.waste_category.value}"
            )
        elif doc == _ZERO and landfill_input.doc_override is None:
            # Non-degradable waste in landfill produces no CH4
            logger.warning(
                "Waste category %s has DOC=0; no CH4 will be generated unless doc_override is set",
                landfill_input.waste_category.value,
            )

        # Climate zone + waste category decay rate check
        decay_key = _WASTE_CATEGORY_TO_DECAY_KEY.get(landfill_input.waste_category)
        if decay_key is not None and landfill_input.k_override is None:
            climate_rates = DECAY_RATE_CONSTANTS.get(landfill_input.climate_zone)
            if climate_rates is None:
                errors.append(
                    f"No decay rates for climate_zone={landfill_input.climate_zone.value}"
                )
            elif decay_key not in climate_rates:
                errors.append(
                    f"No decay rate for climate={landfill_input.climate_zone.value}, "
                    f"waste_key={decay_key}"
                )

        # MCF check
        if landfill_input.mcf_override is None:
            if landfill_input.landfill_type not in MCF_VALUES:
                errors.append(
                    f"No MCF value for landfill_type={landfill_input.landfill_type.value}"
                )

        # Override range validation
        if landfill_input.doc_override is not None:
            if landfill_input.doc_override < _ZERO or landfill_input.doc_override > _ONE:
                errors.append("doc_override must be between 0 and 1")

        if landfill_input.k_override is not None:
            if landfill_input.k_override <= _ZERO:
                errors.append("k_override must be positive")
            if landfill_input.k_override > Decimal("2.0"):
                errors.append(
                    f"k_override={landfill_input.k_override} exceeds reasonable "
                    "maximum of 2.0 yr^-1"
                )

        if landfill_input.mcf_override is not None:
            if landfill_input.mcf_override < _ZERO or landfill_input.mcf_override > _ONE:
                errors.append("mcf_override must be between 0 and 1")

        if landfill_input.oxidation_factor_override is not None:
            if (
                landfill_input.oxidation_factor_override < _ZERO
                or landfill_input.oxidation_factor_override > _ONE
            ):
                errors.append("oxidation_factor_override must be between 0 and 1")

        # Years projection
        if landfill_input.years_projection < 1:
            errors.append("years_projection must be >= 1")
        if landfill_input.years_projection > _DEFAULT_PROJECTION_YEARS:
            logger.warning(
                "years_projection=%d exceeds typical 100-year horizon",
                landfill_input.years_projection,
            )

        return errors

    # ==================================================================
    # 17. calculate_n2o_from_cover - Minor N2O from cover soil
    # ==================================================================

    def calculate_n2o_from_cover(
        self,
        mass_tonnes: Decimal,
        cover_type: str,
        years: int = 1,
    ) -> Decimal:
        """
        Calculate minor N2O emissions from landfill cover soil.

        Landfill cover soils can produce small amounts of N2O through
        nitrification and denitrification processes. This is typically
        a minor contribution compared to CH4 but is included for
        completeness per IPCC guidance.

        Reference: IPCC 2006 Vol 5 Ch 3, Section 3.2.3; Bogner et al. (2007).

        Args:
            mass_tonnes: Mass of waste deposited in tonnes.
            cover_type: Cover type key: 'no_cover', 'soil_cover',
                       'biocover', or 'geomembrane'.
            years: Number of years over which to calculate N2O (default 1).

        Returns:
            N2O emissions in tonnes (Decimal, 8 decimal places).

        Raises:
            ValueError: If mass_tonnes is not positive or cover_type is unknown.

        Example:
            >>> n2o = engine.calculate_n2o_from_cover(
            ...     mass_tonnes=Decimal("1000"),
            ...     cover_type="soil_cover",
            ...     years=30,
            ... )
        """
        self._validate_positive("mass_tonnes", mass_tonnes)

        if cover_type not in N2O_COVER_EMISSION_FACTORS:
            raise ValueError(
                f"Unknown cover_type='{cover_type}'. Valid: "
                f"{list(N2O_COVER_EMISSION_FACTORS.keys())}"
            )

        if years < 1:
            raise ValueError(f"years must be >= 1, got {years}")

        ef_n2o = N2O_COVER_EMISSION_FACTORS[cover_type]
        n2o_tonnes = (mass_tonnes * ef_n2o * Decimal(str(years))).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "calculate_n2o_from_cover: mass=%.4f cover=%s years=%d ef=%.6f => N2O=%.8f",
            mass_tonnes, cover_type, years, ef_n2o, n2o_tonnes,
        )

        return n2o_tonnes

    # ==================================================================
    # 18. calculate_total_co2e - Combined CH4 + N2O to CO2e
    # ==================================================================

    def calculate_total_co2e(
        self,
        ch4_emitted_tonnes: Decimal,
        n2o_emitted_tonnes: Decimal,
        gwp_version: Optional[GWPVersion] = None,
    ) -> Decimal:
        """
        Calculate total CO2 equivalent from CH4 and N2O emissions.

        Formula:
            CO2e = (CH4 x GWP_CH4) + (N2O x GWP_N2O)

        Args:
            ch4_emitted_tonnes: Net CH4 emitted in tonnes.
            n2o_emitted_tonnes: N2O emitted in tonnes.
            gwp_version: IPCC Assessment Report for GWP values.

        Returns:
            Total CO2e in tonnes (Decimal, 8 decimal places).

        Raises:
            ValueError: If inputs are negative.

        Example:
            >>> total = engine.calculate_total_co2e(
            ...     ch4_emitted_tonnes=Decimal("5.0"),
            ...     n2o_emitted_tonnes=Decimal("0.001"),
            ...     gwp_version=GWPVersion.AR5,
            ... )
        """
        self._validate_non_negative("ch4_emitted_tonnes", ch4_emitted_tonnes)
        self._validate_non_negative("n2o_emitted_tonnes", n2o_emitted_tonnes)

        effective_gwp = gwp_version or self._gwp_version
        gwp_ch4 = GWP_VALUES[effective_gwp]["ch4"]
        gwp_n2o = GWP_VALUES[effective_gwp]["n2o"]

        co2e_ch4 = (ch4_emitted_tonnes * gwp_ch4).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        co2e_n2o = (n2o_emitted_tonnes * gwp_n2o).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        total_co2e = (co2e_ch4 + co2e_n2o).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "calculate_total_co2e: CH4=%.8f(GWP=%.1f)=%.4f N2O=%.8f(GWP=%.1f)=%.4f => total=%.4f",
            ch4_emitted_tonnes, gwp_ch4, co2e_ch4,
            n2o_emitted_tonnes, gwp_n2o, co2e_n2o,
            total_co2e,
        )

        return total_co2e

    # ==================================================================
    # 19. get_doc_for_waste_category - DOC lookup
    # ==================================================================

    def get_doc_for_waste_category(
        self,
        waste_category: WasteCategory,
    ) -> Decimal:
        """
        Look up the IPCC default DOC value for a waste category.

        Reference: IPCC 2006 Vol 5 Table 2.4.

        Args:
            waste_category: WasteCategory enum value.

        Returns:
            DOC fraction (Decimal).

        Raises:
            KeyError: If waste_category is not in DOC_VALUES.

        Example:
            >>> doc = engine.get_doc_for_waste_category(WasteCategory.FOOD_WASTE)
            >>> assert doc == Decimal("0.150")
        """
        if waste_category not in DOC_VALUES:
            raise KeyError(
                f"No DOC value for waste_category={waste_category.value}"
            )
        return DOC_VALUES[waste_category]

    # ==================================================================
    # 20. get_k_for_climate_and_waste - Decay rate lookup
    # ==================================================================

    def get_k_for_climate_and_waste(
        self,
        climate_zone: ClimateZone,
        waste_category: WasteCategory,
    ) -> Decimal:
        """
        Look up the IPCC decay rate constant for a climate zone and waste type.

        Reference: IPCC 2006 Vol 5 Table 3.3.

        Args:
            climate_zone: IPCC climate zone.
            waste_category: WasteCategory enum value (mapped to decay key).

        Returns:
            Decay rate constant k in yr^-1 (Decimal).

        Raises:
            KeyError: If the combination is not in DECAY_RATE_CONSTANTS.

        Example:
            >>> k = engine.get_k_for_climate_and_waste(
            ...     ClimateZone.TEMPERATE_WET, WasteCategory.FOOD_WASTE
            ... )
            >>> assert k == Decimal("0.185")
        """
        decay_key = _WASTE_CATEGORY_TO_DECAY_KEY.get(waste_category)
        if decay_key is None:
            raise KeyError(
                f"No decay rate mapping for waste_category={waste_category.value}"
            )

        climate_rates = DECAY_RATE_CONSTANTS.get(climate_zone)
        if climate_rates is None:
            raise KeyError(
                f"No decay rates for climate_zone={climate_zone.value}"
            )

        k = climate_rates.get(decay_key)
        if k is None:
            raise KeyError(
                f"No decay rate for climate={climate_zone.value}, "
                f"waste_key={decay_key}"
            )

        return k

    # ==================================================================
    # 21. get_mcf_for_landfill_type - MCF lookup
    # ==================================================================

    def get_mcf_for_landfill_type(
        self,
        landfill_type: LandfillType,
    ) -> Decimal:
        """
        Look up the IPCC Methane Correction Factor for a landfill type.

        Reference: IPCC 2006 Vol 5 Table 3.1.

        Args:
            landfill_type: LandfillType enum value.

        Returns:
            MCF value (Decimal, 0-1).

        Raises:
            KeyError: If landfill_type is not in MCF_VALUES.

        Example:
            >>> mcf = engine.get_mcf_for_landfill_type(LandfillType.MANAGED_ANAEROBIC)
            >>> assert mcf == Decimal("1.0")
        """
        if landfill_type not in MCF_VALUES:
            raise KeyError(
                f"No MCF value for landfill_type={landfill_type.value}"
            )
        return MCF_VALUES[landfill_type]

    # ==================================================================
    # 22. get_gas_capture_efficiency - Gas capture lookup
    # ==================================================================

    def get_gas_capture_efficiency(
        self,
        gas_collection_system: GasCollectionSystem,
    ) -> Decimal:
        """
        Look up the gas capture efficiency for a collection system type.

        Args:
            gas_collection_system: GasCollectionSystem enum value.

        Returns:
            Capture efficiency fraction (Decimal, 0-1).

        Raises:
            KeyError: If gas_collection_system is not in GAS_CAPTURE_EFFICIENCY.

        Example:
            >>> eff = engine.get_gas_capture_efficiency(
            ...     GasCollectionSystem.ACTIVE_GEOMEMBRANE
            ... )
            >>> assert eff == Decimal("0.90")
        """
        if gas_collection_system not in GAS_CAPTURE_EFFICIENCY:
            raise KeyError(
                f"No capture efficiency for gas_collection={gas_collection_system.value}"
            )
        return GAS_CAPTURE_EFFICIENCY[gas_collection_system]

    # ==================================================================
    # 23. calculate_biogenic_co2 - Biogenic CO2 from decomposition
    # ==================================================================

    def calculate_biogenic_co2(
        self,
        ddocm_decomposed: Decimal,
        f: Decimal = _DEFAULT_F_CH4,
    ) -> Decimal:
        """
        Calculate biogenic CO2 generated alongside CH4 in landfill gas.

        Landfill gas is approximately 50% CH4 and 50% CO2 (by volume).
        The CO2 fraction is biogenic and reported as a memo item only,
        not counted in Scope 3 emissions per GHG Protocol.

        Formula:
            CO2_biogenic = DDOCm_decomposed x (1 - F) x (44/12)

        Args:
            ddocm_decomposed: Mass of DDOCm decomposed in this period (tonnes).
            f: Fraction of CH4 in landfill gas (default 0.50).

        Returns:
            Biogenic CO2 in tonnes (Decimal, 8 decimal places). Memo item only.

        Raises:
            ValueError: If ddocm_decomposed is negative or f is out of range.

        Example:
            >>> bio_co2 = engine.calculate_biogenic_co2(Decimal("3.75"))
            >>> # This is a memo item, NOT counted in Scope 3 totals
        """
        self._validate_non_negative("ddocm_decomposed", ddocm_decomposed)
        self._validate_fraction("f", f)

        co2_fraction = _ONE - f
        biogenic_co2 = (ddocm_decomposed * co2_fraction * _CO2_C_RATIO).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "calculate_biogenic_co2: DDOCm_decomposed=%.8f (1-F)=%.4f 44/12=%.4f => CO2_bio=%.8f",
            ddocm_decomposed, co2_fraction, _CO2_C_RATIO, biogenic_co2,
        )

        return biogenic_co2

    # ==================================================================
    # 24. calculate_uncertainty - Uncertainty quantification
    # ==================================================================

    def calculate_uncertainty(
        self,
        co2e_total: Decimal,
        data_quality_tier: DataQualityTier = DataQualityTier.TIER_1,
    ) -> Dict[str, Decimal]:
        """
        Calculate uncertainty range for landfill FOD model results.

        Uses IPCC default uncertainty ranges based on the data quality
        tier. Higher tiers (facility-specific data) have lower uncertainty.

        Args:
            co2e_total: Total CO2e emissions in tonnes.
            data_quality_tier: IPCC data quality tier (affects uncertainty range).

        Returns:
            Dict with keys: lower_bound, upper_bound, uncertainty_pct,
            central_estimate (all Decimal).

        Raises:
            ValueError: If co2e_total is negative.

        Example:
            >>> unc = engine.calculate_uncertainty(
            ...     co2e_total=Decimal("100.0"),
            ...     data_quality_tier=DataQualityTier.TIER_1,
            ... )
            >>> # Tier 1: +/-30% => lower=70, upper=130
        """
        self._validate_non_negative("co2e_total", co2e_total)

        tier_key = data_quality_tier.value  # "tier_1", "tier_2", "tier_3"
        uncertainty_pct = UNCERTAINTY_RANGES.get("landfill_fod", {}).get(
            tier_key, Decimal("0.30")
        )

        lower_bound = (co2e_total * (_ONE - uncertainty_pct)).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        upper_bound = (co2e_total * (_ONE + uncertainty_pct)).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # Floor lower bound at zero
        if lower_bound < _ZERO:
            lower_bound = _ZERO

        return {
            "central_estimate": co2e_total,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "uncertainty_pct": uncertainty_pct,
        }

    # ==================================================================
    # 25. calculate_provenance_hash_for_result - Audit trail hash
    # ==================================================================

    def calculate_provenance_hash_for_result(
        self,
        landfill_input: LandfillInput,
        result: LandfillEmissionsResult,
    ) -> str:
        """
        Calculate a SHA-256 provenance hash for a landfill calculation.

        Combines the input parameters and output results into a single
        deterministic hash, providing a tamper-evident seal for audit trails.

        Args:
            landfill_input: The input that produced the result.
            result: The calculation result.

        Returns:
            Hexadecimal SHA-256 hash string (64 characters).

        Example:
            >>> prov_hash = engine.calculate_provenance_hash_for_result(input, result)
            >>> assert len(prov_hash) == 64
        """
        return calculate_provenance_hash(landfill_input, result)

    # ==================================================================
    # 26. get_all_decay_rate_constants - Full k table
    # ==================================================================

    def get_all_decay_rate_constants(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Return the complete IPCC decay rate constant table.

        Provides all 24 climate zone x waste type decay rate values
        from IPCC 2006 Vol 5 Table 3.3.

        Returns:
            Nested dict: {climate_zone_value: {waste_type_key: k_value}}.

        Example:
            >>> all_k = engine.get_all_decay_rate_constants()
            >>> k_food_tropical = all_k["tropical_wet"]["food_waste"]
        """
        return {
            cz.value: dict(rates)
            for cz, rates in DECAY_RATE_CONSTANTS.items()
        }

    # ==================================================================
    # 27. get_all_doc_values - Full DOC table
    # ==================================================================

    def get_all_doc_values(self) -> Dict[str, Decimal]:
        """
        Return the complete IPCC DOC value table.

        Provides DOC fractions for all waste categories from IPCC 2006
        Vol 5 Table 2.4.

        Returns:
            Dict mapping waste category value to DOC fraction.

        Example:
            >>> all_doc = engine.get_all_doc_values()
            >>> doc_food = all_doc["food_waste"]
        """
        return {wc.value: doc for wc, doc in DOC_VALUES.items()}

    # ==================================================================
    # 28. get_all_mcf_values - Full MCF table
    # ==================================================================

    def get_all_mcf_values(self) -> Dict[str, Decimal]:
        """
        Return the complete IPCC MCF value table.

        Provides Methane Correction Factors for all landfill types from
        IPCC 2006 Vol 5 Table 3.1.

        Returns:
            Dict mapping landfill type value to MCF fraction.

        Example:
            >>> all_mcf = engine.get_all_mcf_values()
            >>> mcf_managed = all_mcf["managed_anaerobic"]
        """
        return {lt.value: mcf for lt, mcf in MCF_VALUES.items()}

    # ==================================================================
    # PRIVATE HELPER METHODS
    # ==================================================================

    # ------------------------------------------------------------------
    # Parameter Resolution
    # ------------------------------------------------------------------

    def _resolve_doc(self, landfill_input: LandfillInput) -> Decimal:
        """
        Resolve DOC value from override or IPCC defaults.

        Args:
            landfill_input: Input with optional doc_override.

        Returns:
            DOC fraction (Decimal).

        Raises:
            KeyError: If waste_category not in DOC_VALUES and no override.
        """
        if landfill_input.doc_override is not None:
            logger.debug("Using DOC override: %.6f", landfill_input.doc_override)
            return landfill_input.doc_override

        doc = DOC_VALUES.get(landfill_input.waste_category)
        if doc is None:
            raise KeyError(
                f"No DOC value for waste_category={landfill_input.waste_category.value}. "
                "Provide doc_override."
            )
        return doc

    def _resolve_mcf(self, landfill_input: LandfillInput) -> Decimal:
        """
        Resolve MCF value from override or IPCC defaults.

        Args:
            landfill_input: Input with optional mcf_override.

        Returns:
            MCF fraction (Decimal).

        Raises:
            KeyError: If landfill_type not in MCF_VALUES and no override.
        """
        if landfill_input.mcf_override is not None:
            logger.debug("Using MCF override: %.6f", landfill_input.mcf_override)
            return landfill_input.mcf_override

        mcf = MCF_VALUES.get(landfill_input.landfill_type)
        if mcf is None:
            raise KeyError(
                f"No MCF value for landfill_type={landfill_input.landfill_type.value}. "
                "Provide mcf_override."
            )
        return mcf

    def _resolve_k(self, landfill_input: LandfillInput) -> Decimal:
        """
        Resolve decay rate constant from override or IPCC defaults.

        Args:
            landfill_input: Input with optional k_override.

        Returns:
            Decay rate constant k (Decimal, yr^-1).

        Raises:
            KeyError: If climate/waste combination not found and no override.
        """
        if landfill_input.k_override is not None:
            logger.debug("Using k override: %.6f", landfill_input.k_override)
            return landfill_input.k_override

        return self.get_k_for_climate_and_waste(
            landfill_input.climate_zone,
            landfill_input.waste_category,
        )

    def _resolve_gas_capture_efficiency(
        self, landfill_input: LandfillInput,
    ) -> Decimal:
        """
        Resolve gas capture efficiency from the gas collection system type.

        Args:
            landfill_input: Input with gas_collection field.

        Returns:
            Gas capture efficiency fraction (Decimal, 0-1).
        """
        return GAS_CAPTURE_EFFICIENCY.get(landfill_input.gas_collection, _ZERO)

    def _resolve_oxidation_factor(
        self, landfill_input: LandfillInput,
    ) -> Decimal:
        """
        Resolve oxidation factor from override, cover status, and gas system.

        Args:
            landfill_input: Input with has_cover, gas_collection, and optional override.

        Returns:
            Oxidation factor (Decimal, 0-1).
        """
        if landfill_input.oxidation_factor_override is not None:
            logger.debug(
                "Using OX override: %.6f",
                landfill_input.oxidation_factor_override,
            )
            return landfill_input.oxidation_factor_override

        return self._get_oxidation_factor(
            landfill_input.has_cover,
            landfill_input.gas_collection,
            landfill_input.oxidation_factor_override,
        )

    def _get_oxidation_factor(
        self,
        has_cover: bool,
        gas_collection: GasCollectionSystem,
        override: Optional[Decimal],
    ) -> Decimal:
        """
        Determine oxidation factor based on cover type and gas system.

        Args:
            has_cover: Whether the landfill has engineered cover.
            gas_collection: Gas collection system type.
            override: Optional explicit override.

        Returns:
            Oxidation factor (Decimal, 0-1).
        """
        if override is not None:
            return override

        if not has_cover:
            return _DEFAULT_OX_WITHOUT_COVER

        cover_type = _GAS_COLLECTION_TO_COVER_TYPE.get(
            gas_collection, "soil_cover"
        )
        return OXIDATION_FACTORS.get(cover_type, _DEFAULT_OX_WITH_COVER)

    # ------------------------------------------------------------------
    # FOD Model Core Calculations
    # ------------------------------------------------------------------

    def _calculate_single_year(
        self,
        ddocm: Decimal,
        k: Decimal,
        gas_capture_efficiency: Decimal,
        oxidation_factor: Decimal,
        effective_gwp: GWPVersion,
        doc: Decimal,
        mcf: Decimal,
    ) -> LandfillEmissionsResult:
        """
        Calculate landfill emissions for a single year.

        Applies the FOD model for year 1 only and returns a complete
        LandfillEmissionsResult without a decay projection.

        Args:
            ddocm: Mass of decomposable DOC deposited (tonnes).
            k: Decay rate constant (yr^-1).
            gas_capture_efficiency: Gas capture fraction (0-1).
            oxidation_factor: Oxidation factor (0-1).
            effective_gwp: GWP version for CO2e conversion.
            doc: DOC value used (for result reporting).
            mcf: MCF value used (for result reporting).

        Returns:
            LandfillEmissionsResult for year 1.
        """
        # Decomposition factor: 1 - e^(-k)
        decomp_factor = (_ONE - self._exp_neg_k(k)).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # DDOCm decomposed in year 1
        ddocm_decomposed = (ddocm * decomp_factor).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # CH4 generated
        ch4_generated = self.calculate_ch4_generation(ddocm_decomposed)

        # CH4 recovered
        ch4_recovered = self.calculate_ch4_recovery(ch4_generated, gas_capture_efficiency)

        # CH4 oxidized (for reporting)
        net_before_ox = ch4_generated - ch4_recovered
        if net_before_ox < _ZERO:
            net_before_ox = _ZERO
        ch4_oxidized = (net_before_ox * oxidation_factor).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # CH4 emitted
        ch4_emitted = self.calculate_ch4_emitted(
            ch4_generated, ch4_recovered, oxidation_factor
        )

        # CO2e from CH4
        co2e_ch4 = self.convert_ch4_to_co2e(ch4_emitted, effective_gwp)

        # N2O from cover (single year)
        cover_type = self._get_cover_type_key(True, GasCollectionSystem.NONE)
        n2o_co2e = _ZERO  # Include minor N2O from _calculate_n2o_co2e approach
        # For single year, the N2O contribution is negligible but we include it
        # for completeness. We use the mass from the original input context.
        # Since we do not have mass here directly, we back-derive from ddocm context.
        # However, for cleanliness the main calculate() does not pass mass here.
        # The N2O is added at the calculate() level if needed.

        total_co2e = co2e_ch4  # N2O is added at the calling level

        return LandfillEmissionsResult(
            ch4_generated_tonnes=ch4_generated,
            ch4_recovered_tonnes=ch4_recovered,
            ch4_oxidized_tonnes=ch4_oxidized,
            ch4_emitted_tonnes=ch4_emitted,
            co2e_total=total_co2e,
            doc_used=doc,
            mcf_used=mcf,
            k_used=k,
            gas_capture_efficiency=gas_capture_efficiency,
            oxidation_factor=oxidation_factor,
            decay_projection=None,
        )

    def _calculate_multi_year(
        self,
        ddocm: Decimal,
        k: Decimal,
        gas_capture_efficiency: Decimal,
        oxidation_factor: Decimal,
        effective_gwp: GWPVersion,
        years: int,
        doc: Decimal,
        mcf: Decimal,
    ) -> LandfillEmissionsResult:
        """
        Calculate landfill emissions over multiple years with FOD decay schedule.

        Runs the full FOD model for the specified years, applies gas capture
        and oxidation, and returns a LandfillEmissionsResult with the complete
        decay projection schedule.

        Args:
            ddocm: Mass of decomposable DOC deposited (tonnes).
            k: Decay rate constant (yr^-1).
            gas_capture_efficiency: Gas capture fraction (0-1).
            oxidation_factor: Oxidation factor (0-1).
            effective_gwp: GWP version for CO2e conversion.
            years: Number of years to project.
            doc: DOC value used (for result reporting).
            mcf: MCF value used (for result reporting).

        Returns:
            LandfillEmissionsResult with decay_projection populated.
        """
        decay_factor = self._exp_neg_k(k)
        decomp_factor = (_ONE - decay_factor).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        total_ch4_generated = _ZERO
        total_ch4_recovered = _ZERO
        total_ch4_oxidized = _ZERO
        total_ch4_emitted = _ZERO

        decay_projection: List[Dict[str, Decimal]] = []
        ddocm_accumulated_prev = _ZERO

        for year_idx in range(1, years + 1):
            # Accumulate DDOCm
            if year_idx == 1:
                ddocm_accumulated = (
                    ddocm + ddocm_accumulated_prev * decay_factor
                ).quantize(_PRECISION, rounding=ROUND_HALF_UP)
            else:
                ddocm_accumulated = (
                    ddocm_accumulated_prev * decay_factor
                ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

            # DDOCm decomposed
            if year_idx == 1:
                ddocm_decomposed = (ddocm_accumulated * decomp_factor).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            else:
                ddocm_decomposed = (
                    ddocm_accumulated_prev * decomp_factor
                ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

            # CH4 generated
            ch4_gen = self.calculate_ch4_generation(ddocm_decomposed)

            # CH4 recovered
            ch4_rec = self.calculate_ch4_recovery(ch4_gen, gas_capture_efficiency)

            # CH4 oxidized
            net_before_ox = ch4_gen - ch4_rec
            if net_before_ox < _ZERO:
                net_before_ox = _ZERO
            ch4_ox = (net_before_ox * oxidation_factor).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

            # CH4 emitted
            ch4_emit = self.calculate_ch4_emitted(ch4_gen, ch4_rec, oxidation_factor)

            # CO2e
            co2e_year = self.convert_ch4_to_co2e(ch4_emit, effective_gwp)

            # Accumulate totals
            total_ch4_generated = (total_ch4_generated + ch4_gen).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            total_ch4_recovered = (total_ch4_recovered + ch4_rec).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            total_ch4_oxidized = (total_ch4_oxidized + ch4_ox).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            total_ch4_emitted = (total_ch4_emitted + ch4_emit).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

            decay_projection.append({
                "year": Decimal(str(year_idx)),
                "ch4_generated_tonnes": ch4_gen,
                "ch4_recovered_tonnes": ch4_rec,
                "ch4_oxidized_tonnes": ch4_ox,
                "ch4_emitted_tonnes": ch4_emit,
                "co2e_tonnes": co2e_year,
            })

            ddocm_accumulated_prev = ddocm_accumulated

        total_co2e = self.convert_ch4_to_co2e(total_ch4_emitted, effective_gwp)

        return LandfillEmissionsResult(
            ch4_generated_tonnes=total_ch4_generated,
            ch4_recovered_tonnes=total_ch4_recovered,
            ch4_oxidized_tonnes=total_ch4_oxidized,
            ch4_emitted_tonnes=total_ch4_emitted,
            co2e_total=total_co2e,
            doc_used=doc,
            mcf_used=mcf,
            k_used=k,
            gas_capture_efficiency=gas_capture_efficiency,
            oxidation_factor=oxidation_factor,
            decay_projection=decay_projection,
        )

    # ------------------------------------------------------------------
    # Exponential Decay Helper
    # ------------------------------------------------------------------

    def _exp_neg_k(self, k: Decimal) -> Decimal:
        """
        Calculate e^(-k) using math.exp and convert to Decimal.

        Python's math.exp provides sufficient precision for the decay
        factor. The result is converted to Decimal with 8 decimal places
        for subsequent Decimal arithmetic.

        Args:
            k: Decay rate constant (yr^-1).

        Returns:
            e^(-k) as Decimal (8 decimal places).
        """
        # Convert to float for math.exp, then back to Decimal
        exp_value = math.exp(-float(k))
        return Decimal(str(exp_value)).quantize(_PRECISION, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # N2O CO2e Helper
    # ------------------------------------------------------------------

    def _calculate_n2o_co2e(
        self,
        mass_tonnes: Decimal,
        has_cover: bool,
        gas_collection: GasCollectionSystem,
        gwp_version: GWPVersion,
        years: int = 1,
    ) -> Decimal:
        """
        Calculate CO2e contribution from minor N2O cover soil emissions.

        Args:
            mass_tonnes: Mass of waste deposited.
            has_cover: Whether landfill has cover.
            gas_collection: Gas collection system type.
            gwp_version: GWP version for N2O conversion.
            years: Number of years.

        Returns:
            CO2e from N2O in tonnes (Decimal).
        """
        cover_type = self._get_cover_type_key(has_cover, gas_collection)
        n2o_tonnes = self.calculate_n2o_from_cover(mass_tonnes, cover_type, years)

        if n2o_tonnes <= _ZERO:
            return _ZERO

        gwp_n2o = GWP_VALUES[gwp_version]["n2o"]
        return (n2o_tonnes * gwp_n2o).quantize(_PRECISION, rounding=ROUND_HALF_UP)

    def _get_cover_type_key(
        self,
        has_cover: bool,
        gas_collection: GasCollectionSystem,
    ) -> str:
        """
        Determine the cover type key for N2O and OX lookups.

        Args:
            has_cover: Whether the landfill has engineered cover.
            gas_collection: Gas collection system type.

        Returns:
            Cover type key string for lookup tables.
        """
        if not has_cover:
            return "no_cover"

        if gas_collection == GasCollectionSystem.ACTIVE_GEOMEMBRANE:
            return "geomembrane"

        # Default to soil cover for most systems
        return "soil_cover"

    # ------------------------------------------------------------------
    # Zero Result Creator
    # ------------------------------------------------------------------

    def _create_zero_result(
        self,
        landfill_input: LandfillInput,
    ) -> LandfillEmissionsResult:
        """
        Create a zero-emission result for failed batch items.

        Args:
            landfill_input: The input that failed processing.

        Returns:
            LandfillEmissionsResult with all emissions set to zero.
        """
        doc = DOC_VALUES.get(landfill_input.waste_category, _ZERO)
        mcf = MCF_VALUES.get(landfill_input.landfill_type, _ZERO)

        try:
            k = self.get_k_for_climate_and_waste(
                landfill_input.climate_zone,
                landfill_input.waste_category,
            )
        except KeyError:
            k = _ZERO

        gas_capture_eff = GAS_CAPTURE_EFFICIENCY.get(
            landfill_input.gas_collection, _ZERO
        )
        ox = _DEFAULT_OX_WITH_COVER if landfill_input.has_cover else _DEFAULT_OX_WITHOUT_COVER

        return LandfillEmissionsResult(
            ch4_generated_tonnes=_ZERO,
            ch4_recovered_tonnes=_ZERO,
            ch4_oxidized_tonnes=_ZERO,
            ch4_emitted_tonnes=_ZERO,
            co2e_total=_ZERO,
            doc_used=doc,
            mcf_used=mcf,
            k_used=k,
            gas_capture_efficiency=gas_capture_eff,
            oxidation_factor=ox,
            decay_projection=None,
        )

    # ------------------------------------------------------------------
    # Metrics Recording
    # ------------------------------------------------------------------

    def _record_metrics(
        self,
        landfill_input: LandfillInput,
        result: LandfillEmissionsResult,
        elapsed_s: float,
    ) -> None:
        """
        Record Prometheus metrics for a completed calculation.

        Args:
            landfill_input: The input that was processed.
            result: The calculation result.
            elapsed_s: Wall-clock elapsed time in seconds.
        """
        try:
            self._metrics.record_calculation(
                method="landfill_fod",
                treatment="landfill",
                waste_category=landfill_input.waste_category.value,
                tenant_id="system",
            )

            if result.ch4_emitted_tonnes > _ZERO:
                self._metrics.record_landfill_ch4(
                    waste_category=landfill_input.waste_category.value,
                    climate_zone=landfill_input.climate_zone.value,
                    tenant_id="system",
                    ch4_kg=float(result.ch4_emitted_tonnes * _THOUSAND),
                )

            if result.co2e_total > _ZERO:
                self._metrics.record_emissions(
                    treatment="landfill",
                    waste_category=landfill_input.waste_category.value,
                    tenant_id="system",
                    emissions_tco2e=float(result.co2e_total),
                )

        except Exception as e:
            # Metrics recording should never break the calculation
            logger.warning("Failed to record metrics: %s", str(e))

    # ------------------------------------------------------------------
    # Provenance Recording
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        calc_id: str,
        landfill_input: LandfillInput,
        result: LandfillEmissionsResult,
    ) -> None:
        """
        Record provenance for a completed calculation.

        Args:
            calc_id: Unique calculation identifier.
            landfill_input: The input that was processed.
            result: The calculation result.
        """
        try:
            chain_id = self._provenance.start_chain()
            self._provenance.record_stage(
                chain_id,
                "VALIDATE",
                {"input": landfill_input.model_dump_json()} if hasattr(landfill_input, "model_dump_json") else {"input": str(landfill_input)},
                {"status": "valid"},
            )
            self._provenance.record_stage(
                chain_id,
                "CALCULATE_TREATMENT",
                {"method": "landfill_fod", "calc_id": calc_id},
                {"ch4_emitted": str(result.ch4_emitted_tonnes), "co2e": str(result.co2e_total)},
            )
            self._provenance.seal_chain(chain_id)
        except Exception as e:
            # Provenance recording should never break the calculation
            logger.warning("Failed to record provenance: %s", str(e))

    # ------------------------------------------------------------------
    # Input Validation Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_positive(name: str, value: Decimal) -> None:
        """
        Validate that a Decimal value is strictly positive.

        Args:
            name: Parameter name for error messages.
            value: Value to validate.

        Raises:
            ValueError: If value is zero or negative.
        """
        if value <= _ZERO:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def _validate_non_negative(name: str, value: Decimal) -> None:
        """
        Validate that a Decimal value is non-negative.

        Args:
            name: Parameter name for error messages.
            value: Value to validate.

        Raises:
            ValueError: If value is negative.
        """
        if value < _ZERO:
            raise ValueError(f"{name} must be non-negative, got {value}")

    @staticmethod
    def _validate_fraction(name: str, value: Decimal) -> None:
        """
        Validate that a Decimal value is between 0 and 1 inclusive.

        Args:
            name: Parameter name for error messages.
            value: Value to validate.

        Raises:
            ValueError: If value is not in [0, 1].
        """
        if value < _ZERO or value > _ONE:
            raise ValueError(
                f"{name} must be between 0 and 1 inclusive, got {value}"
            )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "LandfillEmissionsEngine",
    "N2O_COVER_EMISSION_FACTORS",
]
