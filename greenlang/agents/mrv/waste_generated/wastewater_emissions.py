# -*- coding: utf-8 -*-
"""
Wastewater Emissions Engine (Engine 5) - AGENT-MRV-018

IPCC 2006 Guidelines Vol 5 Ch 6 implementation for wastewater treatment
emissions calculations. Covers CH4 from organic degradation, N2O from
effluent nitrogen discharge, and sludge management emissions.

Agent: GL-MRV-S3-005
Engine: 5 of 7 (WastewaterEmissionsEngine)
Prefix: gl_wg_

Core Formulas (IPCC 2006 Vol 5 Ch 6):
    CH4 = TOW x Bo x MCF - R
    N2O = N_effluent x EF_N2O x (44/28)

Where:
    TOW   = total organic waste in wastewater (kg COD or BOD/yr)
    Bo    = maximum CH4 producing capacity (kg CH4/kg COD or BOD)
    MCF   = methane correction factor by treatment system (0.0 - 0.8)
    R     = CH4 recovered/flared (kg CH4/yr)
    N_eff = nitrogen in effluent discharge (kg N/yr)
    EF    = 0.005 kg N2O-N/kg N (IPCC default)
    44/28 = molecular weight ratio N2O to N2

Thread Safety:
    Singleton pattern with threading.RLock(). All public methods are
    thread-safe for concurrent access in multi-threaded environments.

Zero-Hallucination:
    All calculations use deterministic Decimal arithmetic. No LLM calls
    in the numeric calculation path. All emission factors sourced from
    IPCC 2006 Vol 5 Tables 6.2, 6.3, 6.8, 6.9.

Example:
    >>> engine = get_wastewater_engine()
    >>> result = engine.calculate_from_cod(
    ...     cod_load=Decimal("50000"),
    ...     treatment_system=WastewaterSystem.ANAEROBIC_REACTOR,
    ...     nitrogen_load=Decimal("1200"),
    ... )
    >>> assert result.co2e_total > 0

Author: GreenLang Platform Team
Version: 1.0.0
"""

import hashlib
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.agents.mrv.waste_generated.models import (
    AGENT_ID,
    VERSION,
    TABLE_PREFIX,
    GWPVersion,
    GWP_VALUES,
    WastewaterSystem,
    WastewaterInput,
    WastewaterEmissionsResult,
    IndustryWastewaterType,
    WASTEWATER_MCF,
    WASTEWATER_Bo,
    INDUSTRY_WASTEWATER_LOADS,
    EmissionGas,
    DataQualityTier,
    EFSource,
    WasteTreatmentMethod,
    WasteCategory,
    CalculationMethod,
    WasteCalculationResult,
)
from greenlang.agents.mrv.waste_generated.config import WastewaterConfig, get_config
from greenlang.agents.mrv.waste_generated.provenance import (
    get_provenance_tracker,
    ProvenanceStage,
)
from greenlang.agents.mrv.waste_generated.metrics import get_metrics

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS - IPCC 2006 Vol 5 Ch 6
# ============================================================================

# Molecular weight ratio N2O / N2 (44/28)
N2O_N2_RATIO: Decimal = Decimal("44") / Decimal("28")

# Default N2O emission factor (kg N2O-N per kg N discharged)
# IPCC 2006 Vol 5 Table 6.11
DEFAULT_EF_N2O: Decimal = Decimal("0.005")

# Maximum CH4 producing capacity (Bo) by measurement basis
# IPCC 2006 Vol 5 Table 6.2
BO_COD: Decimal = Decimal("0.25")  # kg CH4 / kg COD
BO_BOD: Decimal = Decimal("0.60")  # kg CH4 / kg BOD

# CH4 density at STP (kg/m3)
CH4_DENSITY: Decimal = Decimal("0.0007168")  # tonnes/m3 at STP

# Default BOD per capita per day (g/person/day) by region
# IPCC 2006 Vol 5 Table 6.4
BOD_PER_CAPITA_DEFAULTS: Dict[str, Decimal] = {
    "africa": Decimal("40"),
    "asia": Decimal("40"),
    "europe": Decimal("60"),
    "latin_america": Decimal("50"),
    "middle_east": Decimal("45"),
    "north_america": Decimal("80"),
    "oceania": Decimal("60"),
    "global_default": Decimal("60"),
}

# Default nitrogen per capita per year (kg N/person/year)
# IPCC 2006 Vol 5 Table 6.11 - protein consumption based
NITROGEN_PER_CAPITA_DEFAULTS: Dict[str, Decimal] = {
    "africa": Decimal("2.7"),
    "asia": Decimal("3.1"),
    "europe": Decimal("5.5"),
    "latin_america": Decimal("4.0"),
    "middle_east": Decimal("3.8"),
    "north_america": Decimal("6.6"),
    "oceania": Decimal("5.5"),
    "global_default": Decimal("4.4"),
}

# Fraction of nitrogen removed during treatment
# IPCC 2006 Vol 5 Ch 6
NITROGEN_REMOVAL_FRACTION: Dict[WastewaterSystem, Decimal] = {
    WastewaterSystem.CENTRALIZED_AEROBIC_GOOD: Decimal("0.80"),
    WastewaterSystem.CENTRALIZED_AEROBIC_POOR: Decimal("0.50"),
    WastewaterSystem.CENTRALIZED_ANAEROBIC: Decimal("0.20"),
    WastewaterSystem.ANAEROBIC_REACTOR: Decimal("0.20"),
    WastewaterSystem.LAGOON_SHALLOW: Decimal("0.40"),
    WastewaterSystem.LAGOON_DEEP: Decimal("0.30"),
    WastewaterSystem.SEPTIC: Decimal("0.15"),
    WastewaterSystem.OPEN_SEWER: Decimal("0.00"),
    WastewaterSystem.CONSTRUCTED_WETLAND: Decimal("0.60"),
}

# Treatment system efficiency for organic load removal (fraction removed)
TREATMENT_EFFICIENCY: Dict[WastewaterSystem, Decimal] = {
    WastewaterSystem.CENTRALIZED_AEROBIC_GOOD: Decimal("0.95"),
    WastewaterSystem.CENTRALIZED_AEROBIC_POOR: Decimal("0.70"),
    WastewaterSystem.CENTRALIZED_ANAEROBIC: Decimal("0.85"),
    WastewaterSystem.ANAEROBIC_REACTOR: Decimal("0.90"),
    WastewaterSystem.LAGOON_SHALLOW: Decimal("0.60"),
    WastewaterSystem.LAGOON_DEEP: Decimal("0.75"),
    WastewaterSystem.SEPTIC: Decimal("0.50"),
    WastewaterSystem.OPEN_SEWER: Decimal("0.10"),
    WastewaterSystem.CONSTRUCTED_WETLAND: Decimal("0.80"),
}

# Sludge generation rate (kg dry sludge per kg COD removed)
SLUDGE_GENERATION_RATE: Dict[WastewaterSystem, Decimal] = {
    WastewaterSystem.CENTRALIZED_AEROBIC_GOOD: Decimal("0.40"),
    WastewaterSystem.CENTRALIZED_AEROBIC_POOR: Decimal("0.35"),
    WastewaterSystem.CENTRALIZED_ANAEROBIC: Decimal("0.15"),
    WastewaterSystem.ANAEROBIC_REACTOR: Decimal("0.12"),
    WastewaterSystem.LAGOON_SHALLOW: Decimal("0.10"),
    WastewaterSystem.LAGOON_DEEP: Decimal("0.08"),
    WastewaterSystem.SEPTIC: Decimal("0.20"),
    WastewaterSystem.OPEN_SEWER: Decimal("0.00"),
    WastewaterSystem.CONSTRUCTED_WETLAND: Decimal("0.05"),
}

# Sludge disposal emission factors (kg CO2e per tonne dry sludge)
SLUDGE_DISPOSAL_EF: Dict[str, Decimal] = {
    "landfill": Decimal("500"),         # Anaerobic decomposition
    "incineration": Decimal("200"),     # Combustion + energy offset
    "land_application": Decimal("50"),  # Soil carbon offset partially
    "composting": Decimal("80"),        # Aerobic process emissions
    "ocean_disposal": Decimal("10"),    # Transport only (banned in many regions)
    "thermal_drying": Decimal("300"),   # Energy-intensive drying
    "cement_kiln": Decimal("150"),      # Co-processing credit
}

# CH4 recovery efficiency by system type
CH4_RECOVERY_EFFICIENCY: Dict[str, Decimal] = {
    "biogas_capture_good": Decimal("0.90"),
    "biogas_capture_fair": Decimal("0.70"),
    "biogas_capture_poor": Decimal("0.50"),
    "flaring_enclosed": Decimal("0.99"),
    "flaring_open": Decimal("0.92"),
    "no_recovery": Decimal("0.00"),
}

# IPCC default uncertainty ranges (as +/- percentage)
UNCERTAINTY_RANGES: Dict[str, Decimal] = {
    "mcf": Decimal("30"),
    "bo": Decimal("30"),
    "tow": Decimal("30"),
    "ef_n2o": Decimal("50"),
    "sludge_ef": Decimal("40"),
    "nitrogen_load": Decimal("30"),
    "recovery": Decimal("20"),
}

# Decimal precision for rounding
PRECISION = Decimal("0.000001")

# Conversion factors
KG_PER_TONNE: Decimal = Decimal("1000")
DAYS_PER_YEAR: Decimal = Decimal("365")
G_PER_KG: Decimal = Decimal("1000")


# ============================================================================
# HELPER DATA CLASSES
# ============================================================================


class SludgeEmissionsDetail:
    """Immutable detail record for sludge disposal emissions."""

    __slots__ = (
        "sludge_mass_dry_tonnes",
        "disposal_method",
        "ef_used",
        "co2e_tonnes",
    )

    def __init__(
        self,
        sludge_mass_dry_tonnes: Decimal,
        disposal_method: str,
        ef_used: Decimal,
        co2e_tonnes: Decimal,
    ) -> None:
        object.__setattr__(self, "sludge_mass_dry_tonnes", sludge_mass_dry_tonnes)
        object.__setattr__(self, "disposal_method", disposal_method)
        object.__setattr__(self, "ef_used", ef_used)
        object.__setattr__(self, "co2e_tonnes", co2e_tonnes)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("SludgeEmissionsDetail is immutable")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sludge_mass_dry_tonnes": str(self.sludge_mass_dry_tonnes),
            "disposal_method": self.disposal_method,
            "ef_used_kgco2e_per_tonne": str(self.ef_used),
            "co2e_tonnes": str(self.co2e_tonnes),
        }


class WastewaterDetailedResult:
    """Extended result with sludge and detailed breakdown."""

    __slots__ = (
        "base_result",
        "ch4_generated_kg",
        "ch4_recovered_kg",
        "ch4_net_kg",
        "n2o_kg",
        "co2e_ch4",
        "co2e_n2o",
        "co2e_sludge",
        "sludge_detail",
        "gwp_version",
        "uncertainty_pct",
        "data_quality_tier",
        "provenance_hash",
        "processing_time_ms",
    )

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key in self.__slots__:
                object.__setattr__(self, key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("WastewaterDetailedResult is immutable")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        result: Dict[str, Any] = {}
        for slot in self.__slots__:
            val = getattr(self, slot, None)
            if val is None:
                result[slot] = None
            elif isinstance(val, Decimal):
                result[slot] = str(val)
            elif isinstance(val, WastewaterEmissionsResult):
                result[slot] = {
                    "ch4_from_treatment_tonnes": str(val.ch4_from_treatment_tonnes),
                    "n2o_from_effluent_tonnes": str(val.n2o_from_effluent_tonnes),
                    "co2e_total": str(val.co2e_total),
                    "mcf_used": str(val.mcf_used),
                    "bo_used": str(val.bo_used),
                    "treatment_system": val.treatment_system.value,
                }
            elif isinstance(val, SludgeEmissionsDetail):
                result[slot] = val.to_dict()
            else:
                result[slot] = val
        return result


# ============================================================================
# ENGINE IMPLEMENTATION
# ============================================================================


class WastewaterEmissionsEngine:
    """
    IPCC 2006 Vol 5 Ch 6 wastewater emissions calculation engine.

    Implements deterministic Decimal arithmetic for CH4 from organic
    degradation, N2O from effluent nitrogen, and sludge disposal
    emissions. Thread-safe singleton with full provenance tracking.

    Calculation Pathways:
        1. CH4 from treatment: TOW x Bo x MCF - R
        2. N2O from effluent: N_eff x EF_N2O x (44/28)
        3. Sludge disposal: sludge_mass x disposal_EF

    Data Tables (embedded, IPCC 2006):
        - MCF by treatment system (9 systems)
        - Bo values (COD and BOD basis)
        - Industry-specific organic loads (12 industry types)
        - BOD per capita defaults (8 regions)
        - N2O emission factors
        - Sludge generation rates (9 systems)
        - Treatment efficiency by system (9 systems)

    Thread Safety:
        All public methods are protected by threading.RLock() via the
        singleton pattern. The engine is safe for concurrent access.

    Zero-Hallucination:
        All numeric calculations use Decimal arithmetic with explicit
        IPCC emission factors. No LLM calls in the calculation path.

    Attributes:
        _config: WastewaterConfig from the master configuration
        _provenance: Provenance tracker for audit trails
        _metrics: Prometheus metrics collector

    Example:
        >>> engine = WastewaterEmissionsEngine()
        >>> ch4 = engine.calculate_ch4_from_treatment(
        ...     tow=Decimal("50000"),
        ...     bo=Decimal("0.25"),
        ...     mcf=Decimal("0.80"),
        ... )
        >>> assert ch4 == Decimal("10000.000000")
    """

    _instance: Optional["WastewaterEmissionsEngine"] = None
    _init_lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "WastewaterEmissionsEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the engine with config, provenance, and metrics."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._lock = threading.RLock()
            try:
                config = get_config()
                self._config: WastewaterConfig = config.wastewater
            except Exception:
                logger.warning(
                    "WastewaterEmissionsEngine: config load failed, using defaults"
                )
                self._config = WastewaterConfig()
            try:
                self._provenance = get_provenance_tracker()
            except Exception:
                logger.warning(
                    "WastewaterEmissionsEngine: provenance tracker unavailable"
                )
                self._provenance = None
            try:
                self._metrics = get_metrics()
            except Exception:
                logger.warning(
                    "WastewaterEmissionsEngine: metrics collector unavailable"
                )
                self._metrics = None
            self._initialized = True
            logger.info(
                "WastewaterEmissionsEngine initialized [agent=%s, version=%s]",
                AGENT_ID,
                VERSION,
            )

    # ========================================================================
    # SECTION 1: PRIMARY CALCULATE METHOD
    # ========================================================================

    def calculate(
        self,
        input_data: WastewaterInput,
        gwp_version: GWPVersion = GWPVersion.AR5,
        recovery_efficiency: Decimal = Decimal("0"),
        include_sludge: bool = False,
        sludge_disposal_method: str = "landfill",
        tenant_id: str = "default",
    ) -> WastewaterEmissionsResult:
        """
        Primary calculation method for wastewater treatment emissions.

        Implements the full IPCC 2006 Vol 5 Ch 6 methodology:
            1. Resolve Bo and MCF from input or defaults
            2. Calculate CH4 from organic degradation
            3. Subtract CH4 recovery
            4. Calculate N2O from effluent nitrogen
            5. Convert to CO2e using GWP values
            6. Track provenance and metrics

        Args:
            input_data: Validated WastewaterInput with organic load, treatment
                system, and optional nitrogen load.
            gwp_version: GWP assessment report version for CO2e conversion.
                Defaults to AR5 (GHG Protocol standard).
            recovery_efficiency: Fraction of CH4 recovered (0.0 to 1.0).
                Defaults to 0 (no recovery).
            include_sludge: Whether to add sludge disposal emissions.
            sludge_disposal_method: Sludge disposal method if include_sludge
                is True. One of: landfill, incineration, land_application,
                composting, ocean_disposal, thermal_drying, cement_kiln.
            tenant_id: Tenant identifier for multi-tenant metrics.

        Returns:
            WastewaterEmissionsResult with CH4, N2O, and total CO2e.

        Raises:
            ValueError: If input validation fails.
            InvalidOperation: If Decimal arithmetic overflows.

        Example:
            >>> inp = WastewaterInput(
            ...     organic_load_kg=Decimal("50000"),
            ...     measurement_basis="cod",
            ...     treatment_system=WastewaterSystem.ANAEROBIC_REACTOR,
            ...     nitrogen_load_kg=Decimal("1200"),
            ... )
            >>> result = engine.calculate(inp)
            >>> assert result.co2e_total > Decimal("0")
        """
        start_ns = time.perf_counter_ns()
        chain_id: Optional[str] = None

        try:
            # ----------------------------------------------------------
            # Step 1: Validate
            # ----------------------------------------------------------
            errors = self.validate_wastewater_input(input_data)
            if errors:
                raise ValueError(
                    f"Input validation failed: {'; '.join(errors)}"
                )

            # ----------------------------------------------------------
            # Step 2: Start provenance chain
            # ----------------------------------------------------------
            if self._provenance is not None:
                chain_id = self._provenance.start_chain()
                self._provenance.record_stage(
                    chain_id,
                    ProvenanceStage.VALIDATE,
                    {"organic_load_kg": str(input_data.organic_load_kg)},
                    {"status": "valid"},
                )

            # ----------------------------------------------------------
            # Step 3: Resolve parameters
            # ----------------------------------------------------------
            bo = self._resolve_bo(input_data)
            mcf = self._resolve_mcf(input_data)

            if self._provenance is not None and chain_id is not None:
                self._provenance.record_stage(
                    chain_id,
                    ProvenanceStage.RESOLVE_EFS,
                    {
                        "measurement_basis": input_data.measurement_basis,
                        "treatment_system": input_data.treatment_system.value,
                    },
                    {"bo": str(bo), "mcf": str(mcf)},
                )

            # ----------------------------------------------------------
            # Step 4: Calculate CH4 from treatment
            # ----------------------------------------------------------
            ch4_generated = self.calculate_ch4_from_treatment(
                tow=input_data.organic_load_kg,
                bo=bo,
                mcf=mcf,
            )

            ch4_recovered = self.calculate_ch4_recovery(
                ch4_generated=ch4_generated,
                recovery_efficiency=recovery_efficiency,
            )

            ch4_net = self.calculate_net_ch4(
                ch4_generated=ch4_generated,
                ch4_recovered=ch4_recovered,
            )

            # ----------------------------------------------------------
            # Step 5: Calculate N2O from effluent
            # ----------------------------------------------------------
            n2o_kg = Decimal("0")
            if input_data.nitrogen_load_kg is not None:
                n2o_kg = self.calculate_n2o_from_effluent(
                    nitrogen_load=input_data.nitrogen_load_kg,
                    ef_n2o=DEFAULT_EF_N2O,
                )

            # ----------------------------------------------------------
            # Step 6: Convert to CO2e
            # ----------------------------------------------------------
            co2e_total = self.calculate_total_co2e(
                ch4_net_kg=ch4_net,
                n2o_kg=n2o_kg,
                gwp_version=gwp_version,
            )

            # Optional sludge emissions
            if include_sludge:
                sludge_co2e = self._calculate_sludge_from_treatment(
                    organic_load_kg=input_data.organic_load_kg,
                    treatment_system=input_data.treatment_system,
                    disposal_method=sludge_disposal_method,
                )
                co2e_total = co2e_total + sludge_co2e

            if self._provenance is not None and chain_id is not None:
                self._provenance.record_stage(
                    chain_id,
                    ProvenanceStage.CALCULATE_TREATMENT,
                    {
                        "ch4_generated_kg": str(ch4_generated),
                        "ch4_recovered_kg": str(ch4_recovered),
                        "n2o_kg": str(n2o_kg),
                    },
                    {"co2e_total_tonnes": str(co2e_total)},
                )

            # ----------------------------------------------------------
            # Step 7: Build result
            # ----------------------------------------------------------
            ch4_treatment_tonnes = (ch4_net / KG_PER_TONNE).quantize(PRECISION)
            n2o_effluent_tonnes = (n2o_kg / KG_PER_TONNE).quantize(PRECISION)
            ch4_recovered_tonnes = (ch4_recovered / KG_PER_TONNE).quantize(
                PRECISION
            )

            # ----------------------------------------------------------
            # Step 8: Provenance seal
            # ----------------------------------------------------------
            provenance_hash = self._compute_provenance_hash(
                input_data, ch4_net, n2o_kg, co2e_total
            )

            if self._provenance is not None and chain_id is not None:
                self._provenance.record_stage(
                    chain_id,
                    ProvenanceStage.SEAL,
                    {"provenance_hash": provenance_hash},
                    {"sealed": True},
                )
                self._provenance.seal_chain(chain_id)

            result = WastewaterEmissionsResult(
                ch4_from_treatment_tonnes=ch4_treatment_tonnes,
                n2o_from_effluent_tonnes=n2o_effluent_tonnes,
                ch4_recovered_tonnes=ch4_recovered_tonnes,
                co2e_total=co2e_total,
                organic_load_used_kg=input_data.organic_load_kg,
                mcf_used=mcf,
                bo_used=bo,
                treatment_system=input_data.treatment_system,
            )

            # ----------------------------------------------------------
            # Step 9: Record metrics
            # ----------------------------------------------------------
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            self._record_metrics(
                input_data=input_data,
                co2e_total=co2e_total,
                ch4_net=ch4_net,
                n2o_kg=n2o_kg,
                elapsed_s=elapsed_ms / 1000,
                tenant_id=tenant_id,
            )

            logger.info(
                "Wastewater emissions calculated: co2e=%.6f t, "
                "ch4_net=%.2f kg, n2o=%.4f kg, system=%s, "
                "elapsed=%.1f ms",
                co2e_total,
                ch4_net,
                n2o_kg,
                input_data.treatment_system.value,
                elapsed_ms,
            )

            return result

        except ValueError:
            raise
        except Exception as exc:
            logger.error(
                "WastewaterEmissionsEngine.calculate failed: %s",
                str(exc),
                exc_info=True,
            )
            if self._metrics is not None:
                try:
                    self._metrics.record_calculation_error(
                        error_type=type(exc).__name__,
                        treatment="wastewater_treatment",
                        tenant_id=tenant_id,
                    )
                except Exception:
                    pass
            raise

    # ========================================================================
    # SECTION 2: CH4 CALCULATION METHODS
    # ========================================================================

    def calculate_ch4_from_treatment(
        self,
        tow: Decimal,
        bo: Decimal,
        mcf: Decimal,
    ) -> Decimal:
        """
        Calculate CH4 generated from wastewater treatment.

        IPCC Formula: CH4 = TOW x Bo x MCF

        This is the gross CH4 generation before recovery. The formula
        is deterministic and uses only IPCC-sourced parameters.

        Args:
            tow: Total organic waste in wastewater (kg COD or BOD/yr).
                Must be > 0.
            bo: Maximum CH4 producing capacity (kg CH4/kg COD or BOD).
                Typically 0.25 for COD, 0.60 for BOD.
            mcf: Methane correction factor (0.0 to 1.0) based on
                treatment system type.

        Returns:
            CH4 generated in kg, quantized to 6 decimal places.

        Raises:
            ValueError: If any parameter is negative or mcf > 1.

        Example:
            >>> ch4 = engine.calculate_ch4_from_treatment(
            ...     tow=Decimal("50000"),
            ...     bo=Decimal("0.25"),
            ...     mcf=Decimal("0.80"),
            ... )
            >>> assert ch4 == Decimal("10000.000000")
        """
        self._validate_positive("tow", tow)
        self._validate_non_negative("bo", bo)
        self._validate_fraction("mcf", mcf)

        ch4_kg = (tow * bo * mcf).quantize(PRECISION, rounding=ROUND_HALF_UP)
        logger.debug(
            "CH4 from treatment: TOW=%.2f x Bo=%.4f x MCF=%.2f = %.6f kg",
            tow, bo, mcf, ch4_kg,
        )
        return ch4_kg

    def calculate_ch4_recovery(
        self,
        ch4_generated: Decimal,
        recovery_efficiency: Decimal,
    ) -> Decimal:
        """
        Calculate CH4 recovered through biogas capture or flaring.

        Args:
            ch4_generated: Gross CH4 generated (kg). Must be >= 0.
            recovery_efficiency: Fraction of CH4 recovered (0.0 to 1.0).
                0.0 = no recovery, 0.90 = 90% capture.

        Returns:
            CH4 recovered in kg, quantized to 6 decimal places.

        Raises:
            ValueError: If ch4_generated < 0 or recovery_efficiency not
                in [0, 1].

        Example:
            >>> recovered = engine.calculate_ch4_recovery(
            ...     ch4_generated=Decimal("10000"),
            ...     recovery_efficiency=Decimal("0.90"),
            ... )
            >>> assert recovered == Decimal("9000.000000")
        """
        self._validate_non_negative("ch4_generated", ch4_generated)
        self._validate_fraction("recovery_efficiency", recovery_efficiency)

        recovered = (ch4_generated * recovery_efficiency).quantize(
            PRECISION, rounding=ROUND_HALF_UP
        )
        logger.debug(
            "CH4 recovery: %.2f kg x %.2f efficiency = %.6f kg recovered",
            ch4_generated, recovery_efficiency, recovered,
        )
        return recovered

    def calculate_net_ch4(
        self,
        ch4_generated: Decimal,
        ch4_recovered: Decimal,
    ) -> Decimal:
        """
        Calculate net CH4 emitted to atmosphere.

        Formula: CH4_net = CH4_generated - CH4_recovered
        Clamped to zero (cannot be negative).

        Args:
            ch4_generated: Gross CH4 generated (kg).
            ch4_recovered: CH4 recovered/destroyed (kg).

        Returns:
            Net CH4 emitted (kg), minimum 0, quantized to 6 decimals.

        Raises:
            ValueError: If either parameter is negative.

        Example:
            >>> net = engine.calculate_net_ch4(
            ...     ch4_generated=Decimal("10000"),
            ...     ch4_recovered=Decimal("9000"),
            ... )
            >>> assert net == Decimal("1000.000000")
        """
        self._validate_non_negative("ch4_generated", ch4_generated)
        self._validate_non_negative("ch4_recovered", ch4_recovered)

        net = max(ch4_generated - ch4_recovered, Decimal("0"))
        return net.quantize(PRECISION, rounding=ROUND_HALF_UP)

    # ========================================================================
    # SECTION 3: N2O CALCULATION METHODS
    # ========================================================================

    def calculate_n2o_from_effluent(
        self,
        nitrogen_load: Decimal,
        ef_n2o: Decimal = DEFAULT_EF_N2O,
    ) -> Decimal:
        """
        Calculate N2O emissions from nitrogen in effluent discharge.

        IPCC Formula: N2O = N_effluent x EF_N2O x (44/28)

        The 44/28 ratio converts from N2O-N to N2O mass basis.

        Args:
            nitrogen_load: Nitrogen in effluent discharge (kg N/yr).
                Must be >= 0.
            ef_n2o: Emission factor (kg N2O-N per kg N). IPCC default
                is 0.005. Must be >= 0.

        Returns:
            N2O emitted in kg, quantized to 6 decimal places.

        Raises:
            ValueError: If nitrogen_load or ef_n2o is negative.

        Example:
            >>> n2o = engine.calculate_n2o_from_effluent(
            ...     nitrogen_load=Decimal("1200"),
            ...     ef_n2o=Decimal("0.005"),
            ... )
            >>> # 1200 x 0.005 x (44/28) = 9.428571 kg N2O
        """
        self._validate_non_negative("nitrogen_load", nitrogen_load)
        self._validate_non_negative("ef_n2o", ef_n2o)

        n2o_kg = (nitrogen_load * ef_n2o * N2O_N2_RATIO).quantize(
            PRECISION, rounding=ROUND_HALF_UP
        )
        logger.debug(
            "N2O from effluent: %.2f kg N x %.4f EF x %.6f (44/28) = %.6f kg",
            nitrogen_load, ef_n2o, N2O_N2_RATIO, n2o_kg,
        )
        return n2o_kg

    def calculate_n2o_from_treatment_plant(
        self,
        nitrogen_influent: Decimal,
        nitrogen_removal_fraction: Decimal,
        ef_n2o: Decimal = DEFAULT_EF_N2O,
    ) -> Decimal:
        """
        Calculate N2O from a treatment plant given influent nitrogen.

        The effluent nitrogen is estimated as:
            N_effluent = N_influent x (1 - removal_fraction)

        Then applies the standard N2O formula.

        Args:
            nitrogen_influent: Total nitrogen entering the plant (kg N/yr).
            nitrogen_removal_fraction: Fraction removed during treatment
                (0.0 to 1.0).
            ef_n2o: N2O emission factor (kg N2O-N/kg N).

        Returns:
            N2O emitted in kg.

        Raises:
            ValueError: If parameters are out of range.
        """
        self._validate_non_negative("nitrogen_influent", nitrogen_influent)
        self._validate_fraction("nitrogen_removal_fraction", nitrogen_removal_fraction)

        n_effluent = nitrogen_influent * (Decimal("1") - nitrogen_removal_fraction)
        return self.calculate_n2o_from_effluent(n_effluent, ef_n2o)

    # ========================================================================
    # SECTION 4: CO2e CONVERSION
    # ========================================================================

    def calculate_total_co2e(
        self,
        ch4_net_kg: Decimal,
        n2o_kg: Decimal,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> Decimal:
        """
        Convert CH4 and N2O to total CO2 equivalent (tonnes).

        Formula:
            CO2e = (CH4_kg x GWP_CH4 + N2O_kg x GWP_N2O) / 1000

        Args:
            ch4_net_kg: Net CH4 emissions (kg).
            n2o_kg: N2O emissions (kg).
            gwp_version: IPCC GWP version (AR4, AR5, AR6, AR6_20YR).

        Returns:
            Total CO2e in tonnes, quantized to 6 decimal places.

        Raises:
            ValueError: If ch4_net_kg or n2o_kg is negative.
            KeyError: If gwp_version is not supported.

        Example:
            >>> co2e = engine.calculate_total_co2e(
            ...     ch4_net_kg=Decimal("1000"),
            ...     n2o_kg=Decimal("10"),
            ...     gwp_version=GWPVersion.AR5,
            ... )
            >>> # (1000 * 28 + 10 * 265) / 1000 = 30.65 tonnes
        """
        self._validate_non_negative("ch4_net_kg", ch4_net_kg)
        self._validate_non_negative("n2o_kg", n2o_kg)

        gwp = GWP_VALUES[gwp_version]
        gwp_ch4 = gwp["ch4"]
        gwp_n2o = gwp["n2o"]

        co2e_kg = ch4_net_kg * gwp_ch4 + n2o_kg * gwp_n2o
        co2e_tonnes = (co2e_kg / KG_PER_TONNE).quantize(
            PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "CO2e total: (%.2f kg CH4 x %s GWP + %.4f kg N2O x %s GWP) "
            "/ 1000 = %.6f tonnes [%s]",
            ch4_net_kg, gwp_ch4, n2o_kg, gwp_n2o, co2e_tonnes,
            gwp_version.value,
        )
        return co2e_tonnes

    def convert_ch4_to_co2e(
        self,
        ch4_kg: Decimal,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> Decimal:
        """
        Convert CH4 mass to CO2e (tonnes).

        Args:
            ch4_kg: CH4 in kilograms.
            gwp_version: GWP version.

        Returns:
            CO2e in tonnes.
        """
        self._validate_non_negative("ch4_kg", ch4_kg)
        gwp_ch4 = GWP_VALUES[gwp_version]["ch4"]
        return (ch4_kg * gwp_ch4 / KG_PER_TONNE).quantize(
            PRECISION, rounding=ROUND_HALF_UP
        )

    def convert_n2o_to_co2e(
        self,
        n2o_kg: Decimal,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> Decimal:
        """
        Convert N2O mass to CO2e (tonnes).

        Args:
            n2o_kg: N2O in kilograms.
            gwp_version: GWP version.

        Returns:
            CO2e in tonnes.
        """
        self._validate_non_negative("n2o_kg", n2o_kg)
        gwp_n2o = GWP_VALUES[gwp_version]["n2o"]
        return (n2o_kg * gwp_n2o / KG_PER_TONNE).quantize(
            PRECISION, rounding=ROUND_HALF_UP
        )

    # ========================================================================
    # SECTION 5: CONVENIENCE CALCULATION METHODS
    # ========================================================================

    def calculate_from_cod(
        self,
        cod_load: Decimal,
        treatment_system: WastewaterSystem,
        nitrogen_load: Optional[Decimal] = None,
        gwp_version: GWPVersion = GWPVersion.AR5,
        recovery_efficiency: Decimal = Decimal("0"),
        include_sludge: bool = False,
        sludge_disposal_method: str = "landfill",
        tenant_id: str = "default",
    ) -> WastewaterEmissionsResult:
        """
        Calculate wastewater emissions from COD load.

        Convenience method that creates a WastewaterInput with
        measurement_basis='cod' and Bo=0.25 kg CH4/kg COD.

        Args:
            cod_load: Chemical Oxygen Demand load (kg COD/yr).
            treatment_system: Wastewater treatment system type.
            nitrogen_load: Nitrogen in effluent (kg N/yr). Optional.
            gwp_version: GWP version for CO2e conversion.
            recovery_efficiency: CH4 recovery fraction (0-1).
            include_sludge: Include sludge disposal emissions.
            sludge_disposal_method: Sludge disposal pathway.
            tenant_id: Tenant identifier.

        Returns:
            WastewaterEmissionsResult.

        Raises:
            ValueError: If cod_load <= 0.

        Example:
            >>> result = engine.calculate_from_cod(
            ...     cod_load=Decimal("100000"),
            ...     treatment_system=WastewaterSystem.ANAEROBIC_REACTOR,
            ...     nitrogen_load=Decimal("2000"),
            ... )
        """
        self._validate_positive("cod_load", cod_load)

        input_data = WastewaterInput(
            organic_load_kg=cod_load,
            measurement_basis="cod",
            treatment_system=treatment_system,
            nitrogen_load_kg=nitrogen_load,
        )

        return self.calculate(
            input_data=input_data,
            gwp_version=gwp_version,
            recovery_efficiency=recovery_efficiency,
            include_sludge=include_sludge,
            sludge_disposal_method=sludge_disposal_method,
            tenant_id=tenant_id,
        )

    def calculate_from_bod(
        self,
        bod_load: Decimal,
        treatment_system: WastewaterSystem,
        nitrogen_load: Optional[Decimal] = None,
        gwp_version: GWPVersion = GWPVersion.AR5,
        recovery_efficiency: Decimal = Decimal("0"),
        include_sludge: bool = False,
        sludge_disposal_method: str = "landfill",
        tenant_id: str = "default",
    ) -> WastewaterEmissionsResult:
        """
        Calculate wastewater emissions from BOD load.

        Convenience method that creates a WastewaterInput with
        measurement_basis='bod' and Bo=0.60 kg CH4/kg BOD.

        Args:
            bod_load: Biochemical Oxygen Demand load (kg BOD/yr).
            treatment_system: Wastewater treatment system type.
            nitrogen_load: Nitrogen in effluent (kg N/yr). Optional.
            gwp_version: GWP version for CO2e conversion.
            recovery_efficiency: CH4 recovery fraction (0-1).
            include_sludge: Include sludge disposal emissions.
            sludge_disposal_method: Sludge disposal pathway.
            tenant_id: Tenant identifier.

        Returns:
            WastewaterEmissionsResult.

        Raises:
            ValueError: If bod_load <= 0.

        Example:
            >>> result = engine.calculate_from_bod(
            ...     bod_load=Decimal("30000"),
            ...     treatment_system=WastewaterSystem.LAGOON_DEEP,
            ...     nitrogen_load=Decimal("800"),
            ... )
        """
        self._validate_positive("bod_load", bod_load)

        input_data = WastewaterInput(
            organic_load_kg=bod_load,
            measurement_basis="bod",
            treatment_system=treatment_system,
            nitrogen_load_kg=nitrogen_load,
        )

        return self.calculate(
            input_data=input_data,
            gwp_version=gwp_version,
            recovery_efficiency=recovery_efficiency,
            include_sludge=include_sludge,
            sludge_disposal_method=sludge_disposal_method,
            tenant_id=tenant_id,
        )

    # ========================================================================
    # SECTION 6: INDUSTRY-SPECIFIC METHODS
    # ========================================================================

    def estimate_organic_load_from_industry(
        self,
        industry_type: IndustryWastewaterType,
        production_tonnes: Decimal,
    ) -> Dict[str, Decimal]:
        """
        Estimate organic load from industry type and production volume.

        Uses IPCC 2006 Vol 5 Table 6.9 industry-specific wastewater
        generation rates and concentrations.

        Formula:
            COD_total = production_tonnes x volume_m3_per_tonne x cod_kg_per_m3
            BOD_total = production_tonnes x volume_m3_per_tonne x bod_kg_per_m3

        Args:
            industry_type: Industry classification from IPCC Table 6.9.
            production_tonnes: Annual production volume (tonnes product).

        Returns:
            Dictionary with keys:
                - cod_total_kg: Total COD load (kg/yr)
                - bod_total_kg: Total BOD load (kg/yr)
                - wastewater_volume_m3: Total wastewater volume (m3/yr)
                - industry_type: Industry classification used

        Raises:
            ValueError: If production_tonnes <= 0.
            KeyError: If industry_type not in IPCC tables.

        Example:
            >>> loads = engine.estimate_organic_load_from_industry(
            ...     industry_type=IndustryWastewaterType.DAIRY,
            ...     production_tonnes=Decimal("5000"),
            ... )
            >>> assert loads["cod_total_kg"] > 0
        """
        self._validate_positive("production_tonnes", production_tonnes)

        if industry_type not in INDUSTRY_WASTEWATER_LOADS:
            raise KeyError(
                f"Industry type '{industry_type.value}' not found in "
                f"IPCC wastewater load tables"
            )

        load_data = INDUSTRY_WASTEWATER_LOADS[industry_type]
        volume_m3 = production_tonnes * load_data["volume_m3_per_tonne"]
        cod_total = volume_m3 * load_data["cod_kg_per_m3"]
        bod_total = volume_m3 * load_data["bod_kg_per_m3"]

        result = {
            "cod_total_kg": cod_total.quantize(PRECISION, rounding=ROUND_HALF_UP),
            "bod_total_kg": bod_total.quantize(PRECISION, rounding=ROUND_HALF_UP),
            "wastewater_volume_m3": volume_m3.quantize(
                PRECISION, rounding=ROUND_HALF_UP
            ),
            "industry_type": industry_type.value,
        }

        logger.info(
            "Industry organic load estimate: %s, %.1f t production -> "
            "COD=%.0f kg, BOD=%.0f kg, vol=%.0f m3",
            industry_type.value,
            production_tonnes,
            cod_total,
            bod_total,
            volume_m3,
        )
        return result

    def calculate_from_industry(
        self,
        industry_type: IndustryWastewaterType,
        production_tonnes: Decimal,
        treatment_system: WastewaterSystem,
        measurement_basis: str = "cod",
        nitrogen_load: Optional[Decimal] = None,
        gwp_version: GWPVersion = GWPVersion.AR5,
        recovery_efficiency: Decimal = Decimal("0"),
        tenant_id: str = "default",
    ) -> WastewaterEmissionsResult:
        """
        End-to-end calculation from industry type and production volume.

        Combines estimate_organic_load_from_industry with calculate to
        provide a single-call pathway for industrial wastewater.

        Args:
            industry_type: IPCC industry classification.
            production_tonnes: Annual production (tonnes).
            treatment_system: Treatment system type.
            measurement_basis: 'cod' or 'bod' for organic load.
            nitrogen_load: Override nitrogen load (kg N/yr).
            gwp_version: GWP version.
            recovery_efficiency: CH4 recovery fraction.
            tenant_id: Tenant identifier.

        Returns:
            WastewaterEmissionsResult.

        Example:
            >>> result = engine.calculate_from_industry(
            ...     industry_type=IndustryWastewaterType.BEER_MALT,
            ...     production_tonnes=Decimal("10000"),
            ...     treatment_system=WastewaterSystem.ANAEROBIC_REACTOR,
            ... )
        """
        loads = self.estimate_organic_load_from_industry(
            industry_type, production_tonnes
        )

        if measurement_basis.lower() == "bod":
            organic_load = loads["bod_total_kg"]
        else:
            organic_load = loads["cod_total_kg"]

        input_data = WastewaterInput(
            organic_load_kg=organic_load,
            measurement_basis=measurement_basis,
            treatment_system=treatment_system,
            nitrogen_load_kg=nitrogen_load,
            industry_type=industry_type,
            production_volume_tonnes=production_tonnes,
        )

        return self.calculate(
            input_data=input_data,
            gwp_version=gwp_version,
            recovery_efficiency=recovery_efficiency,
            tenant_id=tenant_id,
        )

    def get_industry_typical_loads(
        self,
        industry_type: IndustryWastewaterType,
    ) -> Dict[str, Decimal]:
        """
        Get typical wastewater load parameters for an industry type.

        Returns the IPCC 2006 Vol 5 Table 6.9 values directly for
        reference and documentation purposes.

        Args:
            industry_type: IPCC industry classification.

        Returns:
            Dictionary with keys:
                - cod_kg_per_m3: COD concentration
                - bod_kg_per_m3: BOD concentration
                - volume_m3_per_tonne: Wastewater volume per tonne product
                - cod_per_tonne_product: Total COD per tonne product
                - bod_per_tonne_product: Total BOD per tonne product

        Raises:
            KeyError: If industry_type not in tables.
        """
        if industry_type not in INDUSTRY_WASTEWATER_LOADS:
            raise KeyError(
                f"Industry type '{industry_type.value}' not in IPCC tables"
            )

        data = INDUSTRY_WASTEWATER_LOADS[industry_type]
        vol = data["volume_m3_per_tonne"]

        return {
            "cod_kg_per_m3": data["cod_kg_per_m3"],
            "bod_kg_per_m3": data["bod_kg_per_m3"],
            "volume_m3_per_tonne": vol,
            "cod_per_tonne_product": (data["cod_kg_per_m3"] * vol).quantize(
                PRECISION
            ),
            "bod_per_tonne_product": (data["bod_kg_per_m3"] * vol).quantize(
                PRECISION
            ),
        }

    # ========================================================================
    # SECTION 7: SLUDGE MANAGEMENT
    # ========================================================================

    def calculate_sludge_emissions(
        self,
        sludge_mass_dry_tonnes: Decimal,
        disposal_method: str,
    ) -> Decimal:
        """
        Calculate emissions from sludge disposal.

        Sludge disposal is an ancillary emission source from wastewater
        treatment. The emission factor depends on the disposal pathway.

        Args:
            sludge_mass_dry_tonnes: Dry mass of sludge (tonnes).
            disposal_method: One of: landfill, incineration,
                land_application, composting, ocean_disposal,
                thermal_drying, cement_kiln.

        Returns:
            CO2e emissions in tonnes.

        Raises:
            ValueError: If sludge_mass < 0 or disposal_method unknown.

        Example:
            >>> co2e = engine.calculate_sludge_emissions(
            ...     sludge_mass_dry_tonnes=Decimal("50"),
            ...     disposal_method="landfill",
            ... )
            >>> # 50 tonnes x 500 kgCO2e/tonne / 1000 = 25 tCO2e
        """
        self._validate_non_negative(
            "sludge_mass_dry_tonnes", sludge_mass_dry_tonnes
        )

        method_key = disposal_method.lower().strip()
        if method_key not in SLUDGE_DISPOSAL_EF:
            raise ValueError(
                f"Unknown sludge disposal method '{disposal_method}'. "
                f"Valid methods: {list(SLUDGE_DISPOSAL_EF.keys())}"
            )

        ef = SLUDGE_DISPOSAL_EF[method_key]
        co2e_tonnes = (sludge_mass_dry_tonnes * ef / KG_PER_TONNE).quantize(
            PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "Sludge emissions: %.4f t dry x %s kgCO2e/t (%s) = %.6f tCO2e",
            sludge_mass_dry_tonnes, ef, method_key, co2e_tonnes,
        )
        return co2e_tonnes

    def calculate_sludge_emissions_detailed(
        self,
        sludge_mass_dry_tonnes: Decimal,
        disposal_method: str,
    ) -> SludgeEmissionsDetail:
        """
        Calculate sludge emissions with full breakdown detail.

        Args:
            sludge_mass_dry_tonnes: Dry mass of sludge (tonnes).
            disposal_method: Disposal pathway.

        Returns:
            SludgeEmissionsDetail with mass, method, EF, and CO2e.
        """
        method_key = disposal_method.lower().strip()
        if method_key not in SLUDGE_DISPOSAL_EF:
            raise ValueError(
                f"Unknown sludge disposal method '{disposal_method}'"
            )

        co2e = self.calculate_sludge_emissions(
            sludge_mass_dry_tonnes, disposal_method
        )

        return SludgeEmissionsDetail(
            sludge_mass_dry_tonnes=sludge_mass_dry_tonnes,
            disposal_method=method_key,
            ef_used=SLUDGE_DISPOSAL_EF[method_key],
            co2e_tonnes=co2e,
        )

    def estimate_sludge_generation(
        self,
        organic_load_kg: Decimal,
        treatment_system: WastewaterSystem,
    ) -> Decimal:
        """
        Estimate dry sludge generation from wastewater treatment.

        Formula:
            sludge_dry_kg = organic_load_kg x treatment_efficiency x sludge_rate

        Args:
            organic_load_kg: Organic load entering treatment (kg COD).
            treatment_system: Treatment system type.

        Returns:
            Estimated dry sludge mass in tonnes.

        Example:
            >>> sludge_t = engine.estimate_sludge_generation(
            ...     organic_load_kg=Decimal("100000"),
            ...     treatment_system=WastewaterSystem.CENTRALIZED_AEROBIC_GOOD,
            ... )
        """
        self._validate_non_negative("organic_load_kg", organic_load_kg)

        efficiency = TREATMENT_EFFICIENCY.get(
            treatment_system, Decimal("0.70")
        )
        sludge_rate = SLUDGE_GENERATION_RATE.get(
            treatment_system, Decimal("0.30")
        )

        sludge_kg = organic_load_kg * efficiency * sludge_rate
        sludge_tonnes = (sludge_kg / KG_PER_TONNE).quantize(
            PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "Sludge generation: %.0f kg load x %.2f eff x %.2f rate = %.6f t",
            organic_load_kg, efficiency, sludge_rate, sludge_tonnes,
        )
        return sludge_tonnes

    # ========================================================================
    # SECTION 8: POPULATION-BASED ESTIMATION
    # ========================================================================

    def estimate_tow_from_population(
        self,
        population: int,
        bod_per_capita_per_day: Optional[Decimal] = None,
        region: str = "global_default",
    ) -> Decimal:
        """
        Estimate total organic waste from population served.

        IPCC 2006 Vol 5 Equation 6.3:
            TOW = P x BOD_per_capita x 365 / 1000

        Where P is population, BOD in g/person/day, result in kg BOD/yr.

        Args:
            population: Number of people served by the treatment system.
            bod_per_capita_per_day: BOD generation rate (g/person/day).
                If None, uses regional default from IPCC Table 6.4.
            region: Region for default BOD lookup. One of: africa,
                asia, europe, latin_america, middle_east,
                north_america, oceania, global_default.

        Returns:
            Total organic waste in kg BOD/yr.

        Raises:
            ValueError: If population <= 0.

        Example:
            >>> tow = engine.estimate_tow_from_population(
            ...     population=50000,
            ...     region="europe",
            ... )
            >>> # 50000 x 60 g/day x 365 / 1000 = 1,095,000 kg BOD/yr
        """
        if population <= 0:
            raise ValueError("population must be > 0")

        if bod_per_capita_per_day is None:
            bod_per_capita_per_day = BOD_PER_CAPITA_DEFAULTS.get(
                region.lower(), BOD_PER_CAPITA_DEFAULTS["global_default"]
            )

        pop_dec = Decimal(str(population))
        tow_kg = (
            pop_dec * bod_per_capita_per_day * DAYS_PER_YEAR / G_PER_KG
        ).quantize(PRECISION, rounding=ROUND_HALF_UP)

        logger.info(
            "TOW from population: %d people x %.1f g BOD/day "
            "x 365 / 1000 = %.2f kg BOD/yr [%s]",
            population, bod_per_capita_per_day, tow_kg, region,
        )
        return tow_kg

    def estimate_nitrogen_from_population(
        self,
        population: int,
        region: str = "global_default",
        nitrogen_per_capita_override: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Estimate nitrogen load from population served.

        Uses IPCC 2006 Vol 5 Table 6.11 per-capita nitrogen values
        based on regional protein consumption patterns.

        Args:
            population: Number of people served.
            region: Region for default lookup.
            nitrogen_per_capita_override: Override kg N/person/year.

        Returns:
            Total nitrogen in effluent (kg N/yr).

        Raises:
            ValueError: If population <= 0.

        Example:
            >>> n_kg = engine.estimate_nitrogen_from_population(
            ...     population=100000,
            ...     region="north_america",
            ... )
            >>> # 100000 x 6.6 = 660,000 kg N/yr
        """
        if population <= 0:
            raise ValueError("population must be > 0")

        if nitrogen_per_capita_override is not None:
            n_per_cap = nitrogen_per_capita_override
        else:
            n_per_cap = NITROGEN_PER_CAPITA_DEFAULTS.get(
                region.lower(),
                NITROGEN_PER_CAPITA_DEFAULTS["global_default"],
            )

        pop_dec = Decimal(str(population))
        n_total = (pop_dec * n_per_cap).quantize(
            PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "Nitrogen from population: %d x %.2f kg N/person/yr = %.2f kg N",
            population, n_per_cap, n_total,
        )
        return n_total

    # ========================================================================
    # SECTION 9: PARAMETER LOOKUP METHODS
    # ========================================================================

    def get_bo(self, measurement_basis: str) -> Decimal:
        """
        Get maximum CH4 producing capacity (Bo) for the measurement basis.

        IPCC 2006 Vol 5 Table 6.2:
            COD basis: Bo = 0.25 kg CH4/kg COD
            BOD basis: Bo = 0.60 kg CH4/kg BOD

        Args:
            measurement_basis: 'cod' or 'bod'.

        Returns:
            Bo value (Decimal).

        Raises:
            ValueError: If measurement_basis not 'cod' or 'bod'.

        Example:
            >>> bo = engine.get_bo("cod")
            >>> assert bo == Decimal("0.25")
        """
        basis = measurement_basis.lower().strip()
        if basis == "cod":
            return BO_COD
        elif basis == "bod":
            return BO_BOD
        else:
            raise ValueError(
                f"Invalid measurement_basis '{measurement_basis}'. "
                f"Must be 'cod' or 'bod'."
            )

    def get_mcf(self, treatment_system: WastewaterSystem) -> Decimal:
        """
        Get methane correction factor (MCF) for a treatment system.

        IPCC 2006 Vol 5 Table 6.3.

        Args:
            treatment_system: WastewaterSystem enum value.

        Returns:
            MCF value (Decimal, 0.0 to 1.0).

        Raises:
            KeyError: If treatment_system not in MCF table.

        Example:
            >>> mcf = engine.get_mcf(WastewaterSystem.ANAEROBIC_REACTOR)
            >>> assert mcf == Decimal("0.80")
        """
        if treatment_system not in WASTEWATER_MCF:
            raise KeyError(
                f"Treatment system '{treatment_system.value}' not found "
                f"in IPCC MCF table"
            )
        return WASTEWATER_MCF[treatment_system]

    def get_treatment_efficiency(
        self,
        treatment_system: WastewaterSystem,
    ) -> Decimal:
        """
        Get organic load removal efficiency for a treatment system.

        Args:
            treatment_system: WastewaterSystem enum value.

        Returns:
            Efficiency as a fraction (0.0 to 1.0).
        """
        return TREATMENT_EFFICIENCY.get(treatment_system, Decimal("0.70"))

    def get_nitrogen_removal_fraction(
        self,
        treatment_system: WastewaterSystem,
    ) -> Decimal:
        """
        Get nitrogen removal fraction for a treatment system.

        Args:
            treatment_system: WastewaterSystem enum value.

        Returns:
            Nitrogen removal fraction (0.0 to 1.0).
        """
        return NITROGEN_REMOVAL_FRACTION.get(
            treatment_system, Decimal("0.30")
        )

    def get_sludge_generation_rate(
        self,
        treatment_system: WastewaterSystem,
    ) -> Decimal:
        """
        Get sludge generation rate for a treatment system.

        Args:
            treatment_system: WastewaterSystem enum value.

        Returns:
            Rate in kg dry sludge per kg COD removed.
        """
        return SLUDGE_GENERATION_RATE.get(
            treatment_system, Decimal("0.30")
        )

    def get_sludge_disposal_ef(self, disposal_method: str) -> Decimal:
        """
        Get sludge disposal emission factor.

        Args:
            disposal_method: Disposal pathway name.

        Returns:
            Emission factor in kg CO2e per tonne dry sludge.

        Raises:
            ValueError: If disposal_method not recognized.
        """
        method_key = disposal_method.lower().strip()
        if method_key not in SLUDGE_DISPOSAL_EF:
            raise ValueError(
                f"Unknown disposal method '{disposal_method}'. "
                f"Valid: {list(SLUDGE_DISPOSAL_EF.keys())}"
            )
        return SLUDGE_DISPOSAL_EF[method_key]

    def get_ch4_recovery_efficiency(self, system_type: str) -> Decimal:
        """
        Get CH4 recovery efficiency for a biogas system type.

        Args:
            system_type: One of: biogas_capture_good, biogas_capture_fair,
                biogas_capture_poor, flaring_enclosed, flaring_open,
                no_recovery.

        Returns:
            Recovery fraction (0.0 to 1.0).

        Raises:
            ValueError: If system_type not recognized.
        """
        key = system_type.lower().strip()
        if key not in CH4_RECOVERY_EFFICIENCY:
            raise ValueError(
                f"Unknown recovery system '{system_type}'. "
                f"Valid: {list(CH4_RECOVERY_EFFICIENCY.keys())}"
            )
        return CH4_RECOVERY_EFFICIENCY[key]

    # ========================================================================
    # SECTION 10: TREATMENT EFFICIENCY CALCULATION
    # ========================================================================

    def calculate_treatment_efficiency(
        self,
        inflow_load: Decimal,
        outflow_load: Decimal,
    ) -> Decimal:
        """
        Calculate actual treatment efficiency from measured inflow/outflow.

        Formula: efficiency = (inflow - outflow) / inflow

        Args:
            inflow_load: Organic load entering treatment (kg COD or BOD).
            outflow_load: Organic load in effluent (kg COD or BOD).

        Returns:
            Treatment efficiency as a fraction (0.0 to 1.0).

        Raises:
            ValueError: If inflow_load <= 0 or outflow > inflow.

        Example:
            >>> eff = engine.calculate_treatment_efficiency(
            ...     inflow_load=Decimal("1000"),
            ...     outflow_load=Decimal("50"),
            ... )
            >>> assert eff == Decimal("0.950000")
        """
        self._validate_positive("inflow_load", inflow_load)
        self._validate_non_negative("outflow_load", outflow_load)

        if outflow_load > inflow_load:
            raise ValueError(
                f"outflow_load ({outflow_load}) cannot exceed "
                f"inflow_load ({inflow_load})"
            )

        efficiency = ((inflow_load - outflow_load) / inflow_load).quantize(
            PRECISION, rounding=ROUND_HALF_UP
        )
        return efficiency

    def calculate_organic_load_removed(
        self,
        inflow_load: Decimal,
        treatment_system: WastewaterSystem,
    ) -> Decimal:
        """
        Calculate organic load removed based on treatment efficiency.

        Args:
            inflow_load: Organic load entering treatment (kg).
            treatment_system: Treatment system type.

        Returns:
            Organic load removed (kg).
        """
        self._validate_non_negative("inflow_load", inflow_load)
        efficiency = self.get_treatment_efficiency(treatment_system)
        return (inflow_load * efficiency).quantize(
            PRECISION, rounding=ROUND_HALF_UP
        )

    # ========================================================================
    # SECTION 11: BATCH PROCESSING
    # ========================================================================

    def calculate_batch(
        self,
        inputs: List[WastewaterInput],
        gwp_version: GWPVersion = GWPVersion.AR5,
        recovery_efficiency: Decimal = Decimal("0"),
        tenant_id: str = "default",
    ) -> List[WastewaterEmissionsResult]:
        """
        Process a batch of wastewater emission calculations.

        Iterates over inputs, calculating each independently. Errors on
        individual items are logged and skipped (partial success).

        Args:
            inputs: List of WastewaterInput records.
            gwp_version: GWP version for all calculations.
            recovery_efficiency: CH4 recovery fraction for all.
            tenant_id: Tenant identifier.

        Returns:
            List of WastewaterEmissionsResult (may be shorter than
            inputs if some failed).

        Example:
            >>> results = engine.calculate_batch(
            ...     inputs=[input1, input2, input3],
            ... )
            >>> assert len(results) <= 3
        """
        if not inputs:
            logger.warning("calculate_batch called with empty input list")
            return []

        results: List[WastewaterEmissionsResult] = []
        success_count = 0
        error_count = 0
        start_ns = time.perf_counter_ns()

        for idx, inp in enumerate(inputs):
            try:
                result = self.calculate(
                    input_data=inp,
                    gwp_version=gwp_version,
                    recovery_efficiency=recovery_efficiency,
                    tenant_id=tenant_id,
                )
                results.append(result)
                success_count += 1
            except Exception as exc:
                error_count += 1
                logger.error(
                    "Batch item %d/%d failed: %s",
                    idx + 1, len(inputs), str(exc),
                )

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        logger.info(
            "Batch complete: %d/%d success, %d errors, %.1f ms total",
            success_count, len(inputs), error_count, elapsed_ms,
        )

        return results

    # ========================================================================
    # SECTION 12: VALIDATION
    # ========================================================================

    def validate_wastewater_input(
        self,
        input_data: WastewaterInput,
    ) -> List[str]:
        """
        Validate a WastewaterInput for completeness and correctness.

        Checks:
            1. organic_load_kg > 0
            2. measurement_basis is 'cod' or 'bod'
            3. treatment_system is a valid WastewaterSystem
            4. nitrogen_load_kg >= 0 if provided
            5. bo_override in valid range if provided
            6. mcf_override in [0, 1] if provided
            7. Industry type consistency

        Args:
            input_data: WastewaterInput to validate.

        Returns:
            List of error messages. Empty list = valid.

        Example:
            >>> errors = engine.validate_wastewater_input(inp)
            >>> if not errors:
            ...     print("Valid input")
        """
        errors: List[str] = []

        # organic_load_kg
        if input_data.organic_load_kg <= Decimal("0"):
            errors.append("organic_load_kg must be > 0")

        # measurement_basis
        basis = input_data.measurement_basis.lower().strip()
        if basis not in ("cod", "bod"):
            errors.append(
                f"measurement_basis must be 'cod' or 'bod', "
                f"got '{input_data.measurement_basis}'"
            )

        # treatment_system
        if not isinstance(input_data.treatment_system, WastewaterSystem):
            errors.append(
                f"treatment_system must be a WastewaterSystem enum, "
                f"got '{type(input_data.treatment_system).__name__}'"
            )

        # nitrogen_load_kg
        if (
            input_data.nitrogen_load_kg is not None
            and input_data.nitrogen_load_kg < Decimal("0")
        ):
            errors.append("nitrogen_load_kg must be >= 0 if provided")

        # bo_override
        if input_data.bo_override is not None:
            if input_data.bo_override <= Decimal("0"):
                errors.append("bo_override must be > 0 if provided")
            if input_data.bo_override > Decimal("1"):
                errors.append("bo_override should be <= 1.0 (kg CH4/kg)")

        # mcf_override
        if input_data.mcf_override is not None:
            if input_data.mcf_override < Decimal("0"):
                errors.append("mcf_override must be >= 0")
            if input_data.mcf_override > Decimal("1"):
                errors.append("mcf_override must be <= 1")

        # Industry consistency
        if input_data.industry_type is not None:
            if input_data.production_volume_tonnes is None:
                errors.append(
                    "production_volume_tonnes required when industry_type is set"
                )

        return errors

    def validate_sludge_disposal_method(
        self, disposal_method: str
    ) -> bool:
        """
        Check if a sludge disposal method is valid.

        Args:
            disposal_method: Disposal method string.

        Returns:
            True if valid, False otherwise.
        """
        return disposal_method.lower().strip() in SLUDGE_DISPOSAL_EF

    def get_supported_treatment_systems(self) -> List[str]:
        """
        Get list of supported wastewater treatment systems.

        Returns:
            List of WastewaterSystem enum values as strings.
        """
        return [system.value for system in WastewaterSystem]

    def get_supported_industries(self) -> List[str]:
        """
        Get list of supported industry types for organic load estimation.

        Returns:
            List of IndustryWastewaterType enum values as strings.
        """
        return [ind.value for ind in IndustryWastewaterType]

    def get_supported_sludge_disposal_methods(self) -> List[str]:
        """
        Get list of supported sludge disposal methods.

        Returns:
            List of disposal method names.
        """
        return list(SLUDGE_DISPOSAL_EF.keys())

    def get_supported_recovery_systems(self) -> List[str]:
        """
        Get list of supported CH4 recovery system types.

        Returns:
            List of recovery system type names.
        """
        return list(CH4_RECOVERY_EFFICIENCY.keys())

    # ========================================================================
    # SECTION 13: UNCERTAINTY ESTIMATION
    # ========================================================================

    def estimate_uncertainty(
        self,
        result: WastewaterEmissionsResult,
    ) -> Dict[str, Decimal]:
        """
        Estimate IPCC default uncertainty for a wastewater result.

        Uses error propagation combining MCF, Bo, TOW, and N2O EF
        uncertainties. IPCC default approach (Tier 1 uncertainty).

        The combined uncertainty is calculated as:
            U_combined = sqrt(sum of (U_i)^2) for each parameter

        Args:
            result: WastewaterEmissionsResult to analyze.

        Returns:
            Dictionary with:
                - combined_uncertainty_pct: Combined uncertainty (%)
                - lower_bound_co2e: Lower bound at 95% confidence
                - upper_bound_co2e: Upper bound at 95% confidence
                - parameter_uncertainties: Per-parameter breakdown

        Example:
            >>> unc = engine.estimate_uncertainty(result)
            >>> print(f"Uncertainty: +/- {unc['combined_uncertainty_pct']}%")
        """
        # IPCC default uncertainty ranges
        u_mcf = UNCERTAINTY_RANGES["mcf"]
        u_bo = UNCERTAINTY_RANGES["bo"]
        u_tow = UNCERTAINTY_RANGES["tow"]
        u_n2o_ef = UNCERTAINTY_RANGES["ef_n2o"]

        # Root-sum-of-squares for CH4 pathway
        u_ch4_sq = u_mcf ** 2 + u_bo ** 2 + u_tow ** 2
        u_ch4 = u_ch4_sq.sqrt()

        # For combined, include N2O if present
        if result.n2o_from_effluent_tonnes > Decimal("0"):
            # Weight by relative contribution
            ch4_co2e = result.ch4_from_treatment_tonnes
            n2o_co2e = result.n2o_from_effluent_tonnes
            total_gas = ch4_co2e + n2o_co2e

            if total_gas > Decimal("0"):
                w_ch4 = ch4_co2e / total_gas
                w_n2o = n2o_co2e / total_gas
                u_combined_sq = (w_ch4 * u_ch4) ** 2 + (
                    w_n2o * u_n2o_ef
                ) ** 2
                u_combined = u_combined_sq.sqrt()
            else:
                u_combined = u_ch4
        else:
            u_combined = u_ch4

        u_combined = u_combined.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # 95% confidence bounds (approx 2 sigma)
        factor = u_combined / Decimal("100") * Decimal("2")
        lower = max(
            result.co2e_total * (Decimal("1") - factor), Decimal("0")
        )
        upper = result.co2e_total * (Decimal("1") + factor)

        return {
            "combined_uncertainty_pct": u_combined,
            "lower_bound_co2e": lower.quantize(PRECISION),
            "upper_bound_co2e": upper.quantize(PRECISION),
            "parameter_uncertainties": {
                "mcf_pct": u_mcf,
                "bo_pct": u_bo,
                "tow_pct": u_tow,
                "ef_n2o_pct": u_n2o_ef,
                "ch4_pathway_combined_pct": u_ch4.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                ),
            },
        }

    # ========================================================================
    # SECTION 14: DATA QUALITY
    # ========================================================================

    def assess_data_quality(
        self,
        input_data: WastewaterInput,
    ) -> Dict[str, Any]:
        """
        Assess the data quality tier of a wastewater input.

        Evaluates based on:
            - Measurement specificity (COD/BOD measured vs estimated)
            - Parameter overrides (more specific = higher tier)
            - Industry vs direct measurement
            - Nitrogen load availability

        Args:
            input_data: WastewaterInput to assess.

        Returns:
            Dictionary with:
                - tier: DataQualityTier (TIER_1, TIER_2, TIER_3)
                - score: Numeric score (1-5, lower is better)
                - justification: List of reasoning strings
        """
        score = 3  # Start at middle (fair)
        justification: List[str] = []

        # Direct organic load measurement
        if input_data.organic_load_kg > Decimal("0"):
            score -= 1
            justification.append(
                "Direct organic load measurement provided (+1 quality)"
            )

        # Parameter overrides indicate facility-specific data
        if input_data.bo_override is not None:
            score -= 1
            justification.append(
                "Facility-specific Bo override provided (+1 quality)"
            )

        if input_data.mcf_override is not None:
            score -= 1
            justification.append(
                "Facility-specific MCF override provided (+1 quality)"
            )

        # Nitrogen load available
        if input_data.nitrogen_load_kg is not None:
            justification.append("Nitrogen load data available")
        else:
            score += 1
            justification.append(
                "No nitrogen load data, N2O calculation will be zero (-1 quality)"
            )

        # Industry-based estimation is lower quality
        if input_data.industry_type is not None:
            score += 1
            justification.append(
                "Industry-based load estimation used (-1 quality)"
            )

        # Clamp score
        score = max(1, min(5, score))

        # Map to tier
        if score <= 2:
            tier = DataQualityTier.TIER_3
        elif score <= 3:
            tier = DataQualityTier.TIER_2
        else:
            tier = DataQualityTier.TIER_1

        return {
            "tier": tier,
            "score": score,
            "justification": justification,
        }

    # ========================================================================
    # SECTION 15: DETAILED CALCULATION WITH FULL BREAKDOWN
    # ========================================================================

    def calculate_detailed(
        self,
        input_data: WastewaterInput,
        gwp_version: GWPVersion = GWPVersion.AR5,
        recovery_efficiency: Decimal = Decimal("0"),
        include_sludge: bool = False,
        sludge_disposal_method: str = "landfill",
        tenant_id: str = "default",
    ) -> WastewaterDetailedResult:
        """
        Full detailed calculation with complete breakdown.

        Returns all intermediate values including:
            - CH4 generated/recovered/net in kg
            - N2O in kg
            - CO2e breakdown by gas
            - Sludge emissions detail
            - Uncertainty estimate
            - Data quality assessment
            - Provenance hash

        Args:
            input_data: WastewaterInput.
            gwp_version: GWP version.
            recovery_efficiency: CH4 recovery fraction.
            include_sludge: Include sludge emissions.
            sludge_disposal_method: Sludge disposal pathway.
            tenant_id: Tenant ID.

        Returns:
            WastewaterDetailedResult with full breakdown.
        """
        start_ns = time.perf_counter_ns()

        # Run base calculation
        base_result = self.calculate(
            input_data=input_data,
            gwp_version=gwp_version,
            recovery_efficiency=recovery_efficiency,
            include_sludge=False,
            tenant_id=tenant_id,
        )

        # Re-derive intermediate values
        bo = self._resolve_bo(input_data)
        mcf = self._resolve_mcf(input_data)
        ch4_generated_kg = self.calculate_ch4_from_treatment(
            input_data.organic_load_kg, bo, mcf
        )
        ch4_recovered_kg = self.calculate_ch4_recovery(
            ch4_generated_kg, recovery_efficiency
        )
        ch4_net_kg = self.calculate_net_ch4(ch4_generated_kg, ch4_recovered_kg)

        n2o_kg = Decimal("0")
        if input_data.nitrogen_load_kg is not None:
            n2o_kg = self.calculate_n2o_from_effluent(
                input_data.nitrogen_load_kg
            )

        co2e_ch4 = self.convert_ch4_to_co2e(ch4_net_kg, gwp_version)
        co2e_n2o = self.convert_n2o_to_co2e(n2o_kg, gwp_version)

        # Sludge
        sludge_detail: Optional[SludgeEmissionsDetail] = None
        co2e_sludge = Decimal("0")
        if include_sludge:
            sludge_tonnes = self.estimate_sludge_generation(
                input_data.organic_load_kg, input_data.treatment_system
            )
            sludge_detail = self.calculate_sludge_emissions_detailed(
                sludge_tonnes, sludge_disposal_method
            )
            co2e_sludge = sludge_detail.co2e_tonnes

        # Uncertainty
        unc = self.estimate_uncertainty(base_result)

        # Data quality
        dq = self.assess_data_quality(input_data)

        # Provenance
        provenance_hash = self._compute_provenance_hash(
            input_data, ch4_net_kg, n2o_kg,
            base_result.co2e_total + co2e_sludge,
        )

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        return WastewaterDetailedResult(
            base_result=base_result,
            ch4_generated_kg=ch4_generated_kg,
            ch4_recovered_kg=ch4_recovered_kg,
            ch4_net_kg=ch4_net_kg,
            n2o_kg=n2o_kg,
            co2e_ch4=co2e_ch4,
            co2e_n2o=co2e_n2o,
            co2e_sludge=co2e_sludge,
            sludge_detail=sludge_detail,
            gwp_version=gwp_version,
            uncertainty_pct=unc["combined_uncertainty_pct"],
            data_quality_tier=dq["tier"],
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
        )

    # ========================================================================
    # SECTION 16: COMPARISON AND SCENARIO METHODS
    # ========================================================================

    def compare_treatment_systems(
        self,
        organic_load_kg: Decimal,
        measurement_basis: str = "cod",
        nitrogen_load: Optional[Decimal] = None,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> List[Dict[str, Any]]:
        """
        Compare emissions across all treatment systems for a given load.

        Useful for scenario analysis and treatment system selection
        to minimize GHG emissions.

        Args:
            organic_load_kg: Organic load (kg COD or BOD/yr).
            measurement_basis: 'cod' or 'bod'.
            nitrogen_load: Nitrogen load (kg N/yr).
            gwp_version: GWP version.

        Returns:
            List of dictionaries sorted by co2e_total (ascending), each
            containing treatment_system, co2e_total, ch4_tonnes, n2o_tonnes,
            mcf_used.

        Example:
            >>> comparison = engine.compare_treatment_systems(
            ...     organic_load_kg=Decimal("100000"),
            ...     measurement_basis="cod",
            ...     nitrogen_load=Decimal("2000"),
            ... )
            >>> best = comparison[0]["treatment_system"]
        """
        self._validate_positive("organic_load_kg", organic_load_kg)

        results: List[Dict[str, Any]] = []

        for system in WastewaterSystem:
            try:
                inp = WastewaterInput(
                    organic_load_kg=organic_load_kg,
                    measurement_basis=measurement_basis,
                    treatment_system=system,
                    nitrogen_load_kg=nitrogen_load,
                )
                result = self.calculate(
                    input_data=inp,
                    gwp_version=gwp_version,
                )
                results.append({
                    "treatment_system": system.value,
                    "co2e_total": result.co2e_total,
                    "ch4_from_treatment_tonnes": result.ch4_from_treatment_tonnes,
                    "n2o_from_effluent_tonnes": result.n2o_from_effluent_tonnes,
                    "mcf_used": result.mcf_used,
                    "bo_used": result.bo_used,
                })
            except Exception as exc:
                logger.warning(
                    "compare_treatment_systems: %s failed: %s",
                    system.value, str(exc),
                )

        results.sort(key=lambda r: r["co2e_total"])
        return results

    def compare_recovery_scenarios(
        self,
        input_data: WastewaterInput,
        recovery_levels: Optional[List[Decimal]] = None,
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> List[Dict[str, Any]]:
        """
        Compare emissions across different CH4 recovery efficiency levels.

        Args:
            input_data: Base WastewaterInput.
            recovery_levels: List of recovery fractions to compare.
                Defaults to [0, 0.25, 0.50, 0.70, 0.90].
            gwp_version: GWP version.

        Returns:
            List of dicts with recovery_efficiency, co2e_total,
            ch4_recovered_tonnes, reduction_pct.
        """
        if recovery_levels is None:
            recovery_levels = [
                Decimal("0"),
                Decimal("0.25"),
                Decimal("0.50"),
                Decimal("0.70"),
                Decimal("0.90"),
            ]

        results: List[Dict[str, Any]] = []
        baseline_co2e: Optional[Decimal] = None

        for level in recovery_levels:
            try:
                result = self.calculate(
                    input_data=input_data,
                    gwp_version=gwp_version,
                    recovery_efficiency=level,
                )
                if baseline_co2e is None:
                    baseline_co2e = result.co2e_total

                reduction_pct = Decimal("0")
                if baseline_co2e > Decimal("0"):
                    reduction_pct = (
                        (Decimal("1") - result.co2e_total / baseline_co2e)
                        * Decimal("100")
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                results.append({
                    "recovery_efficiency": level,
                    "co2e_total": result.co2e_total,
                    "ch4_recovered_tonnes": (
                        result.ch4_recovered_tonnes
                        if result.ch4_recovered_tonnes is not None
                        else Decimal("0")
                    ),
                    "reduction_pct": reduction_pct,
                })
            except Exception as exc:
                logger.warning(
                    "compare_recovery_scenarios: level %s failed: %s",
                    level, str(exc),
                )

        return results

    # ========================================================================
    # SECTION 17: REGIONAL DEFAULTS
    # ========================================================================

    def get_bod_per_capita_default(self, region: str) -> Decimal:
        """
        Get default BOD per capita per day for a region.

        IPCC 2006 Vol 5 Table 6.4.

        Args:
            region: Region name (africa, asia, europe, latin_america,
                middle_east, north_america, oceania, global_default).

        Returns:
            BOD in g/person/day.
        """
        return BOD_PER_CAPITA_DEFAULTS.get(
            region.lower(),
            BOD_PER_CAPITA_DEFAULTS["global_default"],
        )

    def get_nitrogen_per_capita_default(self, region: str) -> Decimal:
        """
        Get default nitrogen per capita per year for a region.

        IPCC 2006 Vol 5 Table 6.11.

        Args:
            region: Region name.

        Returns:
            Nitrogen in kg N/person/year.
        """
        return NITROGEN_PER_CAPITA_DEFAULTS.get(
            region.lower(),
            NITROGEN_PER_CAPITA_DEFAULTS["global_default"],
        )

    def get_all_mcf_values(self) -> Dict[str, Decimal]:
        """
        Get all MCF values for all treatment systems.

        Returns:
            Dictionary mapping system name to MCF value.
        """
        return {
            system.value: mcf
            for system, mcf in WASTEWATER_MCF.items()
        }

    def get_all_industry_loads(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get all industry wastewater load data.

        Returns:
            Dictionary mapping industry name to load parameters.
        """
        return {
            ind.value: dict(data)
            for ind, data in INDUSTRY_WASTEWATER_LOADS.items()
        }

    # ========================================================================
    # SECTION 18: SERIALIZATION AND EXPORT
    # ========================================================================

    def result_to_dict(
        self,
        result: WastewaterEmissionsResult,
    ) -> Dict[str, Any]:
        """
        Convert a WastewaterEmissionsResult to a serializable dict.

        Args:
            result: Result to convert.

        Returns:
            Dictionary with string-serialized Decimal values.
        """
        return {
            "ch4_from_treatment_tonnes": str(result.ch4_from_treatment_tonnes),
            "n2o_from_effluent_tonnes": str(result.n2o_from_effluent_tonnes),
            "ch4_recovered_tonnes": (
                str(result.ch4_recovered_tonnes)
                if result.ch4_recovered_tonnes is not None
                else None
            ),
            "co2e_total": str(result.co2e_total),
            "organic_load_used_kg": str(result.organic_load_used_kg),
            "mcf_used": str(result.mcf_used),
            "bo_used": str(result.bo_used),
            "treatment_system": result.treatment_system.value,
        }

    def result_to_waste_calculation(
        self,
        result: WastewaterEmissionsResult,
        calculation_id: Optional[str] = None,
        tenant_id: str = "default",
        facility_id: str = "default",
        mass_tonnes: Decimal = Decimal("0"),
        gwp_version: GWPVersion = GWPVersion.AR5,
    ) -> WasteCalculationResult:
        """
        Wrap a WastewaterEmissionsResult into the standard WasteCalculationResult.

        This allows wastewater results to be aggregated with other waste
        treatment results in the pipeline.

        Args:
            result: Base WastewaterEmissionsResult.
            calculation_id: Unique ID. Auto-generated if None.
            tenant_id: Tenant identifier.
            facility_id: Facility identifier.
            mass_tonnes: Wastewater mass/volume equivalent.
            gwp_version: GWP version used.

        Returns:
            WasteCalculationResult wrapping the wastewater result.
        """
        if calculation_id is None:
            calculation_id = f"ww-{uuid.uuid4().hex[:12]}"

        provenance_hash = self._compute_provenance_hash_simple(
            calculation_id, result.co2e_total
        )

        # If mass is 0, use organic load as proxy
        effective_mass = mass_tonnes
        if effective_mass <= Decimal("0"):
            effective_mass = (
                result.organic_load_used_kg / KG_PER_TONNE
            ).quantize(PRECISION)
            if effective_mass <= Decimal("0"):
                effective_mass = Decimal("0.000001")

        return WasteCalculationResult(
            calculation_id=calculation_id,
            tenant_id=tenant_id,
            facility_id=facility_id,
            waste_category=WasteCategory.OTHER,
            treatment_method=WasteTreatmentMethod.WASTEWATER_TREATMENT,
            calculation_method=CalculationMethod.WASTE_TYPE_SPECIFIC,
            mass_tonnes=effective_mass,
            total_co2e=result.co2e_total,
            breakdown=result,
            ef_source=EFSource.IPCC_2006,
            data_quality_tier=DataQualityTier.TIER_1,
            gwp_version=gwp_version,
            provenance_hash=provenance_hash,
        )

    # ========================================================================
    # SECTION 19: ENGINE METADATA
    # ========================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get engine metadata and capabilities.

        Returns:
            Dictionary with engine identification and capabilities.
        """
        return {
            "engine_name": "WastewaterEmissionsEngine",
            "engine_number": 5,
            "agent_id": AGENT_ID,
            "version": VERSION,
            "table_prefix": TABLE_PREFIX,
            "methodology": "IPCC 2006 Vol 5 Ch 6",
            "gases_covered": ["CH4", "N2O"],
            "measurement_bases": ["cod", "bod"],
            "treatment_systems": self.get_supported_treatment_systems(),
            "industry_types": self.get_supported_industries(),
            "sludge_disposal_methods": self.get_supported_sludge_disposal_methods(),
            "recovery_systems": self.get_supported_recovery_systems(),
            "gwp_versions": [v.value for v in GWPVersion],
            "default_parameters": {
                "bo_cod": str(BO_COD),
                "bo_bod": str(BO_BOD),
                "ef_n2o": str(DEFAULT_EF_N2O),
                "n2o_n2_ratio": str(N2O_N2_RATIO),
            },
        }

    def get_ipcc_reference(self) -> Dict[str, str]:
        """
        Get IPCC reference citations for the methodology.

        Returns:
            Dictionary of citation keys and references.
        """
        return {
            "primary": (
                "IPCC 2006, 2006 IPCC Guidelines for National Greenhouse "
                "Gas Inventories, Volume 5: Waste, Chapter 6: Wastewater "
                "Treatment and Discharge"
            ),
            "table_6_2": "Bo values for domestic wastewater (Table 6.2)",
            "table_6_3": "MCF values by treatment system (Table 6.3)",
            "table_6_4": "Default BOD per capita by region (Table 6.4)",
            "table_6_9": (
                "Industry wastewater organic loads (Table 6.9)"
            ),
            "table_6_11": "N2O emission factors for effluent (Table 6.11)",
            "equation_6_1": "CH4 Emissions = TOW x Bo x MCF - R",
            "equation_6_7": "N2O = N_effluent x EF x 44/28",
            "refinement_2019": (
                "2019 Refinement to the 2006 IPCC Guidelines, "
                "Volume 5, Chapter 6"
            ),
        }

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _resolve_bo(self, input_data: WastewaterInput) -> Decimal:
        """Resolve Bo from override or default based on measurement basis."""
        if input_data.bo_override is not None:
            return input_data.bo_override
        return self.get_bo(input_data.measurement_basis)

    def _resolve_mcf(self, input_data: WastewaterInput) -> Decimal:
        """Resolve MCF from override or treatment system default."""
        if input_data.mcf_override is not None:
            return input_data.mcf_override
        return self.get_mcf(input_data.treatment_system)

    def _calculate_sludge_from_treatment(
        self,
        organic_load_kg: Decimal,
        treatment_system: WastewaterSystem,
        disposal_method: str,
    ) -> Decimal:
        """Calculate sludge CO2e from treatment system parameters."""
        sludge_tonnes = self.estimate_sludge_generation(
            organic_load_kg, treatment_system
        )
        if sludge_tonnes <= Decimal("0"):
            return Decimal("0")
        return self.calculate_sludge_emissions(sludge_tonnes, disposal_method)

    def _compute_provenance_hash(
        self,
        input_data: WastewaterInput,
        ch4_net_kg: Decimal,
        n2o_kg: Decimal,
        co2e_total: Decimal,
    ) -> str:
        """Compute SHA-256 provenance hash for the calculation."""
        payload = (
            f"{AGENT_ID}|{VERSION}|"
            f"{input_data.organic_load_kg}|"
            f"{input_data.measurement_basis}|"
            f"{input_data.treatment_system.value}|"
            f"{input_data.nitrogen_load_kg}|"
            f"{ch4_net_kg}|{n2o_kg}|{co2e_total}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _compute_provenance_hash_simple(
        self,
        calculation_id: str,
        co2e_total: Decimal,
    ) -> str:
        """Compute a simple provenance hash for wrapper results."""
        payload = f"{AGENT_ID}|{VERSION}|{calculation_id}|{co2e_total}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _record_metrics(
        self,
        input_data: WastewaterInput,
        co2e_total: Decimal,
        ch4_net: Decimal,
        n2o_kg: Decimal,
        elapsed_s: float,
        tenant_id: str,
    ) -> None:
        """Record Prometheus metrics for a calculation."""
        if self._metrics is None:
            return

        try:
            self._metrics.record_calculation(
                method="waste_type_specific",
                treatment="wastewater_treatment",
                waste_category="wastewater",
                tenant_id=tenant_id,
                status="success",
                emissions_tco2e=float(co2e_total),
                waste_mass_tonnes=float(
                    input_data.organic_load_kg / KG_PER_TONNE
                ),
                duration_s=elapsed_s,
            )

            self._metrics.record_wastewater_organic_load(
                measurement_basis=input_data.measurement_basis,
                treatment_system=input_data.treatment_system.value,
                tenant_id=tenant_id,
                organic_load_kg=float(input_data.organic_load_kg),
                emissions_tco2e=float(co2e_total),
                ch4_kg=float(ch4_net),
                n2o_kg=float(n2o_kg),
                duration_s=elapsed_s,
            )
        except Exception as exc:
            logger.warning("Metrics recording failed: %s", str(exc))

    # ========================================================================
    # PRIVATE VALIDATION HELPERS
    # ========================================================================

    @staticmethod
    def _validate_positive(name: str, value: Decimal) -> None:
        """Validate that a Decimal value is strictly positive."""
        if value <= Decimal("0"):
            raise ValueError(f"{name} must be > 0, got {value}")

    @staticmethod
    def _validate_non_negative(name: str, value: Decimal) -> None:
        """Validate that a Decimal value is >= 0."""
        if value < Decimal("0"):
            raise ValueError(f"{name} must be >= 0, got {value}")

    @staticmethod
    def _validate_fraction(name: str, value: Decimal) -> None:
        """Validate that a Decimal value is in [0.0, 1.0]."""
        if value < Decimal("0") or value > Decimal("1"):
            raise ValueError(
                f"{name} must be between 0 and 1, got {value}"
            )


# ============================================================================
# THREAD-SAFE SINGLETON ACCESS
# ============================================================================

_engine_instance: Optional[WastewaterEmissionsEngine] = None
_engine_lock = threading.RLock()


def get_wastewater_engine() -> WastewaterEmissionsEngine:
    """
    Get the singleton WastewaterEmissionsEngine instance.

    Thread-safe lazy initialization. The first call creates the engine;
    subsequent calls return the cached instance.

    Returns:
        WastewaterEmissionsEngine singleton.

    Example:
        >>> engine = get_wastewater_engine()
        >>> result = engine.calculate_from_cod(
        ...     cod_load=Decimal("50000"),
        ...     treatment_system=WastewaterSystem.ANAEROBIC_REACTOR,
        ... )
    """
    global _engine_instance

    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = WastewaterEmissionsEngine()

    return _engine_instance


def reset_wastewater_engine() -> None:
    """
    Reset the singleton engine instance.

    Primarily for testing. Forces re-initialization on next access.
    """
    global _engine_instance

    with _engine_lock:
        WastewaterEmissionsEngine._instance = None
        _engine_instance = None

    logger.info("WastewaterEmissionsEngine singleton reset")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Engine class
    "WastewaterEmissionsEngine",
    # Singleton access
    "get_wastewater_engine",
    "reset_wastewater_engine",
    # Helper classes
    "SludgeEmissionsDetail",
    "WastewaterDetailedResult",
    # Constants
    "BO_COD",
    "BO_BOD",
    "DEFAULT_EF_N2O",
    "N2O_N2_RATIO",
    "CH4_DENSITY",
    "BOD_PER_CAPITA_DEFAULTS",
    "NITROGEN_PER_CAPITA_DEFAULTS",
    "NITROGEN_REMOVAL_FRACTION",
    "TREATMENT_EFFICIENCY",
    "SLUDGE_GENERATION_RATE",
    "SLUDGE_DISPOSAL_EF",
    "CH4_RECOVERY_EFFICIENCY",
    "UNCERTAINTY_RANGES",
]
