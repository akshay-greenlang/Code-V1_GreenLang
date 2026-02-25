# -*- coding: utf-8 -*-
"""
DistrictCoolingCalculatorEngine - Engine 4: Cooling Purchase Agent (AGENT-MRV-012)

Core calculation engine for Scope 2 GHG emissions from district cooling
networks, free cooling (natural heat-sink) sources, thermal energy storage
(TES) systems, and multi-source district cooling plants.  Implements the
GHG Protocol Scope 2 Guidance methodology with deterministic Decimal
arithmetic, full calculation trace, and SHA-256 provenance hashing.

District Cooling Network Formula:
    Adjusted_Cooling = Cooling_Output / (1 - Distribution_Loss_Pct)
    Pump_Emissions   = Pump_Energy_kWh  x  Grid_EF
    Gen_Emissions    = Adjusted_Cooling / COP_plant  x  Generation_EF
    Total_Emissions  = Gen_Emissions + Pump_Emissions

Free Cooling Formula (seawater / lake / river / ambient air):
    Pump_Energy (kWh) = Cooling_Output (kWh_th) / COP_free
    Emissions (kgCO2e) = Pump_Energy (kWh) x Grid_EF (kgCO2e/kWh)

Thermal Energy Storage (TES) Formula:
    Charge_Energy (kWh) = (Capacity (kWh_th) / COP_charge) / RT_Eff
    Emissions (kgCO2e)  = Charge_Energy (kWh) x Grid_EF_charge
    Peak_Emissions      = Cooling_Output / COP_peak x Grid_EF_peak
    Emission_Savings    = Peak_Emissions - TES_Emissions

Multi-Source Plant Formula:
    Total_Emissions = SUM( source_fraction x source_emissions )

Zero-Hallucination Guarantees:
    - All calculations use Python ``Decimal`` (8 decimal places)
    - No LLM calls in the calculation path
    - Every step is recorded in the ordered calculation trace
    - SHA-256 provenance hash for every result
    - Deterministic formula evaluation only

Example:
    >>> from greenlang.cooling_purchase.district_cooling_calculator import (
    ...     DistrictCoolingCalculatorEngine,
    ... )
    >>> from greenlang.cooling_purchase.models import (
    ...     DistrictCoolingRequest, FreeCoolingRequest, TESRequest,
    ...     FreeCoolingSource, TESType, DataQualityTier, GWPSource,
    ... )
    >>> from decimal import Decimal
    >>> engine = DistrictCoolingCalculatorEngine()
    >>> result = engine.calculate_district_network(
    ...     cooling_kwh_th=Decimal("100000"),
    ...     region="singapore",
    ... )
    >>> assert result.emissions_kgco2e > Decimal("0")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-012 Cooling Purchase Agent (GL-MRV-X-023)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.cooling_purchase.config import CoolingPurchaseConfig
from greenlang.cooling_purchase.metrics import (
    record_calculation,
    record_error,
)
from greenlang.cooling_purchase.models import (
    CalculationResult,
    CoolingTechnology,
    COOLING_TECHNOLOGY_SPECS,
    DataQualityTier,
    DISTRICT_COOLING_FACTORS,
    DistrictCoolingFactor,
    DistrictCoolingRequest,
    EmissionGas,
    FreeCoolingRequest,
    FreeCoolingSource,
    GasEmissionDetail,
    GWPSource,
    GWP_VALUES,
    TESCalculationResult,
    TESRequest,
    TESType,
    UNIT_CONVERSIONS,
)
from greenlang.cooling_purchase.provenance import (
    CoolingPurchaseProvenance,
    get_provenance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places

# ---------------------------------------------------------------------------
# kWh-thermal to GJ conversion factor (Decimal)
# ---------------------------------------------------------------------------

_KWH_TO_GJ = Decimal("0.0036")

# ---------------------------------------------------------------------------
# Default pump-energy ratio: pump electricity as fraction of cooling output
# ---------------------------------------------------------------------------

_DEFAULT_PUMP_RATIO = Decimal("0.03")  # 3 % of cooling output

# ---------------------------------------------------------------------------
# Default distribution loss percentage
# ---------------------------------------------------------------------------

_DEFAULT_DISTRIBUTION_LOSS_PCT = Decimal("0.08")  # 8 %

# ---------------------------------------------------------------------------
# Default plant COP for district cooling (used with generation EF path)
# ---------------------------------------------------------------------------

_DEFAULT_PLANT_COP = Decimal("4.0")

# ---------------------------------------------------------------------------
# Free-cooling default COP by source type
# ---------------------------------------------------------------------------

_FREE_COOLING_COP_DEFAULTS: Dict[str, Decimal] = {
    FreeCoolingSource.SEAWATER.value: Decimal("20.0"),
    FreeCoolingSource.LAKE.value: Decimal("18.0"),
    FreeCoolingSource.RIVER.value: Decimal("15.0"),
    FreeCoolingSource.AMBIENT_AIR.value: Decimal("10.0"),
}

# ---------------------------------------------------------------------------
# TES round-trip efficiency defaults by type
# ---------------------------------------------------------------------------

_TES_ROUND_TRIP_EFFICIENCY_DEFAULTS: Dict[str, Decimal] = {
    TESType.ICE.value: Decimal("0.85"),
    TESType.CHILLED_WATER.value: Decimal("0.95"),
    TESType.PCM.value: Decimal("0.90"),
}

# ---------------------------------------------------------------------------
# TES default charging COP by type
# ---------------------------------------------------------------------------

_TES_CHARGE_COP_DEFAULTS: Dict[str, Decimal] = {
    TESType.ICE.value: Decimal("3.2"),
    TESType.CHILLED_WATER.value: Decimal("5.5"),
    TESType.PCM.value: Decimal("4.5"),
}

# ---------------------------------------------------------------------------
# TES default peak-hour COP
# ---------------------------------------------------------------------------

_TES_PEAK_COP_DEFAULT = Decimal("5.0")

# ---------------------------------------------------------------------------
# Global default grid emission factor (kgCO2e/kWh)
# Used when no region-specific or request-specific EF is provided.
# Source: IEA Global weighted average 2024.
# ---------------------------------------------------------------------------

_GLOBAL_DEFAULT_GRID_EF = Decimal("0.436")

# ---------------------------------------------------------------------------
# Default generation EF for district cooling (kgCO2e/kWh electrical input)
# This is the grid EF applied to the chiller electricity consumption
# within the district cooling plant.
# ---------------------------------------------------------------------------

_DEFAULT_GENERATION_EF = Decimal("0.436")

# ---------------------------------------------------------------------------
# Gas decomposition ratios for grid electricity (CO2 / CH4 / N2O fractions)
# Based on global average power sector fuel mix.
# Source: IEA Emission Factors (2024), IPCC 2006 Vol 2.
# ---------------------------------------------------------------------------

_GAS_FRACTION_CO2 = Decimal("0.990")
_GAS_FRACTION_CH4 = Decimal("0.005")
_GAS_FRACTION_N2O = Decimal("0.005")

# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------

_ZERO = Decimal("0")
_ONE = Decimal("1")
_THOUSAND = Decimal("1000")


def _q(value: Decimal) -> Decimal:
    """Quantize a Decimal value to 8 decimal places using ROUND_HALF_UP.

    Args:
        value: Raw Decimal value.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed.

    Returns:
        UTC datetime with microsecond component set to zero.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def _canonical_json(data: Dict[str, Any]) -> str:
    """Serialize a dictionary to canonical JSON form for hashing.

    Uses ``sort_keys=True`` and ``default=str`` for deterministic output.

    Args:
        data: Dictionary to serialize.

    Returns:
        Canonical JSON string.
    """
    return json.dumps(data, sort_keys=True, default=str)


# ===========================================================================
# DistrictCoolingCalculatorEngine
# ===========================================================================


class DistrictCoolingCalculatorEngine:
    """Engine 4: District cooling, free cooling, and TES emission calculator.

    Calculates Scope 2 GHG emissions from:

    1. **District cooling networks** -- accounts for generation efficiency,
       distribution losses, and pump energy.
    2. **Free cooling** -- seawater, lake water, river water, and ambient air
       systems where only pump/fan electricity is consumed.
    3. **Thermal energy storage (TES)** -- ice, chilled water, and PCM storage
       with temporal emission shifting analysis.
    4. **Multi-source plants** -- weighted combination of multiple cooling
       generation sources within a single plant.

    All calculations use deterministic ``Decimal`` arithmetic (8 decimal
    places, ``ROUND_HALF_UP``) and record every intermediate step in an
    ordered trace list.  A SHA-256 provenance hash accompanies each result
    for complete audit trail.

    Thread Safety:
        Implemented as a thread-safe singleton using ``_instance`` with a
        class-level ``threading.RLock``.  Instance creation is guarded by
        double-checked locking.  All public methods are safe for concurrent
        invocation because they operate on method-local state (trace lists,
        intermediate Decimals) with no shared mutable data.

    Attributes:
        _config: CoolingPurchaseConfig singleton reference.
        _provenance: CoolingPurchaseProvenance singleton reference.
        _enable_provenance: Whether provenance chain tracking is active.
        _enable_metrics: Whether Prometheus metrics recording is active.

    Example:
        >>> engine = DistrictCoolingCalculatorEngine()
        >>> result = engine.calculate_district_network(
        ...     cooling_kwh_th=Decimal("100000"),
        ...     region="dubai_uae",
        ... )
        >>> assert result.emissions_kgco2e > Decimal("0")
        >>> assert len(result.provenance_hash) == 64
    """

    _instance: Optional[DistrictCoolingCalculatorEngine] = None
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton construction
    # ------------------------------------------------------------------

    def __new__(cls) -> DistrictCoolingCalculatorEngine:
        """Return the singleton instance, creating it on first call.

        Uses double-checked locking with a ``threading.RLock`` to ensure
        exactly one instance is created even under concurrent access.

        Returns:
            The singleton DistrictCoolingCalculatorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the engine with configuration, provenance, and metrics.

        Guarded by ``_initialized`` flag so repeated ``__init__`` calls
        (from repeated singleton access) are no-ops.
        """
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return

            self._config = CoolingPurchaseConfig()
            self._enable_provenance: bool = self._config.enable_provenance
            self._enable_metrics: bool = self._config.enable_metrics

            if self._enable_provenance:
                self._provenance: Optional[CoolingPurchaseProvenance] = (
                    get_provenance()
                )
            else:
                self._provenance = None

            self._initialized = True

            logger.info(
                "DistrictCoolingCalculatorEngine initialized "
                "(provenance=%s, metrics=%s)",
                self._enable_provenance,
                self._enable_metrics,
            )

    # ------------------------------------------------------------------
    # Singleton reset (testing only)
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing.

        After calling ``reset()``, the next ``DistrictCoolingCalculatorEngine()``
        call will create a fresh instance.  This should only be used in test
        fixtures.
        """
        with cls._lock:
            cls._instance = None
        logger.debug("DistrictCoolingCalculatorEngine singleton reset")

    # ==================================================================
    # PUBLIC API: District Cooling
    # ==================================================================

    def calculate_district_cooling(
        self,
        request: DistrictCoolingRequest,
    ) -> CalculationResult:
        """Calculate Scope 2 emissions from a district cooling network.

        Accepts a fully validated ``DistrictCoolingRequest`` Pydantic model
        and delegates to the core ``calculate_district_network`` method.

        Formula:
            Adjusted_Cooling = Cooling_Output / (1 - Distribution_Loss_Pct)
            Pump_Emissions   = Pump_Energy_kWh x Grid_EF
            Gen_Emissions    = Adjusted_Cooling / COP_plant x Generation_EF
            Total            = Gen_Emissions + Pump_Emissions

        Args:
            request: Validated district cooling request with cooling output,
                region, distribution loss, pump energy, and grid EF.

        Returns:
            CalculationResult with total emissions, per-gas breakdown,
            calculation trace, and provenance hash.

        Example:
            >>> from greenlang.cooling_purchase.models import DistrictCoolingRequest
            >>> req = DistrictCoolingRequest(
            ...     cooling_output_kwh_th=Decimal("100000"),
            ...     region="singapore",
            ... )
            >>> result = engine.calculate_district_cooling(req)
            >>> assert result.emissions_kgco2e > Decimal("0")
        """
        return self.calculate_district_network(
            cooling_kwh_th=request.cooling_output_kwh_th,
            region=request.region,
            distribution_loss_pct=request.distribution_loss_pct,
            pump_energy_kwh=request.pump_energy_kwh,
            grid_ef=request.grid_ef_kgco2e_per_kwh,
            facility_id=request.facility_id,
            supplier_id=request.supplier_id,
            tenant_id=request.tenant_id,
            tier=request.calculation_tier,
            gwp_source=request.gwp_source,
        )

    def calculate_district_network(
        self,
        cooling_kwh_th: Decimal,
        region: str = "global_default",
        distribution_loss_pct: Optional[Decimal] = None,
        pump_energy_kwh: Optional[Decimal] = None,
        grid_ef: Optional[Decimal] = None,
        facility_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        tenant_id: str = "default",
        tier: DataQualityTier = DataQualityTier.TIER_1,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> CalculationResult:
        """Calculate district cooling network emissions from parameters.

        This is the primary public method for district cooling calculations.
        It accepts individual parameters (rather than a request model) for
        flexibility and programmatic use.

        Args:
            cooling_kwh_th: Cooling energy output delivered at the building
                meter in kWh thermal.  Must be positive.
            region: Geographic region identifier for emission factor lookup.
                Defaults to ``"global_default"``.
            distribution_loss_pct: Fraction of cooling lost in the
                distribution network (0-1).  Defaults to 0.08 (8%).
            pump_energy_kwh: Metered pump electricity in kWh.  If None,
                estimated as ``cooling_kwh_th * 0.03``.
            grid_ef: Grid electricity emission factor in kgCO2e/kWh for
                pump emissions.  If None, the regional default is used.
            facility_id: Optional facility identifier for metadata.
            supplier_id: Optional supplier identifier for metadata.
            tenant_id: Tenant identifier for multi-tenancy.
            tier: Data quality tier.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            CalculationResult with emissions, trace, and provenance.

        Raises:
            ValueError: If cooling_kwh_th is not positive.
        """
        start_time = time.monotonic()
        calc_id = f"dc_{uuid.uuid4().hex[:12]}"
        trace: List[str] = []
        tier_str = tier.value if isinstance(tier, DataQualityTier) else str(tier)
        gwp_str = gwp_source.value if isinstance(gwp_source, GWPSource) else str(gwp_source)

        try:
            # -- Validation --
            cooling_kwh_th = self._validate_positive_decimal(
                cooling_kwh_th, "cooling_kwh_th",
            )
            trace.append(
                f"[1] Input: cooling_output={cooling_kwh_th} kWh_th, "
                f"region={region}, tier={tier_str}"
            )

            # -- Resolve distribution loss --
            loss_pct = self._resolve_distribution_loss(distribution_loss_pct)
            trace.append(f"[2] Distribution loss: {loss_pct}")

            # -- Adjusted cooling (gross up for losses) --
            adjusted_cooling = self.calculate_adjusted_cooling(
                cooling_kwh_th, loss_pct,
            )
            trace.append(
                f"[3] Adjusted cooling: {cooling_kwh_th} / "
                f"(1 - {loss_pct}) = {adjusted_cooling} kWh_th"
            )

            # -- Distribution loss quantity --
            distribution_loss = self.calculate_distribution_loss(
                cooling_kwh_th, loss_pct,
            )
            trace.append(f"[4] Distribution loss quantity: {distribution_loss} kWh_th")

            # -- Resolve pump energy --
            resolved_pump_kwh = self._resolve_pump_energy(
                pump_energy_kwh, cooling_kwh_th,
            )
            trace.append(f"[5] Pump energy: {resolved_pump_kwh} kWh")

            # -- Resolve grid EF for pumps --
            pump_grid_ef = self._resolve_grid_ef(grid_ef, region)
            trace.append(f"[6] Pump grid EF: {pump_grid_ef} kgCO2e/kWh")

            # -- Pump emissions --
            pump_emissions = _q(resolved_pump_kwh * pump_grid_ef)
            trace.append(
                f"[7] Pump emissions: {resolved_pump_kwh} x "
                f"{pump_grid_ef} = {pump_emissions} kgCO2e"
            )

            # -- Regional emission factor lookup --
            regional_ef = self.get_regional_ef(region)
            trace.append(
                f"[8] Regional EF ({region}): "
                f"{regional_ef} kgCO2e/GJ"
            )

            # -- Convert adjusted cooling to GJ --
            adjusted_cooling_gj = _q(adjusted_cooling * _KWH_TO_GJ)
            trace.append(
                f"[9] Adjusted cooling in GJ: {adjusted_cooling} x "
                f"{_KWH_TO_GJ} = {adjusted_cooling_gj} GJ"
            )

            # -- Generation emissions using regional factor --
            generation_emissions = _q(adjusted_cooling_gj * regional_ef)
            trace.append(
                f"[10] Generation emissions: {adjusted_cooling_gj} GJ x "
                f"{regional_ef} kgCO2e/GJ = {generation_emissions} kgCO2e"
            )

            # -- Total emissions --
            total_emissions = _q(generation_emissions + pump_emissions)
            trace.append(
                f"[11] Total emissions: {generation_emissions} + "
                f"{pump_emissions} = {total_emissions} kgCO2e"
            )

            # -- COP used (district cooling plant system COP) --
            plant_cop = self._get_district_cop(region)
            trace.append(f"[12] District plant COP: {plant_cop}")

            # -- Energy input (for reporting) --
            energy_input_kwh = _q(adjusted_cooling / plant_cop + resolved_pump_kwh)
            trace.append(f"[13] Total energy input: {energy_input_kwh} kWh")

            # -- Gas decomposition --
            gas_breakdown = self.decompose_emissions(
                total_emissions, gwp_str,
            )
            trace.append(
                f"[14] Gas decomposition: "
                f"{len(gas_breakdown)} species"
            )

            # -- Provenance --
            provenance_data = {
                "calculation_id": calc_id,
                "calculation_type": "district_cooling",
                "cooling_kwh_th": str(cooling_kwh_th),
                "region": region,
                "distribution_loss_pct": str(loss_pct),
                "pump_energy_kwh": str(resolved_pump_kwh),
                "pump_grid_ef": str(pump_grid_ef),
                "regional_ef_kgco2e_per_gj": str(regional_ef),
                "generation_emissions_kgco2e": str(generation_emissions),
                "pump_emissions_kgco2e": str(pump_emissions),
                "total_emissions_kgco2e": str(total_emissions),
                "gwp_source": gwp_str,
                "tier": tier_str,
            }
            provenance_hash = self._compute_provenance_hash(provenance_data)
            trace.append(f"[15] Provenance hash: {provenance_hash[:16]}...")

            # -- Record provenance chain --
            self._record_provenance_chain(calc_id, provenance_data)

            # -- Elapsed time --
            elapsed_ms = (time.monotonic() - start_time) * 1000
            trace.append(f"[16] Processing time: {elapsed_ms:.2f} ms")

            # -- Record metrics --
            self._record_district_metrics(
                tenant_id, tier_str, total_emissions,
                cooling_kwh_th, plant_cop, elapsed_ms,
            )

            # -- Build metadata --
            metadata = self._build_metadata(
                facility_id=facility_id,
                supplier_id=supplier_id,
                region=region,
                technology="district_cooling",
                distribution_loss_pct=str(loss_pct),
                pump_energy_kwh=str(resolved_pump_kwh),
                regional_ef_kgco2e_per_gj=str(regional_ef),
                generation_emissions_kgco2e=str(generation_emissions),
                pump_emissions_kgco2e=str(pump_emissions),
            )

            return CalculationResult(
                calculation_id=calc_id,
                calculation_type="district_cooling",
                cooling_output_kwh_th=cooling_kwh_th,
                energy_input_kwh=energy_input_kwh,
                cop_used=plant_cop,
                emissions_kgco2e=total_emissions,
                gas_breakdown=gas_breakdown,
                calculation_tier=tier if isinstance(tier, DataQualityTier) else DataQualityTier(tier),
                provenance_hash=provenance_hash,
                trace_steps=trace,
                timestamp=_utcnow(),
                metadata=metadata,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "District cooling calculation failed (id=%s): %s",
                calc_id, exc, exc_info=True,
            )
            self._record_error_metrics("calculation", "district")
            trace.append(f"[ERROR] {exc}")
            return self._build_error_result(
                calc_id, "district_cooling", cooling_kwh_th,
                tier, trace, elapsed_ms, str(exc),
            )

    def calculate_distribution_loss(
        self,
        cooling_kwh_th: Decimal,
        loss_pct: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate the absolute cooling energy lost in the distribution network.

        Args:
            cooling_kwh_th: Cooling output at the building meter in kWh thermal.
            loss_pct: Distribution loss fraction (0-1).  Defaults to 0.08.

        Returns:
            Distribution loss in kWh thermal.

        Example:
            >>> engine.calculate_distribution_loss(Decimal("100000"))
            Decimal('8695.65217391')
        """
        cooling_kwh_th = self._validate_positive_decimal(
            cooling_kwh_th, "cooling_kwh_th",
        )
        pct = loss_pct if loss_pct is not None else _DEFAULT_DISTRIBUTION_LOSS_PCT
        self._validate_fraction(pct, "loss_pct")
        adjusted = self.calculate_adjusted_cooling(cooling_kwh_th, pct)
        return _q(adjusted - cooling_kwh_th)

    def calculate_adjusted_cooling(
        self,
        cooling_kwh_th: Decimal,
        loss_pct: Optional[Decimal] = None,
    ) -> Decimal:
        """Gross up metered cooling for distribution network losses.

        Formula: Adjusted = Cooling_Output / (1 - Loss_Pct)

        Args:
            cooling_kwh_th: Cooling output at the building meter in kWh thermal.
            loss_pct: Distribution loss fraction (0-1).  Defaults to 0.08.

        Returns:
            Adjusted (grossed-up) cooling in kWh thermal.

        Raises:
            ValueError: If loss_pct >= 1.0 (would cause division by zero).
        """
        cooling_kwh_th = self._validate_positive_decimal(
            cooling_kwh_th, "cooling_kwh_th",
        )
        pct = loss_pct if loss_pct is not None else _DEFAULT_DISTRIBUTION_LOSS_PCT
        self._validate_fraction(pct, "loss_pct")
        denominator = _ONE - pct
        if denominator <= _ZERO:
            raise ValueError(
                f"Distribution loss percentage {pct} must be less than 1.0"
            )
        return _q(cooling_kwh_th / denominator)

    def calculate_pump_energy(
        self,
        cooling_kwh_th: Decimal,
        pump_ratio: Optional[Decimal] = None,
    ) -> Decimal:
        """Estimate pump electricity from cooling output and a pump ratio.

        If no pump ratio is provided, the default 3% is used.  This estimate
        is used when metered pump energy data is unavailable.

        Args:
            cooling_kwh_th: Cooling output in kWh thermal.
            pump_ratio: Fraction of cooling output consumed by pumps (0-1).
                Defaults to 0.03.

        Returns:
            Estimated pump energy in kWh electrical.
        """
        cooling_kwh_th = self._validate_positive_decimal(
            cooling_kwh_th, "cooling_kwh_th",
        )
        ratio = pump_ratio if pump_ratio is not None else _DEFAULT_PUMP_RATIO
        self._validate_fraction(ratio, "pump_ratio")
        return _q(cooling_kwh_th * ratio)

    def get_regional_ef(
        self,
        region: str,
    ) -> Decimal:
        """Look up the regional district cooling emission factor.

        Returns the composite emission factor in kgCO2e per GJ of cooling
        delivered for the specified region.  If the region is not found in
        the ``DISTRICT_COOLING_FACTORS`` table, falls back to the
        ``global_default`` factor.

        Args:
            region: Geographic region identifier (lowercase).

        Returns:
            Emission factor in kgCO2e per GJ of cooling delivered.
        """
        region_lower = region.strip().lower()
        factor = DISTRICT_COOLING_FACTORS.get(region_lower)
        if factor is not None:
            return factor.ef_kgco2e_per_gj

        logger.warning(
            "Region '%s' not found in DISTRICT_COOLING_FACTORS; "
            "falling back to global_default",
            region_lower,
        )
        return DISTRICT_COOLING_FACTORS["global_default"].ef_kgco2e_per_gj

    # ==================================================================
    # PUBLIC API: Free Cooling
    # ==================================================================

    def calculate_free_cooling(
        self,
        request: FreeCoolingRequest,
    ) -> CalculationResult:
        """Calculate Scope 2 emissions from a free cooling system.

        Accepts a validated ``FreeCoolingRequest`` and dispatches to the
        appropriate source-specific method.

        Formula:
            Pump_Energy = Cooling_Output / COP_free
            Emissions   = Pump_Energy x Grid_EF

        Args:
            request: Validated free cooling request with cooling output,
                source type, optional COP override, and grid EF.

        Returns:
            CalculationResult with emissions, trace, and provenance.
        """
        source = request.source
        cop = request.cop_override
        grid_ef = request.grid_ef_kgco2e_per_kwh

        if source == FreeCoolingSource.SEAWATER:
            return self.calculate_seawater_cooling(
                cooling_kwh_th=request.cooling_output_kwh_th,
                cop=cop, grid_ef=grid_ef,
                facility_id=request.facility_id,
                tenant_id=request.tenant_id,
                tier=request.calculation_tier,
                gwp_source=request.gwp_source,
            )
        elif source == FreeCoolingSource.LAKE:
            return self.calculate_lake_cooling(
                cooling_kwh_th=request.cooling_output_kwh_th,
                cop=cop, grid_ef=grid_ef,
                facility_id=request.facility_id,
                tenant_id=request.tenant_id,
                tier=request.calculation_tier,
                gwp_source=request.gwp_source,
            )
        elif source == FreeCoolingSource.RIVER:
            return self.calculate_river_cooling(
                cooling_kwh_th=request.cooling_output_kwh_th,
                cop=cop, grid_ef=grid_ef,
                facility_id=request.facility_id,
                tenant_id=request.tenant_id,
                tier=request.calculation_tier,
                gwp_source=request.gwp_source,
            )
        elif source == FreeCoolingSource.AMBIENT_AIR:
            return self.calculate_ambient_air_cooling(
                cooling_kwh_th=request.cooling_output_kwh_th,
                cop=cop, grid_ef=grid_ef,
                facility_id=request.facility_id,
                tenant_id=request.tenant_id,
                tier=request.calculation_tier,
                gwp_source=request.gwp_source,
            )
        else:
            raise ValueError(f"Unknown free cooling source: {source}")

    def calculate_seawater_cooling(
        self,
        cooling_kwh_th: Decimal,
        cop: Optional[Decimal] = None,
        grid_ef: Optional[Decimal] = None,
        facility_id: Optional[str] = None,
        tenant_id: str = "default",
        tier: DataQualityTier = DataQualityTier.TIER_1,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> CalculationResult:
        """Calculate emissions from seawater free cooling.

        Seawater free cooling uses deep ocean or coastal water as a heat
        sink.  Only pump electricity is consumed.  Default COP is 20.0.

        Args:
            cooling_kwh_th: Cooling output in kWh thermal.
            cop: Measured effective COP override.  Defaults to 20.0.
            grid_ef: Grid EF in kgCO2e/kWh.  Defaults to global default.
            facility_id: Optional facility identifier.
            tenant_id: Tenant identifier.
            tier: Data quality tier.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            CalculationResult for seawater free cooling.
        """
        return self._calculate_free_cooling_core(
            cooling_kwh_th=cooling_kwh_th,
            source=FreeCoolingSource.SEAWATER,
            cop=cop,
            grid_ef=grid_ef,
            facility_id=facility_id,
            tenant_id=tenant_id,
            tier=tier,
            gwp_source=gwp_source,
        )

    def calculate_lake_cooling(
        self,
        cooling_kwh_th: Decimal,
        cop: Optional[Decimal] = None,
        grid_ef: Optional[Decimal] = None,
        facility_id: Optional[str] = None,
        tenant_id: str = "default",
        tier: DataQualityTier = DataQualityTier.TIER_1,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> CalculationResult:
        """Calculate emissions from lake water free cooling.

        Deep lake water cooling uses cold hypolimnion water.  Only pump
        electricity is consumed.  Default COP is 18.0.

        Args:
            cooling_kwh_th: Cooling output in kWh thermal.
            cop: Measured effective COP override.  Defaults to 18.0.
            grid_ef: Grid EF in kgCO2e/kWh.  Defaults to global default.
            facility_id: Optional facility identifier.
            tenant_id: Tenant identifier.
            tier: Data quality tier.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            CalculationResult for lake water free cooling.
        """
        return self._calculate_free_cooling_core(
            cooling_kwh_th=cooling_kwh_th,
            source=FreeCoolingSource.LAKE,
            cop=cop,
            grid_ef=grid_ef,
            facility_id=facility_id,
            tenant_id=tenant_id,
            tier=tier,
            gwp_source=gwp_source,
        )

    def calculate_river_cooling(
        self,
        cooling_kwh_th: Decimal,
        cop: Optional[Decimal] = None,
        grid_ef: Optional[Decimal] = None,
        facility_id: Optional[str] = None,
        tenant_id: str = "default",
        tier: DataQualityTier = DataQualityTier.TIER_1,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> CalculationResult:
        """Calculate emissions from river water free cooling.

        River water cooling uses flowing river or canal water.  Only pump
        electricity is consumed.  Default COP is 15.0.

        Args:
            cooling_kwh_th: Cooling output in kWh thermal.
            cop: Measured effective COP override.  Defaults to 15.0.
            grid_ef: Grid EF in kgCO2e/kWh.  Defaults to global default.
            facility_id: Optional facility identifier.
            tenant_id: Tenant identifier.
            tier: Data quality tier.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            CalculationResult for river water free cooling.
        """
        return self._calculate_free_cooling_core(
            cooling_kwh_th=cooling_kwh_th,
            source=FreeCoolingSource.RIVER,
            cop=cop,
            grid_ef=grid_ef,
            facility_id=facility_id,
            tenant_id=tenant_id,
            tier=tier,
            gwp_source=gwp_source,
        )

    def calculate_ambient_air_cooling(
        self,
        cooling_kwh_th: Decimal,
        cop: Optional[Decimal] = None,
        grid_ef: Optional[Decimal] = None,
        facility_id: Optional[str] = None,
        tenant_id: str = "default",
        tier: DataQualityTier = DataQualityTier.TIER_1,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> CalculationResult:
        """Calculate emissions from ambient air free cooling.

        Ambient air free cooling uses dry coolers or economiser modes.
        Only fan electricity is consumed.  Default COP is 10.0.

        Args:
            cooling_kwh_th: Cooling output in kWh thermal.
            cop: Measured effective COP override.  Defaults to 10.0.
            grid_ef: Grid EF in kgCO2e/kWh.  Defaults to global default.
            facility_id: Optional facility identifier.
            tenant_id: Tenant identifier.
            tier: Data quality tier.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            CalculationResult for ambient air free cooling.
        """
        return self._calculate_free_cooling_core(
            cooling_kwh_th=cooling_kwh_th,
            source=FreeCoolingSource.AMBIENT_AIR,
            cop=cop,
            grid_ef=grid_ef,
            facility_id=facility_id,
            tenant_id=tenant_id,
            tier=tier,
            gwp_source=gwp_source,
        )

    def get_free_cooling_cop(
        self,
        source: FreeCoolingSource,
    ) -> Decimal:
        """Return the default COP for a free cooling source type.

        Args:
            source: Free cooling source (seawater, lake, river, ambient_air).

        Returns:
            Default COP value as Decimal.

        Raises:
            ValueError: If the source type is unknown.
        """
        source_val = source.value if isinstance(source, FreeCoolingSource) else str(source)
        cop = _FREE_COOLING_COP_DEFAULTS.get(source_val)
        if cop is None:
            raise ValueError(f"Unknown free cooling source: {source_val}")
        return cop

    # ==================================================================
    # PUBLIC API: Thermal Energy Storage (TES)
    # ==================================================================

    def calculate_tes(
        self,
        request: TESRequest,
    ) -> TESCalculationResult:
        """Calculate Scope 2 emissions from a TES system.

        Accepts a validated ``TESRequest`` and performs the full TES emission
        calculation including charge energy, emissions from charging, and
        optional emission savings from temporal shifting.

        Args:
            request: Validated TES request with capacity, type, charging
                COP, round-trip efficiency, and grid EFs.

        Returns:
            TESCalculationResult with charge emissions, savings, and
            provenance hash.
        """
        start_time = time.monotonic()
        calc_id = f"tes_{uuid.uuid4().hex[:12]}"
        trace: List[str] = []
        tes_type = request.tes_type
        tes_type_str = tes_type.value if isinstance(tes_type, TESType) else str(tes_type)
        tier = request.calculation_tier
        tier_str = tier.value if isinstance(tier, DataQualityTier) else str(tier)
        gwp_source = request.gwp_source
        gwp_str = gwp_source.value if isinstance(gwp_source, GWPSource) else str(gwp_source)

        try:
            capacity = self._validate_positive_decimal(
                request.tes_capacity_kwh_th, "tes_capacity_kwh_th",
            )
            trace.append(
                f"[1] Input: TES capacity={capacity} kWh_th, "
                f"type={tes_type_str}, tier={tier_str}"
            )

            # -- Resolve COP for charging --
            cop_charge = self._resolve_tes_charge_cop(
                request.cop_charge, tes_type,
            )
            trace.append(f"[2] Charging COP: {cop_charge}")

            # -- Resolve round-trip efficiency --
            rt_eff = self._resolve_tes_round_trip_efficiency(
                request.round_trip_efficiency, tes_type,
            )
            trace.append(f"[3] Round-trip efficiency: {rt_eff}")

            # -- Charge energy --
            charge_energy = self.calculate_tes_charge_energy(
                capacity, cop_charge, rt_eff,
            )
            trace.append(
                f"[4] Charge energy: ({capacity} / {cop_charge}) / "
                f"{rt_eff} = {charge_energy} kWh"
            )

            # -- Grid EF at charge time --
            grid_ef_charge = request.grid_ef_charge_kgco2e_per_kwh
            if grid_ef_charge is None or grid_ef_charge < _ZERO:
                grid_ef_charge = _GLOBAL_DEFAULT_GRID_EF
            trace.append(f"[5] Grid EF (charge): {grid_ef_charge} kgCO2e/kWh")

            # -- TES emissions --
            tes_emissions = self.calculate_tes_emissions(
                charge_energy, grid_ef_charge,
            )
            trace.append(
                f"[6] TES emissions: {charge_energy} x "
                f"{grid_ef_charge} = {tes_emissions} kgCO2e"
            )

            # -- Peak emission savings (temporal shifting) --
            emission_savings = _ZERO
            peak_emissions_avoided = _ZERO

            if request.grid_ef_peak_kgco2e_per_kwh is not None:
                grid_ef_peak = request.grid_ef_peak_kgco2e_per_kwh
                cop_peak = self._resolve_tes_peak_cop(
                    request.cop_peak, tes_type,
                )
                trace.append(
                    f"[7] Peak comparison: COP_peak={cop_peak}, "
                    f"Grid_EF_peak={grid_ef_peak} kgCO2e/kWh"
                )

                emission_savings = self.calculate_tes_savings(
                    cooling_kwh_th=capacity,
                    cop_peak=cop_peak,
                    grid_ef_peak=grid_ef_peak,
                    charge_energy_kwh=charge_energy,
                    grid_ef_charge=grid_ef_charge,
                )
                peak_emissions_avoided = _q(capacity / cop_peak * grid_ef_peak)
                trace.append(
                    f"[8] Peak emissions avoided: "
                    f"{peak_emissions_avoided} kgCO2e"
                )
                trace.append(
                    f"[9] Temporal shift savings: "
                    f"{emission_savings} kgCO2e"
                )
            else:
                trace.append("[7] No peak EF provided; skipping savings calc")

            # -- Gas decomposition --
            gas_breakdown = self.decompose_emissions(
                tes_emissions, gwp_str,
            )
            trace.append(f"[10] Gas decomposition: {len(gas_breakdown)} species")

            # -- Provenance --
            provenance_data = {
                "calculation_id": calc_id,
                "calculation_type": "tes",
                "tes_type": tes_type_str,
                "capacity_kwh_th": str(capacity),
                "cop_charge": str(cop_charge),
                "round_trip_efficiency": str(rt_eff),
                "charge_energy_kwh": str(charge_energy),
                "grid_ef_charge": str(grid_ef_charge),
                "tes_emissions_kgco2e": str(tes_emissions),
                "emission_savings_kgco2e": str(emission_savings),
                "peak_emissions_avoided_kgco2e": str(peak_emissions_avoided),
                "gwp_source": gwp_str,
                "tier": tier_str,
            }
            provenance_hash = self._compute_provenance_hash(provenance_data)
            trace.append(f"[11] Provenance hash: {provenance_hash[:16]}...")

            # -- Record provenance chain --
            self._record_provenance_chain(calc_id, provenance_data)

            # -- Elapsed --
            elapsed_ms = (time.monotonic() - start_time) * 1000
            trace.append(f"[12] Processing time: {elapsed_ms:.2f} ms")

            # -- Metrics --
            self._record_tes_metrics(
                tenant_id=request.tenant_id,
                tier_str=tier_str,
                tes_type_str=tes_type_str,
                emissions=tes_emissions,
                capacity=capacity,
                cop_charge=cop_charge,
                elapsed_ms=elapsed_ms,
            )

            # -- Build metadata --
            metadata = self._build_metadata(
                facility_id=request.facility_id,
                technology=f"tes_{tes_type_str}",
                tes_type=tes_type_str,
                round_trip_efficiency=str(rt_eff),
                charge_energy_kwh=str(charge_energy),
                grid_ef_charge=str(grid_ef_charge),
                emission_savings_kgco2e=str(emission_savings),
            )

            return TESCalculationResult(
                calculation_id=calc_id,
                calculation_type="tes",
                cooling_output_kwh_th=capacity,
                charge_energy_kwh=charge_energy,
                cop_used=cop_charge,
                emissions_kgco2e=tes_emissions,
                emission_savings_kgco2e=emission_savings,
                peak_emissions_avoided_kgco2e=peak_emissions_avoided,
                gas_breakdown=gas_breakdown,
                calculation_tier=tier if isinstance(tier, DataQualityTier) else DataQualityTier(tier),
                provenance_hash=provenance_hash,
                trace_steps=trace,
                timestamp=_utcnow(),
                metadata=metadata,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "TES calculation failed (id=%s): %s",
                calc_id, exc, exc_info=True,
            )
            self._record_error_metrics("calculation", "tes")
            trace.append(f"[ERROR] {exc}")
            return TESCalculationResult(
                calculation_id=calc_id,
                calculation_type="tes",
                cooling_output_kwh_th=request.tes_capacity_kwh_th,
                charge_energy_kwh=_ZERO,
                cop_used=_ONE,
                emissions_kgco2e=_ZERO,
                emission_savings_kgco2e=_ZERO,
                peak_emissions_avoided_kgco2e=_ZERO,
                gas_breakdown=[],
                calculation_tier=tier if isinstance(tier, DataQualityTier) else DataQualityTier(tier),
                provenance_hash="",
                trace_steps=trace,
                timestamp=_utcnow(),
                metadata={"error": str(exc)},
            )

    def calculate_tes_charge_energy(
        self,
        capacity_kwh_th: Decimal,
        cop_charge: Optional[Decimal] = None,
        round_trip_efficiency: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate the electrical energy consumed during TES charging.

        Formula:
            Charge_Energy = (Capacity / COP_charge) / Round_Trip_Efficiency

        Args:
            capacity_kwh_th: TES storage capacity in kWh thermal.
            cop_charge: COP of the chiller during charging.  Defaults to
                technology default (ice=3.2, chilled_water=5.5, pcm=4.5).
            round_trip_efficiency: TES round-trip efficiency (0-1).
                Defaults to technology default.

        Returns:
            Charge energy in kWh electrical.

        Raises:
            ValueError: If capacity, COP, or efficiency are not positive.
        """
        capacity_kwh_th = self._validate_positive_decimal(
            capacity_kwh_th, "capacity_kwh_th",
        )
        cop = cop_charge if cop_charge is not None else _TES_CHARGE_COP_DEFAULTS[TESType.CHILLED_WATER.value]
        self._validate_positive_decimal(cop, "cop_charge")
        eff = round_trip_efficiency if round_trip_efficiency is not None else _TES_ROUND_TRIP_EFFICIENCY_DEFAULTS[TESType.CHILLED_WATER.value]
        if eff <= _ZERO or eff > _ONE:
            raise ValueError(
                f"Round-trip efficiency must be in (0, 1], got {eff}"
            )
        electrical_input = _q(capacity_kwh_th / cop)
        charge_energy = _q(electrical_input / eff)
        return charge_energy

    def calculate_tes_emissions(
        self,
        charge_energy_kwh: Decimal,
        grid_ef_charge: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate emissions from TES charging electricity consumption.

        Formula:
            Emissions = Charge_Energy x Grid_EF_charge

        Args:
            charge_energy_kwh: Electrical energy consumed during charging
                in kWh.
            grid_ef_charge: Grid emission factor during charging period
                in kgCO2e/kWh.  Defaults to global default.

        Returns:
            TES charging emissions in kgCO2e.
        """
        charge_energy_kwh = self._validate_non_negative_decimal(
            charge_energy_kwh, "charge_energy_kwh",
        )
        ef = grid_ef_charge if grid_ef_charge is not None else _GLOBAL_DEFAULT_GRID_EF
        return _q(charge_energy_kwh * ef)

    def calculate_tes_savings(
        self,
        cooling_kwh_th: Decimal,
        cop_peak: Decimal,
        grid_ef_peak: Decimal,
        charge_energy_kwh: Decimal,
        grid_ef_charge: Decimal,
    ) -> Decimal:
        """Calculate emission savings from TES temporal shifting.

        Compares peak-hour emissions (without TES) to off-peak charging
        emissions (with TES) and returns the net saving.

        Formula:
            Peak_Emissions = (Cooling_Output / COP_peak) x Grid_EF_peak
            TES_Emissions  = Charge_Energy x Grid_EF_charge
            Savings        = Peak_Emissions - TES_Emissions

        A positive result indicates the TES system saves emissions. A
        negative result indicates TES actually increases emissions (e.g.
        when off-peak grid carbon intensity is higher than peak).

        Args:
            cooling_kwh_th: Cooling energy delivered by TES in kWh thermal.
            cop_peak: COP of the chiller that would run during peak hours.
            grid_ef_peak: Grid emission factor during peak hours in kgCO2e/kWh.
            charge_energy_kwh: Electrical energy consumed during TES
                charging in kWh.
            grid_ef_charge: Grid emission factor during charging in kgCO2e/kWh.

        Returns:
            Net emission savings in kgCO2e (positive = savings).
        """
        cooling_kwh_th = self._validate_positive_decimal(
            cooling_kwh_th, "cooling_kwh_th",
        )
        cop_peak = self._validate_positive_decimal(cop_peak, "cop_peak")

        peak_energy = _q(cooling_kwh_th / cop_peak)
        peak_emissions = _q(peak_energy * grid_ef_peak)
        tes_emissions = _q(charge_energy_kwh * grid_ef_charge)
        savings = _q(peak_emissions - tes_emissions)
        return savings

    def get_round_trip_efficiency(
        self,
        tes_type: TESType,
    ) -> Decimal:
        """Return the default round-trip efficiency for a TES type.

        Args:
            tes_type: TES technology type (ice, chilled_water, pcm).

        Returns:
            Default round-trip efficiency as Decimal.

        Raises:
            ValueError: If the TES type is unknown.
        """
        tes_val = tes_type.value if isinstance(tes_type, TESType) else str(tes_type)
        eff = _TES_ROUND_TRIP_EFFICIENCY_DEFAULTS.get(tes_val)
        if eff is None:
            raise ValueError(f"Unknown TES type: {tes_val}")
        return eff

    def calculate_ice_tes(
        self,
        capacity_kwh_th: Decimal,
        grid_ef_charge: Decimal,
        grid_ef_peak: Optional[Decimal] = None,
        cop_peak: Optional[Decimal] = None,
    ) -> TESCalculationResult:
        """Convenience method for ice TES calculations.

        Uses ice TES defaults: COP_charge=3.2, round-trip efficiency=0.85.

        Args:
            capacity_kwh_th: Ice TES storage capacity in kWh thermal.
            grid_ef_charge: Grid EF during off-peak charging in kgCO2e/kWh.
            grid_ef_peak: Optional grid EF during peak hours for savings.
            cop_peak: Optional peak-hour chiller COP for savings comparison.

        Returns:
            TESCalculationResult for ice TES.
        """
        return self._calculate_typed_tes(
            capacity_kwh_th=capacity_kwh_th,
            tes_type=TESType.ICE,
            grid_ef_charge=grid_ef_charge,
            grid_ef_peak=grid_ef_peak,
            cop_peak=cop_peak,
        )

    def calculate_chilled_water_tes(
        self,
        capacity_kwh_th: Decimal,
        grid_ef_charge: Decimal,
        grid_ef_peak: Optional[Decimal] = None,
        cop_peak: Optional[Decimal] = None,
    ) -> TESCalculationResult:
        """Convenience method for chilled water TES calculations.

        Uses chilled water TES defaults: COP_charge=5.5, round-trip
        efficiency=0.95.

        Args:
            capacity_kwh_th: Chilled water TES storage capacity in kWh thermal.
            grid_ef_charge: Grid EF during off-peak charging in kgCO2e/kWh.
            grid_ef_peak: Optional grid EF during peak hours for savings.
            cop_peak: Optional peak-hour chiller COP for savings comparison.

        Returns:
            TESCalculationResult for chilled water TES.
        """
        return self._calculate_typed_tes(
            capacity_kwh_th=capacity_kwh_th,
            tes_type=TESType.CHILLED_WATER,
            grid_ef_charge=grid_ef_charge,
            grid_ef_peak=grid_ef_peak,
            cop_peak=cop_peak,
        )

    def calculate_pcm_tes(
        self,
        capacity_kwh_th: Decimal,
        grid_ef_charge: Decimal,
        grid_ef_peak: Optional[Decimal] = None,
        cop_peak: Optional[Decimal] = None,
    ) -> TESCalculationResult:
        """Convenience method for PCM TES calculations.

        Uses PCM TES defaults: COP_charge=4.5, round-trip efficiency=0.90.

        Args:
            capacity_kwh_th: PCM TES storage capacity in kWh thermal.
            grid_ef_charge: Grid EF during off-peak charging in kgCO2e/kWh.
            grid_ef_peak: Optional grid EF during peak hours for savings.
            cop_peak: Optional peak-hour chiller COP for savings comparison.

        Returns:
            TESCalculationResult for PCM TES.
        """
        return self._calculate_typed_tes(
            capacity_kwh_th=capacity_kwh_th,
            tes_type=TESType.PCM,
            grid_ef_charge=grid_ef_charge,
            grid_ef_peak=grid_ef_peak,
            cop_peak=cop_peak,
        )

    # ==================================================================
    # PUBLIC API: Multi-Source Plant
    # ==================================================================

    def calculate_multi_source_plant(
        self,
        sources: List[Dict[str, Any]],
        cooling_kwh_th: Decimal,
        grid_ef: Optional[Decimal] = None,
        facility_id: Optional[str] = None,
        tenant_id: str = "default",
        tier: DataQualityTier = DataQualityTier.TIER_1,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> CalculationResult:
        """Calculate emissions from a multi-source district cooling plant.

        A multi-source plant combines multiple cooling generation sources
        (electric chillers, absorption chillers, free cooling) in a single
        facility.  Emissions are calculated as a weighted sum based on each
        source's fractional contribution.

        Formula:
            Total_Emissions = SUM(fraction_i x emission_intensity_i x cooling)

        Each source dict must contain:
            - ``type`` (str): Source type ("electric", "absorption",
              "free_cooling").
            - ``fraction`` (Decimal or str): Fraction of total cooling from
              this source (0-1).  All fractions must sum to 1.0.
            - ``cop`` (Decimal or str): COP of this source.
            - ``ef`` (Decimal or str, optional): Emission factor for this
              source in kgCO2e/kWh.  Defaults to grid EF for electric
              sources and zero for free cooling.

        Args:
            sources: List of source dictionaries with type, fraction, cop,
                and optional ef.
            cooling_kwh_th: Total cooling output from the plant in kWh
                thermal.
            grid_ef: Grid electricity emission factor in kgCO2e/kWh.
                Used as the default EF for electric sources.
            facility_id: Optional facility identifier.
            tenant_id: Tenant identifier.
            tier: Data quality tier.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            CalculationResult with total weighted emissions.

        Raises:
            ValueError: If fractions do not sum to approximately 1.0,
                if any source has invalid parameters, or if sources list
                is empty.
        """
        start_time = time.monotonic()
        calc_id = f"msp_{uuid.uuid4().hex[:12]}"
        trace: List[str] = []
        tier_str = tier.value if isinstance(tier, DataQualityTier) else str(tier)
        gwp_str = gwp_source.value if isinstance(gwp_source, GWPSource) else str(gwp_source)

        try:
            cooling_kwh_th = self._validate_positive_decimal(
                cooling_kwh_th, "cooling_kwh_th",
            )
            trace.append(
                f"[1] Multi-source plant: cooling={cooling_kwh_th} kWh_th, "
                f"sources={len(sources)}"
            )

            if not sources:
                raise ValueError("sources list must not be empty")

            resolved_grid_ef = grid_ef if grid_ef is not None else _GLOBAL_DEFAULT_GRID_EF
            trace.append(f"[2] Default grid EF: {resolved_grid_ef} kgCO2e/kWh")

            # -- Validate fractions --
            parsed_sources = self._parse_multi_source_list(
                sources, resolved_grid_ef,
            )
            total_fraction = sum(s["fraction"] for s in parsed_sources)
            if abs(total_fraction - _ONE) > Decimal("0.01"):
                raise ValueError(
                    f"Source fractions must sum to 1.0, got {total_fraction}"
                )
            trace.append(f"[3] Total fraction: {total_fraction}")

            # -- Calculate per-source emissions --
            total_emissions = _ZERO
            weighted_cop_inv = _ZERO
            source_details: List[str] = []

            for idx, src in enumerate(parsed_sources):
                src_type = src["type"]
                fraction = src["fraction"]
                cop = src["cop"]
                ef = src["ef"]

                # Cooling allocated to this source
                src_cooling = _q(cooling_kwh_th * fraction)

                # Energy input for this source
                src_energy = _q(src_cooling / cop)

                # Emissions for this source
                src_emissions = _q(src_energy * ef)

                total_emissions = _q(total_emissions + src_emissions)
                weighted_cop_inv = _q(weighted_cop_inv + fraction / cop)

                detail = (
                    f"Source {idx + 1} ({src_type}): "
                    f"fraction={fraction}, COP={cop}, EF={ef}, "
                    f"cooling={src_cooling} kWh_th, "
                    f"energy={src_energy} kWh, "
                    f"emissions={src_emissions} kgCO2e"
                )
                source_details.append(detail)
                trace.append(f"[4.{idx + 1}] {detail}")

            # -- Weighted plant COP (harmonic mean via weighted inverse) --
            weighted_cop = _q(_ONE / weighted_cop_inv) if weighted_cop_inv > _ZERO else _DEFAULT_PLANT_COP
            trace.append(f"[5] Weighted plant COP: {weighted_cop}")
            trace.append(f"[6] Total emissions: {total_emissions} kgCO2e")

            # -- Energy input (for reporting) --
            energy_input_kwh = _q(cooling_kwh_th * weighted_cop_inv)
            trace.append(f"[7] Total energy input: {energy_input_kwh} kWh")

            # -- Gas decomposition --
            gas_breakdown = self.decompose_emissions(
                total_emissions, gwp_str,
            )
            trace.append(f"[8] Gas decomposition: {len(gas_breakdown)} species")

            # -- Weighted plant EF --
            weighted_ef = self.calculate_weighted_plant_ef(sources)
            trace.append(f"[9] Weighted plant EF: {weighted_ef} kgCO2e/kWh")

            # -- Provenance --
            provenance_data = {
                "calculation_id": calc_id,
                "calculation_type": "multi_source_plant",
                "cooling_kwh_th": str(cooling_kwh_th),
                "num_sources": len(parsed_sources),
                "total_emissions_kgco2e": str(total_emissions),
                "weighted_cop": str(weighted_cop),
                "weighted_ef_kgco2e_per_kwh": str(weighted_ef),
                "gwp_source": gwp_str,
                "tier": tier_str,
            }
            provenance_hash = self._compute_provenance_hash(provenance_data)
            trace.append(f"[10] Provenance hash: {provenance_hash[:16]}...")

            # -- Record provenance chain --
            self._record_provenance_chain(calc_id, provenance_data)

            # -- Elapsed --
            elapsed_ms = (time.monotonic() - start_time) * 1000
            trace.append(f"[11] Processing time: {elapsed_ms:.2f} ms")

            # -- Metrics --
            self._record_district_metrics(
                tenant_id, tier_str, total_emissions,
                cooling_kwh_th, weighted_cop, elapsed_ms,
            )

            # -- Metadata --
            metadata = self._build_metadata(
                facility_id=facility_id,
                technology="multi_source_plant",
                num_sources=str(len(parsed_sources)),
                weighted_cop=str(weighted_cop),
                weighted_ef_kgco2e_per_kwh=str(weighted_ef),
            )

            return CalculationResult(
                calculation_id=calc_id,
                calculation_type="multi_source_plant",
                cooling_output_kwh_th=cooling_kwh_th,
                energy_input_kwh=energy_input_kwh,
                cop_used=weighted_cop,
                emissions_kgco2e=total_emissions,
                gas_breakdown=gas_breakdown,
                calculation_tier=tier if isinstance(tier, DataQualityTier) else DataQualityTier(tier),
                provenance_hash=provenance_hash,
                trace_steps=trace,
                timestamp=_utcnow(),
                metadata=metadata,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Multi-source plant calculation failed (id=%s): %s",
                calc_id, exc, exc_info=True,
            )
            self._record_error_metrics("calculation", "district")
            trace.append(f"[ERROR] {exc}")
            return self._build_error_result(
                calc_id, "multi_source_plant", cooling_kwh_th,
                tier, trace, elapsed_ms, str(exc),
            )

    def calculate_weighted_plant_ef(
        self,
        sources: List[Dict[str, Any]],
    ) -> Decimal:
        """Calculate the weighted emission factor for a multi-source plant.

        The weighted EF is computed as:
            Weighted_EF = SUM(fraction_i x EF_i / COP_i)

        giving kgCO2e per kWh of cooling output from the plant.

        Args:
            sources: List of source dicts with ``fraction``, ``cop``, and
                ``ef`` keys (same format as ``calculate_multi_source_plant``).

        Returns:
            Weighted emission factor in kgCO2e per kWh cooling output.
        """
        parsed = self._parse_multi_source_list(
            sources, _GLOBAL_DEFAULT_GRID_EF,
        )
        weighted_ef = _ZERO
        for src in parsed:
            fraction = src["fraction"]
            cop = src["cop"]
            ef = src["ef"]
            weighted_ef = _q(weighted_ef + fraction * ef / cop)
        return weighted_ef

    # ==================================================================
    # PUBLIC API: Gas Decomposition
    # ==================================================================

    def decompose_emissions(
        self,
        total_co2e: Decimal,
        gwp_source: str = "AR6",
    ) -> List[GasEmissionDetail]:
        """Decompose total CO2e into individual gas species.

        Splits the total CO2-equivalent emissions into CO2, CH4, and N2O
        contributions using global average grid electricity gas fractions.
        The decomposition uses GWP values from the specified IPCC source
        to back-calculate individual gas masses from the CO2e total.

        Args:
            total_co2e: Total emissions in kgCO2e.
            gwp_source: IPCC Assessment Report identifier (AR4, AR5, AR6,
                AR6_20YR).

        Returns:
            List of GasEmissionDetail instances for CO2, CH4, and N2O.
        """
        if total_co2e <= _ZERO:
            return [
                GasEmissionDetail(
                    gas=EmissionGas.CO2,
                    quantity_kg=_ZERO,
                    gwp_factor=_ONE,
                    co2e_kg=_ZERO,
                ),
                GasEmissionDetail(
                    gas=EmissionGas.CH4,
                    quantity_kg=_ZERO,
                    gwp_factor=self._get_gwp_ch4(gwp_source),
                    co2e_kg=_ZERO,
                ),
                GasEmissionDetail(
                    gas=EmissionGas.N2O,
                    quantity_kg=_ZERO,
                    gwp_factor=self._get_gwp_n2o(gwp_source),
                    co2e_kg=_ZERO,
                ),
            ]

        gwp_ch4 = self._get_gwp_ch4(gwp_source)
        gwp_n2o = self._get_gwp_n2o(gwp_source)

        # Allocate CO2e to each gas based on grid electricity gas fractions
        co2_co2e = _q(total_co2e * _GAS_FRACTION_CO2)
        ch4_co2e = _q(total_co2e * _GAS_FRACTION_CH4)
        n2o_co2e = _q(total_co2e * _GAS_FRACTION_N2O)

        # Back-calculate mass from CO2e using GWP
        co2_mass = co2_co2e  # GWP of CO2 is 1
        ch4_mass = _q(ch4_co2e / gwp_ch4) if gwp_ch4 > _ZERO else _ZERO
        n2o_mass = _q(n2o_co2e / gwp_n2o) if gwp_n2o > _ZERO else _ZERO

        return [
            GasEmissionDetail(
                gas=EmissionGas.CO2,
                quantity_kg=co2_mass,
                gwp_factor=_ONE,
                co2e_kg=co2_co2e,
            ),
            GasEmissionDetail(
                gas=EmissionGas.CH4,
                quantity_kg=ch4_mass,
                gwp_factor=gwp_ch4,
                co2e_kg=ch4_co2e,
            ),
            GasEmissionDetail(
                gas=EmissionGas.N2O,
                quantity_kg=n2o_mass,
                gwp_factor=gwp_n2o,
                co2e_kg=n2o_co2e,
            ),
        ]

    # ==================================================================
    # PUBLIC API: Batch Operations
    # ==================================================================

    def calculate_district_cooling_batch(
        self,
        requests: List[DistrictCoolingRequest],
    ) -> List[CalculationResult]:
        """Calculate district cooling emissions for a batch of requests.

        Processes each request sequentially and returns a list of results.
        Failures for individual requests do not abort the batch; failed
        items are returned with error information in their trace.

        Args:
            requests: List of DistrictCoolingRequest models.

        Returns:
            List of CalculationResult instances, one per request.
        """
        results: List[CalculationResult] = []
        for req in requests:
            result = self.calculate_district_cooling(req)
            results.append(result)
        return results

    def calculate_free_cooling_batch(
        self,
        requests: List[FreeCoolingRequest],
    ) -> List[CalculationResult]:
        """Calculate free cooling emissions for a batch of requests.

        Processes each request sequentially and returns a list of results.

        Args:
            requests: List of FreeCoolingRequest models.

        Returns:
            List of CalculationResult instances, one per request.
        """
        results: List[CalculationResult] = []
        for req in requests:
            result = self.calculate_free_cooling(req)
            results.append(result)
        return results

    def calculate_tes_batch(
        self,
        requests: List[TESRequest],
    ) -> List[TESCalculationResult]:
        """Calculate TES emissions for a batch of requests.

        Processes each request sequentially and returns a list of results.

        Args:
            requests: List of TESRequest models.

        Returns:
            List of TESCalculationResult instances, one per request.
        """
        results: List[TESCalculationResult] = []
        for req in requests:
            result = self.calculate_tes(req)
            results.append(result)
        return results

    # ==================================================================
    # PUBLIC API: Utility / Lookup Helpers
    # ==================================================================

    def get_all_regional_efs(self) -> Dict[str, Decimal]:
        """Return all regional district cooling emission factors.

        Returns:
            Dictionary mapping region identifiers to emission factors
            in kgCO2e per GJ.
        """
        return {
            region: factor.ef_kgco2e_per_gj
            for region, factor in DISTRICT_COOLING_FACTORS.items()
        }

    def get_all_free_cooling_cops(self) -> Dict[str, Decimal]:
        """Return all default free cooling COP values.

        Returns:
            Dictionary mapping source types to default COP values.
        """
        return dict(_FREE_COOLING_COP_DEFAULTS)

    def get_all_tes_efficiencies(self) -> Dict[str, Decimal]:
        """Return all default TES round-trip efficiencies.

        Returns:
            Dictionary mapping TES types to round-trip efficiencies.
        """
        return dict(_TES_ROUND_TRIP_EFFICIENCY_DEFAULTS)

    def get_tes_charge_cop(
        self,
        tes_type: TESType,
    ) -> Decimal:
        """Return the default charging COP for a TES type.

        Args:
            tes_type: TES technology type.

        Returns:
            Default charging COP as Decimal.

        Raises:
            ValueError: If the TES type is unknown.
        """
        tes_val = tes_type.value if isinstance(tes_type, TESType) else str(tes_type)
        cop = _TES_CHARGE_COP_DEFAULTS.get(tes_val)
        if cop is None:
            raise ValueError(f"Unknown TES type: {tes_val}")
        return cop

    def get_district_cooling_factor(
        self,
        region: str,
    ) -> Optional[DistrictCoolingFactor]:
        """Look up the full DistrictCoolingFactor record for a region.

        Args:
            region: Geographic region identifier (case-insensitive).

        Returns:
            DistrictCoolingFactor if found, None otherwise.
        """
        return DISTRICT_COOLING_FACTORS.get(region.strip().lower())

    def convert_kwh_to_gj(
        self,
        kwh_th: Decimal,
    ) -> Decimal:
        """Convert kilowatt-hours thermal to gigajoules.

        Args:
            kwh_th: Energy in kWh thermal.

        Returns:
            Energy in GJ.
        """
        return _q(kwh_th * _KWH_TO_GJ)

    def convert_gj_to_kwh(
        self,
        gj: Decimal,
    ) -> Decimal:
        """Convert gigajoules to kilowatt-hours thermal.

        Args:
            gj: Energy in GJ.

        Returns:
            Energy in kWh thermal.
        """
        kwh_per_gj = UNIT_CONVERSIONS.get(
            "gj_to_kwh_th", Decimal("277.778"),
        )
        return _q(gj * kwh_per_gj)

    # ==================================================================
    # PRIVATE: Free Cooling Core
    # ==================================================================

    def _calculate_free_cooling_core(
        self,
        cooling_kwh_th: Decimal,
        source: FreeCoolingSource,
        cop: Optional[Decimal] = None,
        grid_ef: Optional[Decimal] = None,
        facility_id: Optional[str] = None,
        tenant_id: str = "default",
        tier: DataQualityTier = DataQualityTier.TIER_1,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> CalculationResult:
        """Core free cooling calculation shared by all source types.

        Formula:
            Pump_Energy = Cooling_Output / COP_free
            Emissions   = Pump_Energy x Grid_EF

        Args:
            cooling_kwh_th: Cooling output in kWh thermal.
            source: Free cooling source type.
            cop: Optional measured COP override.
            grid_ef: Grid EF in kgCO2e/kWh.
            facility_id: Optional facility identifier.
            tenant_id: Tenant identifier.
            tier: Data quality tier.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            CalculationResult with free cooling emissions.
        """
        start_time = time.monotonic()
        source_val = source.value if isinstance(source, FreeCoolingSource) else str(source)
        calc_id = f"fc_{source_val}_{uuid.uuid4().hex[:12]}"
        trace: List[str] = []
        tier_str = tier.value if isinstance(tier, DataQualityTier) else str(tier)
        gwp_str = gwp_source.value if isinstance(gwp_source, GWPSource) else str(gwp_source)

        try:
            cooling_kwh_th = self._validate_positive_decimal(
                cooling_kwh_th, "cooling_kwh_th",
            )
            trace.append(
                f"[1] Input: cooling_output={cooling_kwh_th} kWh_th, "
                f"source={source_val}, tier={tier_str}"
            )

            # -- Resolve COP --
            resolved_cop = self._resolve_free_cooling_cop(cop, source)
            trace.append(f"[2] Free cooling COP ({source_val}): {resolved_cop}")

            # -- Pump/fan energy --
            pump_energy = _q(cooling_kwh_th / resolved_cop)
            trace.append(
                f"[3] Pump energy: {cooling_kwh_th} / {resolved_cop} "
                f"= {pump_energy} kWh"
            )

            # -- Grid EF --
            resolved_ef = grid_ef if grid_ef is not None else _GLOBAL_DEFAULT_GRID_EF
            trace.append(f"[4] Grid EF: {resolved_ef} kgCO2e/kWh")

            # -- Emissions --
            emissions = _q(pump_energy * resolved_ef)
            trace.append(
                f"[5] Emissions: {pump_energy} x {resolved_ef} "
                f"= {emissions} kgCO2e"
            )

            # -- Gas decomposition --
            gas_breakdown = self.decompose_emissions(emissions, gwp_str)
            trace.append(f"[6] Gas decomposition: {len(gas_breakdown)} species")

            # -- Provenance --
            provenance_data = {
                "calculation_id": calc_id,
                "calculation_type": f"free_cooling_{source_val}",
                "cooling_kwh_th": str(cooling_kwh_th),
                "source": source_val,
                "cop": str(resolved_cop),
                "grid_ef": str(resolved_ef),
                "pump_energy_kwh": str(pump_energy),
                "emissions_kgco2e": str(emissions),
                "gwp_source": gwp_str,
                "tier": tier_str,
            }
            provenance_hash = self._compute_provenance_hash(provenance_data)
            trace.append(f"[7] Provenance hash: {provenance_hash[:16]}...")

            # -- Record provenance chain --
            self._record_provenance_chain(calc_id, provenance_data)

            # -- Elapsed --
            elapsed_ms = (time.monotonic() - start_time) * 1000
            trace.append(f"[8] Processing time: {elapsed_ms:.2f} ms")

            # -- Metrics --
            self._record_free_cooling_metrics(
                tenant_id, tier_str, source_val, emissions,
                cooling_kwh_th, resolved_cop, elapsed_ms,
            )

            # -- Metadata --
            metadata = self._build_metadata(
                facility_id=facility_id,
                technology=f"free_cooling_{source_val}",
                source=source_val,
                cop=str(resolved_cop),
                pump_energy_kwh=str(pump_energy),
            )

            # Map free cooling source to CoolingTechnology for metadata
            technology_map = {
                FreeCoolingSource.SEAWATER.value: CoolingTechnology.SEAWATER_FREE.value,
                FreeCoolingSource.LAKE.value: CoolingTechnology.LAKE_FREE.value,
                FreeCoolingSource.RIVER.value: CoolingTechnology.RIVER_FREE.value,
                FreeCoolingSource.AMBIENT_AIR.value: CoolingTechnology.AMBIENT_AIR_FREE.value,
            }
            cooling_tech = technology_map.get(source_val, "free_cooling")
            metadata["cooling_technology"] = cooling_tech

            return CalculationResult(
                calculation_id=calc_id,
                calculation_type=f"free_cooling_{source_val}",
                cooling_output_kwh_th=cooling_kwh_th,
                energy_input_kwh=pump_energy,
                cop_used=resolved_cop,
                emissions_kgco2e=emissions,
                gas_breakdown=gas_breakdown,
                calculation_tier=tier if isinstance(tier, DataQualityTier) else DataQualityTier(tier),
                provenance_hash=provenance_hash,
                trace_steps=trace,
                timestamp=_utcnow(),
                metadata=metadata,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Free cooling (%s) calculation failed (id=%s): %s",
                source_val, calc_id, exc, exc_info=True,
            )
            self._record_error_metrics("calculation", "free_cooling")
            trace.append(f"[ERROR] {exc}")
            return self._build_error_result(
                calc_id, f"free_cooling_{source_val}", cooling_kwh_th,
                tier, trace, elapsed_ms, str(exc),
            )

    # ==================================================================
    # PRIVATE: Typed TES Helper
    # ==================================================================

    def _calculate_typed_tes(
        self,
        capacity_kwh_th: Decimal,
        tes_type: TESType,
        grid_ef_charge: Decimal,
        grid_ef_peak: Optional[Decimal] = None,
        cop_peak: Optional[Decimal] = None,
        facility_id: Optional[str] = None,
        tenant_id: str = "default",
        tier: DataQualityTier = DataQualityTier.TIER_1,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> TESCalculationResult:
        """Build a TESRequest and delegate to calculate_tes.

        Args:
            capacity_kwh_th: TES capacity in kWh thermal.
            tes_type: TES technology type.
            grid_ef_charge: Grid EF during charging.
            grid_ef_peak: Optional peak grid EF.
            cop_peak: Optional peak COP.
            facility_id: Optional facility identifier.
            tenant_id: Tenant identifier.
            tier: Data quality tier.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            TESCalculationResult.
        """
        tes_type_str = tes_type.value if isinstance(tes_type, TESType) else str(tes_type)

        # Use type-specific defaults
        cop_charge_default = _TES_CHARGE_COP_DEFAULTS.get(
            tes_type_str, Decimal("5.0"),
        )
        rt_eff_default = _TES_ROUND_TRIP_EFFICIENCY_DEFAULTS.get(
            tes_type_str, Decimal("0.90"),
        )

        request = TESRequest(
            tes_capacity_kwh_th=capacity_kwh_th,
            tes_type=tes_type if isinstance(tes_type, TESType) else TESType(tes_type),
            cop_charge=cop_charge_default,
            round_trip_efficiency=rt_eff_default,
            grid_ef_charge_kgco2e_per_kwh=grid_ef_charge,
            grid_ef_peak_kgco2e_per_kwh=grid_ef_peak,
            cop_peak=cop_peak,
            facility_id=facility_id,
            tenant_id=tenant_id,
            calculation_tier=tier if isinstance(tier, DataQualityTier) else DataQualityTier(tier),
            gwp_source=gwp_source if isinstance(gwp_source, GWPSource) else GWPSource(gwp_source),
        )
        return self.calculate_tes(request)

    # ==================================================================
    # PRIVATE: Resolution Helpers
    # ==================================================================

    def _resolve_distribution_loss(
        self,
        loss_pct: Optional[Decimal],
    ) -> Decimal:
        """Resolve distribution loss percentage with default fallback.

        Args:
            loss_pct: Provided loss percentage or None.

        Returns:
            Resolved loss percentage as Decimal.
        """
        if loss_pct is not None:
            self._validate_fraction(loss_pct, "distribution_loss_pct")
            return loss_pct
        return self._config.default_distribution_loss

    def _resolve_pump_energy(
        self,
        pump_energy_kwh: Optional[Decimal],
        cooling_kwh_th: Decimal,
    ) -> Decimal:
        """Resolve pump energy with default estimation if not provided.

        If no metered pump energy is available, estimates pump electricity
        as 3% of cooling output.

        Args:
            pump_energy_kwh: Metered pump energy or None.
            cooling_kwh_th: Cooling output in kWh thermal.

        Returns:
            Pump energy in kWh.
        """
        if pump_energy_kwh is not None:
            return self._validate_non_negative_decimal(
                pump_energy_kwh, "pump_energy_kwh",
            )
        return self.calculate_pump_energy(cooling_kwh_th)

    def _resolve_grid_ef(
        self,
        grid_ef: Optional[Decimal],
        region: str = "global_default",
    ) -> Decimal:
        """Resolve grid emission factor with regional/global fallback.

        Priority:
        1. Explicit grid_ef parameter (if provided)
        2. Global default grid EF

        Args:
            grid_ef: Provided grid EF or None.
            region: Region identifier (for logging context).

        Returns:
            Resolved grid EF in kgCO2e/kWh.
        """
        if grid_ef is not None:
            return grid_ef
        return _GLOBAL_DEFAULT_GRID_EF

    def _resolve_free_cooling_cop(
        self,
        cop: Optional[Decimal],
        source: FreeCoolingSource,
    ) -> Decimal:
        """Resolve free cooling COP with source-specific default.

        Args:
            cop: Measured COP override or None.
            source: Free cooling source type.

        Returns:
            Resolved COP value.
        """
        if cop is not None:
            return self._validate_positive_decimal(cop, "cop")
        return self.get_free_cooling_cop(source)

    def _resolve_tes_charge_cop(
        self,
        cop: Optional[Decimal],
        tes_type: TESType,
    ) -> Decimal:
        """Resolve TES charging COP with type-specific default.

        Args:
            cop: Provided charging COP or None.
            tes_type: TES technology type.

        Returns:
            Resolved charging COP.
        """
        if cop is not None:
            return self._validate_positive_decimal(cop, "cop_charge")
        tes_val = tes_type.value if isinstance(tes_type, TESType) else str(tes_type)
        return _TES_CHARGE_COP_DEFAULTS.get(tes_val, Decimal("5.0"))

    def _resolve_tes_peak_cop(
        self,
        cop: Optional[Decimal],
        tes_type: TESType,
    ) -> Decimal:
        """Resolve TES peak-hour COP with default.

        Args:
            cop: Provided peak COP or None.
            tes_type: TES technology type (unused; default applied).

        Returns:
            Resolved peak COP.
        """
        if cop is not None:
            return self._validate_positive_decimal(cop, "cop_peak")
        return _TES_PEAK_COP_DEFAULT

    def _resolve_tes_round_trip_efficiency(
        self,
        eff: Optional[Decimal],
        tes_type: TESType,
    ) -> Decimal:
        """Resolve TES round-trip efficiency with type-specific default.

        Args:
            eff: Provided efficiency or None.
            tes_type: TES technology type.

        Returns:
            Resolved round-trip efficiency.
        """
        if eff is not None:
            if eff <= _ZERO or eff > _ONE:
                raise ValueError(
                    f"Round-trip efficiency must be in (0, 1], got {eff}"
                )
            return eff
        return self.get_round_trip_efficiency(tes_type)

    def _get_district_cop(
        self,
        region: str,
    ) -> Decimal:
        """Return the system COP for a district cooling network region.

        Currently returns the default plant COP (4.0) for all regions.
        This could be extended with region-specific plant COP data.

        Args:
            region: Geographic region identifier.

        Returns:
            District cooling plant COP.
        """
        spec = COOLING_TECHNOLOGY_SPECS.get(
            CoolingTechnology.DISTRICT_COOLING.value,
        )
        if spec is not None:
            return spec.cop_default
        return _DEFAULT_PLANT_COP

    # ==================================================================
    # PRIVATE: Validation Helpers
    # ==================================================================

    def _validate_positive_decimal(
        self,
        value: Any,
        name: str,
    ) -> Decimal:
        """Validate and convert a value to a positive Decimal.

        Args:
            value: Value to validate (Decimal, str, int, float).
            name: Parameter name for error messages.

        Returns:
            Validated positive Decimal.

        Raises:
            ValueError: If value is not positive.
            TypeError: If value cannot be converted to Decimal.
        """
        try:
            dec_value = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a valid number, got {value!r}"
            ) from exc

        if dec_value <= _ZERO:
            raise ValueError(
                f"{name} must be positive, got {dec_value}"
            )
        return dec_value

    def _validate_non_negative_decimal(
        self,
        value: Any,
        name: str,
    ) -> Decimal:
        """Validate and convert a value to a non-negative Decimal.

        Args:
            value: Value to validate.
            name: Parameter name for error messages.

        Returns:
            Validated non-negative Decimal.

        Raises:
            ValueError: If value is negative.
        """
        try:
            dec_value = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a valid number, got {value!r}"
            ) from exc

        if dec_value < _ZERO:
            raise ValueError(
                f"{name} must be non-negative, got {dec_value}"
            )
        return dec_value

    def _validate_fraction(
        self,
        value: Decimal,
        name: str,
    ) -> None:
        """Validate that a Decimal is in [0, 1].

        Args:
            value: Fraction to validate.
            name: Parameter name for error messages.

        Raises:
            ValueError: If value is outside [0, 1].
        """
        if value < _ZERO or value > _ONE:
            raise ValueError(
                f"{name} must be in [0, 1], got {value}"
            )

    # ==================================================================
    # PRIVATE: GWP Helpers
    # ==================================================================

    def _get_gwp_ch4(self, gwp_source: str) -> Decimal:
        """Get the GWP value for CH4 from the specified source.

        Args:
            gwp_source: IPCC Assessment Report identifier.

        Returns:
            CH4 GWP value as Decimal.
        """
        source_upper = gwp_source.upper()
        gwp_set = GWP_VALUES.get(source_upper, GWP_VALUES.get("AR6", {}))
        return gwp_set.get("CH4", Decimal("27.9"))

    def _get_gwp_n2o(self, gwp_source: str) -> Decimal:
        """Get the GWP value for N2O from the specified source.

        Args:
            gwp_source: IPCC Assessment Report identifier.

        Returns:
            N2O GWP value as Decimal.
        """
        source_upper = gwp_source.upper()
        gwp_set = GWP_VALUES.get(source_upper, GWP_VALUES.get("AR6", {}))
        return gwp_set.get("N2O", Decimal("273"))

    # ==================================================================
    # PRIVATE: Multi-Source Parsing
    # ==================================================================

    def _parse_multi_source_list(
        self,
        sources: List[Dict[str, Any]],
        default_ef: Decimal,
    ) -> List[Dict[str, Any]]:
        """Parse and validate multi-source plant source list.

        Converts string values to Decimal and applies defaults for
        missing emission factors.

        Args:
            sources: Raw source dictionaries.
            default_ef: Default grid EF for electric sources.

        Returns:
            List of parsed source dictionaries with Decimal values.

        Raises:
            ValueError: If any source has invalid parameters.
        """
        parsed: List[Dict[str, Any]] = []

        for idx, src in enumerate(sources):
            src_type = str(src.get("type", "electric"))

            # -- Parse fraction --
            try:
                fraction = Decimal(str(src.get("fraction", "0")))
            except (InvalidOperation, TypeError, ValueError):
                raise ValueError(
                    f"Source {idx + 1}: invalid fraction: {src.get('fraction')}"
                )
            if fraction < _ZERO or fraction > _ONE:
                raise ValueError(
                    f"Source {idx + 1}: fraction must be in [0, 1], got {fraction}"
                )

            # -- Parse COP --
            try:
                cop = Decimal(str(src.get("cop", "4.0")))
            except (InvalidOperation, TypeError, ValueError):
                raise ValueError(
                    f"Source {idx + 1}: invalid COP: {src.get('cop')}"
                )
            if cop <= _ZERO:
                raise ValueError(
                    f"Source {idx + 1}: COP must be positive, got {cop}"
                )

            # -- Parse EF --
            ef_raw = src.get("ef")
            if ef_raw is not None:
                try:
                    ef = Decimal(str(ef_raw))
                except (InvalidOperation, TypeError, ValueError):
                    raise ValueError(
                        f"Source {idx + 1}: invalid EF: {ef_raw}"
                    )
            else:
                # Default: electric uses grid EF, free cooling uses grid EF
                # (pump only), absorption uses zero (heat-driven)
                if src_type in ("absorption", "waste_heat"):
                    ef = _ZERO
                else:
                    ef = default_ef

            parsed.append({
                "type": src_type,
                "fraction": fraction,
                "cop": cop,
                "ef": ef,
            })

        return parsed

    # ==================================================================
    # PRIVATE: Provenance and Hashing
    # ==================================================================

    def _compute_provenance_hash(
        self,
        data: Dict[str, Any],
    ) -> str:
        """Compute a SHA-256 hash of calculation data for audit trail.

        Args:
            data: Dictionary of calculation inputs and outputs.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        canonical = _canonical_json(data)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _record_provenance_chain(
        self,
        calc_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Record a provenance chain entry for the calculation.

        If provenance tracking is disabled, this is a no-op.

        Args:
            calc_id: Calculation identifier used as chain ID.
            data: Calculation data to record.
        """
        if not self._enable_provenance or self._provenance is None:
            return

        try:
            chain_id = self._provenance.create_chain(calc_id)
            self._provenance.add_stage(
                chain_id,
                "INPUT_VALIDATION",
                {
                    "calculation_id": calc_id,
                    "calculation_type": data.get("calculation_type", "unknown"),
                },
            )

            calc_type = data.get("calculation_type", "unknown")
            if "district" in calc_type:
                self._provenance.add_stage(
                    chain_id,
                    "DISTRICT_LOSS_ADJUSTMENT",
                    {
                        "distribution_loss_pct": data.get("distribution_loss_pct", ""),
                        "pump_energy_kwh": data.get("pump_energy_kwh", ""),
                        "regional_ef": data.get("regional_ef_kgco2e_per_gj", ""),
                    },
                )
            elif "free_cooling" in calc_type:
                self._provenance.add_stage(
                    chain_id,
                    "FREE_COOLING_CALCULATION",
                    {
                        "source": data.get("source", ""),
                        "cop": data.get("cop", ""),
                        "pump_energy_kwh": data.get("pump_energy_kwh", ""),
                    },
                )
            elif calc_type == "tes":
                self._provenance.add_stage(
                    chain_id,
                    "TES_TEMPORAL_SHIFTING",
                    {
                        "tes_type": data.get("tes_type", ""),
                        "charge_energy_kwh": data.get("charge_energy_kwh", ""),
                        "grid_ef_charge": data.get("grid_ef_charge", ""),
                        "emission_savings_kgco2e": data.get("emission_savings_kgco2e", ""),
                    },
                )
            elif "multi_source" in calc_type:
                self._provenance.add_stage(
                    chain_id,
                    "ENERGY_INPUT_CALCULATION",
                    {
                        "num_sources": data.get("num_sources", 0),
                        "weighted_cop": data.get("weighted_cop", ""),
                    },
                )

            # Grid factor application
            self._provenance.add_stage(
                chain_id,
                "GRID_FACTOR_APPLICATION",
                {
                    "total_emissions_kgco2e": data.get("total_emissions_kgco2e",
                                                        data.get("emissions_kgco2e",
                                                                 data.get("tes_emissions_kgco2e", ""))),
                    "gwp_source": data.get("gwp_source", "AR6"),
                },
            )

            # Gas decomposition
            self._provenance.add_stage(
                chain_id,
                "GAS_DECOMPOSITION",
                {
                    "total_co2e": data.get("total_emissions_kgco2e",
                                           data.get("emissions_kgco2e",
                                                    data.get("tes_emissions_kgco2e", ""))),
                    "gwp_source": data.get("gwp_source", "AR6"),
                },
            )

            # Result finalization
            self._provenance.add_stage(
                chain_id,
                "RESULT_FINALIZATION",
                {
                    "total_co2e_kg": data.get("total_emissions_kgco2e",
                                              data.get("emissions_kgco2e",
                                                       data.get("tes_emissions_kgco2e", ""))),
                    "validation_status": "PASS",
                },
            )

            self._provenance.seal_chain(chain_id)

        except Exception as exc:
            logger.warning(
                "Failed to record provenance chain for %s: %s",
                calc_id, exc,
            )

    # ==================================================================
    # PRIVATE: Metrics Recording
    # ==================================================================

    def _record_district_metrics(
        self,
        tenant_id: str,
        tier_str: str,
        emissions: Decimal,
        cooling_kwh_th: Decimal,
        cop: Decimal,
        elapsed_ms: float,
    ) -> None:
        """Record Prometheus metrics for a district cooling calculation.

        Args:
            tenant_id: Tenant identifier.
            tier_str: Data quality tier string.
            emissions: Total emissions in kgCO2e.
            cooling_kwh_th: Cooling output in kWh thermal.
            cop: COP used.
            elapsed_ms: Processing time in milliseconds.
        """
        if not self._enable_metrics:
            return
        try:
            record_calculation(
                technology="district_cooling",
                calculation_type="district",
                tier=tier_str,
                tenant_id=tenant_id,
                status="success",
                duration_s=elapsed_ms / 1000,
                emissions_kgco2e=float(emissions),
                cooling_kwh_th=float(cooling_kwh_th),
                cop_used=float(cop),
                condenser_type="unknown",
            )
        except Exception as exc:
            logger.debug("Failed to record district metrics: %s", exc)

    def _record_free_cooling_metrics(
        self,
        tenant_id: str,
        tier_str: str,
        source: str,
        emissions: Decimal,
        cooling_kwh_th: Decimal,
        cop: Decimal,
        elapsed_ms: float,
    ) -> None:
        """Record Prometheus metrics for a free cooling calculation.

        Args:
            tenant_id: Tenant identifier.
            tier_str: Data quality tier string.
            source: Free cooling source type.
            emissions: Total emissions in kgCO2e.
            cooling_kwh_th: Cooling output in kWh thermal.
            cop: COP used.
            elapsed_ms: Processing time in milliseconds.
        """
        if not self._enable_metrics:
            return
        try:
            record_calculation(
                technology="free_cooling",
                calculation_type="free_cooling",
                tier=tier_str,
                tenant_id=tenant_id,
                status="success",
                duration_s=elapsed_ms / 1000,
                emissions_kgco2e=float(emissions),
                cooling_kwh_th=float(cooling_kwh_th),
                cop_used=float(cop),
                condenser_type=source,
            )
        except Exception as exc:
            logger.debug("Failed to record free cooling metrics: %s", exc)

    def _record_tes_metrics(
        self,
        tenant_id: str,
        tier_str: str,
        tes_type_str: str,
        emissions: Decimal,
        capacity: Decimal,
        cop_charge: Decimal,
        elapsed_ms: float,
    ) -> None:
        """Record Prometheus metrics for a TES calculation.

        Args:
            tenant_id: Tenant identifier.
            tier_str: Data quality tier string.
            tes_type_str: TES type string.
            emissions: TES charging emissions in kgCO2e.
            capacity: TES capacity in kWh thermal.
            cop_charge: Charging COP.
            elapsed_ms: Processing time in milliseconds.
        """
        if not self._enable_metrics:
            return

        tech_map = {
            TESType.ICE.value: "ice_storage",
            TESType.CHILLED_WATER.value: "chilled_water_storage",
            TESType.PCM.value: "phase_change_material",
        }
        technology = tech_map.get(tes_type_str, "thermal_energy_storage")

        try:
            record_calculation(
                technology=technology,
                calculation_type="tes",
                tier=tier_str,
                tenant_id=tenant_id,
                status="success",
                duration_s=elapsed_ms / 1000,
                emissions_kgco2e=float(emissions),
                cooling_kwh_th=float(capacity),
                cop_used=float(cop_charge),
                condenser_type="unknown",
            )
        except Exception as exc:
            logger.debug("Failed to record TES metrics: %s", exc)

    def _record_error_metrics(
        self,
        error_type: str,
        operation: str,
    ) -> None:
        """Record error metrics.

        Args:
            error_type: Error classification.
            operation: Operation context.
        """
        if not self._enable_metrics:
            return
        try:
            record_error(error_type, operation)
        except Exception as exc:
            logger.debug("Failed to record error metrics: %s", exc)

    # ==================================================================
    # PRIVATE: Result Builders
    # ==================================================================

    def _build_error_result(
        self,
        calc_id: str,
        calc_type: str,
        cooling_kwh_th: Decimal,
        tier: DataQualityTier,
        trace: List[str],
        elapsed_ms: float,
        error_message: str,
    ) -> CalculationResult:
        """Build an error CalculationResult when processing fails.

        Args:
            calc_id: Calculation identifier.
            calc_type: Calculation type string.
            cooling_kwh_th: Original cooling input.
            tier: Data quality tier.
            trace: Trace steps accumulated before failure.
            elapsed_ms: Elapsed processing time.
            error_message: Error description.

        Returns:
            CalculationResult with zero emissions and error metadata.
        """
        return CalculationResult(
            calculation_id=calc_id,
            calculation_type=calc_type,
            cooling_output_kwh_th=cooling_kwh_th,
            energy_input_kwh=_ZERO,
            cop_used=_ONE,
            emissions_kgco2e=_ZERO,
            gas_breakdown=[],
            calculation_tier=tier if isinstance(tier, DataQualityTier) else DataQualityTier(tier),
            provenance_hash="",
            trace_steps=trace,
            timestamp=_utcnow(),
            metadata={
                "error": error_message,
                "processing_time_ms": f"{elapsed_ms:.2f}",
            },
        )

    def _build_metadata(
        self,
        **kwargs: Optional[str],
    ) -> Dict[str, str]:
        """Build a metadata dictionary from keyword arguments.

        Filters out None values and ensures all values are strings.

        Args:
            **kwargs: Key-value pairs for metadata.

        Returns:
            Filtered metadata dictionary.
        """
        return {
            k: str(v)
            for k, v in kwargs.items()
            if v is not None
        }

    # ==================================================================
    # PRIVATE: Summary and Diagnostic Methods
    # ==================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """Return diagnostic information about this engine instance.

        Returns:
            Dictionary with engine version, configuration, and status.
        """
        return {
            "engine": "DistrictCoolingCalculatorEngine",
            "agent": "AGENT-MRV-012",
            "version": self._config.version,
            "enable_provenance": self._enable_provenance,
            "enable_metrics": self._enable_metrics,
            "default_distribution_loss_pct": str(self._config.default_distribution_loss),
            "default_pump_ratio": str(_DEFAULT_PUMP_RATIO),
            "default_plant_cop": str(_DEFAULT_PLANT_COP),
            "global_default_grid_ef": str(_GLOBAL_DEFAULT_GRID_EF),
            "free_cooling_cop_defaults": {
                k: str(v) for k, v in _FREE_COOLING_COP_DEFAULTS.items()
            },
            "tes_round_trip_efficiencies": {
                k: str(v) for k, v in _TES_ROUND_TRIP_EFFICIENCY_DEFAULTS.items()
            },
            "tes_charge_cop_defaults": {
                k: str(v) for k, v in _TES_CHARGE_COP_DEFAULTS.items()
            },
            "district_cooling_regions": list(DISTRICT_COOLING_FACTORS.keys()),
            "precision_decimal_places": 8,
        }

    def validate_engine_ready(self) -> Dict[str, Any]:
        """Validate that the engine is ready for calculations.

        Checks that all required constants, configuration, and services
        are available.

        Returns:
            Dictionary with ``ready`` flag and any error messages.
        """
        errors: List[str] = []

        # Check configuration
        if self._config is None:
            errors.append("Configuration not loaded")

        # Check district cooling factors
        if not DISTRICT_COOLING_FACTORS:
            errors.append("DISTRICT_COOLING_FACTORS table is empty")

        # Check free cooling defaults
        if not _FREE_COOLING_COP_DEFAULTS:
            errors.append("Free cooling COP defaults are empty")

        # Check TES defaults
        if not _TES_ROUND_TRIP_EFFICIENCY_DEFAULTS:
            errors.append("TES round-trip efficiency defaults are empty")

        # Check GWP values
        if not GWP_VALUES:
            errors.append("GWP_VALUES table is empty")

        # Check provenance
        if self._enable_provenance and self._provenance is None:
            errors.append("Provenance tracking enabled but not initialized")

        return {
            "ready": len(errors) == 0,
            "errors": errors,
            "engine": "DistrictCoolingCalculatorEngine",
        }

    # ==================================================================
    # PRIVATE: Comparison and Analysis Helpers
    # ==================================================================

    def compare_district_vs_free_cooling(
        self,
        cooling_kwh_th: Decimal,
        region: str,
        free_source: FreeCoolingSource,
        grid_ef: Optional[Decimal] = None,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> Dict[str, Any]:
        """Compare emissions between district cooling and free cooling.

        Calculates emissions for both district cooling and a free cooling
        alternative for the same cooling demand, enabling comparison and
        decision support.

        Args:
            cooling_kwh_th: Cooling demand in kWh thermal.
            region: District cooling region.
            free_source: Free cooling source type.
            grid_ef: Optional grid EF for free cooling pump emissions.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            Dictionary with both results and the difference.
        """
        district_result = self.calculate_district_network(
            cooling_kwh_th=cooling_kwh_th,
            region=region,
            grid_ef=grid_ef,
            gwp_source=gwp_source,
        )

        free_result = self._calculate_free_cooling_core(
            cooling_kwh_th=cooling_kwh_th,
            source=free_source,
            grid_ef=grid_ef,
            gwp_source=gwp_source,
        )

        difference = _q(
            district_result.emissions_kgco2e - free_result.emissions_kgco2e
        )
        pct_savings = _ZERO
        if district_result.emissions_kgco2e > _ZERO:
            pct_savings = _q(
                difference / district_result.emissions_kgco2e * Decimal("100")
            )

        return {
            "district_cooling": {
                "calculation_id": district_result.calculation_id,
                "emissions_kgco2e": str(district_result.emissions_kgco2e),
                "cop_used": str(district_result.cop_used),
                "region": region,
            },
            "free_cooling": {
                "calculation_id": free_result.calculation_id,
                "emissions_kgco2e": str(free_result.emissions_kgco2e),
                "cop_used": str(free_result.cop_used),
                "source": free_source.value if isinstance(free_source, FreeCoolingSource) else str(free_source),
            },
            "difference_kgco2e": str(difference),
            "pct_savings_with_free_cooling": str(pct_savings),
            "free_cooling_is_lower": free_result.emissions_kgco2e < district_result.emissions_kgco2e,
        }

    def compare_tes_types(
        self,
        capacity_kwh_th: Decimal,
        grid_ef_charge: Decimal,
        grid_ef_peak: Optional[Decimal] = None,
        cop_peak: Optional[Decimal] = None,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> Dict[str, Any]:
        """Compare emissions and savings across all three TES types.

        Calculates TES emissions for ice, chilled water, and PCM storage
        for the same capacity, enabling technology selection.

        Args:
            capacity_kwh_th: TES capacity in kWh thermal.
            grid_ef_charge: Grid EF during off-peak charging.
            grid_ef_peak: Optional peak grid EF for savings.
            cop_peak: Optional peak COP for savings.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            Dictionary with results for all three TES types and ranking.
        """
        ice_result = self.calculate_ice_tes(
            capacity_kwh_th, grid_ef_charge, grid_ef_peak, cop_peak,
        )
        cw_result = self.calculate_chilled_water_tes(
            capacity_kwh_th, grid_ef_charge, grid_ef_peak, cop_peak,
        )
        pcm_result = self.calculate_pcm_tes(
            capacity_kwh_th, grid_ef_charge, grid_ef_peak, cop_peak,
        )

        # Rank by emissions (lowest first)
        ranked = sorted(
            [
                ("ice", ice_result),
                ("chilled_water", cw_result),
                ("pcm", pcm_result),
            ],
            key=lambda x: x[1].emissions_kgco2e,
        )

        return {
            "capacity_kwh_th": str(capacity_kwh_th),
            "grid_ef_charge": str(grid_ef_charge),
            "grid_ef_peak": str(grid_ef_peak) if grid_ef_peak else None,
            "ice": {
                "emissions_kgco2e": str(ice_result.emissions_kgco2e),
                "charge_energy_kwh": str(ice_result.charge_energy_kwh),
                "cop_used": str(ice_result.cop_used),
                "emission_savings_kgco2e": str(ice_result.emission_savings_kgco2e),
                "round_trip_efficiency": str(_TES_ROUND_TRIP_EFFICIENCY_DEFAULTS[TESType.ICE.value]),
            },
            "chilled_water": {
                "emissions_kgco2e": str(cw_result.emissions_kgco2e),
                "charge_energy_kwh": str(cw_result.charge_energy_kwh),
                "cop_used": str(cw_result.cop_used),
                "emission_savings_kgco2e": str(cw_result.emission_savings_kgco2e),
                "round_trip_efficiency": str(_TES_ROUND_TRIP_EFFICIENCY_DEFAULTS[TESType.CHILLED_WATER.value]),
            },
            "pcm": {
                "emissions_kgco2e": str(pcm_result.emissions_kgco2e),
                "charge_energy_kwh": str(pcm_result.charge_energy_kwh),
                "cop_used": str(pcm_result.cop_used),
                "emission_savings_kgco2e": str(pcm_result.emission_savings_kgco2e),
                "round_trip_efficiency": str(_TES_ROUND_TRIP_EFFICIENCY_DEFAULTS[TESType.PCM.value]),
            },
            "ranking_by_emissions": [
                {"type": r[0], "emissions_kgco2e": str(r[1].emissions_kgco2e)}
                for r in ranked
            ],
            "lowest_emissions_type": ranked[0][0],
        }

    def estimate_annual_district_cooling(
        self,
        monthly_cooling_kwh_th: List[Decimal],
        region: str,
        distribution_loss_pct: Optional[Decimal] = None,
        grid_ef: Optional[Decimal] = None,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> Dict[str, Any]:
        """Estimate annual district cooling emissions from monthly data.

        Calculates emissions for each month and aggregates the annual
        total.  Useful for reporting-period emission estimates.

        Args:
            monthly_cooling_kwh_th: List of 12 monthly cooling values
                in kWh thermal.
            region: District cooling region.
            distribution_loss_pct: Optional distribution loss override.
            grid_ef: Optional grid EF override.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            Dictionary with monthly results and annual totals.

        Raises:
            ValueError: If list does not contain exactly 12 values.
        """
        if len(monthly_cooling_kwh_th) != 12:
            raise ValueError(
                f"Expected 12 monthly values, got {len(monthly_cooling_kwh_th)}"
            )

        monthly_results: List[Dict[str, str]] = []
        annual_emissions = _ZERO
        annual_cooling = _ZERO
        month_names = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December",
        ]

        for idx, monthly_kwh in enumerate(monthly_cooling_kwh_th):
            result = self.calculate_district_network(
                cooling_kwh_th=monthly_kwh,
                region=region,
                distribution_loss_pct=distribution_loss_pct,
                grid_ef=grid_ef,
                gwp_source=gwp_source,
            )
            annual_emissions = _q(annual_emissions + result.emissions_kgco2e)
            annual_cooling = _q(annual_cooling + monthly_kwh)
            monthly_results.append({
                "month": month_names[idx],
                "cooling_kwh_th": str(monthly_kwh),
                "emissions_kgco2e": str(result.emissions_kgco2e),
                "calculation_id": result.calculation_id,
            })

        annual_emissions_tonnes = _q(annual_emissions / _THOUSAND)

        # Emission intensity
        emission_intensity = _ZERO
        if annual_cooling > _ZERO:
            emission_intensity = _q(annual_emissions / annual_cooling)

        return {
            "region": region,
            "annual_cooling_kwh_th": str(annual_cooling),
            "annual_emissions_kgco2e": str(annual_emissions),
            "annual_emissions_tonnes_co2e": str(annual_emissions_tonnes),
            "emission_intensity_kgco2e_per_kwh_th": str(emission_intensity),
            "monthly_results": monthly_results,
        }


# ===========================================================================
# Module-level convenience functions
# ===========================================================================


_engine_instance: Optional[DistrictCoolingCalculatorEngine] = None
_engine_lock = threading.Lock()


def get_district_cooling_calculator() -> DistrictCoolingCalculatorEngine:
    """Return the singleton DistrictCoolingCalculatorEngine instance.

    Thread-safe convenience function for obtaining the engine without
    calling the constructor directly.

    Returns:
        The singleton DistrictCoolingCalculatorEngine instance.

    Example:
        >>> engine = get_district_cooling_calculator()
        >>> result = engine.calculate_district_network(
        ...     cooling_kwh_th=Decimal("100000"),
        ...     region="singapore",
        ... )
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = DistrictCoolingCalculatorEngine()
    return _engine_instance


def reset_district_cooling_calculator() -> None:
    """Reset the module-level engine singleton for testing.

    After calling this function, the next call to
    ``get_district_cooling_calculator()`` will create a fresh instance.
    Also resets the class-level singleton.
    """
    global _engine_instance
    _engine_instance = None
    DistrictCoolingCalculatorEngine.reset()
    logger.debug("District cooling calculator module-level singleton reset")
