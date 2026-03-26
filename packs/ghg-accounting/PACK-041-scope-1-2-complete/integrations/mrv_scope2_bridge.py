# -*- coding: utf-8 -*-
"""
MRVScope2Bridge - Bridge to All 5 Scope 2 MRV Agents for PACK-041
=====================================================================

This module routes electricity, steam, heat, and cooling consumption
data to the appropriate Scope 2 MRV agents (MRV-009 through MRV-013)
for dual-reporting emissions calculations (location-based and
market-based per GHG Protocol Scope 2 Guidance).

Routing Table:
    Location-based electricity  --> MRV-009 (gl_scope2_location_)
    Market-based electricity    --> MRV-010 (gl_scope2_market_)
    Purchased steam/heat        --> MRV-011 (gl_scope2_steam_)
    Purchased cooling           --> MRV-012 (gl_scope2_cooling_)
    Dual reporting reconcile    --> MRV-013 (gl_scope2_dual_reporting_)

Key Formulas (deterministic, zero-hallucination):
    location_tco2e = consumption_kwh * grid_ef_kgco2_per_kwh / 1000
    market_tco2e = consumption_kwh * contractual_ef / 1000
    residual_tco2e = (total_kwh - REC_kwh - PPA_kwh) * residual_mix_ef / 1000

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-041 Scope 1-2 Complete
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Scope2Method(str, Enum):
    """Scope 2 accounting methods per GHG Protocol Scope 2 Guidance."""

    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"


class EnergyType(str, Enum):
    """Types of purchased energy for Scope 2."""

    ELECTRICITY = "electricity"
    STEAM = "steam"
    HEAT = "heat"
    COOLING = "cooling"


class MarketInstrument(str, Enum):
    """Market-based contractual instruments."""

    ENERGY_ATTRIBUTE_CERTIFICATE = "energy_attribute_certificate"
    REC = "rec"
    GO = "guarantee_of_origin"
    PPA = "power_purchase_agreement"
    GREEN_TARIFF = "green_tariff"
    SUPPLIER_SPECIFIC = "supplier_specific"
    RESIDUAL_MIX = "residual_mix"


class ReconciliationStatus(str, Enum):
    """Dual reporting reconciliation status."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


# ---------------------------------------------------------------------------
# Grid Emission Factors (kg CO2e per kWh) by region
# ---------------------------------------------------------------------------

GRID_EMISSION_FACTORS: Dict[str, float] = {
    "US_AVERAGE": 0.417,
    "ERCOT": 0.395,
    "PJM": 0.440,
    "MISO": 0.510,
    "CAISO": 0.225,
    "NYISO": 0.280,
    "ISO_NE": 0.305,
    "SPP": 0.475,
    "WECC_NW": 0.310,
    "WECC_SW": 0.420,
    "EU_AVERAGE": 0.276,
    "UK_GRID": 0.233,
    "DE_GRID": 0.385,
    "FR_GRID": 0.052,
    "AU_GRID": 0.680,
    "JP_GRID": 0.470,
    "IN_GRID": 0.708,
    "CN_GRID": 0.581,
    "BR_GRID": 0.074,
}

RESIDUAL_MIX_FACTORS: Dict[str, float] = {
    "US_AVERAGE": 0.450,
    "EU_AVERAGE": 0.380,
    "UK_RESIDUAL": 0.310,
    "DE_RESIDUAL": 0.460,
}

STEAM_EMISSION_FACTORS: Dict[str, float] = {
    "natural_gas_boiler": 0.0665,
    "coal_boiler": 0.0955,
    "district_heating": 0.0750,
    "waste_heat_recovery": 0.0200,
}

COOLING_EMISSION_FACTORS: Dict[str, float] = {
    "electric_chiller": 0.180,
    "absorption_chiller": 0.085,
    "district_cooling": 0.120,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class Scope2AgentConfig(BaseModel):
    """Configuration for Scope 2 agent routing."""

    config_id: str = Field(default_factory=_new_uuid)
    default_grid_region: str = Field(default="US_AVERAGE")
    default_residual_mix_region: str = Field(default="US_AVERAGE")
    require_dual_reporting: bool = Field(default=True)
    rec_retirement_tracking: bool = Field(default=True)


class Scope2Result(BaseModel):
    """Result from a Scope 2 agent execution."""

    result_id: str = Field(default_factory=_new_uuid)
    agent_id: str = Field(default="")
    method: str = Field(default="")
    energy_type: str = Field(default="electricity")
    total_consumption_kwh: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    emission_factor_used: float = Field(default=0.0)
    emission_factor_source: str = Field(default="")
    grid_region: str = Field(default="US_AVERAGE")
    facilities_count: int = Field(default=0)
    records_processed: int = Field(default=0)
    status: str = Field(default="success")
    error_message: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=_utcnow)


class Scope2DualResult(BaseModel):
    """Combined dual-reporting Scope 2 result."""

    result_id: str = Field(default_factory=_new_uuid)
    location_based_tco2e: float = Field(default=0.0)
    market_based_tco2e: float = Field(default=0.0)
    electricity_location_tco2e: float = Field(default=0.0)
    electricity_market_tco2e: float = Field(default=0.0)
    steam_tco2e: float = Field(default=0.0)
    cooling_tco2e: float = Field(default=0.0)
    total_consumption_kwh: float = Field(default=0.0)
    rec_certificates_mwh: float = Field(default=0.0)
    ppa_contracts_mwh: float = Field(default=0.0)
    reconciliation_status: str = Field(default="PASS")
    reconciliation_notes: List[str] = Field(default_factory=list)
    by_facility: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# MRVScope2Bridge
# ---------------------------------------------------------------------------


class MRVScope2Bridge:
    """Bridge to all 5 Scope 2 MRV agents (MRV-009 through MRV-013).

    Routes electricity, steam, heat, and cooling consumption data to the
    appropriate Scope 2 MRV agents for dual-reporting calculations
    (location-based and market-based per GHG Protocol Scope 2 Guidance).

    Attributes:
        config: Agent routing configuration.

    Example:
        >>> bridge = MRVScope2Bridge()
        >>> dual = bridge.execute_full_scope2(all_data)
        >>> assert dual.reconciliation_status == "PASS"
    """

    def __init__(
        self,
        config: Optional[Scope2AgentConfig] = None,
    ) -> None:
        """Initialize MRVScope2Bridge.

        Args:
            config: Agent routing configuration. Uses defaults if None.
        """
        self.config = config or Scope2AgentConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(
            "MRVScope2Bridge initialized: grid=%s, dual_reporting=%s",
            self.config.default_grid_region,
            self.config.require_dual_reporting,
        )

    # -------------------------------------------------------------------------
    # Location-Based
    # -------------------------------------------------------------------------

    def execute_location_based(
        self,
        consumption_data: Dict[str, Any],
        grid_factors: Optional[Dict[str, float]] = None,
    ) -> Scope2Result:
        """Execute location-based Scope 2 calculation via MRV-009.

        Formula: emissions = consumption_kwh * grid_ef / 1000

        Args:
            consumption_data: Dict with consumption_records list, each
                containing facility_id, consumption_kwh, grid_region.
            grid_factors: Override grid emission factors. Uses defaults if None.

        Returns:
            Scope2Result with location-based emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-009"
        factors = grid_factors or GRID_EMISSION_FACTORS

        records = consumption_data.get("consumption_records", [])
        total_kwh = Decimal("0")
        total_emissions = Decimal("0")
        facilities: set = set()
        by_facility: Dict[str, float] = {}

        for record in records:
            facility_id = record.get("facility_id", "default")
            kwh = Decimal(str(record.get("consumption_kwh", 0)))
            region = record.get("grid_region", self.config.default_grid_region)
            ef = Decimal(str(factors.get(region, factors.get("US_AVERAGE", 0.417))))
            facilities.add(facility_id)

            emissions = kwh * ef / Decimal("1000")
            total_kwh += kwh
            total_emissions += emissions

            by_facility[facility_id] = by_facility.get(facility_id, 0.0) + float(
                emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = Scope2Result(
            agent_id=agent_id,
            method=Scope2Method.LOCATION_BASED.value,
            energy_type=EnergyType.ELECTRICITY.value,
            total_consumption_kwh=float(total_kwh),
            total_emissions_tco2e=float(
                total_emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            ),
            emission_factor_source="grid_average",
            grid_region=self.config.default_grid_region,
            facilities_count=len(facilities),
            records_processed=len(records),
            details={"by_facility": by_facility},
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-009 location-based: %.3f tCO2e from %.0f kWh, %d facilities",
            result.total_emissions_tco2e, float(total_kwh), len(facilities),
        )
        return result

    # -------------------------------------------------------------------------
    # Market-Based
    # -------------------------------------------------------------------------

    def execute_market_based(
        self,
        consumption_data: Dict[str, Any],
        instruments: Optional[List[Dict[str, Any]]] = None,
    ) -> Scope2Result:
        """Execute market-based Scope 2 calculation via MRV-010.

        Applies contractual instruments (RECs, PPAs, GOs, supplier-specific)
        to reduce market-based emissions, with residual mix for uncovered
        consumption.

        Args:
            consumption_data: Dict with consumption_records list.
            instruments: List of contractual instruments with type, mwh, ef.

        Returns:
            Scope2Result with market-based emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-010"

        records = consumption_data.get("consumption_records", [])
        instrument_list = instruments or []

        total_kwh = Decimal("0")
        for record in records:
            total_kwh += Decimal(str(record.get("consumption_kwh", 0)))

        total_mwh = total_kwh / Decimal("1000")

        # Calculate instrument-covered MWh
        covered_mwh = Decimal("0")
        instrument_emissions = Decimal("0")
        rec_mwh = Decimal("0")
        ppa_mwh = Decimal("0")

        for inst in instrument_list:
            inst_type = inst.get("type", "rec")
            inst_mwh = Decimal(str(inst.get("mwh", 0)))
            inst_ef = Decimal(str(inst.get("emission_factor_kgco2_per_kwh", 0)))
            covered_mwh += inst_mwh
            instrument_emissions += inst_mwh * Decimal("1000") * inst_ef / Decimal("1000")

            if inst_type in ("rec", "energy_attribute_certificate", "guarantee_of_origin"):
                rec_mwh += inst_mwh
            elif inst_type in ("ppa", "power_purchase_agreement"):
                ppa_mwh += inst_mwh

        # Residual mix for uncovered
        residual_mwh = max(Decimal("0"), total_mwh - covered_mwh)
        residual_ef = Decimal(str(RESIDUAL_MIX_FACTORS.get(
            self.config.default_residual_mix_region, 0.450
        )))
        residual_emissions = residual_mwh * Decimal("1000") * residual_ef / Decimal("1000")

        total_emissions = instrument_emissions + residual_emissions
        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = Scope2Result(
            agent_id=agent_id,
            method=Scope2Method.MARKET_BASED.value,
            energy_type=EnergyType.ELECTRICITY.value,
            total_consumption_kwh=float(total_kwh),
            total_emissions_tco2e=float(
                total_emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            ),
            emission_factor_source="contractual_instruments+residual_mix",
            grid_region=self.config.default_grid_region,
            records_processed=len(records),
            details={
                "instruments_count": len(instrument_list),
                "covered_mwh": float(covered_mwh),
                "residual_mwh": float(residual_mwh),
                "rec_mwh": float(rec_mwh),
                "ppa_mwh": float(ppa_mwh),
                "instrument_emissions_tco2e": float(instrument_emissions.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )),
                "residual_emissions_tco2e": float(residual_emissions.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )),
            },
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-010 market-based: %.3f tCO2e (%.0f MWh covered, %.0f MWh residual)",
            result.total_emissions_tco2e, float(covered_mwh), float(residual_mwh),
        )
        return result

    # -------------------------------------------------------------------------
    # Steam / Heat
    # -------------------------------------------------------------------------

    def execute_steam_heat(
        self,
        purchase_data: Dict[str, Any],
    ) -> Scope2Result:
        """Execute purchased steam/heat emissions calculation via MRV-011.

        Args:
            purchase_data: Dict with steam_records list, each containing
                facility_id, consumption_kwh, source_type.

        Returns:
            Scope2Result with steam/heat emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-011"

        records = purchase_data.get("steam_records", [])
        total_kwh = Decimal("0")
        total_emissions = Decimal("0")

        for record in records:
            kwh = Decimal(str(record.get("consumption_kwh", 0)))
            source = record.get("source_type", "natural_gas_boiler")
            ef = Decimal(str(STEAM_EMISSION_FACTORS.get(source, 0.0665)))
            emissions = kwh * ef / Decimal("1000")
            total_kwh += kwh
            total_emissions += emissions

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = Scope2Result(
            agent_id=agent_id,
            method="purchased_steam_heat",
            energy_type=EnergyType.STEAM.value,
            total_consumption_kwh=float(total_kwh),
            total_emissions_tco2e=float(
                total_emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            ),
            emission_factor_source="steam_emission_factors",
            records_processed=len(records),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-011 steam/heat: %.3f tCO2e from %.0f kWh",
            result.total_emissions_tco2e, float(total_kwh),
        )
        return result

    # -------------------------------------------------------------------------
    # Cooling
    # -------------------------------------------------------------------------

    def execute_cooling(
        self,
        purchase_data: Dict[str, Any],
    ) -> Scope2Result:
        """Execute purchased cooling emissions calculation via MRV-012.

        Args:
            purchase_data: Dict with cooling_records list, each containing
                facility_id, consumption_kwh, source_type.

        Returns:
            Scope2Result with cooling emissions.
        """
        start_time = time.monotonic()
        agent_id = "MRV-012"

        records = purchase_data.get("cooling_records", [])
        total_kwh = Decimal("0")
        total_emissions = Decimal("0")

        for record in records:
            kwh = Decimal(str(record.get("consumption_kwh", 0)))
            source = record.get("source_type", "electric_chiller")
            ef = Decimal(str(COOLING_EMISSION_FACTORS.get(source, 0.180)))
            emissions = kwh * ef / Decimal("1000")
            total_kwh += kwh
            total_emissions += emissions

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = Scope2Result(
            agent_id=agent_id,
            method="purchased_cooling",
            energy_type=EnergyType.COOLING.value,
            total_consumption_kwh=float(total_kwh),
            total_emissions_tco2e=float(
                total_emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            ),
            emission_factor_source="cooling_emission_factors",
            records_processed=len(records),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "MRV-012 cooling: %.3f tCO2e from %.0f kWh",
            result.total_emissions_tco2e, float(total_kwh),
        )
        return result

    # -------------------------------------------------------------------------
    # Dual Reconciliation
    # -------------------------------------------------------------------------

    def execute_dual_reconciliation(
        self,
        location_result: Scope2Result,
        market_result: Scope2Result,
    ) -> Dict[str, Any]:
        """Execute dual reporting reconciliation via MRV-013.

        Validates consistency between location-based and market-based
        totals and checks for common errors.

        Args:
            location_result: Location-based Scope 2 result.
            market_result: Market-based Scope 2 result.

        Returns:
            Dict with reconciliation status and notes.
        """
        start_time = time.monotonic()
        notes: List[str] = []
        status = ReconciliationStatus.PASS

        # Check consumption consistency
        loc_kwh = location_result.total_consumption_kwh
        mkt_kwh = market_result.total_consumption_kwh
        if loc_kwh > 0 and abs(loc_kwh - mkt_kwh) / loc_kwh > 0.01:
            notes.append(
                f"Consumption mismatch: location={loc_kwh:.0f} kWh vs "
                f"market={mkt_kwh:.0f} kWh"
            )
            status = ReconciliationStatus.WARNING

        # Market should typically be <= location (instruments reduce it)
        if market_result.total_emissions_tco2e > location_result.total_emissions_tco2e * 1.05:
            notes.append(
                "Market-based emissions exceed location-based by >5%. "
                "Verify contractual instruments."
            )
            status = ReconciliationStatus.WARNING

        if not notes:
            notes.append("Dual reporting reconciliation passed all checks")

        elapsed_ms = (time.monotonic() - start_time) * 1000

        reconciliation = {
            "status": status.value,
            "location_tco2e": location_result.total_emissions_tco2e,
            "market_tco2e": market_result.total_emissions_tco2e,
            "difference_tco2e": round(
                location_result.total_emissions_tco2e - market_result.total_emissions_tco2e, 3
            ),
            "notes": notes,
            "processing_time_ms": elapsed_ms,
            "provenance_hash": _compute_hash({
                "location": location_result.total_emissions_tco2e,
                "market": market_result.total_emissions_tco2e,
            }),
        }

        self.logger.info(
            "MRV-013 reconciliation: %s (loc=%.3f, mkt=%.3f, diff=%.3f)",
            status.value,
            location_result.total_emissions_tco2e,
            market_result.total_emissions_tco2e,
            reconciliation["difference_tco2e"],
        )
        return reconciliation

    # -------------------------------------------------------------------------
    # Full Scope 2
    # -------------------------------------------------------------------------

    def execute_full_scope2(
        self,
        all_data: Dict[str, Any],
    ) -> Scope2DualResult:
        """Execute complete Scope 2 dual-reporting calculation.

        Runs location-based, market-based, steam/heat, cooling, and
        reconciliation in sequence.

        Args:
            all_data: Dict containing electricity_data, instruments,
                steam_data, cooling_data keys.

        Returns:
            Scope2DualResult with complete dual-reporting output.
        """
        start_time = time.monotonic()

        electricity_data = all_data.get("electricity_data", {})
        instruments = all_data.get("instruments", [])
        steam_data = all_data.get("steam_data", {})
        cooling_data = all_data.get("cooling_data", {})

        # Location-based
        location_result = self.execute_location_based(electricity_data)

        # Market-based
        market_result = self.execute_market_based(electricity_data, instruments)

        # Steam/Heat
        steam_result = self.execute_steam_heat(steam_data)

        # Cooling
        cooling_result = self.execute_cooling(cooling_data)

        # Reconciliation
        reconciliation = self.execute_dual_reconciliation(location_result, market_result)

        # Aggregate
        location_total = float(
            Decimal(str(location_result.total_emissions_tco2e))
            + Decimal(str(steam_result.total_emissions_tco2e))
            + Decimal(str(cooling_result.total_emissions_tco2e))
        )
        market_total = float(
            Decimal(str(market_result.total_emissions_tco2e))
            + Decimal(str(steam_result.total_emissions_tco2e))
            + Decimal(str(cooling_result.total_emissions_tco2e))
        )

        rec_mwh = market_result.details.get("rec_mwh", 0.0)
        ppa_mwh = market_result.details.get("ppa_mwh", 0.0)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        dual_result = Scope2DualResult(
            location_based_tco2e=round(location_total, 3),
            market_based_tco2e=round(market_total, 3),
            electricity_location_tco2e=location_result.total_emissions_tco2e,
            electricity_market_tco2e=market_result.total_emissions_tco2e,
            steam_tco2e=steam_result.total_emissions_tco2e,
            cooling_tco2e=cooling_result.total_emissions_tco2e,
            total_consumption_kwh=location_result.total_consumption_kwh,
            rec_certificates_mwh=rec_mwh,
            ppa_contracts_mwh=ppa_mwh,
            reconciliation_status=reconciliation["status"],
            reconciliation_notes=reconciliation["notes"],
            processing_time_ms=elapsed_ms,
        )
        dual_result.provenance_hash = _compute_hash(dual_result)

        self.logger.info(
            "Full Scope 2: location=%.3f tCO2e, market=%.3f tCO2e, "
            "steam=%.3f, cooling=%.3f, reconciliation=%s",
            location_total, market_total,
            steam_result.total_emissions_tco2e,
            cooling_result.total_emissions_tco2e,
            reconciliation["status"],
        )
        return dual_result
