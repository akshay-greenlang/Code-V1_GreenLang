# -*- coding: utf-8 -*-
"""
UpstreamLeasedPipelineEngine - Orchestrated 10-stage pipeline for upstream leased assets.

This module implements the UpstreamLeasedPipelineEngine for AGENT-MRV-021
(Upstream Leased Assets, GHG Protocol Scope 3 Category 8). It orchestrates a 10-stage
pipeline from raw input to compliant output with full audit trail, covering buildings,
vehicles, equipment, and IT assets that are leased by the reporting company.

The 10 stages are:
1. VALIDATE: Asset inventory, lease types, floor areas > 0, valid asset categories
2. CLASSIFY: Lease type (operating/finance), asset category, energy sources
3. NORMALIZE: Unit conversions (sqft->sqm, therms->kWh, gallons->litres, currency->USD)
4. RESOLVE_EFS: Grid EFs, fuel EFs, EUI benchmarks, EEIO factors per asset
5. CALCULATE: Emissions per asset using appropriate engine (building/vehicle/equipment/IT)
6. ALLOCATE: Multi-tenant allocation factors and partial-year proration
7. AGGREGATE: By asset_type, building_type, country, energy_source. Hot-spot analysis
8. COMPLIANCE: Run all 7 frameworks + 10 DC rules
9. PROVENANCE: SHA-256 chain, record each stage
10. SEAL: Merkle root, final hash, timestamp, immutable

Calculation Methods:
- Asset-specific: Metered energy data per leased asset
- Lessor-specific: Primary data from lessor/landlord
- Average-data: Benchmark EUI/energy intensity by asset type
- Spend-based: EEIO factors with CPI deflation

Key Formulas:
  Building asset-specific: CO2e = (elec*grid_ef + gas*gas_ef + heat*dh_ef + cool*dc_ef) * alloc * (months/12)
  Building average:        CO2e = area * EUI * grid_ef * alloc * (months/12)
  Vehicle:                 CO2e = km * count * ef_per_km (+ WTT)
  Equipment:               CO2e = power * hours * load_factor * count * grid_ef
  IT:                      CO2e = power * pue * hours * util * count * grid_ef
  Spend:                   CO2e = spend_usd_deflated * eeio_factor
  Lessor:                  CO2e = lessor_co2e * alloc * (months/12)

Example:
    >>> from greenlang.agents.mrv.upstream_leased_assets.upstream_leased_pipeline import (
    ...     UpstreamLeasedPipelineEngine,
    ... )
    >>> engine = UpstreamLeasedPipelineEngine()
    >>> result = engine.calculate({"assets": [...], "reporting_period": "2025"})
    >>> print(f"Total: {result['total_co2e']} kgCO2e")

Module: greenlang.agents.mrv.upstream_leased_assets.upstream_leased_pipeline
Agent: AGENT-MRV-021
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import logging
import hashlib
import json
from threading import RLock

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "upstream_leased_pipeline_engine"
ENGINE_VERSION: str = "1.0.0"

_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_TWELVE = Decimal("12")
_HUNDRED = Decimal("100")

# Unit conversion constants
_SQFT_TO_SQM = Decimal("0.09290304")
_THERMS_TO_KWH = Decimal("29.3001")
_MMBTU_TO_KWH = Decimal("293.07107")
_GALLONS_TO_LITRES = Decimal("3.78541")
_MILES_TO_KM = Decimal("1.60934")

# CPI deflators (base year 2024 = 1.000)
CPI_DEFLATORS: Dict[int, Decimal] = {
    2020: Decimal("0.8835"),
    2021: Decimal("0.9247"),
    2022: Decimal("0.9635"),
    2023: Decimal("0.9821"),
    2024: Decimal("1.0000"),
    2025: Decimal("1.0180"),
    2026: Decimal("1.0360"),
}

# Currency exchange rates to USD (approximate 2024 annual avg)
CURRENCY_RATES: Dict[str, Decimal] = {
    "USD": Decimal("1.0000"),
    "EUR": Decimal("1.0850"),
    "GBP": Decimal("1.2650"),
    "CAD": Decimal("0.7420"),
    "AUD": Decimal("0.6530"),
    "JPY": Decimal("0.006667"),
    "CNY": Decimal("0.1389"),
    "INR": Decimal("0.01199"),
    "CHF": Decimal("1.1320"),
    "SGD": Decimal("0.7440"),
    "BRL": Decimal("0.2000"),
    "ZAR": Decimal("0.0556"),
}

# Grid emission factors (kgCO2e/kWh) by country - IEA 2024
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.3937"),
    "GB": Decimal("0.2121"),
    "DE": Decimal("0.3850"),
    "FR": Decimal("0.0564"),
    "JP": Decimal("0.4700"),
    "CA": Decimal("0.1200"),
    "AU": Decimal("0.6700"),
    "IN": Decimal("0.7080"),
    "CN": Decimal("0.5550"),
    "BR": Decimal("0.0740"),
    "GLOBAL": Decimal("0.4360"),
}

# Natural gas emission factor (kgCO2e/kWh) - DEFRA 2024
GAS_EMISSION_FACTOR: Decimal = Decimal("0.18293")
GAS_WTT_FACTOR: Decimal = Decimal("0.02533")

# District heating emission factor (kgCO2e/kWh)
DISTRICT_HEATING_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.1700"),
    "GB": Decimal("0.1510"),
    "DE": Decimal("0.1660"),
    "FR": Decimal("0.1280"),
    "GLOBAL": Decimal("0.1600"),
}

# District cooling emission factor (kgCO2e/kWh)
DISTRICT_COOLING_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.1400"),
    "GB": Decimal("0.1200"),
    "GLOBAL": Decimal("0.1300"),
}

# EUI benchmarks (kWh/sqm/year) by building type and climate zone
# Zone 1=Hot, 2=Warm, 3=Mixed, 4=Cool, 5=Cold
EUI_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    "office": {
        "1": Decimal("280"), "2": Decimal("250"), "3": Decimal("220"),
        "4": Decimal("200"), "5": Decimal("230"),
    },
    "retail": {
        "1": Decimal("320"), "2": Decimal("290"), "3": Decimal("260"),
        "4": Decimal("240"), "5": Decimal("270"),
    },
    "warehouse": {
        "1": Decimal("120"), "2": Decimal("100"), "3": Decimal("90"),
        "4": Decimal("80"), "5": Decimal("95"),
    },
    "industrial": {
        "1": Decimal("350"), "2": Decimal("310"), "3": Decimal("280"),
        "4": Decimal("260"), "5": Decimal("290"),
    },
    "data_center": {
        "1": Decimal("1800"), "2": Decimal("1700"), "3": Decimal("1600"),
        "4": Decimal("1500"), "5": Decimal("1550"),
    },
    "hotel": {
        "1": Decimal("340"), "2": Decimal("300"), "3": Decimal("270"),
        "4": Decimal("250"), "5": Decimal("280"),
    },
    "healthcare": {
        "1": Decimal("420"), "2": Decimal("380"), "3": Decimal("350"),
        "4": Decimal("320"), "5": Decimal("360"),
    },
    "education": {
        "1": Decimal("240"), "2": Decimal("210"), "3": Decimal("190"),
        "4": Decimal("170"), "5": Decimal("200"),
    },
}

# Vehicle emission factors (kgCO2e per km) - DEFRA 2024
VEHICLE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "small_car": {"ef_per_km": Decimal("0.13890"), "wtt_per_km": Decimal("0.03290")},
    "medium_car": {"ef_per_km": Decimal("0.17130"), "wtt_per_km": Decimal("0.04074")},
    "large_car": {"ef_per_km": Decimal("0.22740"), "wtt_per_km": Decimal("0.05404")},
    "suv": {"ef_per_km": Decimal("0.21230"), "wtt_per_km": Decimal("0.05045")},
    "light_van": {"ef_per_km": Decimal("0.24928"), "wtt_per_km": Decimal("0.05925")},
    "heavy_van": {"ef_per_km": Decimal("0.31024"), "wtt_per_km": Decimal("0.07374")},
    "light_truck": {"ef_per_km": Decimal("0.47123"), "wtt_per_km": Decimal("0.11200")},
    "heavy_truck": {"ef_per_km": Decimal("0.88461"), "wtt_per_km": Decimal("0.21024")},
}

# Equipment load factors by type
EQUIPMENT_LOAD_FACTORS: Dict[str, Decimal] = {
    "manufacturing": Decimal("0.65"),
    "construction": Decimal("0.55"),
    "generator": Decimal("0.70"),
    "agricultural": Decimal("0.50"),
    "mining": Decimal("0.60"),
    "hvac": Decimal("0.45"),
}

# IT PUE (Power Usage Effectiveness) defaults
IT_PUE_DEFAULTS: Dict[str, Decimal] = {
    "server": Decimal("1.58"),
    "network_switch": Decimal("1.58"),
    "storage": Decimal("1.58"),
    "desktop": Decimal("1.00"),
    "laptop": Decimal("1.00"),
    "printer": Decimal("1.00"),
    "copier": Decimal("1.00"),
}

# IT asset power consumption (kW) defaults
IT_POWER_DEFAULTS: Dict[str, Decimal] = {
    "server": Decimal("0.500"),
    "network_switch": Decimal("0.150"),
    "storage": Decimal("0.250"),
    "desktop": Decimal("0.100"),
    "laptop": Decimal("0.050"),
    "printer": Decimal("0.200"),
    "copier": Decimal("0.500"),
}

# IT utilization defaults
IT_UTILIZATION_DEFAULTS: Dict[str, Decimal] = {
    "server": Decimal("0.40"),
    "network_switch": Decimal("0.30"),
    "storage": Decimal("0.50"),
    "desktop": Decimal("0.25"),
    "laptop": Decimal("0.20"),
    "printer": Decimal("0.05"),
    "copier": Decimal("0.05"),
}

# EEIO spend-based factors (kgCO2e per USD) by sector
EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "531110": {"name": "Lessors of residential buildings", "ef": Decimal("0.20100")},
    "531120": {"name": "Lessors of nonresidential buildings", "ef": Decimal("0.22500")},
    "531130": {"name": "Lessors of miniwarehouses/storage", "ef": Decimal("0.18700")},
    "532100": {"name": "Automotive equipment rental", "ef": Decimal("0.31200")},
    "532200": {"name": "Consumer goods rental", "ef": Decimal("0.28500")},
    "532400": {"name": "Commercial equipment rental", "ef": Decimal("0.35100")},
    "532300": {"name": "General rental centers", "ef": Decimal("0.30000")},
    "511210": {"name": "Software / IT leasing", "ef": Decimal("0.15800")},
    "334100": {"name": "Computer and peripheral equipment", "ef": Decimal("0.25600")},
}

# Valid asset categories
VALID_ASSET_CATEGORIES = frozenset({
    "building", "vehicle", "equipment", "it_asset",
})

VALID_BUILDING_TYPES = frozenset({
    "office", "retail", "warehouse", "industrial",
    "data_center", "hotel", "healthcare", "education",
})

VALID_VEHICLE_TYPES = frozenset({
    "small_car", "medium_car", "large_car", "suv",
    "light_van", "heavy_van", "light_truck", "heavy_truck",
})

VALID_EQUIPMENT_TYPES = frozenset({
    "manufacturing", "construction", "generator",
    "agricultural", "mining", "hvac",
})

VALID_IT_TYPES = frozenset({
    "server", "network_switch", "storage",
    "desktop", "laptop", "printer", "copier",
})

OPERATING_LEASE_TYPES = frozenset({
    "operating_lease", "operating", "short_term", "low_value",
})

FINANCE_LEASE_TYPES = frozenset({
    "finance_lease", "capital_lease", "finance", "capital",
    "ifrs16_right_of_use",
})


# ==============================================================================
# PIPELINE STATUS
# ==============================================================================


class PipelineStatus:
    """Pipeline execution status constants."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


# ==============================================================================
# UPSTREAM LEASED PIPELINE ENGINE
# ==============================================================================


class UpstreamLeasedPipelineEngine:
    """
    UpstreamLeasedPipelineEngine - Orchestrated 10-stage pipeline.

    This engine coordinates the complete upstream leased assets emissions
    calculation workflow through 10 sequential stages, from input validation
    to sealed audit trail. It supports buildings (8 types x 5 zones),
    vehicles (8 types), equipment (6 types), and IT assets (7 types) plus
    average-data and spend-based fallback methods.

    Thread Safety:
        Singleton with RLock. All mutable state is protected.

    Example:
        >>> engine = UpstreamLeasedPipelineEngine()
        >>> result = engine.calculate({
        ...     "assets": [{
        ...         "asset_id": "BLD-001",
        ...         "asset_category": "building",
        ...         "building_type": "office",
        ...         "lease_type": "operating",
        ...         "area_sqm": "5000",
        ...         "country": "US",
        ...         "electricity_kwh": "120000",
        ...         "natural_gas_kwh": "50000",
        ...     }],
        ...     "reporting_period": "2025",
        ... })
        >>> assert result["status"] == PipelineStatus.SUCCESS
    """

    _instance: Optional["UpstreamLeasedPipelineEngine"] = None
    _lock: RLock = RLock()

    def __new__(cls) -> "UpstreamLeasedPipelineEngine":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize UpstreamLeasedPipelineEngine."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._compliance_engine: Optional[Any] = None
        self._provenance_chains: Dict[str, List[Dict[str, Any]]] = {}
        self._calculations_performed: int = 0
        self._initialized = True
        logger.info(
            "UpstreamLeasedPipelineEngine initialized (version %s)",
            ENGINE_VERSION,
        )

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (testing only)."""
        with cls._lock:
            cls._instance = None
            logger.info("UpstreamLeasedPipelineEngine singleton reset")

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def calculate(self, input_data: dict) -> dict:
        """
        Execute the 10-stage upstream leased assets emissions pipeline.

        Args:
            input_data: Dictionary containing:
                - assets: List of asset dicts with energy/lease data
                - reporting_period: Reporting period label
                - enable_compliance: (optional) bool, default True

        Returns:
            Dictionary with pipeline results.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If pipeline execution fails.
        """
        chain_id = f"ula-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._provenance_chains[chain_id] = []
        stage_durations: Dict[str, float] = {}

        try:
            # Stage 1: VALIDATE
            start = datetime.now(timezone.utc)
            validated = self._stage_validate(input_data)
            duration_ms = self._elapsed_ms(start)
            stage_durations["VALIDATE"] = duration_ms
            self._record_provenance(
                chain_id, "validate",
                input_data, {"valid": True, "asset_count": len(validated["assets"])}
            )
            logger.info(
                "[%s] Stage VALIDATE completed in %.2fms (%d assets)",
                chain_id, duration_ms, len(validated["assets"]),
            )

            # Stage 2: CLASSIFY
            start = datetime.now(timezone.utc)
            classified = self._stage_classify(validated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CLASSIFY"] = duration_ms
            self._record_provenance(
                chain_id, "classify",
                validated, {"classified_count": len(classified["assets"])}
            )
            logger.info("[%s] Stage CLASSIFY completed in %.2fms", chain_id, duration_ms)

            # Stage 3: NORMALIZE
            start = datetime.now(timezone.utc)
            normalized = self._stage_normalize(classified)
            duration_ms = self._elapsed_ms(start)
            stage_durations["NORMALIZE"] = duration_ms
            self._record_provenance(
                chain_id, "normalize",
                classified, {"normalized_count": len(normalized["assets"])}
            )
            logger.info("[%s] Stage NORMALIZE completed in %.2fms", chain_id, duration_ms)

            # Stage 4: RESOLVE_EFS
            start = datetime.now(timezone.utc)
            resolved = self._stage_resolve_efs(normalized)
            duration_ms = self._elapsed_ms(start)
            stage_durations["RESOLVE_EFS"] = duration_ms
            self._record_provenance(
                chain_id, "resolve_efs",
                normalized, {"ef_resolved_count": len(resolved["assets"])}
            )
            logger.info("[%s] Stage RESOLVE_EFS completed in %.2fms", chain_id, duration_ms)

            # Stage 5: CALCULATE
            start = datetime.now(timezone.utc)
            calculated = self._stage_calculate(resolved)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CALCULATE"] = duration_ms
            self._record_provenance(
                chain_id, "calculate",
                resolved, {"total_raw_co2e": str(calculated.get("total_raw_co2e", 0))}
            )
            logger.info(
                "[%s] Stage CALCULATE completed in %.2fms (raw_co2e=%s)",
                chain_id, duration_ms, calculated.get("total_raw_co2e", 0),
            )

            # Stage 6: ALLOCATE
            start = datetime.now(timezone.utc)
            allocated = self._stage_allocate(calculated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["ALLOCATE"] = duration_ms
            self._record_provenance(
                chain_id, "allocate",
                calculated, {"total_allocated_co2e": str(allocated.get("total_co2e", 0))}
            )
            logger.info(
                "[%s] Stage ALLOCATE completed in %.2fms (alloc_co2e=%s)",
                chain_id, duration_ms, allocated.get("total_co2e", 0),
            )

            # Stage 7: AGGREGATE
            start = datetime.now(timezone.utc)
            aggregated = self._stage_aggregate(allocated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["AGGREGATE"] = duration_ms
            self._record_provenance(
                chain_id, "aggregate",
                allocated, {"aggregation_keys": list(aggregated.get("by_asset_type", {}).keys())}
            )
            logger.info("[%s] Stage AGGREGATE completed in %.2fms", chain_id, duration_ms)

            # Stage 8: COMPLIANCE
            start = datetime.now(timezone.utc)
            enable_compliance = input_data.get("enable_compliance", True)
            if enable_compliance:
                compliance_result = self._stage_compliance(aggregated)
            else:
                compliance_result = {"overall_status": "skipped"}
            aggregated["compliance"] = compliance_result
            duration_ms = self._elapsed_ms(start)
            stage_durations["COMPLIANCE"] = duration_ms
            self._record_provenance(
                chain_id, "compliance",
                aggregated, {"compliance_status": compliance_result.get("overall_status", "N/A")}
            )
            logger.info(
                "[%s] Stage COMPLIANCE completed in %.2fms (status=%s)",
                chain_id, duration_ms, compliance_result.get("overall_status", "N/A"),
            )

            # Stage 9: PROVENANCE
            start = datetime.now(timezone.utc)
            provenance_data = self._stage_provenance(chain_id, aggregated)
            aggregated["provenance_chain"] = provenance_data
            duration_ms = self._elapsed_ms(start)
            stage_durations["PROVENANCE"] = duration_ms
            logger.info("[%s] Stage PROVENANCE completed in %.2fms", chain_id, duration_ms)

            # Stage 10: SEAL
            start = datetime.now(timezone.utc)
            sealed = self._stage_seal(chain_id, aggregated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["SEAL"] = duration_ms
            logger.info("[%s] Stage SEAL completed in %.2fms", chain_id, duration_ms)

            sealed["chain_id"] = chain_id
            sealed["status"] = PipelineStatus.SUCCESS
            sealed["stage_durations_ms"] = stage_durations
            sealed["total_pipeline_ms"] = sum(stage_durations.values())

            total_dur = sealed["total_pipeline_ms"]
            self._calculations_performed += 1
            logger.info(
                "[%s] Pipeline completed in %.2fms. Total: %s kgCO2e",
                chain_id, total_dur, sealed.get("total_co2e", 0),
            )
            return sealed

        except ValueError:
            raise
        except Exception as e:
            logger.error("[%s] Pipeline failed: %s", chain_id, e, exc_info=True)
            raise RuntimeError(f"Pipeline execution failed: {e}") from e
        finally:
            self._provenance_chains.pop(chain_id, None)

    def calculate_batch(self, inputs: list) -> dict:
        """
        Process a batch of independent pipeline inputs.

        Args:
            inputs: List of input dicts, each suitable for calculate().

        Returns:
            Dictionary with batch results.
        """
        start_time = datetime.now(timezone.utc)
        results: List[dict] = []
        errors: List[dict] = []

        logger.info("Starting batch calculation (%d inputs)", len(inputs))

        for idx, inp in enumerate(inputs):
            try:
                result = self.calculate(inp)
                results.append(result)
            except Exception as e:
                logger.error("Batch input %d failed: %s", idx, e)
                errors.append({"index": idx, "error": str(e)})

        total_co2e = sum(
            (self._safe_decimal(r.get("total_co2e", 0)) for r in results),
            _ZERO,
        )

        duration_ms = self._elapsed_ms(start_time)

        if not errors:
            batch_status = "completed"
        elif not results:
            batch_status = "failed"
        else:
            batch_status = "partial"

        batch_hash = self._hash_json({
            "results_count": len(results),
            "errors_count": len(errors),
            "total_co2e": str(total_co2e),
        })

        logger.info(
            "Batch completed in %.2fms. Success: %d, Failed: %d, Total: %s kgCO2e",
            duration_ms, len(results), len(errors), total_co2e,
        )

        return {
            "results": results,
            "total_co2e": str(total_co2e),
            "count": len(results),
            "errors": errors,
            "status": batch_status,
            "duration_ms": duration_ms,
            "provenance_hash": batch_hash,
        }

    def calculate_lessor_method(self, lessor_data: dict) -> dict:
        """
        Calculate emissions using lessor-specific data.

        Convenience method for when the lessor provides their total
        emissions and the lessee applies allocation and proration.

        Args:
            lessor_data: Dictionary containing:
                - assets: List with lessor_co2e, allocation_factor, months
                - reporting_period: Reporting period label

        Returns:
            Pipeline result dict.
        """
        for asset in lessor_data.get("assets", []):
            asset.setdefault("calculation_method", "lessor_specific")
        return self.calculate(lessor_data)

    def calculate_average_data(self, avg_data: dict) -> dict:
        """
        Calculate emissions using average-data (benchmark EUI) method.

        Args:
            avg_data: Dictionary containing:
                - assets: List with area_sqm, building_type, climate_zone
                - reporting_period: Reporting period label

        Returns:
            Pipeline result dict.
        """
        for asset in avg_data.get("assets", []):
            asset.setdefault("calculation_method", "average_data")
            asset.setdefault("asset_category", "building")
        return self.calculate(avg_data)

    def calculate_spend_based(self, spend_data: dict) -> dict:
        """
        Calculate emissions using EEIO spend-based method.

        Args:
            spend_data: Dictionary containing:
                - assets: List with spend_amount, currency, naics_code
                - reporting_period: Reporting period label

        Returns:
            Pipeline result dict.
        """
        for asset in spend_data.get("assets", []):
            asset.setdefault("calculation_method", "spend_based")
        return self.calculate(spend_data)

    # ==========================================================================
    # STAGE METHODS (PRIVATE)
    # ==========================================================================

    def _stage_validate(self, input_data: dict) -> dict:
        """
        Stage 1: VALIDATE - Input validation.

        Checks asset inventory, lease types, floor areas, categories.
        """
        errors: List[str] = []

        assets = input_data.get("assets")
        if not assets or not isinstance(assets, list):
            raise ValueError("'assets' must be a non-empty list")

        validated_assets: List[dict] = []

        for idx, asset in enumerate(assets):
            asset_errors: List[str] = []
            asset_id = asset.get("asset_id", f"ASSET-{idx}")

            # Asset category validation
            category = str(asset.get("asset_category", "building")).lower()
            if category not in VALID_ASSET_CATEGORIES:
                asset_errors.append(
                    f"Invalid asset_category '{category}'. "
                    f"Valid: {sorted(VALID_ASSET_CATEGORIES)}"
                )

            # Lease type validation
            lease_type = str(asset.get("lease_type", "operating")).lower().replace("-", "_").replace(" ", "_")
            calc_method = str(asset.get("calculation_method", "")).lower()

            # For spend-based, lease type is optional
            if calc_method != "spend_based":
                if lease_type not in OPERATING_LEASE_TYPES and lease_type not in FINANCE_LEASE_TYPES:
                    asset_errors.append(
                        f"Invalid lease_type '{lease_type}'. Use operating or finance."
                    )

            # Building-specific validation
            if category == "building":
                building_type = str(asset.get("building_type", "office")).lower()
                if building_type not in VALID_BUILDING_TYPES:
                    asset_errors.append(
                        f"Invalid building_type '{building_type}'. "
                        f"Valid: {sorted(VALID_BUILDING_TYPES)}"
                    )

                # Area validation (unless spend-based or lessor method)
                if calc_method not in ("spend_based", "lessor_specific"):
                    area = asset.get("area_sqm") or asset.get("area_sqft")
                    if area is not None:
                        try:
                            area_val = Decimal(str(area))
                            if area_val <= _ZERO:
                                asset_errors.append("Floor area must be > 0")
                        except (InvalidOperation, ValueError):
                            asset_errors.append(f"Invalid area value: {area}")
                    elif calc_method == "average_data":
                        asset_errors.append(
                            "area_sqm or area_sqft required for average-data method"
                        )

            # Vehicle-specific validation
            elif category == "vehicle":
                vehicle_type = str(asset.get("vehicle_type", "medium_car")).lower()
                if vehicle_type not in VALID_VEHICLE_TYPES:
                    asset_errors.append(
                        f"Invalid vehicle_type '{vehicle_type}'. "
                        f"Valid: {sorted(VALID_VEHICLE_TYPES)}"
                    )

            # Equipment-specific validation
            elif category == "equipment":
                equip_type = str(asset.get("equipment_type", "manufacturing")).lower()
                if equip_type not in VALID_EQUIPMENT_TYPES:
                    asset_errors.append(
                        f"Invalid equipment_type '{equip_type}'. "
                        f"Valid: {sorted(VALID_EQUIPMENT_TYPES)}"
                    )

            # IT asset validation
            elif category == "it_asset":
                it_type = str(asset.get("it_type", "server")).lower()
                if it_type not in VALID_IT_TYPES:
                    asset_errors.append(
                        f"Invalid it_type '{it_type}'. "
                        f"Valid: {sorted(VALID_IT_TYPES)}"
                    )

            if asset_errors:
                errors.extend([f"Asset {asset_id}: {e}" for e in asset_errors])
            else:
                validated = dict(asset)
                validated["asset_id"] = asset_id
                validated_assets.append(validated)

        if errors:
            raise ValueError(
                f"Validation failed ({len(errors)} errors): "
                + "; ".join(errors[:10])
                + ("..." if len(errors) > 10 else "")
            )

        if not validated_assets:
            raise ValueError("No valid assets after validation")

        result = dict(input_data)
        result["assets"] = validated_assets
        return result

    def _stage_classify(self, validated: dict) -> dict:
        """
        Stage 2: CLASSIFY - Classify lease type, asset category, energy sources.
        """
        classified_assets: List[dict] = []

        for asset in validated["assets"]:
            classified = dict(asset)

            # Parse category
            category = str(asset.get("asset_category", "building")).lower()
            classified["_category"] = category

            # Parse lease type
            lease_type = str(asset.get("lease_type", "operating")).lower().replace("-", "_").replace(" ", "_")
            if lease_type in FINANCE_LEASE_TYPES:
                classified["_lease_class"] = "finance"
                classified["_in_category_8"] = False
            else:
                classified["_lease_class"] = "operating"
                classified["_in_category_8"] = True

            # Parse calculation method
            calc_method = str(asset.get("calculation_method", "")).lower()
            if not calc_method:
                if asset.get("lessor_co2e") is not None:
                    calc_method = "lessor_specific"
                elif asset.get("spend_amount") is not None:
                    calc_method = "spend_based"
                elif category == "building" and (
                    asset.get("electricity_kwh") is not None
                    or asset.get("natural_gas_kwh") is not None
                ):
                    calc_method = "asset_specific"
                elif category == "building":
                    calc_method = "average_data"
                else:
                    calc_method = "asset_specific"
            classified["_calc_method"] = calc_method

            # Classify building subtype
            if category == "building":
                classified["_building_type"] = str(
                    asset.get("building_type", "office")
                ).lower()
                classified["_climate_zone"] = str(
                    asset.get("climate_zone", "3")
                )

            # Classify vehicle subtype
            elif category == "vehicle":
                classified["_vehicle_type"] = str(
                    asset.get("vehicle_type", "medium_car")
                ).lower()

            # Classify equipment subtype
            elif category == "equipment":
                classified["_equipment_type"] = str(
                    asset.get("equipment_type", "manufacturing")
                ).lower()

            # Classify IT subtype
            elif category == "it_asset":
                classified["_it_type"] = str(
                    asset.get("it_type", "server")
                ).lower()

            # Country / region
            classified["_country"] = str(
                asset.get("country", asset.get("region", "GLOBAL"))
            ).upper()

            # Identify energy sources present
            energy_sources: List[str] = []
            if self._safe_decimal(asset.get("electricity_kwh", 0)) > _ZERO:
                energy_sources.append("electricity")
            if self._safe_decimal(asset.get("natural_gas_kwh", 0)) > _ZERO:
                energy_sources.append("natural_gas")
            if self._safe_decimal(asset.get("district_heating_kwh", 0)) > _ZERO:
                energy_sources.append("district_heating")
            if self._safe_decimal(asset.get("district_cooling_kwh", 0)) > _ZERO:
                energy_sources.append("district_cooling")
            classified["_energy_sources"] = energy_sources

            classified_assets.append(classified)

        result = dict(validated)
        result["assets"] = classified_assets
        return result

    def _stage_normalize(self, classified: dict) -> dict:
        """
        Stage 3: NORMALIZE - Unit conversions.
        """
        normalized_assets: List[dict] = []

        for asset in classified["assets"]:
            norm = dict(asset)

            # sqft -> sqm
            if "area_sqft" in asset and "area_sqm" not in asset:
                sqft = self._safe_decimal(asset["area_sqft"])
                norm["area_sqm"] = str(
                    (sqft * _SQFT_TO_SQM).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                )
                norm["_original_area_sqft"] = str(sqft)

            # therms -> kWh (natural gas)
            if "natural_gas_therms" in asset and "natural_gas_kwh" not in asset:
                therms = self._safe_decimal(asset["natural_gas_therms"])
                norm["natural_gas_kwh"] = str(
                    (therms * _THERMS_TO_KWH).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                )

            # MMBtu -> kWh
            if "energy_mmbtu" in asset and "energy_kwh" not in asset:
                mmbtu = self._safe_decimal(asset["energy_mmbtu"])
                norm["energy_kwh"] = str(
                    (mmbtu * _MMBTU_TO_KWH).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                )

            # gallons -> litres (fuel)
            if "fuel_gallons" in asset and "fuel_litres" not in asset:
                gallons = self._safe_decimal(asset["fuel_gallons"])
                norm["fuel_litres"] = str(
                    (gallons * _GALLONS_TO_LITRES).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                )

            # miles -> km (vehicles)
            if "annual_km_per_vehicle" not in asset:
                if "annual_miles_per_vehicle" in asset:
                    miles = self._safe_decimal(asset["annual_miles_per_vehicle"])
                    norm["annual_km_per_vehicle"] = str(
                        (miles * _MILES_TO_KM).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                    )

            # Currency conversion for spend-based
            if asset.get("_calc_method") == "spend_based":
                amount = self._safe_decimal(asset.get("spend_amount", 0))
                currency = str(asset.get("currency", "USD")).upper()
                reporting_year = int(asset.get("reporting_year", 2024))

                # Convert to USD
                rate = CURRENCY_RATES.get(currency, _ONE)
                amount_usd = (amount * rate).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

                # CPI deflation to 2024 base
                deflator = CPI_DEFLATORS.get(reporting_year, _ONE)
                if deflator > _ZERO:
                    amount_usd_deflated = (amount_usd / deflator).quantize(
                        _QUANT_8DP, rounding=ROUND_HALF_UP
                    )
                else:
                    amount_usd_deflated = amount_usd

                norm["_spend_usd"] = str(amount_usd)
                norm["_spend_usd_deflated"] = str(amount_usd_deflated)
                norm["_currency_rate"] = str(rate)
                norm["_cpi_deflator"] = str(deflator)

            # Months in period (for partial-year proration)
            months = self._safe_decimal(asset.get("months_in_period", "12"))
            if months <= _ZERO:
                months = _TWELVE
            if months > _TWELVE:
                months = _TWELVE
            norm["_months"] = str(months)

            normalized_assets.append(norm)

        result = dict(classified)
        result["assets"] = normalized_assets
        return result

    def _stage_resolve_efs(self, normalized: dict) -> dict:
        """
        Stage 4: RESOLVE_EFS - Resolve emission factors per asset.
        """
        resolved_assets: List[dict] = []

        for asset in normalized["assets"]:
            resolved = dict(asset)
            country = asset.get("_country", "GLOBAL")
            calc_method = asset.get("_calc_method", "asset_specific")
            category = asset.get("_category", "building")

            # Grid emission factor
            grid_ef = GRID_EMISSION_FACTORS.get(
                country, GRID_EMISSION_FACTORS["GLOBAL"]
            )
            resolved["_grid_ef"] = str(grid_ef)

            # Natural gas EF
            resolved["_gas_ef"] = str(GAS_EMISSION_FACTOR)
            resolved["_gas_wtt_ef"] = str(GAS_WTT_FACTOR)

            # District heating EF
            dh_ef = DISTRICT_HEATING_FACTORS.get(
                country, DISTRICT_HEATING_FACTORS["GLOBAL"]
            )
            resolved["_dh_ef"] = str(dh_ef)

            # District cooling EF
            dc_ef = DISTRICT_COOLING_FACTORS.get(
                country, DISTRICT_COOLING_FACTORS["GLOBAL"]
            )
            resolved["_dc_ef"] = str(dc_ef)

            # EUI benchmark for average-data
            if calc_method == "average_data" and category == "building":
                building_type = asset.get("_building_type", "office")
                climate_zone = asset.get("_climate_zone", "3")
                eui_by_zone = EUI_BENCHMARKS.get(building_type, EUI_BENCHMARKS["office"])
                eui = eui_by_zone.get(climate_zone, eui_by_zone.get("3", Decimal("220")))
                resolved["_eui_kwh_per_sqm"] = str(eui)

            # Vehicle EF
            if category == "vehicle":
                vtype = asset.get("_vehicle_type", "medium_car")
                vef = VEHICLE_EMISSION_FACTORS.get(
                    vtype, VEHICLE_EMISSION_FACTORS["medium_car"]
                )
                resolved["_vehicle_ef_per_km"] = str(vef["ef_per_km"])
                resolved["_vehicle_wtt_per_km"] = str(vef["wtt_per_km"])

            # Equipment load factor
            if category == "equipment":
                etype = asset.get("_equipment_type", "manufacturing")
                load_factor = EQUIPMENT_LOAD_FACTORS.get(
                    etype, Decimal("0.60")
                )
                resolved["_load_factor"] = str(load_factor)

            # IT PUE and power defaults
            if category == "it_asset":
                itype = asset.get("_it_type", "server")
                pue = self._safe_decimal(
                    asset.get("pue", str(IT_PUE_DEFAULTS.get(itype, Decimal("1.58"))))
                )
                power_kw = self._safe_decimal(
                    asset.get("power_kw", str(IT_POWER_DEFAULTS.get(itype, Decimal("0.500"))))
                )
                util = self._safe_decimal(
                    asset.get("utilization", str(IT_UTILIZATION_DEFAULTS.get(itype, Decimal("0.40"))))
                )
                resolved["_pue"] = str(pue)
                resolved["_power_kw"] = str(power_kw)
                resolved["_utilization"] = str(util)

            # EEIO factor for spend-based
            if calc_method == "spend_based":
                naics = str(asset.get("naics_code", "531120"))
                eeio_entry = EEIO_FACTORS.get(naics, EEIO_FACTORS.get("531120", {}))
                eeio_ef = eeio_entry.get("ef", Decimal("0.22500"))
                resolved["_eeio_ef"] = str(eeio_ef)
                resolved["_eeio_naics"] = naics
                resolved["_eeio_name"] = eeio_entry.get("name", "")

            resolved["_ef_source"] = self._determine_ef_source(calc_method, country)
            resolved_assets.append(resolved)

        result = dict(normalized)
        result["assets"] = resolved_assets
        return result

    def _stage_calculate(self, resolved: dict) -> dict:
        """
        Stage 5: CALCULATE - Emissions per asset.
        """
        calculated_assets: List[dict] = []
        total_raw_co2e = _ZERO

        for asset in resolved["assets"]:
            calc = dict(asset)
            calc_method = asset.get("_calc_method", "asset_specific")
            category = asset.get("_category", "building")

            co2e = _ZERO
            wtt = _ZERO

            if calc_method == "lessor_specific":
                co2e, wtt = self._calc_lessor(asset)
            elif calc_method == "spend_based":
                co2e, wtt = self._calc_spend(asset)
            elif category == "building" and calc_method == "average_data":
                co2e, wtt = self._calc_building_average(asset)
            elif category == "building":
                co2e, wtt = self._calc_building_specific(asset)
            elif category == "vehicle":
                co2e, wtt = self._calc_vehicle(asset)
            elif category == "equipment":
                co2e, wtt = self._calc_equipment(asset)
            elif category == "it_asset":
                co2e, wtt = self._calc_it_asset(asset)
            else:
                logger.warning(
                    "Unknown calc path: method=%s category=%s",
                    calc_method, category,
                )

            total = (co2e + wtt).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
            calc["_raw_co2e"] = str(co2e)
            calc["_raw_wtt"] = str(wtt)
            calc["_raw_total"] = str(total)

            total_raw_co2e += total
            calculated_assets.append(calc)

        result = dict(resolved)
        result["assets"] = calculated_assets
        result["total_raw_co2e"] = str(
            total_raw_co2e.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        )
        return result

    def _stage_allocate(self, calculated: dict) -> dict:
        """
        Stage 6: ALLOCATE - Multi-tenant allocation and partial-year proration.
        """
        allocated_assets: List[dict] = []
        total_co2e = _ZERO

        for asset in calculated["assets"]:
            alloc = dict(asset)
            raw_total = self._safe_decimal(asset.get("_raw_total", "0"))

            # Allocation factor (1.0 = single tenant / 100% of asset)
            alloc_factor = self._safe_decimal(
                asset.get("allocation_factor", "1")
            )
            if alloc_factor <= _ZERO:
                alloc_factor = _ONE
            if alloc_factor > _ONE:
                alloc_factor = _ONE

            # Partial-year proration
            months = self._safe_decimal(asset.get("_months", "12"))
            proration = (months / _TWELVE).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

            # Sub-lease exclusion
            sub_lease_fraction = self._safe_decimal(
                asset.get("sub_lease_fraction", "0")
            )
            sub_lease_factor = (_ONE - sub_lease_fraction).quantize(
                _QUANT_8DP, rounding=ROUND_HALF_UP
            )
            if sub_lease_factor < _ZERO:
                sub_lease_factor = _ZERO

            # Final allocated CO2e
            allocated_co2e = (
                raw_total * alloc_factor * proration * sub_lease_factor
            ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

            alloc["_allocation_factor"] = str(alloc_factor)
            alloc["_proration"] = str(proration)
            alloc["_sub_lease_factor"] = str(sub_lease_factor)
            alloc["_allocated_co2e"] = str(allocated_co2e)

            total_co2e += allocated_co2e
            allocated_assets.append(alloc)

        result = dict(calculated)
        result["assets"] = allocated_assets
        result["total_co2e"] = str(
            total_co2e.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        )
        return result

    def _stage_aggregate(self, allocated: dict) -> dict:
        """
        Stage 7: AGGREGATE - By asset_type, building_type, country, energy_source.
        """
        by_asset_type: Dict[str, Decimal] = {}
        by_building_type: Dict[str, Decimal] = {}
        by_country: Dict[str, Decimal] = {}
        by_energy_source: Dict[str, Decimal] = {}
        by_calc_method: Dict[str, Decimal] = {}

        asset_results: List[dict] = []

        for asset in allocated["assets"]:
            co2e = self._safe_decimal(asset.get("_allocated_co2e", "0"))
            category = asset.get("_category", "unknown")
            country = asset.get("_country", "GLOBAL")
            calc_method = asset.get("_calc_method", "unknown")

            # By asset type
            by_asset_type[category] = by_asset_type.get(category, _ZERO) + co2e

            # By building type
            if category == "building":
                bt = asset.get("_building_type", "unknown")
                by_building_type[bt] = by_building_type.get(bt, _ZERO) + co2e

            # By country
            by_country[country] = by_country.get(country, _ZERO) + co2e

            # By energy source
            for source in asset.get("_energy_sources", []):
                by_energy_source[source] = by_energy_source.get(source, _ZERO) + co2e

            # By calc method
            by_calc_method[calc_method] = by_calc_method.get(calc_method, _ZERO) + co2e

            # Per-asset result for output
            asset_results.append({
                "asset_id": asset.get("asset_id"),
                "asset_category": category,
                "lease_type": asset.get("_lease_class", "operating"),
                "calculation_method": calc_method,
                "country": country,
                "raw_co2e": asset.get("_raw_total", "0"),
                "allocation_factor": asset.get("_allocation_factor", "1"),
                "proration": asset.get("_proration", "1"),
                "sub_lease_factor": asset.get("_sub_lease_factor", "1"),
                "allocated_co2e": asset.get("_allocated_co2e", "0"),
            })

        total_co2e = self._safe_decimal(allocated.get("total_co2e", "0"))

        # Hot-spot analysis
        hot_spots: List[dict] = []
        for ar in sorted(asset_results, key=lambda x: self._safe_decimal(x["allocated_co2e"]), reverse=True):
            ar_co2e = self._safe_decimal(ar["allocated_co2e"])
            if total_co2e > _ZERO:
                pct = (ar_co2e / total_co2e * _HUNDRED).quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)
            else:
                pct = _ZERO
            hot_spots.append({
                "asset_id": ar["asset_id"],
                "co2e": ar["allocated_co2e"],
                "percentage": str(pct),
            })

        # Total leased area
        total_area_sqm = _ZERO
        for asset in allocated["assets"]:
            if asset.get("_category") == "building":
                area = self._safe_decimal(asset.get("area_sqm", 0))
                total_area_sqm += area

        intensity_per_sqm = _ZERO
        if total_area_sqm > _ZERO:
            intensity_per_sqm = (total_co2e / total_area_sqm).quantize(
                _QUANT_8DP, rounding=ROUND_HALF_UP
            )

        result = dict(allocated)
        result["asset_results"] = asset_results
        result["by_asset_type"] = {k: str(v.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)) for k, v in by_asset_type.items()}
        result["by_building_type"] = {k: str(v.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)) for k, v in by_building_type.items()}
        result["by_country"] = {k: str(v.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)) for k, v in by_country.items()}
        result["by_energy_source"] = {k: str(v.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)) for k, v in by_energy_source.items()}
        result["by_calc_method"] = {k: str(v.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)) for k, v in by_calc_method.items()}
        result["hot_spots"] = hot_spots[:20]
        result["total_area_sqm"] = str(total_area_sqm.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
        result["intensity_per_sqm"] = str(intensity_per_sqm)
        result["asset_count"] = len(asset_results)
        result["reporting_period"] = allocated.get("reporting_period", "")
        result["method"] = "mixed" if len(by_calc_method) > 1 else list(by_calc_method.keys())[0] if by_calc_method else "unknown"

        return result

    def _stage_compliance(self, aggregated: dict) -> dict:
        """
        Stage 8: COMPLIANCE - Run all 7 frameworks + 10 DC rules.
        """
        engine = self._get_compliance_engine()
        if engine is None:
            return {"overall_status": "unavailable", "reason": "Compliance engine not loaded"}

        try:
            fw_results = engine.check_all_frameworks(aggregated)
            summary = engine.get_compliance_summary(fw_results)

            # Double-counting check on assets
            assets = aggregated.get("assets", [])
            dc_findings = engine.check_double_counting(assets)

            summary["double_counting_findings"] = dc_findings
            summary["double_counting_count"] = len(dc_findings)

            return summary
        except Exception as e:
            logger.error("Compliance check failed: %s", e, exc_info=True)
            return {
                "overall_status": "error",
                "error": str(e),
            }

    def _stage_provenance(self, chain_id: str, data: dict) -> dict:
        """
        Stage 9: PROVENANCE - Generate SHA-256 chain record.
        """
        chain = self._provenance_chains.get(chain_id, [])
        return {
            "chain_id": chain_id,
            "chain_length": len(chain),
            "stages": [entry.get("stage", "") for entry in chain],
            "chain_hashes": [entry.get("chain_hash", "") for entry in chain],
        }

    def _stage_seal(self, chain_id: str, aggregated: dict) -> dict:
        """
        Stage 10: SEAL - Merkle root, final hash, timestamp, immutable.
        """
        chain = self._provenance_chains.get(chain_id, [])

        leaf_hashes = [entry.get("output_hash", "") for entry in chain]
        merkle_root = self._build_merkle_root(leaf_hashes)

        chain_str = json.dumps(chain, sort_keys=True, default=str)
        # Build result without internal fields for clean output
        clean_data = {
            k: v for k, v in aggregated.items()
            if not k.startswith("_") and k != "assets"
        }
        result_str = json.dumps(clean_data, sort_keys=True, default=str)
        combined = f"{chain_str}|{result_str}|{merkle_root}"
        provenance_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        sealed = {}
        # Copy non-internal fields
        for k, v in aggregated.items():
            if k == "assets":
                continue
            sealed[k] = v

        sealed["asset_results"] = aggregated.get("asset_results", [])
        sealed["provenance_hash"] = provenance_hash
        sealed["merkle_root"] = merkle_root
        sealed["sealed_at"] = datetime.now(timezone.utc).isoformat()
        sealed["provenance_chain_length"] = len(chain)
        sealed["immutable"] = True
        sealed["agent_id"] = "GL-MRV-S3-008"
        sealed["agent_component"] = "AGENT-MRV-021"
        sealed["version"] = ENGINE_VERSION

        return sealed

    # ==========================================================================
    # CALCULATION METHODS (PRIVATE)
    # ==========================================================================

    def _calc_building_specific(self, asset: dict) -> Tuple[Decimal, Decimal]:
        """
        Building asset-specific calculation.

        CO2e = (elec*grid_ef + gas*gas_ef + heat*dh_ef + cool*dc_ef)
        WTT for gas included separately.
        """
        grid_ef = self._safe_decimal(asset.get("_grid_ef", "0.4360"))
        gas_ef = self._safe_decimal(asset.get("_gas_ef", str(GAS_EMISSION_FACTOR)))
        gas_wtt = self._safe_decimal(asset.get("_gas_wtt_ef", str(GAS_WTT_FACTOR)))
        dh_ef = self._safe_decimal(asset.get("_dh_ef", "0.1600"))
        dc_ef = self._safe_decimal(asset.get("_dc_ef", "0.1300"))

        elec_kwh = self._safe_decimal(asset.get("electricity_kwh", "0"))
        gas_kwh = self._safe_decimal(asset.get("natural_gas_kwh", "0"))
        heat_kwh = self._safe_decimal(asset.get("district_heating_kwh", "0"))
        cool_kwh = self._safe_decimal(asset.get("district_cooling_kwh", "0"))

        co2e_elec = (elec_kwh * grid_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        co2e_gas = (gas_kwh * gas_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        co2e_heat = (heat_kwh * dh_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        co2e_cool = (cool_kwh * dc_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        total_co2e = co2e_elec + co2e_gas + co2e_heat + co2e_cool

        wtt_gas = (gas_kwh * gas_wtt).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return total_co2e, wtt_gas

    def _calc_building_average(self, asset: dict) -> Tuple[Decimal, Decimal]:
        """
        Building average-data calculation.

        CO2e = area_sqm * EUI * grid_ef
        """
        area_sqm = self._safe_decimal(asset.get("area_sqm", "0"))
        eui = self._safe_decimal(asset.get("_eui_kwh_per_sqm", "220"))
        grid_ef = self._safe_decimal(asset.get("_grid_ef", "0.4360"))

        total_energy_kwh = (area_sqm * eui).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        co2e = (total_energy_kwh * grid_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        # No separate WTT for average-data (embedded in EUI factors)
        wtt = _ZERO

        return co2e, wtt

    def _calc_vehicle(self, asset: dict) -> Tuple[Decimal, Decimal]:
        """
        Vehicle calculation.

        CO2e = annual_km * count * ef_per_km
        WTT  = annual_km * count * wtt_per_km
        """
        annual_km = self._safe_decimal(asset.get("annual_km_per_vehicle", "15000"))
        count = self._safe_decimal(asset.get("vehicle_count", "1"))
        ef_per_km = self._safe_decimal(asset.get("_vehicle_ef_per_km", "0.17130"))
        wtt_per_km = self._safe_decimal(asset.get("_vehicle_wtt_per_km", "0.04074"))

        total_km = (annual_km * count).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        co2e = (total_km * ef_per_km).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        wtt = (total_km * wtt_per_km).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return co2e, wtt

    def _calc_equipment(self, asset: dict) -> Tuple[Decimal, Decimal]:
        """
        Equipment calculation.

        CO2e = power_kw * annual_hours * load_factor * count * grid_ef
        """
        power_kw = self._safe_decimal(asset.get("power_kw", "50"))
        annual_hours = self._safe_decimal(asset.get("annual_hours", "2000"))
        load_factor = self._safe_decimal(asset.get("_load_factor", "0.60"))
        count = self._safe_decimal(asset.get("equipment_count", "1"))
        grid_ef = self._safe_decimal(asset.get("_grid_ef", "0.4360"))

        energy_kwh = (
            power_kw * annual_hours * load_factor * count
        ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        co2e = (energy_kwh * grid_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        wtt = _ZERO  # Grid WTT not separately tracked for equipment

        return co2e, wtt

    def _calc_it_asset(self, asset: dict) -> Tuple[Decimal, Decimal]:
        """
        IT asset calculation.

        CO2e = power_kw * PUE * annual_hours * utilization * count * grid_ef
        """
        power_kw = self._safe_decimal(asset.get("_power_kw", "0.500"))
        pue = self._safe_decimal(asset.get("_pue", "1.58"))
        annual_hours = self._safe_decimal(asset.get("annual_hours", "8760"))
        util = self._safe_decimal(asset.get("_utilization", "0.40"))
        count = self._safe_decimal(asset.get("asset_count", asset.get("it_count", "1")))
        grid_ef = self._safe_decimal(asset.get("_grid_ef", "0.4360"))

        energy_kwh = (
            power_kw * pue * annual_hours * util * count
        ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        co2e = (energy_kwh * grid_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        wtt = _ZERO

        return co2e, wtt

    def _calc_spend(self, asset: dict) -> Tuple[Decimal, Decimal]:
        """
        Spend-based EEIO calculation.

        CO2e = spend_usd_deflated * eeio_factor
        """
        spend_usd_deflated = self._safe_decimal(
            asset.get("_spend_usd_deflated", "0")
        )
        eeio_ef = self._safe_decimal(asset.get("_eeio_ef", "0.22500"))

        co2e = (spend_usd_deflated * eeio_ef).quantize(
            _QUANT_8DP, rounding=ROUND_HALF_UP
        )
        wtt = _ZERO

        return co2e, wtt

    def _calc_lessor(self, asset: dict) -> Tuple[Decimal, Decimal]:
        """
        Lessor-specific calculation.

        CO2e = lessor_co2e (already calculated by lessor)
        Allocation and proration applied in Stage 6.
        """
        lessor_co2e = self._safe_decimal(asset.get("lessor_co2e", "0"))
        return lessor_co2e, _ZERO

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def _determine_ef_source(self, calc_method: str, country: str) -> str:
        """Determine the emission factor source string."""
        if calc_method == "spend_based":
            return "EEIO"
        if calc_method == "lessor_specific":
            return "LESSOR"
        if country == "US":
            return "eGRID_2023"
        if country == "GB":
            return "DEFRA_2024"
        return "IEA_2024"

    def _get_compliance_engine(self) -> Optional[Any]:
        """Get or create ComplianceCheckerEngine (lazy loading)."""
        if self._compliance_engine is None:
            try:
                from greenlang.agents.mrv.upstream_leased_assets.compliance_checker import (
                    ComplianceCheckerEngine,
                )
                self._compliance_engine = ComplianceCheckerEngine.get_instance()
            except ImportError:
                logger.debug("ComplianceCheckerEngine not available")
                self._compliance_engine = None
        return self._compliance_engine

    def _record_provenance(
        self,
        chain_id: str,
        stage: str,
        input_data: Any,
        output_data: Any,
    ) -> None:
        """Record a provenance entry for a pipeline stage."""
        input_str = self._canonical_json(input_data)
        output_str = self._canonical_json(output_data)

        input_hash = hashlib.sha256(input_str.encode("utf-8")).hexdigest()
        output_hash = hashlib.sha256(output_str.encode("utf-8")).hexdigest()

        chain = self._provenance_chains.get(chain_id)
        if chain is None:
            return

        previous_hash = chain[-1]["chain_hash"] if chain else "0" * 64
        chain_input = f"{previous_hash}|{stage}|{input_hash}|{output_hash}"
        chain_hash = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()

        entry = {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hash": input_hash,
            "output_hash": output_hash,
            "chain_hash": chain_hash,
        }
        chain.append(entry)

    def _build_merkle_root(self, leaf_hashes: List[str]) -> str:
        """Build a Merkle tree root from leaf hashes."""
        if not leaf_hashes:
            return hashlib.sha256(b"empty").hexdigest()

        current_level = list(leaf_hashes)

        while len(current_level) > 1:
            next_level: List[str] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha256(
                    f"{left}{right}".encode("utf-8")
                ).hexdigest()
                next_level.append(combined)
            current_level = next_level

        return current_level[0]

    @staticmethod
    def _elapsed_ms(start: datetime) -> float:
        """Calculate milliseconds elapsed since start."""
        return (datetime.now(timezone.utc) - start).total_seconds() * 1000.0

    @staticmethod
    def _safe_decimal(value: Any) -> Decimal:
        """Safely convert any value to Decimal."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return _ZERO

    @staticmethod
    def _canonical_json(data: Any) -> str:
        """Produce deterministic JSON serialization for hashing."""
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(data)

    @staticmethod
    def _hash_json(data: Any) -> str:
        """SHA-256 hash of canonical JSON representation."""
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get_engine_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "calculations_performed": self._calculations_performed,
            "supported_asset_categories": sorted(VALID_ASSET_CATEGORIES),
            "supported_building_types": sorted(VALID_BUILDING_TYPES),
            "supported_vehicle_types": sorted(VALID_VEHICLE_TYPES),
            "supported_equipment_types": sorted(VALID_EQUIPMENT_TYPES),
            "supported_it_types": sorted(VALID_IT_TYPES),
            "grid_ef_countries": sorted(GRID_EMISSION_FACTORS.keys()),
            "eeio_naics_codes": sorted(EEIO_FACTORS.keys()),
        }


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

def get_pipeline_engine() -> UpstreamLeasedPipelineEngine:
    """
    Get the singleton UpstreamLeasedPipelineEngine instance.

    Returns:
        UpstreamLeasedPipelineEngine singleton.

    Example:
        >>> engine = get_pipeline_engine()
        >>> result = engine.calculate({"assets": [...]})
    """
    return UpstreamLeasedPipelineEngine()


def reset_pipeline_engine() -> None:
    """
    Reset the pipeline engine singleton (for testing only).

    Example:
        >>> reset_pipeline_engine()
    """
    UpstreamLeasedPipelineEngine.reset_instance()


__all__ = [
    "ENGINE_ID",
    "ENGINE_VERSION",
    "PipelineStatus",
    "UpstreamLeasedPipelineEngine",
    "get_pipeline_engine",
    "reset_pipeline_engine",
]
