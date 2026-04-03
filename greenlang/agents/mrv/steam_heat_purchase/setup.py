# -*- coding: utf-8 -*-
"""
Steam/Heat Purchase Agent Service Facade - AGENT-MRV-011

This module provides the unified service interface for the Steam/Heat Purchase Agent.
Implements thread-safe singleton pattern and delegates to 7 specialized engines.

Architecture:
    - SteamHeatDatabaseEngine: Emission factors, fuel data, regional data
    - SteamCalculatorEngine: Steam emissions (Tier 1/2/3)
    - HeatingCalculatorEngine: District heating, boilers, heat pumps
    - CoolingCalculatorEngine: Chillers, district cooling
    - CHPAllocationEngine: Combined heat & power allocation (4 methods)
    - UncertaintyQuantifierEngine: Monte Carlo, error propagation
    - ComplianceCheckerEngine: 8 regulatory frameworks
    - SteamHeatPipelineEngine: End-to-end orchestration

Usage:
    >>> service = get_service(config)
    >>> result = service.calculate_steam_emissions(request)
    >>> compliance = service.check_compliance(result)

Author: GreenLang Backend Team
Date: 2026-02-22
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

__all__ = [
    "SteamHeatPurchaseService",
    "get_service",
    "create_service",
    "reset_service",
]

# Conditional imports for all 7 engines
_DATABASE_AVAILABLE = False
_STEAM_CALC_AVAILABLE = False
_HEATING_CALC_AVAILABLE = False
_COOLING_CALC_AVAILABLE = False
_CHP_AVAILABLE = False
_UNCERTAINTY_AVAILABLE = False
_COMPLIANCE_AVAILABLE = False
_PIPELINE_AVAILABLE = False
_CONFIG_AVAILABLE = False
_METRICS_AVAILABLE = False
_PROVENANCE_AVAILABLE = False

try:
    from greenlang.agents.mrv.steam_heat_purchase.database import SteamHeatDatabaseEngine
    _DATABASE_AVAILABLE = True
except ImportError:
    logger.warning("SteamHeatDatabaseEngine not available")
    SteamHeatDatabaseEngine = None

try:
    from greenlang.agents.mrv.steam_heat_purchase.steam_calculator import SteamCalculatorEngine
    _STEAM_CALC_AVAILABLE = True
except ImportError:
    logger.warning("SteamCalculatorEngine not available")
    SteamCalculatorEngine = None

try:
    from greenlang.agents.mrv.steam_heat_purchase.heating_calculator import HeatingCalculatorEngine
    _HEATING_CALC_AVAILABLE = True
except ImportError:
    logger.warning("HeatingCalculatorEngine not available")
    HeatingCalculatorEngine = None

try:
    from greenlang.agents.mrv.steam_heat_purchase.cooling_calculator import CoolingCalculatorEngine
    _COOLING_CALC_AVAILABLE = True
except ImportError:
    logger.warning("CoolingCalculatorEngine not available")
    CoolingCalculatorEngine = None

try:
    from greenlang.agents.mrv.steam_heat_purchase.chp_allocation import CHPAllocationEngine
    _CHP_AVAILABLE = True
except ImportError:
    logger.warning("CHPAllocationEngine not available")
    CHPAllocationEngine = None

try:
    from greenlang.agents.mrv.steam_heat_purchase.uncertainty import UncertaintyQuantifierEngine
    _UNCERTAINTY_AVAILABLE = True
except ImportError:
    logger.warning("UncertaintyQuantifierEngine not available")
    UncertaintyQuantifierEngine = None

try:
    from greenlang.agents.mrv.steam_heat_purchase.compliance import ComplianceCheckerEngine
    _COMPLIANCE_AVAILABLE = True
except ImportError:
    logger.warning("ComplianceCheckerEngine not available")
    ComplianceCheckerEngine = None

try:
    from greenlang.agents.mrv.steam_heat_purchase.pipeline import SteamHeatPipelineEngine
    _PIPELINE_AVAILABLE = True
except ImportError:
    logger.warning("SteamHeatPipelineEngine not available")
    SteamHeatPipelineEngine = None

try:
    from greenlang.agents.mrv.steam_heat_purchase.config import SteamHeatConfig
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("SteamHeatConfig not available")
    SteamHeatConfig = None

try:
    from greenlang.observability.metrics import record_metric, increment_counter
    _METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Metrics not available")
    def record_metric(*args, **kwargs): pass
    def increment_counter(*args, **kwargs): pass

try:
    from greenlang.provenance.tracker import ProvenanceTracker
    _PROVENANCE_AVAILABLE = True
except ImportError:
    logger.warning("ProvenanceTracker not available")
    ProvenanceTracker = None

# Utility functions

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of data for provenance."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

def _round_decimal(value: float, precision: int = 4) -> Decimal:
    """Round value to Decimal with specified precision."""
    if value is None:
        return None
    quantize_str = "0." + "0" * precision
    return Decimal(str(value)).quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

class SteamHeatPurchaseService:
    """
    Service facade for Steam/Heat Purchase Agent (AGENT-MRV-011).

    Provides unified interface to 7 specialized engines:
    - Database: Emission factors, fuel data
    - Steam Calculator: Steam emissions (Tier 1/2/3)
    - Heating Calculator: District heating, boilers, heat pumps
    - Cooling Calculator: Chillers, district cooling
    - CHP Allocation: Combined heat & power (4 methods)
    - Uncertainty Quantifier: Monte Carlo, error propagation
    - Compliance Checker: 8 regulatory frameworks
    - Pipeline: End-to-end orchestration

    Thread-safe singleton implementation with lazy engine initialization.

    Attributes:
        config: Service configuration
        db_engine: Database engine instance
        steam_engine: Steam calculator instance
        heating_engine: Heating calculator instance
        cooling_engine: Cooling calculator instance
        chp_engine: CHP allocation engine instance
        uncertainty_engine: Uncertainty quantifier instance
        compliance_engine: Compliance checker instance
        pipeline_engine: Pipeline orchestrator instance
        provenance_tracker: Provenance tracking instance

    Example:
        >>> service = get_service(config)
        >>> result = service.calculate_steam_emissions({
        ...     "steam_purchased_mwh": 1000.0,
        ...     "supplier_emission_factor": 0.25,
        ...     "calculation_method": "supplier_specific"
        ... })
        >>> assert result["status"] == "success"
        >>> compliance = service.check_compliance(result)
    """

    _instance: Optional[SteamHeatPurchaseService] = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        """Double-checked locking singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize service with lazy engine loading.

        Args:
            config: Service configuration (SteamHeatConfig or dict)
        """
        # Prevent re-initialization
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._init_time = utcnow()
        self._request_count = 0
        self._error_count = 0

        # Load config
        if config is None:
            if _CONFIG_AVAILABLE and SteamHeatConfig:
                self.config = SteamHeatConfig()
            else:
                self.config = {}
                logger.warning("No config provided and SteamHeatConfig not available")
        elif isinstance(config, dict):
            if _CONFIG_AVAILABLE and SteamHeatConfig:
                self.config = SteamHeatConfig(**config)
            else:
                self.config = config
        else:
            self.config = config

        # Initialize engines (lazy)
        self.db_engine = None
        self.steam_engine = None
        self.heating_engine = None
        self.cooling_engine = None
        self.chp_engine = None
        self.uncertainty_engine = None
        self.compliance_engine = None
        self.pipeline_engine = None
        self.provenance_tracker = None

        logger.info("SteamHeatPurchaseService initialized")

    def _ensure_engines(self):
        """Lazy initialization of all engines."""
        if self.db_engine is None:
            if _DATABASE_AVAILABLE and SteamHeatDatabaseEngine:
                self.db_engine = SteamHeatDatabaseEngine(self.config)
            else:
                raise RuntimeError("SteamHeatDatabaseEngine not available")

        if self.steam_engine is None:
            if _STEAM_CALC_AVAILABLE and SteamCalculatorEngine:
                self.steam_engine = SteamCalculatorEngine(self.config, self.db_engine)
            else:
                raise RuntimeError("SteamCalculatorEngine not available")

        if self.heating_engine is None:
            if _HEATING_CALC_AVAILABLE and HeatingCalculatorEngine:
                self.heating_engine = HeatingCalculatorEngine(self.config, self.db_engine)
            else:
                raise RuntimeError("HeatingCalculatorEngine not available")

        if self.cooling_engine is None:
            if _COOLING_CALC_AVAILABLE and CoolingCalculatorEngine:
                self.cooling_engine = CoolingCalculatorEngine(self.config, self.db_engine)
            else:
                raise RuntimeError("CoolingCalculatorEngine not available")

        if self.chp_engine is None:
            if _CHP_AVAILABLE and CHPAllocationEngine:
                self.chp_engine = CHPAllocationEngine(self.config, self.db_engine)
            else:
                raise RuntimeError("CHPAllocationEngine not available")

        if self.uncertainty_engine is None:
            if _UNCERTAINTY_AVAILABLE and UncertaintyQuantifierEngine:
                self.uncertainty_engine = UncertaintyQuantifierEngine(self.config)
            else:
                raise RuntimeError("UncertaintyQuantifierEngine not available")

        if self.compliance_engine is None:
            if _COMPLIANCE_AVAILABLE and ComplianceCheckerEngine:
                self.compliance_engine = ComplianceCheckerEngine(self.config)
            else:
                raise RuntimeError("ComplianceCheckerEngine not available")

        if self.pipeline_engine is None:
            if _PIPELINE_AVAILABLE and SteamHeatPipelineEngine:
                self.pipeline_engine = SteamHeatPipelineEngine(
                    config=self.config,
                    db_engine=self.db_engine,
                    steam_engine=self.steam_engine,
                    heating_engine=self.heating_engine,
                    cooling_engine=self.cooling_engine,
                    chp_engine=self.chp_engine,
                    uncertainty_engine=self.uncertainty_engine,
                    compliance_engine=self.compliance_engine
                )
            else:
                raise RuntimeError("SteamHeatPipelineEngine not available")

        if self.provenance_tracker is None and _PROVENANCE_AVAILABLE and ProvenanceTracker:
            self.provenance_tracker = ProvenanceTracker()

    # -------------------------------------------------------------------------
    # Pipeline Operations (delegate to SteamHeatPipelineEngine)
    # -------------------------------------------------------------------------

    def calculate_steam_emissions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate steam purchase emissions.

        Args:
            request: Calculation request with keys:
                - steam_purchased_mwh: Steam quantity (MWh)
                - calculation_method: supplier_specific, regional_grid, fuel_mix
                - supplier_emission_factor: kg CO2e/MWh (supplier_specific)
                - fuel_mix: List of fuels (fuel_mix method)
                - region: Region code (regional_grid)
                - facility_id: Optional facility identifier
                - reporting_period: Optional period (YYYY-MM)

        Returns:
            Calculation result with emissions, method, provenance

        Raises:
            ValueError: Invalid request data
            RuntimeError: Engine not available
        """
        start_time = time.time()
        self._request_count += 1

        try:
            self._ensure_engines()

            # Validate request
            if not isinstance(request, dict):
                raise ValueError("Request must be a dictionary")
            if "steam_purchased_mwh" not in request:
                raise ValueError("steam_purchased_mwh required")

            # Delegate to pipeline
            result = self.pipeline_engine.calculate_steam_emissions(request)

            # Add service metadata
            result["service_metadata"] = {
                "service": "SteamHeatPurchaseService",
                "method": "calculate_steam_emissions",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": utcnow().isoformat()
            }

            # Track metrics
            if _METRICS_AVAILABLE:
                record_metric(
                    "steam_heat_purchase.calculation.duration",
                    (time.time() - start_time) * 1000,
                    tags={"type": "steam", "method": request.get("calculation_method", "unknown")}
                )
                increment_counter(
                    "steam_heat_purchase.calculation.count",
                    tags={"type": "steam", "status": result.get("status", "unknown")}
                )

            return result

        except Exception as e:
            self._error_count += 1
            logger.error("Steam calculation failed: %s", e, exc_info=True)

            if _METRICS_AVAILABLE:
                increment_counter(
                    "steam_heat_purchase.calculation.error",
                    tags={"type": "steam", "error": type(e).__name__}
                )

            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": utcnow().isoformat()
            }

    def calculate_heating_emissions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate heating purchase emissions.

        Args:
            request: Calculation request with keys:
                - heating_type: district_heating, boiler, heat_pump
                - energy_purchased_mwh: Energy quantity
                - calculation_method: supplier_specific, regional_grid, technology_default
                - supplier_emission_factor: Optional supplier EF
                - region: Region code
                - technology: Technology type (heat_pump)
                - cop: Coefficient of performance (heat_pump)
                - facility_id: Optional facility identifier

        Returns:
            Calculation result with emissions, method, provenance
        """
        start_time = time.time()
        self._request_count += 1

        try:
            self._ensure_engines()

            if not isinstance(request, dict):
                raise ValueError("Request must be a dictionary")
            if "heating_type" not in request:
                raise ValueError("heating_type required")
            if "energy_purchased_mwh" not in request:
                raise ValueError("energy_purchased_mwh required")

            result = self.pipeline_engine.calculate_heating_emissions(request)

            result["service_metadata"] = {
                "service": "SteamHeatPurchaseService",
                "method": "calculate_heating_emissions",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": utcnow().isoformat()
            }

            if _METRICS_AVAILABLE:
                record_metric(
                    "steam_heat_purchase.calculation.duration",
                    (time.time() - start_time) * 1000,
                    tags={"type": "heating", "heating_type": request.get("heating_type", "unknown")}
                )
                increment_counter(
                    "steam_heat_purchase.calculation.count",
                    tags={"type": "heating", "status": result.get("status", "unknown")}
                )

            return result

        except Exception as e:
            self._error_count += 1
            logger.error("Heating calculation failed: %s", e, exc_info=True)

            if _METRICS_AVAILABLE:
                increment_counter(
                    "steam_heat_purchase.calculation.error",
                    tags={"type": "heating", "error": type(e).__name__}
                )

            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": utcnow().isoformat()
            }

    def calculate_cooling_emissions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate cooling purchase emissions.

        Args:
            request: Calculation request with keys:
                - cooling_type: district_cooling, chiller, absorption_chiller
                - energy_purchased_mwh: Energy quantity
                - calculation_method: supplier_specific, regional_grid, technology_default
                - supplier_emission_factor: Optional supplier EF
                - region: Region code
                - technology: Chiller technology
                - eer: Energy efficiency ratio (chiller)
                - facility_id: Optional facility identifier

        Returns:
            Calculation result with emissions, method, provenance
        """
        start_time = time.time()
        self._request_count += 1

        try:
            self._ensure_engines()

            if not isinstance(request, dict):
                raise ValueError("Request must be a dictionary")
            if "cooling_type" not in request:
                raise ValueError("cooling_type required")
            if "energy_purchased_mwh" not in request:
                raise ValueError("energy_purchased_mwh required")

            result = self.pipeline_engine.calculate_cooling_emissions(request)

            result["service_metadata"] = {
                "service": "SteamHeatPurchaseService",
                "method": "calculate_cooling_emissions",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": utcnow().isoformat()
            }

            if _METRICS_AVAILABLE:
                record_metric(
                    "steam_heat_purchase.calculation.duration",
                    (time.time() - start_time) * 1000,
                    tags={"type": "cooling", "cooling_type": request.get("cooling_type", "unknown")}
                )
                increment_counter(
                    "steam_heat_purchase.calculation.count",
                    tags={"type": "cooling", "status": result.get("status", "unknown")}
                )

            return result

        except Exception as e:
            self._error_count += 1
            logger.error("Cooling calculation failed: %s", e, exc_info=True)

            if _METRICS_AVAILABLE:
                increment_counter(
                    "steam_heat_purchase.calculation.error",
                    tags={"type": "cooling", "error": type(e).__name__}
                )

            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": utcnow().isoformat()
            }

    def calculate_chp_emissions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate CHP emissions with allocation.

        Args:
            request: Calculation request with keys:
                - allocation_method: energy, exergy, efficiency, carbon
                - total_fuel_input_mwh: Total fuel consumed
                - electricity_output_mwh: Electricity generated
                - heat_output_mwh: Heat generated
                - fuel_type: Fuel type code
                - thermal_efficiency: Heat output efficiency
                - electrical_efficiency: Electricity output efficiency
                - facility_id: Optional facility identifier

        Returns:
            Calculation result with allocated emissions
        """
        start_time = time.time()
        self._request_count += 1

        try:
            self._ensure_engines()

            if not isinstance(request, dict):
                raise ValueError("Request must be a dictionary")
            if "allocation_method" not in request:
                raise ValueError("allocation_method required")
            if "total_fuel_input_mwh" not in request:
                raise ValueError("total_fuel_input_mwh required")

            result = self.pipeline_engine.calculate_chp_emissions(request)

            result["service_metadata"] = {
                "service": "SteamHeatPurchaseService",
                "method": "calculate_chp_emissions",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": utcnow().isoformat()
            }

            if _METRICS_AVAILABLE:
                record_metric(
                    "steam_heat_purchase.calculation.duration",
                    (time.time() - start_time) * 1000,
                    tags={"type": "chp", "method": request.get("allocation_method", "unknown")}
                )
                increment_counter(
                    "steam_heat_purchase.calculation.count",
                    tags={"type": "chp", "status": result.get("status", "unknown")}
                )

            return result

        except Exception as e:
            self._error_count += 1
            logger.error("CHP calculation failed: %s", e, exc_info=True)

            if _METRICS_AVAILABLE:
                increment_counter(
                    "steam_heat_purchase.calculation.error",
                    tags={"type": "chp", "error": type(e).__name__}
                )

            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": utcnow().isoformat()
            }

    def run_batch(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run batch calculations for multiple requests.

        Args:
            requests: List of calculation requests (any type)

        Returns:
            Batch result with individual results and summary
        """
        start_time = time.time()

        try:
            self._ensure_engines()

            if not isinstance(requests, list):
                raise ValueError("Requests must be a list")

            result = self.pipeline_engine.run_batch(requests)

            result["service_metadata"] = {
                "service": "SteamHeatPurchaseService",
                "method": "run_batch",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": utcnow().isoformat()
            }

            if _METRICS_AVAILABLE:
                record_metric(
                    "steam_heat_purchase.batch.duration",
                    (time.time() - start_time) * 1000,
                    tags={"count": len(requests)}
                )
                increment_counter(
                    "steam_heat_purchase.batch.count",
                    tags={"status": result.get("status", "unknown")}
                )

            return result

        except Exception as e:
            self._error_count += 1
            logger.error("Batch calculation failed: %s", e, exc_info=True)

            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": utcnow().isoformat()
            }

    def aggregate_results(
        self,
        calc_ids: List[str],
        agg_type: str = "sum"
    ) -> Dict[str, Any]:
        """
        Aggregate multiple calculation results.

        Args:
            calc_ids: List of calculation IDs to aggregate
            agg_type: Aggregation type (sum, average, weighted_average)

        Returns:
            Aggregated result
        """
        start_time = time.time()

        try:
            self._ensure_engines()

            if not isinstance(calc_ids, list):
                raise ValueError("calc_ids must be a list")

            result = self.pipeline_engine.aggregate_results(calc_ids, agg_type)

            result["service_metadata"] = {
                "service": "SteamHeatPurchaseService",
                "method": "aggregate_results",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": utcnow().isoformat()
            }

            return result

        except Exception as e:
            self._error_count += 1
            logger.error("Aggregation failed: %s", e, exc_info=True)

            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": utcnow().isoformat()
            }

    # -------------------------------------------------------------------------
    # Database Operations (delegate to SteamHeatDatabaseEngine)
    # -------------------------------------------------------------------------

    def get_fuel_emission_factor(self, fuel_type: str) -> Dict[str, Any]:
        """
        Get emission factor for fuel type.

        Args:
            fuel_type: Fuel type code

        Returns:
            Emission factor data
        """
        try:
            self._ensure_engines()
            return self.db_engine.get_fuel_emission_factor(fuel_type)
        except Exception as e:
            logger.error("Failed to get fuel EF: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def get_all_fuel_emission_factors(self) -> Dict[str, Any]:
        """
        Get all fuel emission factors.

        Returns:
            All fuel emission factors
        """
        try:
            self._ensure_engines()
            return self.db_engine.get_all_fuel_emission_factors()
        except Exception as e:
            logger.error("Failed to get all fuel EFs: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def get_district_heating_factor(self, region: str) -> Dict[str, Any]:
        """
        Get district heating emission factor for region.

        Args:
            region: Region code

        Returns:
            District heating emission factor
        """
        try:
            self._ensure_engines()
            return self.db_engine.get_district_heating_factor(region)
        except Exception as e:
            logger.error("Failed to get district heating factor: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def get_cooling_system_factor(self, technology: str) -> Dict[str, Any]:
        """
        Get cooling system emission factor.

        Args:
            technology: Cooling technology type

        Returns:
            Cooling system emission factor
        """
        try:
            self._ensure_engines()
            return self.db_engine.get_cooling_system_factor(technology)
        except Exception as e:
            logger.error("Failed to get cooling factor: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def get_chp_defaults(self, fuel_type: str) -> Dict[str, Any]:
        """
        Get CHP default parameters for fuel type.

        Args:
            fuel_type: Fuel type code

        Returns:
            CHP default parameters
        """
        try:
            self._ensure_engines()
            return self.db_engine.get_chp_defaults(fuel_type)
        except Exception as e:
            logger.error("Failed to get CHP defaults: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    # -------------------------------------------------------------------------
    # CHP Operations (delegate to CHPAllocationEngine)
    # -------------------------------------------------------------------------

    def allocate_chp(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate CHP emissions between heat and power.

        Args:
            request: Allocation request with allocation_method, fuel input, outputs

        Returns:
            Allocated emissions
        """
        start_time = time.time()

        try:
            self._ensure_engines()
            result = self.chp_engine.allocate_emissions(request)

            if _METRICS_AVAILABLE:
                record_metric(
                    "steam_heat_purchase.chp.allocation.duration",
                    (time.time() - start_time) * 1000,
                    tags={"method": request.get("allocation_method", "unknown")}
                )

            return result

        except Exception as e:
            logger.error("CHP allocation failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def compare_chp_methods(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare all 4 CHP allocation methods.

        Args:
            request: CHP parameters

        Returns:
            Comparison of all methods
        """
        try:
            self._ensure_engines()
            return self.chp_engine.compare_allocation_methods(request)
        except Exception as e:
            logger.error("CHP comparison failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def compute_primary_energy_savings(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute primary energy savings from CHP.

        Args:
            request: CHP parameters with reference efficiencies

        Returns:
            Primary energy savings calculation
        """
        try:
            self._ensure_engines()
            return self.chp_engine.compute_primary_energy_savings(request)
        except Exception as e:
            logger.error("PES calculation failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    # -------------------------------------------------------------------------
    # Uncertainty & Compliance
    # -------------------------------------------------------------------------

    def quantify_uncertainty(
        self,
        calc_result: Dict[str, Any],
        method: str = "monte_carlo"
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty in calculation result.

        Args:
            calc_result: Calculation result to analyze
            method: monte_carlo or error_propagation

        Returns:
            Uncertainty quantification
        """
        start_time = time.time()

        try:
            self._ensure_engines()
            result = self.uncertainty_engine.quantify_uncertainty(calc_result, method)

            if _METRICS_AVAILABLE:
                record_metric(
                    "steam_heat_purchase.uncertainty.duration",
                    (time.time() - start_time) * 1000,
                    tags={"method": method}
                )

            return result

        except Exception as e:
            logger.error("Uncertainty quantification failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def check_compliance(
        self,
        calc_result: Dict[str, Any],
        frameworks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check compliance against regulatory frameworks.

        Args:
            calc_result: Calculation result to check
            frameworks: List of framework codes (or None for all)

        Returns:
            Compliance check results
        """
        start_time = time.time()

        try:
            self._ensure_engines()
            result = self.compliance_engine.check_compliance(calc_result, frameworks)

            if _METRICS_AVAILABLE:
                record_metric(
                    "steam_heat_purchase.compliance.duration",
                    (time.time() - start_time) * 1000,
                    tags={"frameworks": len(frameworks) if frameworks else "all"}
                )

            return result

        except Exception as e:
            logger.error("Compliance check failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def get_compliance_frameworks(self) -> List[str]:
        """
        Get list of supported compliance frameworks.

        Returns:
            List of framework codes
        """
        try:
            self._ensure_engines()
            return self.compliance_engine.get_supported_frameworks()
        except Exception as e:
            logger.error("Failed to get frameworks: %s", e, exc_info=True)
            return []

    # -------------------------------------------------------------------------
    # Utility Operations
    # -------------------------------------------------------------------------

    def get_calculation(self, calc_id: str) -> Dict[str, Any]:
        """
        Retrieve stored calculation by ID.

        Args:
            calc_id: Calculation identifier

        Returns:
            Calculation result
        """
        try:
            self._ensure_engines()
            return self.db_engine.get_calculation(calc_id)
        except Exception as e:
            logger.error("Failed to get calculation: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def compare_energy_sources(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare emissions from different energy sources.

        Args:
            requests: List of calculation requests to compare

        Returns:
            Comparison result with rankings
        """
        start_time = time.time()

        try:
            self._ensure_engines()

            # Run all calculations
            results = []
            for req in requests:
                # Determine type and calculate
                if "steam_purchased_mwh" in req:
                    result = self.calculate_steam_emissions(req)
                elif "heating_type" in req:
                    result = self.calculate_heating_emissions(req)
                elif "cooling_type" in req:
                    result = self.calculate_cooling_emissions(req)
                elif "allocation_method" in req:
                    result = self.calculate_chp_emissions(req)
                else:
                    result = {"status": "error", "error": "Unknown request type"}

                results.append(result)

            # Compare results
            comparison = {
                "status": "success",
                "comparison_count": len(results),
                "results": results,
                "ranking": sorted(
                    [
                        {
                            "index": i,
                            "emissions_kg_co2e": r.get("total_emissions_kg_co2e", float('inf')),
                            "source": r.get("energy_source", "unknown")
                        }
                        for i, r in enumerate(results)
                        if r.get("status") == "success"
                    ],
                    key=lambda x: x["emissions_kg_co2e"]
                ),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": utcnow().isoformat()
            }

            return comparison

        except Exception as e:
            logger.error("Energy source comparison failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all engines.

        Returns:
            Health status for each engine
        """
        health = {
            "status": "healthy",
            "timestamp": utcnow().isoformat(),
            "engines": {}
        }

        try:
            self._ensure_engines()

            # Check each engine
            engines = {
                "database": self.db_engine,
                "steam_calculator": self.steam_engine,
                "heating_calculator": self.heating_engine,
                "cooling_calculator": self.cooling_engine,
                "chp_allocation": self.chp_engine,
                "uncertainty_quantifier": self.uncertainty_engine,
                "compliance_checker": self.compliance_engine,
                "pipeline": self.pipeline_engine
            }

            for name, engine in engines.items():
                if engine is None:
                    health["engines"][name] = {"status": "unavailable"}
                    health["status"] = "degraded"
                else:
                    try:
                        # Try to call health_check if available
                        if hasattr(engine, "health_check"):
                            health["engines"][name] = engine.health_check()
                        else:
                            health["engines"][name] = {"status": "healthy"}
                    except Exception as e:
                        health["engines"][name] = {"status": "error", "error": str(e)}
                        health["status"] = "unhealthy"

        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)

        return health

    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Service statistics (uptime, request count, error rate)
        """
        uptime_seconds = (utcnow() - self._init_time).total_seconds()

        return {
            "status": "success",
            "service": "SteamHeatPurchaseService",
            "initialized_at": self._init_time.isoformat(),
            "uptime_seconds": uptime_seconds,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._request_count
                if self._request_count > 0
                else 0.0
            ),
            "timestamp": utcnow().isoformat()
        }

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information.

        Returns:
            Service metadata and capabilities
        """
        return {
            "status": "success",
            "service": "SteamHeatPurchaseService",
            "version": "1.0.0",
            "agent_code": "AGENT-MRV-011",
            "capabilities": [
                "steam_emissions_calculation",
                "heating_emissions_calculation",
                "cooling_emissions_calculation",
                "chp_allocation",
                "uncertainty_quantification",
                "compliance_checking",
                "batch_processing",
                "result_aggregation"
            ],
            "calculation_methods": {
                "steam": ["supplier_specific", "regional_grid", "fuel_mix"],
                "heating": ["supplier_specific", "regional_grid", "technology_default"],
                "cooling": ["supplier_specific", "regional_grid", "technology_default"],
                "chp_allocation": ["energy", "exergy", "efficiency", "carbon"]
            },
            "supported_frameworks": self.get_compliance_frameworks(),
            "engines": {
                "database": _DATABASE_AVAILABLE,
                "steam_calculator": _STEAM_CALC_AVAILABLE,
                "heating_calculator": _HEATING_CALC_AVAILABLE,
                "cooling_calculator": _COOLING_CALC_AVAILABLE,
                "chp_allocation": _CHP_AVAILABLE,
                "uncertainty_quantifier": _UNCERTAINTY_AVAILABLE,
                "compliance_checker": _COMPLIANCE_AVAILABLE,
                "pipeline": _PIPELINE_AVAILABLE
            },
            "timestamp": utcnow().isoformat()
        }

    @classmethod
    def reset(cls):
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
            logger.info("SteamHeatPurchaseService reset")

# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def get_service(config: Optional[Any] = None) -> SteamHeatPurchaseService:
    """
    Get or create service singleton.

    Args:
        config: Optional service configuration

    Returns:
        Service instance

    Example:
        >>> service = get_service()
        >>> result = service.calculate_steam_emissions(request)
    """
    return SteamHeatPurchaseService(config)

def create_service(config: Optional[Any] = None) -> SteamHeatPurchaseService:
    """
    Create service instance (alias for get_service).

    Args:
        config: Optional service configuration

    Returns:
        Service instance
    """
    return SteamHeatPurchaseService(config)

def reset_service():
    """
    Reset service singleton (for testing).

    Warning:
        This will destroy the current service instance.
        Use only in test teardown.
    """
    SteamHeatPurchaseService.reset()
