"""
CoolingPurchaseService - Unified service facade for AGENT-MRV-012.

This module implements the CoolingPurchaseService facade class that wraps
all 7 engines and provides a unified API for purchased cooling calculations.
Thread-safe singleton pattern with lazy-loaded engine dependencies.

Example:
    >>> service = get_service()
    >>> result = service.calculate_electric_chiller(request)
    >>> assert result.emissions_kg_co2e > 0
"""

import threading
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime
import logging

# Try importing all engines with fallback to None
try:
    from greenlang.cooling_purchase.cooling_database import CoolingDatabaseEngine
except ImportError:
    CoolingDatabaseEngine = None

try:
    from greenlang.cooling_purchase.electric_chiller_calculator import (
        ElectricChillerCalculatorEngine,
    )
except ImportError:
    ElectricChillerCalculatorEngine = None

try:
    from greenlang.cooling_purchase.absorption_cooling_calculator import (
        AbsorptionCoolingCalculatorEngine,
    )
except ImportError:
    AbsorptionCoolingCalculatorEngine = None

try:
    from greenlang.cooling_purchase.district_cooling_calculator import (
        DistrictCoolingCalculatorEngine,
    )
except ImportError:
    DistrictCoolingCalculatorEngine = None

try:
    from greenlang.cooling_purchase.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None

try:
    from greenlang.cooling_purchase.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None

try:
    from greenlang.cooling_purchase.cooling_purchase_pipeline import (
        CoolingPurchasePipelineEngine,
    )
except ImportError:
    CoolingPurchasePipelineEngine = None

# Import models
try:
    from greenlang.cooling_purchase.models import (
        ElectricChillerRequest,
        AbsorptionCoolingRequest,
        DistrictCoolingRequest,
        FreeCoolingRequest,
        TESRequest,
        BatchCoolingRequest,
        AggregationRequest,
        UncertaintyRequest,
        CalculationResult,
        TESCalculationResult,
        BatchCalculationResult,
        AggregationResult,
        UncertaintyResult,
        ComplianceCheckResult,
        CoolingTechnologySpec,
        CoolingTechnology,
        EfficiencyMetric,
        CoolingUnit,
        HeatSource,
        GWPSource,
    )
except ImportError:
    # Fallback to Any if models not available
    (
        ElectricChillerRequest,
        AbsorptionCoolingRequest,
        DistrictCoolingRequest,
        FreeCoolingRequest,
        TESRequest,
        BatchCoolingRequest,
        AggregationRequest,
        UncertaintyRequest,
        CalculationResult,
        TESCalculationResult,
        BatchCalculationResult,
        AggregationResult,
        UncertaintyResult,
        ComplianceCheckResult,
        CoolingTechnologySpec,
        CoolingTechnology,
        EfficiencyMetric,
        CoolingUnit,
        HeatSource,
        GWPSource,
    ) = (Any,) * 18

logger = logging.getLogger(__name__)


class CoolingPurchaseService:
    """
    Thread-safe singleton service facade for purchased cooling calculations.

    This class provides a unified API for all cooling purchase calculation
    engines, managing lazy initialization and thread safety.

    Attributes:
        _instance: Singleton instance
        _lock: Thread lock for singleton pattern
        _db_engine: Database engine instance
        _electric_engine: Electric chiller calculator
        _absorption_engine: Absorption cooling calculator
        _district_engine: District cooling calculator
        _uncertainty_engine: Uncertainty quantifier
        _compliance_engine: Compliance checker
        _pipeline_engine: Pipeline orchestrator

    Example:
        >>> service = CoolingPurchaseService.get_instance()
        >>> result = service.calculate_electric_chiller(request)
    """

    _instance: Optional["CoolingPurchaseService"] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self):
        """Initialize service with None engines (lazy-loaded)."""
        self._db_engine = None
        self._electric_engine = None
        self._absorption_engine = None
        self._district_engine = None
        self._uncertainty_engine = None
        self._compliance_engine = None
        self._pipeline_engine = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "CoolingPurchaseService":
        """Get singleton instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _ensure_db_engine(self):
        """Lazy-load database engine."""
        if self._db_engine is None:
            if CoolingDatabaseEngine is None:
                raise RuntimeError("CoolingDatabaseEngine not available")
            self._db_engine = CoolingDatabaseEngine()

    def _ensure_electric_engine(self):
        """Lazy-load electric chiller calculator."""
        if self._electric_engine is None:
            if ElectricChillerCalculatorEngine is None:
                raise RuntimeError("ElectricChillerCalculatorEngine not available")
            self._ensure_db_engine()
            self._electric_engine = ElectricChillerCalculatorEngine(self._db_engine)

    def _ensure_absorption_engine(self):
        """Lazy-load absorption cooling calculator."""
        if self._absorption_engine is None:
            if AbsorptionCoolingCalculatorEngine is None:
                raise RuntimeError("AbsorptionCoolingCalculatorEngine not available")
            self._ensure_db_engine()
            self._absorption_engine = AbsorptionCoolingCalculatorEngine(self._db_engine)

    def _ensure_district_engine(self):
        """Lazy-load district cooling calculator."""
        if self._district_engine is None:
            if DistrictCoolingCalculatorEngine is None:
                raise RuntimeError("DistrictCoolingCalculatorEngine not available")
            self._ensure_db_engine()
            self._district_engine = DistrictCoolingCalculatorEngine(self._db_engine)

    def _ensure_uncertainty_engine(self):
        """Lazy-load uncertainty quantifier."""
        if self._uncertainty_engine is None:
            if UncertaintyQuantifierEngine is None:
                raise RuntimeError("UncertaintyQuantifierEngine not available")
            self._uncertainty_engine = UncertaintyQuantifierEngine()

    def _ensure_compliance_engine(self):
        """Lazy-load compliance checker."""
        if self._compliance_engine is None:
            if ComplianceCheckerEngine is None:
                raise RuntimeError("ComplianceCheckerEngine not available")
            self._compliance_engine = ComplianceCheckerEngine()

    def _ensure_pipeline_engine(self):
        """Lazy-load pipeline orchestrator."""
        if self._pipeline_engine is None:
            if CoolingPurchasePipelineEngine is None:
                raise RuntimeError("CoolingPurchasePipelineEngine not available")
            self._ensure_db_engine()
            self._ensure_electric_engine()
            self._ensure_absorption_engine()
            self._ensure_district_engine()
            self._ensure_uncertainty_engine()
            self._ensure_compliance_engine()
            self._pipeline_engine = CoolingPurchasePipelineEngine(
                db_engine=self._db_engine,
                electric_engine=self._electric_engine,
                absorption_engine=self._absorption_engine,
                district_engine=self._district_engine,
                uncertainty_engine=self._uncertainty_engine,
                compliance_engine=self._compliance_engine,
            )

    # ==================== Electric Chiller Methods ====================

    def calculate_electric_chiller(
        self, request: ElectricChillerRequest
    ) -> CalculationResult:
        """
        Calculate emissions from electric chiller.

        Args:
            request: Electric chiller calculation request

        Returns:
            Calculation result with emissions

        Raises:
            RuntimeError: If engine not available
        """
        self._ensure_electric_engine()
        return self._electric_engine.calculate(request)

    def calculate_electric_chiller_full_load(
        self,
        cooling_kwh_th: Decimal,
        cop: Decimal,
        grid_ef_kg_co2e_kwh: Decimal,
        refrigerant_gwp: Optional[Decimal] = None,
        refrigerant_charge_kg: Optional[Decimal] = None,
        leak_rate_percent: Optional[Decimal] = None,
    ) -> CalculationResult:
        """
        Calculate electric chiller at full load.

        Args:
            cooling_kwh_th: Cooling output (kWh thermal)
            cop: Coefficient of performance
            grid_ef_kg_co2e_kwh: Grid emission factor
            refrigerant_gwp: Refrigerant GWP (optional)
            refrigerant_charge_kg: Refrigerant charge (optional)
            leak_rate_percent: Annual leak rate (optional)

        Returns:
            Calculation result
        """
        self._ensure_electric_engine()
        return self._electric_engine.calculate_full_load(
            cooling_kwh_th=cooling_kwh_th,
            cop=cop,
            grid_ef_kg_co2e_kwh=grid_ef_kg_co2e_kwh,
            refrigerant_gwp=refrigerant_gwp,
            refrigerant_charge_kg=refrigerant_charge_kg,
            leak_rate_percent=leak_rate_percent,
        )

    def calculate_iplv(
        self,
        cop_100: Decimal,
        cop_75: Decimal,
        cop_50: Decimal,
        cop_25: Decimal,
    ) -> Decimal:
        """
        Calculate Integrated Part Load Value (IPLV).

        Args:
            cop_100: COP at 100% load
            cop_75: COP at 75% load
            cop_50: COP at 50% load
            cop_25: COP at 25% load

        Returns:
            IPLV value
        """
        self._ensure_electric_engine()
        return self._electric_engine.calculate_iplv(cop_100, cop_75, cop_50, cop_25)

    # ==================== Absorption Cooling Methods ====================

    def calculate_absorption_cooling(
        self, request: AbsorptionCoolingRequest
    ) -> CalculationResult:
        """
        Calculate emissions from absorption cooling.

        Args:
            request: Absorption cooling calculation request

        Returns:
            Calculation result with emissions
        """
        self._ensure_absorption_engine()
        return self._absorption_engine.calculate(request)

    def calculate_single_effect(
        self,
        cooling_kwh_th: Decimal,
        thermal_cop: Decimal,
        heat_source: HeatSource,
        heat_ef_kg_co2e_kwh: Decimal,
        parasitic_electric_kwh: Optional[Decimal] = None,
        grid_ef_kg_co2e_kwh: Optional[Decimal] = None,
    ) -> CalculationResult:
        """
        Calculate single-effect absorption cooling.

        Args:
            cooling_kwh_th: Cooling output
            thermal_cop: Thermal COP
            heat_source: Heat source type
            heat_ef_kg_co2e_kwh: Heat emission factor
            parasitic_electric_kwh: Parasitic electricity
            grid_ef_kg_co2e_kwh: Grid emission factor

        Returns:
            Calculation result
        """
        self._ensure_absorption_engine()
        return self._absorption_engine.calculate_single_effect(
            cooling_kwh_th=cooling_kwh_th,
            thermal_cop=thermal_cop,
            heat_source=heat_source,
            heat_ef_kg_co2e_kwh=heat_ef_kg_co2e_kwh,
            parasitic_electric_kwh=parasitic_electric_kwh,
            grid_ef_kg_co2e_kwh=grid_ef_kg_co2e_kwh,
        )

    def calculate_double_effect(
        self,
        cooling_kwh_th: Decimal,
        thermal_cop: Decimal,
        heat_source: HeatSource,
        heat_ef_kg_co2e_kwh: Decimal,
        parasitic_electric_kwh: Optional[Decimal] = None,
        grid_ef_kg_co2e_kwh: Optional[Decimal] = None,
    ) -> CalculationResult:
        """
        Calculate double-effect absorption cooling.

        Args:
            cooling_kwh_th: Cooling output
            thermal_cop: Thermal COP
            heat_source: Heat source type
            heat_ef_kg_co2e_kwh: Heat emission factor
            parasitic_electric_kwh: Parasitic electricity
            grid_ef_kg_co2e_kwh: Grid emission factor

        Returns:
            Calculation result
        """
        self._ensure_absorption_engine()
        return self._absorption_engine.calculate_double_effect(
            cooling_kwh_th=cooling_kwh_th,
            thermal_cop=thermal_cop,
            heat_source=heat_source,
            heat_ef_kg_co2e_kwh=heat_ef_kg_co2e_kwh,
            parasitic_electric_kwh=parasitic_electric_kwh,
            grid_ef_kg_co2e_kwh=grid_ef_kg_co2e_kwh,
        )

    # ==================== District/Free/TES Methods ====================

    def calculate_district_cooling(
        self, request: DistrictCoolingRequest
    ) -> CalculationResult:
        """
        Calculate emissions from district cooling.

        Args:
            request: District cooling calculation request

        Returns:
            Calculation result with emissions
        """
        self._ensure_district_engine()
        return self._district_engine.calculate_district(request)

    def calculate_free_cooling(self, request: FreeCoolingRequest) -> CalculationResult:
        """
        Calculate emissions from free cooling.

        Args:
            request: Free cooling calculation request

        Returns:
            Calculation result with emissions
        """
        self._ensure_district_engine()
        return self._district_engine.calculate_free_cooling(request)

    def calculate_tes(self, request: TESRequest) -> TESCalculationResult:
        """
        Calculate thermal energy storage economics.

        Args:
            request: TES calculation request

        Returns:
            TES calculation result with cost savings
        """
        self._ensure_district_engine()
        return self._district_engine.calculate_tes(request)

    # ==================== Database Lookup Methods ====================

    def get_technology_spec(
        self, technology: CoolingTechnology
    ) -> CoolingTechnologySpec:
        """
        Get cooling technology specifications.

        Args:
            technology: Cooling technology type

        Returns:
            Technology specifications
        """
        self._ensure_db_engine()
        return self._db_engine.get_technology_spec(technology)

    def get_default_cop(self, technology: CoolingTechnology) -> Decimal:
        """
        Get default COP for technology.

        Args:
            technology: Cooling technology type

        Returns:
            Default COP value
        """
        self._ensure_db_engine()
        return self._db_engine.get_default_cop(technology)

    def get_district_ef(self, region: str) -> Decimal:
        """
        Get district cooling emission factor.

        Args:
            region: Region code

        Returns:
            Emission factor (kg CO2e/kWh)
        """
        self._ensure_db_engine()
        return self._db_engine.get_district_ef(region)

    def get_heat_source_ef(self, source: HeatSource) -> Decimal:
        """
        Get heat source emission factor.

        Args:
            source: Heat source type

        Returns:
            Emission factor (kg CO2e/kWh)
        """
        self._ensure_db_engine()
        return self._db_engine.get_heat_source_ef(source)

    def get_refrigerant_gwp(
        self, refrigerant: str, gwp_source: GWPSource = GWPSource.AR5
    ) -> Decimal:
        """
        Get refrigerant GWP value.

        Args:
            refrigerant: Refrigerant name
            gwp_source: GWP assessment source

        Returns:
            GWP value (kg CO2e/kg)
        """
        self._ensure_db_engine()
        return self._db_engine.get_refrigerant_gwp(refrigerant, gwp_source)

    def convert_efficiency(
        self,
        value: Decimal,
        from_metric: EfficiencyMetric,
        to_metric: EfficiencyMetric,
    ) -> Decimal:
        """
        Convert between efficiency metrics.

        Args:
            value: Efficiency value
            from_metric: Source metric
            to_metric: Target metric

        Returns:
            Converted value
        """
        self._ensure_db_engine()
        return self._db_engine.convert_efficiency(value, from_metric, to_metric)

    def convert_cooling_units(
        self, value: Decimal, from_unit: CoolingUnit, to_unit: CoolingUnit
    ) -> Decimal:
        """
        Convert between cooling units.

        Args:
            value: Cooling value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value
        """
        self._ensure_db_engine()
        return self._db_engine.convert_cooling_units(value, from_unit, to_unit)

    # ==================== Pipeline Methods ====================

    def run_electric_pipeline(
        self,
        request: ElectricChillerRequest,
        calculate_uncertainty: bool = True,
        check_compliance: bool = True,
        compliance_frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete electric chiller pipeline.

        Args:
            request: Electric chiller request
            calculate_uncertainty: Whether to calculate uncertainty
            check_compliance: Whether to check compliance
            compliance_frameworks: Frameworks to check

        Returns:
            Pipeline result with all outputs
        """
        self._ensure_pipeline_engine()
        return self._pipeline_engine.run_electric_pipeline(
            request=request,
            calculate_uncertainty=calculate_uncertainty,
            check_compliance=check_compliance,
            compliance_frameworks=compliance_frameworks,
        )

    def run_absorption_pipeline(
        self,
        request: AbsorptionCoolingRequest,
        calculate_uncertainty: bool = True,
        check_compliance: bool = True,
        compliance_frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete absorption cooling pipeline.

        Args:
            request: Absorption cooling request
            calculate_uncertainty: Whether to calculate uncertainty
            check_compliance: Whether to check compliance
            compliance_frameworks: Frameworks to check

        Returns:
            Pipeline result with all outputs
        """
        self._ensure_pipeline_engine()
        return self._pipeline_engine.run_absorption_pipeline(
            request=request,
            calculate_uncertainty=calculate_uncertainty,
            check_compliance=check_compliance,
            compliance_frameworks=compliance_frameworks,
        )

    def run_free_cooling_pipeline(
        self,
        request: FreeCoolingRequest,
        calculate_uncertainty: bool = True,
        check_compliance: bool = True,
        compliance_frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete free cooling pipeline.

        Args:
            request: Free cooling request
            calculate_uncertainty: Whether to calculate uncertainty
            check_compliance: Whether to check compliance
            compliance_frameworks: Frameworks to check

        Returns:
            Pipeline result with all outputs
        """
        self._ensure_pipeline_engine()
        return self._pipeline_engine.run_free_cooling_pipeline(
            request=request,
            calculate_uncertainty=calculate_uncertainty,
            check_compliance=check_compliance,
            compliance_frameworks=compliance_frameworks,
        )

    def run_tes_pipeline(
        self,
        request: TESRequest,
        calculate_uncertainty: bool = True,
        check_compliance: bool = True,
        compliance_frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete TES pipeline.

        Args:
            request: TES request
            calculate_uncertainty: Whether to calculate uncertainty
            check_compliance: Whether to check compliance
            compliance_frameworks: Frameworks to check

        Returns:
            Pipeline result with all outputs
        """
        self._ensure_pipeline_engine()
        return self._pipeline_engine.run_tes_pipeline(
            request=request,
            calculate_uncertainty=calculate_uncertainty,
            check_compliance=check_compliance,
            compliance_frameworks=compliance_frameworks,
        )

    def run_district_pipeline(
        self,
        request: DistrictCoolingRequest,
        calculate_uncertainty: bool = True,
        check_compliance: bool = True,
        compliance_frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete district cooling pipeline.

        Args:
            request: District cooling request
            calculate_uncertainty: Whether to calculate uncertainty
            check_compliance: Whether to check compliance
            compliance_frameworks: Frameworks to check

        Returns:
            Pipeline result with all outputs
        """
        self._ensure_pipeline_engine()
        return self._pipeline_engine.run_district_pipeline(
            request=request,
            calculate_uncertainty=calculate_uncertainty,
            check_compliance=check_compliance,
            compliance_frameworks=compliance_frameworks,
        )

    def run_batch(self, request: BatchCoolingRequest) -> BatchCalculationResult:
        """
        Run batch cooling calculations.

        Args:
            request: Batch calculation request

        Returns:
            Batch calculation results
        """
        self._ensure_pipeline_engine()
        return self._pipeline_engine.run_batch(request)

    # ==================== Uncertainty & Compliance Methods ====================

    def quantify_uncertainty(self, request: UncertaintyRequest) -> UncertaintyResult:
        """
        Quantify calculation uncertainty.

        Args:
            request: Uncertainty quantification request

        Returns:
            Uncertainty analysis result
        """
        self._ensure_uncertainty_engine()
        return self._uncertainty_engine.quantify(request)

    def check_compliance(
        self,
        result: CalculationResult,
        frameworks: Optional[List[str]] = None,
    ) -> List[ComplianceCheckResult]:
        """
        Check compliance with frameworks.

        Args:
            result: Calculation result to check
            frameworks: Frameworks to check (None = all)

        Returns:
            List of compliance check results
        """
        self._ensure_compliance_engine()
        return self._compliance_engine.check(result, frameworks)

    # ==================== Aggregation Methods ====================

    def aggregate_results(self, request: AggregationRequest) -> AggregationResult:
        """
        Aggregate multiple calculation results.

        Args:
            request: Aggregation request

        Returns:
            Aggregated result
        """
        self._ensure_pipeline_engine()
        return self._pipeline_engine.aggregate_results(request)

    # ==================== Utility Methods ====================

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all engines.

        Returns:
            Health status dictionary
        """
        health = {
            "service": "CoolingPurchaseService",
            "version": self.get_version(),
            "timestamp": datetime.utcnow().isoformat(),
            "engines": {},
        }

        # Check each engine
        engines = {
            "database": (self._db_engine, CoolingDatabaseEngine),
            "electric_chiller": (self._electric_engine, ElectricChillerCalculatorEngine),
            "absorption_cooling": (
                self._absorption_engine,
                AbsorptionCoolingCalculatorEngine,
            ),
            "district_cooling": (self._district_engine, DistrictCoolingCalculatorEngine),
            "uncertainty": (self._uncertainty_engine, UncertaintyQuantifierEngine),
            "compliance": (self._compliance_engine, ComplianceCheckerEngine),
            "pipeline": (self._pipeline_engine, CoolingPurchasePipelineEngine),
        }

        for name, (instance, cls) in engines.items():
            if instance is not None:
                health["engines"][name] = {
                    "status": "initialized",
                    "available": True,
                }
            elif cls is not None:
                health["engines"][name] = {
                    "status": "not_initialized",
                    "available": True,
                }
            else:
                health["engines"][name] = {
                    "status": "unavailable",
                    "available": False,
                }

        health["overall_status"] = (
            "healthy"
            if all(e["available"] for e in health["engines"].values())
            else "degraded"
        )

        return health

    def get_version(self) -> str:
        """Get service version."""
        return "1.0.0"

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about all engines.

        Returns:
            Engine information dictionary
        """
        info = {
            "service": "CoolingPurchaseService",
            "version": self.get_version(),
            "engines": {},
        }

        # Database engine
        if self._db_engine is not None:
            info["engines"]["database"] = {
                "initialized": True,
                "class": "CoolingDatabaseEngine",
            }

        # Electric chiller engine
        if self._electric_engine is not None:
            info["engines"]["electric_chiller"] = {
                "initialized": True,
                "class": "ElectricChillerCalculatorEngine",
            }

        # Absorption cooling engine
        if self._absorption_engine is not None:
            info["engines"]["absorption_cooling"] = {
                "initialized": True,
                "class": "AbsorptionCoolingCalculatorEngine",
            }

        # District cooling engine
        if self._district_engine is not None:
            info["engines"]["district_cooling"] = {
                "initialized": True,
                "class": "DistrictCoolingCalculatorEngine",
            }

        # Uncertainty engine
        if self._uncertainty_engine is not None:
            info["engines"]["uncertainty"] = {
                "initialized": True,
                "class": "UncertaintyQuantifierEngine",
            }

        # Compliance engine
        if self._compliance_engine is not None:
            info["engines"]["compliance"] = {
                "initialized": True,
                "class": "ComplianceCheckerEngine",
            }

        # Pipeline engine
        if self._pipeline_engine is not None:
            info["engines"]["pipeline"] = {
                "initialized": True,
                "class": "CoolingPurchasePipelineEngine",
            }

        return info


# ==================== Module-Level Functions ====================


def get_service() -> CoolingPurchaseService:
    """
    Get singleton service instance.

    Returns:
        CoolingPurchaseService singleton

    Example:
        >>> service = get_service()
        >>> result = service.calculate_electric_chiller(request)
    """
    return CoolingPurchaseService.get_instance()


def reset_service():
    """
    Reset singleton service (for testing).

    Example:
        >>> reset_service()
        >>> service = get_service()  # Fresh instance
    """
    CoolingPurchaseService.reset()


# ==================== Public API ====================

__all__ = [
    "CoolingPurchaseService",
    "get_service",
    "reset_service",
]
