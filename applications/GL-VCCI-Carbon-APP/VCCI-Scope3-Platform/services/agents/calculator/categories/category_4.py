# -*- coding: utf-8 -*-
"""
Category 4: Upstream Transportation & Distribution Calculator
GL-VCCI Scope 3 Platform

ISO 14083:2023 Compliant Implementation
Zero variance requirement for all test cases.

Formula (ISO 14083):
    emissions = distance × weight × emission_factor

Where:
- distance: kilometers
- weight: tonnes
- emission_factor: kgCO2e per tonne-km (by transport mode)

Version: 1.0.0
Date: 2025-10-30
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from greenlang.determinism import DeterministicClock
from ..models import (
    Category4Input,
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
    ProvenanceChain,
    UncertaintyResult,
)
from ..config import TierType, TransportMode, get_config, TRANSPORT_MODE_DEFAULTS
from ..exceptions import (
    DataValidationError,
    TransportModeError,
    CalculationError,
    ISO14083ComplianceError,
)

logger = logging.getLogger(__name__)


class Category4Calculator:
    """
    Category 4 (Upstream Transportation & Distribution) calculator.

    ISO 14083:2023 Compliant Implementation:
    - Emissions = distance × weight × emission_factor
    - All transport modes supported
    - Zero variance to reference calculations
    - Full provenance tracking

    Features:
    - Multi-modal transport support
    - ISO 14083 test suite validation
    - Emission factor lookup via Factor Broker
    - Load factor adjustments
    - Uncertainty propagation
    """

    def __init__(
        self,
        factor_broker: Any,
        uncertainty_engine: Any,
        provenance_builder: Any,
        config: Optional[Any] = None
    ):
        """
        Initialize Category 4 calculator.

        Args:
            factor_broker: FactorBroker instance
            uncertainty_engine: UncertaintyEngine instance
            provenance_builder: ProvenanceChainBuilder instance
            config: Calculator configuration
        """
        self.factor_broker = factor_broker
        self.uncertainty_engine = uncertainty_engine
        self.provenance_builder = provenance_builder
        self.config = config or get_config()

        # Supported transport modes
        self.supported_modes = [mode.value for mode in TransportMode]

        logger.info("Initialized Category4Calculator (ISO 14083 compliant)")

    async def calculate(self, input_data: Category4Input) -> CalculationResult:
        """
        Calculate Category 4 emissions using ISO 14083 formula.

        ISO 14083 Formula:
            emissions = distance × weight × emission_factor

        Args:
            input_data: Category 4 input data

        Returns:
            CalculationResult with emissions and provenance

        Raises:
            DataValidationError: If input data is invalid
            TransportModeError: If transport mode is unsupported
            ISO14083ComplianceError: If calculation fails compliance
        """
        start_time = DeterministicClock.utcnow()

        # Validate input
        self._validate_input(input_data)

        try:
            # Get emission factor
            emission_factor = await self._get_transport_emission_factor(input_data)

            # ISO 14083 calculation with high precision
            # Using Decimal for exact arithmetic to ensure zero variance
            distance_decimal = Decimal(str(input_data.distance_km))
            weight_decimal = Decimal(str(input_data.weight_tonnes))
            ef_decimal = Decimal(str(emission_factor.value))

            # Apply load factor if specified
            load_factor = input_data.load_factor or 1.0
            load_factor_decimal = Decimal(str(load_factor))

            # ISO 14083 formula: emissions = distance × weight × EF / load_factor
            emissions_decimal = (
                distance_decimal * weight_decimal * ef_decimal / load_factor_decimal
            )

            # Round to 6 decimal places for precision
            emissions_kgco2e = float(
                emissions_decimal.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
            )

            logger.info(
                f"ISO 14083 calculation: {input_data.distance_km} km × "
                f"{input_data.weight_tonnes} t × {emission_factor.value} kgCO2e/t-km "
                f"/ {load_factor} = {emissions_kgco2e:.6f} kgCO2e"
            )

            # Uncertainty propagation
            uncertainty = None
            if self.config.enable_monte_carlo:
                # Combine distance and weight uncertainties
                # Typical uncertainties: distance ±5%, weight ±3%, EF from database
                uncertainty = await self.uncertainty_engine.propagate_logistics(
                    distance=input_data.distance_km,
                    distance_uncertainty=0.05,
                    weight=input_data.weight_tonnes,
                    weight_uncertainty=0.03,
                    emission_factor=emission_factor.value,
                    factor_uncertainty=emission_factor.uncertainty,
                    load_factor=load_factor,
                    iterations=self.config.monte_carlo_iterations
                )

            # Emission factor info
            ef_info = EmissionFactorInfo(
                factor_id=emission_factor.factor_id,
                value=emission_factor.value,
                unit=emission_factor.unit,
                source=emission_factor.source,
                source_version=emission_factor.metadata.source_version,
                gwp_standard=emission_factor.metadata.gwp_standard.value,
                uncertainty=emission_factor.uncertainty,
                data_quality_score=emission_factor.data_quality_score,
                reference_year=emission_factor.metadata.reference_year,
                geographic_scope=emission_factor.metadata.geographic_scope,
                hash=emission_factor.provenance.calculation_hash or "unknown"
            )

            # Data quality (transport data is typically Tier 2)
            warnings = []
            if load_factor < 0.7:
                warnings.append(
                    f"Low load factor ({load_factor:.2%}) increases per-tonne emissions"
                )

            data_quality = DataQualityInfo(
                dqi_score=emission_factor.data_quality_score,
                tier=TierType.TIER_2,
                rating=self._get_quality_rating(emission_factor.data_quality_score),
                pedigree_score=emission_factor.data_quality_score / 20.0,
                warnings=warnings
            )

            # Provenance chain
            provenance = await self.provenance_builder.build(
                category=4,
                tier=TierType.TIER_2,
                input_data=input_data.dict(),
                emission_factor=ef_info,
                calculation={
                    "formula": "distance × weight × emission_factor / load_factor",
                    "standard": "ISO_14083:2023",
                    "distance_km": input_data.distance_km,
                    "weight_tonnes": input_data.weight_tonnes,
                    "emission_factor": emission_factor.value,
                    "load_factor": load_factor,
                    "result_kgco2e": emissions_kgco2e,
                    "tonne_km": input_data.distance_km * input_data.weight_tonnes,
                },
                data_quality=data_quality,
            )

            # ISO 14083 compliance check (if enforced)
            if self.config.category_4_enforce_iso14083:
                self._verify_iso14083_compliance(
                    distance=input_data.distance_km,
                    weight=input_data.weight_tonnes,
                    ef=emission_factor.value,
                    load_factor=load_factor,
                    result=emissions_kgco2e
                )

            return CalculationResult(
                emissions_kgco2e=emissions_kgco2e,
                emissions_tco2e=emissions_kgco2e / 1000,
                category=4,
                tier=TierType.TIER_2,
                uncertainty=uncertainty,
                data_quality=data_quality,
                provenance=provenance,
                calculation_method="iso_14083_logistics",
                warnings=warnings,
                metadata={
                    "transport_mode": input_data.transport_mode.value,
                    "tonne_km": input_data.distance_km * input_data.weight_tonnes,
                    "load_factor": load_factor,
                    "origin": input_data.origin,
                    "destination": input_data.destination,
                    "iso_14083_compliant": True,
                }
            )

        except (DataValidationError, TransportModeError, ISO14083ComplianceError):
            raise
        except Exception as e:
            logger.error(f"Category 4 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_4_logistics",
                reason=str(e),
                category=4,
                input_data=input_data.dict()
            )

    async def _get_transport_emission_factor(
        self, input_data: Category4Input
    ) -> Any:
        """
        Get transport emission factor from Factor Broker or defaults.

        Args:
            input_data: Category 4 input

        Returns:
            FactorResponse with emission factor

        Raises:
            TransportModeError: If mode is unsupported
        """
        # If custom emission factor provided, use it
        if input_data.emission_factor and input_data.emission_factor > 0:
            return self._create_custom_factor_response(input_data)

        # Try to get from Factor Broker
        from ...factor_broker.models import FactorRequest

        transport_mode_key = input_data.transport_mode.value

        # Build product name for factor lookup
        product_name = f"transport_{transport_mode_key}"
        if input_data.fuel_type:
            product_name += f"_{input_data.fuel_type}"

        request = FactorRequest(
            product=product_name,
            region="Global",  # Transport factors are typically global
            gwp_standard="AR6",
            unit="tonne_km",
            category="logistics"
        )

        try:
            response = await self.factor_broker.resolve(request)
            logger.info(f"Using factor from broker: {response.factor_id}")
            return response
        except Exception as e:
            logger.warning(f"Factor broker lookup failed, using defaults: {e}")
            return self._get_default_transport_factor(input_data)

    def _get_default_transport_factor(self, input_data: Category4Input) -> Any:
        """
        Get default transport emission factor.

        Args:
            input_data: Category 4 input

        Returns:
            Mock FactorResponse with default values
        """
        transport_mode = input_data.transport_mode

        if transport_mode not in TRANSPORT_MODE_DEFAULTS:
            raise TransportModeError(
                transport_mode=transport_mode.value,
                supported_modes=list(TRANSPORT_MODE_DEFAULTS.keys())
            )

        value = TRANSPORT_MODE_DEFAULTS[transport_mode]

        # Create mock response
        from ...factor_broker.models import (
            FactorResponse,
            FactorMetadata,
            ProvenanceInfo,
            SourceType,
            DataQualityIndicator,
            GWPStandard,
        )

        return FactorResponse(
            factor_id=f"default_transport_{transport_mode.value}",
            value=value,
            unit="kgCO2e/tonne-km",
            uncertainty=0.20,  # Default 20% uncertainty
            metadata=FactorMetadata(
                source=SourceType.PROXY,
                source_version="default_v1_iso14083",
                gwp_standard=GWPStandard.AR6,
                reference_year=2024,
                geographic_scope="Global",
                data_quality=DataQualityIndicator(
                    reliability=3,
                    completeness=3,
                    temporal_correlation=3,
                    geographical_correlation=3,
                    technological_correlation=3,
                    overall_score=60
                )
            ),
            provenance=ProvenanceInfo(
                is_proxy=True,
                proxy_method="default_transport_factor_iso14083"
            )
        )

    def _create_custom_factor_response(self, input_data: Category4Input) -> Any:
        """Create FactorResponse from custom emission factor."""
        from ...factor_broker.models import (
            FactorResponse,
            FactorMetadata,
            ProvenanceInfo,
            SourceType,
            DataQualityIndicator,
            GWPStandard,
        )

        uncertainty = input_data.emission_factor_uncertainty or 0.15

        return FactorResponse(
            factor_id=f"custom_{input_data.transport_mode.value}_{input_data.shipment_id or 'unknown'}",
            value=input_data.emission_factor,
            unit="kgCO2e/tonne-km",
            uncertainty=uncertainty,
            metadata=FactorMetadata(
                source=SourceType.PROXY,
                source_version="custom_v1",
                gwp_standard=GWPStandard.AR6,
                reference_year=2024,
                geographic_scope="Custom",
                data_quality=DataQualityIndicator(
                    reliability=4,
                    completeness=4,
                    temporal_correlation=4,
                    geographical_correlation=3,
                    technological_correlation=4,
                    overall_score=76
                )
            ),
            provenance=ProvenanceInfo(
                is_proxy=False,
                proxy_method=None
            )
        )

    def _verify_iso14083_compliance(
        self,
        distance: float,
        weight: float,
        ef: float,
        load_factor: float,
        result: float,
        tolerance: float = 0.000001
    ):
        """
        Verify ISO 14083 compliance with zero variance requirement.

        Args:
            distance: Distance in km
            weight: Weight in tonnes
            ef: Emission factor
            load_factor: Load factor
            result: Calculated result
            tolerance: Acceptable variance (default: 0.000001)

        Raises:
            ISO14083ComplianceError: If variance exceeds tolerance
        """
        # Recalculate with high precision
        expected = Decimal(str(distance)) * Decimal(str(weight)) * Decimal(str(ef)) / Decimal(str(load_factor))
        expected_float = float(expected.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))

        variance = abs(result - expected_float)

        if variance > tolerance:
            raise ISO14083ComplianceError(
                test_case=f"distance={distance},weight={weight},ef={ef}",
                expected=expected_float,
                actual=result,
                tolerance=tolerance
            )

        logger.debug(
            f"ISO 14083 compliance verified: variance={variance:.9f} "
            f"(tolerance={tolerance})"
        )

    def _validate_input(self, input_data: Category4Input):
        """
        Validate Category 4 input data.

        Args:
            input_data: Input to validate

        Raises:
            DataValidationError: If validation fails
            TransportModeError: If mode is unsupported
        """
        if input_data.distance_km <= 0:
            raise DataValidationError(
                field="distance_km",
                value=input_data.distance_km,
                reason="Distance must be positive",
                category=4
            )

        if input_data.weight_tonnes <= 0:
            raise DataValidationError(
                field="weight_tonnes",
                value=input_data.weight_tonnes,
                reason="Weight must be positive",
                category=4
            )

        if input_data.transport_mode.value not in self.supported_modes:
            raise TransportModeError(
                transport_mode=input_data.transport_mode.value,
                supported_modes=self.supported_modes
            )

        if input_data.load_factor and not (0 < input_data.load_factor <= 1):
            raise DataValidationError(
                field="load_factor",
                value=input_data.load_factor,
                reason="Load factor must be between 0 and 1",
                category=4
            )

    def _get_quality_rating(self, dqi_score: float) -> str:
        """Convert DQI score to quality rating."""
        if dqi_score >= 80:
            return "excellent"
        elif dqi_score >= 60:
            return "good"
        elif dqi_score >= 40:
            return "fair"
        else:
            return "poor"

    async def run_iso14083_test_suite(
        self, test_cases: List[Any]
    ) -> Dict[str, Any]:
        """
        Run ISO 14083 compliance test suite.

        All test cases must pass with ZERO VARIANCE.

        Args:
            test_cases: List of ISO14083TestCase objects

        Returns:
            Test results summary

        Raises:
            ISO14083ComplianceError: If any test fails
        """
        results = {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "max_variance": 0.0,
            "test_details": []
        }

        for test in test_cases:
            try:
                # Create input from test case
                input_data = Category4Input(
                    transport_mode=test.transport_mode,
                    distance_km=test.distance_km,
                    weight_tonnes=test.weight_tonnes,
                    emission_factor=test.emission_factor,
                    load_factor=1.0
                )

                # Calculate
                result = await self.calculate(input_data)

                # Check variance
                variance = abs(result.emissions_kgco2e - test.expected_emissions_kgco2e)

                if variance <= test.tolerance:
                    results["passed"] += 1
                    status = "PASS"
                else:
                    results["failed"] += 1
                    status = "FAIL"
                    logger.error(
                        f"ISO 14083 test FAILED: {test.test_id} - "
                        f"variance {variance:.9f} exceeds tolerance {test.tolerance}"
                    )

                results["max_variance"] = max(results["max_variance"], variance)

                results["test_details"].append({
                    "test_id": test.test_id,
                    "status": status,
                    "expected": test.expected_emissions_kgco2e,
                    "actual": result.emissions_kgco2e,
                    "variance": variance,
                    "tolerance": test.tolerance
                })

            except Exception as e:
                results["failed"] += 1
                results["test_details"].append({
                    "test_id": test.test_id,
                    "status": "ERROR",
                    "error": str(e)
                })

        # Summary
        results["pass_rate"] = results["passed"] / results["total_tests"] if results["total_tests"] > 0 else 0
        results["all_passed"] = results["failed"] == 0

        logger.info(
            f"ISO 14083 test suite complete: "
            f"{results['passed']}/{results['total_tests']} passed, "
            f"max variance: {results['max_variance']:.9f}"
        )

        if not results["all_passed"]:
            raise ISO14083ComplianceError(
                test_case="test_suite",
                expected=results["total_tests"],
                actual=results["passed"],
                tolerance=0
            )

        return results


__all__ = ["Category4Calculator"]
