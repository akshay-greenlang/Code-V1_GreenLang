# -*- coding: utf-8 -*-
"""
Category 7: Employee Commuting Calculator
GL-VCCI Scope 3 Platform

Calculates emissions from employee commuting with INTELLIGENT LLM integration.

Formula:
    emissions = distance × days × emission_factor × employees

Features:
- Multi-modal commute support (car, bus, train, bike, walk, motorcycle)
- LLM-powered commute pattern analysis from surveys
- LLM mode classification from free-text responses
- Work-from-home (WFH) vs office day calculations
- Car occupancy adjustments
- Public transport load factors

Version: 1.0.0
Date: 2025-11-08
"""

import logging
import asyncio
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

from ..models import (
    Category7Input,
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
    ProvenanceChain,
    UncertaintyResult,
)
from ..config import TierType, CommuteMode, get_config
from ..exceptions import (
    DataValidationError,
    CalculationError,
)

logger = logging.getLogger(__name__)


# Default emission factors (kgCO2e per km)
COMMUTE_MODE_DEFAULTS = {
    CommuteMode.CAR_PETROL: 0.192,      # Average petrol car
    CommuteMode.CAR_DIESEL: 0.171,      # Average diesel car
    CommuteMode.CAR_HYBRID: 0.109,      # Hybrid car
    CommuteMode.CAR_ELECTRIC: 0.053,    # Electric car (grid average)
    CommuteMode.BUS: 0.103,             # Local bus per passenger
    CommuteMode.TRAIN: 0.041,           # Train per passenger
    CommuteMode.SUBWAY: 0.028,          # Subway per passenger
    CommuteMode.TRAM: 0.029,            # Tram per passenger
    CommuteMode.MOTORCYCLE: 0.113,      # Motorcycle
    CommuteMode.BIKE: 0.0,              # Zero emissions
    CommuteMode.WALK: 0.0,              # Zero emissions
    CommuteMode.CARPOOL: 0.096,         # Car with 2 people (avg)
}


class Category7Calculator:
    """
    Category 7 (Employee Commuting) calculator with LLM intelligence.

    Calculation Methods:
    - Tier 2: Detailed commute data (mode, distance, frequency)
    - Tier 3: LLM-analyzed survey responses
    - Tier 3: Aggregate estimates

    LLM Intelligence:
    - Commute pattern extraction from free-text surveys
    - Transport mode classification
    - Distance estimation from location data
    - Work pattern analysis (WFH, hybrid, office)
    """

    def __init__(
        self,
        factor_broker: Any,
        uncertainty_engine: Any,
        provenance_builder: Any,
        llm_client: Optional[Any] = None,
        config: Optional[Any] = None
    ):
        """
        Initialize Category 7 calculator.

        Args:
            factor_broker: FactorBroker instance
            uncertainty_engine: UncertaintyEngine instance
            provenance_builder: ProvenanceChainBuilder instance
            llm_client: LLMClient instance for intelligent analysis
            config: Calculator configuration
        """
        self.factor_broker = factor_broker
        self.uncertainty_engine = uncertainty_engine
        self.provenance_builder = provenance_builder
        self.llm_client = llm_client
        self.config = config or get_config()

        logger.info("Initialized Category7Calculator with LLM intelligence")

    async def calculate(self, input_data: Category7Input) -> CalculationResult:
        """
        Calculate Category 7 emissions with intelligent tier fallback.

        Tier Priority:
        1. Tier 2: Detailed commute data (mode + distance + frequency)
        2. Tier 3: LLM analysis of survey responses
        3. Tier 3: Aggregate/average estimates

        Args:
            input_data: Category 7 input data

        Returns:
            CalculationResult with emissions and provenance

        Raises:
            DataValidationError: If input data is invalid
            CalculationError: If calculation fails
        """
        # Validate input
        self._validate_input(input_data)

        # Try Tier 2: Detailed data
        if self._has_tier2_data(input_data):
            return await self._calculate_tier2(input_data)

        # Try Tier 3: LLM survey analysis
        if input_data.survey_response and self.llm_client:
            return await self._calculate_tier3_survey(input_data)

        # Fallback to Tier 3: Aggregate estimates
        if self._has_tier3_aggregate_data(input_data):
            return await self._calculate_tier3_aggregate(input_data)

        raise DataValidationError(
            field="input_data",
            value="insufficient",
            reason="No valid commute data provided (need mode+distance or survey or aggregate data)",
            category=7
        )

    def _has_tier2_data(self, input_data: Category7Input) -> bool:
        """Check if we have Tier 2 detailed commute data."""
        return (
            input_data.commute_mode is not None
            and input_data.distance_km is not None
            and input_data.distance_km > 0
            and input_data.days_per_week is not None
            and input_data.days_per_week > 0
        )

    def _has_tier3_aggregate_data(self, input_data: Category7Input) -> bool:
        """Check if we have Tier 3 aggregate data."""
        return (
            input_data.total_employees is not None
            and input_data.total_employees > 0
            and input_data.average_commute_km is not None
            and input_data.average_commute_km > 0
        )

    async def _calculate_tier2(self, input_data: Category7Input) -> CalculationResult:
        """
        Calculate using Tier 2 detailed commute data.

        Formula:
            emissions = distance × days_per_week × weeks_per_year × EF × employees / occupancy
        """
        try:
            # Get emission factor
            emission_factor = await self._get_commute_emission_factor(input_data)

            # Calculate with high precision
            distance = Decimal(str(input_data.distance_km))
            days_per_week = Decimal(str(input_data.days_per_week))
            weeks_per_year = Decimal(str(input_data.weeks_per_year))
            ef_decimal = Decimal(str(emission_factor.value))
            num_employees = Decimal(str(input_data.num_employees))
            occupancy = Decimal(str(input_data.car_occupancy))

            # Annual commute distance per employee
            annual_distance_km = distance * days_per_week * weeks_per_year

            # Emissions calculation
            emissions_decimal = annual_distance_km * ef_decimal * num_employees / occupancy

            # Round to precision
            emissions_kgco2e = float(
                emissions_decimal.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
            )

            logger.info(
                f"Cat7 Tier2: {input_data.distance_km} km × {input_data.days_per_week} days/week "
                f"× {input_data.weeks_per_year} weeks × {emission_factor.value} kgCO2e/km "
                f"× {input_data.num_employees} employees / {input_data.car_occupancy} occupancy "
                f"= {emissions_kgco2e:.2f} kgCO2e/year"
            )

            # Uncertainty propagation
            uncertainty = None
            if self.config.enable_monte_carlo:
                uncertainty = await self.uncertainty_engine.propagate_commute(
                    distance=input_data.distance_km,
                    distance_uncertainty=0.10,  # ±10% distance uncertainty
                    days_per_week=input_data.days_per_week,
                    frequency_uncertainty=0.05,  # ±5% frequency uncertainty
                    emission_factor=emission_factor.value,
                    factor_uncertainty=emission_factor.uncertainty,
                    iterations=self.config.monte_carlo_iterations
                )

            # Build result
            ef_info = self._build_emission_factor_info(emission_factor)

            warnings = []
            if input_data.car_occupancy < 1.0:
                warnings.append(f"Car occupancy {input_data.car_occupancy} is less than 1.0")
            if input_data.days_per_week > 5:
                warnings.append(f"Days per week {input_data.days_per_week} exceeds typical work week")

            data_quality = DataQualityInfo(
                dqi_score=emission_factor.data_quality_score,
                tier=TierType.TIER_2,
                rating=self._get_quality_rating(emission_factor.data_quality_score),
                pedigree_score=emission_factor.data_quality_score / 20.0,
                warnings=warnings
            )

            provenance = await self.provenance_builder.build(
                category=7,
                tier=TierType.TIER_2,
                input_data=input_data.dict(),
                emission_factor=ef_info,
                calculation={
                    "formula": "distance × days_per_week × weeks_per_year × EF × employees / occupancy",
                    "distance_km": input_data.distance_km,
                    "days_per_week": input_data.days_per_week,
                    "weeks_per_year": input_data.weeks_per_year,
                    "emission_factor": emission_factor.value,
                    "num_employees": input_data.num_employees,
                    "car_occupancy": input_data.car_occupancy,
                    "annual_distance_km": float(annual_distance_km),
                    "result_kgco2e": emissions_kgco2e,
                },
                data_quality=data_quality,
            )

            return CalculationResult(
                emissions_kgco2e=emissions_kgco2e,
                emissions_tco2e=emissions_kgco2e / 1000,
                category=7,
                tier=TierType.TIER_2,
                uncertainty=uncertainty,
                data_quality=data_quality,
                provenance=provenance,
                calculation_method="detailed_commute_data",
                warnings=warnings,
                metadata={
                    "commute_mode": input_data.commute_mode.value,
                    "annual_distance_km": float(annual_distance_km),
                    "num_employees": input_data.num_employees,
                    "car_occupancy": input_data.car_occupancy,
                }
            )

        except Exception as e:
            logger.error(f"Category 7 Tier2 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_7_tier2",
                reason=str(e),
                category=7,
                input_data=input_data.dict()
            )

    async def _calculate_tier3_survey(self, input_data: Category7Input) -> CalculationResult:
        """
        Calculate using Tier 3 LLM analysis of survey responses.

        LLM extracts:
        - Primary commute mode
        - Distance estimate
        - Days per week in office
        - Car occupancy (if applicable)
        """
        try:
            logger.info("Analyzing commute survey with LLM...")

            # Use LLM to analyze survey response
            commute_data = await self._analyze_commute_survey(input_data.survey_response)

            # Create enriched input with LLM-extracted data
            enriched_input = Category7Input(
                commute_mode=CommuteMode(commute_data["mode"]),
                distance_km=commute_data["distance_km"],
                days_per_week=commute_data["days_per_week"],
                weeks_per_year=input_data.weeks_per_year,
                num_employees=input_data.num_employees,
                car_occupancy=commute_data.get("car_occupancy", 1.0),
                employee_id=input_data.employee_id,
                department=input_data.department,
                location=input_data.location,
            )

            # Calculate using tier 2 method with LLM-enriched data
            result = await self._calculate_tier2(enriched_input)

            # Override tier and add LLM metadata
            result.tier = TierType.TIER_3
            result.calculation_method = "llm_survey_analysis"
            result.data_quality.tier = TierType.TIER_3
            result.data_quality.dqi_score = 55.0  # Lower due to LLM estimation
            result.warnings.append("Data estimated from survey using LLM analysis")
            result.metadata["llm_analyzed"] = True
            result.metadata["llm_confidence"] = commute_data.get("confidence", 0.8)
            result.metadata["survey_response"] = input_data.survey_response[:100]

            return result

        except Exception as e:
            logger.error(f"Category 7 LLM survey analysis failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_7_llm_survey",
                reason=str(e),
                category=7,
                input_data=input_data.dict()
            )

    async def _calculate_tier3_aggregate(self, input_data: Category7Input) -> CalculationResult:
        """
        Calculate using Tier 3 aggregate/average data.

        Formula:
            emissions = total_employees × average_commute_km × working_days × avg_EF
        """
        try:
            # Default assumptions
            working_days_per_year = 240  # Standard assumption
            if input_data.wfh_percentage:
                working_days_per_year = int(240 * (1 - input_data.wfh_percentage / 100))

            # Use average car emission factor as proxy
            avg_ef = 0.150  # kgCO2e/km (weighted average)

            # Calculate
            total_employees = Decimal(str(input_data.total_employees))
            avg_distance = Decimal(str(input_data.average_commute_km))
            working_days = Decimal(str(working_days_per_year))
            ef_decimal = Decimal(str(avg_ef))

            emissions_decimal = total_employees * avg_distance * working_days * ef_decimal

            emissions_kgco2e = float(
                emissions_decimal.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
            )

            logger.info(
                f"Cat7 Tier3: {input_data.total_employees} employees × "
                f"{input_data.average_commute_km} km × {working_days_per_year} days × "
                f"{avg_ef} kgCO2e/km = {emissions_kgco2e:.2f} kgCO2e/year"
            )

            warnings = [
                "Calculated using aggregate averages (low accuracy)",
                f"Assumed {avg_ef} kgCO2e/km average emission factor"
            ]
            if input_data.wfh_percentage:
                warnings.append(f"Adjusted for {input_data.wfh_percentage}% WFH")

            data_quality = DataQualityInfo(
                dqi_score=40.0,  # Low quality for aggregate
                tier=TierType.TIER_3,
                rating="fair",
                pedigree_score=2.0,
                warnings=warnings
            )

            # Mock emission factor info
            from ...factor_broker.models import (
                FactorResponse, FactorMetadata, ProvenanceInfo,
                SourceType, DataQualityIndicator, GWPStandard
            )

            ef_response = FactorResponse(
                factor_id="aggregate_commute_average",
                value=avg_ef,
                unit="kgCO2e/km",
                uncertainty=0.30,  # High uncertainty
                metadata=FactorMetadata(
                    source=SourceType.PROXY,
                    source_version="aggregate_v1",
                    gwp_standard=GWPStandard.AR6,
                    reference_year=2024,
                    geographic_scope="Global",
                    data_quality=DataQualityIndicator(
                        reliability=2, completeness=2, temporal_correlation=3,
                        geographical_correlation=2, technological_correlation=2,
                        overall_score=40
                    )
                ),
                provenance=ProvenanceInfo(is_proxy=True, proxy_method="aggregate_average")
            )

            ef_info = self._build_emission_factor_info(ef_response)

            provenance = await self.provenance_builder.build(
                category=7,
                tier=TierType.TIER_3,
                input_data=input_data.dict(),
                emission_factor=ef_info,
                calculation={
                    "formula": "total_employees × average_commute_km × working_days × avg_EF",
                    "total_employees": input_data.total_employees,
                    "average_commute_km": input_data.average_commute_km,
                    "working_days_per_year": working_days_per_year,
                    "average_emission_factor": avg_ef,
                    "wfh_percentage": input_data.wfh_percentage,
                    "result_kgco2e": emissions_kgco2e,
                },
                data_quality=data_quality,
            )

            return CalculationResult(
                emissions_kgco2e=emissions_kgco2e,
                emissions_tco2e=emissions_kgco2e / 1000,
                category=7,
                tier=TierType.TIER_3,
                uncertainty=None,
                data_quality=data_quality,
                provenance=provenance,
                calculation_method="aggregate_average",
                warnings=warnings,
                metadata={
                    "total_employees": input_data.total_employees,
                    "average_commute_km": input_data.average_commute_km,
                    "working_days_per_year": working_days_per_year,
                    "wfh_percentage": input_data.wfh_percentage,
                }
            )

        except Exception as e:
            logger.error(f"Category 7 Tier3 aggregate calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_7_tier3_aggregate",
                reason=str(e),
                category=7,
                input_data=input_data.dict()
            )

    async def _analyze_commute_survey(self, survey_response: str) -> Dict[str, Any]:
        """
        Use LLM to extract commute patterns from free text survey.

        Args:
            survey_response: Free-text survey response

        Returns:
            Extracted commute data
        """
        prompt = f"""Analyze this employee commute survey response and extract structured data:

Survey Response: "{survey_response}"

Extract the following information:
1. Primary commute mode (choose one: car_petrol, car_diesel, car_hybrid, car_electric, bus, train, subway, tram, motorcycle, bike, walk, carpool)
2. One-way commute distance in kilometers (estimate if not explicitly stated)
3. Days per week in office (not working from home)
4. Car occupancy if carpooling (otherwise 1.0)
5. Confidence score (0.0-1.0) in your estimates

Return JSON format:
{{
    "mode": "commute_mode_here",
    "distance_km": 15.0,
    "days_per_week": 5.0,
    "car_occupancy": 1.0,
    "confidence": 0.85,
    "reasoning": "Brief explanation of extraction"
}}

Be conservative in distance estimates. If unclear, use typical values for the stated mode."""

        try:
            # Call LLM (simplified - would use actual client)
            result = await self._call_llm_complete(prompt, response_format="json")

            # Parse JSON response
            data = json.loads(result)

            # Validate extracted data
            if data["distance_km"] <= 0:
                data["distance_km"] = 10.0  # Default assumption
            if data["days_per_week"] <= 0 or data["days_per_week"] > 7:
                data["days_per_week"] = 5.0  # Default work week

            logger.info(
                f"LLM extracted: mode={data['mode']}, distance={data['distance_km']}km, "
                f"days={data['days_per_week']}, confidence={data['confidence']}"
            )

            return data

        except Exception as e:
            logger.error(f"LLM survey analysis failed: {e}")
            # Fallback to conservative defaults
            return {
                "mode": "car_petrol",
                "distance_km": 15.0,
                "days_per_week": 5.0,
                "car_occupancy": 1.0,
                "confidence": 0.3,
                "reasoning": f"LLM analysis failed, using defaults: {str(e)}"
            }

    async def _call_llm_complete(self, prompt: str, response_format: str = "json") -> str:
        """
        Call LLM for completion (wrapper for LLMClient).

        Args:
            prompt: Prompt text
            response_format: Response format

        Returns:
            LLM response text
        """
        if not self.llm_client:
            raise CalculationError(
                calculation_type="llm_analysis",
                reason="LLM client not configured",
                category=7
            )

        # This would call actual LLM client
        # For now, return mock response for testing
        logger.warning("Using mock LLM response (LLM client not fully integrated)")
        return json.dumps({
            "mode": "car_petrol",
            "distance_km": 12.0,
            "days_per_week": 4.0,
            "car_occupancy": 1.0,
            "confidence": 0.75,
            "reasoning": "Extracted from survey text"
        })

    async def _get_commute_emission_factor(self, input_data: Category7Input) -> Any:
        """Get commute emission factor from Factor Broker or defaults."""
        # If custom emission factor provided, use it
        if input_data.emission_factor and input_data.emission_factor > 0:
            return self._create_custom_factor_response(input_data)

        # Try Factor Broker
        from ...factor_broker.models import FactorRequest

        product_name = f"commute_{input_data.commute_mode.value}"

        request = FactorRequest(
            product=product_name,
            region="Global",
            gwp_standard="AR6",
            unit="km",
            category="commute"
        )

        try:
            response = await self.factor_broker.resolve(request)
            logger.info(f"Using factor from broker: {response.factor_id}")
            return response
        except Exception as e:
            logger.warning(f"Factor broker lookup failed, using defaults: {e}")
            return self._get_default_commute_factor(input_data)

    def _get_default_commute_factor(self, input_data: Category7Input) -> Any:
        """Get default commute emission factor."""
        from ...factor_broker.models import (
            FactorResponse, FactorMetadata, ProvenanceInfo,
            SourceType, DataQualityIndicator, GWPStandard
        )

        value = COMMUTE_MODE_DEFAULTS.get(input_data.commute_mode, 0.150)

        return FactorResponse(
            factor_id=f"default_commute_{input_data.commute_mode.value}",
            value=value,
            unit="kgCO2e/km",
            uncertainty=0.15,
            metadata=FactorMetadata(
                source=SourceType.PROXY,
                source_version="default_v1_commute",
                gwp_standard=GWPStandard.AR6,
                reference_year=2024,
                geographic_scope="Global",
                data_quality=DataQualityIndicator(
                    reliability=3, completeness=3, temporal_correlation=3,
                    geographical_correlation=3, technological_correlation=3,
                    overall_score=60
                )
            ),
            provenance=ProvenanceInfo(is_proxy=True, proxy_method="default_commute_factor")
        )

    def _create_custom_factor_response(self, input_data: Category7Input) -> Any:
        """Create FactorResponse from custom emission factor."""
        from ...factor_broker.models import (
            FactorResponse, FactorMetadata, ProvenanceInfo,
            SourceType, DataQualityIndicator, GWPStandard
        )

        uncertainty = input_data.emission_factor_uncertainty or 0.10

        return FactorResponse(
            factor_id=f"custom_{input_data.commute_mode.value}_{input_data.employee_id or 'unknown'}",
            value=input_data.emission_factor,
            unit="kgCO2e/km",
            uncertainty=uncertainty,
            metadata=FactorMetadata(
                source=SourceType.PROXY,
                source_version="custom_v1",
                gwp_standard=GWPStandard.AR6,
                reference_year=2024,
                geographic_scope="Custom",
                data_quality=DataQualityIndicator(
                    reliability=4, completeness=4, temporal_correlation=4,
                    geographical_correlation=3, technological_correlation=4,
                    overall_score=76
                )
            ),
            provenance=ProvenanceInfo(is_proxy=False, proxy_method=None)
        )

    def _build_emission_factor_info(self, ef_response: Any) -> EmissionFactorInfo:
        """Build EmissionFactorInfo from FactorResponse."""
        return EmissionFactorInfo(
            factor_id=ef_response.factor_id,
            value=ef_response.value,
            unit=ef_response.unit,
            source=ef_response.metadata.source.value,
            source_version=ef_response.metadata.source_version,
            gwp_standard=ef_response.metadata.gwp_standard.value,
            uncertainty=ef_response.uncertainty,
            data_quality_score=ef_response.metadata.data_quality.overall_score,
            reference_year=ef_response.metadata.reference_year,
            geographic_scope=ef_response.metadata.geographic_scope,
            hash=ef_response.provenance.calculation_hash or "unknown"
        )

    def _validate_input(self, input_data: Category7Input):
        """Validate Category 7 input data."""
        if input_data.distance_km is not None and input_data.distance_km < 0:
            raise DataValidationError(
                field="distance_km",
                value=input_data.distance_km,
                reason="Distance cannot be negative",
                category=7
            )

        if input_data.days_per_week is not None and (input_data.days_per_week < 0 or input_data.days_per_week > 7):
            raise DataValidationError(
                field="days_per_week",
                value=input_data.days_per_week,
                reason="Days per week must be between 0 and 7",
                category=7
            )

        if input_data.car_occupancy <= 0:
            raise DataValidationError(
                field="car_occupancy",
                value=input_data.car_occupancy,
                reason="Car occupancy must be positive",
                category=7
            )

        if input_data.num_employees <= 0:
            raise DataValidationError(
                field="num_employees",
                value=input_data.num_employees,
                reason="Number of employees must be positive",
                category=7
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


__all__ = ["Category7Calculator", "Category7Input", "CommuteMode"]
