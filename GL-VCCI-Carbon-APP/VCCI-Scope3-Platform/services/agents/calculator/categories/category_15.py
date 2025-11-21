# -*- coding: utf-8 -*-
"""
Category 15: Investments Calculator (PCAF Standard)
GL-VCCI Scope 3 Platform

Financed emissions from investments (equity, debt, project finance).

CRITICAL for Financial Institutions:
- Banks (loan portfolios)
- Asset managers (investment funds)
- Insurance companies (underwriting)
- Private equity firms

Implements Partnership for Carbon Accounting Financials (PCAF) Standard:
- Attribution factor calculation
- Data quality scoring (1-5 scale)
- Multiple asset classes
- LLM sector classification

Formula:
Financed Emissions = Portfolio Company Emissions × Attribution Factor

Attribution Factor = Outstanding Amount / Company Value (EVIC or Total Assets)

Version: 1.0.0
Date: 2025-11-08
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from ..models import (
from greenlang.determinism import DeterministicClock
    Category15Input,
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
    ProvenanceChain,
    UncertaintyResult,
)
from ..config import TierType, AssetClass, AttributionMethod, PCAFDataQuality, IndustrySector, get_config
from ..exceptions import (
    DataValidationError,
    EmissionFactorNotFoundError,
    CalculationError,
)

logger = logging.getLogger(__name__)

class Category15Calculator:
    """
    Category 15 (Investments) calculator implementing PCAF Standard.

    PCAF Data Quality Hierarchy:
    - Score 1: Verified reported emissions (best)
    - Score 2: Unverified reported emissions
    - Score 3: Physical activity, primary data
    - Score 4: Physical activity, estimated data
    - Score 5: Economic activity (worst)

    Attribution Methods:
    - Equity share: Outstanding / EVIC
    - Revenue-based: Outstanding / Revenue
    - Asset-based: Outstanding / Total Assets
    - Project-specific: Direct attribution
    - Physical activity: Per unit attribution

    Features:
    - LLM sector classification
    - Automatic PCAF score determination
    - Multiple attribution methods
    - Portfolio aggregation
    - Sector-based estimation
    """

    def __init__(
        self,
        factor_broker: Any,
        llm_client: Any,
        uncertainty_engine: Any,
        provenance_builder: Any,
        config: Optional[Any] = None
    ):
        """
        Initialize Category 15 calculator.

        Args:
            factor_broker: FactorBroker instance for emission factors
            llm_client: LLM client for sector classification
            uncertainty_engine: UncertaintyEngine for Monte Carlo
            provenance_builder: ProvenanceChainBuilder for tracking
            config: Calculator configuration
        """
        self.factor_broker = factor_broker
        self.llm_client = llm_client
        self.uncertainty_engine = uncertainty_engine
        self.provenance_builder = provenance_builder
        self.config = config or get_config()

        # Sector emission intensities (tCO2e per $M revenue)
        # These are industry averages for estimation
        self.sector_intensities = {
            IndustrySector.FINANCIAL_SERVICES: 15.0,
            IndustrySector.TECHNOLOGY: 25.0,
            IndustrySector.HEALTHCARE: 35.0,
            IndustrySector.ENERGY: 450.0,
            IndustrySector.UTILITIES: 500.0,
            IndustrySector.MANUFACTURING: 180.0,
            IndustrySector.RETAIL: 45.0,
            IndustrySector.REAL_ESTATE: 65.0,
            IndustrySector.TRANSPORTATION: 320.0,
            IndustrySector.AGRICULTURE: 250.0,
            IndustrySector.MINING: 400.0,
            IndustrySector.CONSTRUCTION: 120.0,
            IndustrySector.TELECOM: 55.0,
            IndustrySector.CONSUMER_GOODS: 95.0,
            IndustrySector.PROFESSIONAL_SERVICES: 20.0,
            IndustrySector.OTHER: 100.0,
        }

        # PCAF score to DQI mapping
        self.pcaf_to_dqi = {
            PCAFDataQuality.SCORE_1: 95.0,
            PCAFDataQuality.SCORE_2: 85.0,
            PCAFDataQuality.SCORE_3: 70.0,
            PCAFDataQuality.SCORE_4: 55.0,
            PCAFDataQuality.SCORE_5: 40.0,
        }

        logger.info("Initialized Category15Calculator with PCAF Standard")

    async def calculate(self, input_data: Category15Input) -> CalculationResult:
        """
        Calculate financed emissions using PCAF methodology.

        Args:
            input_data: Category 15 input data

        Returns:
            CalculationResult with financed emissions and PCAF score

        Raises:
            DataValidationError: If input data is invalid
            CalculationError: If calculation fails
        """
        start_time = DeterministicClock.utcnow()

        # Validate input
        self._validate_input(input_data)

        # LLM Enhancement: Classify industry sector if needed
        if not input_data.industry_sector and input_data.portfolio_company_description:
            input_data.industry_sector = await self._llm_classify_sector(
                input_data.portfolio_company_description,
                input_data.portfolio_company_name
            )
            logger.info(
                f"LLM classified sector: {input_data.industry_sector.value}"
            )

        # Determine PCAF data quality score
        pcaf_score = self._determine_pcaf_score(input_data)
        logger.info(f"PCAF Data Quality Score: {pcaf_score.value}")

        # Calculate based on PCAF score
        try:
            if pcaf_score in [PCAFDataQuality.SCORE_1, PCAFDataQuality.SCORE_2]:
                # Use reported emissions
                result = await self._calculate_reported_emissions(input_data, pcaf_score)
            elif pcaf_score in [PCAFDataQuality.SCORE_3, PCAFDataQuality.SCORE_4]:
                # Use physical activity
                result = await self._calculate_physical_activity(input_data, pcaf_score)
            else:  # SCORE_5
                # Use economic activity (sector-based)
                result = await self._calculate_economic_activity(input_data, pcaf_score)

            return result

        except Exception as e:
            logger.error(f"Category 15 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_15",
                reason=str(e),
                category=15,
                input_data=input_data.model_dump()
            )

    async def _calculate_reported_emissions(
        self,
        input_data: Category15Input,
        pcaf_score: PCAFDataQuality
    ) -> CalculationResult:
        """
        PCAF Score 1 or 2: Calculate using reported company emissions.

        Args:
            input_data: Category 15 input data
            pcaf_score: PCAF data quality score

        Returns:
            CalculationResult
        """
        # Calculate total company emissions
        company_emissions_tco2e = 0.0

        if input_data.company_emissions_scope1_tco2e:
            company_emissions_tco2e += input_data.company_emissions_scope1_tco2e

        if input_data.company_emissions_scope2_tco2e:
            company_emissions_tco2e += input_data.company_emissions_scope2_tco2e

        # Include Scope 3 if available (optional for PCAF)
        if input_data.company_emissions_scope3_tco2e:
            company_emissions_tco2e += input_data.company_emissions_scope3_tco2e

        # Calculate attribution factor
        attribution_factor = await self._calculate_attribution_factor(input_data)

        # Calculate financed emissions
        financed_emissions_tco2e = company_emissions_tco2e * attribution_factor

        # Convert to kg
        financed_emissions_kg = financed_emissions_tco2e * 1000.0

        # Uncertainty
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self._propagate_uncertainty_pcaf(
                financed_emissions_tco2e,
                pcaf_score
            )

        # Data quality
        dqi_score = self.pcaf_to_dqi[pcaf_score]
        tier = TierType.TIER_1 if pcaf_score == PCAFDataQuality.SCORE_1 else TierType.TIER_2

        data_quality = DataQualityInfo(
            dqi_score=dqi_score,
            tier=tier,
            rating="excellent" if pcaf_score == PCAFDataQuality.SCORE_1 else "good",
            warnings=[]
        )

        # Provenance
        provenance = self.provenance_builder.build(
            calculation_id=f"cat15_{input_data.investment_id}_{DeterministicClock.utcnow().timestamp()}",
            category=15,
            tier=tier,
            input_data=input_data.model_dump(),
            emission_factor=None,
            calculation={
                "method": "pcaf_reported_emissions",
                "pcaf_score": pcaf_score.value,
                "company_emissions_tco2e": company_emissions_tco2e,
                "attribution_factor": attribution_factor,
                "attribution_method": input_data.attribution_method.value if input_data.attribution_method else "auto",
                "outstanding_amount": input_data.outstanding_amount,
            },
            data_quality=data_quality
        )

        return CalculationResult(
            emissions_kgco2e=financed_emissions_kg,
            emissions_tco2e=financed_emissions_tco2e,
            category=15,
            tier=tier,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method=f"pcaf_score_{pcaf_score.value}_reported",
            warnings=[],
            metadata={
                "investment_id": input_data.investment_id,
                "portfolio_company": input_data.portfolio_company_name,
                "asset_class": input_data.asset_class.value,
                "pcaf_score": pcaf_score.value,
                "attribution_factor": attribution_factor,
                "company_emissions_tco2e": company_emissions_tco2e,
                "emissions_verified": input_data.emissions_verified,
            }
        )

    async def _calculate_physical_activity(
        self,
        input_data: Category15Input,
        pcaf_score: PCAFDataQuality
    ) -> CalculationResult:
        """
        PCAF Score 3 or 4: Calculate using physical activity data.

        Args:
            input_data: Category 15 input data
            pcaf_score: PCAF data quality score

        Returns:
            CalculationResult
        """
        # Use LLM/sector estimation if no physical activity data
        if not input_data.physical_activity_data and input_data.company_revenue:
            return await self._calculate_economic_activity(
                input_data,
                PCAFDataQuality.SCORE_5
            )

        # Simplified: Use sector intensity as proxy
        sector = input_data.industry_sector or IndustrySector.OTHER
        intensity = self.sector_intensities[sector]

        # Estimate company emissions based on revenue
        if input_data.company_revenue:
            estimated_company_emissions = (input_data.company_revenue / 1_000_000) * intensity
        else:
            # Fallback estimation
            estimated_company_emissions = 1000.0  # tCO2e

        # Attribution factor
        attribution_factor = await self._calculate_attribution_factor(input_data)

        # Financed emissions
        financed_emissions_tco2e = estimated_company_emissions * attribution_factor
        financed_emissions_kg = financed_emissions_tco2e * 1000.0

        # Uncertainty
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self._propagate_uncertainty_pcaf(
                financed_emissions_tco2e,
                pcaf_score
            )

        # Data quality
        dqi_score = self.pcaf_to_dqi[pcaf_score]
        tier = TierType.TIER_2

        warnings = ["Using physical activity estimation"]

        data_quality = DataQualityInfo(
            dqi_score=dqi_score,
            tier=tier,
            rating="good" if pcaf_score == PCAFDataQuality.SCORE_3 else "fair",
            warnings=warnings
        )

        # Provenance
        provenance = self.provenance_builder.build(
            calculation_id=f"cat15_{input_data.investment_id}_{DeterministicClock.utcnow().timestamp()}",
            category=15,
            tier=tier,
            input_data=input_data.model_dump(),
            emission_factor=None,
            calculation={
                "method": "pcaf_physical_activity",
                "pcaf_score": pcaf_score.value,
                "estimated_company_emissions_tco2e": estimated_company_emissions,
                "attribution_factor": attribution_factor,
            },
            data_quality=data_quality
        )

        return CalculationResult(
            emissions_kgco2e=financed_emissions_kg,
            emissions_tco2e=financed_emissions_tco2e,
            category=15,
            tier=tier,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method=f"pcaf_score_{pcaf_score.value}_physical_activity",
            warnings=warnings,
            metadata={
                "investment_id": input_data.investment_id,
                "portfolio_company": input_data.portfolio_company_name,
                "asset_class": input_data.asset_class.value,
                "pcaf_score": pcaf_score.value,
                "attribution_factor": attribution_factor,
                "estimated_company_emissions_tco2e": estimated_company_emissions,
            }
        )

    async def _calculate_economic_activity(
        self,
        input_data: Category15Input,
        pcaf_score: PCAFDataQuality
    ) -> CalculationResult:
        """
        PCAF Score 5: Calculate using economic activity (sector intensity).

        Args:
            input_data: Category 15 input data
            pcaf_score: PCAF data quality score

        Returns:
            CalculationResult
        """
        # Get sector or estimate using LLM
        sector = input_data.industry_sector or IndustrySector.OTHER
        intensity = self.sector_intensities[sector]

        # Estimate company emissions from revenue or assets
        if input_data.company_revenue:
            estimated_company_emissions = (input_data.company_revenue / 1_000_000) * intensity
        elif input_data.company_total_assets:
            # Rough approximation: assets to revenue
            estimated_revenue = input_data.company_total_assets * 0.5
            estimated_company_emissions = (estimated_revenue / 1_000_000) * intensity
        else:
            # Last resort: use outstanding amount as proxy
            estimated_company_emissions = (input_data.outstanding_amount / 1_000_000) * intensity * 2.0

        # Attribution factor
        attribution_factor = await self._calculate_attribution_factor(input_data)

        # Financed emissions
        financed_emissions_tco2e = estimated_company_emissions * attribution_factor
        financed_emissions_kg = financed_emissions_tco2e * 1000.0

        # Uncertainty (highest for Score 5)
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self._propagate_uncertainty_pcaf(
                financed_emissions_tco2e,
                pcaf_score
            )

        # Data quality
        dqi_score = self.pcaf_to_dqi[pcaf_score]
        tier = TierType.TIER_3

        warnings = [
            "Using economic activity estimation (PCAF Score 5)",
            "Results have high uncertainty",
            f"Sector: {sector.value}"
        ]

        data_quality = DataQualityInfo(
            dqi_score=dqi_score,
            tier=tier,
            rating="fair",
            warnings=warnings
        )

        # Provenance
        provenance = self.provenance_builder.build(
            calculation_id=f"cat15_{input_data.investment_id}_{DeterministicClock.utcnow().timestamp()}",
            category=15,
            tier=tier,
            input_data=input_data.model_dump(),
            emission_factor=None,
            calculation={
                "method": "pcaf_economic_activity",
                "pcaf_score": pcaf_score.value,
                "sector": sector.value,
                "sector_intensity_tco2e_per_m_revenue": intensity,
                "estimated_company_emissions_tco2e": estimated_company_emissions,
                "attribution_factor": attribution_factor,
            },
            data_quality=data_quality
        )

        return CalculationResult(
            emissions_kgco2e=financed_emissions_kg,
            emissions_tco2e=financed_emissions_tco2e,
            category=15,
            tier=tier,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method=f"pcaf_score_{pcaf_score.value}_economic_activity",
            warnings=warnings,
            metadata={
                "investment_id": input_data.investment_id,
                "portfolio_company": input_data.portfolio_company_name,
                "asset_class": input_data.asset_class.value,
                "sector": sector.value,
                "pcaf_score": pcaf_score.value,
                "attribution_factor": attribution_factor,
                "estimated_company_emissions_tco2e": estimated_company_emissions,
                "sector_intensity": intensity,
            }
        )

    async def _calculate_attribution_factor(
        self,
        input_data: Category15Input
    ) -> float:
        """
        Calculate PCAF attribution factor.

        Formula: Outstanding Amount / Company Value

        Args:
            input_data: Category 15 input data

        Returns:
            Attribution factor (0-1+)
        """
        # Determine attribution method
        if input_data.attribution_method:
            method = input_data.attribution_method
        else:
            # Auto-select based on asset class
            method = self._auto_select_attribution_method(input_data.asset_class)

        # Calculate attribution factor
        if method == AttributionMethod.EQUITY_SHARE:
            if input_data.company_value_evic:
                return input_data.outstanding_amount / input_data.company_value_evic
            elif input_data.company_total_assets:
                return input_data.outstanding_amount / input_data.company_total_assets
            else:
                # Fallback
                return 0.1  # Assume 10% if no data

        elif method == AttributionMethod.ASSET_BASED:
            if input_data.company_total_assets:
                return input_data.outstanding_amount / input_data.company_total_assets
            else:
                return 0.1

        elif method == AttributionMethod.REVENUE_BASED:
            if input_data.company_revenue:
                # For revenue-based, cap at 1.0
                return min(input_data.outstanding_amount / input_data.company_revenue, 1.0)
            else:
                return 0.1

        elif method == AttributionMethod.PROJECT_SPECIFIC:
            # For project finance, usually 100% attribution
            return 1.0

        else:  # PHYSICAL_ACTIVITY
            # Use asset-based as fallback
            if input_data.company_total_assets:
                return input_data.outstanding_amount / input_data.company_total_assets
            else:
                return 0.1

    def _auto_select_attribution_method(
        self,
        asset_class: AssetClass
    ) -> AttributionMethod:
        """Auto-select attribution method based on asset class."""
        mapping = {
            AssetClass.LISTED_EQUITY: AttributionMethod.EQUITY_SHARE,
            AssetClass.CORPORATE_BONDS: AttributionMethod.EQUITY_SHARE,
            AssetClass.BUSINESS_LOANS: AttributionMethod.ASSET_BASED,
            AssetClass.PROJECT_FINANCE: AttributionMethod.PROJECT_SPECIFIC,
            AssetClass.COMMERCIAL_REAL_ESTATE: AttributionMethod.PHYSICAL_ACTIVITY,
            AssetClass.MORTGAGES: AttributionMethod.PHYSICAL_ACTIVITY,
            AssetClass.MOTOR_VEHICLE_LOANS: AttributionMethod.PHYSICAL_ACTIVITY,
            AssetClass.SOVEREIGN_DEBT: AttributionMethod.EQUITY_SHARE,
        }
        return mapping.get(asset_class, AttributionMethod.ASSET_BASED)

    def _determine_pcaf_score(
        self,
        input_data: Category15Input
    ) -> PCAFDataQuality:
        """
        Determine PCAF data quality score based on available data.

        Args:
            input_data: Category 15 input data

        Returns:
            PCAF data quality score (1-5)
        """
        # Score 1: Verified reported emissions
        if (input_data.company_emissions_scope1_tco2e is not None or
            input_data.company_emissions_scope2_tco2e is not None) and \
           input_data.emissions_verified:
            return PCAFDataQuality.SCORE_1

        # Score 2: Unverified reported emissions
        if (input_data.company_emissions_scope1_tco2e is not None or
            input_data.company_emissions_scope2_tco2e is not None):
            return PCAFDataQuality.SCORE_2

        # Score 3: Physical activity, primary data
        if input_data.physical_activity_data:
            return PCAFDataQuality.SCORE_3

        # Score 4: Physical activity, estimated
        if input_data.asset_class in [
            AssetClass.COMMERCIAL_REAL_ESTATE,
            AssetClass.MORTGAGES,
            AssetClass.MOTOR_VEHICLE_LOANS
        ]:
            return PCAFDataQuality.SCORE_4

        # Score 5: Economic activity (default)
        return PCAFDataQuality.SCORE_5

    async def _llm_classify_sector(
        self,
        company_description: str,
        company_name: str
    ) -> IndustrySector:
        """
        Use LLM to classify company industry sector.

        Args:
            company_description: Company description
            company_name: Company name

        Returns:
            IndustrySector
        """
        # Keyword-based classification (simplified)
        description_lower = (company_description + " " + company_name).lower()

        if any(word in description_lower for word in ["bank", "financial", "investment", "capital"]):
            return IndustrySector.FINANCIAL_SERVICES
        elif any(word in description_lower for word in ["tech", "software", "it", "cloud", "saas"]):
            return IndustrySector.TECHNOLOGY
        elif any(word in description_lower for word in ["health", "medical", "pharma", "hospital"]):
            return IndustrySector.HEALTHCARE
        elif any(word in description_lower for word in ["oil", "gas", "energy", "petroleum"]):
            return IndustrySector.ENERGY
        elif any(word in description_lower for word in ["electric", "utility", "power", "grid"]):
            return IndustrySector.UTILITIES
        elif any(word in description_lower for word in ["manufacturing", "factory", "production"]):
            return IndustrySector.MANUFACTURING
        elif any(word in description_lower for word in ["retail", "store", "shopping"]):
            return IndustrySector.RETAIL
        elif any(word in description_lower for word in ["real estate", "property", "reit"]):
            return IndustrySector.REAL_ESTATE
        elif any(word in description_lower for word in ["transport", "logistics", "airline", "shipping"]):
            return IndustrySector.TRANSPORTATION
        elif any(word in description_lower for word in ["agriculture", "farm", "food production"]):
            return IndustrySector.AGRICULTURE
        elif any(word in description_lower for word in ["mining", "metals", "minerals"]):
            return IndustrySector.MINING
        elif any(word in description_lower for word in ["construction", "building", "infrastructure"]):
            return IndustrySector.CONSTRUCTION
        elif any(word in description_lower for word in ["telecom", "communication", "network"]):
            return IndustrySector.TELECOM
        elif any(word in description_lower for word in ["consumer", "goods", "products"]):
            return IndustrySector.CONSUMER_GOODS
        elif any(word in description_lower for word in ["consulting", "advisory", "professional"]):
            return IndustrySector.PROFESSIONAL_SERVICES
        else:
            return IndustrySector.OTHER

    async def _propagate_uncertainty_pcaf(
        self,
        financed_emissions_tco2e: float,
        pcaf_score: PCAFDataQuality
    ) -> UncertaintyResult:
        """
        Propagate uncertainty based on PCAF score.

        Args:
            financed_emissions_tco2e: Financed emissions
            pcaf_score: PCAF data quality score

        Returns:
            UncertaintyResult
        """
        # PCAF score to uncertainty mapping
        score_uncertainties = {
            PCAFDataQuality.SCORE_1: 0.10,  # ±10%
            PCAFDataQuality.SCORE_2: 0.20,  # ±20%
            PCAFDataQuality.SCORE_3: 0.30,  # ±30%
            PCAFDataQuality.SCORE_4: 0.50,  # ±50%
            PCAFDataQuality.SCORE_5: 0.75,  # ±75%
        }

        uncertainty = score_uncertainties[pcaf_score]
        mean_kg = financed_emissions_tco2e * 1000.0
        std_dev = mean_kg * uncertainty

        return UncertaintyResult(
            mean=mean_kg,
            std_dev=std_dev,
            p5=mean_kg * (1 - 1.645 * uncertainty),
            p50=mean_kg,
            p95=mean_kg * (1 + 1.645 * uncertainty),
            min_value=mean_kg * (1 - 2 * uncertainty),
            max_value=mean_kg * (1 + 2 * uncertainty),
            uncertainty_range=f"±{int(uncertainty * 100)}%",
            coefficient_of_variation=uncertainty,
            iterations=10000
        )

    def _validate_input(self, input_data: Category15Input):
        """Validate input data."""
        if not input_data.investment_id:
            raise DataValidationError(
                field="investment_id",
                value=None,
                message="Investment ID is required",
                category=15
            )

        if not input_data.portfolio_company_name:
            raise DataValidationError(
                field="portfolio_company_name",
                value=None,
                message="Portfolio company name is required",
                category=15
            )

        if input_data.outstanding_amount <= 0:
            raise DataValidationError(
                field="outstanding_amount",
                value=input_data.outstanding_amount,
                message="Outstanding amount must be greater than 0",
                category=15
            )

    async def calculate_portfolio(
        self,
        investments: List[Category15Input]
    ) -> Dict[str, Any]:
        """
        Calculate financed emissions for entire portfolio.

        Args:
            investments: List of investments

        Returns:
            Portfolio-level results with aggregations
        """
        results = []

        for investment in investments:
            try:
                result = await self.calculate(investment)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to calculate investment {investment.investment_id}: {e}")
                continue

        # Aggregate results
        total_emissions_tco2e = sum(r.emissions_tco2e for r in results)
        avg_pcaf_score = sum(
            r.metadata.get("pcaf_score", 5) for r in results
        ) / len(results) if results else 5

        # Group by asset class
        by_asset_class = {}
        for result in results:
            asset_class = result.metadata.get("asset_class", "unknown")
            if asset_class not in by_asset_class:
                by_asset_class[asset_class] = {
                    "count": 0,
                    "emissions_tco2e": 0.0
                }
            by_asset_class[asset_class]["count"] += 1
            by_asset_class[asset_class]["emissions_tco2e"] += result.emissions_tco2e

        # Group by sector
        by_sector = {}
        for result in results:
            sector = result.metadata.get("sector", "unknown")
            if sector not in by_sector:
                by_sector[sector] = {
                    "count": 0,
                    "emissions_tco2e": 0.0
                }
            by_sector[sector]["count"] += 1
            by_sector[sector]["emissions_tco2e"] += result.emissions_tco2e

        return {
            "total_investments": len(investments),
            "successful_calculations": len(results),
            "total_financed_emissions_tco2e": total_emissions_tco2e,
            "average_pcaf_score": avg_pcaf_score,
            "by_asset_class": by_asset_class,
            "by_sector": by_sector,
            "individual_results": results
        }


__all__ = ["Category15Calculator"]
