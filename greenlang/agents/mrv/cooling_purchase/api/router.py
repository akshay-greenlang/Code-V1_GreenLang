"""
FastAPI Router for AGENT-MRV-012 (Cooling Purchase Agent).

This module implements 20 REST endpoints for calculating Scope 2 cooling purchase
emissions across 6 technology types (electric chillers, absorption, district,
free cooling, TES temporal shifting, and batch processing).

Endpoints:
    - 6 calculation endpoints (electric, absorption, district, free-cooling, tes, batch)
    - 2 technology lookup endpoints
    - 3 factor lookup endpoints
    - 4 facility/supplier management endpoints
    - 4 analysis endpoints (uncertainty, compliance, aggregate)
    - 1 health check endpoint

Example:
    >>> from greenlang.agents.mrv.cooling_purchase.api.router import router
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends
from pydantic import BaseModel, Field, validator
from decimal import Decimal
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import uuid
import logging

from greenlang.agents.mrv.cooling_purchase.setup import get_service
from greenlang.agents.mrv.cooling_purchase.models import (
    CoolingTechnology,
    AbsorptionType,
    HeatSource,
    DistrictCoolingRegion,
    FreeCoolingSource,
    TESType,
    TierLevel,
    GWPSource,
    UncertaintySource,
    ComplianceFramework,
)

logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST BODY MODELS
# ============================================================================


class ElectricChillerRequestBody(BaseModel):
    """Request body for electric chiller calculation."""

    cooling_output_kwh_th: Decimal = Field(..., gt=0, description="Cooling output in kWh-thermal")
    technology: CoolingTechnology = Field(..., description="Chiller technology type")
    cop_override: Optional[Decimal] = Field(None, gt=0, description="Override COP value")
    use_iplv: bool = Field(False, description="Use IPLV weighted average instead of rated COP")
    cop_100: Optional[Decimal] = Field(None, gt=0, description="COP at 100% load")
    cop_75: Optional[Decimal] = Field(None, gt=0, description="COP at 75% load")
    cop_50: Optional[Decimal] = Field(None, gt=0, description="COP at 50% load")
    cop_25: Optional[Decimal] = Field(None, gt=0, description="COP at 25% load")
    grid_ef: Decimal = Field(..., gt=0, description="Grid emission factor kg CO2e/kWh")
    auxiliary_pct: Decimal = Field(Decimal("0.15"), ge=0, le=1, description="Auxiliary equipment %")
    tenant_id: str = Field(..., description="Tenant identifier")
    tier: TierLevel = Field(TierLevel.TIER_1, description="Calculation tier")
    gwp_source: GWPSource = Field(GWPSource.AR6, description="GWP source")
    reporting_period: str = Field(..., description="Reporting period YYYY-MM")

    @validator("cop_100", "cop_75", "cop_50", "cop_25")
    def validate_cop_range(cls, v):
        """Validate COP in realistic range."""
        if v is not None and (v < Decimal("0.5") or v > Decimal("20")):
            raise ValueError("COP must be between 0.5 and 20")
        return v


class AbsorptionCoolingRequestBody(BaseModel):
    """Request body for absorption chiller calculation."""

    cooling_output_kwh_th: Decimal = Field(..., gt=0, description="Cooling output in kWh-thermal")
    absorption_type: AbsorptionType = Field(..., description="Absorption chiller type")
    heat_source: HeatSource = Field(..., description="Heat source type")
    cop_override: Optional[Decimal] = Field(None, gt=0, description="Override COP value")
    parasitic_ratio: Decimal = Field(Decimal("0.02"), ge=0, le=0.5, description="Parasitic electric ratio")
    grid_ef: Decimal = Field(..., gt=0, description="Grid emission factor kg CO2e/kWh")
    heat_source_ef_override: Optional[Decimal] = Field(None, ge=0, description="Heat source EF override")
    tenant_id: str = Field(..., description="Tenant identifier")
    tier: TierLevel = Field(TierLevel.TIER_1, description="Calculation tier")
    gwp_source: GWPSource = Field(GWPSource.AR6, description="GWP source")
    reporting_period: str = Field(..., description="Reporting period YYYY-MM")


class DistrictCoolingRequestBody(BaseModel):
    """Request body for district cooling calculation."""

    cooling_output_kwh_th: Decimal = Field(..., gt=0, description="Cooling output in kWh-thermal")
    region: DistrictCoolingRegion = Field(..., description="District cooling region")
    distribution_loss_pct: Decimal = Field(Decimal("0.05"), ge=0, le=0.5, description="Distribution loss %")
    pump_energy_kwh: Decimal = Field(Decimal("0"), ge=0, description="Pump energy kWh")
    grid_ef: Decimal = Field(..., gt=0, description="Grid emission factor kg CO2e/kWh")
    tenant_id: str = Field(..., description="Tenant identifier")
    tier: TierLevel = Field(TierLevel.TIER_1, description="Calculation tier")
    gwp_source: GWPSource = Field(GWPSource.AR6, description="GWP source")
    reporting_period: str = Field(..., description="Reporting period YYYY-MM")


class FreeCoolingRequestBody(BaseModel):
    """Request body for free cooling calculation."""

    cooling_output_kwh_th: Decimal = Field(..., gt=0, description="Cooling output in kWh-thermal")
    source: FreeCoolingSource = Field(..., description="Free cooling source")
    cop_override: Optional[Decimal] = Field(None, gt=0, description="Override COP value")
    grid_ef: Decimal = Field(..., gt=0, description="Grid emission factor kg CO2e/kWh")
    tenant_id: str = Field(..., description="Tenant identifier")
    tier: TierLevel = Field(TierLevel.TIER_1, description="Calculation tier")
    gwp_source: GWPSource = Field(GWPSource.AR6, description="GWP source")
    reporting_period: str = Field(..., description="Reporting period YYYY-MM")


class TESRequestBody(BaseModel):
    """Request body for TES temporal shifting calculation."""

    tes_capacity_kwh_th: Decimal = Field(..., gt=0, description="TES capacity in kWh-thermal")
    tes_type: TESType = Field(..., description="TES type")
    cop_charge: Decimal = Field(..., gt=0, description="COP during charging")
    round_trip_efficiency: Decimal = Field(Decimal("0.85"), gt=0, le=1, description="Round-trip efficiency")
    grid_ef_charge: Decimal = Field(..., gt=0, description="Grid EF during charging kg CO2e/kWh")
    grid_ef_peak: Decimal = Field(..., gt=0, description="Grid EF during peak kg CO2e/kWh")
    cop_peak: Decimal = Field(..., gt=0, description="COP without TES (displaced)")
    tenant_id: str = Field(..., description="Tenant identifier")
    tier: TierLevel = Field(TierLevel.TIER_1, description="Calculation tier")
    gwp_source: GWPSource = Field(GWPSource.AR6, description="GWP source")
    reporting_period: str = Field(..., description="Reporting period YYYY-MM")


class BatchRequestBody(BaseModel):
    """Request body for batch calculation."""

    calculations: List[Dict[str, Any]] = Field(..., description="List of calculation requests")
    tenant_id: str = Field(..., description="Tenant identifier")

    @validator("calculations")
    def validate_calculations(cls, v):
        """Validate calculations list."""
        if len(v) == 0:
            raise ValueError("At least one calculation required")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 calculations per batch")
        return v


class FacilityRequestBody(BaseModel):
    """Request body for facility registration."""

    facility_name: str = Field(..., min_length=1, max_length=255, description="Facility name")
    location: str = Field(..., min_length=1, max_length=255, description="Facility location")
    cooling_technologies: List[CoolingTechnology] = Field(..., description="Installed technologies")
    total_capacity_kwh_th: Decimal = Field(..., gt=0, description="Total cooling capacity")
    tenant_id: str = Field(..., description="Tenant identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SupplierRequestBody(BaseModel):
    """Request body for supplier registration."""

    supplier_name: str = Field(..., min_length=1, max_length=255, description="Supplier name")
    region: DistrictCoolingRegion = Field(..., description="Operating region")
    emission_factor: Decimal = Field(..., gt=0, description="Emission factor kg CO2e/kWh-th")
    renewable_pct: Decimal = Field(Decimal("0"), ge=0, le=1, description="Renewable energy %")
    tenant_id: str = Field(..., description="Tenant identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class UncertaintyRequestBody(BaseModel):
    """Request body for uncertainty analysis."""

    calculation_type: Literal["electric", "absorption", "district", "free-cooling", "tes"] = Field(
        ..., description="Calculation type"
    )
    input_data: Dict[str, Any] = Field(..., description="Input data for calculation")
    uncertainty_sources: List[UncertaintySource] = Field(..., description="Uncertainty sources")
    confidence_level: Decimal = Field(Decimal("0.95"), gt=0, lt=1, description="Confidence level")
    monte_carlo_runs: int = Field(1000, ge=100, le=100000, description="Monte Carlo runs")
    tenant_id: str = Field(..., description="Tenant identifier")


class ComplianceCheckRequestBody(BaseModel):
    """Request body for compliance check."""

    calculation_results: Dict[str, Any] = Field(..., description="Calculation results")
    frameworks: List[ComplianceFramework] = Field(..., description="Frameworks to check")
    tenant_id: str = Field(..., description="Tenant identifier")


class AggregateRequestBody(BaseModel):
    """Request body for aggregation."""

    calculation_ids: List[str] = Field(..., description="Calculation IDs to aggregate")
    aggregation_level: Literal["technology", "facility", "supplier", "period"] = Field(
        ..., description="Aggregation level"
    )
    tenant_id: str = Field(..., description="Tenant identifier")


# ============================================================================
# ROUTER FACTORY
# ============================================================================


def create_router() -> APIRouter:
    """Create and configure the Cooling Purchase router."""
    router = APIRouter(prefix="/api/v1/cooling-purchase", tags=["cooling-purchase"])

    # ========================================================================
    # CALCULATE ENDPOINTS (6)
    # ========================================================================

    @router.post("/calculate/electric")
    async def calculate_electric_chiller(body: ElectricChillerRequestBody) -> Dict[str, Any]:
        """
        Calculate emissions from electric chiller cooling purchase.

        This endpoint calculates Scope 2 emissions from electric chillers using:
        - Rated COP or IPLV weighted average (25%@100% + 50%@75% + 21%@50% + 4%@25%)
        - Grid emission factor
        - Auxiliary equipment (pumps, cooling towers, fans)

        Args:
            body: Electric chiller calculation request

        Returns:
            Dict containing emissions (kg CO2e), electricity consumption (kWh),
            effective COP, auxiliary consumption, provenance hash, and metadata

        Raises:
            HTTPException: If calculation fails
        """
        try:
            service = get_service()
            logger.info(
                f"Electric chiller calculation: {body.cooling_output_kwh_th} kWh-th, "
                f"tech={body.technology}, tenant={body.tenant_id}"
            )

            # Build input dictionary
            input_data = {
                "cooling_output_kwh_th": body.cooling_output_kwh_th,
                "technology": body.technology,
                "cop_override": body.cop_override,
                "use_iplv": body.use_iplv,
                "cop_100": body.cop_100,
                "cop_75": body.cop_75,
                "cop_50": body.cop_50,
                "cop_25": body.cop_25,
                "grid_ef": body.grid_ef,
                "auxiliary_pct": body.auxiliary_pct,
                "tenant_id": body.tenant_id,
                "tier": body.tier,
                "gwp_source": body.gwp_source,
                "reporting_period": body.reporting_period,
            }

            result = service.calculate_electric_chiller(input_data)

            return {
                "status": "success",
                "calculation_id": str(uuid.uuid4()),
                "calculation_type": "electric_chiller",
                "emissions_kg_co2e": float(result["emissions_kg_co2e"]),
                "electricity_consumption_kwh": float(result["electricity_consumption_kwh"]),
                "effective_cop": float(result["effective_cop"]),
                "auxiliary_consumption_kwh": float(result.get("auxiliary_consumption_kwh", 0)),
                "provenance_hash": result["provenance_hash"],
                "tier": body.tier.value,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": result.get("metadata", {}),
            }

        except ValueError as e:
            logger.error(f"Electric chiller calculation validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Electric chiller calculation failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

    @router.post("/calculate/absorption")
    async def calculate_absorption_cooling(body: AbsorptionCoolingRequestBody) -> Dict[str, Any]:
        """
        Calculate emissions from absorption chiller cooling purchase.

        This endpoint calculates Scope 2 emissions from absorption chillers using:
        - Thermal COP (0.6-1.4 depending on type)
        - Heat source emissions (waste heat, natural gas, steam, biomass, solar)
        - Parasitic electric consumption (pumps, controls)

        Args:
            body: Absorption chiller calculation request

        Returns:
            Dict containing emissions (kg CO2e), heat consumption (kWh-th),
            electric consumption (kWh), effective COP, provenance hash, metadata

        Raises:
            HTTPException: If calculation fails
        """
        try:
            service = get_service()
            logger.info(
                f"Absorption cooling calculation: {body.cooling_output_kwh_th} kWh-th, "
                f"type={body.absorption_type}, heat_source={body.heat_source}, tenant={body.tenant_id}"
            )

            input_data = {
                "cooling_output_kwh_th": body.cooling_output_kwh_th,
                "absorption_type": body.absorption_type,
                "heat_source": body.heat_source,
                "cop_override": body.cop_override,
                "parasitic_ratio": body.parasitic_ratio,
                "grid_ef": body.grid_ef,
                "heat_source_ef_override": body.heat_source_ef_override,
                "tenant_id": body.tenant_id,
                "tier": body.tier,
                "gwp_source": body.gwp_source,
                "reporting_period": body.reporting_period,
            }

            result = service.calculate_absorption_cooling(input_data)

            return {
                "status": "success",
                "calculation_id": str(uuid.uuid4()),
                "calculation_type": "absorption_cooling",
                "emissions_kg_co2e": float(result["emissions_kg_co2e"]),
                "heat_consumption_kwh_th": float(result["heat_consumption_kwh_th"]),
                "electric_consumption_kwh": float(result["electric_consumption_kwh"]),
                "effective_cop": float(result["effective_cop"]),
                "heat_source_ef": float(result["heat_source_ef"]),
                "provenance_hash": result["provenance_hash"],
                "tier": body.tier.value,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": result.get("metadata", {}),
            }

        except ValueError as e:
            logger.error(f"Absorption cooling calculation validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Absorption cooling calculation failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

    @router.post("/calculate/district")
    async def calculate_district_cooling(body: DistrictCoolingRequestBody) -> Dict[str, Any]:
        """
        Calculate emissions from district cooling purchase.

        This endpoint calculates Scope 2 emissions from district cooling using:
        - Regional district cooling emission factors (0.02-0.15 kg CO2e/kWh-th)
        - Distribution losses (typically 5-10%)
        - Pump energy consumption

        Args:
            body: District cooling calculation request

        Returns:
            Dict containing emissions (kg CO2e), cooling purchased (kWh-th),
            distribution losses, pump emissions, provenance hash, metadata

        Raises:
            HTTPException: If calculation fails
        """
        try:
            service = get_service()
            logger.info(
                f"District cooling calculation: {body.cooling_output_kwh_th} kWh-th, "
                f"region={body.region}, tenant={body.tenant_id}"
            )

            input_data = {
                "cooling_output_kwh_th": body.cooling_output_kwh_th,
                "region": body.region,
                "distribution_loss_pct": body.distribution_loss_pct,
                "pump_energy_kwh": body.pump_energy_kwh,
                "grid_ef": body.grid_ef,
                "tenant_id": body.tenant_id,
                "tier": body.tier,
                "gwp_source": body.gwp_source,
                "reporting_period": body.reporting_period,
            }

            result = service.calculate_district_cooling(input_data)

            return {
                "status": "success",
                "calculation_id": str(uuid.uuid4()),
                "calculation_type": "district_cooling",
                "emissions_kg_co2e": float(result["emissions_kg_co2e"]),
                "cooling_purchased_kwh_th": float(result["cooling_purchased_kwh_th"]),
                "distribution_losses_kwh_th": float(result["distribution_losses_kwh_th"]),
                "pump_emissions_kg_co2e": float(result["pump_emissions_kg_co2e"]),
                "district_ef": float(result["district_ef"]),
                "provenance_hash": result["provenance_hash"],
                "tier": body.tier.value,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": result.get("metadata", {}),
            }

        except ValueError as e:
            logger.error(f"District cooling calculation validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"District cooling calculation failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

    @router.post("/calculate/free-cooling")
    async def calculate_free_cooling(body: FreeCoolingRequestBody) -> Dict[str, Any]:
        """
        Calculate emissions from free cooling systems.

        This endpoint calculates Scope 2 emissions from free cooling using:
        - High COP values (15-50 depending on source)
        - Minimal electricity for circulation pumps/fans
        - Sources: cooling towers, seawater, groundwater, air economizers

        Args:
            body: Free cooling calculation request

        Returns:
            Dict containing emissions (kg CO2e), electricity consumption (kWh),
            effective COP, provenance hash, metadata

        Raises:
            HTTPException: If calculation fails
        """
        try:
            service = get_service()
            logger.info(
                f"Free cooling calculation: {body.cooling_output_kwh_th} kWh-th, "
                f"source={body.source}, tenant={body.tenant_id}"
            )

            input_data = {
                "cooling_output_kwh_th": body.cooling_output_kwh_th,
                "source": body.source,
                "cop_override": body.cop_override,
                "grid_ef": body.grid_ef,
                "tenant_id": body.tenant_id,
                "tier": body.tier,
                "gwp_source": body.gwp_source,
                "reporting_period": body.reporting_period,
            }

            result = service.calculate_free_cooling(input_data)

            return {
                "status": "success",
                "calculation_id": str(uuid.uuid4()),
                "calculation_type": "free_cooling",
                "emissions_kg_co2e": float(result["emissions_kg_co2e"]),
                "electricity_consumption_kwh": float(result["electricity_consumption_kwh"]),
                "effective_cop": float(result["effective_cop"]),
                "provenance_hash": result["provenance_hash"],
                "tier": body.tier.value,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": result.get("metadata", {}),
            }

        except ValueError as e:
            logger.error(f"Free cooling calculation validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Free cooling calculation failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

    @router.post("/calculate/tes")
    async def calculate_tes_temporal_shifting(body: TESRequestBody) -> Dict[str, Any]:
        """
        Calculate emissions savings from TES temporal shifting.

        This endpoint calculates Scope 2 emissions from TES systems using:
        - Charge during off-peak with lower grid EF
        - Discharge during peak to displace high-EF chiller operation
        - Round-trip efficiency losses
        - Net emissions = charge_emissions - displaced_emissions

        Args:
            body: TES temporal shifting calculation request

        Returns:
            Dict containing net emissions (kg CO2e), charge emissions, displaced emissions,
            emissions savings, round-trip losses, provenance hash, metadata

        Raises:
            HTTPException: If calculation fails
        """
        try:
            service = get_service()
            logger.info(
                f"TES temporal shifting calculation: {body.tes_capacity_kwh_th} kWh-th, "
                f"type={body.tes_type}, tenant={body.tenant_id}"
            )

            input_data = {
                "tes_capacity_kwh_th": body.tes_capacity_kwh_th,
                "tes_type": body.tes_type,
                "cop_charge": body.cop_charge,
                "round_trip_efficiency": body.round_trip_efficiency,
                "grid_ef_charge": body.grid_ef_charge,
                "grid_ef_peak": body.grid_ef_peak,
                "cop_peak": body.cop_peak,
                "tenant_id": body.tenant_id,
                "tier": body.tier,
                "gwp_source": body.gwp_source,
                "reporting_period": body.reporting_period,
            }

            result = service.calculate_tes_temporal_shifting(input_data)

            return {
                "status": "success",
                "calculation_id": str(uuid.uuid4()),
                "calculation_type": "tes_temporal_shifting",
                "net_emissions_kg_co2e": float(result["net_emissions_kg_co2e"]),
                "charge_emissions_kg_co2e": float(result["charge_emissions_kg_co2e"]),
                "displaced_emissions_kg_co2e": float(result["displaced_emissions_kg_co2e"]),
                "emissions_savings_kg_co2e": float(result["emissions_savings_kg_co2e"]),
                "round_trip_losses_kwh_th": float(result["round_trip_losses_kwh_th"]),
                "provenance_hash": result["provenance_hash"],
                "tier": body.tier.value,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": result.get("metadata", {}),
            }

        except ValueError as e:
            logger.error(f"TES temporal shifting calculation validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"TES temporal shifting calculation failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

    @router.post("/calculate/batch")
    async def calculate_batch(body: BatchRequestBody) -> Dict[str, Any]:
        """
        Calculate emissions for batch of cooling calculations.

        This endpoint processes multiple calculations in a single request.
        Supports all calculation types: electric, absorption, district, free-cooling, tes.
        Returns individual results + aggregated totals.

        Args:
            body: Batch calculation request (max 1000 calculations)

        Returns:
            Dict containing individual results, aggregated totals, success count,
            error count, provenance hashes

        Raises:
            HTTPException: If batch processing fails
        """
        try:
            service = get_service()
            logger.info(f"Batch calculation: {len(body.calculations)} calculations, tenant={body.tenant_id}")

            results = []
            errors = []
            total_emissions = Decimal("0")

            for idx, calc in enumerate(body.calculations):
                try:
                    calc_type = calc.get("calculation_type")
                    calc["tenant_id"] = body.tenant_id

                    if calc_type == "electric":
                        result = service.calculate_electric_chiller(calc)
                    elif calc_type == "absorption":
                        result = service.calculate_absorption_cooling(calc)
                    elif calc_type == "district":
                        result = service.calculate_district_cooling(calc)
                    elif calc_type == "free-cooling":
                        result = service.calculate_free_cooling(calc)
                    elif calc_type == "tes":
                        result = service.calculate_tes_temporal_shifting(calc)
                    else:
                        raise ValueError(f"Unknown calculation type: {calc_type}")

                    total_emissions += result["emissions_kg_co2e"]
                    results.append(
                        {
                            "index": idx,
                            "status": "success",
                            "calculation_type": calc_type,
                            "emissions_kg_co2e": float(result["emissions_kg_co2e"]),
                            "provenance_hash": result["provenance_hash"],
                        }
                    )

                except Exception as e:
                    logger.warning(f"Batch calculation {idx} failed: {str(e)}")
                    errors.append({"index": idx, "error": str(e), "calculation": calc})

            return {
                "status": "success",
                "batch_id": str(uuid.uuid4()),
                "total_calculations": len(body.calculations),
                "successful_calculations": len(results),
                "failed_calculations": len(errors),
                "total_emissions_kg_co2e": float(total_emissions),
                "results": results,
                "errors": errors,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Batch calculation failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Batch calculation failed: {str(e)}")

    # ========================================================================
    # TECHNOLOGY LOOKUP ENDPOINTS (2)
    # ========================================================================

    @router.get("/technologies")
    async def list_technologies() -> Dict[str, Any]:
        """
        List all 18 cooling technologies with COP profiles.

        Returns all supported cooling technologies with their default COP values,
        IPLV profiles, and applicability.

        Returns:
            Dict containing list of technologies with COP profiles

        Raises:
            HTTPException: If lookup fails
        """
        try:
            service = get_service()
            logger.info("Listing all cooling technologies")

            technologies = service.list_technologies()

            return {
                "status": "success",
                "total_technologies": len(technologies),
                "technologies": technologies,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Technology listing failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Technology listing failed: {str(e)}")

    @router.get("/technologies/{technology_id}")
    async def get_technology(technology_id: str = Path(..., description="Technology identifier")) -> Dict[str, Any]:
        """
        Get specific cooling technology COP profile.

        Returns detailed COP profile for a specific technology including:
        - Rated COP
        - IPLV profile (100%, 75%, 50%, 25% load points)
        - Technology characteristics
        - Applicability and best practices

        Args:
            technology_id: Technology identifier (e.g., "AIR_COOLED_CHILLER_SCROLL")

        Returns:
            Dict containing technology COP profile and metadata

        Raises:
            HTTPException: If technology not found or lookup fails
        """
        try:
            service = get_service()
            logger.info(f"Getting technology: {technology_id}")

            try:
                tech = CoolingTechnology(technology_id)
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Technology not found: {technology_id}")

            technology = service.get_technology(tech)

            if not technology:
                raise HTTPException(status_code=404, detail=f"Technology not found: {technology_id}")

            return {
                "status": "success",
                "technology": technology,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Technology lookup failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Technology lookup failed: {str(e)}")

    # ========================================================================
    # FACTOR LOOKUP ENDPOINTS (3)
    # ========================================================================

    @router.get("/factors/district/{region}")
    async def get_district_cooling_factor(
        region: str = Path(..., description="District cooling region")
    ) -> Dict[str, Any]:
        """
        Get district cooling emission factor by region.

        Returns emission factor (kg CO2e/kWh-th) for district cooling in a specific region.
        Factors range from 0.02 (Nordic renewables) to 0.15 (Middle East gas).

        Args:
            region: District cooling region (e.g., "NORDIC_EUROPE", "MIDDLE_EAST_GCC")

        Returns:
            Dict containing emission factor and regional metadata

        Raises:
            HTTPException: If region not found or lookup fails
        """
        try:
            service = get_service()
            logger.info(f"Getting district cooling factor: {region}")

            try:
                region_enum = DistrictCoolingRegion(region)
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Region not found: {region}")

            factor = service.get_district_cooling_factor(region_enum)

            if not factor:
                raise HTTPException(status_code=404, detail=f"Factor not found for region: {region}")

            return {
                "status": "success",
                "region": region,
                "emission_factor_kg_co2e_per_kwh_th": float(factor["emission_factor"]),
                "renewable_pct": float(factor.get("renewable_pct", 0)),
                "source": factor.get("source", "Default"),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"District cooling factor lookup failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Factor lookup failed: {str(e)}")

    @router.get("/factors/heat-source/{source}")
    async def get_heat_source_factor(source: str = Path(..., description="Heat source type")) -> Dict[str, Any]:
        """
        Get heat source emission factor.

        Returns emission factor (kg CO2e/kWh-th) for different heat sources used
        in absorption chillers: waste heat, natural gas, steam, biomass, solar.

        Args:
            source: Heat source type (e.g., "NATURAL_GAS", "WASTE_HEAT")

        Returns:
            Dict containing emission factor and source metadata

        Raises:
            HTTPException: If source not found or lookup fails
        """
        try:
            service = get_service()
            logger.info(f"Getting heat source factor: {source}")

            try:
                source_enum = HeatSource(source)
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Heat source not found: {source}")

            factor = service.get_heat_source_factor(source_enum)

            if factor is None:
                raise HTTPException(status_code=404, detail=f"Factor not found for heat source: {source}")

            return {
                "status": "success",
                "heat_source": source,
                "emission_factor_kg_co2e_per_kwh_th": float(factor["emission_factor"]),
                "source": factor.get("source", "Default"),
                "is_renewable": factor.get("is_renewable", False),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Heat source factor lookup failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Factor lookup failed: {str(e)}")

    @router.get("/factors/refrigerants")
    async def list_refrigerant_gwps(
        gwp_source: GWPSource = Query(GWPSource.AR6, description="GWP source (AR5 or AR6)")
    ) -> Dict[str, Any]:
        """
        List all refrigerant GWP values.

        Returns GWP values for all refrigerants used in cooling systems.
        Includes HFCs, HCFCs, natural refrigerants, and low-GWP alternatives.

        Args:
            gwp_source: GWP source (AR5 or AR6)

        Returns:
            Dict containing refrigerant GWPs and metadata

        Raises:
            HTTPException: If lookup fails
        """
        try:
            service = get_service()
            logger.info(f"Listing refrigerant GWPs: source={gwp_source}")

            refrigerants = service.list_refrigerant_gwps(gwp_source)

            return {
                "status": "success",
                "gwp_source": gwp_source.value,
                "total_refrigerants": len(refrigerants),
                "refrigerants": refrigerants,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Refrigerant GWP listing failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Refrigerant GWP listing failed: {str(e)}")

    # ========================================================================
    # FACILITY/SUPPLIER ENDPOINTS (4)
    # ========================================================================

    @router.post("/facilities")
    async def register_facility(body: FacilityRequestBody) -> Dict[str, Any]:
        """
        Register cooling facility.

        Registers a facility with its installed cooling technologies and capacity.
        Used for tracking facility-level emissions and generating reports.

        Args:
            body: Facility registration request

        Returns:
            Dict containing facility_id and registration confirmation

        Raises:
            HTTPException: If registration fails
        """
        try:
            service = get_service()
            logger.info(f"Registering facility: {body.facility_name}, tenant={body.tenant_id}")

            facility_data = {
                "facility_name": body.facility_name,
                "location": body.location,
                "cooling_technologies": [tech.value for tech in body.cooling_technologies],
                "total_capacity_kwh_th": body.total_capacity_kwh_th,
                "tenant_id": body.tenant_id,
                "metadata": body.metadata or {},
            }

            facility = service.register_facility(facility_data)

            return {
                "status": "success",
                "facility_id": facility["facility_id"],
                "facility_name": facility["facility_name"],
                "location": facility["location"],
                "cooling_technologies": facility["cooling_technologies"],
                "total_capacity_kwh_th": float(facility["total_capacity_kwh_th"]),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except ValueError as e:
            logger.error(f"Facility registration validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Facility registration failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Facility registration failed: {str(e)}")

    @router.get("/facilities/{facility_id}")
    async def get_facility(
        facility_id: str = Path(..., description="Facility identifier"), tenant_id: str = Query(..., description="Tenant identifier")
    ) -> Dict[str, Any]:
        """
        Get cooling facility details.

        Returns facility details including installed technologies, capacity,
        and historical emissions data.

        Args:
            facility_id: Facility identifier (UUID)
            tenant_id: Tenant identifier

        Returns:
            Dict containing facility details and metadata

        Raises:
            HTTPException: If facility not found or access denied
        """
        try:
            service = get_service()
            logger.info(f"Getting facility: {facility_id}, tenant={tenant_id}")

            facility = service.get_facility(facility_id, tenant_id)

            if not facility:
                raise HTTPException(status_code=404, detail=f"Facility not found: {facility_id}")

            return {
                "status": "success",
                "facility": facility,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Facility lookup failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Facility lookup failed: {str(e)}")

    @router.post("/suppliers")
    async def register_supplier(body: SupplierRequestBody) -> Dict[str, Any]:
        """
        Register district cooling supplier.

        Registers a district cooling supplier with their emission factor and
        renewable energy percentage. Used for Scope 2 reporting.

        Args:
            body: Supplier registration request

        Returns:
            Dict containing supplier_id and registration confirmation

        Raises:
            HTTPException: If registration fails
        """
        try:
            service = get_service()
            logger.info(f"Registering supplier: {body.supplier_name}, tenant={body.tenant_id}")

            supplier_data = {
                "supplier_name": body.supplier_name,
                "region": body.region.value,
                "emission_factor": body.emission_factor,
                "renewable_pct": body.renewable_pct,
                "tenant_id": body.tenant_id,
                "metadata": body.metadata or {},
            }

            supplier = service.register_supplier(supplier_data)

            return {
                "status": "success",
                "supplier_id": supplier["supplier_id"],
                "supplier_name": supplier["supplier_name"],
                "region": supplier["region"],
                "emission_factor": float(supplier["emission_factor"]),
                "renewable_pct": float(supplier["renewable_pct"]),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except ValueError as e:
            logger.error(f"Supplier registration validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Supplier registration failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Supplier registration failed: {str(e)}")

    @router.get("/suppliers/{supplier_id}")
    async def get_supplier(
        supplier_id: str = Path(..., description="Supplier identifier"), tenant_id: str = Query(..., description="Tenant identifier")
    ) -> Dict[str, Any]:
        """
        Get district cooling supplier details.

        Returns supplier details including emission factor, renewable percentage,
        and service area.

        Args:
            supplier_id: Supplier identifier (UUID)
            tenant_id: Tenant identifier

        Returns:
            Dict containing supplier details and metadata

        Raises:
            HTTPException: If supplier not found or access denied
        """
        try:
            service = get_service()
            logger.info(f"Getting supplier: {supplier_id}, tenant={tenant_id}")

            supplier = service.get_supplier(supplier_id, tenant_id)

            if not supplier:
                raise HTTPException(status_code=404, detail=f"Supplier not found: {supplier_id}")

            return {
                "status": "success",
                "supplier": supplier,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Supplier lookup failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Supplier lookup failed: {str(e)}")

    # ========================================================================
    # ANALYSIS ENDPOINTS (4)
    # ========================================================================

    @router.post("/uncertainty")
    async def run_uncertainty_analysis(body: UncertaintyRequestBody) -> Dict[str, Any]:
        """
        Run uncertainty analysis on cooling calculation.

        Performs Monte Carlo simulation to quantify uncertainty in emissions
        estimates. Sources: activity data, emission factors, COP values, GWP.

        Args:
            body: Uncertainty analysis request

        Returns:
            Dict containing mean, std dev, confidence intervals, sensitivity indices

        Raises:
            HTTPException: If analysis fails
        """
        try:
            service = get_service()
            logger.info(
                f"Running uncertainty analysis: type={body.calculation_type}, "
                f"runs={body.monte_carlo_runs}, tenant={body.tenant_id}"
            )

            analysis_data = {
                "calculation_type": body.calculation_type,
                "input_data": body.input_data,
                "uncertainty_sources": [s.value for s in body.uncertainty_sources],
                "confidence_level": body.confidence_level,
                "monte_carlo_runs": body.monte_carlo_runs,
                "tenant_id": body.tenant_id,
            }

            result = service.run_uncertainty_analysis(analysis_data)

            return {
                "status": "success",
                "analysis_id": str(uuid.uuid4()),
                "calculation_type": body.calculation_type,
                "mean_emissions_kg_co2e": float(result["mean_emissions"]),
                "std_dev_kg_co2e": float(result["std_dev"]),
                "confidence_interval_lower": float(result["ci_lower"]),
                "confidence_interval_upper": float(result["ci_upper"]),
                "confidence_level": float(body.confidence_level),
                "relative_uncertainty_pct": float(result["relative_uncertainty_pct"]),
                "sensitivity_indices": result.get("sensitivity_indices", {}),
                "monte_carlo_runs": body.monte_carlo_runs,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except ValueError as e:
            logger.error(f"Uncertainty analysis validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Uncertainty analysis failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Uncertainty analysis failed: {str(e)}")

    @router.post("/compliance/check")
    async def check_compliance(body: ComplianceCheckRequestBody) -> Dict[str, Any]:
        """
        Check compliance against regulatory frameworks.

        Validates calculation results against 7 frameworks:
        - GHG Protocol Scope 2 Guidance
        - ISO 14064-1:2018
        - CSRD/ESRS E1
        - CDP Climate Change
        - SBTi Corporate Net-Zero Standard
        - TCFD
        - GRI 302/305

        Args:
            body: Compliance check request

        Returns:
            Dict containing compliance status, findings, recommendations

        Raises:
            HTTPException: If check fails
        """
        try:
            service = get_service()
            logger.info(f"Running compliance check: frameworks={body.frameworks}, tenant={body.tenant_id}")

            check_data = {
                "calculation_results": body.calculation_results,
                "frameworks": [f.value for f in body.frameworks],
                "tenant_id": body.tenant_id,
            }

            result = service.check_compliance(check_data)

            return {
                "status": "success",
                "check_id": str(uuid.uuid4()),
                "overall_compliance": result["overall_compliance"],
                "frameworks_checked": len(body.frameworks),
                "frameworks_passed": result["frameworks_passed"],
                "frameworks_failed": result["frameworks_failed"],
                "findings": result["findings"],
                "recommendations": result.get("recommendations", []),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except ValueError as e:
            logger.error(f"Compliance check validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Compliance check failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")

    @router.get("/compliance/frameworks")
    async def list_compliance_frameworks() -> Dict[str, Any]:
        """
        List all compliance frameworks.

        Returns list of 7 supported compliance frameworks with their
        requirements and validation criteria.

        Returns:
            Dict containing list of frameworks with requirements

        Raises:
            HTTPException: If listing fails
        """
        try:
            service = get_service()
            logger.info("Listing compliance frameworks")

            frameworks = service.list_compliance_frameworks()

            return {
                "status": "success",
                "total_frameworks": len(frameworks),
                "frameworks": frameworks,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Framework listing failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Framework listing failed: {str(e)}")

    @router.post("/aggregate")
    async def aggregate_results(body: AggregateRequestBody) -> Dict[str, Any]:
        """
        Aggregate cooling calculation results.

        Aggregates emissions across multiple calculations by:
        - Technology type
        - Facility
        - Supplier
        - Reporting period

        Args:
            body: Aggregation request

        Returns:
            Dict containing aggregated emissions, breakdowns, trends

        Raises:
            HTTPException: If aggregation fails
        """
        try:
            service = get_service()
            logger.info(
                f"Aggregating results: {len(body.calculation_ids)} calculations, "
                f"level={body.aggregation_level}, tenant={body.tenant_id}"
            )

            aggregation_data = {
                "calculation_ids": body.calculation_ids,
                "aggregation_level": body.aggregation_level,
                "tenant_id": body.tenant_id,
            }

            result = service.aggregate_results(aggregation_data)

            return {
                "status": "success",
                "aggregation_id": str(uuid.uuid4()),
                "aggregation_level": body.aggregation_level,
                "total_calculations": len(body.calculation_ids),
                "total_emissions_kg_co2e": float(result["total_emissions"]),
                "breakdown": result["breakdown"],
                "trends": result.get("trends", {}),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except ValueError as e:
            logger.error(f"Aggregation validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Aggregation failed: {str(e)}")

    # ========================================================================
    # HEALTH ENDPOINT (1)
    # ========================================================================

    @router.get("/health")
    async def health_check() -> Dict[str, Any]:
        """
        Health check endpoint.

        Returns service health status and version information.

        Returns:
            Dict containing health status, version, and uptime

        Raises:
            HTTPException: If health check fails
        """
        try:
            service = get_service()
            logger.info("Health check")

            health = service.health_check()

            return {
                "status": "healthy",
                "service": "cooling-purchase-agent",
                "version": health.get("version", "1.0.0"),
                "database": health.get("database", "connected"),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

    return router


# ============================================================================
# MODULE-LEVEL ROUTER
# ============================================================================

router = create_router()
