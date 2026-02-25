# -*- coding: utf-8 -*-
"""
Agricultural Emissions REST API Router - AGENT-MRV-008
=======================================================

20 REST endpoints for the Agricultural Emissions Agent
(GL-MRV-SCOPE1-008).

Prefix: ``/api/v1/agricultural-emissions``

Endpoints:
     1. POST   /calculations                         - Execute single calculation
     2. POST   /calculations/batch                   - Execute batch calculations
     3. GET    /calculations/{id}                    - Get calculation by ID
     4. GET    /calculations                         - List calculations with filters
     5. DELETE /calculations/{id}                    - Delete calculation
     6. POST   /farms                                - Register farm
     7. GET    /farms                                - List farms
     8. PUT    /farms/{id}                           - Update farm metadata
     9. POST   /livestock                            - Register livestock herd
    10. GET    /livestock                            - List livestock herds
    11. PUT    /livestock/{id}                       - Update livestock herd
    12. POST   /cropland-inputs                      - Record cropland inputs
    13. GET    /cropland-inputs                      - List cropland inputs
    14. POST   /rice-fields                          - Register rice field
    15. GET    /rice-fields                          - List rice fields
    16. POST   /compliance/check                     - Run compliance check
    17. GET    /compliance/{id}                      - Get compliance result
    18. POST   /uncertainty                          - Run Monte Carlo analysis
    19. GET    /aggregations                         - Get aggregated emissions
    20. GET    /health                               - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query, Path
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.debug("FastAPI not installed; router unavailable")


# ===================================================================
# Request body models (Pydantic)
# ===================================================================

if FASTAPI_AVAILABLE:

    # --------------------------------------------------------------
    # Calculation models
    # --------------------------------------------------------------

    class SingleCalculationRequest(BaseModel):
        """Request body for a single agricultural emission calculation.

        Supports enteric fermentation (CH4 from livestock digestion),
        manure management (CH4 and N2O), agricultural soils (direct and
        indirect N2O from synthetic fertiliser, organic amendments, crop
        residues, liming, and urea), rice cultivation (CH4 from flooded
        paddies), and field burning of agricultural residues (CH4, N2O,
        CO, NOx) using IPCC Tier 1/2/3 methodologies.
        """

        tenant_id: str = Field(
            default="default",
            description="Owning tenant identifier",
        )
        farm_id: Optional[str] = Field(
            default=None,
            description="Reference to the registered farm",
        )
        emission_source: str = Field(
            ...,
            description="Agricultural emission source "
            "(enteric_fermentation, manure_management, "
            "agricultural_soils, rice_cultivation, "
            "field_burning, liming, urea_application)",
        )
        calculation_method: str = Field(
            default="ipcc_tier_1",
            description="IPCC calculation tier "
            "(ipcc_tier_1, ipcc_tier_2, ipcc_tier_3, "
            "mass_balance, direct_measurement, spend_based)",
        )
        gwp_source: str = Field(
            default="AR6",
            description="GWP source (AR4, AR5, AR6, AR6_20YR)",
        )

        # -- Enteric fermentation parameters ----------------------------
        animal_type: Optional[str] = Field(
            default=None,
            description="Livestock animal type "
            "(dairy_cattle, non_dairy_cattle, buffalo, sheep, "
            "goats, camels, horses, mules_asses, swine, "
            "poultry, deer, elk, alpacas, llamas, rabbits, "
            "fur_bearing_animals, ostrich)",
        )
        head_count: Optional[int] = Field(
            default=None, ge=0,
            description="Number of animals in the herd/flock",
        )
        body_weight_kg: Optional[float] = Field(
            default=None, gt=0,
            description="Average live body weight in kg",
        )
        milk_yield_kg_day: Optional[float] = Field(
            default=None, ge=0,
            description="Daily milk yield per animal in kg/day "
            "(dairy cattle only)",
        )
        fat_pct: Optional[float] = Field(
            default=None, ge=0, le=100,
            description="Milk fat content as a percentage",
        )
        feed_digestibility_pct: Optional[float] = Field(
            default=None, ge=0, le=100,
            description="Feed digestibility (DE%) as a percentage "
            "of gross energy",
        )
        ym_pct: Optional[float] = Field(
            default=None, ge=0, le=25,
            description="Methane conversion factor (Ym%) as a "
            "percentage of gross energy intake",
        )
        activity_coefficient: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Activity coefficient (Ca) for net energy "
            "calculations (0.0=confined, 0.17=grazing, "
            "0.36=hilly terrain)",
        )
        pregnancy_factor: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Pregnancy factor (NEp/NE maintenance) "
            "for breeding females",
        )

        # -- Manure management parameters --------------------------------
        manure_system: Optional[str] = Field(
            default=None,
            description="Manure management system type "
            "(daily_spread, solid_storage, dry_lot, "
            "liquid_slurry, liquid_slurry_crust, "
            "anaerobic_lagoon, pit_storage_below, "
            "pit_storage_above, deep_bedding_no_mixing, "
            "deep_bedding_active_mixing, composting_vessel, "
            "composting_static, composting_passive, "
            "dry_lot_cattle, range_paddock, poultry_with_litter, "
            "poultry_without_litter, biogas_digester)",
        )
        mcf: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Methane correction factor for the manure "
            "management system (0.0-1.0)",
        )
        bo: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Maximum methane producing capacity "
            "(m3 CH4/kg VS)",
        )
        vs_rate: Optional[float] = Field(
            default=None, ge=0.0,
            description="Volatile solids excretion rate "
            "(kg VS/head/day)",
        )
        temperature_c: Optional[float] = Field(
            default=None, ge=-60.0, le=60.0,
            description="Annual average temperature in degrees "
            "Celsius for MCF calculations",
        )
        nex_rate: Optional[float] = Field(
            default=None, ge=0.0,
            description="Nitrogen excretion rate "
            "(kg N/head/year)",
        )

        # -- Agricultural soils parameters --------------------------------
        synthetic_n_kg: Optional[float] = Field(
            default=None, ge=0,
            description="Synthetic fertiliser nitrogen applied "
            "(kg N)",
        )
        organic_n_kg: Optional[float] = Field(
            default=None, ge=0,
            description="Organic nitrogen applied (manure, "
            "compost, sewage sludge) in kg N",
        )
        crop_residue_n_kg: Optional[float] = Field(
            default=None, ge=0,
            description="Crop residue nitrogen returned to soil "
            "(kg N)",
        )
        mineralisation_n_kg: Optional[float] = Field(
            default=None, ge=0,
            description="Nitrogen from mineralisation of soil "
            "organic matter (kg N)",
        )
        pasture_range_n_kg: Optional[float] = Field(
            default=None, ge=0,
            description="Nitrogen deposited by grazing animals "
            "on pasture, range, and paddock (kg N)",
        )
        ef1: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Direct N2O emission factor "
            "(kg N2O-N/kg N input). IPCC default 0.01",
        )
        frac_gasf: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of synthetic N volatilised as "
            "NH3 and NOx (FRAC_GASF). IPCC default 0.10",
        )
        frac_gasm: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of organic N volatilised as "
            "NH3 and NOx (FRAC_GASM). IPCC default 0.20",
        )
        frac_leach: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of all N additions leached or "
            "run off (FRAC_LEACH). IPCC default 0.30",
        )
        ef4: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Indirect N2O EF from atmospheric "
            "deposition (kg N2O-N/kg NH3-N+NOx-N). "
            "IPCC default 0.01",
        )
        ef5: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Indirect N2O EF from leaching/runoff "
            "(kg N2O-N/kg N leached). IPCC default 0.0075",
        )

        # -- Liming and urea parameters ----------------------------------
        limestone_tonnes: Optional[float] = Field(
            default=None, ge=0,
            description="Limestone (CaCO3) applied in metric "
            "tonnes",
        )
        dolomite_tonnes: Optional[float] = Field(
            default=None, ge=0,
            description="Dolomite (CaMg(CO3)2) applied in metric "
            "tonnes",
        )
        urea_tonnes: Optional[float] = Field(
            default=None, ge=0,
            description="Urea (CO(NH2)2) applied in metric tonnes",
        )
        limestone_ef: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Limestone CO2 emission factor. "
            "IPCC default 0.12",
        )
        dolomite_ef: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Dolomite CO2 emission factor. "
            "IPCC default 0.13",
        )
        urea_ef: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Urea CO2 emission factor. "
            "IPCC default 0.20",
        )

        # -- Rice cultivation parameters ----------------------------------
        rice_area_ha: Optional[float] = Field(
            default=None, gt=0,
            description="Rice paddy area in hectares",
        )
        cultivation_days: Optional[int] = Field(
            default=None, gt=0, le=365,
            description="Number of cultivation days per season",
        )
        water_regime: Optional[str] = Field(
            default=None,
            description="Water management regime "
            "(continuously_flooded, "
            "intermittently_flooded_single, "
            "intermittently_flooded_multiple, "
            "rainfed_regular, rainfed_drought_prone, "
            "deep_water, upland)",
        )
        pre_season_flooding: Optional[str] = Field(
            default=None,
            description="Pre-season flooding status "
            "(flooded_long, flooded_short, non_flooded)",
        )
        organic_amendments: Optional[List[Dict[str, Any]]] = Field(
            default=None,
            description="Organic amendments applied to rice "
            "paddy - list of {type, amount_tonnes_ha, "
            "timing} dictionaries",
        )
        rice_baseline_ef: Optional[float] = Field(
            default=None, ge=0.0, le=20.0,
            description="Baseline CH4 emission factor "
            "(kg CH4/ha/day). IPCC default 1.30",
        )

        # -- Field burning parameters ------------------------------------
        crop_type: Optional[str] = Field(
            default=None,
            description="Crop type for field burning "
            "(rice, wheat, maize, sugarcane, barley, "
            "sorghum, millet, oats, rye, cotton, "
            "soybean, groundnut, lentils, other_cereal, "
            "other_pulse, other_oilseed)",
        )
        area_burned_ha: Optional[float] = Field(
            default=None, gt=0,
            description="Area of field burned in hectares",
        )
        crop_yield_tonnes_ha: Optional[float] = Field(
            default=None, ge=0,
            description="Crop yield in tonnes per hectare",
        )
        burn_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of residues actually burned "
            "(0.0-1.0). IPCC default 0.25",
        )
        combustion_factor: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Combustion factor for residue burning "
            "(0.0-1.0). IPCC default 0.80",
        )
        residue_to_crop_ratio: Optional[float] = Field(
            default=None, gt=0,
            description="Residue-to-crop ratio (kg dry matter "
            "residue / kg crop yield)",
        )
        dry_matter_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Dry matter fraction of harvested "
            "residue (0.0-1.0)",
        )

        # -- Climate zone ------------------------------------------------
        climate_zone: Optional[str] = Field(
            default=None,
            description="IPCC climate zone "
            "(TROPICAL_WET, TROPICAL_DRY, "
            "WARM_TEMPERATE_WET, WARM_TEMPERATE_DRY, "
            "COOL_TEMPERATE_WET, COOL_TEMPERATE_DRY, "
            "BOREAL_WET, BOREAL_DRY)",
        )

        # -- Compliance ---------------------------------------------------
        compliance_frameworks: Optional[List[str]] = Field(
            default=None,
            description="Regulatory frameworks for compliance "
            "(GHG_PROTOCOL, IPCC, CSRD, UNFCCC, "
            "US_EPA, UK_SECR, NATIONAL_INVENTORY)",
        )

        # -- Metadata ----------------------------------------------------
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Arbitrary key-value metadata attached "
            "to this calculation",
        )

    class BatchCalculationBody(BaseModel):
        """Request body for batch agricultural emission calculations."""

        calculations: List[Dict[str, Any]] = Field(
            ..., min_length=1, max_length=10000,
            description="List of calculation request dictionaries "
            "(max 10,000 per batch)",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source applied to all calculations "
            "(AR4, AR5, AR6, AR6_20YR)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier applied to all "
            "calculations in the batch",
        )

    # --------------------------------------------------------------
    # Farm models
    # --------------------------------------------------------------

    class FarmBody(BaseModel):
        """Request body for registering an agricultural farm."""

        name: str = Field(
            ..., min_length=1, max_length=500,
            description="Human-readable farm name",
        )
        farm_type: str = Field(
            ...,
            description="Farm type "
            "(dairy, beef, mixed_livestock, arable, "
            "rice_paddy, mixed_farming, horticulture, "
            "plantation, pastoral, intensive_livestock, "
            "organic, smallholder)",
        )
        country_code: str = Field(
            ..., min_length=2, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )
        climate_zone: str = Field(
            ...,
            description="IPCC climate zone "
            "(TROPICAL_WET, TROPICAL_DRY, "
            "WARM_TEMPERATE_WET, WARM_TEMPERATE_DRY, "
            "COOL_TEMPERATE_WET, COOL_TEMPERATE_DRY, "
            "BOREAL_WET, BOREAL_DRY)",
        )
        area_ha: float = Field(
            ..., gt=0,
            description="Total farm area in hectares",
        )
        latitude: float = Field(
            ..., ge=-90, le=90,
            description="WGS84 latitude in decimal degrees",
        )
        longitude: float = Field(
            ..., ge=-180, le=180,
            description="WGS84 longitude in decimal degrees",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier",
        )
        soil_type: Optional[str] = Field(
            default=None,
            description="Predominant soil type "
            "(high_activity_clay, low_activity_clay, "
            "sandy, spodic, volcanic, wetland, organic)",
        )
        elevation_m: Optional[float] = Field(
            default=None,
            description="Elevation above sea level in metres",
        )
        annual_rainfall_mm: Optional[float] = Field(
            default=None, ge=0,
            description="Annual average rainfall in mm",
        )
        irrigation_type: Optional[str] = Field(
            default=None,
            description="Irrigation type "
            "(rainfed, irrigated_surface, irrigated_drip, "
            "irrigated_sprinkler, irrigated_flood, none)",
        )

    class FarmUpdateBody(BaseModel):
        """Request body for updating a farm's attributes."""

        name: Optional[str] = Field(
            default=None, min_length=1, max_length=500,
            description="Human-readable farm name",
        )
        farm_type: Optional[str] = Field(
            default=None,
            description="Farm type",
        )
        climate_zone: Optional[str] = Field(
            default=None,
            description="IPCC climate zone",
        )
        area_ha: Optional[float] = Field(
            default=None, gt=0,
            description="Total farm area in hectares",
        )
        country_code: Optional[str] = Field(
            default=None, min_length=2, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )
        soil_type: Optional[str] = Field(
            default=None,
            description="Predominant soil type",
        )
        elevation_m: Optional[float] = Field(
            default=None,
            description="Elevation above sea level in metres",
        )
        annual_rainfall_mm: Optional[float] = Field(
            default=None, ge=0,
            description="Annual average rainfall in mm",
        )
        irrigation_type: Optional[str] = Field(
            default=None,
            description="Irrigation type",
        )

    # --------------------------------------------------------------
    # Livestock models
    # --------------------------------------------------------------

    class LivestockBody(BaseModel):
        """Request body for registering a livestock herd or flock."""

        farm_id: str = Field(
            ..., min_length=1,
            description="Reference to the registered farm",
        )
        animal_type: str = Field(
            ...,
            description="Livestock animal type "
            "(dairy_cattle, non_dairy_cattle, buffalo, sheep, "
            "goats, camels, horses, mules_asses, swine, "
            "poultry, deer, elk, alpacas, llamas, rabbits, "
            "fur_bearing_animals, ostrich)",
        )
        head_count: int = Field(
            ..., ge=1,
            description="Number of animals in the herd/flock",
        )
        body_weight_kg: float = Field(
            ..., gt=0,
            description="Average live body weight in kg",
        )
        milk_yield_kg_day: Optional[float] = Field(
            default=None, ge=0,
            description="Daily milk yield per animal in kg/day "
            "(dairy cattle only)",
        )
        fat_pct: Optional[float] = Field(
            default=None, ge=0, le=100,
            description="Milk fat content as a percentage",
        )
        feed_digestibility_pct: Optional[float] = Field(
            default=None, ge=0, le=100,
            description="Feed digestibility (DE%) as a percentage",
        )
        ym_pct: Optional[float] = Field(
            default=None, ge=0, le=25,
            description="Methane conversion factor (Ym%)",
        )
        manure_system: Optional[str] = Field(
            default=None,
            description="Manure management system type",
        )
        feed_type: Optional[str] = Field(
            default=None,
            description="Predominant feed type "
            "(pasture, concentrate, mixed, crop_residue, "
            "silage, hay, total_mixed_ration)",
        )
        grazing_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of time spent grazing (0.0-1.0)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    class LivestockUpdateBody(BaseModel):
        """Request body for updating a livestock herd's attributes."""

        head_count: Optional[int] = Field(
            default=None, ge=1,
            description="Number of animals in the herd/flock",
        )
        body_weight_kg: Optional[float] = Field(
            default=None, gt=0,
            description="Average live body weight in kg",
        )
        milk_yield_kg_day: Optional[float] = Field(
            default=None, ge=0,
            description="Daily milk yield per animal in kg/day",
        )
        fat_pct: Optional[float] = Field(
            default=None, ge=0, le=100,
            description="Milk fat content as a percentage",
        )
        feed_digestibility_pct: Optional[float] = Field(
            default=None, ge=0, le=100,
            description="Feed digestibility (DE%) as a percentage",
        )
        ym_pct: Optional[float] = Field(
            default=None, ge=0, le=25,
            description="Methane conversion factor (Ym%)",
        )
        manure_system: Optional[str] = Field(
            default=None,
            description="Manure management system type",
        )
        feed_type: Optional[str] = Field(
            default=None,
            description="Predominant feed type",
        )
        grazing_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of time spent grazing (0.0-1.0)",
        )

    # --------------------------------------------------------------
    # Cropland input model
    # --------------------------------------------------------------

    class CroplandInputBody(BaseModel):
        """Request body for recording cropland nitrogen and liming inputs."""

        farm_id: str = Field(
            ..., min_length=1,
            description="Reference to the registered farm",
        )
        reporting_year: Optional[int] = Field(
            default=None, ge=1990, le=2100,
            description="Reporting year for the cropland inputs",
        )
        synthetic_n_kg: Optional[float] = Field(
            default=None, ge=0,
            description="Synthetic fertiliser nitrogen applied "
            "(kg N)",
        )
        organic_n_kg: Optional[float] = Field(
            default=None, ge=0,
            description="Organic nitrogen applied (manure, "
            "compost, sewage sludge) in kg N",
        )
        crop_residue_n_kg: Optional[float] = Field(
            default=None, ge=0,
            description="Crop residue nitrogen returned to soil "
            "(kg N)",
        )
        mineralisation_n_kg: Optional[float] = Field(
            default=None, ge=0,
            description="Nitrogen from mineralisation of soil "
            "organic matter (kg N)",
        )
        limestone_tonnes: Optional[float] = Field(
            default=None, ge=0,
            description="Limestone (CaCO3) applied in metric "
            "tonnes",
        )
        dolomite_tonnes: Optional[float] = Field(
            default=None, ge=0,
            description="Dolomite (CaMg(CO3)2) applied in metric "
            "tonnes",
        )
        urea_tonnes: Optional[float] = Field(
            default=None, ge=0,
            description="Urea (CO(NH2)2) applied in metric tonnes",
        )
        crop_type: Optional[str] = Field(
            default=None,
            description="Crop type grown on this land",
        )
        area_ha: Optional[float] = Field(
            default=None, gt=0,
            description="Cropland area in hectares",
        )
        tillage_practice: Optional[str] = Field(
            default=None,
            description="Tillage practice "
            "(conventional, reduced, no_till)",
        )
        notes: Optional[str] = Field(
            default=None, max_length=2000,
            description="Optional notes about the cropland inputs",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    # --------------------------------------------------------------
    # Rice field model
    # --------------------------------------------------------------

    class RiceFieldBody(BaseModel):
        """Request body for registering a rice paddy field."""

        farm_id: str = Field(
            ..., min_length=1,
            description="Reference to the registered farm",
        )
        area_ha: float = Field(
            ..., gt=0,
            description="Rice paddy area in hectares",
        )
        cultivation_days: int = Field(
            ..., gt=0, le=365,
            description="Number of cultivation days per season",
        )
        water_regime: str = Field(
            ...,
            description="Water management regime "
            "(continuously_flooded, "
            "intermittently_flooded_single, "
            "intermittently_flooded_multiple, "
            "rainfed_regular, rainfed_drought_prone, "
            "deep_water, upland)",
        )
        pre_season_flooding: Optional[str] = Field(
            default=None,
            description="Pre-season flooding status "
            "(flooded_long, flooded_short, non_flooded)",
        )
        organic_amendments: Optional[List[Dict[str, Any]]] = Field(
            default=None,
            description="Organic amendments applied - list of "
            "{type, amount_tonnes_ha, timing} dictionaries",
        )
        rice_variety: Optional[str] = Field(
            default=None,
            description="Rice variety or cultivar name",
        )
        seasons_per_year: Optional[int] = Field(
            default=None, ge=1, le=3,
            description="Number of rice seasons per year",
        )
        soil_type: Optional[str] = Field(
            default=None,
            description="Paddy soil type",
        )
        baseline_ef: Optional[float] = Field(
            default=None, ge=0.0, le=20.0,
            description="Baseline CH4 emission factor "
            "(kg CH4/ha/day). IPCC default 1.30",
        )
        notes: Optional[str] = Field(
            default=None, max_length=2000,
            description="Optional notes about the rice field",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    # --------------------------------------------------------------
    # Field burning model
    # --------------------------------------------------------------

    class FieldBurningBody(BaseModel):
        """Request body for recording a field burning event."""

        farm_id: str = Field(
            ..., min_length=1,
            description="Reference to the registered farm",
        )
        crop_type: str = Field(
            ...,
            description="Crop type whose residues are burned "
            "(rice, wheat, maize, sugarcane, barley, "
            "sorghum, millet, oats, rye, cotton, "
            "soybean, groundnut, lentils, other_cereal, "
            "other_pulse, other_oilseed)",
        )
        area_burned_ha: float = Field(
            ..., gt=0,
            description="Area of field burned in hectares",
        )
        crop_yield_tonnes_ha: float = Field(
            ..., ge=0,
            description="Crop yield in tonnes per hectare",
        )
        burn_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of residues actually burned "
            "(0.0-1.0). IPCC default 0.25",
        )
        combustion_factor: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Combustion factor (0.0-1.0). "
            "IPCC default 0.80",
        )
        residue_to_crop_ratio: Optional[float] = Field(
            default=None, gt=0,
            description="Residue-to-crop ratio",
        )
        dry_matter_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Dry matter fraction of residue (0.0-1.0)",
        )
        burn_date: Optional[str] = Field(
            default=None,
            description="Date of the burning event (ISO-8601)",
        )
        notes: Optional[str] = Field(
            default=None, max_length=2000,
            description="Optional notes about the burning event",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    # --------------------------------------------------------------
    # Compliance & uncertainty models
    # --------------------------------------------------------------

    class ComplianceCheckBody(BaseModel):
        """Request body for a compliance check."""

        calculation_id: str = Field(
            default="",
            description="ID of a previous calculation (optional)",
        )
        frameworks: List[str] = Field(
            default_factory=list,
            description="Frameworks to check "
            "(empty = all frameworks). "
            "Options: GHG_PROTOCOL, IPCC, CSRD, "
            "UNFCCC, US_EPA, UK_SECR, NATIONAL_INVENTORY",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Tenant identifier for scoping",
        )

    class UncertaintyBody(BaseModel):
        """Request body for Monte Carlo uncertainty analysis."""

        calculation_id: str = Field(
            ..., description="ID of a previous calculation",
        )
        method: str = Field(
            default="monte_carlo",
            description="Uncertainty analysis method "
            "(monte_carlo, analytical, error_propagation)",
        )
        iterations: int = Field(
            default=5000, gt=0, le=1_000_000,
            description="Monte Carlo iterations",
        )
        seed: int = Field(
            default=42, ge=0,
            description="Random seed for reproducibility",
        )
        confidence_level: float = Field(
            default=95.0, gt=0, lt=100,
            description="Confidence level percentage",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Agricultural Emissions FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints.

    Raises:
        RuntimeError: If FastAPI is not installed.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the agricultural emissions router"
        )

    router = APIRouter(
        prefix="/api/v1/agricultural-emissions",
        tags=["Agricultural Emissions"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the AgriculturalEmissionsService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.agricultural_emissions.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Agricultural Emissions service "
                "not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculations - Execute single calculation
    # ==================================================================

    @router.post("/calculations", status_code=201)
    async def create_calculation(
        body: SingleCalculationRequest,
    ) -> Dict[str, Any]:
        """Execute a single agricultural emission calculation.

        Computes GHG emissions (CO2, CH4, N2O) for an agricultural
        emission source using the specified IPCC tier methodology.
        Supports enteric fermentation, manure management, agricultural
        soils (direct and indirect N2O), liming, urea application,
        rice cultivation, and field burning of crop residues.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "tenant_id": body.tenant_id,
            "emission_source": body.emission_source,
            "calculation_method": body.calculation_method,
            "gwp_source": body.gwp_source,
        }

        # Optional farm reference
        if body.farm_id is not None:
            request_data["farm_id"] = body.farm_id

        # Enteric fermentation parameters
        if body.animal_type is not None:
            request_data["animal_type"] = body.animal_type
        if body.head_count is not None:
            request_data["head_count"] = body.head_count
        if body.body_weight_kg is not None:
            request_data["body_weight_kg"] = body.body_weight_kg
        if body.milk_yield_kg_day is not None:
            request_data["milk_yield_kg_day"] = (
                body.milk_yield_kg_day
            )
        if body.fat_pct is not None:
            request_data["fat_pct"] = body.fat_pct
        if body.feed_digestibility_pct is not None:
            request_data["feed_digestibility_pct"] = (
                body.feed_digestibility_pct
            )
        if body.ym_pct is not None:
            request_data["ym_pct"] = body.ym_pct
        if body.activity_coefficient is not None:
            request_data["activity_coefficient"] = (
                body.activity_coefficient
            )
        if body.pregnancy_factor is not None:
            request_data["pregnancy_factor"] = body.pregnancy_factor

        # Manure management parameters
        if body.manure_system is not None:
            request_data["manure_system"] = body.manure_system
        if body.mcf is not None:
            request_data["mcf"] = body.mcf
        if body.bo is not None:
            request_data["bo"] = body.bo
        if body.vs_rate is not None:
            request_data["vs_rate"] = body.vs_rate
        if body.temperature_c is not None:
            request_data["temperature_c"] = body.temperature_c
        if body.nex_rate is not None:
            request_data["nex_rate"] = body.nex_rate

        # Agricultural soils parameters
        if body.synthetic_n_kg is not None:
            request_data["synthetic_n_kg"] = body.synthetic_n_kg
        if body.organic_n_kg is not None:
            request_data["organic_n_kg"] = body.organic_n_kg
        if body.crop_residue_n_kg is not None:
            request_data["crop_residue_n_kg"] = (
                body.crop_residue_n_kg
            )
        if body.mineralisation_n_kg is not None:
            request_data["mineralisation_n_kg"] = (
                body.mineralisation_n_kg
            )
        if body.pasture_range_n_kg is not None:
            request_data["pasture_range_n_kg"] = (
                body.pasture_range_n_kg
            )
        if body.ef1 is not None:
            request_data["ef1"] = body.ef1
        if body.frac_gasf is not None:
            request_data["frac_gasf"] = body.frac_gasf
        if body.frac_gasm is not None:
            request_data["frac_gasm"] = body.frac_gasm
        if body.frac_leach is not None:
            request_data["frac_leach"] = body.frac_leach
        if body.ef4 is not None:
            request_data["ef4"] = body.ef4
        if body.ef5 is not None:
            request_data["ef5"] = body.ef5

        # Liming and urea parameters
        if body.limestone_tonnes is not None:
            request_data["limestone_tonnes"] = body.limestone_tonnes
        if body.dolomite_tonnes is not None:
            request_data["dolomite_tonnes"] = body.dolomite_tonnes
        if body.urea_tonnes is not None:
            request_data["urea_tonnes"] = body.urea_tonnes
        if body.limestone_ef is not None:
            request_data["limestone_ef"] = body.limestone_ef
        if body.dolomite_ef is not None:
            request_data["dolomite_ef"] = body.dolomite_ef
        if body.urea_ef is not None:
            request_data["urea_ef"] = body.urea_ef

        # Rice cultivation parameters
        if body.rice_area_ha is not None:
            request_data["rice_area_ha"] = body.rice_area_ha
        if body.cultivation_days is not None:
            request_data["cultivation_days"] = body.cultivation_days
        if body.water_regime is not None:
            request_data["water_regime"] = body.water_regime
        if body.pre_season_flooding is not None:
            request_data["pre_season_flooding"] = (
                body.pre_season_flooding
            )
        if body.organic_amendments is not None:
            request_data["organic_amendments"] = (
                body.organic_amendments
            )
        if body.rice_baseline_ef is not None:
            request_data["rice_baseline_ef"] = body.rice_baseline_ef

        # Field burning parameters
        if body.crop_type is not None:
            request_data["crop_type"] = body.crop_type
        if body.area_burned_ha is not None:
            request_data["area_burned_ha"] = body.area_burned_ha
        if body.crop_yield_tonnes_ha is not None:
            request_data["crop_yield_tonnes_ha"] = (
                body.crop_yield_tonnes_ha
            )
        if body.burn_fraction is not None:
            request_data["burn_fraction"] = body.burn_fraction
        if body.combustion_factor is not None:
            request_data["combustion_factor"] = (
                body.combustion_factor
            )
        if body.residue_to_crop_ratio is not None:
            request_data["residue_to_crop_ratio"] = (
                body.residue_to_crop_ratio
            )
        if body.dry_matter_fraction is not None:
            request_data["dry_matter_fraction"] = (
                body.dry_matter_fraction
            )

        # Climate zone
        if body.climate_zone is not None:
            request_data["climate_zone"] = body.climate_zone

        # Compliance frameworks
        if body.compliance_frameworks is not None:
            request_data["compliance_frameworks"] = (
                body.compliance_frameworks
            )

        # Metadata
        if body.metadata is not None:
            request_data["metadata"] = body.metadata

        try:
            result = svc.calculate(request_data)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_calculation failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /calculations/batch - Execute batch calculations
    # ==================================================================

    @router.post("/calculations/batch", status_code=201)
    async def create_batch_calculation(
        body: BatchCalculationBody,
    ) -> Dict[str, Any]:
        """Execute batch agricultural emission calculations.

        Processes up to 10,000 calculation requests in a single batch.
        Each item in the ``calculations`` list follows the same schema
        as the single calculate endpoint. Optionally applies a shared
        ``gwp_source`` and ``tenant_id`` to all items.
        """
        svc = _get_service()

        if len(body.calculations) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Batch size exceeds maximum of 10,000 "
                "calculations",
            )

        try:
            result = svc.calculate_batch(
                body.calculations,
                gwp_source=body.gwp_source,
                tenant_id=body.tenant_id,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_batch_calculation failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. GET /calculations/{id} - Get calculation by ID
    # ==================================================================

    @router.get("/calculations/{calc_id}", status_code=200)
    async def get_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get an agricultural calculation result by its unique ID.

        Returns the full calculation result including emissions
        breakdown by gas (CO2, CH4, N2O), emission source details,
        IPCC tier methodology used, and provenance hash.
        """
        svc = _get_service()

        for calc in svc._calculations:
            if calc.get("calculation_id") == calc_id:
                return calc

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calc_id} not found",
        )

    # ==================================================================
    # 4. GET /calculations - List calculations with filters
    # ==================================================================

    @router.get("/calculations", status_code=200)
    async def list_calculations(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        emission_source: Optional[str] = Query(
            None,
            description="Filter by emission source "
            "(enteric_fermentation, manure_management, "
            "agricultural_soils, rice_cultivation, "
            "field_burning, liming, urea_application)",
        ),
        animal_type: Optional[str] = Query(
            None,
            description="Filter by animal type "
            "(dairy_cattle, non_dairy_cattle, buffalo, "
            "sheep, goats, swine, poultry, etc.)",
        ),
        farm_id: Optional[str] = Query(
            None,
            description="Filter by farm identifier",
        ),
        from_date: Optional[str] = Query(
            None,
            description="Filter calculations from this date "
            "(ISO-8601)",
        ),
        to_date: Optional[str] = Query(
            None,
            description="Filter calculations up to this date "
            "(ISO-8601)",
        ),
    ) -> Dict[str, Any]:
        """List agricultural calculation results with pagination.

        Supports filtering by tenant, emission source, animal type,
        farm, and date range. Returns paginated results with total
        count.
        """
        svc = _get_service()
        all_calcs = list(svc._calculations)

        # Apply filters
        if tenant_id is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("tenant_id") == tenant_id
            ]
        if emission_source is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("emission_source") == emission_source
            ]
        if animal_type is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("animal_type") == animal_type
            ]
        if farm_id is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("farm_id") == farm_id
            ]
        if from_date is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("calculated_at", "") >= from_date
            ]
        if to_date is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("calculated_at", "") <= to_date
            ]

        total = len(all_calcs)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_calcs[start:end]

        return {
            "calculations": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    # ==================================================================
    # 5. DELETE /calculations/{id} - Delete calculation
    # ==================================================================

    @router.delete("/calculations/{calc_id}", status_code=200)
    async def delete_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Delete an agricultural calculation result by its unique ID.

        Permanently removes the calculation record. This operation
        cannot be undone.
        """
        svc = _get_service()

        for i, calc in enumerate(svc._calculations):
            if calc.get("calculation_id") == calc_id:
                svc._calculations.pop(i)
                svc._total_calculations = max(
                    0, svc._total_calculations - 1,
                )
                return {
                    "deleted": True,
                    "calculation_id": calc_id,
                }

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calc_id} not found",
        )

    # ==================================================================
    # 6. POST /farms - Register farm
    # ==================================================================

    @router.post("/farms", status_code=201)
    async def create_farm(
        body: FarmBody,
    ) -> Dict[str, Any]:
        """Register a new agricultural farm.

        Creates a farm record with geographic location, climate zone,
        farm type, area, and soil characteristics. Every farm is
        scoped to a tenant and serves as the anchor for livestock
        herds, cropland inputs, rice fields, and field burning events.
        """
        svc = _get_service()
        try:
            result = svc.register_farm(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_farm failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 7. GET /farms - List farms
    # ==================================================================

    @router.get("/farms", status_code=200)
    async def list_farms(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        farm_type: Optional[str] = Query(
            None,
            description="Filter by farm type "
            "(dairy, beef, mixed_livestock, arable, "
            "rice_paddy, mixed_farming, etc.)",
        ),
        country_code: Optional[str] = Query(
            None,
            description="Filter by ISO 3166-1 alpha-2 country code",
        ),
        climate_zone: Optional[str] = Query(
            None,
            description="Filter by IPCC climate zone",
        ),
    ) -> Dict[str, Any]:
        """List registered farms with pagination and filters.

        Returns farms filtered by tenant, farm type, country code,
        and climate zone.
        """
        svc = _get_service()
        try:
            result = svc.list_farms(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                farm_type=farm_type,
                country_code=country_code,
                climate_zone=climate_zone,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_farms failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 8. PUT /farms/{id} - Update farm metadata
    # ==================================================================

    @router.put("/farms/{farm_id}", status_code=200)
    async def update_farm(
        body: FarmUpdateBody,
        farm_id: str = Path(
            ..., description="Farm identifier",
        ),
    ) -> Dict[str, Any]:
        """Update an existing farm's attributes.

        Only non-null fields in the request body will be updated.
        Immutable fields (tenant_id, latitude, longitude) cannot be
        changed through this endpoint.
        """
        svc = _get_service()
        update_data = {
            k: v for k, v in body.model_dump().items()
            if v is not None
        }
        try:
            result = svc.update_farm(farm_id, update_data)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Farm {farm_id} not found",
                )
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "update_farm failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. POST /livestock - Register livestock herd
    # ==================================================================

    @router.post("/livestock", status_code=201)
    async def create_livestock(
        body: LivestockBody,
    ) -> Dict[str, Any]:
        """Register a new livestock herd or flock.

        Creates a livestock record linked to a farm with animal type,
        head count, body weight, and feeding parameters. Livestock
        records provide the activity data for enteric fermentation
        and manure management emission calculations.
        """
        svc = _get_service()
        try:
            result = svc.register_livestock(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_livestock failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 10. GET /livestock - List livestock herds
    # ==================================================================

    @router.get("/livestock", status_code=200)
    async def list_livestock(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        farm_id: Optional[str] = Query(
            None, description="Filter by farm identifier",
        ),
        animal_type: Optional[str] = Query(
            None,
            description="Filter by animal type "
            "(dairy_cattle, non_dairy_cattle, buffalo, "
            "sheep, goats, swine, poultry, etc.)",
        ),
    ) -> Dict[str, Any]:
        """List registered livestock herds with pagination and filters.

        Returns livestock records filtered by tenant, farm, and
        animal type.
        """
        svc = _get_service()
        try:
            result = svc.list_livestock(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                farm_id=farm_id,
                animal_type=animal_type,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_livestock failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 11. PUT /livestock/{id} - Update livestock herd
    # ==================================================================

    @router.put("/livestock/{herd_id}", status_code=200)
    async def update_livestock(
        body: LivestockUpdateBody,
        herd_id: str = Path(
            ..., description="Livestock herd identifier",
        ),
    ) -> Dict[str, Any]:
        """Update an existing livestock herd's attributes.

        Only non-null fields in the request body will be updated.
        Allows adjusting head count, body weight, milk yield, feeding
        parameters, and manure management system as conditions change.
        """
        svc = _get_service()
        update_data = {
            k: v for k, v in body.model_dump().items()
            if v is not None
        }
        try:
            result = svc.update_livestock(herd_id, update_data)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Livestock herd {herd_id} not found",
                )
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "update_livestock failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. POST /cropland-inputs - Record cropland inputs
    # ==================================================================

    @router.post("/cropland-inputs", status_code=201)
    async def create_cropland_input(
        body: CroplandInputBody,
    ) -> Dict[str, Any]:
        """Record cropland nitrogen and liming inputs.

        Captures synthetic fertiliser, organic nitrogen, crop residue
        nitrogen, limestone, dolomite, and urea application data for
        a farm. Cropland inputs provide the activity data for
        agricultural soils N2O calculations and liming/urea CO2
        calculations.
        """
        svc = _get_service()
        try:
            result = svc.record_cropland_input(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_cropland_input failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 13. GET /cropland-inputs - List cropland inputs
    # ==================================================================

    @router.get("/cropland-inputs", status_code=200)
    async def list_cropland_inputs(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        farm_id: Optional[str] = Query(
            None, description="Filter by farm identifier",
        ),
        crop_type: Optional[str] = Query(
            None,
            description="Filter by crop type",
        ),
        reporting_year: Optional[int] = Query(
            None, ge=1990, le=2100,
            description="Filter by reporting year",
        ),
    ) -> Dict[str, Any]:
        """List cropland input records with pagination and filters.

        Returns cropland inputs filtered by tenant, farm, crop type,
        and reporting year.
        """
        svc = _get_service()
        try:
            result = svc.list_cropland_inputs(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                farm_id=farm_id,
                crop_type=crop_type,
                reporting_year=reporting_year,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_cropland_inputs failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 14. POST /rice-fields - Register rice field
    # ==================================================================

    @router.post("/rice-fields", status_code=201)
    async def create_rice_field(
        body: RiceFieldBody,
    ) -> Dict[str, Any]:
        """Register a new rice paddy field.

        Creates a rice field record linked to a farm with area,
        cultivation period, water management regime, pre-season
        flooding status, and organic amendment details. Rice field
        records provide the activity data for CH4 emission
        calculations from flooded rice paddies using IPCC
        methodology.
        """
        svc = _get_service()
        try:
            result = svc.register_rice_field(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_rice_field failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 15. GET /rice-fields - List rice fields
    # ==================================================================

    @router.get("/rice-fields", status_code=200)
    async def list_rice_fields(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        farm_id: Optional[str] = Query(
            None, description="Filter by farm identifier",
        ),
        water_regime: Optional[str] = Query(
            None,
            description="Filter by water management regime "
            "(continuously_flooded, "
            "intermittently_flooded_single, "
            "intermittently_flooded_multiple, "
            "rainfed_regular, rainfed_drought_prone, "
            "deep_water, upland)",
        ),
    ) -> Dict[str, Any]:
        """List registered rice fields with pagination and filters.

        Returns rice field records filtered by tenant, farm, and
        water management regime.
        """
        svc = _get_service()
        try:
            result = svc.list_rice_fields(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                farm_id=farm_id,
                water_regime=water_regime,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_rice_fields failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. POST /compliance/check - Run compliance check
    # ==================================================================

    @router.post("/compliance/check", status_code=200)
    async def check_compliance(
        body: ComplianceCheckBody,
    ) -> Dict[str, Any]:
        """Run regulatory compliance check against multiple frameworks.

        Evaluates the agricultural emission calculation against
        applicable frameworks: GHG Protocol Corporate Standard, IPCC
        2006/2019 Guidelines (Vol 4 - Agriculture, Forestry and Other
        Land Use), CSRD/ESRS E1, UNFCCC National Inventory
        requirements, US EPA Subpart H/S, and UK SECR guidance.
        """
        svc = _get_service()
        try:
            result = svc.check_compliance(body.model_dump())
            return result
        except Exception as exc:
            logger.error(
                "check_compliance failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 17. GET /compliance/{id} - Get compliance result
    # ==================================================================

    @router.get("/compliance/{compliance_id}", status_code=200)
    async def get_compliance_result(
        compliance_id: str = Path(
            ..., description="Compliance check identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a compliance check result by its unique identifier.

        Returns the full compliance assessment including per-framework
        status, rule violations, recommendations, and overall
        compliance score.
        """
        svc = _get_service()

        for result in svc._compliance_results:
            if result.get("id") == compliance_id:
                return result

        raise HTTPException(
            status_code=404,
            detail=f"Compliance result {compliance_id} not found",
        )

    # ==================================================================
    # 18. POST /uncertainty - Run Monte Carlo analysis
    # ==================================================================

    @router.post("/uncertainty", status_code=200)
    async def run_uncertainty(
        body: UncertaintyBody,
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on a calculation.

        Requires a previous ``calculation_id``. Performs Monte Carlo
        simulation by sampling emission factor distributions, activity
        data uncertainty, and parameter uncertainty according to IPCC
        guidelines. Returns statistical characterization including
        mean, standard deviation, confidence intervals, percentiles,
        and coefficient of variation.
        """
        svc = _get_service()
        try:
            result = svc.run_uncertainty(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "run_uncertainty failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 19. GET /aggregations - Get aggregated emissions
    # ==================================================================

    @router.get("/aggregations", status_code=200)
    async def get_aggregations(
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier for scoping",
        ),
        period: Optional[str] = Query(
            None,
            description="Reporting period "
            "(monthly, quarterly, annual, custom)",
        ),
        group_by: Optional[str] = Query(
            None,
            description="Comma-separated fields to group by "
            "(emission_source, animal_type, crop_type, "
            "farm_id)",
        ),
        emission_source: Optional[str] = Query(
            None,
            description="Filter by emission source "
            "(enteric_fermentation, manure_management, "
            "agricultural_soils, rice_cultivation, "
            "field_burning, liming, urea_application)",
        ),
        animal_type: Optional[str] = Query(
            None,
            description="Filter by animal type",
        ),
        crop_type: Optional[str] = Query(
            None,
            description="Filter by crop type",
        ),
        farm_ids: Optional[str] = Query(
            None,
            description="Comma-separated farm ID filter",
        ),
        start_date: Optional[str] = Query(
            None, description="Start date (ISO-8601)",
        ),
        end_date: Optional[str] = Query(
            None, description="End date (ISO-8601)",
        ),
    ) -> Dict[str, Any]:
        """Get aggregated agricultural emissions.

        Aggregates calculation results by specified grouping fields
        and reporting period. Supports filtering by tenant, emission
        source, animal type, crop type, farm, and date range. Returns
        totals for CO2, CH4, N2O, and total CO2e, with breakdowns by
        enteric fermentation, manure management, soils, rice, liming,
        urea, and field burning.
        """
        svc = _get_service()
        try:
            agg_data: Dict[str, Any] = {
                "tenant_id": tenant_id,
            }
            if period is not None:
                agg_data["period"] = period
            if group_by is not None:
                agg_data["group_by"] = [
                    g.strip() for g in group_by.split(",")
                    if g.strip()
                ]
            if emission_source is not None:
                agg_data["emission_source"] = emission_source
            if animal_type is not None:
                agg_data["animal_type"] = animal_type
            if crop_type is not None:
                agg_data["crop_type"] = crop_type
            if farm_ids is not None:
                agg_data["farm_ids"] = [
                    f.strip()
                    for f in farm_ids.split(",")
                    if f.strip()
                ]
            if start_date is not None:
                agg_data["start_date"] = start_date
            if end_date is not None:
                agg_data["end_date"] = end_date

            result = svc.aggregate(agg_data)
            return result
        except Exception as exc:
            logger.error(
                "get_aggregations failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 20. GET /health - Health check
    # ==================================================================

    @router.get("/health", status_code=200)
    async def health_check() -> Dict[str, Any]:
        """Health check for the Agricultural Emissions service.

        Returns service status, version, uptime, and summary
        statistics about registered farms, livestock herds, cropland
        inputs, rice fields, and calculations.
        """
        try:
            svc = _get_service()
            now = datetime.now(timezone.utc).isoformat()

            return {
                "status": "healthy",
                "service": "agricultural-emissions",
                "version": "1.0.0",
                "timestamp": now,
                "stats": {
                    "total_calculations": getattr(
                        svc, "_total_calculations", 0,
                    ),
                    "total_farms": len(
                        getattr(svc, "_farms", []),
                    ),
                    "total_livestock_herds": len(
                        getattr(svc, "_livestock", []),
                    ),
                    "total_cropland_inputs": len(
                        getattr(svc, "_cropland_inputs", []),
                    ),
                    "total_rice_fields": len(
                        getattr(svc, "_rice_fields", []),
                    ),
                    "total_compliance_checks": len(
                        getattr(svc, "_compliance_results", []),
                    ),
                },
            }
        except HTTPException:
            # Service not initialized but endpoint is reachable
            now = datetime.now(timezone.utc).isoformat()
            return {
                "status": "degraded",
                "service": "agricultural-emissions",
                "version": "1.0.0",
                "timestamp": now,
                "detail": "Service not yet initialized",
                "stats": {},
            }
        except Exception as exc:
            logger.error(
                "health_check failed: %s", exc, exc_info=True,
            )
            now = datetime.now(timezone.utc).isoformat()
            return {
                "status": "unhealthy",
                "service": "agricultural-emissions",
                "version": "1.0.0",
                "timestamp": now,
                "detail": str(exc),
                "stats": {},
            }

    return router


# ===================================================================
# Public API
# ===================================================================

__all__ = ["create_router"]
