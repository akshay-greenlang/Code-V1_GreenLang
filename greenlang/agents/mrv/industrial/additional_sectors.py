# -*- coding: utf-8 -*-
"""
Additional Industrial Sector MRV Agents
========================================

This module contains MRV agents for:
    - GL-MRV-IND-008: Pharmaceutical Manufacturing
    - GL-MRV-IND-009: Electronics Manufacturing
    - GL-MRV-IND-010: Automotive Manufacturing
    - GL-MRV-IND-011: Textiles Production
    - GL-MRV-IND-012: Mining Operations
    - GL-MRV-IND-013: Plastics Production

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from pydantic import Field

from .base import (
    IndustrialMRVBaseAgent,
    IndustrialMRVInput,
    IndustrialMRVOutput,
    CalculationStep,
    EmissionFactor,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# GL-MRV-IND-008: PHARMACEUTICAL MRV
# =============================================================================

class PharmaProcessType(str, Enum):
    """Pharmaceutical process types."""
    API_SYNTHESIS = "API_SYNTHESIS"  # Active Pharmaceutical Ingredient
    FORMULATION = "FORMULATION"
    PACKAGING = "PACKAGING"
    FERMENTATION = "FERMENTATION"
    EXTRACTION = "EXTRACTION"


class PharmaMRVInput(IndustrialMRVInput):
    """Input model for Pharmaceutical MRV."""
    process_type: PharmaProcessType = Field(default=PharmaProcessType.API_SYNTHESIS)
    solvent_usage_kg: Optional[Decimal] = Field(None, ge=0)
    hvac_area_m2: Optional[Decimal] = Field(None, ge=0)


class PharmaMRVOutput(IndustrialMRVOutput):
    """Output model for Pharmaceutical MRV."""
    process_type: str = Field(default="")
    solvent_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    hvac_emissions_tco2e: Decimal = Field(default=Decimal("0"))


class PharmaceuticalMRVAgent(IndustrialMRVBaseAgent[PharmaMRVInput, PharmaMRVOutput]):
    """
    GL-MRV-IND-008: Pharmaceutical Manufacturing MRV Agent

    Emission Sources:
        - Process: Heating, cooling, synthesis reactions
        - Solvents: VOC emissions (often captured)
        - HVAC: Cleanroom climate control
        - Electricity: High electricity intensity

    Emission Factors (tCO2e/t product):
        - API Synthesis: 5.0-15.0 (high energy intensity)
        - Formulation: 0.5-2.0
        - Fermentation: 2.0-5.0
    """

    AGENT_ID = "GL-MRV-IND-008"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Pharmaceutical"
    CBAM_CN_CODE = "2936-3006"
    CBAM_PRODUCT_CATEGORY = "Pharmaceutical Products"

    EMISSION_FACTORS = {
        PharmaProcessType.API_SYNTHESIS: Decimal("10.0"),
        PharmaProcessType.FORMULATION: Decimal("1.5"),
        PharmaProcessType.PACKAGING: Decimal("0.3"),
        PharmaProcessType.FERMENTATION: Decimal("3.5"),
        PharmaProcessType.EXTRACTION: Decimal("2.0"),
    }

    ELECTRICITY_KWH_PER_T = Decimal("5000")  # High for cleanrooms

    def _load_emission_factors(self) -> None:
        self._emission_factors = {
            pt.value: EmissionFactor(
                factor_id=f"pharma_{pt.value.lower()}",
                value=self.EMISSION_FACTORS[pt],
                unit="tCO2e/t_product",
                source="Pharma Industry Benchmarks",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=30.0
            )
            for pt in PharmaProcessType
        }

    def calculate_emissions(self, input_data: PharmaMRVInput) -> PharmaMRVOutput:
        steps, factors_used = [], []
        calc_id = self._generate_calculation_id(input_data.facility_id, input_data.reporting_period)

        ef = self._emission_factors[input_data.process_type.value]
        factors_used.append(ef)

        process_emissions = self._round_emissions(input_data.production_tonnes * ef.value)
        steps.append(CalculationStep(
            step_number=1,
            description=f"Process emissions ({input_data.process_type.value})",
            formula="production_tonnes * emission_factor",
            inputs={"production_tonnes": str(input_data.production_tonnes), "ef": str(ef.value)},
            output_value=process_emissions,
            output_unit="tCO2e",
            source=ef.source
        ))

        electricity_emissions = Decimal("0")
        if input_data.grid_emission_factor_kg_co2_per_kwh:
            kwh = input_data.electricity_kwh or (input_data.production_tonnes * self.ELECTRICITY_KWH_PER_T)
            electricity_emissions = self._round_emissions(kwh * input_data.grid_emission_factor_kg_co2_per_kwh / Decimal("1000"))

        steps.append(CalculationStep(
            step_number=2,
            description="Electricity emissions",
            formula="kwh * grid_factor / 1000",
            inputs={"kwh": str(input_data.electricity_kwh or Decimal("0"))},
            output_value=electricity_emissions,
            output_unit="tCO2e",
            source="Grid factor"
        ))

        total = process_emissions + electricity_emissions
        intensity = total / input_data.production_tonnes if input_data.production_tonnes > 0 else Decimal("0")

        return PharmaMRVOutput(
            calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(), facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period, production_tonnes=input_data.production_tonnes,
            process_type=input_data.process_type.value, scope_1_emissions_tco2e=process_emissions,
            scope_2_emissions_tco2e=electricity_emissions, total_emissions_tco2e=self._round_emissions(total),
            emission_intensity_tco2e_per_t=self._round_intensity(intensity),
            cbam_output=self._create_cbam_output(input_data.production_tonnes, process_emissions, electricity_emissions),
            calculation_steps=steps, emission_factors_used=factors_used,
            data_quality=input_data.data_quality, verification_status=VerificationStatus.UNVERIFIED, is_valid=True
        )


# =============================================================================
# GL-MRV-IND-009: ELECTRONICS MRV
# =============================================================================

class ElectronicsProductType(str, Enum):
    """Electronics product types."""
    SEMICONDUCTORS = "SEMICONDUCTORS"
    PCB = "PCB"
    DISPLAYS = "DISPLAYS"
    CONSUMER_ELECTRONICS = "CONSUMER_ELECTRONICS"
    COMPONENTS = "COMPONENTS"


class ElectronicsMRVInput(IndustrialMRVInput):
    """Input model for Electronics MRV."""
    product_type: ElectronicsProductType = Field(default=ElectronicsProductType.COMPONENTS)
    pfc_usage_kg: Optional[Decimal] = Field(None, ge=0, description="PFC gases used (SF6, NF3, etc.)")
    pfc_gwp: Decimal = Field(default=Decimal("23500"), description="GWP of PFC (SF6 default)")
    cleanroom_area_m2: Optional[Decimal] = Field(None, ge=0)


class ElectronicsMRVOutput(IndustrialMRVOutput):
    """Output model for Electronics MRV."""
    product_type: str = Field(default="")
    pfc_emissions_tco2e: Decimal = Field(default=Decimal("0"))


class ElectronicsMRVAgent(IndustrialMRVBaseAgent[ElectronicsMRVInput, ElectronicsMRVOutput]):
    """
    GL-MRV-IND-009: Electronics Manufacturing MRV Agent

    Emission Sources:
        - PFCs: High-GWP gases used in semiconductor fab (SF6, NF3, CF4)
        - Electricity: Extremely high for fabs (cleanrooms, precision equipment)
        - Process: Heat treatment, soldering

    Semiconductor fabs are among the most energy-intensive facilities.
    """

    AGENT_ID = "GL-MRV-IND-009"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Electronics"
    CBAM_CN_CODE = "8541-8542"
    CBAM_PRODUCT_CATEGORY = "Electronics"

    EMISSION_FACTORS = {
        ElectronicsProductType.SEMICONDUCTORS: Decimal("15.0"),  # Per wafer area proxy
        ElectronicsProductType.PCB: Decimal("2.0"),
        ElectronicsProductType.DISPLAYS: Decimal("5.0"),
        ElectronicsProductType.CONSUMER_ELECTRONICS: Decimal("1.0"),
        ElectronicsProductType.COMPONENTS: Decimal("0.8"),
    }

    ELECTRICITY_KWH_PER_T = Decimal("8000")  # Very high for fabs
    PFC_DESTRUCTION_RATE = Decimal("0.90")  # 90% abatement typical

    def _load_emission_factors(self) -> None:
        self._emission_factors = {
            pt.value: EmissionFactor(
                factor_id=f"elec_{pt.value.lower()}",
                value=self.EMISSION_FACTORS[pt],
                unit="tCO2e/t_product",
                source="Electronics Industry Benchmarks",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=30.0
            )
            for pt in ElectronicsProductType
        }

    def calculate_emissions(self, input_data: ElectronicsMRVInput) -> ElectronicsMRVOutput:
        steps, factors_used = [], []
        calc_id = self._generate_calculation_id(input_data.facility_id, input_data.reporting_period)

        ef = self._emission_factors[input_data.product_type.value]
        factors_used.append(ef)

        process_emissions = self._round_emissions(input_data.production_tonnes * ef.value)
        steps.append(CalculationStep(
            step_number=1, description=f"Process emissions ({input_data.product_type.value})",
            formula="production_tonnes * ef", inputs={"tonnes": str(input_data.production_tonnes)},
            output_value=process_emissions, output_unit="tCO2e", source=ef.source
        ))

        pfc_emissions = Decimal("0")
        if input_data.pfc_usage_kg:
            released = input_data.pfc_usage_kg * (Decimal("1") - self.PFC_DESTRUCTION_RATE)
            pfc_emissions = self._round_emissions(released * input_data.pfc_gwp / Decimal("1000"))

        steps.append(CalculationStep(
            step_number=2, description="PFC emissions",
            formula="pfc_kg * (1 - destruction_rate) * GWP / 1000",
            inputs={"pfc_kg": str(input_data.pfc_usage_kg or 0)},
            output_value=pfc_emissions, output_unit="tCO2e", source="GHG Protocol"
        ))

        electricity_emissions = Decimal("0")
        if input_data.grid_emission_factor_kg_co2_per_kwh:
            kwh = input_data.electricity_kwh or (input_data.production_tonnes * self.ELECTRICITY_KWH_PER_T)
            electricity_emissions = self._round_emissions(kwh * input_data.grid_emission_factor_kg_co2_per_kwh / Decimal("1000"))

        scope_1 = process_emissions + pfc_emissions
        total = scope_1 + electricity_emissions
        intensity = total / input_data.production_tonnes if input_data.production_tonnes > 0 else Decimal("0")

        return ElectronicsMRVOutput(
            calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(), facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period, production_tonnes=input_data.production_tonnes,
            product_type=input_data.product_type.value, pfc_emissions_tco2e=pfc_emissions,
            scope_1_emissions_tco2e=self._round_emissions(scope_1), scope_2_emissions_tco2e=electricity_emissions,
            total_emissions_tco2e=self._round_emissions(total), emission_intensity_tco2e_per_t=self._round_intensity(intensity),
            cbam_output=self._create_cbam_output(input_data.production_tonnes, scope_1, electricity_emissions),
            calculation_steps=steps, emission_factors_used=factors_used,
            data_quality=input_data.data_quality, verification_status=VerificationStatus.UNVERIFIED, is_valid=True
        )


# =============================================================================
# GL-MRV-IND-010: AUTOMOTIVE MRV
# =============================================================================

class AutomotiveProcessType(str, Enum):
    """Automotive manufacturing process types."""
    ASSEMBLY = "ASSEMBLY"
    BODY_SHOP = "BODY_SHOP"  # Stamping, welding
    PAINT_SHOP = "PAINT_SHOP"
    POWERTRAIN = "POWERTRAIN"
    BATTERY_PACK = "BATTERY_PACK"  # EV specific


class AutomotiveMRVInput(IndustrialMRVInput):
    """Input model for Automotive MRV."""
    process_type: AutomotiveProcessType = Field(default=AutomotiveProcessType.ASSEMBLY)
    vehicles_produced: Optional[int] = Field(None, ge=0, description="Number of vehicles (alternative to tonnes)")
    average_vehicle_weight_kg: Decimal = Field(default=Decimal("1500"))
    paint_voc_kg: Optional[Decimal] = Field(None, ge=0)


class AutomotiveMRVOutput(IndustrialMRVOutput):
    """Output model for Automotive MRV."""
    process_type: str = Field(default="")
    vehicles_produced: int = Field(default=0)
    emissions_per_vehicle_tco2e: Decimal = Field(default=Decimal("0"))


class AutomotiveMRVAgent(IndustrialMRVBaseAgent[AutomotiveMRVInput, AutomotiveMRVOutput]):
    """
    GL-MRV-IND-010: Automotive Manufacturing MRV Agent

    Emission Sources:
        - Paint shop: Major source (VOCs, natural gas for curing)
        - Body shop: Welding, stamping
        - Assembly: Electricity dominated
        - Powertrain: Machining, heat treatment

    Industry Benchmark: ~0.5-1.0 tCO2e per vehicle
    """

    AGENT_ID = "GL-MRV-IND-010"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Automotive"
    CBAM_CN_CODE = "8701-8705"
    CBAM_PRODUCT_CATEGORY = "Motor Vehicles"

    # tCO2e per vehicle by process
    EF_PER_VEHICLE = {
        AutomotiveProcessType.ASSEMBLY: Decimal("0.30"),
        AutomotiveProcessType.BODY_SHOP: Decimal("0.15"),
        AutomotiveProcessType.PAINT_SHOP: Decimal("0.35"),
        AutomotiveProcessType.POWERTRAIN: Decimal("0.20"),
        AutomotiveProcessType.BATTERY_PACK: Decimal("0.40"),
    }

    def _load_emission_factors(self) -> None:
        self._emission_factors = {
            pt.value: EmissionFactor(
                factor_id=f"auto_{pt.value.lower()}",
                value=self.EF_PER_VEHICLE[pt],
                unit="tCO2e/vehicle",
                source="Automotive Industry Benchmarks",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=20.0
            )
            for pt in AutomotiveProcessType
        }

    def calculate_emissions(self, input_data: AutomotiveMRVInput) -> AutomotiveMRVOutput:
        steps, factors_used = [], []
        calc_id = self._generate_calculation_id(input_data.facility_id, input_data.reporting_period)

        # Convert production to vehicle count if needed
        if input_data.vehicles_produced:
            vehicle_count = input_data.vehicles_produced
        else:
            vehicle_count = int(input_data.production_tonnes * Decimal("1000") / input_data.average_vehicle_weight_kg)

        ef = self._emission_factors[input_data.process_type.value]
        factors_used.append(ef)

        direct_emissions = self._round_emissions(Decimal(str(vehicle_count)) * ef.value)
        steps.append(CalculationStep(
            step_number=1, description=f"Direct emissions ({input_data.process_type.value})",
            formula="vehicles * ef_per_vehicle",
            inputs={"vehicles": str(vehicle_count), "ef": str(ef.value)},
            output_value=direct_emissions, output_unit="tCO2e", source=ef.source
        ))

        electricity_emissions = Decimal("0")
        if input_data.grid_emission_factor_kg_co2_per_kwh and input_data.electricity_kwh:
            electricity_emissions = self._round_emissions(
                input_data.electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh / Decimal("1000")
            )

        total = direct_emissions + electricity_emissions
        per_vehicle = total / Decimal(str(vehicle_count)) if vehicle_count > 0 else Decimal("0")

        return AutomotiveMRVOutput(
            calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(), facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period, production_tonnes=input_data.production_tonnes,
            process_type=input_data.process_type.value, vehicles_produced=vehicle_count,
            emissions_per_vehicle_tco2e=self._round_intensity(per_vehicle),
            scope_1_emissions_tco2e=direct_emissions, scope_2_emissions_tco2e=electricity_emissions,
            total_emissions_tco2e=self._round_emissions(total),
            emission_intensity_tco2e_per_t=self._round_intensity(per_vehicle),
            cbam_output=self._create_cbam_output(input_data.production_tonnes, direct_emissions, electricity_emissions),
            calculation_steps=steps, emission_factors_used=factors_used,
            data_quality=input_data.data_quality, verification_status=VerificationStatus.UNVERIFIED, is_valid=True
        )


# =============================================================================
# GL-MRV-IND-011: TEXTILES MRV
# =============================================================================

class TextileProcessType(str, Enum):
    """Textile production process types."""
    SPINNING = "SPINNING"
    WEAVING = "WEAVING"
    DYEING = "DYEING"
    FINISHING = "FINISHING"
    GARMENT = "GARMENT"


class TextilesMRVInput(IndustrialMRVInput):
    """Input model for Textiles MRV."""
    process_type: TextileProcessType = Field(default=TextileProcessType.DYEING)
    water_usage_m3: Optional[Decimal] = Field(None, ge=0)
    steam_usage_gj: Optional[Decimal] = Field(None, ge=0)


class TextilesMRVOutput(IndustrialMRVOutput):
    """Output model for Textiles MRV."""
    process_type: str = Field(default="")
    water_emissions_tco2e: Decimal = Field(default=Decimal("0"))


class TextilesMRVAgent(IndustrialMRVBaseAgent[TextilesMRVInput, TextilesMRVOutput]):
    """
    GL-MRV-IND-011: Textiles Production MRV Agent

    Emission Sources:
        - Dyeing: High thermal energy for heating water
        - Finishing: Steam, chemicals
        - Spinning/Weaving: Electricity dominated

    The dyeing/finishing wet processes are the most emission-intensive.
    """

    AGENT_ID = "GL-MRV-IND-011"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Textiles"
    CBAM_CN_CODE = "5001-6310"
    CBAM_PRODUCT_CATEGORY = "Textiles"

    EF_BY_PROCESS = {
        TextileProcessType.SPINNING: Decimal("0.50"),
        TextileProcessType.WEAVING: Decimal("0.40"),
        TextileProcessType.DYEING: Decimal("2.50"),
        TextileProcessType.FINISHING: Decimal("1.50"),
        TextileProcessType.GARMENT: Decimal("0.30"),
    }

    def _load_emission_factors(self) -> None:
        self._emission_factors = {
            pt.value: EmissionFactor(
                factor_id=f"textile_{pt.value.lower()}",
                value=self.EF_BY_PROCESS[pt],
                unit="tCO2e/t_textile",
                source="Textile Industry Benchmarks",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=25.0
            )
            for pt in TextileProcessType
        }

    def calculate_emissions(self, input_data: TextilesMRVInput) -> TextilesMRVOutput:
        steps, factors_used = [], []
        calc_id = self._generate_calculation_id(input_data.facility_id, input_data.reporting_period)

        ef = self._emission_factors[input_data.process_type.value]
        factors_used.append(ef)

        process_emissions = self._round_emissions(input_data.production_tonnes * ef.value)
        steps.append(CalculationStep(
            step_number=1, description=f"Process emissions ({input_data.process_type.value})",
            formula="tonnes * ef", inputs={"tonnes": str(input_data.production_tonnes)},
            output_value=process_emissions, output_unit="tCO2e", source=ef.source
        ))

        electricity_emissions = Decimal("0")
        if input_data.grid_emission_factor_kg_co2_per_kwh and input_data.electricity_kwh:
            electricity_emissions = self._round_emissions(
                input_data.electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh / Decimal("1000")
            )

        total = process_emissions + electricity_emissions
        intensity = total / input_data.production_tonnes if input_data.production_tonnes > 0 else Decimal("0")

        return TextilesMRVOutput(
            calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(), facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period, production_tonnes=input_data.production_tonnes,
            process_type=input_data.process_type.value,
            scope_1_emissions_tco2e=process_emissions, scope_2_emissions_tco2e=electricity_emissions,
            total_emissions_tco2e=self._round_emissions(total),
            emission_intensity_tco2e_per_t=self._round_intensity(intensity),
            cbam_output=self._create_cbam_output(input_data.production_tonnes, process_emissions, electricity_emissions),
            calculation_steps=steps, emission_factors_used=factors_used,
            data_quality=input_data.data_quality, verification_status=VerificationStatus.UNVERIFIED, is_valid=True
        )


# =============================================================================
# GL-MRV-IND-012: MINING MRV
# =============================================================================

class MiningType(str, Enum):
    """Mining operation types."""
    COAL = "COAL"
    IRON_ORE = "IRON_ORE"
    COPPER = "COPPER"
    GOLD = "GOLD"
    LITHIUM = "LITHIUM"
    RARE_EARTH = "RARE_EARTH"


class MiningMRVInput(IndustrialMRVInput):
    """Input model for Mining MRV."""
    mining_type: MiningType = Field(default=MiningType.IRON_ORE)
    diesel_litres: Optional[Decimal] = Field(None, ge=0)
    explosives_kg: Optional[Decimal] = Field(None, ge=0)
    strip_ratio: Optional[Decimal] = Field(default=Decimal("2.0"), ge=0, description="Waste to ore ratio")


class MiningMRVOutput(IndustrialMRVOutput):
    """Output model for Mining MRV."""
    mining_type: str = Field(default="")
    diesel_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    blasting_emissions_tco2e: Decimal = Field(default=Decimal("0"))


class MiningMRVAgent(IndustrialMRVBaseAgent[MiningMRVInput, MiningMRVOutput]):
    """
    GL-MRV-IND-012: Mining Operations MRV Agent

    Emission Sources:
        - Mobile equipment: Haul trucks, loaders (diesel)
        - Blasting: Explosives
        - Processing: Crushing, grinding (electricity)
        - Ventilation: Underground mines

    Emission intensity varies greatly by ore grade and strip ratio.
    """

    AGENT_ID = "GL-MRV-IND-012"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Mining"
    CBAM_CN_CODE = "2601-2621"
    CBAM_PRODUCT_CATEGORY = "Mining Products"

    # tCO2e per tonne of ore
    EF_BY_TYPE = {
        MiningType.COAL: Decimal("0.020"),
        MiningType.IRON_ORE: Decimal("0.015"),
        MiningType.COPPER: Decimal("0.050"),  # Lower grade = more processing
        MiningType.GOLD: Decimal("0.800"),    # Very low grade
        MiningType.LITHIUM: Decimal("0.100"),
        MiningType.RARE_EARTH: Decimal("0.200"),
    }

    EF_DIESEL_KG_CO2_PER_L = Decimal("2.68")
    EF_EXPLOSIVES_KG_CO2_PER_KG = Decimal("0.20")

    def _load_emission_factors(self) -> None:
        self._emission_factors = {
            mt.value: EmissionFactor(
                factor_id=f"mining_{mt.value.lower()}",
                value=self.EF_BY_TYPE[mt],
                unit="tCO2e/t_ore",
                source="Mining Industry Benchmarks",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=30.0
            )
            for mt in MiningType
        }

    def calculate_emissions(self, input_data: MiningMRVInput) -> MiningMRVOutput:
        steps, factors_used = [], []
        calc_id = self._generate_calculation_id(input_data.facility_id, input_data.reporting_period)

        ef = self._emission_factors[input_data.mining_type.value]
        factors_used.append(ef)

        # Adjust for strip ratio
        adjusted_ef = ef.value * (Decimal("1") + (input_data.strip_ratio or Decimal("2.0")) / Decimal("10"))
        base_emissions = self._round_emissions(input_data.production_tonnes * adjusted_ef)

        steps.append(CalculationStep(
            step_number=1, description=f"Base mining emissions ({input_data.mining_type.value})",
            formula="tonnes * ef * (1 + strip_ratio/10)",
            inputs={"tonnes": str(input_data.production_tonnes), "strip_ratio": str(input_data.strip_ratio)},
            output_value=base_emissions, output_unit="tCO2e", source=ef.source
        ))

        diesel_emissions = Decimal("0")
        if input_data.diesel_litres:
            diesel_emissions = self._round_emissions(
                input_data.diesel_litres * self.EF_DIESEL_KG_CO2_PER_L / Decimal("1000")
            )

        blasting_emissions = Decimal("0")
        if input_data.explosives_kg:
            blasting_emissions = self._round_emissions(
                input_data.explosives_kg * self.EF_EXPLOSIVES_KG_CO2_PER_KG / Decimal("1000")
            )

        electricity_emissions = Decimal("0")
        if input_data.grid_emission_factor_kg_co2_per_kwh and input_data.electricity_kwh:
            electricity_emissions = self._round_emissions(
                input_data.electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh / Decimal("1000")
            )

        scope_1 = base_emissions + diesel_emissions + blasting_emissions
        total = scope_1 + electricity_emissions
        intensity = total / input_data.production_tonnes if input_data.production_tonnes > 0 else Decimal("0")

        return MiningMRVOutput(
            calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(), facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period, production_tonnes=input_data.production_tonnes,
            mining_type=input_data.mining_type.value,
            diesel_emissions_tco2e=diesel_emissions, blasting_emissions_tco2e=blasting_emissions,
            scope_1_emissions_tco2e=self._round_emissions(scope_1), scope_2_emissions_tco2e=electricity_emissions,
            total_emissions_tco2e=self._round_emissions(total),
            emission_intensity_tco2e_per_t=self._round_intensity(intensity),
            cbam_output=self._create_cbam_output(input_data.production_tonnes, scope_1, electricity_emissions),
            calculation_steps=steps, emission_factors_used=factors_used,
            data_quality=input_data.data_quality, verification_status=VerificationStatus.UNVERIFIED, is_valid=True
        )


# =============================================================================
# GL-MRV-IND-013: PLASTICS MRV
# =============================================================================

class PlasticType(str, Enum):
    """Plastic resin types."""
    POLYETHYLENE = "POLYETHYLENE"  # PE (HDPE, LDPE, LLDPE)
    POLYPROPYLENE = "POLYPROPYLENE"  # PP
    PVC = "PVC"
    PET = "PET"
    POLYSTYRENE = "POLYSTYRENE"  # PS
    ABS = "ABS"


class PlasticsMRVInput(IndustrialMRVInput):
    """Input model for Plastics MRV."""
    plastic_type: PlasticType = Field(default=PlasticType.POLYETHYLENE)
    recycled_content_ratio: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    feedstock_origin: str = Field(default="fossil", description="fossil or bio-based")


class PlasticsMRVOutput(IndustrialMRVOutput):
    """Output model for Plastics MRV."""
    plastic_type: str = Field(default="")
    recycled_content_ratio: Decimal = Field(default=Decimal("0"))
    feedstock_emissions_tco2e: Decimal = Field(default=Decimal("0"))


class PlasticsMRVAgent(IndustrialMRVBaseAgent[PlasticsMRVInput, PlasticsMRVOutput]):
    """
    GL-MRV-IND-013: Plastics Production MRV Agent

    Emission Sources:
        - Feedstock: Fossil carbon in monomers
        - Process: Polymerization energy
        - Electricity: Extruders, compounding

    Cradle-to-gate emissions for virgin plastics are ~2-4 tCO2e/t.
    Recycled content reduces this significantly.
    """

    AGENT_ID = "GL-MRV-IND-013"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Plastics"
    CBAM_CN_CODE = "3901-3914"
    CBAM_PRODUCT_CATEGORY = "Plastics"

    EF_BY_TYPE = {
        PlasticType.POLYETHYLENE: Decimal("1.80"),
        PlasticType.POLYPROPYLENE: Decimal("1.70"),
        PlasticType.PVC: Decimal("2.10"),
        PlasticType.PET: Decimal("2.50"),
        PlasticType.POLYSTYRENE: Decimal("3.00"),
        PlasticType.ABS: Decimal("3.30"),
    }

    EF_RECYCLED_REDUCTION = Decimal("0.70")  # 70% reduction for recycled

    def _load_emission_factors(self) -> None:
        self._emission_factors = {
            pt.value: EmissionFactor(
                factor_id=f"plastic_{pt.value.lower()}",
                value=self.EF_BY_TYPE[pt],
                unit="tCO2e/t_plastic",
                source="PlasticsEurope Eco-profiles",
                region="global",
                valid_from="2020-01-01",
                uncertainty_percent=15.0
            )
            for pt in PlasticType
        }

    def calculate_emissions(self, input_data: PlasticsMRVInput) -> PlasticsMRVOutput:
        steps, factors_used = [], []
        calc_id = self._generate_calculation_id(input_data.facility_id, input_data.reporting_period)

        ef = self._emission_factors[input_data.plastic_type.value]
        factors_used.append(ef)

        # Adjust for recycled content
        virgin_share = Decimal("1") - input_data.recycled_content_ratio
        recycled_share = input_data.recycled_content_ratio

        virgin_ef = ef.value
        recycled_ef = ef.value * (Decimal("1") - self.EF_RECYCLED_REDUCTION)
        blended_ef = (virgin_share * virgin_ef) + (recycled_share * recycled_ef)

        process_emissions = self._round_emissions(input_data.production_tonnes * blended_ef)
        steps.append(CalculationStep(
            step_number=1, description=f"Process emissions ({input_data.plastic_type.value})",
            formula="tonnes * (virgin_share * virgin_ef + recycled_share * recycled_ef)",
            inputs={
                "tonnes": str(input_data.production_tonnes),
                "recycled_ratio": str(input_data.recycled_content_ratio),
                "virgin_ef": str(virgin_ef), "recycled_ef": str(recycled_ef)
            },
            output_value=process_emissions, output_unit="tCO2e", source=ef.source
        ))

        electricity_emissions = Decimal("0")
        if input_data.grid_emission_factor_kg_co2_per_kwh and input_data.electricity_kwh:
            electricity_emissions = self._round_emissions(
                input_data.electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh / Decimal("1000")
            )

        total = process_emissions + electricity_emissions
        intensity = total / input_data.production_tonnes if input_data.production_tonnes > 0 else Decimal("0")

        return PlasticsMRVOutput(
            calculation_id=calc_id, agent_id=self.AGENT_ID, agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(), facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period, production_tonnes=input_data.production_tonnes,
            plastic_type=input_data.plastic_type.value,
            recycled_content_ratio=input_data.recycled_content_ratio,
            feedstock_emissions_tco2e=process_emissions,
            scope_1_emissions_tco2e=process_emissions, scope_2_emissions_tco2e=electricity_emissions,
            total_emissions_tco2e=self._round_emissions(total),
            emission_intensity_tco2e_per_t=self._round_intensity(intensity),
            cbam_output=self._create_cbam_output(input_data.production_tonnes, process_emissions, electricity_emissions),
            calculation_steps=steps, emission_factors_used=factors_used,
            data_quality=input_data.data_quality, verification_status=VerificationStatus.UNVERIFIED, is_valid=True
        )
