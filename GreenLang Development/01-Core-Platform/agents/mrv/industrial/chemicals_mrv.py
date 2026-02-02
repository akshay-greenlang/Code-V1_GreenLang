# -*- coding: utf-8 -*-
"""
GL-MRV-IND-003: Chemicals Production MRV Agent
===============================================

Industrial MRV agent for chemical sector emissions measurement, reporting, and verification.
Covers major chemical products including ammonia, methanol, ethylene, and bulk chemicals.

Chemical Products:
    - Ammonia (NH3): Primary feedstock for fertilizers
    - Methanol (CH3OH): Base chemical for many derivatives
    - Ethylene (C2H4): Building block for polyethylene
    - Propylene (C3H6): Building block for polypropylene
    - Chlorine (Cl2): Produced via chlor-alkali process
    - Hydrogen (H2): Industrial hydrogen production

Sources:
    - IPCC 2006 Guidelines, Volume 3, Chapter 3 (Chemical Industry)
    - IEA Chemical Industry Analysis
    - European Chemical Industry Council (CEFIC)
    - CBAM Implementing Regulation (EU) 2023/956 Annex III

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
    DataQuality,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ChemicalProduct(str, Enum):
    """Chemical products covered by this agent."""
    AMMONIA = "AMMONIA"
    METHANOL = "METHANOL"
    ETHYLENE = "ETHYLENE"
    PROPYLENE = "PROPYLENE"
    CHLORINE = "CHLORINE"
    HYDROGEN_GRAY = "HYDROGEN_GRAY"
    HYDROGEN_BLUE = "HYDROGEN_BLUE"
    HYDROGEN_GREEN = "HYDROGEN_GREEN"
    UREA = "UREA"
    NITRIC_ACID = "NITRIC_ACID"


class FeedstockType(str, Enum):
    """Feedstock types for chemical production."""
    NATURAL_GAS = "NATURAL_GAS"
    NAPHTHA = "NAPHTHA"
    COAL = "COAL"
    BIOMASS = "BIOMASS"
    ELECTROLYSIS = "ELECTROLYSIS"


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class ChemicalsMRVInput(IndustrialMRVInput):
    """Input model for Chemicals Production MRV."""

    chemical_product: ChemicalProduct = Field(
        ..., description="Type of chemical product"
    )

    feedstock_type: FeedstockType = Field(
        default=FeedstockType.NATURAL_GAS,
        description="Primary feedstock type"
    )

    # Process-specific inputs
    feedstock_tonnes: Optional[Decimal] = Field(None, ge=0)
    steam_tonnes: Optional[Decimal] = Field(None, ge=0)
    co2_captured_tonnes: Optional[Decimal] = Field(None, ge=0)  # For CCS


class ChemicalsMRVOutput(IndustrialMRVOutput):
    """Output model for Chemicals Production MRV."""

    chemical_product: str = Field(default="")
    feedstock_type: str = Field(default="")

    # Emissions breakdown
    feedstock_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    process_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    energy_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    ccs_credit_tco2e: Decimal = Field(default=Decimal("0"))


# =============================================================================
# CHEMICALS MRV AGENT
# =============================================================================

class ChemicalsProductionMRVAgent(IndustrialMRVBaseAgent[ChemicalsMRVInput, ChemicalsMRVOutput]):
    """
    GL-MRV-IND-003: Chemicals Production MRV Agent

    Implements zero-hallucination emissions calculation for chemical production.

    Emission Factors (tCO2/t product):
        - Ammonia (NG): 1.9 (IPCC default)
        - Methanol (NG): 0.67 (IPCC 2006)
        - Ethylene (Naphtha): 1.73 (IEA)
        - Hydrogen Gray: 10.0 (per t H2)
        - Hydrogen Blue: 2.0 (with CCS)
        - Hydrogen Green: 0.0 (renewable electrolysis)

    Example:
        >>> agent = ChemicalsProductionMRVAgent()
        >>> input_data = ChemicalsMRVInput(
        ...     facility_id="CHEM_001",
        ...     reporting_period="2024-Q1",
        ...     production_tonnes=Decimal("100000"),
        ...     chemical_product=ChemicalProduct.AMMONIA,
        ...     feedstock_type=FeedstockType.NATURAL_GAS
        ... )
        >>> result = agent.process(input_data)
    """

    AGENT_ID = "GL-MRV-IND-003"
    AGENT_VERSION = "1.0.0"
    SECTOR = "Chemicals"
    CBAM_CN_CODE = "2804-2942"
    CBAM_PRODUCT_CATEGORY = "Chemicals"

    # Emission factors (tCO2/t product) - IPCC 2006 / IEA
    EMISSION_FACTORS = {
        ChemicalProduct.AMMONIA: {
            FeedstockType.NATURAL_GAS: Decimal("1.90"),
            FeedstockType.COAL: Decimal("3.80"),
            FeedstockType.NAPHTHA: Decimal("2.50"),
        },
        ChemicalProduct.METHANOL: {
            FeedstockType.NATURAL_GAS: Decimal("0.67"),
            FeedstockType.COAL: Decimal("1.50"),
        },
        ChemicalProduct.ETHYLENE: {
            FeedstockType.NAPHTHA: Decimal("1.73"),
            FeedstockType.NATURAL_GAS: Decimal("1.10"),  # Ethane cracking
        },
        ChemicalProduct.PROPYLENE: {
            FeedstockType.NAPHTHA: Decimal("1.50"),
            FeedstockType.NATURAL_GAS: Decimal("1.20"),
        },
        ChemicalProduct.CHLORINE: {
            FeedstockType.ELECTROLYSIS: Decimal("0.30"),  # Direct only
        },
        ChemicalProduct.HYDROGEN_GRAY: {
            FeedstockType.NATURAL_GAS: Decimal("10.00"),
        },
        ChemicalProduct.HYDROGEN_BLUE: {
            FeedstockType.NATURAL_GAS: Decimal("2.00"),  # With CCS
        },
        ChemicalProduct.HYDROGEN_GREEN: {
            FeedstockType.ELECTROLYSIS: Decimal("0.00"),
        },
        ChemicalProduct.UREA: {
            FeedstockType.NATURAL_GAS: Decimal("1.57"),
        },
        ChemicalProduct.NITRIC_ACID: {
            FeedstockType.NATURAL_GAS: Decimal("0.30"),  # Process N2O included
        },
    }

    # Electricity consumption (kWh/t product)
    ELECTRICITY_CONSUMPTION = {
        ChemicalProduct.AMMONIA: Decimal("100"),
        ChemicalProduct.METHANOL: Decimal("150"),
        ChemicalProduct.ETHYLENE: Decimal("300"),
        ChemicalProduct.CHLORINE: Decimal("2500"),  # Electrolysis
        ChemicalProduct.HYDROGEN_GREEN: Decimal("50000"),  # High electricity
    }

    def _load_emission_factors(self) -> None:
        """Load chemical sector emission factors."""
        self._emission_factors = {}
        for product, feedstocks in self.EMISSION_FACTORS.items():
            for feedstock, value in feedstocks.items():
                key = f"{product.value}_{feedstock.value}"
                self._emission_factors[key] = EmissionFactor(
                    factor_id=f"chem_{key.lower()}",
                    value=value,
                    unit="tCO2/t_product",
                    source="IPCC 2006 Vol 3 Ch 3 / IEA",
                    region="global",
                    valid_from="2006-01-01",
                    uncertainty_percent=15.0
                )

    def calculate_emissions(self, input_data: ChemicalsMRVInput) -> ChemicalsMRVOutput:
        """
        Calculate chemical production emissions - ZERO HALLUCINATION.

        Args:
            input_data: Validated chemical production input

        Returns:
            Complete MRV output with emissions breakdown
        """
        steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []
        step_num = 0

        calc_id = self._generate_calculation_id(
            input_data.facility_id, input_data.reporting_period
        )

        # Step 1: Get emission factor
        step_num += 1
        ef_key = f"{input_data.chemical_product.value}_{input_data.feedstock_type.value}"
        product_ef = self._get_product_emission_factor(
            input_data.chemical_product, input_data.feedstock_type
        )
        factors_used.append(product_ef)

        # Step 2: Calculate process emissions
        step_num += 1
        process_emissions = input_data.production_tonnes * product_ef.value
        process_emissions = self._round_emissions(process_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description=f"Process emissions for {input_data.chemical_product.value}",
            formula="production_tonnes * emission_factor",
            inputs={
                "production_tonnes": str(input_data.production_tonnes),
                "chemical_product": input_data.chemical_product.value,
                "feedstock_type": input_data.feedstock_type.value,
                "emission_factor": str(product_ef.value)
            },
            output_value=process_emissions,
            output_unit="tCO2e",
            source=product_ef.source
        ))

        # Step 3: Calculate electricity emissions
        step_num += 1
        electricity_emissions = self._calculate_electricity_emissions(input_data)
        electricity_emissions = self._round_emissions(electricity_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Electricity emissions",
            formula="electricity_kwh * grid_factor / 1000",
            inputs={
                "electricity_kwh": str(input_data.electricity_kwh or Decimal("0")),
                "grid_factor": str(
                    input_data.grid_emission_factor_kg_co2_per_kwh or Decimal("0")
                )
            },
            output_value=electricity_emissions,
            output_unit="tCO2e",
            source="Grid emission factor"
        ))

        # Step 4: CCS credit (if applicable)
        step_num += 1
        ccs_credit = Decimal("0")
        if input_data.co2_captured_tonnes:
            ccs_credit = -input_data.co2_captured_tonnes
            ccs_credit = self._round_emissions(ccs_credit)

        steps.append(CalculationStep(
            step_number=step_num,
            description="CCS credit (captured CO2)",
            formula="-co2_captured_tonnes",
            inputs={
                "co2_captured_tonnes": str(input_data.co2_captured_tonnes or Decimal("0"))
            },
            output_value=ccs_credit,
            output_unit="tCO2e",
            source="CCS monitoring"
        ))

        # Step 5: Total emissions
        step_num += 1
        scope_1 = process_emissions + ccs_credit
        scope_2 = electricity_emissions
        total_emissions = scope_1 + scope_2
        total_emissions = self._round_emissions(total_emissions)

        steps.append(CalculationStep(
            step_number=step_num,
            description="Total emissions calculation",
            formula="process + electricity + ccs_credit",
            inputs={
                "process_emissions": str(process_emissions),
                "electricity_emissions": str(electricity_emissions),
                "ccs_credit": str(ccs_credit)
            },
            output_value=total_emissions,
            output_unit="tCO2e",
            source="Summation"
        ))

        # Calculate intensity
        emission_intensity = (
            total_emissions / input_data.production_tonnes
            if input_data.production_tonnes > 0 else Decimal("0")
        )
        emission_intensity = self._round_intensity(emission_intensity)

        # Create CBAM output
        cbam_output = self._create_cbam_output(
            production_tonnes=input_data.production_tonnes,
            direct_emissions=scope_1,
            indirect_emissions=scope_2
        )

        return ChemicalsMRVOutput(
            calculation_id=calc_id,
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            facility_id=input_data.facility_id,
            reporting_period=input_data.reporting_period,
            production_tonnes=input_data.production_tonnes,
            chemical_product=input_data.chemical_product.value,
            feedstock_type=input_data.feedstock_type.value,
            feedstock_emissions_tco2e=Decimal("0"),
            process_emissions_tco2e=process_emissions,
            energy_emissions_tco2e=electricity_emissions,
            ccs_credit_tco2e=ccs_credit,
            scope_1_emissions_tco2e=self._round_emissions(scope_1),
            scope_2_emissions_tco2e=self._round_emissions(scope_2),
            total_emissions_tco2e=total_emissions,
            emission_intensity_tco2e_per_t=emission_intensity,
            cbam_output=cbam_output,
            calculation_steps=steps,
            emission_factors_used=factors_used,
            data_quality=input_data.data_quality,
            verification_status=VerificationStatus.UNVERIFIED,
            is_valid=True
        )

    def _get_product_emission_factor(
        self,
        product: ChemicalProduct,
        feedstock: FeedstockType
    ) -> EmissionFactor:
        """Get emission factor for product/feedstock combination."""
        ef_key = f"{product.value}_{feedstock.value}"
        if ef_key in self._emission_factors:
            return self._emission_factors[ef_key]

        # Try to find any factor for this product
        for key, ef in self._emission_factors.items():
            if key.startswith(product.value):
                return ef

        raise ValueError(
            f"No emission factor for {product.value} with {feedstock.value}"
        )

    def _calculate_electricity_emissions(self, input_data: ChemicalsMRVInput) -> Decimal:
        """Calculate electricity emissions."""
        if input_data.grid_emission_factor_kg_co2_per_kwh is None:
            return Decimal("0")

        electricity_kwh = input_data.electricity_kwh
        if electricity_kwh is None:
            default_consumption = self.ELECTRICITY_CONSUMPTION.get(
                input_data.chemical_product, Decimal("200")
            )
            electricity_kwh = input_data.production_tonnes * default_consumption

        emissions_kg = electricity_kwh * input_data.grid_emission_factor_kg_co2_per_kwh
        return emissions_kg / Decimal("1000")
