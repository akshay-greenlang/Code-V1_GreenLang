# -*- coding: utf-8 -*-
"""
SB 253 Scope 2 Location-Based Electricity Calculator
=====================================================

Calculates indirect GHG emissions from purchased electricity using
grid average emission factors from EPA eGRID.

Emission Factors: EPA eGRID 2023
California Factor: CAMX = 0.254 kg CO2e/kWh

Formula:
    Emissions (kg CO2e) = Electricity (kWh) x Grid Factor (kg CO2e/kWh)

GHG Protocol Requirement:
    Location-based method is MANDATORY for SB 253 dual reporting.

Accuracy Target: +/- 2%

Author: GreenLang Framework Team
Version: 1.0.0
Date: 2025-12-04
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from ..base import BaseCalculator, CalculationResult, AuditRecord


@dataclass
class ElectricityInput:
    """Input data for electricity consumption."""
    facility_id: str
    egrid_subregion: str
    quantity_kwh: float
    reporting_period_start: str
    reporting_period_end: str
    source_document_id: Optional[str] = None
    utility_name: Optional[str] = None


@dataclass
class GridFactorData:
    """Grid emission factor with metadata."""
    factor: float
    name: str
    states: List[str]
    source: str
    source_uri: str
    data_year: int


class LocationBasedCalculator(BaseCalculator):
    """
    Calculate Scope 2 location-based emissions from purchased electricity.

    This calculator uses EPA eGRID subregional emission factors.
    Location-based method is MANDATORY under GHG Protocol Scope 2 Guidance.

    Key California Factor:
        CAMX (California) = 0.254 kg CO2e/kWh

    This is significantly lower than US average (0.417 kg CO2e/kWh) due to
    California's high renewable energy penetration.

    Formula:
        Emissions (kg CO2e) = Electricity (kWh) x Grid Factor (kg CO2e/kWh)

    Accuracy Target: +/- 2%
    """

    CALCULATOR_ID = "sb253-scope2-location-v1"
    CALCULATOR_VERSION = "1.0.0"

    # EPA eGRID 2023 Subregional Emission Factors (kg CO2e/kWh)
    # Source: https://www.epa.gov/egrid/download-data
    # Data Year: 2022 (released November 2024)
    EGRID_FACTORS: Dict[str, GridFactorData] = {
        # California (WECC California) - KEY FOR SB 253
        "CAMX": GridFactorData(
            factor=0.254,
            name="WECC California",
            states=["CA"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # Southwest
        "AZNM": GridFactorData(
            factor=0.458,
            name="WECC Southwest",
            states=["AZ", "NM"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # Northwest
        "NWPP": GridFactorData(
            factor=0.354,
            name="WECC Northwest",
            states=["WA", "OR", "ID", "MT", "WY", "NV", "UT", "CO"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # Rockies
        "RMPA": GridFactorData(
            factor=0.684,
            name="WECC Rockies",
            states=["CO", "NE", "WY"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # Texas
        "ERCT": GridFactorData(
            factor=0.376,
            name="ERCOT All",
            states=["TX"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # Florida
        "FRCC": GridFactorData(
            factor=0.421,
            name="FRCC All",
            states=["FL"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # Midwest East
        "MROE": GridFactorData(
            factor=0.651,
            name="MRO East",
            states=["WI", "MI"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # Midwest West
        "MROW": GridFactorData(
            factor=0.583,
            name="MRO West",
            states=["MN", "IA", "ND", "SD", "NE", "MT"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # New England
        "NEWE": GridFactorData(
            factor=0.183,
            name="NPCC New England",
            states=["CT", "MA", "ME", "NH", "RI", "VT"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # New York City/Westchester
        "NYCW": GridFactorData(
            factor=0.260,
            name="NPCC NYC/Westchester",
            states=["NY"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # New York Upstate
        "NYUP": GridFactorData(
            factor=0.160,
            name="NPCC Upstate NY",
            states=["NY"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # Long Island
        "NYLI": GridFactorData(
            factor=0.366,
            name="NPCC Long Island",
            states=["NY"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # PJM East (Mid-Atlantic)
        "RFCE": GridFactorData(
            factor=0.355,
            name="RFC East",
            states=["PA", "NJ", "MD", "DE", "DC"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # Michigan
        "RFCM": GridFactorData(
            factor=0.572,
            name="RFC Michigan",
            states=["MI"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # PJM West
        "RFCW": GridFactorData(
            factor=0.568,
            name="RFC West",
            states=["OH", "IN", "KY", "WV"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # SERC Midwest
        "SRMW": GridFactorData(
            factor=0.713,
            name="SERC Midwest",
            states=["MO", "IL", "AR"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # SERC Mississippi Valley
        "SRMV": GridFactorData(
            factor=0.432,
            name="SERC Mississippi Valley",
            states=["LA", "MS", "AR"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # SERC South
        "SRSO": GridFactorData(
            factor=0.455,
            name="SERC South",
            states=["GA", "AL"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # SERC Tennessee Valley
        "SRTV": GridFactorData(
            factor=0.478,
            name="SERC Tennessee Valley",
            states=["TN", "NC", "KY"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # SERC Virginia/Carolina
        "SRVC": GridFactorData(
            factor=0.379,
            name="SERC Virginia/Carolina",
            states=["VA", "NC", "SC"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # SPP North
        "SPNO": GridFactorData(
            factor=0.583,
            name="SPP North",
            states=["KS", "NE", "OK"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
        # SPP South
        "SPSO": GridFactorData(
            factor=0.512,
            name="SPP South",
            states=["OK", "TX", "LA", "AR"],
            source="EPA eGRID 2023",
            source_uri="https://www.epa.gov/egrid",
            data_year=2022,
        ),
    }

    # US National Average for reference
    US_NATIONAL_AVERAGE = 0.417  # kg CO2e/kWh

    def __init__(self):
        super().__init__(
            calculator_id=self.CALCULATOR_ID,
            version=self.CALCULATOR_VERSION
        )

    def calculate(
        self,
        inputs: List[Dict[str, Any]]
    ) -> CalculationResult:
        """
        Calculate location-based Scope 2 emissions.

        This method is DETERMINISTIC - same inputs always produce same outputs.

        Args:
            inputs: List of electricity consumption records with:
                - facility_id: str
                - egrid_subregion: str (CAMX for California)
                - quantity_kwh: float (electricity consumed)
                - reporting_period_start: str (ISO date)
                - reporting_period_end: str (ISO date)

        Returns:
            CalculationResult with:
                - total_emissions_kg_co2e
                - total_emissions_mt_co2e
                - emissions_by_source (by eGRID subregion)
                - audit_records (complete provenance)

        Raises:
            ValueError: If eGRID subregion is unknown
        """
        total_emissions_kg = 0.0
        total_kwh = 0.0
        emissions_by_region: Dict[str, Dict[str, Any]] = {}
        audit_records: List[AuditRecord] = []

        for input_dict in inputs:
            # Parse input
            input_data = self._parse_input(input_dict)

            # Validate eGRID subregion
            subregion = input_data.egrid_subregion.upper()
            if subregion not in self.EGRID_FACTORS:
                raise ValueError(
                    f"Unknown eGRID subregion: {input_data.egrid_subregion}. "
                    f"Supported regions: {list(self.EGRID_FACTORS.keys())}"
                )

            # Get grid factor
            grid_data = self.EGRID_FACTORS[subregion]
            grid_factor = grid_data.factor

            # Validate input
            self.validate_positive(input_data.quantity_kwh, "quantity_kwh")

            # Calculate emissions (DETERMINISTIC)
            emissions_kg = input_data.quantity_kwh * grid_factor

            # Aggregate totals
            total_emissions_kg += emissions_kg
            total_kwh += input_data.quantity_kwh

            # Track by region
            if subregion not in emissions_by_region:
                emissions_by_region[subregion] = {
                    "emissions_kg_co2e": 0.0,
                    "kwh": 0.0,
                    "factor": grid_factor,
                    "region_name": grid_data.name,
                }
            emissions_by_region[subregion]["emissions_kg_co2e"] += emissions_kg
            emissions_by_region[subregion]["kwh"] += input_data.quantity_kwh

            # Create audit record
            audit_record = self._create_audit_record(
                input_data=input_data,
                grid_data=grid_data,
                emissions_kg=emissions_kg
            )
            audit_records.append(audit_record)

        # Convert to metric tonnes
        total_emissions_mt = total_emissions_kg / 1000.0

        return CalculationResult(
            success=True,
            scope="2",
            category="location_based",
            total_emissions_kg_co2e=self.round_emissions(total_emissions_kg),
            total_emissions_mt_co2e=self.round_emissions(total_emissions_mt, 3),
            emissions_by_source=emissions_by_region,
            audit_records=audit_records,
            calculation_timestamp=self.get_timestamp(),
            calculator_id=self.CALCULATOR_ID,
            calculator_version=self.CALCULATOR_VERSION,
            metadata={
                "total_kwh": total_kwh,
                "method": "location_based",
                "factor_source": "EPA eGRID 2023",
                "regions_included": list(emissions_by_region.keys()),
            }
        )

    def _parse_input(self, input_dict: Dict[str, Any]) -> ElectricityInput:
        """Parse dictionary input into typed dataclass."""
        return ElectricityInput(
            facility_id=input_dict["facility_id"],
            egrid_subregion=input_dict["egrid_subregion"],
            quantity_kwh=float(input_dict["quantity_kwh"]),
            reporting_period_start=input_dict["reporting_period_start"],
            reporting_period_end=input_dict["reporting_period_end"],
            source_document_id=input_dict.get("source_document_id"),
            utility_name=input_dict.get("utility_name"),
        )

    def _create_audit_record(
        self,
        input_data: ElectricityInput,
        grid_data: GridFactorData,
        emissions_kg: float
    ) -> AuditRecord:
        """Create audit record with SHA-256 provenance hash."""

        # Deterministic input dictionary
        input_dict = {
            "facility_id": input_data.facility_id,
            "egrid_subregion": input_data.egrid_subregion,
            "quantity_kwh": input_data.quantity_kwh,
            "reporting_period_start": input_data.reporting_period_start,
            "reporting_period_end": input_data.reporting_period_end,
        }

        input_hash = hashlib.sha256(
            json.dumps(input_dict, sort_keys=True).encode()
        ).hexdigest()

        # Deterministic output dictionary
        output_dict = {
            "emissions_kg_co2e": self.round_emissions(emissions_kg),
        }

        output_hash = hashlib.sha256(
            json.dumps(output_dict, sort_keys=True).encode()
        ).hexdigest()

        return AuditRecord(
            calculation_id=f"{self.CALCULATOR_ID}-{input_hash[:12]}",
            timestamp=self.get_timestamp(),
            scope="2",
            category="location_based",
            input_hash=input_hash,
            output_hash=output_hash,
            emission_factor_source=grid_data.source,
            emission_factor_version="2023",
            emission_factor_value=grid_data.factor,
            emission_factor_unit="kg CO2e/kWh",
            gwp_basis="IPCC AR6",
            calculation_formula=(
                f"emissions = {input_data.quantity_kwh:.2f} kWh x "
                f"{grid_data.factor} kg CO2e/kWh = {emissions_kg:.4f} kg CO2e"
            ),
            inputs=input_dict,
            outputs=output_dict
        )

    def get_california_factor(self) -> float:
        """Get California (CAMX) grid emission factor."""
        return self.EGRID_FACTORS["CAMX"].factor

    def get_supported_regions(self) -> List[str]:
        """Get list of supported eGRID subregions."""
        return list(self.EGRID_FACTORS.keys())

    def get_region_factor(self, subregion: str) -> Optional[float]:
        """Get emission factor for a specific subregion."""
        grid_data = self.EGRID_FACTORS.get(subregion.upper())
        return grid_data.factor if grid_data else None
