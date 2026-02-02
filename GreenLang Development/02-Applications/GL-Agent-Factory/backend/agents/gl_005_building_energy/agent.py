"""
GL-005: Building Energy Agent

This module implements the Building Energy Agent for calculating
building energy consumption and emissions aligned with EPBD, CRREM,
and energy performance certificate (EPC) requirements.

The agent supports:
- Energy Use Intensity (EUI) calculation
- GHG emissions from building operations
- CRREM pathway alignment analysis
- Stranding risk assessment
- Energy Performance Certificate (EPC) rating

Example:
    >>> agent = BuildingEnergyAgent()
    >>> result = agent.run(BuildingEnergyInput(
    ...     building_type=BuildingType.OFFICE,
    ...     floor_area_sqm=10000,
    ...     annual_electricity_kwh=500000,
    ...     annual_gas_m3=50000,
    ...     region="DE"
    ... ))
    >>> print(f"EUI: {result.data.eui_kwh_sqm} kWh/m2")
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class BuildingType(str, Enum):
    """Building type classifications."""

    OFFICE = "office"
    RETAIL = "retail"
    HOTEL = "hotel"
    RESIDENTIAL = "residential"
    INDUSTRIAL = "industrial"
    WAREHOUSE = "warehouse"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    DATA_CENTER = "data_center"


class EPCRating(str, Enum):
    """Energy Performance Certificate ratings."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class StrandingRisk(str, Enum):
    """CRREM stranding risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    STRANDED = "stranded"


class BuildingEnergyInput(BaseModel):
    """
    Input model for Building Energy Agent.

    Attributes:
        building_id: Unique building identifier
        building_type: Type of building
        floor_area_sqm: Gross floor area in square meters
        year_built: Year of construction
        region: Geographic region (ISO 3166)
        annual_electricity_kwh: Annual electricity consumption
        annual_gas_m3: Annual natural gas consumption
        annual_district_heating_kwh: Annual district heating
        annual_district_cooling_kwh: Annual district cooling
        renewable_generation_kwh: On-site renewable generation
        occupancy_rate: Average occupancy percentage
        operating_hours: Annual operating hours
    """

    building_id: Optional[str] = Field(None, description="Building identifier")
    building_type: BuildingType = Field(..., description="Building type")
    floor_area_sqm: float = Field(..., ge=1, description="Floor area in m2")
    year_built: Optional[int] = Field(None, ge=1800, le=2030, description="Construction year")
    region: str = Field("EU", min_length=2, max_length=3, description="Region code")

    # Energy consumption
    annual_electricity_kwh: float = Field(0, ge=0, description="Electricity kWh/year")
    annual_gas_m3: float = Field(0, ge=0, description="Natural gas m3/year")
    annual_district_heating_kwh: float = Field(0, ge=0, description="District heating kWh/year")
    annual_district_cooling_kwh: float = Field(0, ge=0, description="District cooling kWh/year")
    renewable_generation_kwh: float = Field(0, ge=0, description="On-site renewables kWh/year")

    # Operational data
    occupancy_rate: float = Field(100, ge=0, le=100, description="Occupancy %")
    operating_hours: Optional[int] = Field(None, ge=0, le=8760, description="Hours/year")

    # Analysis parameters
    target_year: int = Field(2050, ge=2024, le=2100, description="Target year for pathway")

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("region")
    def validate_region(cls, v: str) -> str:
        """Validate region code."""
        return v.upper()


class BuildingEnergyOutput(BaseModel):
    """
    Output model for Building Energy Agent.

    Includes EUI, emissions, and pathway analysis.
    """

    building_type: str = Field(..., description="Building type")
    floor_area_sqm: float = Field(..., description="Floor area")

    # Energy metrics
    total_energy_kwh: float = Field(..., description="Total energy consumption")
    eui_kwh_sqm: float = Field(..., description="Energy Use Intensity kWh/m2")
    electricity_kwh: float = Field(..., description="Electricity consumption")
    gas_kwh: float = Field(..., description="Gas consumption (converted)")
    renewable_share_pct: float = Field(..., description="Renewable energy %")

    # Emissions metrics
    total_emissions_kgco2e: float = Field(..., description="Total GHG emissions")
    emissions_intensity_kgco2e_sqm: float = Field(..., description="Emissions intensity")
    scope1_emissions: float = Field(..., description="Scope 1 (on-site combustion)")
    scope2_emissions: float = Field(..., description="Scope 2 (electricity)")

    # Performance ratings
    epc_rating: str = Field(..., description="EPC rating estimate")
    epc_score: float = Field(..., description="EPC score 0-100")

    # CRREM pathway analysis
    crrem_target_intensity: float = Field(..., description="CRREM 2050 target kgCO2e/m2")
    crrem_excess_intensity: float = Field(..., description="Excess above pathway")
    stranding_year: Optional[int] = Field(None, description="Projected stranding year")
    stranding_risk: str = Field(..., description="Stranding risk level")
    decarbonization_gap_pct: float = Field(..., description="Gap to 2050 target %")

    # Recommendations
    improvement_potential_kwh: float = Field(..., description="Potential energy savings")
    improvement_potential_pct: float = Field(..., description="Potential savings %")

    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class EnergyFactor(BaseModel):
    """Energy conversion and emission factor."""

    value: float
    unit: str
    source: str


class BuildingEnergyAgent:
    """
    GL-005: Building Energy Agent.

    This agent calculates building energy performance and carbon emissions
    using zero-hallucination deterministic calculations:
    - EUI = total_energy / floor_area
    - Emissions = energy * emission_factor

    Aligned with:
    - EPBD (Energy Performance of Buildings Directive)
    - CRREM (Carbon Risk Real Estate Monitor)
    - ISO 52000 series

    Attributes:
        emission_factors: Grid emission factors by region
        epc_thresholds: EPC rating thresholds
        crrem_pathways: CRREM decarbonization pathways

    Example:
        >>> agent = BuildingEnergyAgent()
        >>> result = agent.run(BuildingEnergyInput(
        ...     building_type=BuildingType.OFFICE,
        ...     floor_area_sqm=10000,
        ...     annual_electricity_kwh=500000
        ... ))
        >>> assert result.eui_kwh_sqm > 0
    """

    AGENT_ID = "buildings/energy_performance_v1"
    VERSION = "1.0.0"
    DESCRIPTION = "Building energy and emissions calculator with CRREM alignment"

    # Natural gas to kWh conversion (1 m3 = 10.55 kWh)
    GAS_M3_TO_KWH = 10.55

    # Grid emission factors (kgCO2e/kWh)
    GRID_EMISSION_FACTORS: Dict[str, EnergyFactor] = {
        "DE": EnergyFactor(value=0.366, unit="kgCO2e/kWh", source="IEA 2024"),
        "FR": EnergyFactor(value=0.052, unit="kgCO2e/kWh", source="IEA 2024"),
        "UK": EnergyFactor(value=0.207, unit="kgCO2e/kWh", source="DEFRA 2024"),
        "US": EnergyFactor(value=0.417, unit="kgCO2e/kWh", source="EPA eGRID 2024"),
        "EU": EnergyFactor(value=0.276, unit="kgCO2e/kWh", source="IEA 2024"),
        "NL": EnergyFactor(value=0.328, unit="kgCO2e/kWh", source="IEA 2024"),
        "ES": EnergyFactor(value=0.182, unit="kgCO2e/kWh", source="IEA 2024"),
        "IT": EnergyFactor(value=0.256, unit="kgCO2e/kWh", source="IEA 2024"),
        "PL": EnergyFactor(value=0.635, unit="kgCO2e/kWh", source="IEA 2024"),
    }

    # Natural gas emission factor
    GAS_EMISSION_FACTOR = 0.185  # kgCO2e/kWh (IPCC)

    # District heating emission factor (varies by region, using EU average)
    DISTRICT_HEATING_FACTOR = 0.150  # kgCO2e/kWh

    # District cooling emission factor
    DISTRICT_COOLING_FACTOR = 0.100  # kgCO2e/kWh

    # EPC rating thresholds (kWh/m2/year)
    EPC_THRESHOLDS: Dict[BuildingType, Dict[EPCRating, float]] = {
        BuildingType.OFFICE: {
            EPCRating.A_PLUS: 50,
            EPCRating.A: 75,
            EPCRating.B: 100,
            EPCRating.C: 135,
            EPCRating.D: 175,
            EPCRating.E: 225,
            EPCRating.F: 300,
            EPCRating.G: float("inf"),
        },
        BuildingType.RETAIL: {
            EPCRating.A_PLUS: 100,
            EPCRating.A: 150,
            EPCRating.B: 200,
            EPCRating.C: 270,
            EPCRating.D: 350,
            EPCRating.E: 450,
            EPCRating.F: 600,
            EPCRating.G: float("inf"),
        },
        BuildingType.HOTEL: {
            EPCRating.A_PLUS: 80,
            EPCRating.A: 120,
            EPCRating.B: 160,
            EPCRating.C: 216,
            EPCRating.D: 280,
            EPCRating.E: 360,
            EPCRating.F: 480,
            EPCRating.G: float("inf"),
        },
    }

    # CRREM 2050 target intensities (kgCO2e/m2/year)
    CRREM_TARGETS_2050: Dict[BuildingType, float] = {
        BuildingType.OFFICE: 4.5,
        BuildingType.RETAIL: 8.0,
        BuildingType.HOTEL: 7.5,
        BuildingType.RESIDENTIAL: 3.0,
        BuildingType.INDUSTRIAL: 12.0,
        BuildingType.WAREHOUSE: 6.0,
        BuildingType.HEALTHCARE: 15.0,
        BuildingType.EDUCATION: 5.0,
        BuildingType.DATA_CENTER: 50.0,
    }

    # Typical EUI benchmarks (kWh/m2/year)
    EUI_BENCHMARKS: Dict[BuildingType, Dict[str, float]] = {
        BuildingType.OFFICE: {"typical": 180, "best": 70, "worst": 350},
        BuildingType.RETAIL: {"typical": 350, "best": 150, "worst": 600},
        BuildingType.HOTEL: {"typical": 280, "best": 120, "worst": 450},
        BuildingType.RESIDENTIAL: {"typical": 120, "best": 40, "worst": 250},
        BuildingType.INDUSTRIAL: {"typical": 250, "best": 100, "worst": 500},
        BuildingType.WAREHOUSE: {"typical": 150, "best": 50, "worst": 300},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Building Energy Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict] = []

        logger.info(f"BuildingEnergyAgent initialized (version {self.VERSION})")

    def run(self, input_data: BuildingEnergyInput) -> BuildingEnergyOutput:
        """
        Execute the building energy calculation.

        ZERO-HALLUCINATION calculations:
        - EUI = total_energy / floor_area
        - Emissions = SUM(energy_source * emission_factor)

        Args:
            input_data: Validated building input data

        Returns:
            Calculation result with energy metrics and ratings
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []

        logger.info(
            f"Calculating building energy: type={input_data.building_type}, "
            f"area={input_data.floor_area_sqm}m2, region={input_data.region}"
        )

        try:
            # Step 1: Convert all energy to kWh
            electricity_kwh = input_data.annual_electricity_kwh
            gas_kwh = input_data.annual_gas_m3 * self.GAS_M3_TO_KWH
            district_heating_kwh = input_data.annual_district_heating_kwh
            district_cooling_kwh = input_data.annual_district_cooling_kwh

            total_energy = (
                electricity_kwh +
                gas_kwh +
                district_heating_kwh +
                district_cooling_kwh
            )

            self._track_step("energy_conversion", {
                "electricity_kwh": electricity_kwh,
                "gas_m3": input_data.annual_gas_m3,
                "gas_kwh": gas_kwh,
                "district_heating_kwh": district_heating_kwh,
                "district_cooling_kwh": district_cooling_kwh,
                "total_energy_kwh": total_energy,
            })

            # Step 2: ZERO-HALLUCINATION CALCULATION
            # EUI = total_energy / floor_area
            eui = total_energy / input_data.floor_area_sqm

            self._track_step("eui_calculation", {
                "formula": "EUI = total_energy / floor_area",
                "total_energy": total_energy,
                "floor_area": input_data.floor_area_sqm,
                "eui": eui,
            })

            # Step 3: Calculate renewable share
            renewable_share = 0.0
            if total_energy > 0:
                renewable_share = (input_data.renewable_generation_kwh / total_energy) * 100

            # Step 4: Calculate emissions
            grid_factor = self._get_grid_factor(input_data.region)

            scope2_emissions = electricity_kwh * grid_factor.value
            scope1_emissions = gas_kwh * self.GAS_EMISSION_FACTOR
            district_emissions = (
                district_heating_kwh * self.DISTRICT_HEATING_FACTOR +
                district_cooling_kwh * self.DISTRICT_COOLING_FACTOR
            )

            total_emissions = scope1_emissions + scope2_emissions + district_emissions
            emissions_intensity = total_emissions / input_data.floor_area_sqm

            self._track_step("emissions_calculation", {
                "scope1_emissions": scope1_emissions,
                "scope2_emissions": scope2_emissions,
                "district_emissions": district_emissions,
                "total_emissions": total_emissions,
                "emissions_intensity": emissions_intensity,
                "grid_factor_used": grid_factor.value,
            })

            # Step 5: Calculate EPC rating
            epc_rating, epc_score = self._calculate_epc_rating(
                input_data.building_type,
                eui,
            )

            self._track_step("epc_rating", {
                "eui": eui,
                "rating": epc_rating.value,
                "score": epc_score,
            })

            # Step 6: CRREM pathway analysis
            crrem_target = self.CRREM_TARGETS_2050.get(
                input_data.building_type, 10.0
            )
            crrem_excess = max(0, emissions_intensity - crrem_target)
            stranding_year, stranding_risk = self._calculate_stranding_risk(
                emissions_intensity,
                crrem_target,
                input_data.target_year,
            )
            decarbonization_gap = ((emissions_intensity - crrem_target) / emissions_intensity * 100) if emissions_intensity > 0 else 0

            self._track_step("crrem_analysis", {
                "crrem_target_2050": crrem_target,
                "current_intensity": emissions_intensity,
                "excess_intensity": crrem_excess,
                "stranding_year": stranding_year,
                "stranding_risk": stranding_risk.value,
                "decarbonization_gap_pct": decarbonization_gap,
            })

            # Step 7: Calculate improvement potential
            benchmark = self.EUI_BENCHMARKS.get(input_data.building_type, {})
            best_eui = benchmark.get("best", eui * 0.5)
            improvement_kwh = max(0, (eui - best_eui) * input_data.floor_area_sqm)
            improvement_pct = ((eui - best_eui) / eui * 100) if eui > 0 else 0

            # Step 8: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 9: Create output
            output = BuildingEnergyOutput(
                building_type=input_data.building_type.value,
                floor_area_sqm=input_data.floor_area_sqm,
                total_energy_kwh=round(total_energy, 2),
                eui_kwh_sqm=round(eui, 2),
                electricity_kwh=round(electricity_kwh, 2),
                gas_kwh=round(gas_kwh, 2),
                renewable_share_pct=round(renewable_share, 2),
                total_emissions_kgco2e=round(total_emissions, 2),
                emissions_intensity_kgco2e_sqm=round(emissions_intensity, 4),
                scope1_emissions=round(scope1_emissions, 2),
                scope2_emissions=round(scope2_emissions, 2),
                epc_rating=epc_rating.value,
                epc_score=round(epc_score, 1),
                crrem_target_intensity=crrem_target,
                crrem_excess_intensity=round(crrem_excess, 4),
                stranding_year=stranding_year,
                stranding_risk=stranding_risk.value,
                decarbonization_gap_pct=round(max(0, decarbonization_gap), 2),
                improvement_potential_kwh=round(improvement_kwh, 2),
                improvement_potential_pct=round(max(0, improvement_pct), 2),
                provenance_hash=provenance_hash,
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"Building energy calculation complete: EUI={eui:.1f} kWh/m2, "
                f"emissions={emissions_intensity:.2f} kgCO2e/m2, EPC={epc_rating.value} "
                f"(duration: {duration_ms:.2f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"Building energy calculation failed: {str(e)}", exc_info=True)
            raise

    def _get_grid_factor(self, region: str) -> EnergyFactor:
        """Get grid emission factor for region."""
        return self.GRID_EMISSION_FACTORS.get(
            region,
            self.GRID_EMISSION_FACTORS["EU"]
        )

    def _calculate_epc_rating(
        self,
        building_type: BuildingType,
        eui: float,
    ) -> Tuple[EPCRating, float]:
        """
        Calculate EPC rating based on EUI.

        ZERO-HALLUCINATION: Uses deterministic threshold lookup.
        """
        thresholds = self.EPC_THRESHOLDS.get(
            building_type,
            self.EPC_THRESHOLDS[BuildingType.OFFICE]
        )

        # Find rating band
        for rating in EPCRating:
            if eui <= thresholds.get(rating, float("inf")):
                # Calculate score (0-100 scale, higher is better)
                if rating == EPCRating.A_PLUS:
                    score = 100
                elif rating == EPCRating.G:
                    score = 0
                else:
                    # Linear interpolation within band
                    ratings_list = list(EPCRating)
                    idx = ratings_list.index(rating)
                    score = 100 - (idx * 12.5)

                return rating, score

        return EPCRating.G, 0

    def _calculate_stranding_risk(
        self,
        current_intensity: float,
        target_2050: float,
        target_year: int,
    ) -> Tuple[Optional[int], StrandingRisk]:
        """
        Calculate CRREM stranding year and risk level.

        ZERO-HALLUCINATION: Linear pathway interpolation.
        """
        current_year = datetime.now().year
        years_to_2050 = 2050 - current_year

        if current_intensity <= target_2050:
            return None, StrandingRisk.LOW

        # Calculate required annual reduction
        required_reduction = (current_intensity - target_2050) / years_to_2050

        # Assume standard annual reduction rate of 3%
        typical_reduction = current_intensity * 0.03

        if typical_reduction >= required_reduction:
            return None, StrandingRisk.LOW

        # Calculate stranding year (when current trajectory crosses pathway)
        # Simplified: linear extrapolation
        excess_ratio = current_intensity / target_2050

        if excess_ratio > 3:
            stranding_year = current_year + 5
            risk = StrandingRisk.STRANDED
        elif excess_ratio > 2:
            stranding_year = current_year + 10
            risk = StrandingRisk.HIGH
        elif excess_ratio > 1.5:
            stranding_year = current_year + 15
            risk = StrandingRisk.MEDIUM
        else:
            stranding_year = current_year + 20
            risk = StrandingRisk.LOW

        return stranding_year, risk

    def _track_step(self, step_type: str, data: Dict[str, Any]) -> None:
        """Track a calculation step for provenance."""
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_building_types(self) -> List[str]:
        """Get list of supported building types."""
        return [bt.value for bt in BuildingType]


# Pack specification
PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "buildings/energy_performance_v1",
    "name": "Building Energy Agent",
    "version": "1.0.0",
    "summary": "Building energy consumption and emissions calculator",
    "tags": ["buildings", "energy", "epc", "crrem", "epbd"],
    "owners": ["buildings-team"],
    "compute": {
        "entrypoint": "python://agents.gl_005_building_energy.agent:BuildingEnergyAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "ef://iea/grid-factors/2024"},
        {"ref": "ef://crrem/pathways/2024"},
    ],
    "provenance": {
        "methodology": "ISO 52000, CRREM",
        "enable_audit": True,
    },
}
