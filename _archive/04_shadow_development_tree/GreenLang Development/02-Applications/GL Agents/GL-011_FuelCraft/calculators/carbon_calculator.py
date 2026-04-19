"""
GL-011 FuelCraft - Carbon Calculator

Deterministic carbon intensity and emissions calculations:
- Energy-weighted blend intensity
- TTW/WTT/WTW boundary emissions
- kgCO2e/MJ calculations
- Emission factor governance with effective dates

Standards:
- GHG Protocol (Scope 1, 2, 3)
- IMO CII (Carbon Intensity Indicator)
- EU FuelEU Maritime
- CORSIA (Aviation)
- IPCC 2006 Guidelines
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json


class EmissionBoundary(Enum):
    """Lifecycle emission boundary."""
    TTW = "TTW"    # Tank-to-Wake (direct combustion only)
    WTT = "WTT"    # Well-to-Tank (upstream/production)
    WTW = "WTW"    # Well-to-Wake (full lifecycle)


class EmissionScope(Enum):
    """GHG Protocol emission scope."""
    SCOPE_1 = "scope_1"   # Direct emissions
    SCOPE_2 = "scope_2"   # Indirect (electricity)
    SCOPE_3 = "scope_3"   # Value chain


class GHGType(Enum):
    """Greenhouse gas types."""
    CO2 = "CO2"       # Carbon dioxide
    CH4 = "CH4"       # Methane
    N2O = "N2O"       # Nitrous oxide
    CO2E = "CO2e"     # CO2 equivalent


@dataclass(frozen=True)
class EmissionFactor:
    """
    Emission factor with full provenance.

    All factors in kgCO2e per unit (MJ, kg, m3, etc.)
    """
    factor_id: str
    fuel_type: str
    boundary: EmissionBoundary
    factor_value: Decimal       # kgCO2e/unit
    factor_unit: str            # Base unit (MJ, kg, L, etc.)
    ghg_type: GHGType
    scope: EmissionScope
    source_standard: str        # IPCC, DEFRA, EPA, etc.
    effective_date: date
    expiry_date: Optional[date]
    region: str                 # Global, EU, US, etc.
    uncertainty_pct: Decimal    # Uncertainty percentage
    notes: str = ""

    def is_valid(self, check_date: date) -> bool:
        """Check if factor is valid for given date."""
        if check_date < self.effective_date:
            return False
        if self.expiry_date and check_date > self.expiry_date:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "fuel_type": self.fuel_type,
            "boundary": self.boundary.value,
            "factor_value": str(self.factor_value),
            "factor_unit": self.factor_unit,
            "ghg_type": self.ghg_type.value,
            "scope": self.scope.value,
            "source_standard": self.source_standard,
            "effective_date": self.effective_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "region": self.region,
            "uncertainty_pct": str(self.uncertainty_pct),
            "notes": self.notes
        }


# Default emission factors database
# Source: IPCC 2006, IMO 2021, DEFRA 2023
DEFAULT_EMISSION_FACTORS: Dict[str, List[EmissionFactor]] = {
    "diesel": [
        EmissionFactor(
            factor_id="diesel_ttw_ipcc2006",
            fuel_type="diesel",
            boundary=EmissionBoundary.TTW,
            factor_value=Decimal("0.0741"),  # kgCO2e/MJ
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_1,
            source_standard="IPCC 2006",
            effective_date=date(2006, 1, 1),
            expiry_date=None,
            region="Global",
            uncertainty_pct=Decimal("5.0"),
            notes="Default diesel combustion factor"
        ),
        EmissionFactor(
            factor_id="diesel_wtt_eu",
            fuel_type="diesel",
            boundary=EmissionBoundary.WTT,
            factor_value=Decimal("0.0161"),  # kgCO2e/MJ
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_3,
            source_standard="JEC v5",
            effective_date=date(2020, 1, 1),
            expiry_date=None,
            region="EU",
            uncertainty_pct=Decimal("20.0"),
            notes="EU diesel upstream emissions"
        ),
    ],
    "heavy_fuel_oil": [
        EmissionFactor(
            factor_id="hfo_ttw_imo",
            fuel_type="heavy_fuel_oil",
            boundary=EmissionBoundary.TTW,
            factor_value=Decimal("0.0771"),  # kgCO2e/MJ (3.114 / 40.4)
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_1,
            source_standard="IMO MEPC.364(79)",
            effective_date=date(2022, 6, 17),
            expiry_date=None,
            region="Global",
            uncertainty_pct=Decimal("2.0"),
            notes="IMO default HFO factor"
        ),
        EmissionFactor(
            factor_id="hfo_wtt_global",
            fuel_type="heavy_fuel_oil",
            boundary=EmissionBoundary.WTT,
            factor_value=Decimal("0.0124"),  # kgCO2e/MJ
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_3,
            source_standard="GREET 2022",
            effective_date=date(2022, 1, 1),
            expiry_date=None,
            region="Global",
            uncertainty_pct=Decimal("25.0")
        ),
    ],
    "marine_fuel_oil": [
        EmissionFactor(
            factor_id="mfo_ttw_imo",
            fuel_type="marine_fuel_oil",
            boundary=EmissionBoundary.TTW,
            factor_value=Decimal("0.0773"),  # kgCO2e/MJ (3.114 / 40.2)
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_1,
            source_standard="IMO MEPC.364(79)",
            effective_date=date(2022, 6, 17),
            expiry_date=None,
            region="Global",
            uncertainty_pct=Decimal("2.0")
        ),
    ],
    "natural_gas": [
        EmissionFactor(
            factor_id="lng_ttw_imo",
            fuel_type="natural_gas",
            boundary=EmissionBoundary.TTW,
            factor_value=Decimal("0.0561"),  # kgCO2e/MJ
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_1,
            source_standard="IMO MEPC.364(79)",
            effective_date=date(2022, 6, 17),
            expiry_date=None,
            region="Global",
            uncertainty_pct=Decimal("5.0"),
            notes="LNG combustion including methane slip"
        ),
        EmissionFactor(
            factor_id="lng_wtt_global",
            fuel_type="natural_gas",
            boundary=EmissionBoundary.WTT,
            factor_value=Decimal("0.0183"),  # kgCO2e/MJ
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_3,
            source_standard="GREET 2022",
            effective_date=date(2022, 1, 1),
            expiry_date=None,
            region="Global",
            uncertainty_pct=Decimal("30.0")
        ),
    ],
    "hydrogen": [
        EmissionFactor(
            factor_id="h2_green_ttw",
            fuel_type="hydrogen",
            boundary=EmissionBoundary.TTW,
            factor_value=Decimal("0.0000"),  # Zero combustion CO2
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_1,
            source_standard="IMO MEPC",
            effective_date=date(2023, 1, 1),
            expiry_date=None,
            region="Global",
            uncertainty_pct=Decimal("0.0"),
            notes="Green hydrogen - zero combustion emissions"
        ),
        EmissionFactor(
            factor_id="h2_grey_wtt",
            fuel_type="hydrogen",
            boundary=EmissionBoundary.WTT,
            factor_value=Decimal("0.0833"),  # 10 kgCO2e/kgH2 / 120 MJ/kg
            factor_unit="MJ",
            ghg_type=GHGType.CO2E,
            scope=EmissionScope.SCOPE_3,
            source_standard="IEA 2023",
            effective_date=date(2023, 1, 1),
            expiry_date=None,
            region="Global",
            uncertainty_pct=Decimal("20.0"),
            notes="Grey hydrogen from SMR"
        ),
    ],
}


@dataclass
class CarbonInput:
    """Input for carbon calculation."""
    fuel_type: str
    energy_mj: Decimal
    boundary: EmissionBoundary = EmissionBoundary.WTW
    region: str = "Global"
    reference_date: date = field(default_factory=date.today)


@dataclass
class CarbonResult:
    """
    Result of carbon calculation with provenance.
    """
    fuel_type: str
    energy_mj: Decimal
    boundary: EmissionBoundary
    # Emissions by boundary
    ttw_emissions_kg_co2e: Decimal
    wtt_emissions_kg_co2e: Decimal
    wtw_emissions_kg_co2e: Decimal
    # Carbon intensity
    ttw_intensity_kg_co2e_mj: Decimal
    wtt_intensity_kg_co2e_mj: Decimal
    wtw_intensity_kg_co2e_mj: Decimal
    # Factors used
    factors_used: List[EmissionFactor]
    # Provenance
    provenance_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "fuel_type": self.fuel_type,
            "energy_mj": str(self.energy_mj),
            "boundary": self.boundary.value,
            "ttw_emissions_kg_co2e": str(self.ttw_emissions_kg_co2e),
            "wtt_emissions_kg_co2e": str(self.wtt_emissions_kg_co2e),
            "wtw_emissions_kg_co2e": str(self.wtw_emissions_kg_co2e),
            "factors": [f.factor_id for f in self.factors_used],
            "timestamp": self.timestamp.isoformat()
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fuel_type": self.fuel_type,
            "energy_mj": str(self.energy_mj),
            "boundary": self.boundary.value,
            "ttw_emissions_kg_co2e": str(self.ttw_emissions_kg_co2e),
            "wtt_emissions_kg_co2e": str(self.wtt_emissions_kg_co2e),
            "wtw_emissions_kg_co2e": str(self.wtw_emissions_kg_co2e),
            "ttw_intensity_kg_co2e_mj": str(self.ttw_intensity_kg_co2e_mj),
            "wtt_intensity_kg_co2e_mj": str(self.wtt_intensity_kg_co2e_mj),
            "wtw_intensity_kg_co2e_mj": str(self.wtw_intensity_kg_co2e_mj),
            "factors_used": [f.to_dict() for f in self.factors_used],
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp.isoformat()
        }


class CarbonCalculator:
    """
    Deterministic carbon intensity calculator.

    Provides ZERO-HALLUCINATION calculations for:
    - TTW/WTT/WTW boundary emissions
    - Energy-weighted blend carbon intensity
    - Emission factor governance with date validity
    - GHG Protocol scope attribution

    All calculations use Decimal arithmetic.
    """

    NAME: str = "CarbonCalculator"
    VERSION: str = "1.0.0"

    PRECISION: int = 6

    def __init__(
        self,
        emission_factors: Optional[Dict[str, List[EmissionFactor]]] = None
    ):
        """
        Initialize calculator.

        Args:
            emission_factors: Custom emission factor database.
                            Defaults to DEFAULT_EMISSION_FACTORS.
        """
        self._factors = emission_factors or DEFAULT_EMISSION_FACTORS

    def calculate(
        self,
        carbon_input: CarbonInput,
        precision: int = 6
    ) -> CarbonResult:
        """
        Calculate emissions for fuel consumption - DETERMINISTIC.

        Args:
            carbon_input: Input parameters
            precision: Output decimal places

        Returns:
            CarbonResult with full provenance

        Raises:
            ValueError: If fuel type not found or no valid factor
        """
        fuel_type = carbon_input.fuel_type
        energy_mj = carbon_input.energy_mj
        ref_date = carbon_input.reference_date
        region = carbon_input.region

        if fuel_type not in self._factors:
            raise ValueError(f"Unknown fuel type: {fuel_type}")

        steps: List[Dict[str, Any]] = []
        factors_used: List[EmissionFactor] = []

        # Step 1: Get TTW factor
        ttw_factor = self._get_factor(fuel_type, EmissionBoundary.TTW, ref_date, region)
        if ttw_factor is None:
            raise ValueError(f"No valid TTW factor for {fuel_type} on {ref_date}")

        factors_used.append(ttw_factor)
        ttw_emissions = energy_mj * ttw_factor.factor_value
        ttw_intensity = ttw_factor.factor_value

        steps.append({
            "step": 1,
            "operation": "calculate_ttw",
            "factor_id": ttw_factor.factor_id,
            "factor_value": str(ttw_factor.factor_value),
            "emissions_kg_co2e": str(ttw_emissions)
        })

        # Step 2: Get WTT factor
        wtt_factor = self._get_factor(fuel_type, EmissionBoundary.WTT, ref_date, region)
        if wtt_factor is not None:
            factors_used.append(wtt_factor)
            wtt_emissions = energy_mj * wtt_factor.factor_value
            wtt_intensity = wtt_factor.factor_value
        else:
            wtt_emissions = Decimal("0")
            wtt_intensity = Decimal("0")

        steps.append({
            "step": 2,
            "operation": "calculate_wtt",
            "factor_id": wtt_factor.factor_id if wtt_factor else None,
            "factor_value": str(wtt_intensity),
            "emissions_kg_co2e": str(wtt_emissions)
        })

        # Step 3: Calculate WTW (total lifecycle)
        wtw_emissions = ttw_emissions + wtt_emissions
        wtw_intensity = ttw_intensity + wtt_intensity

        steps.append({
            "step": 3,
            "operation": "calculate_wtw",
            "wtw_emissions_kg_co2e": str(wtw_emissions),
            "wtw_intensity_kg_co2e_mj": str(wtw_intensity)
        })

        # Apply precision
        quantize_str = "0." + "0" * precision

        return CarbonResult(
            fuel_type=fuel_type,
            energy_mj=energy_mj,
            boundary=carbon_input.boundary,
            ttw_emissions_kg_co2e=ttw_emissions.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            wtt_emissions_kg_co2e=wtt_emissions.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            wtw_emissions_kg_co2e=wtw_emissions.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            ttw_intensity_kg_co2e_mj=ttw_intensity.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            wtt_intensity_kg_co2e_mj=wtt_intensity.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            wtw_intensity_kg_co2e_mj=wtw_intensity.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            factors_used=factors_used,
            calculation_steps=steps
        )

    def calculate_blend_intensity(
        self,
        fuel_energies: List[Tuple[str, Decimal]],  # (fuel_type, energy_mj)
        boundary: EmissionBoundary = EmissionBoundary.WTW,
        reference_date: date = None,
        region: str = "Global"
    ) -> Tuple[Decimal, List[Dict[str, Any]]]:
        """
        Calculate energy-weighted blend carbon intensity.

        CI_blend = sum(E_i * CI_i) / sum(E_i)

        Args:
            fuel_energies: List of (fuel_type, energy_mj) tuples
            boundary: Emission boundary (TTW/WTT/WTW)
            reference_date: Date for factor validity
            region: Region for factor selection

        Returns:
            Tuple of (blend_intensity, calculation_details)
        """
        ref_date = reference_date or date.today()
        total_weighted_emissions = Decimal("0")
        total_energy = Decimal("0")
        details: List[Dict[str, Any]] = []

        for fuel_type, energy_mj in fuel_energies:
            # Get appropriate factor
            if boundary == EmissionBoundary.TTW:
                factor = self._get_factor(fuel_type, EmissionBoundary.TTW, ref_date, region)
                intensity = factor.factor_value if factor else Decimal("0")
            elif boundary == EmissionBoundary.WTT:
                factor = self._get_factor(fuel_type, EmissionBoundary.WTT, ref_date, region)
                intensity = factor.factor_value if factor else Decimal("0")
            else:  # WTW
                ttw = self._get_factor(fuel_type, EmissionBoundary.TTW, ref_date, region)
                wtt = self._get_factor(fuel_type, EmissionBoundary.WTT, ref_date, region)
                intensity = (ttw.factor_value if ttw else Decimal("0")) + \
                           (wtt.factor_value if wtt else Decimal("0"))

            emissions = energy_mj * intensity
            total_weighted_emissions += emissions
            total_energy += energy_mj

            details.append({
                "fuel_type": fuel_type,
                "energy_mj": str(energy_mj),
                "intensity_kg_co2e_mj": str(intensity),
                "emissions_kg_co2e": str(emissions)
            })

        if total_energy == Decimal("0"):
            blend_intensity = Decimal("0")
        else:
            blend_intensity = total_weighted_emissions / total_energy

        return blend_intensity.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), details

    def _get_factor(
        self,
        fuel_type: str,
        boundary: EmissionBoundary,
        ref_date: date,
        region: str
    ) -> Optional[EmissionFactor]:
        """
        Get valid emission factor for fuel and boundary.

        Priority:
        1. Region-specific factor valid on date
        2. Global factor valid on date
        3. None if no valid factor

        Args:
            fuel_type: Fuel type
            boundary: Emission boundary
            ref_date: Reference date
            region: Region preference

        Returns:
            EmissionFactor or None
        """
        if fuel_type not in self._factors:
            return None

        factors = self._factors[fuel_type]

        # Filter by boundary and validity
        valid_factors = [
            f for f in factors
            if f.boundary == boundary and f.is_valid(ref_date)
        ]

        if not valid_factors:
            return None

        # Prefer region-specific
        region_factors = [f for f in valid_factors if f.region == region]
        if region_factors:
            # Return most recent
            return sorted(region_factors, key=lambda f: f.effective_date, reverse=True)[0]

        # Fall back to global
        global_factors = [f for f in valid_factors if f.region == "Global"]
        if global_factors:
            return sorted(global_factors, key=lambda f: f.effective_date, reverse=True)[0]

        # Return any valid factor
        return sorted(valid_factors, key=lambda f: f.effective_date, reverse=True)[0]

    def get_factor(
        self,
        fuel_type: str,
        boundary: EmissionBoundary,
        reference_date: date = None,
        region: str = "Global"
    ) -> Optional[EmissionFactor]:
        """Public interface to get emission factor."""
        ref_date = reference_date or date.today()
        return self._get_factor(fuel_type, boundary, ref_date, region)

    def list_fuels(self) -> List[str]:
        """List available fuel types."""
        return list(self._factors.keys())

    def list_factors(self, fuel_type: str) -> List[EmissionFactor]:
        """List all factors for a fuel type."""
        return self._factors.get(fuel_type, [])
