"""
CO2e Calculator for GL-003 UNIFIEDSTEAM

Provides comprehensive CO2e calculations for steam system operations,
including fuel-based emissions, electricity emissions, and steam carbon
intensity calculations with full uncertainty quantification.

Author: GL-003 Climate Intelligence Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import logging

from .emission_factors import (
    EmissionFactorDatabase,
    EmissionFactor,
    FuelType,
    GridRegion,
    EmissionScope,
)

logger = logging.getLogger(__name__)


@dataclass
class FuelConsumptionEstimate:
    """
    Fuel consumption estimate with uncertainty.

    Attributes:
        fuel_type: Type of fuel
        consumption_gj: Fuel consumption in GJ
        boiler_efficiency: Boiler efficiency (0-1)
        steam_output_gj: Steam energy output in GJ
        uncertainty_pct: Uncertainty percentage
    """
    fuel_type: FuelType
    consumption_gj: Decimal
    boiler_efficiency: Decimal
    steam_output_gj: Decimal
    uncertainty_pct: Decimal = Decimal("5.0")

    def get_bounds(self) -> Tuple[Decimal, Decimal]:
        """Return uncertainty bounds for fuel consumption."""
        unc = self.consumption_gj * (self.uncertainty_pct / Decimal("100"))
        return (
            self.consumption_gj - unc * Decimal("1.96"),
            self.consumption_gj + unc * Decimal("1.96")
        )


@dataclass
class EmissionsBreakdown:
    """
    Detailed emissions breakdown by source and gas.

    Attributes:
        scope: Emission scope (1, 2, or 3)
        source: Emission source description
        co2_kg: CO2 emissions in kg
        ch4_kg: CH4 emissions in kg
        n2o_kg: N2O emissions in kg
        co2e_kg: Total CO2e emissions in kg
        uncertainty_pct: Uncertainty percentage
    """
    scope: EmissionScope
    source: str
    co2_kg: Decimal
    ch4_kg: Decimal
    n2o_kg: Decimal
    co2e_kg: Decimal
    uncertainty_pct: Decimal
    calculation_hash: str = ""

    def get_bounds(self) -> Tuple[Decimal, Decimal]:
        """Return 95% confidence interval bounds."""
        unc = self.co2e_kg * (self.uncertainty_pct / Decimal("100"))
        return (
            self.co2e_kg - unc * Decimal("1.96"),
            self.co2e_kg + unc * Decimal("1.96")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scope": self.scope.value,
            "source": self.source,
            "co2_kg": str(self.co2_kg),
            "ch4_kg": str(self.ch4_kg),
            "n2o_kg": str(self.n2o_kg),
            "co2e_kg": str(self.co2e_kg),
            "uncertainty_pct": str(self.uncertainty_pct),
            "calculation_hash": self.calculation_hash,
        }


@dataclass
class SteamCarbonIntensity:
    """
    Carbon intensity of steam production.

    Attributes:
        intensity_kg_co2e_per_gj: kg CO2e per GJ of steam
        intensity_kg_co2e_per_tonne: kg CO2e per tonne of steam
        methodology: Calculation methodology used
        fuel_mix: Fuel mix used for calculation
    """
    intensity_kg_co2e_per_gj: Decimal
    intensity_kg_co2e_per_tonne: Decimal
    methodology: str
    fuel_mix: Dict[FuelType, Decimal]
    boiler_efficiency: Decimal
    uncertainty_pct: Decimal = Decimal("10.0")

    def get_intensity_bounds(self) -> Tuple[Decimal, Decimal]:
        """Return bounds for intensity per GJ."""
        unc = self.intensity_kg_co2e_per_gj * (self.uncertainty_pct / Decimal("100"))
        return (
            self.intensity_kg_co2e_per_gj - unc * Decimal("1.96"),
            self.intensity_kg_co2e_per_gj + unc * Decimal("1.96")
        )


@dataclass
class ClimateImpactResult:
    """
    Complete climate impact calculation result.

    Includes all emissions by scope, totals, intensity metrics,
    and full provenance for audit.
    """
    calculation_id: str
    timestamp: datetime
    period_start: datetime
    period_end: datetime

    # Total emissions by scope
    scope_1_total_kg: Decimal
    scope_2_total_kg: Decimal
    scope_3_total_kg: Decimal
    total_co2e_kg: Decimal

    # Detailed breakdown
    emissions_breakdown: List[EmissionsBreakdown]

    # Intensity metrics
    steam_carbon_intensity: SteamCarbonIntensity

    # Energy metrics
    total_steam_gj: Decimal
    total_fuel_gj: Decimal
    total_electricity_kwh: Decimal

    # Uncertainty
    total_uncertainty_pct: Decimal
    total_lower_kg: Decimal
    total_upper_kg: Decimal

    # Provenance
    emission_factor_version: str
    methodology: str
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "calculation_id": self.calculation_id,
            "timestamp": self.timestamp.isoformat(),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "emissions": {
                "scope_1_kg": str(self.scope_1_total_kg),
                "scope_2_kg": str(self.scope_2_total_kg),
                "scope_3_kg": str(self.scope_3_total_kg),
                "total_co2e_kg": str(self.total_co2e_kg),
                "uncertainty_pct": str(self.total_uncertainty_pct),
                "lower_bound_kg": str(self.total_lower_kg),
                "upper_bound_kg": str(self.total_upper_kg),
            },
            "intensity": {
                "kg_co2e_per_gj": str(self.steam_carbon_intensity.intensity_kg_co2e_per_gj),
                "kg_co2e_per_tonne": str(self.steam_carbon_intensity.intensity_kg_co2e_per_tonne),
            },
            "energy": {
                "steam_gj": str(self.total_steam_gj),
                "fuel_gj": str(self.total_fuel_gj),
                "electricity_kwh": str(self.total_electricity_kwh),
            },
            "breakdown": [e.to_dict() for e in self.emissions_breakdown],
            "provenance": {
                "emission_factor_version": self.emission_factor_version,
                "methodology": self.methodology,
                "audit_trail": self.audit_trail,
            },
        }


class CO2eCalculator:
    """
    CO2e Calculator for steam system climate impact.

    Provides deterministic, auditable CO2e calculations with full
    uncertainty quantification for steam system operations.

    Example:
        >>> calculator = CO2eCalculator()
        >>> result = calculator.calculate_fuel_emissions(
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     fuel_consumption_gj=Decimal("1000"),
        ... )
        >>> print(f"Emissions: {result.co2e_kg} kg CO2e")
    """

    # Steam enthalpy reference (kJ/kg at typical conditions)
    STEAM_ENTHALPY_TYPICAL = Decimal("2750")  # Saturated steam at ~150 psig

    def __init__(
        self,
        emission_db: Optional[EmissionFactorDatabase] = None,
        default_boiler_efficiency: Decimal = Decimal("0.82"),
        default_grid_region: GridRegion = GridRegion.US_AVERAGE,
    ):
        """
        Initialize CO2e calculator.

        Args:
            emission_db: Emission factor database (creates default if None)
            default_boiler_efficiency: Default boiler efficiency
            default_grid_region: Default grid region for electricity
        """
        self.emission_db = emission_db or EmissionFactorDatabase()
        self.default_boiler_efficiency = default_boiler_efficiency
        self.default_grid_region = default_grid_region
        self._audit_log: List[Dict[str, Any]] = []

    def calculate_fuel_emissions(
        self,
        fuel_type: FuelType,
        fuel_consumption_gj: Decimal,
        include_uncertainty: bool = True,
    ) -> EmissionsBreakdown:
        """
        Calculate emissions from fuel combustion.

        Args:
            fuel_type: Type of fuel
            fuel_consumption_gj: Fuel consumption in GJ
            include_uncertainty: Include uncertainty bounds

        Returns:
            EmissionsBreakdown with detailed results
        """
        # Get emission factor
        factor = self.emission_db.get_fuel_factor(fuel_type)

        # Calculate emissions
        co2_kg = factor.co2_factor * fuel_consumption_gj
        ch4_kg = factor.ch4_factor * fuel_consumption_gj
        n2o_kg = factor.n2o_factor * fuel_consumption_gj
        co2e_kg = factor.co2e_factor * fuel_consumption_gj

        calc_hash = self._compute_hash(
            fuel_type.value,
            str(fuel_consumption_gj),
            factor.version,
        )

        result = EmissionsBreakdown(
            scope=EmissionScope.SCOPE_1,
            source=f"Fuel combustion: {fuel_type.value}",
            co2_kg=co2_kg.quantize(Decimal("0.01"), ROUND_HALF_UP),
            ch4_kg=ch4_kg.quantize(Decimal("0.0001"), ROUND_HALF_UP),
            n2o_kg=n2o_kg.quantize(Decimal("0.00001"), ROUND_HALF_UP),
            co2e_kg=co2e_kg.quantize(Decimal("0.01"), ROUND_HALF_UP),
            uncertainty_pct=factor.uncertainty_pct,
            calculation_hash=calc_hash,
        )

        self._log_calculation("fuel_emissions", {
            "fuel_type": fuel_type.value,
            "consumption_gj": str(fuel_consumption_gj),
            "co2e_kg": str(result.co2e_kg),
        })

        return result

    def calculate_electricity_emissions(
        self,
        electricity_kwh: Decimal,
        grid_region: Optional[GridRegion] = None,
    ) -> EmissionsBreakdown:
        """
        Calculate emissions from electricity consumption.

        Args:
            electricity_kwh: Electricity consumption in kWh
            grid_region: Grid region (uses default if None)

        Returns:
            EmissionsBreakdown for Scope 2 emissions
        """
        region = grid_region or self.default_grid_region
        factor = self.emission_db.get_grid_factor(region)

        co2e_kg = factor.co2e_factor * electricity_kwh

        calc_hash = self._compute_hash(
            region.value,
            str(electricity_kwh),
            factor.version,
        )

        result = EmissionsBreakdown(
            scope=EmissionScope.SCOPE_2,
            source=f"Electricity: {region.value}",
            co2_kg=co2e_kg.quantize(Decimal("0.01"), ROUND_HALF_UP),
            ch4_kg=Decimal("0"),
            n2o_kg=Decimal("0"),
            co2e_kg=co2e_kg.quantize(Decimal("0.01"), ROUND_HALF_UP),
            uncertainty_pct=factor.uncertainty_pct,
            calculation_hash=calc_hash,
        )

        return result

    def estimate_fuel_from_steam(
        self,
        steam_mass_kg: Decimal,
        steam_enthalpy_kj_kg: Optional[Decimal] = None,
        feedwater_enthalpy_kj_kg: Decimal = Decimal("420"),  # ~100C feedwater
        boiler_efficiency: Optional[Decimal] = None,
        fuel_type: FuelType = FuelType.NATURAL_GAS,
    ) -> FuelConsumptionEstimate:
        """
        Estimate fuel consumption from steam production.

        Args:
            steam_mass_kg: Steam produced in kg
            steam_enthalpy_kj_kg: Steam enthalpy (uses default if None)
            feedwater_enthalpy_kj_kg: Feedwater enthalpy
            boiler_efficiency: Boiler efficiency (uses default if None)
            fuel_type: Type of fuel used

        Returns:
            FuelConsumptionEstimate with uncertainty
        """
        steam_enthalpy = steam_enthalpy_kj_kg or self.STEAM_ENTHALPY_TYPICAL
        efficiency = boiler_efficiency or self.default_boiler_efficiency

        # Calculate steam energy
        steam_energy_kj = steam_mass_kg * (steam_enthalpy - feedwater_enthalpy_kj_kg)
        steam_energy_gj = steam_energy_kj / Decimal("1000000")

        # Calculate fuel required
        fuel_gj = steam_energy_gj / efficiency

        # Estimate uncertainty based on efficiency uncertainty
        # Typical boiler efficiency uncertainty is 2-3%
        efficiency_unc = Decimal("3.0")
        enthalpy_unc = Decimal("2.0")
        combined_unc = (efficiency_unc ** 2 + enthalpy_unc ** 2).sqrt()

        return FuelConsumptionEstimate(
            fuel_type=fuel_type,
            consumption_gj=fuel_gj.quantize(Decimal("0.001"), ROUND_HALF_UP),
            boiler_efficiency=efficiency,
            steam_output_gj=steam_energy_gj.quantize(Decimal("0.001"), ROUND_HALF_UP),
            uncertainty_pct=combined_unc,
        )

    def calculate_steam_carbon_intensity(
        self,
        fuel_mix: Dict[FuelType, Decimal],
        boiler_efficiency: Optional[Decimal] = None,
    ) -> SteamCarbonIntensity:
        """
        Calculate carbon intensity of steam production.

        Args:
            fuel_mix: Dictionary of fuel types and their percentage shares (0-1)
            boiler_efficiency: Overall boiler efficiency

        Returns:
            SteamCarbonIntensity with intensity metrics
        """
        efficiency = boiler_efficiency or self.default_boiler_efficiency

        # Validate fuel mix sums to 1
        total_share = sum(fuel_mix.values())
        if abs(total_share - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(f"Fuel mix shares must sum to 1.0, got {total_share}")

        # Calculate weighted average emission factor
        weighted_factor = Decimal("0")
        weighted_uncertainty = Decimal("0")

        for fuel_type, share in fuel_mix.items():
            factor = self.emission_db.get_fuel_factor(fuel_type)
            weighted_factor += factor.co2e_factor * share
            weighted_uncertainty += (factor.uncertainty_pct * share) ** 2

        weighted_uncertainty = weighted_uncertainty.sqrt()

        # Calculate intensity per GJ of steam output
        # Intensity = emission factor / efficiency
        intensity_per_gj = weighted_factor / efficiency

        # Calculate intensity per tonne of steam
        # Assumes typical steam enthalpy of 2750 kJ/kg (2.75 GJ/tonne)
        steam_energy_per_tonne = Decimal("2.33")  # GJ/tonne (2330 kJ/kg net)
        intensity_per_tonne = intensity_per_gj * steam_energy_per_tonne

        return SteamCarbonIntensity(
            intensity_kg_co2e_per_gj=intensity_per_gj.quantize(
                Decimal("0.01"), ROUND_HALF_UP
            ),
            intensity_kg_co2e_per_tonne=intensity_per_tonne.quantize(
                Decimal("0.01"), ROUND_HALF_UP
            ),
            methodology="Weighted average fuel mix with boiler efficiency",
            fuel_mix=fuel_mix,
            boiler_efficiency=efficiency,
            uncertainty_pct=weighted_uncertainty + Decimal("3.0"),  # Add efficiency unc
        )

    def calculate_complete_impact(
        self,
        period_start: datetime,
        period_end: datetime,
        steam_production_kg: Decimal,
        fuel_consumptions: Dict[FuelType, Decimal],
        electricity_kwh: Decimal = Decimal("0"),
        grid_region: Optional[GridRegion] = None,
        boiler_efficiency: Optional[Decimal] = None,
    ) -> ClimateImpactResult:
        """
        Calculate complete climate impact for a period.

        Args:
            period_start: Start of period
            period_end: End of period
            steam_production_kg: Total steam produced in kg
            fuel_consumptions: Fuel consumption by type (GJ)
            electricity_kwh: Electricity consumption (kWh)
            grid_region: Grid region for electricity
            boiler_efficiency: Boiler efficiency

        Returns:
            Complete ClimateImpactResult
        """
        import uuid

        emissions_breakdown: List[EmissionsBreakdown] = []

        # Calculate fuel emissions (Scope 1)
        scope_1_total = Decimal("0")
        total_fuel_gj = Decimal("0")

        for fuel_type, consumption_gj in fuel_consumptions.items():
            if consumption_gj > 0:
                emission = self.calculate_fuel_emissions(fuel_type, consumption_gj)
                emissions_breakdown.append(emission)
                scope_1_total += emission.co2e_kg
                total_fuel_gj += consumption_gj

        # Calculate electricity emissions (Scope 2)
        scope_2_total = Decimal("0")
        if electricity_kwh > 0:
            elec_emission = self.calculate_electricity_emissions(
                electricity_kwh, grid_region
            )
            emissions_breakdown.append(elec_emission)
            scope_2_total = elec_emission.co2e_kg

        # Scope 3 (placeholder - could include upstream fuel emissions)
        scope_3_total = Decimal("0")

        # Total emissions
        total_co2e = scope_1_total + scope_2_total + scope_3_total

        # Calculate steam energy
        steam_energy_gj = (
            steam_production_kg * self.STEAM_ENTHALPY_TYPICAL / Decimal("1000000")
        )

        # Calculate fuel mix and intensity
        fuel_mix: Dict[FuelType, Decimal] = {}
        if total_fuel_gj > 0:
            for fuel_type, consumption in fuel_consumptions.items():
                if consumption > 0:
                    fuel_mix[fuel_type] = consumption / total_fuel_gj
        else:
            fuel_mix[FuelType.NATURAL_GAS] = Decimal("1.0")  # Default

        intensity = self.calculate_steam_carbon_intensity(
            fuel_mix, boiler_efficiency
        )

        # Calculate combined uncertainty
        if emissions_breakdown:
            unc_squared = sum(
                (e.uncertainty_pct / Decimal("100") * e.co2e_kg) ** 2
                for e in emissions_breakdown
            )
            combined_unc = unc_squared.sqrt()
            total_unc_pct = (
                combined_unc / total_co2e * Decimal("100")
                if total_co2e > 0 else Decimal("10")
            )
        else:
            total_unc_pct = Decimal("10")

        unc_margin = total_co2e * (total_unc_pct / Decimal("100")) * Decimal("1.96")

        calc_id = f"CLIMATE-{uuid.uuid4().hex[:8].upper()}"

        result = ClimateImpactResult(
            calculation_id=calc_id,
            timestamp=datetime.now(timezone.utc),
            period_start=period_start,
            period_end=period_end,
            scope_1_total_kg=scope_1_total.quantize(Decimal("0.01"), ROUND_HALF_UP),
            scope_2_total_kg=scope_2_total.quantize(Decimal("0.01"), ROUND_HALF_UP),
            scope_3_total_kg=scope_3_total.quantize(Decimal("0.01"), ROUND_HALF_UP),
            total_co2e_kg=total_co2e.quantize(Decimal("0.01"), ROUND_HALF_UP),
            emissions_breakdown=emissions_breakdown,
            steam_carbon_intensity=intensity,
            total_steam_gj=steam_energy_gj.quantize(Decimal("0.01"), ROUND_HALF_UP),
            total_fuel_gj=total_fuel_gj.quantize(Decimal("0.01"), ROUND_HALF_UP),
            total_electricity_kwh=electricity_kwh,
            total_uncertainty_pct=total_unc_pct.quantize(Decimal("0.1"), ROUND_HALF_UP),
            total_lower_kg=(total_co2e - unc_margin).quantize(
                Decimal("0.01"), ROUND_HALF_UP
            ),
            total_upper_kg=(total_co2e + unc_margin).quantize(
                Decimal("0.01"), ROUND_HALF_UP
            ),
            emission_factor_version=self.emission_db.version,
            methodology="GHG Protocol Scope 1/2 calculation",
            audit_trail=[{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "calculate_complete_impact",
                "calculation_id": calc_id,
            }],
        )

        self._log_calculation("complete_impact", {
            "calculation_id": calc_id,
            "total_co2e_kg": str(result.total_co2e_kg),
        })

        return result

    def calculate_savings_impact(
        self,
        baseline_steam_kg: Decimal,
        actual_steam_kg: Decimal,
        fuel_type: FuelType = FuelType.NATURAL_GAS,
        boiler_efficiency: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Calculate climate impact of steam savings.

        Args:
            baseline_steam_kg: Baseline steam consumption in kg
            actual_steam_kg: Actual steam consumption in kg
            fuel_type: Primary fuel type
            boiler_efficiency: Boiler efficiency

        Returns:
            Dictionary with savings impact metrics
        """
        efficiency = boiler_efficiency or self.default_boiler_efficiency

        # Calculate steam savings
        steam_savings_kg = baseline_steam_kg - actual_steam_kg
        steam_savings_gj = (
            steam_savings_kg * self.STEAM_ENTHALPY_TYPICAL / Decimal("1000000")
        )

        # Estimate fuel savings
        fuel_savings_gj = steam_savings_gj / efficiency

        # Calculate emissions avoided
        factor = self.emission_db.get_fuel_factor(fuel_type)
        emissions_avoided_kg = factor.co2e_factor * fuel_savings_gj

        return {
            "steam_savings_kg": str(steam_savings_kg.quantize(Decimal("0.1"))),
            "steam_savings_gj": str(steam_savings_gj.quantize(Decimal("0.001"))),
            "fuel_savings_gj": str(fuel_savings_gj.quantize(Decimal("0.001"))),
            "emissions_avoided_kg_co2e": str(
                emissions_avoided_kg.quantize(Decimal("0.01"))
            ),
            "emissions_avoided_tonnes_co2e": str(
                (emissions_avoided_kg / Decimal("1000")).quantize(Decimal("0.001"))
            ),
            "fuel_type": fuel_type.value,
            "boiler_efficiency": str(efficiency),
            "emission_factor": str(factor.co2e_factor),
            "calculation_hash": self._compute_hash(
                str(baseline_steam_kg),
                str(actual_steam_kg),
                fuel_type.value,
            ),
        }

    def _compute_hash(self, *args) -> str:
        """Compute deterministic hash for audit."""
        data = "|".join(args)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _log_calculation(self, calc_type: str, details: Dict[str, Any]):
        """Log calculation to audit trail."""
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "calculation_type": calc_type,
            **details,
        })

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log."""
        return self._audit_log.copy()
