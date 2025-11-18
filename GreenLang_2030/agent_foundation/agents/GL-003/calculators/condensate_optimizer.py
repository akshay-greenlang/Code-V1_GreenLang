"""
Condensate Recovery Optimizer - Zero Hallucination

Calculates condensate return optimization, flash steam recovery,
and heat recovery potential from condensate systems.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ASHRAE, Spirax Sarco Steam Engineering Principles
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List
from dataclasses import dataclass
from .provenance import ProvenanceTracker
from .steam_properties import SteamPropertiesCalculator


@dataclass
class CondensateData:
    """Condensate system data."""
    condensate_flow_rate_kg_hr: float
    condensate_temperature_c: float
    condensate_pressure_bar: float
    flash_vessel_pressure_bar: float
    feedwater_temperature_c: float
    steam_generation_pressure_bar: float
    return_rate_percent: float  # Current condensate return rate


@dataclass
class CondensateResults:
    """Results from condensate optimization."""
    flash_steam_available_kg_hr: float
    flash_steam_energy_gj_hr: float
    condensate_heat_recovery_gj_hr: float
    total_recovery_potential_gj_hr: float
    current_return_rate_percent: float
    optimal_return_rate_percent: float
    annual_savings_potential: float
    flash_vessel_sizing: Dict
    recommendations: List[str]
    provenance: Dict


class CondensateOptimizer:
    """
    Optimize condensate recovery and flash steam utilization.

    Zero Hallucination Guarantee:
    - Pure thermodynamic calculations
    - IAPWS steam properties
    - No LLM inference
    """

    def __init__(self, version: str = "1.0.0"):
        """Initialize optimizer."""
        self.version = version
        self.steam_calc = SteamPropertiesCalculator(version)

    def optimize_condensate_system(
        self,
        data: CondensateData,
        energy_cost_per_gj: float = 20.0,
        water_cost_per_tonne: float = 2.0,
        treatment_cost_per_tonne: float = 5.0
    ) -> CondensateResults:
        """
        Optimize condensate recovery system.

        Returns energy recovery potential and optimization recommendations.
        """
        tracker = ProvenanceTracker(
            calculation_id=f"condensate_opt_{id(data)}",
            calculation_type="condensate_optimization",
            version=self.version
        )

        tracker.record_inputs(data.__dict__)

        # Step 1: Calculate flash steam generation
        flash_steam_data = self._calculate_flash_steam(data, tracker)

        # Step 2: Calculate condensate heat recovery
        condensate_heat = self._calculate_condensate_heat_recovery(data, tracker)

        # Step 3: Calculate total recovery potential
        total_recovery = flash_steam_data['energy_gj_hr'] + condensate_heat

        # Step 4: Determine optimal return rate
        optimal_return = self._calculate_optimal_return_rate(data, tracker)

        # Step 5: Calculate economic benefits
        annual_savings = self._calculate_annual_savings(
            total_recovery,
            data.return_rate_percent,
            optimal_return,
            energy_cost_per_gj,
            water_cost_per_tonne,
            treatment_cost_per_tonne,
            data.condensate_flow_rate_kg_hr,
            tracker
        )

        # Step 6: Size flash vessel
        flash_vessel = self._size_flash_vessel(
            flash_steam_data['flash_rate_kg_hr'],
            data.flash_vessel_pressure_bar,
            tracker
        )

        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(
            data,
            flash_steam_data,
            float(optimal_return),
            float(annual_savings)
        )

        return CondensateResults(
            flash_steam_available_kg_hr=flash_steam_data['flash_rate_kg_hr'],
            flash_steam_energy_gj_hr=float(flash_steam_data['energy_gj_hr']),
            condensate_heat_recovery_gj_hr=float(condensate_heat),
            total_recovery_potential_gj_hr=float(total_recovery),
            current_return_rate_percent=data.return_rate_percent,
            optimal_return_rate_percent=float(optimal_return),
            annual_savings_potential=float(annual_savings),
            flash_vessel_sizing=flash_vessel,
            recommendations=recommendations,
            provenance=tracker.get_provenance_record(total_recovery).to_dict()
        )

    def _calculate_flash_steam(
        self,
        data: CondensateData,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate flash steam generation when high-pressure condensate
        is discharged to lower pressure.

        Flash fraction = (h_initial - h_saturated_liquid) / h_fg
        """
        m_condensate = Decimal(str(data.condensate_flow_rate_kg_hr))
        P_initial = Decimal(str(data.condensate_pressure_bar))
        T_initial = Decimal(str(data.condensate_temperature_c))
        P_flash = Decimal(str(data.flash_vessel_pressure_bar))

        # Initial condensate enthalpy
        h_initial = Decimal(str(
            self.steam_calc.enthalpy_from_pressure_temperature(
                float(P_initial),
                float(T_initial)
            )
        ))

        # Saturated liquid enthalpy at flash pressure
        T_sat_flash = Decimal(str(
            self.steam_calc.saturation_temperature_from_pressure(float(P_flash))
        ))

        h_liquid_flash = Decimal(str(
            self.steam_calc.enthalpy_from_pressure_temperature(
                float(P_flash),
                float(T_sat_flash)
            )
        ))

        # Latent heat at flash pressure (hfg)
        h_vapor_flash = Decimal(str(
            self.steam_calc.enthalpy_from_pressure_temperature(
                float(P_flash),
                float(T_sat_flash) + 1.0  # Slightly superheated
            )
        ))

        h_fg = h_vapor_flash - h_liquid_flash

        # Flash fraction
        if h_fg > Decimal('0'):
            flash_fraction = (h_initial - h_liquid_flash) / h_fg
            flash_fraction = max(Decimal('0'), min(flash_fraction, Decimal('0.30')))  # Typical max ~30%
        else:
            flash_fraction = Decimal('0')

        # Flash steam flow rate
        m_flash = m_condensate * flash_fraction

        # Energy content of flash steam (GJ/hr)
        flash_energy_kj_hr = m_flash * h_vapor_flash
        flash_energy_gj_hr = flash_energy_kj_hr / Decimal('1000000')

        tracker.record_step(
            operation="flash_steam_calculation",
            description="Calculate flash steam generation",
            inputs={
                'initial_enthalpy_kj_kg': h_initial,
                'flash_liquid_enthalpy_kj_kg': h_liquid_flash,
                'latent_heat_kj_kg': h_fg
            },
            output_value=flash_fraction,
            output_name="flash_fraction",
            formula="x_flash = (h_in - h_f) / h_fg",
            units="fraction"
        )

        return {
            'flash_fraction': float(flash_fraction),
            'flash_rate_kg_hr': float(m_flash),
            'energy_gj_hr': flash_energy_gj_hr
        }

    def _calculate_condensate_heat_recovery(
        self,
        data: CondensateData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate sensible heat that can be recovered from condensate
        by using it to preheat feedwater.
        """
        m_condensate = Decimal(str(data.condensate_flow_rate_kg_hr))
        T_condensate = Decimal(str(data.condensate_temperature_c))
        T_feedwater = Decimal(str(data.feedwater_temperature_c))

        # Specific heat of water (approximately constant)
        Cp = Decimal('4.18')  # kJ/(kg·K)

        # Heat available
        Q_kj_hr = m_condensate * Cp * (T_condensate - T_feedwater)

        # Convert to GJ/hr
        Q_gj_hr = Q_kj_hr / Decimal('1000000')

        tracker.record_step(
            operation="condensate_heat_recovery",
            description="Calculate sensible heat recovery from condensate",
            inputs={
                'condensate_flow_kg_hr': m_condensate,
                'condensate_temp_c': T_condensate,
                'feedwater_temp_c': T_feedwater
            },
            output_value=Q_gj_hr,
            output_name="condensate_heat_gj_hr",
            formula="Q = m * Cp * (T_cond - T_fw)",
            units="GJ/hr"
        )

        return Q_gj_hr

    def _calculate_optimal_return_rate(
        self,
        data: CondensateData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate optimal condensate return rate based on economics.

        Target: 90-95% return rate (industry best practice)
        """
        current_rate = Decimal(str(data.return_rate_percent))

        # Optimal target depends on system characteristics
        # For typical industrial systems: 90% is achievable
        if current_rate < Decimal('70'):
            optimal = Decimal('85')  # Conservative target
        elif current_rate < Decimal('85'):
            optimal = Decimal('90')  # Standard target
        else:
            optimal = Decimal('95')  # Best practice target

        tracker.record_step(
            operation="optimal_return_rate",
            description="Determine optimal condensate return rate",
            inputs={'current_return_rate_percent': current_rate},
            output_value=optimal,
            output_name="optimal_return_rate_percent",
            formula="Based on industry best practices",
            units="%"
        )

        return optimal

    def _calculate_annual_savings(
        self,
        total_recovery_gj_hr: Decimal,
        current_return_percent: float,
        optimal_return_percent: Decimal,
        energy_cost: float,
        water_cost: float,
        treatment_cost: float,
        condensate_flow: float,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate annual savings from improving condensate return."""
        current = Decimal(str(current_return_percent))
        optimal = Decimal(str(optimal_return_percent))
        improvement = optimal - current

        if improvement <= Decimal('0'):
            return Decimal('0')

        # Energy savings (8760 hours/year)
        energy_savings = total_recovery_gj_hr * Decimal('8760') * (improvement / Decimal('100'))
        energy_cost_savings = energy_savings * Decimal(str(energy_cost))

        # Water and treatment savings
        m_flow = Decimal(str(condensate_flow))
        annual_flow_tonnes = (m_flow * Decimal('8760')) / Decimal('1000')
        water_saved_tonnes = annual_flow_tonnes * (improvement / Decimal('100'))

        water_savings = water_saved_tonnes * Decimal(str(water_cost))
        treatment_savings = water_saved_tonnes * Decimal(str(treatment_cost))

        total_savings = energy_cost_savings + water_savings + treatment_savings

        tracker.record_step(
            operation="annual_savings",
            description="Calculate total annual savings potential",
            inputs={
                'energy_cost_savings': energy_cost_savings,
                'water_cost_savings': water_savings,
                'treatment_cost_savings': treatment_savings
            },
            output_value=total_savings,
            output_name="annual_savings",
            formula="Total = Energy + Water + Treatment savings",
            units="currency"
        )

        return total_savings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def _size_flash_vessel(
        self,
        flash_steam_kg_hr: float,
        pressure_bar: float,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Size flash vessel based on vapor load and separation efficiency."""
        m_flash = Decimal(str(flash_steam_kg_hr))
        P = Decimal(str(pressure_bar))

        # Vapor density at flash pressure (simplified)
        # ρ_vapor ≈ P / (R * T) for ideal gas approximation
        T_sat_k = Decimal(str(
            self.steam_calc.saturation_temperature_from_pressure(float(P))
        )) + Decimal('273.15')

        R = Decimal('0.4615')  # kJ/(kg·K)
        rho_vapor = (P * Decimal('100')) / (R * T_sat_k)  # kg/m³

        # Volumetric flow rate
        V_vapor_m3_hr = m_flash / rho_vapor

        # Vessel sizing (simplified)
        # Vessel diameter based on vapor velocity limit (~20-30 m/s for separation)
        v_max = Decimal('25')  # m/s
        A_cross = V_vapor_m3_hr / (v_max * Decimal('3600'))  # m²

        # Diameter: A = π * D² / 4
        D = Decimal('2') * (A_cross / Decimal('3.14159')).sqrt()

        # Length typically 2-3 times diameter for good separation
        L = D * Decimal('2.5')

        return {
            'diameter_m': float(D.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'length_m': float(L.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            'volume_m3': float((Decimal('3.14159') * (D / Decimal('2')) ** 2 * L).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            'design_pressure_bar': float(P * Decimal('1.5'))  # Safety factor
        }

    def _generate_recommendations(
        self,
        data: CondensateData,
        flash_data: Dict,
        optimal_return: float,
        annual_savings: float
    ) -> List[str]:
        """Generate condensate system recommendations."""
        recommendations = []

        # Return rate improvement
        improvement_needed = optimal_return - data.return_rate_percent
        if improvement_needed > 5:
            recommendations.append(
                f"Increase condensate return rate from {data.return_rate_percent:.1f}% to {optimal_return:.1f}%. "
                f"Potential annual savings: ${annual_savings:,.0f}"
            )

        # Flash steam recovery
        if flash_data['flash_rate_kg_hr'] > 10:
            recommendations.append(
                f"Install flash vessel to recover {flash_data['flash_rate_kg_hr']:.1f} kg/hr of flash steam. "
                f"Energy recovery potential: {flash_data['energy_gj_hr']:.2f} GJ/hr"
            )

        # System improvements
        if data.return_rate_percent < 70:
            recommendations.append(
                "Install condensate recovery system with pumps and return lines"
            )

        if data.return_rate_percent < 85:
            recommendations.append(
                "Inspect and repair steam traps to reduce condensate loss"
            )

        recommendations.append(
            "Use recovered condensate to preheat boiler feedwater (deaerator feedwater)"
        )

        return recommendations
