"""
Stoichiometric Calculator

Calculates theoretical air requirement and combustion products for complete combustion.
Zero-hallucination design using fundamental combustion chemistry.

Reference: ASHRAE Fundamentals, Combustion and Fuels chapter
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)


class StoichiometricCalculator:
    """Calculate stoichiometric combustion parameters"""

    # Molecular weights (kg/kmol)
    MW = {
        'C': 12.011,
        'H': 1.008,
        'O': 15.999,
        'N': 14.007,
        'S': 32.065,
        'O2': 31.998,
        'N2': 28.014,
        'CO2': 44.01,
        'H2O': 18.015,
        'SO2': 64.064
    }

    def calculate(self, fuel_type: str, fuel_composition: Dict[str, float], 
                  air_fuel_ratio: float) -> Dict[str, float]:
        """
        Calculate stoichiometric parameters
        
        Args:
            fuel_type: Type of fuel
            fuel_composition: Weight percentages (C, H, O, N, S)
            air_fuel_ratio: Actual air-fuel ratio (kg air / kg fuel)
            
        Returns:
            Dict with stoichiometric parameters
        """
        
        # Calculate theoretical O2 requirement (kg O2 / kg fuel)
        c_frac = fuel_composition.get('C', 0) / 100
        h_frac = fuel_composition.get('H', 0) / 100
        o_frac = fuel_composition.get('O', 0) / 100
        s_frac = fuel_composition.get('S', 0) / 100
        
        # Stoichiometric O2: C + O2 -> CO2, 2H2 + O2 -> 2H2O, S + O2 -> SO2
        o2_required = (c_frac * (self.MW['O2'] / self.MW['C']) +
                      h_frac * (self.MW['O2'] / (2 * self.MW['H'])) +
                      s_frac * (self.MW['O2'] / self.MW['S']) -
                      o_frac)
        
        # Theoretical air requirement (assuming 23.15% O2 by mass in air)
        theoretical_air = o2_required / 0.2315
        
        # Excess air
        excess_air_percent = ((air_fuel_ratio - theoretical_air) / theoretical_air * 100
                             if theoretical_air > 0 else 0)
        
        # Combustion products
        co2_produced = c_frac * (self.MW['CO2'] / self.MW['C'])
        h2o_produced = h_frac * (self.MW['H2O'] / (2 * self.MW['H']))
        so2_produced = s_frac * (self.MW['SO2'] / self.MW['S'])
        
        # N2 from air + fuel N
        n2_from_air = air_fuel_ratio * 0.7685  # 76.85% N2 by mass in air
        n2_from_fuel = fuel_composition.get('N', 0) / 100
        n2_total = n2_from_air + n2_from_fuel
        
        # Excess O2
        excess_o2 = o2_required * (excess_air_percent / 100) if excess_air_percent > 0 else 0
        
        # Total flue gas
        total_flue_gas = co2_produced + h2o_produced + so2_produced + n2_total + excess_o2
        
        return {
            'theoretical_air_kg_per_kg_fuel': theoretical_air,
            'excess_air_percent': excess_air_percent,
            'o2_required_kg_per_kg_fuel': o2_required,
            'co2_produced_kg_per_kg_fuel': co2_produced,
            'h2o_produced_kg_per_kg_fuel': h2o_produced,
            'so2_produced_kg_per_kg_fuel': so2_produced,
            'n2_in_flue_gas_kg_per_kg_fuel': n2_total,
            'excess_o2_kg_per_kg_fuel': excess_o2,
            'total_flue_gas_kg_per_kg_fuel': total_flue_gas,
            'theoretical_co2_percent': (co2_produced / total_flue_gas * 100) if total_flue_gas > 0 else 0,
            'theoretical_o2_percent': (excess_o2 / total_flue_gas * 100) if total_flue_gas > 0 else 0
        }
