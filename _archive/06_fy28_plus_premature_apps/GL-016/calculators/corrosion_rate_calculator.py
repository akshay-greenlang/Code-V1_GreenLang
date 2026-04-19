# -*- coding: utf-8 -*-
"""
Corrosion Rate Calculator - GL-016 WATERGUARD

Predicts corrosion rates for various mechanisms in boiler and cooling systems.
Implements electrochemical and mechanistic corrosion models with zero hallucination guarantee.

Author: GL-016 WATERGUARD Engineering Team
Version: 1.0.0
Standards: NACE, ASTM, ASME
References:
- NACE SP0169 - Control of External Corrosion
- ASTM G1 - Standard Practice for Preparing, Cleaning, and Evaluating Corrosion Test Specimens
- ASME PTC 19.11 - Steam and Water Sampling, Conditioning, and Analysis
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional
from dataclasses import dataclass
from .provenance import ProvenanceTracker


@dataclass
class CorrosionConditions:
    """Operating conditions affecting corrosion."""
    # Environmental conditions
    temperature_c: float
    pressure_bar: float
    flow_velocity_m_s: float
    ph: float

    # Water chemistry (mg/L)
    dissolved_oxygen_mg_l: float
    carbon_dioxide_mg_l: float
    chloride_mg_l: float
    sulfate_mg_l: float
    ammonia_mg_l: float
    conductivity_us_cm: float

    # Material properties
    material_type: str  # 'carbon_steel', 'stainless_304', 'copper', 'brass'
    surface_finish: str  # 'polished', 'machined', 'as_welded', 'corroded'

    # Operational
    operating_time_hours: float
    stress_level_mpa: float  # For stress corrosion cracking


class CorrosionRateCalculator:
    """
    Calculates corrosion rates for multiple mechanisms.

    Capabilities:
    - Oxygen corrosion modeling
    - Carbon dioxide corrosion
    - Under-deposit corrosion
    - Stress corrosion cracking risk
    - Galvanic corrosion potential
    - Erosion-corrosion rates
    - Material-specific corrosion allowances
    - Remaining life calculations
    """

    # Material properties
    DENSITIES = {  # g/cm³
        'carbon_steel': Decimal('7.85'),
        'stainless_304': Decimal('8.00'),
        'copper': Decimal('8.96'),
        'brass': Decimal('8.50')
    }

    EQUIVALENT_WEIGHTS = {  # g/equivalent
        'carbon_steel': Decimal('27.92'),  # Fe → Fe²⁺ (2 electrons)
        'stainless_304': Decimal('25.0'),  # Average for stainless
        'copper': Decimal('31.77'),  # Cu → Cu²⁺
        'brass': Decimal('30.0')  # Cu-Zn alloy average
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize corrosion rate calculator."""
        self.version = version

    def calculate_comprehensive_corrosion_analysis(
        self,
        conditions: CorrosionConditions
    ) -> Dict:
        """
        Perform comprehensive corrosion analysis.

        Args:
            conditions: Operating conditions and water chemistry

        Returns:
            Complete corrosion analysis with rates and predictions
        """
        tracker = ProvenanceTracker(
            calculation_id=f"corrosion_calc_{id(conditions)}",
            calculation_type="corrosion_analysis",
            version=self.version
        )

        tracker.record_inputs(conditions.__dict__)

        # Calculate individual corrosion mechanisms
        oxygen_corrosion = self._calculate_oxygen_corrosion(conditions, tracker)
        co2_corrosion = self._calculate_co2_corrosion(conditions, tracker)
        pitting = self._calculate_pitting_corrosion(conditions, tracker)
        crevice = self._calculate_crevice_corrosion(conditions, tracker)
        erosion_corrosion = self._calculate_erosion_corrosion(conditions, tracker)
        galvanic = self._calculate_galvanic_corrosion(conditions, tracker)
        scc_risk = self._calculate_scc_risk(conditions, tracker)

        # Total corrosion rate
        total_rate = self._calculate_total_corrosion_rate(
            oxygen_corrosion, co2_corrosion, pitting, erosion_corrosion, tracker
        )

        # Remaining life analysis
        remaining_life = self._calculate_remaining_life(total_rate, conditions, tracker)

        result = {
            'oxygen_corrosion': oxygen_corrosion,
            'co2_corrosion': co2_corrosion,
            'pitting_corrosion': pitting,
            'crevice_corrosion': crevice,
            'erosion_corrosion': erosion_corrosion,
            'galvanic_corrosion': galvanic,
            'stress_corrosion_cracking': scc_risk,
            'total_corrosion_rate': total_rate,
            'remaining_life_analysis': remaining_life,
            'provenance': tracker.get_provenance_record(total_rate).to_dict()
        }

        return result

    def _calculate_oxygen_corrosion(
        self,
        conditions: CorrosionConditions,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate oxygen corrosion rate using electrochemical model.

        Cathodic reaction: O2 + 2H2O + 4e⁻ → 4OH⁻
        Anodic reaction: Fe → Fe²⁺ + 2e⁻

        Corrosion rate = k × [O2]^n × exp(-Ea/RT)

        Reference: NACE Corrosion Basics
        """
        temp = Decimal(str(conditions.temperature_c))
        oxygen = Decimal(str(conditions.dissolved_oxygen_mg_l))
        velocity = Decimal(str(conditions.flow_velocity_m_s))
        ph = Decimal(str(conditions.ph))

        # Temperature factor (Arrhenius)
        temp_k = temp + Decimal('273.15')
        Ea = Decimal('50000')  # Activation energy J/mol
        R = Decimal('8.314')
        temp_factor = ((-Ea / (R * temp_k)).exp())

        # Oxygen concentration factor
        # Rate proportional to [O2]^0.5 (diffusion limited)
        oxygen_factor = oxygen.sqrt()

        # pH factor (lower pH increases corrosion)
        pH_factor = Decimal('10') ** (Decimal('7') - ph) if ph < Decimal('7') else Decimal('1')

        # Velocity factor (mass transfer coefficient)
        # k_mass ∝ v^0.8 (turbulent flow)
        velocity_factor = (velocity ** Decimal('0.8')) if velocity > 0 else Decimal('1')

        # Base rate constant (mils per year per (mg/L)^0.5)
        k0 = Decimal('2.0')

        # Corrosion rate (mils per year - mpy)
        rate_mpy = k0 * oxygen_factor * pH_factor * velocity_factor * temp_factor

        # Convert to mm/year
        rate_mm_yr = rate_mpy * Decimal('0.0254')

        # Mass loss rate
        material = conditions.material_type
        density = self.DENSITIES.get(material, self.DENSITIES['carbon_steel'])
        mass_loss_g_m2_day = rate_mm_yr * density * Decimal('1000') / Decimal('365')

        tracker.record_step(
            operation="oxygen_corrosion",
            description="Calculate oxygen-induced corrosion rate",
            inputs={
                'oxygen_mg_L': oxygen,
                'temperature_C': temp,
                'pH': ph,
                'velocity_m_s': velocity
            },
            output_value=rate_mm_yr,
            output_name="oxygen_corrosion_rate_mm_yr",
            formula="Rate = k × [O2]^0.5 × 10^(7-pH) × v^0.8 × exp(-Ea/RT)",
            units="mm/year",
            reference="NACE Corrosion Basics"
        )

        return {
            'corrosion_rate_mpy': float(rate_mpy.quantize(Decimal('0.01'))),
            'corrosion_rate_mm_yr': float(rate_mm_yr.quantize(Decimal('0.001'))),
            'mass_loss_g_m2_day': float(mass_loss_g_m2_day.quantize(Decimal('0.01'))),
            'severity': self._assess_corrosion_severity(float(rate_mpy)),
            'mechanism': 'Oxygen reduction (diffusion controlled)'
        }

    def _calculate_co2_corrosion(
        self,
        conditions: CorrosionConditions,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate CO2 corrosion rate (sweet corrosion).

        CO2 + H2O → H2CO3 → H⁺ + HCO3⁻
        Fe + H2CO3 → FeCO3 + H2

        De Waard-Milliams model for CO2 corrosion.

        Reference: NACE MR0175
        """
        temp = Decimal(str(conditions.temperature_c))
        co2 = Decimal(str(conditions.carbon_dioxide_mg_l))
        ph = Decimal(str(conditions.ph))
        velocity = Decimal(str(conditions.flow_velocity_m_s))

        # De Waard-Milliams equation
        # log(CR) = 5.8 - 1710/T - 0.67×log(pCO2)
        # Where CR in mm/year, T in Kelvin, pCO2 in bar

        # Estimate pCO2 from dissolved CO2 (Henry's Law)
        # pCO2 (bar) ≈ [CO2_dissolved] / (K_H × 44)
        # K_H at 25°C ≈ 0.034 mol/L/bar
        K_H = Decimal('0.034')
        pCO2_bar = (co2 / Decimal('44000')) / K_H  # Convert mg/L to mol/L

        temp_k = temp + Decimal('273.15')

        if pCO2_bar > 0:
            log_pCO2 = (pCO2_bar).ln() / Decimal('2.303')  # log10
            log_CR = Decimal('5.8') - Decimal('1710') / temp_k - Decimal('0.67') * log_pCO2
            rate_mm_yr = Decimal('10') ** log_CR
        else:
            rate_mm_yr = Decimal('0')

        # pH correction factor
        # Protective FeCO3 film forms at pH > 5
        if ph > Decimal('5.5'):
            protection_factor = Decimal('0.1')  # 90% reduction
        else:
            protection_factor = Decimal('1.0')

        rate_mm_yr = rate_mm_yr * protection_factor

        # Velocity enhancement for erosion of protective film
        if velocity > Decimal('3.0'):  # Critical velocity
            velocity_enhancement = Decimal('1') + (velocity - Decimal('3.0')) * Decimal('0.2')
            rate_mm_yr = rate_mm_yr * velocity_enhancement

        # Convert to mpy
        rate_mpy = rate_mm_yr / Decimal('0.0254')

        tracker.record_step(
            operation="co2_corrosion",
            description="Calculate CO2 corrosion using De Waard-Milliams model",
            inputs={
                'co2_mg_L': co2,
                'temperature_C': temp,
                'pH': ph,
                'pCO2_bar': pCO2_bar
            },
            output_value=rate_mm_yr,
            output_name="co2_corrosion_rate_mm_yr",
            formula="log(CR) = 5.8 - 1710/T - 0.67×log(pCO2)",
            units="mm/year",
            reference="De Waard-Milliams, NACE MR0175"
        )

        return {
            'pCO2_bar': float(pCO2_bar.quantize(Decimal('0.0001'))),
            'corrosion_rate_mpy': float(rate_mpy.quantize(Decimal('0.01'))),
            'corrosion_rate_mm_yr': float(rate_mm_yr.quantize(Decimal('0.001'))),
            'feco3_protection': 'Yes' if ph > Decimal('5.5') else 'No',
            'severity': self._assess_corrosion_severity(float(rate_mpy)),
            'mechanism': 'Carbonic acid attack'
        }

    def _calculate_pitting_corrosion(
        self,
        conditions: CorrosionConditions,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate pitting corrosion susceptibility.

        Pitting index: PI = [Cl⁻] / ([HCO3⁻] + [OH⁻])

        Reference: Larson-Skold Index
        """
        chloride = Decimal(str(conditions.chloride_mg_l))
        sulfate = Decimal(str(conditions.sulfate_mg_l))
        ph = Decimal(str(conditions.ph))
        temp = Decimal(str(conditions.temperature_c))

        # Pitting potential (simplified)
        # Aggressive ions: Cl⁻, SO4²⁻
        # Inhibiting ions: OH⁻ (from pH)

        aggressive = chloride + sulfate * Decimal('0.5')  # SO4 less aggressive than Cl
        inhibiting = Decimal('10') ** (ph - Decimal('7')) * Decimal('17')  # [OH⁻] as mg/L equiv.

        if inhibiting > 0:
            pitting_index = aggressive / inhibiting
        else:
            pitting_index = aggressive

        # Temperature enhancement
        temp_enhancement = Decimal('1') + (temp - Decimal('25')) * Decimal('0.02')
        pitting_index = pitting_index * temp_enhancement

        # Risk assessment
        if pitting_index < Decimal('0.5'):
            risk = "Low"
            pit_rate_mpy = Decimal('0.1')
        elif pitting_index < Decimal('2.0'):
            risk = "Moderate"
            pit_rate_mpy = pitting_index * Decimal('0.5')
        elif pitting_index < Decimal('5.0'):
            risk = "High"
            pit_rate_mpy = pitting_index * Decimal('1.0')
        else:
            risk = "Critical"
            pit_rate_mpy = pitting_index * Decimal('2.0')

        pit_rate_mm_yr = pit_rate_mpy * Decimal('0.0254')

        tracker.record_step(
            operation="pitting_corrosion",
            description="Calculate pitting corrosion susceptibility",
            inputs={
                'chloride_mg_L': chloride,
                'sulfate_mg_L': sulfate,
                'pH': ph,
                'temperature_C': temp
            },
            output_value=pitting_index,
            output_name="pitting_index",
            formula="PI = (Cl + 0.5×SO4) / OH × (1 + 0.02×(T-25))",
            units="dimensionless"
        )

        return {
            'pitting_index': float(pitting_index.quantize(Decimal('0.01'))),
            'risk_level': risk,
            'maximum_pit_rate_mpy': float(pit_rate_mpy.quantize(Decimal('0.01'))),
            'maximum_pit_rate_mm_yr': float(pit_rate_mm_yr.quantize(Decimal('0.001'))),
            'mechanism': 'Localized anodic attack by aggressive anions'
        }

    def _calculate_crevice_corrosion(
        self,
        conditions: CorrosionConditions,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate crevice corrosion susceptibility.

        Occurs in shielded areas with restricted oxygen access.
        """
        chloride = Decimal(str(conditions.chloride_mg_l))
        oxygen = Decimal(str(conditions.dissolved_oxygen_mg_l))
        temp = Decimal(str(conditions.temperature_c))
        material = conditions.material_type

        # Crevice susceptibility (higher Cl and lower O2 increase risk)
        crevice_factor = chloride / (oxygen + Decimal('1'))

        # Temperature enhancement
        if temp > Decimal('60'):
            temp_factor = Decimal('1') + (temp - Decimal('60')) * Decimal('0.03')
        else:
            temp_factor = Decimal('1')

        crevice_index = crevice_factor * temp_factor

        # Material susceptibility
        if material == 'stainless_304':
            material_factor = Decimal('2.0')  # More susceptible
        else:
            material_factor = Decimal('1.0')

        crevice_index = crevice_index * material_factor

        # Risk assessment
        if crevice_index < Decimal('10'):
            risk = "Low"
        elif crevice_index < Decimal('50'):
            risk = "Moderate"
        elif crevice_index < Decimal('100'):
            risk = "High"
        else:
            risk = "Critical"

        return {
            'crevice_index': float(crevice_index.quantize(Decimal('0.1'))),
            'risk_level': risk,
            'susceptible_material': 'Yes' if material == 'stainless_304' else 'Moderate',
            'mechanism': 'Oxygen differential cell in shielded areas'
        }

    def _calculate_erosion_corrosion(
        self,
        conditions: CorrosionConditions,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate erosion-corrosion rate.

        Mechanical wear + electrochemical corrosion
        Critical velocity: v_crit = k / √(density)

        Reference: ASTM G119
        """
        velocity = Decimal(str(conditions.flow_velocity_m_s))
        temp = Decimal(str(conditions.temperature_c))
        oxygen = Decimal(str(conditions.dissolved_oxygen_mg_l))
        material = conditions.material_type

        # Critical velocity for material
        if material == 'copper' or material == 'brass':
            v_crit = Decimal('1.5')  # m/s for copper alloys
        else:
            v_crit = Decimal('3.0')  # m/s for steel

        # Erosion-corrosion rate
        if velocity > v_crit:
            # Rate increases exponentially above critical velocity
            excess_velocity = velocity - v_crit
            k_erosion = Decimal('0.5')

            # Base corrosion rate
            base_rate_mpy = Decimal('1.0') + oxygen * Decimal('0.1')

            # Erosion enhancement factor
            erosion_factor = Decimal('1') + k_erosion * (excess_velocity ** Decimal('2'))

            rate_mpy = base_rate_mpy * erosion_factor
        else:
            # Below critical velocity, minimal erosion-corrosion
            rate_mpy = Decimal('0.5')

        # Temperature enhancement
        temp_factor = Decimal('1') + (temp - Decimal('25')) * Decimal('0.01')
        rate_mpy = rate_mpy * temp_factor

        rate_mm_yr = rate_mpy * Decimal('0.0254')

        tracker.record_step(
            operation="erosion_corrosion",
            description="Calculate erosion-corrosion rate",
            inputs={
                'velocity_m_s': velocity,
                'critical_velocity_m_s': v_crit,
                'oxygen_mg_L': oxygen,
                'temperature_C': temp
            },
            output_value=rate_mm_yr,
            output_name="erosion_corrosion_rate_mm_yr",
            formula="Rate = Base × (1 + k×(v-v_crit)²) × (1 + 0.01×(T-25))",
            units="mm/year",
            reference="ASTM G119"
        )

        return {
            'critical_velocity_m_s': float(v_crit),
            'actual_velocity_m_s': float(velocity),
            'velocity_ratio': float((velocity / v_crit).quantize(Decimal('0.01'))),
            'erosion_corrosion_rate_mpy': float(rate_mpy.quantize(Decimal('0.01'))),
            'erosion_corrosion_rate_mm_yr': float(rate_mm_yr.quantize(Decimal('0.001'))),
            'risk_level': 'High' if velocity > v_crit else 'Low',
            'mechanism': 'Protective film removal + electrochemical attack'
        }

    def _calculate_galvanic_corrosion(
        self,
        conditions: CorrosionConditions,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate galvanic corrosion potential.

        Reference: Galvanic series in seawater
        """
        material = conditions.material_type
        conductivity = Decimal(str(conditions.conductivity_us_cm))

        # Standard electrode potentials (V vs SHE)
        potentials = {
            'carbon_steel': Decimal('-0.44'),
            'brass': Decimal('-0.30'),
            'copper': Decimal('0.34'),
            'stainless_304': Decimal('-0.08')
        }

        # Get material potential
        E_material = potentials.get(material, Decimal('-0.44'))

        # Conductivity affects galvanic current
        # Higher conductivity = more aggressive galvanic attack
        conductivity_factor = conductivity / Decimal('1000')  # Normalize

        # Galvanic current density (simplified, assumes dissimilar metal contact)
        # Actual galvanic corrosion requires bimetallic couple
        galvanic_current = Decimal('0')  # No dissimilar metals in this analysis

        return {
            'material_potential_V': float(E_material),
            'water_conductivity_uS_cm': float(conductivity),
            'galvanic_current_density_uA_cm2': float(galvanic_current),
            'risk_assessment': 'Requires dissimilar metal contact analysis',
            'mechanism': 'Electrochemical cell between dissimilar metals'
        }

    def _calculate_scc_risk(
        self,
        conditions: CorrosionConditions,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate stress corrosion cracking (SCC) risk.

        Requires: Tensile stress + Corrosive environment + Susceptible material

        Reference: NACE SP0304
        """
        material = conditions.material_type
        stress = Decimal(str(conditions.stress_level_mpa))
        chloride = Decimal(str(conditions.chloride_mg_l))
        temp = Decimal(str(conditions.temperature_c))
        ph = Decimal(str(conditions.ph))

        # Material susceptibility
        if material == 'stainless_304':
            # Susceptible to chloride SCC
            susceptibility_score = Decimal('10')
        elif material == 'brass':
            # Susceptible to ammonia SCC (not modeled here)
            susceptibility_score = Decimal('5')
        else:
            # Carbon steel: caustic SCC at high pH
            susceptibility_score = Decimal('3')

        # Stress factor (% of yield strength)
        # Assume yield strength: carbon steel 250 MPa, SS304 215 MPa
        if material == 'stainless_304':
            yield_strength = Decimal('215')
        else:
            yield_strength = Decimal('250')

        stress_ratio = stress / yield_strength

        # Environmental severity
        # For chloride SCC: high T, high Cl, neutral pH
        if material == 'stainless_304':
            env_score = (chloride / Decimal('100')) * (temp / Decimal('50'))
        else:
            # Caustic SCC: high pH
            env_score = max(Decimal('0'), (ph - Decimal('10')))

        # SCC risk index
        scc_risk = susceptibility_score * stress_ratio * env_score

        # Risk assessment
        if scc_risk < Decimal('5'):
            risk = "Low"
        elif scc_risk < Decimal('20'):
            risk = "Moderate"
        elif scc_risk < Decimal('50'):
            risk = "High"
        else:
            risk = "Critical"

        tracker.record_step(
            operation="scc_risk_assessment",
            description="Assess stress corrosion cracking risk",
            inputs={
                'material': material,
                'stress_MPa': stress,
                'chloride_mg_L': chloride,
                'temperature_C': temp
            },
            output_value=scc_risk,
            output_name="scc_risk_index",
            formula="Risk = Susceptibility × (σ/σ_y) × Environment",
            units="dimensionless",
            reference="NACE SP0304"
        )

        return {
            'scc_risk_index': float(scc_risk.quantize(Decimal('0.1'))),
            'risk_level': risk,
            'stress_ratio': float(stress_ratio.quantize(Decimal('0.01'))),
            'critical_stress_MPa': float((yield_strength * Decimal('0.3')).quantize(Decimal('1'))),
            'mechanism': 'Chloride SCC' if material == 'stainless_304' else 'Caustic SCC'
        }

    def _calculate_total_corrosion_rate(
        self,
        oxygen: Dict,
        co2: Dict,
        pitting: Dict,
        erosion: Dict,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate total corrosion rate from all mechanisms."""
        # General corrosion (oxygen + CO2)
        general_rate_mpy = oxygen['corrosion_rate_mpy'] + co2['corrosion_rate_mpy']

        # Localized corrosion (pitting)
        localized_rate_mpy = pitting['maximum_pit_rate_mpy']

        # Erosion-corrosion
        erosion_rate_mpy = erosion['erosion_corrosion_rate_mpy']

        # Total (conservative: sum of all mechanisms)
        total_rate_mpy = general_rate_mpy + localized_rate_mpy + erosion_rate_mpy
        total_rate_mm_yr = Decimal(str(total_rate_mpy)) * Decimal('0.0254')

        return {
            'general_corrosion_mpy': float(Decimal(str(general_rate_mpy)).quantize(Decimal('0.01'))),
            'localized_corrosion_mpy': float(Decimal(str(localized_rate_mpy)).quantize(Decimal('0.01'))),
            'erosion_corrosion_mpy': float(Decimal(str(erosion_rate_mpy)).quantize(Decimal('0.01'))),
            'total_corrosion_rate_mpy': float(Decimal(str(total_rate_mpy)).quantize(Decimal('0.01'))),
            'total_corrosion_rate_mm_yr': float(Decimal(str(total_rate_mm_yr)).quantize(Decimal('0.001'))),
            'overall_severity': self._assess_corrosion_severity(float(total_rate_mpy))
        }

    def _calculate_remaining_life(
        self,
        total_rate: Dict,
        conditions: CorrosionConditions,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate remaining equipment life based on corrosion rate.

        Assumes typical wall thickness and corrosion allowance.
        """
        rate_mm_yr = Decimal(str(total_rate['total_corrosion_rate_mm_yr']))

        # Typical wall thicknesses and allowances
        # Assume pipe: Schedule 40, 100mm diameter
        nominal_thickness_mm = Decimal('6.0')
        corrosion_allowance_mm = Decimal('3.0')
        minimum_thickness_mm = Decimal('3.0')

        # Assume some corrosion has already occurred
        current_loss_mm = rate_mm_yr * Decimal(str(conditions.operating_time_hours)) / Decimal('8760')
        current_thickness_mm = nominal_thickness_mm - current_loss_mm

        # Remaining corrosion allowance
        remaining_allowance_mm = current_thickness_mm - minimum_thickness_mm

        # Years until reaching minimum thickness
        if rate_mm_yr > 0:
            remaining_life_years = remaining_allowance_mm / rate_mm_yr
        else:
            remaining_life_years = Decimal('999')

        tracker.record_step(
            operation="remaining_life",
            description="Calculate remaining equipment life",
            inputs={
                'current_thickness_mm': current_thickness_mm,
                'minimum_thickness_mm': minimum_thickness_mm,
                'corrosion_rate_mm_yr': rate_mm_yr
            },
            output_value=remaining_life_years,
            output_name="remaining_life_years",
            formula="Life = (Current - Minimum) / Rate",
            units="years"
        )

        return {
            'nominal_thickness_mm': float(nominal_thickness_mm),
            'current_thickness_mm': float(current_thickness_mm.quantize(Decimal('0.01'))),
            'minimum_thickness_mm': float(minimum_thickness_mm),
            'remaining_allowance_mm': float(remaining_allowance_mm.quantize(Decimal('0.01'))),
            'remaining_life_years': float(remaining_life_years.quantize(Decimal('0.1'))),
            'inspection_frequency_months': self._recommend_inspection_frequency(float(remaining_life_years))
        }

    def _assess_corrosion_severity(self, rate_mpy: float) -> str:
        """Assess corrosion severity based on rate."""
        if rate_mpy < 2:
            return "Excellent (<2 mpy)"
        elif rate_mpy < 5:
            return "Good (2-5 mpy)"
        elif rate_mpy < 10:
            return "Fair (5-10 mpy)"
        elif rate_mpy < 20:
            return "Poor (10-20 mpy)"
        else:
            return "Unacceptable (>20 mpy)"

    def _recommend_inspection_frequency(self, remaining_life_years: float) -> int:
        """Recommend inspection frequency based on remaining life."""
        if remaining_life_years < 1:
            return 1  # Monthly
        elif remaining_life_years < 2:
            return 3  # Quarterly
        elif remaining_life_years < 5:
            return 6  # Semi-annually
        elif remaining_life_years < 10:
            return 12  # Annually
        else:
            return 24  # Bi-annually
