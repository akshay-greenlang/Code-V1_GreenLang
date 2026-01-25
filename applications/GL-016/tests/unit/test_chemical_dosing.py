# -*- coding: utf-8 -*-
"""
Unit tests for Chemical Dosing Calculator - GL-016 WATERGUARD

Comprehensive test suite covering:
- Oxygen scavenger dosing for different chemicals
- Scale inhibitor calculations
- pH adjustment dosing
- Biocide schedules
- Cost calculations

Target: 95%+ code coverage
"""

import pytest
import math
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from hypothesis import given, strategies as st, settings, assume

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.provenance import ProvenanceTracker


# ============================================================================
# Chemical Dosing Calculator Class (for testing)
# ============================================================================

class ChemicalDosingCalculator:
    """
    Chemical dosing calculator for boiler water treatment.

    Calculates dosing requirements for:
    - Oxygen scavengers (sulfite, hydrazine, DEHA, carbohydrazide)
    - Scale inhibitors (phosphate, polymer)
    - pH adjusters (caustic, amine)
    - Biocides (chlorine, non-oxidizing)
    """

    # Chemical properties
    OXYGEN_SCAVENGER_RATIOS = {
        'sodium_sulfite': Decimal('8.0'),      # 8 ppm sulfite per 1 ppm O2
        'sodium_bisulfite': Decimal('7.0'),     # 7 ppm bisulfite per 1 ppm O2
        'hydrazine': Decimal('1.0'),            # 1 ppm hydrazine per 1 ppm O2
        'deha': Decimal('2.5'),                 # 2.5 ppm DEHA per 1 ppm O2
        'carbohydrazide': Decimal('1.5'),       # 1.5 ppm carbohydrazide per 1 ppm O2
        'erythorbic_acid': Decimal('4.0'),      # 4 ppm erythorbic acid per 1 ppm O2
    }

    CHEMICAL_COSTS_PER_KG = {
        'sodium_sulfite': Decimal('1.50'),
        'sodium_bisulfite': Decimal('1.20'),
        'hydrazine': Decimal('15.00'),
        'deha': Decimal('8.00'),
        'carbohydrazide': Decimal('12.00'),
        'trisodium_phosphate': Decimal('2.50'),
        'disodium_phosphate': Decimal('2.80'),
        'sodium_hydroxide': Decimal('0.80'),
        'sulfuric_acid': Decimal('0.60'),
        'morpholine': Decimal('6.00'),
        'cyclohexylamine': Decimal('7.50'),
    }

    def __init__(self, version: str = "1.0.0"):
        self.version = version

    def calculate_oxygen_scavenger_dosing(
        self,
        dissolved_oxygen_ppm: float,
        feedwater_flow_m3_hr: float,
        scavenger_type: str = 'sodium_sulfite',
        target_residual_ppm: float = 20.0,
        scavenger_concentration_percent: float = 100.0
    ) -> Dict[str, Any]:
        """
        Calculate oxygen scavenger dosing requirements.

        Dosing Rate = (O2 x Ratio + Residual) x Flow / Concentration
        """
        tracker = ProvenanceTracker(
            calculation_id=f"o2_scavenger_{id(self)}",
            calculation_type="oxygen_scavenger_dosing",
            version=self.version
        )

        do = Decimal(str(dissolved_oxygen_ppm))
        flow = Decimal(str(feedwater_flow_m3_hr))
        residual = Decimal(str(target_residual_ppm))
        concentration = Decimal(str(scavenger_concentration_percent)) / Decimal('100')

        tracker.record_inputs({
            'dissolved_oxygen_ppm': do,
            'feedwater_flow_m3_hr': flow,
            'scavenger_type': scavenger_type,
            'target_residual_ppm': residual
        })

        # Validate scavenger type
        if scavenger_type not in self.OXYGEN_SCAVENGER_RATIOS:
            raise ValueError(f"Unknown scavenger type: {scavenger_type}")

        ratio = self.OXYGEN_SCAVENGER_RATIOS[scavenger_type]

        # Calculate required dosing (ppm in feedwater)
        required_ppm = (do * ratio) + residual

        # Convert to mass flow (kg/hr)
        # ppm = mg/L, flow in m3/hr, 1 m3 = 1000 L
        mass_flow_kg_hr = (required_ppm * flow * Decimal('1000')) / Decimal('1000000')

        # Adjust for solution concentration
        solution_flow_kg_hr = mass_flow_kg_hr / concentration

        # Daily consumption
        daily_consumption_kg = solution_flow_kg_hr * Decimal('24')

        # Cost calculation
        chemical_cost = self.CHEMICAL_COSTS_PER_KG.get(scavenger_type, Decimal('2.00'))
        daily_cost = daily_consumption_kg * chemical_cost

        tracker.record_step(
            operation="oxygen_scavenger_dosing",
            description=f"Calculate {scavenger_type} dosing for oxygen removal",
            inputs={
                'dissolved_oxygen': do,
                'ratio': ratio,
                'residual': residual,
                'flow': flow
            },
            output_value=solution_flow_kg_hr,
            output_name="solution_flow_kg_hr",
            formula="Dose = (O2 x Ratio + Residual) x Flow / Conc",
            units="kg/hr"
        )

        return {
            'scavenger_type': scavenger_type,
            'required_ppm': float(required_ppm.quantize(Decimal('0.01'))),
            'solution_flow_kg_hr': float(solution_flow_kg_hr.quantize(Decimal('0.001'))),
            'solution_flow_L_hr': float((solution_flow_kg_hr * Decimal('1000')).quantize(Decimal('0.01'))),
            'daily_consumption_kg': float(daily_consumption_kg.quantize(Decimal('0.1'))),
            'daily_cost': float(daily_cost.quantize(Decimal('0.01'))),
            'stoichiometric_ratio': float(ratio),
            'provenance': tracker.get_provenance_record(solution_flow_kg_hr).to_dict()
        }

    def calculate_phosphate_dosing(
        self,
        feedwater_flow_m3_hr: float,
        target_phosphate_ppm: float = 30.0,
        current_phosphate_ppm: float = 0.0,
        phosphate_type: str = 'trisodium_phosphate',
        solution_concentration_percent: float = 30.0
    ) -> Dict[str, Any]:
        """
        Calculate phosphate dosing for scale control.

        Phosphate Treatment Program:
        - Congruent phosphate: 2-6 ppm PO4, Na/PO4 ratio 2.6:1
        - Coordinated phosphate: variable based on pressure
        - Phosphate-polymer: with dispersants
        """
        tracker = ProvenanceTracker(
            calculation_id=f"phosphate_{id(self)}",
            calculation_type="phosphate_dosing",
            version=self.version
        )

        flow = Decimal(str(feedwater_flow_m3_hr))
        target = Decimal(str(target_phosphate_ppm))
        current = Decimal(str(current_phosphate_ppm))
        concentration = Decimal(str(solution_concentration_percent)) / Decimal('100')

        tracker.record_inputs({
            'feedwater_flow_m3_hr': flow,
            'target_phosphate_ppm': target,
            'current_phosphate_ppm': current,
            'phosphate_type': phosphate_type
        })

        # Required phosphate increase
        required_increase = target - current
        if required_increase < 0:
            required_increase = Decimal('0')

        # Mass flow calculation
        mass_flow_kg_hr = (required_increase * flow * Decimal('1000')) / Decimal('1000000')

        # Adjust for solution concentration
        solution_flow_kg_hr = mass_flow_kg_hr / concentration

        # Daily values
        daily_consumption_kg = solution_flow_kg_hr * Decimal('24')

        # Cost
        chemical_cost = self.CHEMICAL_COSTS_PER_KG.get(phosphate_type, Decimal('2.50'))
        daily_cost = daily_consumption_kg * chemical_cost

        tracker.record_step(
            operation="phosphate_dosing",
            description="Calculate phosphate dosing for scale control",
            inputs={'target': target, 'current': current, 'flow': flow},
            output_value=solution_flow_kg_hr,
            output_name="solution_flow_kg_hr",
            formula="Dose = (Target - Current) x Flow / Conc",
            units="kg/hr"
        )

        return {
            'phosphate_type': phosphate_type,
            'required_increase_ppm': float(required_increase.quantize(Decimal('0.01'))),
            'solution_flow_kg_hr': float(solution_flow_kg_hr.quantize(Decimal('0.001'))),
            'solution_flow_L_hr': float((solution_flow_kg_hr * Decimal('1000')).quantize(Decimal('0.01'))),
            'daily_consumption_kg': float(daily_consumption_kg.quantize(Decimal('0.1'))),
            'daily_cost': float(daily_cost.quantize(Decimal('0.01'))),
            'provenance': tracker.get_provenance_record(solution_flow_kg_hr).to_dict()
        }

    def calculate_ph_adjustment_dosing(
        self,
        current_ph: float,
        target_ph: float,
        water_volume_m3: float,
        alkalinity_ppm: float = 100.0,
        chemical_type: str = 'sodium_hydroxide'
    ) -> Dict[str, Any]:
        """
        Calculate pH adjustment chemical dosing.

        For raising pH: NaOH, Na2CO3
        For lowering pH: H2SO4, HCl
        """
        tracker = ProvenanceTracker(
            calculation_id=f"ph_adjust_{id(self)}",
            calculation_type="ph_adjustment",
            version=self.version
        )

        ph_current = Decimal(str(current_ph))
        ph_target = Decimal(str(target_ph))
        volume = Decimal(str(water_volume_m3))
        alk = Decimal(str(alkalinity_ppm))

        tracker.record_inputs({
            'current_ph': ph_current,
            'target_ph': ph_target,
            'water_volume_m3': volume,
            'alkalinity_ppm': alk,
            'chemical_type': chemical_type
        })

        # Calculate pH change required
        ph_change = ph_target - ph_current

        # Estimate chemical requirement based on alkalinity
        # Higher alkalinity = more chemical needed (buffering capacity)
        buffering_factor = Decimal('1') + (alk / Decimal('500'))

        # Approximate dosing (simplified model)
        # This is a rough estimate - actual depends on water chemistry
        if ph_change > 0:  # Need to raise pH
            # NaOH dosing: roughly 0.4 kg per m3 per pH unit for typical water
            base_dose = abs(ph_change) * Decimal('0.4') * volume * buffering_factor
            action = "raise"
        elif ph_change < 0:  # Need to lower pH
            # H2SO4 dosing: roughly 0.3 kg per m3 per pH unit
            base_dose = abs(ph_change) * Decimal('0.3') * volume * buffering_factor
            action = "lower"
        else:
            base_dose = Decimal('0')
            action = "none"

        # Cost calculation
        chemical_cost = self.CHEMICAL_COSTS_PER_KG.get(chemical_type, Decimal('1.00'))
        total_cost = base_dose * chemical_cost

        tracker.record_step(
            operation="ph_adjustment",
            description=f"Calculate chemical to {action} pH",
            inputs={'ph_change': ph_change, 'buffering_factor': buffering_factor},
            output_value=base_dose,
            output_name="chemical_dose_kg",
            formula="Dose = |pH_change| x Factor x Volume x Buffering",
            units="kg"
        )

        return {
            'chemical_type': chemical_type,
            'ph_change_required': float(ph_change.quantize(Decimal('0.01'))),
            'action': action,
            'chemical_dose_kg': float(base_dose.quantize(Decimal('0.01'))),
            'buffering_factor': float(buffering_factor.quantize(Decimal('0.01'))),
            'total_cost': float(total_cost.quantize(Decimal('0.01'))),
            'provenance': tracker.get_provenance_record(base_dose).to_dict()
        }

    def calculate_biocide_dosing(
        self,
        system_volume_m3: float,
        biocide_type: str = 'chlorine',
        target_concentration_ppm: float = 2.0,
        frequency_hours: float = 24.0,
        decay_rate_per_hour: float = 0.1
    ) -> Dict[str, Any]:
        """
        Calculate biocide dosing schedule.

        Biocide types:
        - Oxidizing: chlorine, bromine, chlorine dioxide
        - Non-oxidizing: glutaraldehyde, isothiazolone, quaternary ammonium
        """
        tracker = ProvenanceTracker(
            calculation_id=f"biocide_{id(self)}",
            calculation_type="biocide_dosing",
            version=self.version
        )

        volume = Decimal(str(system_volume_m3))
        target = Decimal(str(target_concentration_ppm))
        frequency = Decimal(str(frequency_hours))
        decay = Decimal(str(decay_rate_per_hour))

        tracker.record_inputs({
            'system_volume_m3': volume,
            'biocide_type': biocide_type,
            'target_concentration_ppm': target,
            'frequency_hours': frequency,
            'decay_rate_per_hour': decay
        })

        # Calculate dose needed considering decay
        # After time t: C = C0 * e^(-decay * t)
        # At dosing time, concentration has decayed to:
        # C_remaining = target * e^(-decay * frequency)

        import math
        decay_factor = Decimal(str(math.exp(-float(decay * frequency))))
        remaining_concentration = target * decay_factor

        # Need to dose to bring back up to target
        dose_concentration = target - remaining_concentration

        # Convert to mass (kg per dose)
        # ppm = mg/L = g/m3, so mass (kg) = ppm * volume / 1000
        dose_mass_kg = (dose_concentration * volume) / Decimal('1000')

        # Daily consumption (doses per day)
        doses_per_day = Decimal('24') / frequency
        daily_consumption_kg = dose_mass_kg * doses_per_day

        # Contact time recommendation
        if biocide_type in ['chlorine', 'bromine']:
            contact_time_min = 30
        else:
            contact_time_min = 60

        tracker.record_step(
            operation="biocide_dosing",
            description=f"Calculate {biocide_type} dosing schedule",
            inputs={
                'target': target,
                'decay_factor': decay_factor,
                'dose_concentration': dose_concentration
            },
            output_value=dose_mass_kg,
            output_name="dose_mass_kg",
            formula="Dose = (Target - Remaining) x Volume / 1000",
            units="kg"
        )

        return {
            'biocide_type': biocide_type,
            'dose_mass_kg': float(dose_mass_kg.quantize(Decimal('0.001'))),
            'dose_concentration_ppm': float(dose_concentration.quantize(Decimal('0.01'))),
            'doses_per_day': float(doses_per_day.quantize(Decimal('0.1'))),
            'daily_consumption_kg': float(daily_consumption_kg.quantize(Decimal('0.01'))),
            'contact_time_min': contact_time_min,
            'schedule': [
                {'hour': int(i * float(frequency)), 'dose_kg': float(dose_mass_kg.quantize(Decimal('0.001')))}
                for i in range(int(24 / float(frequency)))
            ],
            'provenance': tracker.get_provenance_record(dose_mass_kg).to_dict()
        }

    def calculate_scale_inhibitor_dosing(
        self,
        feedwater_flow_m3_hr: float,
        hardness_ppm: float,
        target_dose_ppm: float = 5.0,
        inhibitor_type: str = 'phosphonate',
        solution_concentration_percent: float = 25.0
    ) -> Dict[str, Any]:
        """
        Calculate scale inhibitor dosing.

        Inhibitor types:
        - Phosphonate (HEDP, ATMP, PBTC)
        - Polymer (polyacrylate, phosphinocarboxylic acid)
        - Combined (phosphonate + polymer)
        """
        tracker = ProvenanceTracker(
            calculation_id=f"scale_inhibitor_{id(self)}",
            calculation_type="scale_inhibitor_dosing",
            version=self.version
        )

        flow = Decimal(str(feedwater_flow_m3_hr))
        hardness = Decimal(str(hardness_ppm))
        target = Decimal(str(target_dose_ppm))
        concentration = Decimal(str(solution_concentration_percent)) / Decimal('100')

        tracker.record_inputs({
            'feedwater_flow_m3_hr': flow,
            'hardness_ppm': hardness,
            'target_dose_ppm': target,
            'inhibitor_type': inhibitor_type
        })

        # Adjust dose based on hardness
        # Higher hardness may need higher dose
        hardness_factor = Decimal('1') + (hardness / Decimal('500'))
        adjusted_dose = target * hardness_factor

        # Mass flow (kg/hr)
        mass_flow_kg_hr = (adjusted_dose * flow * Decimal('1000')) / Decimal('1000000')

        # Solution flow
        solution_flow_kg_hr = mass_flow_kg_hr / concentration

        # Daily values
        daily_consumption_kg = solution_flow_kg_hr * Decimal('24')

        tracker.record_step(
            operation="scale_inhibitor_dosing",
            description=f"Calculate {inhibitor_type} dosing",
            inputs={
                'target': target,
                'hardness_factor': hardness_factor,
                'flow': flow
            },
            output_value=solution_flow_kg_hr,
            output_name="solution_flow_kg_hr",
            formula="Dose = Target x Hardness_Factor x Flow / Conc",
            units="kg/hr"
        )

        return {
            'inhibitor_type': inhibitor_type,
            'adjusted_dose_ppm': float(adjusted_dose.quantize(Decimal('0.01'))),
            'hardness_factor': float(hardness_factor.quantize(Decimal('0.01'))),
            'solution_flow_kg_hr': float(solution_flow_kg_hr.quantize(Decimal('0.001'))),
            'daily_consumption_kg': float(daily_consumption_kg.quantize(Decimal('0.1'))),
            'provenance': tracker.get_provenance_record(solution_flow_kg_hr).to_dict()
        }

    def calculate_total_treatment_cost(
        self,
        dosing_results: List[Dict[str, Any]],
        operating_hours_per_day: float = 24.0
    ) -> Dict[str, Any]:
        """
        Calculate total water treatment cost from all dosing programs.
        """
        tracker = ProvenanceTracker(
            calculation_id=f"total_cost_{id(self)}",
            calculation_type="total_treatment_cost",
            version=self.version
        )

        total_daily_cost = Decimal('0')
        cost_breakdown = []

        for result in dosing_results:
            if 'daily_cost' in result:
                cost = Decimal(str(result['daily_cost']))
                total_daily_cost += cost
                cost_breakdown.append({
                    'program': result.get('scavenger_type') or result.get('phosphate_type')
                              or result.get('biocide_type') or result.get('inhibitor_type'),
                    'daily_cost': float(cost.quantize(Decimal('0.01')))
                })

        monthly_cost = total_daily_cost * Decimal('30')
        annual_cost = total_daily_cost * Decimal('365')

        tracker.record_inputs({
            'number_of_programs': len(dosing_results),
            'operating_hours_per_day': operating_hours_per_day
        })

        tracker.record_step(
            operation="total_cost",
            description="Sum all treatment program costs",
            inputs={'programs': len(cost_breakdown)},
            output_value=total_daily_cost,
            output_name="total_daily_cost",
            formula="Sum(all daily costs)",
            units="$/day"
        )

        return {
            'total_daily_cost': float(total_daily_cost.quantize(Decimal('0.01'))),
            'total_monthly_cost': float(monthly_cost.quantize(Decimal('0.01'))),
            'total_annual_cost': float(annual_cost.quantize(Decimal('0.01'))),
            'cost_breakdown': cost_breakdown,
            'provenance': tracker.get_provenance_record(total_daily_cost).to_dict()
        }


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def calculator():
    """Create ChemicalDosingCalculator instance."""
    return ChemicalDosingCalculator(version="1.0.0-test")


@pytest.fixture
def standard_feedwater_params():
    """Standard feedwater parameters."""
    return {
        'flow_m3_hr': 50.0,
        'dissolved_oxygen_ppm': 0.02,
        'hardness_ppm': 200.0,
        'alkalinity_ppm': 150.0,
        'ph': 7.5
    }


# ============================================================================
# Oxygen Scavenger Tests
# ============================================================================

@pytest.mark.unit
class TestOxygenScavengerDosing:
    """Test oxygen scavenger dosing calculations."""

    def test_sulfite_dosing_basic(self, calculator):
        """Test basic sodium sulfite dosing."""
        result = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=0.02,
            feedwater_flow_m3_hr=50.0,
            scavenger_type='sodium_sulfite',
            target_residual_ppm=20.0
        )

        # Expected: (0.02 * 8 + 20) * 50 * 1000 / 1000000 = 1.008 kg/hr
        assert result['required_ppm'] == pytest.approx(20.16, rel=0.01)
        assert 'provenance' in result

    def test_hydrazine_dosing(self, calculator):
        """Test hydrazine dosing (1:1 ratio)."""
        result = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=0.02,
            feedwater_flow_m3_hr=50.0,
            scavenger_type='hydrazine',
            target_residual_ppm=0.1  # Much lower residual
        )

        # Hydrazine ratio is 1:1, so required = 0.02 + 0.1 = 0.12 ppm
        assert result['required_ppm'] == pytest.approx(0.12, rel=0.01)
        assert result['stoichiometric_ratio'] == 1.0

    @pytest.mark.parametrize("scavenger,ratio", [
        ('sodium_sulfite', 8.0),
        ('sodium_bisulfite', 7.0),
        ('hydrazine', 1.0),
        ('deha', 2.5),
        ('carbohydrazide', 1.5),
    ])
    def test_scavenger_ratios(self, calculator, scavenger, ratio):
        """Test different scavenger stoichiometric ratios."""
        result = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=1.0,
            feedwater_flow_m3_hr=100.0,
            scavenger_type=scavenger,
            target_residual_ppm=0.0
        )

        assert result['stoichiometric_ratio'] == ratio
        # Required = O2 * ratio + residual = 1.0 * ratio
        assert result['required_ppm'] == pytest.approx(ratio, rel=0.01)

    def test_invalid_scavenger_type(self, calculator):
        """Test with invalid scavenger type."""
        with pytest.raises(ValueError):
            calculator.calculate_oxygen_scavenger_dosing(
                dissolved_oxygen_ppm=0.02,
                feedwater_flow_m3_hr=50.0,
                scavenger_type='unknown_chemical'
            )

    def test_high_oxygen_content(self, calculator):
        """Test with high dissolved oxygen."""
        result = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=8.0,  # Air-saturated water
            feedwater_flow_m3_hr=50.0,
            scavenger_type='sodium_sulfite',
            target_residual_ppm=20.0
        )

        # High O2 requires much more scavenger
        assert result['required_ppm'] == pytest.approx(84.0, rel=0.01)  # 8*8 + 20

    def test_dilute_solution(self, calculator):
        """Test with dilute scavenger solution."""
        result = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=0.02,
            feedwater_flow_m3_hr=50.0,
            scavenger_type='sodium_sulfite',
            target_residual_ppm=20.0,
            scavenger_concentration_percent=25.0  # 25% solution
        )

        result_full = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=0.02,
            feedwater_flow_m3_hr=50.0,
            scavenger_type='sodium_sulfite',
            target_residual_ppm=20.0,
            scavenger_concentration_percent=100.0  # Pure
        )

        # 25% solution needs 4x the volume
        assert result['solution_flow_kg_hr'] == pytest.approx(
            result_full['solution_flow_kg_hr'] * 4,
            rel=0.01
        )


# ============================================================================
# Phosphate Dosing Tests
# ============================================================================

@pytest.mark.unit
class TestPhosphateDosing:
    """Test phosphate dosing calculations."""

    def test_basic_phosphate_dosing(self, calculator):
        """Test basic phosphate dosing."""
        result = calculator.calculate_phosphate_dosing(
            feedwater_flow_m3_hr=50.0,
            target_phosphate_ppm=30.0,
            current_phosphate_ppm=10.0
        )

        # Required increase = 30 - 10 = 20 ppm
        assert result['required_increase_ppm'] == pytest.approx(20.0, rel=0.01)
        assert 'provenance' in result

    def test_phosphate_already_at_target(self, calculator):
        """Test when phosphate already at target."""
        result = calculator.calculate_phosphate_dosing(
            feedwater_flow_m3_hr=50.0,
            target_phosphate_ppm=30.0,
            current_phosphate_ppm=30.0
        )

        assert result['required_increase_ppm'] == 0.0
        assert result['solution_flow_kg_hr'] == 0.0

    def test_phosphate_above_target(self, calculator):
        """Test when phosphate above target (no dosing needed)."""
        result = calculator.calculate_phosphate_dosing(
            feedwater_flow_m3_hr=50.0,
            target_phosphate_ppm=30.0,
            current_phosphate_ppm=40.0
        )

        assert result['required_increase_ppm'] == 0.0

    @pytest.mark.parametrize("target,current,expected_increase", [
        (30.0, 0.0, 30.0),
        (30.0, 15.0, 15.0),
        (50.0, 10.0, 40.0),
        (20.0, 20.0, 0.0),
    ])
    def test_phosphate_parametrized(self, calculator, target, current, expected_increase):
        """Parametrized phosphate dosing tests."""
        result = calculator.calculate_phosphate_dosing(
            feedwater_flow_m3_hr=50.0,
            target_phosphate_ppm=target,
            current_phosphate_ppm=current
        )

        assert result['required_increase_ppm'] == pytest.approx(expected_increase, rel=0.01)


# ============================================================================
# pH Adjustment Tests
# ============================================================================

@pytest.mark.unit
class TestPHAdjustmentDosing:
    """Test pH adjustment dosing calculations."""

    def test_raise_ph(self, calculator):
        """Test raising pH with caustic."""
        result = calculator.calculate_ph_adjustment_dosing(
            current_ph=7.0,
            target_ph=9.0,
            water_volume_m3=100.0,
            alkalinity_ppm=100.0,
            chemical_type='sodium_hydroxide'
        )

        assert result['action'] == 'raise'
        assert result['ph_change_required'] == pytest.approx(2.0, rel=0.01)
        assert result['chemical_dose_kg'] > 0

    def test_lower_ph(self, calculator):
        """Test lowering pH with acid."""
        result = calculator.calculate_ph_adjustment_dosing(
            current_ph=10.0,
            target_ph=8.5,
            water_volume_m3=100.0,
            alkalinity_ppm=100.0,
            chemical_type='sulfuric_acid'
        )

        assert result['action'] == 'lower'
        assert result['ph_change_required'] == pytest.approx(-1.5, rel=0.01)
        assert result['chemical_dose_kg'] > 0

    def test_no_adjustment_needed(self, calculator):
        """Test when pH already at target."""
        result = calculator.calculate_ph_adjustment_dosing(
            current_ph=8.5,
            target_ph=8.5,
            water_volume_m3=100.0,
            alkalinity_ppm=100.0
        )

        assert result['action'] == 'none'
        assert result['chemical_dose_kg'] == 0.0

    def test_high_alkalinity_buffering(self, calculator):
        """Test high alkalinity requires more chemical."""
        result_low_alk = calculator.calculate_ph_adjustment_dosing(
            current_ph=7.0,
            target_ph=8.0,
            water_volume_m3=100.0,
            alkalinity_ppm=50.0
        )

        result_high_alk = calculator.calculate_ph_adjustment_dosing(
            current_ph=7.0,
            target_ph=8.0,
            water_volume_m3=100.0,
            alkalinity_ppm=500.0
        )

        # Higher alkalinity = more chemical needed
        assert result_high_alk['chemical_dose_kg'] > result_low_alk['chemical_dose_kg']


# ============================================================================
# Biocide Dosing Tests
# ============================================================================

@pytest.mark.unit
class TestBiocideDosing:
    """Test biocide dosing calculations."""

    def test_chlorine_dosing(self, calculator):
        """Test chlorine biocide dosing."""
        result = calculator.calculate_biocide_dosing(
            system_volume_m3=500.0,
            biocide_type='chlorine',
            target_concentration_ppm=2.0,
            frequency_hours=8.0
        )

        assert result['biocide_type'] == 'chlorine'
        assert result['doses_per_day'] == 3.0
        assert result['contact_time_min'] == 30
        assert len(result['schedule']) == 3

    def test_non_oxidizing_biocide(self, calculator):
        """Test non-oxidizing biocide dosing."""
        result = calculator.calculate_biocide_dosing(
            system_volume_m3=500.0,
            biocide_type='glutaraldehyde',
            target_concentration_ppm=50.0,
            frequency_hours=24.0
        )

        assert result['contact_time_min'] == 60  # Longer for non-oxidizing
        assert result['doses_per_day'] == 1.0

    def test_decay_rate_effect(self, calculator):
        """Test effect of decay rate on dosing."""
        result_slow_decay = calculator.calculate_biocide_dosing(
            system_volume_m3=500.0,
            biocide_type='chlorine',
            target_concentration_ppm=2.0,
            frequency_hours=8.0,
            decay_rate_per_hour=0.05  # Slow decay
        )

        result_fast_decay = calculator.calculate_biocide_dosing(
            system_volume_m3=500.0,
            biocide_type='chlorine',
            target_concentration_ppm=2.0,
            frequency_hours=8.0,
            decay_rate_per_hour=0.2  # Fast decay
        )

        # Fast decay requires more chemical per dose
        assert result_fast_decay['dose_mass_kg'] > result_slow_decay['dose_mass_kg']

    def test_dosing_schedule(self, calculator):
        """Test biocide dosing schedule generation."""
        result = calculator.calculate_biocide_dosing(
            system_volume_m3=500.0,
            biocide_type='chlorine',
            target_concentration_ppm=2.0,
            frequency_hours=6.0
        )

        # 24/6 = 4 doses per day
        assert len(result['schedule']) == 4
        assert result['schedule'][0]['hour'] == 0
        assert result['schedule'][1]['hour'] == 6
        assert result['schedule'][2]['hour'] == 12
        assert result['schedule'][3]['hour'] == 18


# ============================================================================
# Scale Inhibitor Tests
# ============================================================================

@pytest.mark.unit
class TestScaleInhibitorDosing:
    """Test scale inhibitor dosing calculations."""

    def test_basic_inhibitor_dosing(self, calculator):
        """Test basic scale inhibitor dosing."""
        result = calculator.calculate_scale_inhibitor_dosing(
            feedwater_flow_m3_hr=50.0,
            hardness_ppm=200.0,
            target_dose_ppm=5.0,
            inhibitor_type='phosphonate'
        )

        assert result['inhibitor_type'] == 'phosphonate'
        assert result['hardness_factor'] > 1.0
        assert 'provenance' in result

    def test_hardness_factor_effect(self, calculator):
        """Test hardness factor increases dose."""
        result_low = calculator.calculate_scale_inhibitor_dosing(
            feedwater_flow_m3_hr=50.0,
            hardness_ppm=100.0,
            target_dose_ppm=5.0
        )

        result_high = calculator.calculate_scale_inhibitor_dosing(
            feedwater_flow_m3_hr=50.0,
            hardness_ppm=500.0,
            target_dose_ppm=5.0
        )

        # Higher hardness = higher adjusted dose
        assert result_high['adjusted_dose_ppm'] > result_low['adjusted_dose_ppm']

    @pytest.mark.parametrize("hardness,expected_factor", [
        (0, 1.0),
        (250, 1.5),
        (500, 2.0),
        (1000, 3.0),
    ])
    def test_hardness_factor_calculation(self, calculator, hardness, expected_factor):
        """Test hardness factor calculation."""
        result = calculator.calculate_scale_inhibitor_dosing(
            feedwater_flow_m3_hr=50.0,
            hardness_ppm=hardness,
            target_dose_ppm=5.0
        )

        assert result['hardness_factor'] == pytest.approx(expected_factor, rel=0.01)


# ============================================================================
# Total Cost Calculation Tests
# ============================================================================

@pytest.mark.unit
class TestTotalCostCalculation:
    """Test total treatment cost calculations."""

    def test_total_cost_single_program(self, calculator):
        """Test total cost with single program."""
        scavenger_result = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=0.02,
            feedwater_flow_m3_hr=50.0,
            scavenger_type='sodium_sulfite'
        )

        total_result = calculator.calculate_total_treatment_cost([scavenger_result])

        assert total_result['total_daily_cost'] == pytest.approx(
            scavenger_result['daily_cost'],
            rel=0.01
        )

    def test_total_cost_multiple_programs(self, calculator):
        """Test total cost with multiple programs."""
        scavenger = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=0.02,
            feedwater_flow_m3_hr=50.0,
            scavenger_type='sodium_sulfite'
        )

        phosphate = calculator.calculate_phosphate_dosing(
            feedwater_flow_m3_hr=50.0,
            target_phosphate_ppm=30.0,
            current_phosphate_ppm=10.0
        )

        total = calculator.calculate_total_treatment_cost([scavenger, phosphate])

        expected_daily = scavenger['daily_cost'] + phosphate['daily_cost']
        assert total['total_daily_cost'] == pytest.approx(expected_daily, rel=0.01)
        assert len(total['cost_breakdown']) == 2

    def test_monthly_annual_calculation(self, calculator):
        """Test monthly and annual cost calculation."""
        scavenger = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=0.02,
            feedwater_flow_m3_hr=50.0,
            scavenger_type='sodium_sulfite'
        )

        total = calculator.calculate_total_treatment_cost([scavenger])

        assert total['total_monthly_cost'] == pytest.approx(
            total['total_daily_cost'] * 30,
            rel=0.01
        )
        assert total['total_annual_cost'] == pytest.approx(
            total['total_daily_cost'] * 365,
            rel=0.01
        )


# ============================================================================
# Determinism Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.determinism
class TestDeterminism:
    """Test calculation determinism."""

    def test_scavenger_deterministic(self, calculator):
        """Test oxygen scavenger calculation is deterministic."""
        results = [
            calculator.calculate_oxygen_scavenger_dosing(
                dissolved_oxygen_ppm=0.02,
                feedwater_flow_m3_hr=50.0,
                scavenger_type='sodium_sulfite'
            )
            for _ in range(10)
        ]

        hashes = [r['provenance']['provenance_hash'] for r in results]
        assert len(set(hashes)) == 1

    def test_phosphate_deterministic(self, calculator):
        """Test phosphate calculation is deterministic."""
        results = [
            calculator.calculate_phosphate_dosing(
                feedwater_flow_m3_hr=50.0,
                target_phosphate_ppm=30.0,
                current_phosphate_ppm=10.0
            )
            for _ in range(10)
        ]

        hashes = [r['provenance']['provenance_hash'] for r in results]
        assert len(set(hashes)) == 1

    def test_biocide_deterministic(self, calculator):
        """Test biocide calculation is deterministic."""
        results = [
            calculator.calculate_biocide_dosing(
                system_volume_m3=500.0,
                biocide_type='chlorine',
                target_concentration_ppm=2.0
            )
            for _ in range(10)
        ]

        hashes = [r['provenance']['provenance_hash'] for r in results]
        assert len(set(hashes)) == 1


# ============================================================================
# Property-Based Tests
# ============================================================================

@pytest.mark.unit
class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        do=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        flow=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        residual=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50)
    def test_scavenger_always_positive(self, calculator, do, flow, residual):
        """Test scavenger dosing is always positive."""
        result = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=do,
            feedwater_flow_m3_hr=flow,
            scavenger_type='sodium_sulfite',
            target_residual_ppm=residual
        )

        assert result['required_ppm'] >= 0
        assert result['solution_flow_kg_hr'] >= 0

    @given(
        volume=st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        target=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=30)
    def test_biocide_dose_proportional_to_volume(self, calculator, volume, target):
        """Test biocide dose scales with volume."""
        result = calculator.calculate_biocide_dosing(
            system_volume_m3=volume,
            biocide_type='chlorine',
            target_concentration_ppm=target
        )

        # Dose should scale linearly with volume
        result_double = calculator.calculate_biocide_dosing(
            system_volume_m3=volume * 2,
            biocide_type='chlorine',
            target_concentration_ppm=target
        )

        assert result_double['dose_mass_kg'] == pytest.approx(
            result['dose_mass_kg'] * 2,
            rel=0.01
        )


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.unit
class TestIntegration:
    """Test integration between chemical dosing calculations."""

    def test_complete_treatment_program(self, calculator, standard_feedwater_params):
        """Test complete water treatment program calculation."""
        # Oxygen scavenger
        scavenger = calculator.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen_ppm=standard_feedwater_params['dissolved_oxygen_ppm'],
            feedwater_flow_m3_hr=standard_feedwater_params['flow_m3_hr'],
            scavenger_type='sodium_sulfite'
        )

        # Phosphate
        phosphate = calculator.calculate_phosphate_dosing(
            feedwater_flow_m3_hr=standard_feedwater_params['flow_m3_hr'],
            target_phosphate_ppm=30.0,
            current_phosphate_ppm=5.0
        )

        # Scale inhibitor
        inhibitor = calculator.calculate_scale_inhibitor_dosing(
            feedwater_flow_m3_hr=standard_feedwater_params['flow_m3_hr'],
            hardness_ppm=standard_feedwater_params['hardness_ppm']
        )

        # pH adjustment
        ph_adjust = calculator.calculate_ph_adjustment_dosing(
            current_ph=standard_feedwater_params['ph'],
            target_ph=9.0,
            water_volume_m3=100.0,
            alkalinity_ppm=standard_feedwater_params['alkalinity_ppm']
        )

        # Total cost
        total = calculator.calculate_total_treatment_cost(
            [scavenger, phosphate, inhibitor]
        )

        assert total['total_daily_cost'] > 0
        assert len(total['cost_breakdown']) == 3
        assert 'provenance' in total
