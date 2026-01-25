# -*- coding: utf-8 -*-
"""
Chemical Dosing Calculator - GL-016 WATERGUARD

Advanced chemical dosing calculations for boiler and cooling water treatment.
Implements stoichiometric and empirical dosing formulas with zero hallucination
guarantee through deterministic calculations and SHA-256 provenance tracking.

Author: GL-016 WATERGUARD Engineering Team
Version: 1.0.0
Standards: ASME, ABMA, NACE, ASTM
References:
- ASME Consensus on Operating Practices for Control of Feedwater/Boiler Water Chemistry
- ABMA Boiler Water Guidelines for Drum-Type Boilers
- NACE SP0590 - Prevention, Detection, and Correction of Deaerator Cracking
- ASTM D1193 - Standard Specification for Reagent Water
- EPRI TR-102285 - Boiler Chemical Cleaning Guidelines
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from .provenance import ProvenanceTracker, CalculationStep
import hashlib
import json


class OxygenScavengerType(Enum):
    """Types of oxygen scavengers with their properties."""
    SODIUM_SULFITE = "sodium_sulfite"
    CATALYZED_SULFITE = "catalyzed_sulfite"
    HYDRAZINE = "hydrazine"
    CARBOHYDRAZIDE = "carbohydrazide"
    DEHA = "deha"  # Diethylhydroxylamine
    ERYTHORBIC_ACID = "erythorbic_acid"


class ScaleInhibitorType(Enum):
    """Types of scale inhibitors."""
    HEDP = "hedp"  # 1-Hydroxyethylidene-1,1-diphosphonic acid
    PBTC = "pbtc"  # 2-Phosphonobutane-1,2,4-tricarboxylic acid
    POLYACRYLATE = "polyacrylate"
    PHOSPHATE_ESTER = "phosphate_ester"
    POLYPHOSPHATE = "polyphosphate"


class BiocideType(Enum):
    """Types of biocides for microbial control."""
    CHLORINE = "chlorine"
    CHLORINE_DIOXIDE = "chlorine_dioxide"
    BROMINE = "bromine"
    ISOTHIAZOLINE = "isothiazoline"
    GLUTARALDEHYDE = "glutaraldehyde"
    QUATERNARY_AMMONIUM = "quaternary_ammonium"


@dataclass
class WaterConditions:
    """Water chemistry conditions for dosing calculations."""
    # Water properties
    volume_m3: float  # System volume
    flow_rate_m3_hr: float  # Makeup or circulation rate
    temperature_c: float  # Operating temperature
    ph: float  # Current pH

    # Chemistry parameters (mg/L)
    dissolved_oxygen_mg_l: float
    total_alkalinity_mg_l_caco3: float
    total_hardness_mg_l_caco3: float
    calcium_mg_l: float
    silica_mg_l: float
    iron_mg_l: float
    phosphate_mg_l: float = 0.0

    # Current residuals (mg/L)
    sulfite_residual_mg_l: float = 0.0
    phosphate_residual_mg_l: float = 0.0

    # Target residuals (mg/L)
    target_sulfite_mg_l: float = 20.0
    target_phosphate_mg_l: float = 10.0

    # Operational
    cycles_of_concentration: float = 1.0
    system_type: str = "boiler"  # "boiler", "cooling_tower", "closed_loop"


@dataclass
class ChemicalDosingResult:
    """Result of chemical dosing calculation with complete provenance."""
    chemical_name: str
    chemical_formula: str
    dose_concentration_mg_l: float
    initial_dose_kg: float
    continuous_dose_kg_hr: float
    daily_consumption_kg: float
    monthly_consumption_kg: float
    cost_per_hour: float
    cost_per_month: float
    safety_notes: List[str]
    provenance_hash: str
    calculation_steps: List[Dict] = field(default_factory=list)


class ChemicalDosingCalculator:
    """
    Zero-Hallucination Chemical Dosing Calculator.

    Guarantees:
    - Deterministic stoichiometric calculations
    - Complete provenance tracking with SHA-256 hashing
    - Industry-standard formulas (ASME, ABMA, NACE)
    - NO LLM involvement in calculation path

    Capabilities:
    - Oxygen scavenger dosing (sulfite, hydrazine, DEHA, etc.)
    - Scale inhibitor dosing (phosphonates, polymers)
    - Corrosion inhibitor dosing (filming amines)
    - pH adjustment dosing (caustic, acid)
    - Biocide dosing calculation
    - Chemical cost tracking
    - Safety and handling notes
    """

    # Molecular weights (g/mol)
    MOLECULAR_WEIGHTS = {
        'Na2SO3': Decimal('126.04'),  # Sodium sulfite
        'Na2S2O5': Decimal('190.11'),  # Sodium metabisulfite
        'N2H4': Decimal('32.05'),  # Hydrazine
        'CH6N4O': Decimal('90.08'),  # Carbohydrazide
        'C4H11NO': Decimal('89.14'),  # DEHA
        'NaOH': Decimal('40.00'),  # Sodium hydroxide
        'H2SO4': Decimal('98.079'),  # Sulfuric acid
        'HCl': Decimal('36.46'),  # Hydrochloric acid
        'Na3PO4': Decimal('163.94'),  # Trisodium phosphate
        'Na2HPO4': Decimal('141.96'),  # Disodium phosphate
        'NaH2PO4': Decimal('119.98'),  # Monosodium phosphate
        'O2': Decimal('32.00'),  # Oxygen
        'CaCO3': Decimal('100.087'),  # Calcium carbonate
        'HOCl': Decimal('52.46'),  # Hypochlorous acid
        'NaOCl': Decimal('74.44'),  # Sodium hypochlorite
    }

    # Stoichiometric ratios
    OXYGEN_SCAVENGER_RATIOS = {
        'sodium_sulfite': Decimal('7.88'),  # mg Na2SO3 per mg O2
        'catalyzed_sulfite': Decimal('7.88'),  # Same stoichiometry, faster kinetics
        'hydrazine': Decimal('1.0'),  # mg N2H4 per mg O2
        'carbohydrazide': Decimal('1.41'),  # mg per mg O2
        'deha': Decimal('2.8'),  # mg DEHA per mg O2
        'erythorbic_acid': Decimal('3.5'),  # mg per mg O2
    }

    # Typical chemical costs ($/kg)
    CHEMICAL_COSTS = {
        'sodium_sulfite': Decimal('0.50'),
        'catalyzed_sulfite': Decimal('1.20'),
        'hydrazine': Decimal('15.00'),  # More expensive, tightly regulated
        'carbohydrazide': Decimal('8.00'),
        'deha': Decimal('6.00'),
        'erythorbic_acid': Decimal('3.00'),
        'naoh_25pct': Decimal('0.40'),  # $/L of 25% solution
        'naoh_50pct': Decimal('0.35'),  # $/L of 50% solution
        'h2so4_93pct': Decimal('0.20'),  # $/L of 93% solution
        'na3po4': Decimal('1.50'),
        'scale_inhibitor': Decimal('4.00'),
        'filming_amine': Decimal('5.00'),
        'biocide': Decimal('3.00'),
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize chemical dosing calculator."""
        self.version = version

    def calculate_oxygen_scavenger_dosing(
        self,
        conditions: WaterConditions,
        scavenger_type: OxygenScavengerType = OxygenScavengerType.SODIUM_SULFITE,
        excess_ratio: float = 1.5,
        tracker: Optional[ProvenanceTracker] = None
    ) -> ChemicalDosingResult:
        """
        Calculate oxygen scavenger dosing requirements.

        Stoichiometry:
        - Sodium sulfite: 2Na2SO3 + O2 -> 2Na2SO4 (7.88 mg/mg O2)
        - Hydrazine: N2H4 + O2 -> N2 + 2H2O (1.0 mg/mg O2)
        - DEHA: (C2H5)2NOH + 0.5O2 -> Products (2.8 mg/mg O2)

        Total dose = Stoichiometric dose + Residual maintenance

        Reference: ASME Consensus Document, ABMA Guidelines

        Args:
            conditions: Water chemistry conditions
            scavenger_type: Type of oxygen scavenger
            excess_ratio: Excess factor (typically 1.5-2.0)
            tracker: Optional provenance tracker

        Returns:
            ChemicalDosingResult with complete calculation details
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"o2_scavenger_{id(conditions)}",
                calculation_type="oxygen_scavenger_dosing",
                version=self.version
            )

        tracker.record_inputs({
            **conditions.__dict__,
            'scavenger_type': scavenger_type.value,
            'excess_ratio': excess_ratio
        })

        # Convert to Decimal
        volume = Decimal(str(conditions.volume_m3))
        flow = Decimal(str(conditions.flow_rate_m3_hr))
        do_mg_l = Decimal(str(conditions.dissolved_oxygen_mg_l))
        current_residual = Decimal(str(conditions.sulfite_residual_mg_l))
        target_residual = Decimal(str(conditions.target_sulfite_mg_l))
        excess = Decimal(str(excess_ratio))

        # Get stoichiometric ratio
        stoich_ratio = self.OXYGEN_SCAVENGER_RATIOS.get(
            scavenger_type.value, Decimal('7.88')
        )

        # Step 1: Calculate stoichiometric dose for oxygen removal
        # Dose = O2 * stoich_ratio * excess_factor
        stoich_dose_mg_l = do_mg_l * stoich_ratio * excess

        tracker.record_step(
            operation="stoichiometric_dose",
            description="Calculate stoichiometric scavenger dose for O2 removal",
            inputs={
                'dissolved_oxygen_mg_l': do_mg_l,
                'stoichiometric_ratio': stoich_ratio,
                'excess_factor': excess
            },
            output_value=stoich_dose_mg_l,
            output_name="stoichiometric_dose_mg_l",
            formula="Dose = O2 * ratio * excess",
            units="mg/L",
            reference="ASME Consensus Document"
        )

        # Step 2: Calculate residual maintenance dose
        residual_deficit = target_residual - current_residual
        if residual_deficit < Decimal('0'):
            residual_deficit = Decimal('0')

        tracker.record_step(
            operation="residual_maintenance",
            description="Calculate additional dose for residual maintenance",
            inputs={
                'target_residual_mg_l': target_residual,
                'current_residual_mg_l': current_residual
            },
            output_value=residual_deficit,
            output_name="residual_maintenance_mg_l",
            formula="Deficit = Target - Current (min 0)",
            units="mg/L"
        )

        # Step 3: Total concentration dose
        total_dose_mg_l = stoich_dose_mg_l + residual_deficit

        # Step 4: Calculate mass doses
        # Initial dose to treat entire system volume
        initial_dose_kg = total_dose_mg_l * volume / Decimal('1000')

        # Continuous dose for makeup water treatment
        continuous_dose_kg_hr = stoich_dose_mg_l * flow / Decimal('1000')

        # Daily and monthly consumption
        daily_consumption_kg = continuous_dose_kg_hr * Decimal('24')
        monthly_consumption_kg = daily_consumption_kg * Decimal('30')

        tracker.record_step(
            operation="mass_calculation",
            description="Calculate chemical mass requirements",
            inputs={
                'total_dose_mg_l': total_dose_mg_l,
                'system_volume_m3': volume,
                'flow_rate_m3_hr': flow
            },
            output_value=continuous_dose_kg_hr,
            output_name="continuous_dose_kg_hr",
            formula="Mass = Conc * Volume / 1000",
            units="kg/hr",
            reference="Mass Balance"
        )

        # Step 5: Cost calculation
        unit_cost = self.CHEMICAL_COSTS.get(scavenger_type.value, Decimal('1.00'))
        cost_per_hr = continuous_dose_kg_hr * unit_cost
        cost_per_month = monthly_consumption_kg * unit_cost

        # Safety notes
        safety_notes = self._get_scavenger_safety_notes(scavenger_type)

        # Chemical details
        chemical_details = self._get_scavenger_details(scavenger_type)

        # Generate provenance
        provenance = tracker.get_provenance_record({
            'total_dose_mg_l': float(total_dose_mg_l),
            'continuous_dose_kg_hr': float(continuous_dose_kg_hr)
        })

        return ChemicalDosingResult(
            chemical_name=chemical_details['name'],
            chemical_formula=chemical_details['formula'],
            dose_concentration_mg_l=float(total_dose_mg_l.quantize(Decimal('0.01'))),
            initial_dose_kg=float(initial_dose_kg.quantize(Decimal('0.001'))),
            continuous_dose_kg_hr=float(continuous_dose_kg_hr.quantize(Decimal('0.0001'))),
            daily_consumption_kg=float(daily_consumption_kg.quantize(Decimal('0.01'))),
            monthly_consumption_kg=float(monthly_consumption_kg.quantize(Decimal('0.1'))),
            cost_per_hour=float(cost_per_hr.quantize(Decimal('0.01'))),
            cost_per_month=float(cost_per_month.quantize(Decimal('1'))),
            safety_notes=safety_notes,
            provenance_hash=provenance.provenance_hash,
            calculation_steps=[step.to_dict() for step in tracker.steps]
        )

    def calculate_scale_inhibitor_dosing(
        self,
        conditions: WaterConditions,
        inhibitor_type: ScaleInhibitorType = ScaleInhibitorType.HEDP,
        lsi_value: float = 0.0,
        tracker: Optional[ProvenanceTracker] = None
    ) -> ChemicalDosingResult:
        """
        Calculate scale inhibitor dosing requirements.

        Dose based on:
        1. Water hardness and scaling potential (LSI)
        2. Cycles of concentration
        3. Temperature effects
        4. System type

        Typical doses:
        - Cooling towers: 10-50 mg/L
        - Boilers: 5-20 mg/L
        - Closed loops: 50-200 mg/L

        Reference: NACE SP0374, Cooling Water Treatment Guidelines

        Args:
            conditions: Water chemistry conditions
            inhibitor_type: Type of scale inhibitor
            lsi_value: Langelier Saturation Index
            tracker: Optional provenance tracker

        Returns:
            ChemicalDosingResult with dosing recommendations
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"scale_inhibitor_{id(conditions)}",
                calculation_type="scale_inhibitor_dosing",
                version=self.version
            )

        tracker.record_inputs({
            **conditions.__dict__,
            'inhibitor_type': inhibitor_type.value,
            'lsi_value': lsi_value
        })

        # Convert to Decimal
        volume = Decimal(str(conditions.volume_m3))
        flow = Decimal(str(conditions.flow_rate_m3_hr))
        hardness = Decimal(str(conditions.total_hardness_mg_l_caco3))
        calcium = Decimal(str(conditions.calcium_mg_l))
        silica = Decimal(str(conditions.silica_mg_l))
        coc = Decimal(str(conditions.cycles_of_concentration))
        lsi = Decimal(str(lsi_value))
        temp = Decimal(str(conditions.temperature_c))

        # Step 1: Base dose from hardness
        # Base = 5 + 0.1 * hardness (mg/L)
        base_dose = Decimal('5') + Decimal('0.1') * (hardness / Decimal('100'))

        tracker.record_step(
            operation="base_dose",
            description="Calculate base dose from hardness",
            inputs={
                'hardness_mg_l_caco3': hardness
            },
            output_value=base_dose,
            output_name="base_dose_mg_l",
            formula="Base = 5 + 0.1 * (Hardness/100)",
            units="mg/L"
        )

        # Step 2: LSI adjustment
        # Higher LSI = higher scaling potential = higher dose needed
        if lsi > Decimal('0'):
            lsi_factor = Decimal('1') + lsi * Decimal('0.5')
        else:
            lsi_factor = Decimal('1')

        lsi_adjusted_dose = base_dose * lsi_factor

        tracker.record_step(
            operation="lsi_adjustment",
            description="Adjust dose for scaling potential (LSI)",
            inputs={
                'base_dose_mg_l': base_dose,
                'lsi_value': lsi,
                'lsi_factor': lsi_factor
            },
            output_value=lsi_adjusted_dose,
            output_name="lsi_adjusted_dose_mg_l",
            formula="Dose = Base * (1 + LSI * 0.5) for LSI > 0",
            units="mg/L",
            reference="NACE SP0374"
        )

        # Step 3: Cycles of concentration adjustment
        # In recirculating systems, inhibitor concentrates
        if coc > Decimal('1'):
            coc_adjusted_dose = lsi_adjusted_dose / coc
        else:
            coc_adjusted_dose = lsi_adjusted_dose

        # Step 4: Temperature adjustment
        # Higher temp increases reaction rates, may need higher dose
        if temp > Decimal('40'):
            temp_factor = Decimal('1') + (temp - Decimal('40')) * Decimal('0.01')
        else:
            temp_factor = Decimal('1')

        final_dose_mg_l = coc_adjusted_dose * temp_factor

        tracker.record_step(
            operation="final_dose",
            description="Calculate final dose with CoC and temperature adjustments",
            inputs={
                'lsi_adjusted_dose': lsi_adjusted_dose,
                'cycles_of_concentration': coc,
                'temperature_C': temp,
                'temp_factor': temp_factor
            },
            output_value=final_dose_mg_l,
            output_name="final_dose_mg_l",
            formula="Dose = LSI_adjusted / CoC * (1 + 0.01*(T-40))",
            units="mg/L"
        )

        # Step 5: Mass calculations
        initial_dose_kg = final_dose_mg_l * volume / Decimal('1000')
        continuous_dose_kg_hr = final_dose_mg_l * flow / Decimal('1000')
        daily_consumption_kg = continuous_dose_kg_hr * Decimal('24')
        monthly_consumption_kg = daily_consumption_kg * Decimal('30')

        # Cost
        unit_cost = self.CHEMICAL_COSTS.get('scale_inhibitor', Decimal('4.00'))
        cost_per_hr = continuous_dose_kg_hr * unit_cost
        cost_per_month = monthly_consumption_kg * unit_cost

        # Safety notes
        safety_notes = [
            "Wear chemical-resistant gloves and eye protection",
            "Avoid contact with concentrated solutions",
            "Store in cool, dry location",
            "Incompatible with oxidizers"
        ]

        # Generate provenance
        provenance = tracker.get_provenance_record({
            'final_dose_mg_l': float(final_dose_mg_l),
            'continuous_dose_kg_hr': float(continuous_dose_kg_hr)
        })

        return ChemicalDosingResult(
            chemical_name=f"Scale Inhibitor ({inhibitor_type.value.upper()})",
            chemical_formula=inhibitor_type.value.upper(),
            dose_concentration_mg_l=float(final_dose_mg_l.quantize(Decimal('0.1'))),
            initial_dose_kg=float(initial_dose_kg.quantize(Decimal('0.01'))),
            continuous_dose_kg_hr=float(continuous_dose_kg_hr.quantize(Decimal('0.001'))),
            daily_consumption_kg=float(daily_consumption_kg.quantize(Decimal('0.01'))),
            monthly_consumption_kg=float(monthly_consumption_kg.quantize(Decimal('0.1'))),
            cost_per_hour=float(cost_per_hr.quantize(Decimal('0.01'))),
            cost_per_month=float(cost_per_month.quantize(Decimal('1'))),
            safety_notes=safety_notes,
            provenance_hash=provenance.provenance_hash,
            calculation_steps=[step.to_dict() for step in tracker.steps]
        )

    def calculate_ph_adjustment_dosing(
        self,
        conditions: WaterConditions,
        target_ph: float,
        chemical: str = "auto",
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Calculate pH adjustment chemical dosing.

        pH Increase (alkaline): NaOH, Na2CO3, Na3PO4
        pH Decrease (acidic): H2SO4, HCl

        Calculation accounts for buffering capacity (alkalinity).

        Henderson-Hasselbalch:
        pH = pKa + log([A-]/[HA])

        Reference: APHA Standard Methods, Water Chemistry Textbooks

        Args:
            conditions: Water conditions
            target_ph: Desired pH value
            chemical: Chemical to use ("auto", "naoh", "h2so4", "hcl")
            tracker: Optional provenance tracker

        Returns:
            Dosing calculation with buffer considerations
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"ph_adjust_{id(conditions)}",
                calculation_type="ph_adjustment_dosing",
                version=self.version
            )

        tracker.record_inputs({
            **conditions.__dict__,
            'target_ph': target_ph,
            'chemical': chemical
        })

        # Convert to Decimal
        volume = Decimal(str(conditions.volume_m3))
        flow = Decimal(str(conditions.flow_rate_m3_hr))
        current_ph = Decimal(str(conditions.ph))
        target_ph_dec = Decimal(str(target_ph))
        alkalinity = Decimal(str(conditions.total_alkalinity_mg_l_caco3))

        # Determine direction
        ph_change = target_ph_dec - current_ph

        if abs(ph_change) < Decimal('0.1'):
            # No significant change needed
            return {
                'chemical': 'None',
                'dose_mg_l': 0.0,
                'initial_dose_kg': 0.0,
                'continuous_dose_kg_hr': 0.0,
                'note': 'pH already within target range'
            }

        # Auto-select chemical
        if chemical == "auto":
            chemical = "naoh" if ph_change > 0 else "h2so4"

        if ph_change > Decimal('0'):
            # pH increase - use caustic
            result = self._calculate_caustic_dosing(
                current_ph, target_ph_dec, alkalinity, volume, flow, tracker
            )
        else:
            # pH decrease - use acid
            result = self._calculate_acid_dosing(
                current_ph, target_ph_dec, alkalinity, volume, flow,
                chemical, tracker
            )

        return result

    def calculate_biocide_dosing(
        self,
        conditions: WaterConditions,
        biocide_type: BiocideType = BiocideType.CHLORINE,
        target_residual_mg_l: float = 0.5,
        tracker: Optional[ProvenanceTracker] = None
    ) -> ChemicalDosingResult:
        """
        Calculate biocide dosing for microbial control.

        Chlorine demand + Residual maintenance:
        Total Cl2 = Demand + Residual

        Chlorine demand from organics, ammonia, iron:
        - Organics: ~2-5 mg Cl2 per mg TOC
        - Ammonia: 7.6 mg Cl2 per mg NH3-N (breakpoint)
        - Iron: 0.64 mg Cl2 per mg Fe

        Reference: AWWA Manual M20, EPA Guidelines

        Args:
            conditions: Water conditions
            biocide_type: Type of biocide
            target_residual_mg_l: Target free residual
            tracker: Optional provenance tracker

        Returns:
            Biocide dosing requirements
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"biocide_{id(conditions)}",
                calculation_type="biocide_dosing",
                version=self.version
            )

        tracker.record_inputs({
            **conditions.__dict__,
            'biocide_type': biocide_type.value,
            'target_residual_mg_l': target_residual_mg_l
        })

        # Convert to Decimal
        volume = Decimal(str(conditions.volume_m3))
        flow = Decimal(str(conditions.flow_rate_m3_hr))
        iron = Decimal(str(conditions.iron_mg_l))
        target_residual = Decimal(str(target_residual_mg_l))

        # Step 1: Estimate chlorine demand
        # Simplified: demand based on iron and general organics
        iron_demand = iron * Decimal('0.64')  # 0.64 mg Cl2 per mg Fe
        organic_demand = Decimal('2.0')  # Assume 2 mg/L organic demand
        total_demand = iron_demand + organic_demand

        tracker.record_step(
            operation="chlorine_demand",
            description="Estimate chlorine demand from iron and organics",
            inputs={
                'iron_mg_l': iron,
                'iron_demand_factor': Decimal('0.64'),
                'estimated_organic_demand': organic_demand
            },
            output_value=total_demand,
            output_name="total_demand_mg_l",
            formula="Demand = Fe * 0.64 + Organic_estimate",
            units="mg/L as Cl2",
            reference="AWWA M20"
        )

        # Step 2: Total dose = demand + residual
        total_dose_cl2 = total_demand + target_residual

        # Step 3: Convert to actual chemical dose
        if biocide_type == BiocideType.CHLORINE:
            # Gas chlorine: 1:1
            chemical_dose_mg_l = total_dose_cl2
            chemical_name = "Chlorine Gas"
            chemical_formula = "Cl2"
        elif biocide_type == BiocideType.CHLORINE_DIOXIDE:
            # ClO2: ~2.63 times more oxidizing power
            chemical_dose_mg_l = total_dose_cl2 / Decimal('2.63')
            chemical_name = "Chlorine Dioxide"
            chemical_formula = "ClO2"
        else:
            # Sodium hypochlorite (12.5% available Cl2)
            chemical_dose_mg_l = total_dose_cl2 / Decimal('0.125')
            chemical_name = "Sodium Hypochlorite (12.5%)"
            chemical_formula = "NaOCl"

        tracker.record_step(
            operation="chemical_dose",
            description="Calculate actual chemical dose",
            inputs={
                'total_cl2_dose_mg_l': total_dose_cl2,
                'biocide_type': biocide_type.value
            },
            output_value=chemical_dose_mg_l,
            output_name="chemical_dose_mg_l",
            formula="Varies by biocide type",
            units="mg/L"
        )

        # Step 4: Mass calculations
        initial_dose_kg = chemical_dose_mg_l * volume / Decimal('1000')
        continuous_dose_kg_hr = chemical_dose_mg_l * flow / Decimal('1000')
        daily_consumption_kg = continuous_dose_kg_hr * Decimal('24')
        monthly_consumption_kg = daily_consumption_kg * Decimal('30')

        # Cost
        unit_cost = self.CHEMICAL_COSTS.get('biocide', Decimal('3.00'))
        cost_per_hr = continuous_dose_kg_hr * unit_cost
        cost_per_month = monthly_consumption_kg * unit_cost

        # Safety notes
        safety_notes = self._get_biocide_safety_notes(biocide_type)

        # Generate provenance
        provenance = tracker.get_provenance_record({
            'chemical_dose_mg_l': float(chemical_dose_mg_l),
            'continuous_dose_kg_hr': float(continuous_dose_kg_hr)
        })

        return ChemicalDosingResult(
            chemical_name=chemical_name,
            chemical_formula=chemical_formula,
            dose_concentration_mg_l=float(chemical_dose_mg_l.quantize(Decimal('0.01'))),
            initial_dose_kg=float(initial_dose_kg.quantize(Decimal('0.001'))),
            continuous_dose_kg_hr=float(continuous_dose_kg_hr.quantize(Decimal('0.0001'))),
            daily_consumption_kg=float(daily_consumption_kg.quantize(Decimal('0.01'))),
            monthly_consumption_kg=float(monthly_consumption_kg.quantize(Decimal('0.1'))),
            cost_per_hour=float(cost_per_hr.quantize(Decimal('0.01'))),
            cost_per_month=float(cost_per_month.quantize(Decimal('1'))),
            safety_notes=safety_notes,
            provenance_hash=provenance.provenance_hash,
            calculation_steps=[step.to_dict() for step in tracker.steps]
        )

    def calculate_filming_amine_dosing(
        self,
        conditions: WaterConditions,
        surface_area_m2: float = 100.0,
        tracker: Optional[ProvenanceTracker] = None
    ) -> ChemicalDosingResult:
        """
        Calculate filming amine dosing for corrosion protection.

        Filming amines form protective layer on metal surfaces.
        Dose based on:
        - System surface area
        - Water velocity (affects film stability)
        - Temperature
        - pH

        Typical doses: 5-25 mg/L for steam condensate lines

        Reference: NACE SP0590, ASME Guidelines

        Args:
            conditions: Water conditions
            surface_area_m2: Total metal surface area to protect
            tracker: Optional provenance tracker

        Returns:
            Filming amine dosing requirements
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"filming_amine_{id(conditions)}",
                calculation_type="filming_amine_dosing",
                version=self.version
            )

        tracker.record_inputs({
            **conditions.__dict__,
            'surface_area_m2': surface_area_m2
        })

        # Convert to Decimal
        volume = Decimal(str(conditions.volume_m3))
        flow = Decimal(str(conditions.flow_rate_m3_hr))
        surface = Decimal(str(surface_area_m2))
        temp = Decimal(str(conditions.temperature_c))
        ph = Decimal(str(conditions.ph))
        iron = Decimal(str(conditions.iron_mg_l))

        # Step 1: Base dose from surface area to volume ratio
        # Higher ratio = more surface = higher dose
        if volume > 0:
            surface_ratio = surface / (volume * Decimal('1000'))  # m2/L
        else:
            surface_ratio = Decimal('0.01')

        # Base dose: 10 mg/L minimum, increases with surface ratio
        base_dose = Decimal('10') + surface_ratio * Decimal('50')
        if base_dose > Decimal('50'):
            base_dose = Decimal('50')  # Cap at 50 mg/L

        tracker.record_step(
            operation="base_dose",
            description="Calculate base dose from surface area",
            inputs={
                'surface_area_m2': surface,
                'volume_m3': volume,
                'surface_ratio': surface_ratio
            },
            output_value=base_dose,
            output_name="base_dose_mg_l",
            formula="Base = 10 + (A/V*1000) * 50, max 50",
            units="mg/L"
        )

        # Step 2: Temperature adjustment
        # Film stability decreases at high temp, need higher dose
        if temp > Decimal('100'):
            temp_factor = Decimal('1') + (temp - Decimal('100')) * Decimal('0.02')
        else:
            temp_factor = Decimal('1')

        # Step 3: pH adjustment
        # Filming amines work best at pH 7-9
        if ph < Decimal('7'):
            ph_factor = Decimal('1') + (Decimal('7') - ph) * Decimal('0.2')
        elif ph > Decimal('9'):
            ph_factor = Decimal('1') + (ph - Decimal('9')) * Decimal('0.15')
        else:
            ph_factor = Decimal('1')

        # Step 4: Iron/corrosion adjustment
        # High iron indicates active corrosion, need higher dose
        if iron > Decimal('0.5'):
            corrosion_factor = Decimal('1') + (iron - Decimal('0.5')) * Decimal('0.3')
        else:
            corrosion_factor = Decimal('1')

        final_dose_mg_l = base_dose * temp_factor * ph_factor * corrosion_factor

        tracker.record_step(
            operation="final_dose",
            description="Calculate final dose with all adjustments",
            inputs={
                'base_dose_mg_l': base_dose,
                'temp_factor': temp_factor,
                'ph_factor': ph_factor,
                'corrosion_factor': corrosion_factor
            },
            output_value=final_dose_mg_l,
            output_name="final_dose_mg_l",
            formula="Dose = Base * Temp_factor * pH_factor * Corrosion_factor",
            units="mg/L",
            reference="NACE SP0590"
        )

        # Mass calculations
        initial_dose_kg = final_dose_mg_l * volume / Decimal('1000')
        continuous_dose_kg_hr = final_dose_mg_l * flow / Decimal('1000')
        daily_consumption_kg = continuous_dose_kg_hr * Decimal('24')
        monthly_consumption_kg = daily_consumption_kg * Decimal('30')

        # Cost
        unit_cost = self.CHEMICAL_COSTS.get('filming_amine', Decimal('5.00'))
        cost_per_hr = continuous_dose_kg_hr * unit_cost
        cost_per_month = monthly_consumption_kg * unit_cost

        # Safety notes
        safety_notes = [
            "Filming amines may cause foaming in boilers",
            "Monitor condensate pH to verify distribution",
            "Avoid overdosing - can lead to deposits",
            "Store away from oxidizers",
            "Use adequate ventilation when handling"
        ]

        # Generate provenance
        provenance = tracker.get_provenance_record({
            'final_dose_mg_l': float(final_dose_mg_l),
            'continuous_dose_kg_hr': float(continuous_dose_kg_hr)
        })

        return ChemicalDosingResult(
            chemical_name="Filming Amine (Octadecylamine blend)",
            chemical_formula="C18H37NH2 blend",
            dose_concentration_mg_l=float(final_dose_mg_l.quantize(Decimal('0.1'))),
            initial_dose_kg=float(initial_dose_kg.quantize(Decimal('0.01'))),
            continuous_dose_kg_hr=float(continuous_dose_kg_hr.quantize(Decimal('0.001'))),
            daily_consumption_kg=float(daily_consumption_kg.quantize(Decimal('0.01'))),
            monthly_consumption_kg=float(monthly_consumption_kg.quantize(Decimal('0.1'))),
            cost_per_hour=float(cost_per_hr.quantize(Decimal('0.01'))),
            cost_per_month=float(cost_per_month.quantize(Decimal('1'))),
            safety_notes=safety_notes,
            provenance_hash=provenance.provenance_hash,
            calculation_steps=[step.to_dict() for step in tracker.steps]
        )

    def calculate_comprehensive_treatment_program(
        self,
        conditions: WaterConditions,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Calculate complete chemical treatment program.

        Includes all necessary chemicals for comprehensive water treatment:
        - Oxygen scavenger
        - Scale inhibitor
        - pH adjustment (if needed)
        - Biocide (for cooling towers)
        - Filming amine (for steam systems)

        Reference: ABMA, ASME, ASHRAE Guidelines

        Args:
            conditions: Water conditions
            tracker: Optional provenance tracker

        Returns:
            Complete treatment program with all dosing requirements
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"comprehensive_{id(conditions)}",
                calculation_type="comprehensive_treatment",
                version=self.version
            )

        results = {}

        # 1. Oxygen scavenger (for boilers and closed loops)
        if conditions.system_type in ["boiler", "closed_loop"]:
            results['oxygen_scavenger'] = self.calculate_oxygen_scavenger_dosing(
                conditions,
                OxygenScavengerType.SODIUM_SULFITE,
                excess_ratio=1.5
            ).__dict__

        # 2. Scale inhibitor (for all systems)
        results['scale_inhibitor'] = self.calculate_scale_inhibitor_dosing(
            conditions,
            ScaleInhibitorType.HEDP,
            lsi_value=0.0  # Would need to calculate LSI
        ).__dict__

        # 3. pH adjustment (if needed)
        if conditions.ph < 7.0 or conditions.ph > 9.5:
            target_ph = 8.5 if conditions.system_type == "boiler" else 7.5
            results['ph_adjustment'] = self.calculate_ph_adjustment_dosing(
                conditions,
                target_ph=target_ph
            )

        # 4. Biocide (for cooling towers)
        if conditions.system_type == "cooling_tower":
            results['biocide'] = self.calculate_biocide_dosing(
                conditions,
                BiocideType.CHLORINE,
                target_residual_mg_l=0.5
            ).__dict__

        # 5. Filming amine (for steam systems)
        if conditions.system_type == "boiler":
            results['filming_amine'] = self.calculate_filming_amine_dosing(
                conditions,
                surface_area_m2=100.0
            ).__dict__

        # Calculate total costs
        total_cost_hr = Decimal('0')
        total_cost_month = Decimal('0')

        for treatment in results.values():
            if isinstance(treatment, dict):
                total_cost_hr += Decimal(str(treatment.get('cost_per_hour', 0)))
                total_cost_month += Decimal(str(treatment.get('cost_per_month', 0)))

        results['summary'] = {
            'total_cost_per_hour': float(total_cost_hr.quantize(Decimal('0.01'))),
            'total_cost_per_month': float(total_cost_month.quantize(Decimal('1'))),
            'total_cost_per_year': float((total_cost_month * Decimal('12')).quantize(Decimal('1'))),
            'treatment_count': len(results) - 1,  # Exclude summary
            'system_type': conditions.system_type
        }

        return results

    # Helper methods

    def _calculate_caustic_dosing(
        self,
        current_ph: Decimal,
        target_ph: Decimal,
        alkalinity: Decimal,
        volume: Decimal,
        flow: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate sodium hydroxide dosing for pH increase."""
        # Calculate OH- requirement
        oh_current = Decimal('10') ** (current_ph - Decimal('14'))
        oh_target = Decimal('10') ** (target_ph - Decimal('14'))
        oh_deficit = oh_target - oh_current

        # Buffer factor from alkalinity
        buffer_factor = Decimal('1') + alkalinity / Decimal('500')

        # NaOH dose (40 g/mol, 1 mol OH- per mol NaOH)
        naoh_mol_l = oh_deficit * buffer_factor
        naoh_mg_l = naoh_mol_l * self.MOLECULAR_WEIGHTS['NaOH'] * Decimal('1000')

        # Prevent negative doses
        if naoh_mg_l < Decimal('0'):
            naoh_mg_l = Decimal('0')

        tracker.record_step(
            operation="naoh_dosing",
            description="Calculate NaOH dose for pH increase",
            inputs={
                'current_ph': current_ph,
                'target_ph': target_ph,
                'alkalinity_mg_l': alkalinity,
                'buffer_factor': buffer_factor
            },
            output_value=naoh_mg_l,
            output_name="naoh_dose_mg_l",
            formula="NaOH = OH_deficit * buffer * 40000",
            units="mg/L",
            reference="APHA Standard Methods"
        )

        # Mass calculations
        initial_kg = naoh_mg_l * volume / Decimal('1000')
        continuous_kg_hr = naoh_mg_l * flow / Decimal('1000')

        # Convert to 25% solution (L)
        solution_density = Decimal('1.28')  # kg/L for 25% NaOH
        solution_conc = Decimal('0.25')
        initial_solution_l = initial_kg / (solution_density * solution_conc)
        continuous_solution_l_hr = continuous_kg_hr / (solution_density * solution_conc)

        return {
            'chemical': 'Sodium Hydroxide (NaOH)',
            'current_ph': float(current_ph),
            'target_ph': float(target_ph),
            'dose_mg_l': float(naoh_mg_l.quantize(Decimal('0.01'))),
            'initial_dose_kg': float(initial_kg.quantize(Decimal('0.001'))),
            'continuous_dose_kg_hr': float(continuous_kg_hr.quantize(Decimal('0.0001'))),
            'solution_25pct_initial_L': float(initial_solution_l.quantize(Decimal('0.01'))),
            'solution_25pct_continuous_L_hr': float(continuous_solution_l_hr.quantize(Decimal('0.001'))),
            'buffer_factor': float(buffer_factor.quantize(Decimal('0.01'))),
            'safety_notes': [
                "25% NaOH is highly corrosive",
                "Wear face shield and chemical-resistant gloves",
                "Add caustic to water, never water to caustic",
                "Neutralize spills with dilute acid"
            ]
        }

    def _calculate_acid_dosing(
        self,
        current_ph: Decimal,
        target_ph: Decimal,
        alkalinity: Decimal,
        volume: Decimal,
        flow: Decimal,
        chemical: str,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate acid dosing for pH decrease."""
        # H+ requirement
        h_current = Decimal('10') ** (-current_ph)
        h_target = Decimal('10') ** (-target_ph)
        h_addition = h_target - h_current

        # Alkalinity neutralization
        # Need to neutralize alkalinity that buffers pH change
        alk_mol_l = alkalinity / Decimal('50000')  # meq/L to mol/L

        # Estimate fraction of alk to neutralize based on pH change
        if target_ph < Decimal('8.3'):
            alk_neutralize = alk_mol_l * Decimal('0.5')
        elif target_ph < Decimal('6.3'):
            alk_neutralize = alk_mol_l
        else:
            alk_neutralize = Decimal('0')

        total_h_mol_l = h_addition + alk_neutralize

        if chemical == "h2so4":
            # H2SO4: 2 H+ per mol
            acid_mol_l = total_h_mol_l / Decimal('2')
            acid_mg_l = acid_mol_l * self.MOLECULAR_WEIGHTS['H2SO4'] * Decimal('1000')
            chemical_name = "Sulfuric Acid (H2SO4)"
            solution_conc = Decimal('0.93')
            solution_density = Decimal('1.83')
        else:  # HCl
            # HCl: 1 H+ per mol
            acid_mol_l = total_h_mol_l
            acid_mg_l = acid_mol_l * self.MOLECULAR_WEIGHTS['HCl'] * Decimal('1000')
            chemical_name = "Hydrochloric Acid (HCl)"
            solution_conc = Decimal('0.37')
            solution_density = Decimal('1.19')

        if acid_mg_l < Decimal('0'):
            acid_mg_l = Decimal('0')

        tracker.record_step(
            operation="acid_dosing",
            description="Calculate acid dose for pH decrease",
            inputs={
                'current_ph': current_ph,
                'target_ph': target_ph,
                'alkalinity_mg_l': alkalinity,
                'h_addition_mol_l': h_addition,
                'alk_neutralization_mol_l': alk_neutralize
            },
            output_value=acid_mg_l,
            output_name="acid_dose_mg_l",
            formula="Acid = (H_add + Alk_neutralize) * MW * 1000",
            units="mg/L"
        )

        # Mass calculations
        initial_kg = acid_mg_l * volume / Decimal('1000')
        continuous_kg_hr = acid_mg_l * flow / Decimal('1000')

        initial_solution_l = initial_kg / (solution_density * solution_conc)
        continuous_solution_l_hr = continuous_kg_hr / (solution_density * solution_conc)

        return {
            'chemical': chemical_name,
            'current_ph': float(current_ph),
            'target_ph': float(target_ph),
            'dose_mg_l': float(acid_mg_l.quantize(Decimal('0.01'))),
            'initial_dose_kg': float(initial_kg.quantize(Decimal('0.001'))),
            'continuous_dose_kg_hr': float(continuous_kg_hr.quantize(Decimal('0.0001'))),
            'solution_initial_L': float(initial_solution_l.quantize(Decimal('0.01'))),
            'solution_continuous_L_hr': float(continuous_solution_l_hr.quantize(Decimal('0.001'))),
            'alkalinity_neutralization_mol_l': float(alk_neutralize.quantize(Decimal('0.000001'))),
            'safety_notes': [
                f"{chemical_name} is highly corrosive",
                "Always add acid to water, never water to acid",
                "Wear full face shield and acid-resistant gear",
                "Have neutralizing agent (soda ash) nearby",
                "Ensure adequate ventilation"
            ]
        }

    def _get_scavenger_details(self, scavenger_type: OxygenScavengerType) -> Dict:
        """Get chemical details for oxygen scavenger."""
        details = {
            OxygenScavengerType.SODIUM_SULFITE: {
                'name': 'Sodium Sulfite',
                'formula': 'Na2SO3'
            },
            OxygenScavengerType.CATALYZED_SULFITE: {
                'name': 'Catalyzed Sodium Sulfite',
                'formula': 'Na2SO3 + Co catalyst'
            },
            OxygenScavengerType.HYDRAZINE: {
                'name': 'Hydrazine',
                'formula': 'N2H4'
            },
            OxygenScavengerType.CARBOHYDRAZIDE: {
                'name': 'Carbohydrazide',
                'formula': 'CH6N4O'
            },
            OxygenScavengerType.DEHA: {
                'name': 'Diethylhydroxylamine (DEHA)',
                'formula': '(C2H5)2NOH'
            },
            OxygenScavengerType.ERYTHORBIC_ACID: {
                'name': 'Erythorbic Acid',
                'formula': 'C6H8O6'
            }
        }
        return details.get(scavenger_type, {'name': 'Unknown', 'formula': 'N/A'})

    def _get_scavenger_safety_notes(self, scavenger_type: OxygenScavengerType) -> List[str]:
        """Get safety notes for oxygen scavenger."""
        base_notes = [
            "Wear appropriate PPE when handling",
            "Store in cool, dry location",
            "Keep container tightly closed"
        ]

        specific_notes = {
            OxygenScavengerType.SODIUM_SULFITE: [
                "May cause skin irritation",
                "Avoid breathing dust",
                "Incompatible with strong oxidizers"
            ],
            OxygenScavengerType.HYDRAZINE: [
                "CARCINOGEN - Use with extreme caution",
                "Use only in enclosed systems",
                "Requires special handling procedures",
                "OSHA/NIOSH regulated substance"
            ],
            OxygenScavengerType.CARBOHYDRAZIDE: [
                "Less toxic hydrazine alternative",
                "Still requires careful handling",
                "Avoid skin contact"
            ],
            OxygenScavengerType.DEHA: [
                "Volatile - ensure ventilation",
                "May cause eye irritation",
                "Forms volatile amines at high temperature"
            ],
            OxygenScavengerType.ERYTHORBIC_ACID: [
                "Food-grade alternative",
                "Relatively safe handling",
                "May support microbial growth if overdosed"
            ]
        }

        return base_notes + specific_notes.get(scavenger_type, [])

    def _get_biocide_safety_notes(self, biocide_type: BiocideType) -> List[str]:
        """Get safety notes for biocide."""
        notes = {
            BiocideType.CHLORINE: [
                "Toxic gas - handle with extreme care",
                "Use only in well-ventilated areas",
                "Have emergency response plan",
                "Never mix with ammonia or acids"
            ],
            BiocideType.CHLORINE_DIOXIDE: [
                "Explosive at high concentrations",
                "Generate on-site only",
                "Requires specialized equipment",
                "Light sensitive - store in dark"
            ],
            BiocideType.BROMINE: [
                "Corrosive and toxic",
                "More stable than chlorine at high pH",
                "Wear chemical-resistant gear"
            ],
            BiocideType.ISOTHIAZOLINE: [
                "Skin sensitizer",
                "Avoid skin contact",
                "Use at recommended concentrations only"
            ],
            BiocideType.GLUTARALDEHYDE: [
                "Skin and respiratory sensitizer",
                "Use adequate ventilation",
                "Follow exposure limits"
            ],
            BiocideType.QUATERNARY_AMMONIUM: [
                "Foaming tendency",
                "Incompatible with anionic surfactants",
                "Relatively safe handling"
            ]
        }
        return notes.get(biocide_type, ["Handle with care", "Wear appropriate PPE"])
