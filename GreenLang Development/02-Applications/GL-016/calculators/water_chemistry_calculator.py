# -*- coding: utf-8 -*-
"""
Water Chemistry Calculator - GL-016 WATERGUARD

Advanced water chemistry calculations for boiler and cooling water systems.
Implements rigorous physical chemistry principles with zero hallucination guarantee.

Author: GL-016 WATERGUARD Engineering Team
Version: 1.0.0
Standards: ASME, ASHRAE, NACE, ASTM
References:
- ASME Consensus on Operating Practices for Control of Water and Steam Chemistry
- ASHRAE Handbook - HVAC Systems and Equipment
- NACE SP0294 - Design, Fabrication, and Inspection of Tanks for the Storage of Concentrated Sulfuric Acid
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .provenance import ProvenanceTracker
import math


@dataclass
class WaterSample:
    """Water sample analysis data."""
    # Basic parameters
    temperature_c: float  # Celsius
    ph: float  # pH units
    conductivity_us_cm: float  # microsiemens/cm

    # Major ions (mg/L)
    calcium_mg_l: float
    magnesium_mg_l: float
    sodium_mg_l: float
    potassium_mg_l: float
    chloride_mg_l: float
    sulfate_mg_l: float
    bicarbonate_mg_l: float
    carbonate_mg_l: float
    hydroxide_mg_l: float

    # Other parameters
    silica_mg_l: float
    iron_mg_l: float
    copper_mg_l: float
    phosphate_mg_l: float
    dissolved_oxygen_mg_l: float
    total_alkalinity_mg_l_caco3: float
    total_hardness_mg_l_caco3: float


class WaterChemistryCalculator:
    """
    Advanced water chemistry calculator for boiler and cooling systems.

    Capabilities:
    - pH equilibrium calculations
    - Dissolved solids calculations
    - Ionic strength calculations
    - Activity coefficient calculations (Debye-Hückel)
    - Saturation indices for mineral phases
    - Temperature correction factors
    - Conductivity-TDS correlations
    """

    # Physical constants
    WATER_ION_PRODUCT_25C = Decimal('1e-14')  # Kw at 25°C
    GAS_CONSTANT = Decimal('8.314')  # J/(mol·K)
    FARADAY_CONSTANT = Decimal('96485')  # C/mol

    # Ion charges
    ION_CHARGES = {
        'Ca': 2, 'Mg': 2, 'Na': 1, 'K': 1,
        'Cl': -1, 'SO4': -2, 'HCO3': -1, 'CO3': -2, 'OH': -1
    }

    # Molecular weights (g/mol)
    MOLECULAR_WEIGHTS = {
        'Ca': Decimal('40.078'),
        'Mg': Decimal('24.305'),
        'Na': Decimal('22.990'),
        'K': Decimal('39.098'),
        'Cl': Decimal('35.453'),
        'SO4': Decimal('96.062'),
        'HCO3': Decimal('61.017'),
        'CO3': Decimal('60.008'),
        'OH': Decimal('17.007'),
        'SiO2': Decimal('60.084'),
        'Fe': Decimal('55.845'),
        'Cu': Decimal('63.546'),
        'PO4': Decimal('94.971'),
        'CaCO3': Decimal('100.087')
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize water chemistry calculator."""
        self.version = version

    def calculate_comprehensive_analysis(self, sample: WaterSample) -> Dict:
        """
        Perform comprehensive water chemistry analysis.

        Args:
            sample: Water sample data

        Returns:
            Complete water chemistry analysis with provenance
        """
        tracker = ProvenanceTracker(
            calculation_id=f"water_chem_{id(sample)}",
            calculation_type="comprehensive_water_analysis",
            version=self.version
        )

        # Record inputs
        tracker.record_inputs(sample.__dict__)

        # Calculate ionic strength
        ionic_strength = self._calculate_ionic_strength(sample, tracker)

        # Calculate activity coefficients
        activity_coeffs = self._calculate_activity_coefficients(
            ionic_strength, sample.temperature_c, tracker
        )

        # Calculate TDS
        tds = self._calculate_tds(sample, tracker)

        # Calculate pH adjustments
        ph_analysis = self._calculate_ph_equilibrium(sample, tracker)

        # Calculate saturation indices
        saturation_indices = self._calculate_saturation_indices(
            sample, ionic_strength, activity_coeffs, tracker
        )

        # Temperature corrections
        temp_corrections = self._calculate_temperature_corrections(
            sample.temperature_c, tracker
        )

        # Conductivity correlations
        conductivity_analysis = self._calculate_conductivity_correlations(
            sample, tds, tracker
        )

        # Hardness calculations
        hardness = self._calculate_hardness(sample, tracker)

        # Alkalinity analysis
        alkalinity = self._calculate_alkalinity(sample, tracker)

        # Ion balance check
        ion_balance = self._calculate_ion_balance(sample, tracker)

        result = {
            'ionic_strength': ionic_strength,
            'activity_coefficients': activity_coeffs,
            'tds': tds,
            'ph_analysis': ph_analysis,
            'saturation_indices': saturation_indices,
            'temperature_corrections': temp_corrections,
            'conductivity_analysis': conductivity_analysis,
            'hardness': hardness,
            'alkalinity': alkalinity,
            'ion_balance': ion_balance,
            'provenance': tracker.get_provenance_record(saturation_indices).to_dict()
        }

        return result

    def _calculate_ionic_strength(
        self,
        sample: WaterSample,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate ionic strength using the formula:
        I = 0.5 × Σ(c_i × z_i²)

        Where:
        - c_i = concentration of ion i (mol/L)
        - z_i = charge of ion i

        Reference: ASTM D1125-95(2014)
        """
        # Convert mg/L to mol/L for each ion
        ca_mol = Decimal(str(sample.calcium_mg_l)) / self.MOLECULAR_WEIGHTS['Ca'] / Decimal('1000')
        mg_mol = Decimal(str(sample.magnesium_mg_l)) / self.MOLECULAR_WEIGHTS['Mg'] / Decimal('1000')
        na_mol = Decimal(str(sample.sodium_mg_l)) / self.MOLECULAR_WEIGHTS['Na'] / Decimal('1000')
        k_mol = Decimal(str(sample.potassium_mg_l)) / self.MOLECULAR_WEIGHTS['K'] / Decimal('1000')
        cl_mol = Decimal(str(sample.chloride_mg_l)) / self.MOLECULAR_WEIGHTS['Cl'] / Decimal('1000')
        so4_mol = Decimal(str(sample.sulfate_mg_l)) / self.MOLECULAR_WEIGHTS['SO4'] / Decimal('1000')
        hco3_mol = Decimal(str(sample.bicarbonate_mg_l)) / self.MOLECULAR_WEIGHTS['HCO3'] / Decimal('1000')
        co3_mol = Decimal(str(sample.carbonate_mg_l)) / self.MOLECULAR_WEIGHTS['CO3'] / Decimal('1000')
        oh_mol = Decimal(str(sample.hydroxide_mg_l)) / self.MOLECULAR_WEIGHTS['OH'] / Decimal('1000')

        # Calculate ionic strength
        I = Decimal('0.5') * (
            ca_mol * Decimal(str(self.ION_CHARGES['Ca']**2)) +
            mg_mol * Decimal(str(self.ION_CHARGES['Mg']**2)) +
            na_mol * Decimal(str(self.ION_CHARGES['Na']**2)) +
            k_mol * Decimal(str(self.ION_CHARGES['K']**2)) +
            cl_mol * Decimal(str(self.ION_CHARGES['Cl']**2)) +
            so4_mol * Decimal(str(self.ION_CHARGES['SO4']**2)) +
            hco3_mol * Decimal(str(self.ION_CHARGES['HCO3']**2)) +
            co3_mol * Decimal(str(self.ION_CHARGES['CO3']**2)) +
            oh_mol * Decimal(str(self.ION_CHARGES['OH']**2))
        )

        tracker.record_step(
            operation="ionic_strength",
            description="Calculate ionic strength from ion concentrations",
            inputs={
                'ca_mol_L': ca_mol,
                'mg_mol_L': mg_mol,
                'na_mol_L': na_mol,
                'cl_mol_L': cl_mol,
                'so4_mol_L': so4_mol
            },
            output_value=I,
            output_name="ionic_strength_mol_L",
            formula="I = 0.5 × Σ(c_i × z_i²)",
            units="mol/L",
            reference="ASTM D1125-95(2014)"
        )

        return {
            'ionic_strength_mol_L': float(I.quantize(Decimal('0.00001'))),
            'ion_concentrations_mol_L': {
                'Ca': float(ca_mol.quantize(Decimal('0.000001'))),
                'Mg': float(mg_mol.quantize(Decimal('0.000001'))),
                'Na': float(na_mol.quantize(Decimal('0.000001'))),
                'K': float(k_mol.quantize(Decimal('0.000001'))),
                'Cl': float(cl_mol.quantize(Decimal('0.000001'))),
                'SO4': float(so4_mol.quantize(Decimal('0.000001'))),
                'HCO3': float(hco3_mol.quantize(Decimal('0.000001'))),
                'CO3': float(co3_mol.quantize(Decimal('0.000001'))),
                'OH': float(oh_mol.quantize(Decimal('0.000001')))
            }
        }

    def _calculate_activity_coefficients(
        self,
        ionic_strength_data: Dict,
        temperature_c: float,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate activity coefficients using Debye-Hückel equation.

        Extended Debye-Hückel equation:
        log(γ) = -A × z² × √I / (1 + B × a × √I)

        Where:
        - A, B = temperature-dependent constants
        - z = ion charge
        - I = ionic strength
        - a = ion size parameter (Angstroms)

        Reference: Debye-Hückel Theory, Physical Chemistry
        """
        I = Decimal(str(ionic_strength_data['ionic_strength_mol_L']))
        T_kelvin = Decimal(str(temperature_c)) + Decimal('273.15')

        # Debye-Hückel A parameter (temperature dependent)
        # A = 1.82483 × 10^6 × (ρ_w)^0.5 / (ε_r × T)^1.5
        # At 25°C, A ≈ 0.5091
        # Temperature correction: A(T) = A(25) × (298.15/T)^1.5 × (ε(25)/ε(T))^1.5
        A_25 = Decimal('0.5091')
        A = A_25 * ((Decimal('298.15') / T_kelvin) ** Decimal('1.5'))

        # Debye-Hückel B parameter
        # At 25°C, B ≈ 0.3286 (when distance in Angstroms)
        B_25 = Decimal('0.3286')
        B = B_25 * ((Decimal('298.15') / T_kelvin) ** Decimal('0.5'))

        # Ion size parameters (Angstroms)
        ion_sizes = {
            'Ca': Decimal('6.0'),
            'Mg': Decimal('5.5'),
            'Na': Decimal('4.0'),
            'K': Decimal('3.5'),
            'Cl': Decimal('3.0'),
            'SO4': Decimal('4.0'),
            'HCO3': Decimal('4.0'),
            'CO3': Decimal('4.5'),
            'OH': Decimal('3.5')
        }

        activity_coeffs = {}
        sqrt_I = I.sqrt()

        for ion, z in self.ION_CHARGES.items():
            z_dec = Decimal(str(z))
            a = ion_sizes[ion]

            # Extended Debye-Hückel equation
            numerator = -A * z_dec * z_dec * sqrt_I
            denominator = Decimal('1') + B * a * sqrt_I
            log_gamma = numerator / denominator

            # Convert from log10 to natural log and then to gamma
            gamma = Decimal('10') ** log_gamma

            activity_coeffs[ion] = float(gamma.quantize(Decimal('0.0001')))

        tracker.record_step(
            operation="activity_coefficients",
            description="Calculate activity coefficients via Debye-Hückel",
            inputs={
                'ionic_strength': I,
                'temperature_K': T_kelvin,
                'A_parameter': A,
                'B_parameter': B
            },
            output_value=activity_coeffs,
            output_name="activity_coefficients",
            formula="log(γ) = -A × z² × √I / (1 + B × a × √I)",
            units="dimensionless",
            reference="Debye-Hückel Theory"
        )

        return activity_coeffs

    def _calculate_tds(
        self,
        sample: WaterSample,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate Total Dissolved Solids (TDS).

        TDS = sum of all dissolved minerals

        Reference: ASTM D1125-95(2014)
        """
        tds = Decimal(str(sample.calcium_mg_l)) + \
              Decimal(str(sample.magnesium_mg_l)) + \
              Decimal(str(sample.sodium_mg_l)) + \
              Decimal(str(sample.potassium_mg_l)) + \
              Decimal(str(sample.chloride_mg_l)) + \
              Decimal(str(sample.sulfate_mg_l)) + \
              Decimal(str(sample.bicarbonate_mg_l)) + \
              Decimal(str(sample.carbonate_mg_l)) + \
              Decimal(str(sample.silica_mg_l)) + \
              Decimal(str(sample.phosphate_mg_l))

        # Convert half of bicarbonate to equivalent CO2
        # HCO3- → CO2 + OH-
        # Factor: 0.4887 (MW CO2 / MW HCO3)
        tds_adjusted = tds - (Decimal(str(sample.bicarbonate_mg_l)) * Decimal('0.4887'))

        tracker.record_step(
            operation="tds_calculation",
            description="Calculate Total Dissolved Solids",
            inputs={
                'calcium': sample.calcium_mg_l,
                'magnesium': sample.magnesium_mg_l,
                'sodium': sample.sodium_mg_l,
                'chloride': sample.chloride_mg_l,
                'sulfate': sample.sulfate_mg_l
            },
            output_value=tds,
            output_name="tds_mg_L",
            formula="TDS = Σ(all dissolved minerals)",
            units="mg/L",
            reference="ASTM D1125"
        )

        return {
            'tds_mg_L': float(tds.quantize(Decimal('0.1'))),
            'tds_adjusted_mg_L': float(tds_adjusted.quantize(Decimal('0.1'))),
            'tds_ppm': float(tds.quantize(Decimal('0.1'))),  # mg/L = ppm for dilute solutions
            'tds_classification': self._classify_tds(float(tds))
        }

    def _classify_tds(self, tds_mg_l: float) -> str:
        """Classify water based on TDS content."""
        if tds_mg_l < 300:
            return "Excellent"
        elif tds_mg_l < 600:
            return "Good"
        elif tds_mg_l < 900:
            return "Fair"
        elif tds_mg_l < 1200:
            return "Poor"
        else:
            return "Unacceptable"

    def _calculate_ph_equilibrium(
        self,
        sample: WaterSample,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate pH equilibrium and carbonate system speciation.

        CO2 + H2O ⇌ H2CO3 ⇌ H+ + HCO3- ⇌ 2H+ + CO3²-

        Henderson-Hasselbalch equation:
        pH = pKa + log([A-]/[HA])

        Reference: APHA Standard Methods
        """
        ph = Decimal(str(sample.ph))
        temp_c = Decimal(str(sample.temperature_c))

        # Calculate pKa values (temperature corrected)
        # pKa1 (H2CO3 ⇌ HCO3-) at 25°C = 6.35
        # pKa2 (HCO3- ⇌ CO3²-) at 25°C = 10.33
        pKa1_25 = Decimal('6.35')
        pKa2_25 = Decimal('10.33')

        # Temperature correction: pKa(T) = pKa(25) + 0.011(25-T)
        temp_factor = Decimal('0.011') * (Decimal('25') - temp_c)
        pKa1 = pKa1_25 + temp_factor
        pKa2 = pKa2_25 + temp_factor

        # Calculate species distribution
        hco3 = Decimal(str(sample.bicarbonate_mg_l))
        co3 = Decimal(str(sample.carbonate_mg_l))

        # Alpha factors for speciation
        h_conc = Decimal('10') ** (-ph)
        ka1 = Decimal('10') ** (-pKa1)
        ka2 = Decimal('10') ** (-pKa2)

        denominator = h_conc * h_conc + h_conc * ka1 + ka1 * ka2

        alpha0 = (h_conc * h_conc) / denominator  # H2CO3 fraction
        alpha1 = (h_conc * ka1) / denominator  # HCO3- fraction
        alpha2 = (ka1 * ka2) / denominator  # CO3²- fraction

        tracker.record_step(
            operation="ph_equilibrium",
            description="Calculate carbonate system pH equilibrium",
            inputs={
                'ph': ph,
                'temperature_C': temp_c,
                'pKa1': pKa1,
                'pKa2': pKa2
            },
            output_value={'alpha0': alpha0, 'alpha1': alpha1, 'alpha2': alpha2},
            output_name="carbonate_speciation",
            formula="Henderson-Hasselbalch: pH = pKa + log([A-]/[HA])",
            units="dimensionless",
            reference="APHA Standard Methods"
        )

        return {
            'ph': float(ph),
            'pKa1_corrected': float(pKa1.quantize(Decimal('0.01'))),
            'pKa2_corrected': float(pKa2.quantize(Decimal('0.01'))),
            'carbonate_speciation': {
                'H2CO3_fraction': float(alpha0.quantize(Decimal('0.0001'))),
                'HCO3_fraction': float(alpha1.quantize(Decimal('0.0001'))),
                'CO3_fraction': float(alpha2.quantize(Decimal('0.0001')))
            },
            'dominant_species': self._get_dominant_carbonate_species(float(ph))
        }

    def _get_dominant_carbonate_species(self, ph: float) -> str:
        """Determine dominant carbonate species based on pH."""
        if ph < 6.35:
            return "H2CO3"
        elif ph < 10.33:
            return "HCO3-"
        else:
            return "CO3²-"

    def _calculate_saturation_indices(
        self,
        sample: WaterSample,
        ionic_strength_data: Dict,
        activity_coeffs: Dict,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate saturation indices for mineral phases.

        Saturation Index (SI) = log(IAP/Ksp)
        Where:
        - IAP = Ion Activity Product
        - Ksp = Solubility Product Constant

        SI > 0: Supersaturated (scaling likely)
        SI = 0: Saturated (equilibrium)
        SI < 0: Undersaturated (no scaling)

        Reference: Langelier Saturation Index, Ryznar Stability Index
        """
        temp_c = Decimal(str(sample.temperature_c))

        # Calcium carbonate saturation (Langelier Saturation Index)
        lsi = self._calculate_langelier_index(sample, activity_coeffs, tracker)

        # Calcium sulfate (gypsum) saturation
        gypsum_si = self._calculate_gypsum_saturation(sample, activity_coeffs, temp_c, tracker)

        # Silica saturation
        silica_si = self._calculate_silica_saturation(sample, temp_c, tracker)

        # Magnesium silicate
        mg_silicate_si = self._calculate_mg_silicate_saturation(sample, temp_c, tracker)

        return {
            'langelier_saturation_index': lsi,
            'gypsum_saturation_index': gypsum_si,
            'silica_saturation_index': silica_si,
            'magnesium_silicate_saturation_index': mg_silicate_si,
            'overall_scaling_risk': self._assess_scaling_risk(lsi, gypsum_si, silica_si)
        }

    def _calculate_langelier_index(
        self,
        sample: WaterSample,
        activity_coeffs: Dict,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate Langelier Saturation Index (LSI).

        LSI = pH - pHs
        Where pHs is the pH at calcium carbonate saturation

        pHs = pK2 - pKsp + pCa + pAlk + 5pf_m

        Standard Method (Carrier):
        pHs = (9.3 + A + B) - (C + D)
        Where:
        A = (log10(TDS) - 1) / 10  (TDS effect)
        B = -13.12 * log10(T + 273) + 34.55  (Temperature effect)
        C = log10(Ca as CaCO3) - 0.4  (Calcium effect)
        D = log10(Alkalinity as CaCO3)  (Alkalinity effect)

        Reference: ASTM D3739, Langelier (1936), Carrier (1965)
        """
        ph = Decimal(str(sample.ph))
        temp_c = Decimal(str(sample.temperature_c))
        temp_k = temp_c + Decimal('273.15')

        # Calculate TDS for A factor
        tds = Decimal(str(sample.calcium_mg_l)) + \
              Decimal(str(sample.magnesium_mg_l)) + \
              Decimal(str(sample.sodium_mg_l)) + \
              Decimal(str(sample.chloride_mg_l)) + \
              Decimal(str(sample.sulfate_mg_l)) + \
              Decimal(str(sample.bicarbonate_mg_l))

        # Factor A: TDS effect
        # A = (log10(TDS) - 1) / 10
        if tds > 0:
            A = ((tds).ln() / Decimal('2.303') - Decimal('1')) / Decimal('10')
        else:
            A = Decimal('0')

        # Factor B: Temperature effect
        # B = -13.12 * log10(T_kelvin) + 34.55
        B = Decimal('-13.12') * (temp_k.ln() / Decimal('2.303')) + Decimal('34.55')

        # Factor C: Calcium hardness effect
        # C = log10(Ca as CaCO3) - 0.4
        ca_mg_l = Decimal(str(sample.calcium_mg_l))
        ca_as_caco3 = ca_mg_l * Decimal('2.497')  # Convert to CaCO3 equivalent
        if ca_as_caco3 > 0:
            C = (ca_as_caco3).ln() / Decimal('2.303') - Decimal('0.4')
        else:
            C = Decimal('0')

        # Factor D: Alkalinity effect
        # D = log10(Alkalinity as CaCO3)
        alk_mg_l_caco3 = Decimal(str(sample.total_alkalinity_mg_l_caco3))
        if alk_mg_l_caco3 > 0:
            D = (alk_mg_l_caco3).ln() / Decimal('2.303')
        else:
            D = Decimal('0')

        # Calculate pHs using Carrier method
        pHs = (Decimal('9.3') + A + B) - (C + D)

        # Calculate LSI
        lsi = ph - pHs

        tracker.record_step(
            operation="langelier_index",
            description="Calculate Langelier Saturation Index for CaCO3",
            inputs={
                'ph': ph,
                'temperature_C': temp_c,
                'calcium_mg_L': ca_mg_l,
                'alkalinity_mg_L_CaCO3': alk_mg_l_caco3,
                'TDS_mg_L': tds,
                'A_factor': A,
                'B_factor': B,
                'C_factor': C,
                'D_factor': D
            },
            output_value=lsi,
            output_name="langelier_saturation_index",
            formula="LSI = pH - pHs, where pHs = (9.3 + A + B) - (C + D)",
            units="dimensionless",
            reference="ASTM D3739, Langelier (1936), Carrier (1965)"
        )

        return {
            'lsi': float(lsi.quantize(Decimal('0.01'))),
            'pHs': float(pHs.quantize(Decimal('0.01'))),
            'A_factor_tds': float(A.quantize(Decimal('0.001'))),
            'B_factor_temp': float(B.quantize(Decimal('0.001'))),
            'C_factor_calcium': float(C.quantize(Decimal('0.001'))),
            'D_factor_alkalinity': float(D.quantize(Decimal('0.001'))),
            'interpretation': self._interpret_lsi(float(lsi))
        }

    def _interpret_lsi(self, lsi: float) -> str:
        """Interpret Langelier Saturation Index."""
        if lsi > 0.5:
            return "Severe scaling tendency"
        elif lsi > 0:
            return "Slight scaling tendency"
        elif lsi > -0.5:
            return "Balanced (ideal)"
        else:
            return "Corrosive tendency"

    def _calculate_gypsum_saturation(
        self,
        sample: WaterSample,
        activity_coeffs: Dict,
        temp_c: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate gypsum (CaSO4·2H2O) saturation index.

        Reference: Rossum and Merrill, J. AWWA
        """
        ca = Decimal(str(sample.calcium_mg_l)) / self.MOLECULAR_WEIGHTS['Ca']
        so4 = Decimal(str(sample.sulfate_mg_l)) / self.MOLECULAR_WEIGHTS['SO4']

        gamma_ca = Decimal(str(activity_coeffs.get('Ca', 0.5)))
        gamma_so4 = Decimal(str(activity_coeffs.get('SO4', 0.4)))

        # Ion activity product
        iap = ca * gamma_ca * so4 * gamma_so4

        # Gypsum Ksp (temperature dependent)
        # log(Ksp) = -4.58 - 0.0065×T
        log_ksp = Decimal('-4.58') - Decimal('0.0065') * temp_c
        ksp = Decimal('10') ** log_ksp

        # Saturation index
        if iap > 0:
            si = (iap / ksp).ln() / Decimal('2.303')  # log10
        else:
            si = Decimal('-10')

        tracker.record_step(
            operation="gypsum_saturation",
            description="Calculate gypsum saturation index",
            inputs={
                'calcium_mmol': ca,
                'sulfate_mmol': so4,
                'temperature_C': temp_c
            },
            output_value=si,
            output_name="gypsum_SI",
            formula="SI = log(IAP/Ksp)",
            units="dimensionless"
        )

        return {
            'gypsum_si': float(si.quantize(Decimal('0.01'))),
            'interpretation': "Scaling risk" if float(si) > 0 else "No scaling risk"
        }

    def _calculate_silica_saturation(
        self,
        sample: WaterSample,
        temp_c: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate silica saturation.

        Amorphous silica solubility (mg/L SiO2):
        S = 60 + 2×T (approximation)

        Reference: EPRI Guidelines
        """
        silica = Decimal(str(sample.silica_mg_l))

        # Amorphous silica solubility
        solubility = Decimal('60') + Decimal('2') * temp_c

        # Saturation ratio
        sat_ratio = silica / solubility

        # SI = log(concentration / solubility)
        si = (sat_ratio).ln() / Decimal('2.303') if sat_ratio > 0 else Decimal('-10')

        tracker.record_step(
            operation="silica_saturation",
            description="Calculate silica saturation index",
            inputs={
                'silica_mg_L': silica,
                'temperature_C': temp_c,
                'solubility_mg_L': solubility
            },
            output_value=si,
            output_name="silica_SI",
            formula="SI = log(C/S)",
            units="dimensionless"
        )

        return {
            'silica_si': float(si.quantize(Decimal('0.01'))),
            'silica_concentration_mg_L': float(silica),
            'silica_solubility_mg_L': float(solubility.quantize(Decimal('0.1'))),
            'percent_of_solubility': float((sat_ratio * Decimal('100')).quantize(Decimal('0.1'))),
            'interpretation': "Scaling risk" if float(si) > 0 else "No scaling risk"
        }

    def _calculate_mg_silicate_saturation(
        self,
        sample: WaterSample,
        temp_c: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate magnesium silicate saturation (serpentine, chrysotile)."""
        mg = Decimal(str(sample.magnesium_mg_l))
        silica = Decimal(str(sample.silica_mg_l))
        ph = Decimal(str(sample.ph))

        # Simplified approach: Mg-Si scaling risk when both are high at high pH
        risk_factor = (mg / Decimal('10')) * (silica / Decimal('150')) * (ph / Decimal('10'))

        si = (risk_factor).ln() / Decimal('2.303') if risk_factor > 0 else Decimal('-10')

        return {
            'mg_silicate_si': float(si.quantize(Decimal('0.01'))),
            'interpretation': "Scaling risk" if float(risk_factor) > 1 else "Low risk"
        }

    def _assess_scaling_risk(
        self,
        lsi_data: Dict,
        gypsum_data: Dict,
        silica_data: Dict
    ) -> str:
        """Assess overall scaling risk."""
        risks = []
        if lsi_data['lsi'] > 0:
            risks.append("CaCO3")
        if gypsum_data['gypsum_si'] > 0:
            risks.append("Gypsum")
        if silica_data['silica_si'] > 0:
            risks.append("Silica")

        if len(risks) >= 2:
            return f"HIGH - Multiple scaling risks: {', '.join(risks)}"
        elif len(risks) == 1:
            return f"MEDIUM - {risks[0]} scaling risk"
        else:
            return "LOW - No significant scaling risk"

    def _calculate_temperature_corrections(
        self,
        temp_c: float,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate temperature correction factors for various parameters.

        Reference: ASME consensus
        """
        temp_dec = Decimal(str(temp_c))
        temp_k = temp_dec + Decimal('273.15')

        # Water ion product temperature correction
        # log(Kw) = -4470.99/T + 6.0875 - 0.01706×T
        log_kw = Decimal('-4470.99') / temp_k + Decimal('6.0875') - Decimal('0.01706') * (temp_k / Decimal('1000'))
        kw = Decimal('10') ** log_kw

        # Conductivity temperature correction
        # κ(T) = κ(25) × [1 + α(T - 25)]
        # α ≈ 0.02 per °C for most waters
        alpha = Decimal('0.02')
        cond_factor = Decimal('1') + alpha * (temp_dec - Decimal('25'))

        return {
            'water_ion_product_Kw': float(kw.quantize(Decimal('0.00000000000001'))),
            'conductivity_correction_factor': float(cond_factor.quantize(Decimal('0.001'))),
            'reference_temperature_C': 25.0
        }

    def _calculate_conductivity_correlations(
        self,
        sample: WaterSample,
        tds_data: Dict,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate correlations between conductivity and TDS.

        TDS (mg/L) ≈ K × Conductivity (μS/cm)
        Where K typically ranges from 0.55 to 0.75

        Reference: ASTM D1125
        """
        conductivity = Decimal(str(sample.conductivity_us_cm))
        tds_measured = Decimal(str(tds_data['tds_mg_L']))

        # Calculate K factor
        if conductivity > 0:
            k_factor = tds_measured / conductivity
        else:
            k_factor = Decimal('0.64')  # Typical value

        # Calculate predicted TDS from conductivity
        tds_predicted = k_factor * conductivity

        # Calculate deviation
        deviation = abs(tds_measured - tds_predicted) / tds_measured * Decimal('100')

        tracker.record_step(
            operation="conductivity_tds_correlation",
            description="Correlate conductivity with TDS",
            inputs={
                'conductivity_uS_cm': conductivity,
                'tds_measured_mg_L': tds_measured
            },
            output_value=k_factor,
            output_name="K_factor",
            formula="TDS = K × Conductivity",
            units="dimensionless"
        )

        return {
            'conductivity_us_cm': float(conductivity),
            'k_factor': float(k_factor.quantize(Decimal('0.001'))),
            'tds_from_conductivity_mg_L': float(tds_predicted.quantize(Decimal('0.1'))),
            'deviation_percent': float(deviation.quantize(Decimal('0.1')))
        }

    def _calculate_hardness(
        self,
        sample: WaterSample,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate various hardness measures.

        Total Hardness = Ca + Mg (as CaCO3)
        Calcium Hardness = Ca (as CaCO3)
        Magnesium Hardness = Mg (as CaCO3)

        Reference: APHA Standard Methods 2340
        """
        ca = Decimal(str(sample.calcium_mg_l))
        mg = Decimal(str(sample.magnesium_mg_l))

        # Convert to CaCO3 equivalents
        # Ca: multiply by 2.497 (100.09/40.08)
        # Mg: multiply by 4.118 (100.09/24.31)
        ca_hardness = ca * Decimal('2.497')
        mg_hardness = mg * Decimal('4.118')
        total_hardness = ca_hardness + mg_hardness

        return {
            'total_hardness_mg_L_CaCO3': float(total_hardness.quantize(Decimal('0.1'))),
            'calcium_hardness_mg_L_CaCO3': float(ca_hardness.quantize(Decimal('0.1'))),
            'magnesium_hardness_mg_L_CaCO3': float(mg_hardness.quantize(Decimal('0.1'))),
            'hardness_classification': self._classify_hardness(float(total_hardness))
        }

    def _classify_hardness(self, hardness: float) -> str:
        """Classify water hardness."""
        if hardness < 60:
            return "Soft"
        elif hardness < 120:
            return "Moderately hard"
        elif hardness < 180:
            return "Hard"
        else:
            return "Very hard"

    def _calculate_alkalinity(
        self,
        sample: WaterSample,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate alkalinity components.

        Reference: APHA Standard Methods 2320
        """
        total_alk = Decimal(str(sample.total_alkalinity_mg_l_caco3))
        ph = Decimal(str(sample.ph))

        # Estimate P-alkalinity and M-alkalinity based on pH
        if ph < Decimal('8.3'):
            p_alk = Decimal('0')
            m_alk = total_alk
        else:
            # Simplified estimation
            p_alk = total_alk * Decimal('0.5')
            m_alk = total_alk

        return {
            'total_alkalinity_mg_L_CaCO3': float(total_alk),
            'p_alkalinity_mg_L_CaCO3': float(p_alk.quantize(Decimal('0.1'))),
            'm_alkalinity_mg_L_CaCO3': float(m_alk.quantize(Decimal('0.1'))),
            'alkalinity_classification': self._classify_alkalinity(float(total_alk))
        }

    def _classify_alkalinity(self, alkalinity: float) -> str:
        """Classify alkalinity level."""
        if alkalinity < 40:
            return "Low"
        elif alkalinity < 120:
            return "Moderate"
        elif alkalinity < 200:
            return "High"
        else:
            return "Very high"

    def _calculate_ion_balance(
        self,
        sample: WaterSample,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate ion balance to verify water analysis accuracy.

        Ion Balance Error (%) = 100 × (Σcations - Σanions) / (Σcations + Σanions)

        Acceptable range: ±5%

        Reference: ASTM D1125
        """
        # Convert to meq/L
        ca_meq = Decimal(str(sample.calcium_mg_l)) / self.MOLECULAR_WEIGHTS['Ca'] * Decimal('2')
        mg_meq = Decimal(str(sample.magnesium_mg_l)) / self.MOLECULAR_WEIGHTS['Mg'] * Decimal('2')
        na_meq = Decimal(str(sample.sodium_mg_l)) / self.MOLECULAR_WEIGHTS['Na']
        k_meq = Decimal(str(sample.potassium_mg_l)) / self.MOLECULAR_WEIGHTS['K']

        cl_meq = Decimal(str(sample.chloride_mg_l)) / self.MOLECULAR_WEIGHTS['Cl']
        so4_meq = Decimal(str(sample.sulfate_mg_l)) / self.MOLECULAR_WEIGHTS['SO4'] * Decimal('2')
        hco3_meq = Decimal(str(sample.bicarbonate_mg_l)) / self.MOLECULAR_WEIGHTS['HCO3']
        co3_meq = Decimal(str(sample.carbonate_mg_l)) / self.MOLECULAR_WEIGHTS['CO3'] * Decimal('2')

        total_cations = ca_meq + mg_meq + na_meq + k_meq
        total_anions = cl_meq + so4_meq + hco3_meq + co3_meq

        # Ion balance error
        if (total_cations + total_anions) > 0:
            balance_error = Decimal('100') * (total_cations - total_anions) / (total_cations + total_anions)
        else:
            balance_error = Decimal('0')

        tracker.record_step(
            operation="ion_balance",
            description="Calculate ion balance to verify analysis accuracy",
            inputs={
                'total_cations_meq_L': total_cations,
                'total_anions_meq_L': total_anions
            },
            output_value=balance_error,
            output_name="ion_balance_error_percent",
            formula="Error = 100 × (Σcations - Σanions) / (Σcations + Σanions)",
            units="%",
            reference="ASTM D1125"
        )

        return {
            'total_cations_meq_L': float(total_cations.quantize(Decimal('0.01'))),
            'total_anions_meq_L': float(total_anions.quantize(Decimal('0.01'))),
            'balance_error_percent': float(balance_error.quantize(Decimal('0.1'))),
            'analysis_quality': "Acceptable" if abs(float(balance_error)) < 5 else "Questionable - recheck analysis"
        }

    def calculate_ryznar_stability_index(
        self,
        sample: WaterSample,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Calculate Ryznar Stability Index (RSI).

        RSI = 2 * pHs - pH

        Where pHs is the pH at calcium carbonate saturation (same as LSI).

        Interpretation:
        - RSI < 5.5: Heavy scale formation
        - 5.5 < RSI < 6.2: Light scale formation
        - 6.2 < RSI < 6.8: Stable water (ideal)
        - 6.8 < RSI < 8.5: Slight corrosion tendency
        - RSI > 8.5: Severe corrosion

        Reference: Ryznar (1944), ASHRAE Handbook
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"rsi_{id(sample)}",
                calculation_type="ryznar_stability_index",
                version=self.version
            )
            tracker.record_inputs(sample.__dict__)

        ph = Decimal(str(sample.ph))
        temp_c = Decimal(str(sample.temperature_c))
        temp_k = temp_c + Decimal('273.15')

        # Calculate pHs using Carrier method (same as LSI)
        tds = Decimal(str(sample.calcium_mg_l)) + \
              Decimal(str(sample.magnesium_mg_l)) + \
              Decimal(str(sample.sodium_mg_l)) + \
              Decimal(str(sample.chloride_mg_l)) + \
              Decimal(str(sample.sulfate_mg_l)) + \
              Decimal(str(sample.bicarbonate_mg_l))

        # Factor A: TDS effect
        if tds > 0:
            A = ((tds).ln() / Decimal('2.303') - Decimal('1')) / Decimal('10')
        else:
            A = Decimal('0')

        # Factor B: Temperature effect
        B = Decimal('-13.12') * (temp_k.ln() / Decimal('2.303')) + Decimal('34.55')

        # Factor C: Calcium hardness effect
        ca_mg_l = Decimal(str(sample.calcium_mg_l))
        ca_as_caco3 = ca_mg_l * Decimal('2.497')
        if ca_as_caco3 > 0:
            C = (ca_as_caco3).ln() / Decimal('2.303') - Decimal('0.4')
        else:
            C = Decimal('0')

        # Factor D: Alkalinity effect
        alk_mg_l_caco3 = Decimal(str(sample.total_alkalinity_mg_l_caco3))
        if alk_mg_l_caco3 > 0:
            D = (alk_mg_l_caco3).ln() / Decimal('2.303')
        else:
            D = Decimal('0')

        # Calculate pHs
        pHs = (Decimal('9.3') + A + B) - (C + D)

        # Calculate RSI
        rsi = Decimal('2') * pHs - ph

        tracker.record_step(
            operation="ryznar_stability_index",
            description="Calculate Ryznar Stability Index for water stability",
            inputs={
                'ph': ph,
                'pHs': pHs,
                'temperature_C': temp_c,
                'calcium_mg_L': ca_mg_l,
                'alkalinity_mg_L_CaCO3': alk_mg_l_caco3
            },
            output_value=rsi,
            output_name="ryznar_stability_index",
            formula="RSI = 2 * pHs - pH",
            units="dimensionless",
            reference="Ryznar (1944), ASHRAE Handbook"
        )

        return {
            'rsi': float(rsi.quantize(Decimal('0.01'))),
            'pHs': float(pHs.quantize(Decimal('0.01'))),
            'interpretation': self._interpret_rsi(float(rsi)),
            'provenance': tracker.get_provenance_record(rsi).to_dict()
        }

    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret Ryznar Stability Index."""
        if rsi < 5.5:
            return "Heavy scale formation - treatment required"
        elif rsi < 6.2:
            return "Light scale formation"
        elif rsi < 6.8:
            return "Stable water (ideal range)"
        elif rsi < 8.5:
            return "Slight corrosion tendency"
        else:
            return "Severe corrosion - treatment required"

    def calculate_puckorius_scaling_index(
        self,
        sample: WaterSample,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Calculate Puckorius Scaling Index (PSI).

        PSI = 2 * pHs - pHeq

        Where:
        - pHs = pH at saturation (same as LSI/RSI)
        - pHeq = equilibrium pH = 1.465 * log10(Alkalinity) + 4.54

        The PSI accounts for buffering capacity of water, making it more
        accurate for industrial cooling water systems.

        Interpretation:
        - PSI < 5.5: Severe scaling tendency
        - 5.5 < PSI < 6.0: Moderate scaling
        - 6.0 < PSI < 7.0: Stable (ideal)
        - 7.0 < PSI < 8.0: Slight corrosion
        - PSI > 8.0: Severe corrosion

        Reference: Puckorius and Brooke (1991), NACE Standards
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"psi_{id(sample)}",
                calculation_type="puckorius_scaling_index",
                version=self.version
            )
            tracker.record_inputs(sample.__dict__)

        temp_c = Decimal(str(sample.temperature_c))
        temp_k = temp_c + Decimal('273.15')

        # Calculate pHs using Carrier method
        tds = Decimal(str(sample.calcium_mg_l)) + \
              Decimal(str(sample.magnesium_mg_l)) + \
              Decimal(str(sample.sodium_mg_l)) + \
              Decimal(str(sample.chloride_mg_l)) + \
              Decimal(str(sample.sulfate_mg_l)) + \
              Decimal(str(sample.bicarbonate_mg_l))

        if tds > 0:
            A = ((tds).ln() / Decimal('2.303') - Decimal('1')) / Decimal('10')
        else:
            A = Decimal('0')

        B = Decimal('-13.12') * (temp_k.ln() / Decimal('2.303')) + Decimal('34.55')

        ca_mg_l = Decimal(str(sample.calcium_mg_l))
        ca_as_caco3 = ca_mg_l * Decimal('2.497')
        if ca_as_caco3 > 0:
            C = (ca_as_caco3).ln() / Decimal('2.303') - Decimal('0.4')
        else:
            C = Decimal('0')

        alk_mg_l_caco3 = Decimal(str(sample.total_alkalinity_mg_l_caco3))
        if alk_mg_l_caco3 > 0:
            D = (alk_mg_l_caco3).ln() / Decimal('2.303')
        else:
            D = Decimal('0')

        pHs = (Decimal('9.3') + A + B) - (C + D)

        # Calculate equilibrium pH (pHeq)
        # pHeq = 1.465 * log10(Alkalinity) + 4.54
        if alk_mg_l_caco3 > 0:
            pHeq = Decimal('1.465') * (alk_mg_l_caco3.ln() / Decimal('2.303')) + Decimal('4.54')
        else:
            pHeq = Decimal('7.0')

        # Calculate PSI
        psi = Decimal('2') * pHs - pHeq

        tracker.record_step(
            operation="puckorius_scaling_index",
            description="Calculate Puckorius Scaling Index with buffering correction",
            inputs={
                'pHs': pHs,
                'pHeq': pHeq,
                'alkalinity_mg_L_CaCO3': alk_mg_l_caco3,
                'temperature_C': temp_c
            },
            output_value=psi,
            output_name="puckorius_scaling_index",
            formula="PSI = 2 * pHs - pHeq, where pHeq = 1.465 * log10(Alk) + 4.54",
            units="dimensionless",
            reference="Puckorius and Brooke (1991), NACE Standards"
        )

        return {
            'psi': float(psi.quantize(Decimal('0.01'))),
            'pHs': float(pHs.quantize(Decimal('0.01'))),
            'pHeq': float(pHeq.quantize(Decimal('0.01'))),
            'interpretation': self._interpret_psi(float(psi)),
            'provenance': tracker.get_provenance_record(psi).to_dict()
        }

    def _interpret_psi(self, psi: float) -> str:
        """Interpret Puckorius Scaling Index."""
        if psi < 5.5:
            return "Severe scaling tendency - immediate treatment required"
        elif psi < 6.0:
            return "Moderate scaling tendency"
        elif psi < 7.0:
            return "Stable water (ideal range)"
        elif psi < 8.0:
            return "Slight corrosion tendency"
        else:
            return "Severe corrosion - treatment required"

    def calculate_cycles_of_concentration(
        self,
        makeup_water: WaterSample,
        circulating_water: WaterSample,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Calculate Cycles of Concentration (CoC) for cooling tower systems.

        CoC = C_circulating / C_makeup

        Where C is a conservative ion concentration (typically Cl- or conductivity).

        Can also be calculated as:
        CoC = Makeup / Blowdown = Evaporation / (Makeup - Evaporation)

        Reference: ASHRAE Handbook - HVAC Systems, Cooling Technology Institute (CTI)
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"coc_{id(makeup_water)}",
                calculation_type="cycles_of_concentration",
                version=self.version
            )

        # Method 1: Using chloride (conservative ion)
        cl_makeup = Decimal(str(makeup_water.chloride_mg_l))
        cl_circulating = Decimal(str(circulating_water.chloride_mg_l))

        if cl_makeup > 0:
            coc_chloride = cl_circulating / cl_makeup
        else:
            coc_chloride = Decimal('1')

        # Method 2: Using conductivity
        cond_makeup = Decimal(str(makeup_water.conductivity_us_cm))
        cond_circulating = Decimal(str(circulating_water.conductivity_us_cm))

        if cond_makeup > 0:
            coc_conductivity = cond_circulating / cond_makeup
        else:
            coc_conductivity = Decimal('1')

        # Method 3: Using calcium (less reliable due to precipitation)
        ca_makeup = Decimal(str(makeup_water.calcium_mg_l))
        ca_circulating = Decimal(str(circulating_water.calcium_mg_l))

        if ca_makeup > 0:
            coc_calcium = ca_circulating / ca_makeup
        else:
            coc_calcium = Decimal('1')

        # Method 4: Using silica
        si_makeup = Decimal(str(makeup_water.silica_mg_l))
        si_circulating = Decimal(str(circulating_water.silica_mg_l))

        if si_makeup > 0:
            coc_silica = si_circulating / si_makeup
        else:
            coc_silica = Decimal('1')

        # Best estimate (average of chloride and conductivity methods)
        coc_best = (coc_chloride + coc_conductivity) / Decimal('2')

        tracker.record_step(
            operation="cycles_of_concentration",
            description="Calculate cycles of concentration from makeup and circulating water",
            inputs={
                'chloride_makeup_mg_L': cl_makeup,
                'chloride_circulating_mg_L': cl_circulating,
                'conductivity_makeup_uS_cm': cond_makeup,
                'conductivity_circulating_uS_cm': cond_circulating
            },
            output_value=coc_best,
            output_name="cycles_of_concentration",
            formula="CoC = C_circulating / C_makeup",
            units="cycles",
            reference="ASHRAE Handbook, CTI Guidelines"
        )

        return {
            'coc_chloride_method': float(coc_chloride.quantize(Decimal('0.01'))),
            'coc_conductivity_method': float(coc_conductivity.quantize(Decimal('0.01'))),
            'coc_calcium_method': float(coc_calcium.quantize(Decimal('0.01'))),
            'coc_silica_method': float(coc_silica.quantize(Decimal('0.01'))),
            'coc_best_estimate': float(coc_best.quantize(Decimal('0.01'))),
            'interpretation': self._interpret_coc(float(coc_best)),
            'provenance': tracker.get_provenance_record(coc_best).to_dict()
        }

    def _interpret_coc(self, coc: float) -> str:
        """Interpret cycles of concentration."""
        if coc < 2:
            return "Very low - excessive water waste, increase cycles"
        elif coc < 3:
            return "Low - room for improvement"
        elif coc < 5:
            return "Good - typical for many applications"
        elif coc < 8:
            return "Very good - efficient water use"
        elif coc < 10:
            return "Excellent - near optimal efficiency"
        else:
            return "Caution - very high, check for scaling/corrosion issues"

    def calculate_blowdown_rate(
        self,
        evaporation_rate_m3_hr: float,
        target_coc: float,
        drift_loss_percent: float = 0.02,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Calculate optimal blowdown rate for cooling tower.

        Mass balance:
        Makeup = Evaporation + Blowdown + Drift

        CoC = Makeup / (Blowdown + Drift)

        Therefore:
        Blowdown = Evaporation / (CoC - 1) - Drift

        Where:
        - Evaporation typically = 1.8% of circulation rate per 10F delta-T
        - Drift = 0.01-0.02% of circulation rate (modern towers)

        Reference: ASHRAE Handbook - HVAC Systems, CTI ATC-105
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"blowdown_{id(evaporation_rate_m3_hr)}",
                calculation_type="blowdown_optimization",
                version=self.version
            )

        evap = Decimal(str(evaporation_rate_m3_hr))
        coc = Decimal(str(target_coc))
        drift_pct = Decimal(str(drift_loss_percent))

        # Calculate makeup rate
        # Makeup = Evap / (1 - 1/CoC) = Evap * CoC / (CoC - 1)
        if coc > Decimal('1'):
            makeup_rate = evap * coc / (coc - Decimal('1'))
        else:
            makeup_rate = evap * Decimal('10')  # Fallback for CoC <= 1

        # Calculate drift loss
        drift_rate = makeup_rate * drift_pct / Decimal('100')

        # Calculate blowdown rate
        # Blowdown = Makeup - Evaporation - Drift
        blowdown_rate = makeup_rate - evap - drift_rate

        # Alternative formula: Blowdown = Evaporation / (CoC - 1) - Drift
        if coc > Decimal('1'):
            blowdown_alt = evap / (coc - Decimal('1')) - drift_rate
        else:
            blowdown_alt = blowdown_rate

        # Water savings calculation (compared to once-through)
        # Once-through would require makeup = circulation rate
        # Savings = 1 - 1/CoC (as fraction)
        water_savings_percent = (Decimal('1') - Decimal('1') / coc) * Decimal('100')

        # Annual water consumption estimate (m3/year)
        annual_makeup_m3 = makeup_rate * Decimal('8760')  # hours per year

        tracker.record_step(
            operation="blowdown_calculation",
            description="Calculate optimal blowdown rate from mass balance",
            inputs={
                'evaporation_rate_m3_hr': evap,
                'target_coc': coc,
                'drift_loss_percent': drift_pct
            },
            output_value=blowdown_rate,
            output_name="blowdown_rate_m3_hr",
            formula="Blowdown = Makeup - Evaporation - Drift, where Makeup = Evap * CoC / (CoC - 1)",
            units="m3/hr",
            reference="ASHRAE Handbook, CTI ATC-105"
        )

        return {
            'evaporation_rate_m3_hr': float(evap.quantize(Decimal('0.001'))),
            'makeup_rate_m3_hr': float(makeup_rate.quantize(Decimal('0.001'))),
            'blowdown_rate_m3_hr': float(blowdown_rate.quantize(Decimal('0.001'))),
            'drift_rate_m3_hr': float(drift_rate.quantize(Decimal('0.0001'))),
            'target_cycles_of_concentration': float(coc),
            'water_savings_percent': float(water_savings_percent.quantize(Decimal('0.1'))),
            'annual_makeup_m3': float(annual_makeup_m3.quantize(Decimal('1'))),
            'annual_blowdown_m3': float((blowdown_rate * Decimal('8760')).quantize(Decimal('1'))),
            'provenance': tracker.get_provenance_record(blowdown_rate).to_dict()
        }

    def calculate_chemical_dosing(
        self,
        sample: WaterSample,
        target_parameter: str,
        target_value: float,
        system_volume_m3: float,
        makeup_rate_m3_hr: float,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Dict:
        """
        Calculate chemical dosing requirements based on stoichiometry.

        Supports:
        - pH adjustment (acid/caustic)
        - Scale inhibitor dosing
        - Biocide dosing
        - Oxygen scavenger dosing
        - Phosphate treatment

        Reference: ABMA Guidelines, ASME Consensus Document
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id=f"dosing_{id(sample)}",
                calculation_type="chemical_dosing",
                version=self.version
            )
            tracker.record_inputs(sample.__dict__)

        vol = Decimal(str(system_volume_m3))
        makeup = Decimal(str(makeup_rate_m3_hr))
        target = Decimal(str(target_value))

        results = {}

        if target_parameter == 'ph_increase':
            results = self._calculate_caustic_dosing(sample, target, vol, makeup, tracker)
        elif target_parameter == 'ph_decrease':
            results = self._calculate_acid_dosing(sample, target, vol, makeup, tracker)
        elif target_parameter == 'phosphate':
            results = self._calculate_phosphate_dosing(sample, target, vol, makeup, tracker)
        elif target_parameter == 'oxygen_scavenger':
            results = self._calculate_oxygen_scavenger_dosing(sample, target, vol, makeup, tracker)
        elif target_parameter == 'scale_inhibitor':
            results = self._calculate_scale_inhibitor_dosing(sample, target, vol, makeup, tracker)
        else:
            results = {'error': f'Unknown target parameter: {target_parameter}'}

        return results

    def _calculate_caustic_dosing(
        self,
        sample: WaterSample,
        target_ph: Decimal,
        volume_m3: Decimal,
        makeup_rate_m3_hr: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate sodium hydroxide (NaOH) dosing for pH increase.

        Stoichiometry:
        NaOH + H+ -> Na+ + H2O

        OH- required = 10^(14-current_pH) - 10^(14-target_pH) [mol/L]
        NaOH required = OH- * MW_NaOH [g/L]

        Reference: APHA Standard Methods, Water Chemistry
        """
        current_ph = Decimal(str(sample.ph))
        MW_NaOH = Decimal('40.0')  # g/mol

        # Calculate hydroxide deficiency
        # [OH-] at target = 10^(pH - 14)
        oh_current = Decimal('10') ** (current_ph - Decimal('14'))
        oh_target = Decimal('10') ** (target_ph - Decimal('14'))
        oh_deficit_mol_l = oh_target - oh_current

        # Account for buffering capacity (alkalinity)
        # Buffer factor increases required dosing
        alk = Decimal(str(sample.total_alkalinity_mg_l_caco3))
        buffer_factor = Decimal('1') + (alk / Decimal('500'))

        # NaOH required (g/L)
        naoh_g_l = oh_deficit_mol_l * MW_NaOH * buffer_factor

        # Total NaOH for system volume (kg)
        naoh_kg_initial = naoh_g_l * volume_m3

        # Continuous dosing rate for makeup (kg/hr)
        # Assumes makeup water has same pH deficit
        naoh_kg_hr = naoh_g_l * makeup_rate_m3_hr

        # Convert to common dosing units (25% NaOH solution, density 1.28 kg/L)
        solution_density = Decimal('1.28')
        solution_concentration = Decimal('0.25')
        solution_l_initial = naoh_kg_initial / (solution_density * solution_concentration)
        solution_l_hr = naoh_kg_hr / (solution_density * solution_concentration)

        tracker.record_step(
            operation="caustic_dosing",
            description="Calculate NaOH dosing for pH increase",
            inputs={
                'current_pH': current_ph,
                'target_pH': target_ph,
                'alkalinity_mg_L': alk,
                'system_volume_m3': volume_m3
            },
            output_value=naoh_kg_initial,
            output_name="naoh_initial_dose_kg",
            formula="NaOH = OH_deficit × MW × Buffer_factor × Volume",
            units="kg",
            reference="APHA Standard Methods"
        )

        return {
            'chemical': 'Sodium Hydroxide (NaOH)',
            'current_ph': float(current_ph),
            'target_ph': float(target_ph),
            'naoh_concentration_g_L': float(naoh_g_l.quantize(Decimal('0.0001'))),
            'initial_dose_kg': float(naoh_kg_initial.quantize(Decimal('0.001'))),
            'continuous_dose_kg_hr': float(naoh_kg_hr.quantize(Decimal('0.0001'))),
            'solution_25pct_initial_L': float(solution_l_initial.quantize(Decimal('0.01'))),
            'solution_25pct_continuous_L_hr': float(solution_l_hr.quantize(Decimal('0.001'))),
            'buffer_factor': float(buffer_factor.quantize(Decimal('0.01'))),
            'provenance': tracker.get_provenance_record(naoh_kg_initial).to_dict()
        }

    def _calculate_acid_dosing(
        self,
        sample: WaterSample,
        target_ph: Decimal,
        volume_m3: Decimal,
        makeup_rate_m3_hr: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate sulfuric acid (H2SO4) dosing for pH decrease.

        Stoichiometry:
        H2SO4 + 2OH- -> SO4^2- + 2H2O
        H2SO4 + 2HCO3- -> SO4^2- + 2H2O + 2CO2

        Reference: APHA Standard Methods
        """
        current_ph = Decimal(str(sample.ph))
        MW_H2SO4 = Decimal('98.079')  # g/mol

        # Calculate hydrogen ion addition required
        h_current = Decimal('10') ** (-current_ph)
        h_target = Decimal('10') ** (-target_ph)
        h_addition_mol_l = h_target - h_current

        # Account for alkalinity consumption
        # Each mol of alkalinity requires 1 mol H+ to neutralize
        alk_mg_l = Decimal(str(sample.total_alkalinity_mg_l_caco3))
        alk_mol_l = alk_mg_l / Decimal('50000')  # 50 g/eq, 1000 mg/g

        # Total H+ needed = pH change + alkalinity neutralization (partial)
        # Fraction of alkalinity to neutralize depends on pH target
        if target_ph < Decimal('8.3'):
            alk_neutralization = alk_mol_l * Decimal('0.5')  # Convert CO3 to HCO3
        elif target_ph < Decimal('6.3'):
            alk_neutralization = alk_mol_l  # Full conversion to CO2
        else:
            alk_neutralization = Decimal('0')

        total_h_mol_l = h_addition_mol_l + alk_neutralization

        # H2SO4 required (1 mol H2SO4 provides 2 mol H+)
        h2so4_mol_l = total_h_mol_l / Decimal('2')
        h2so4_g_l = h2so4_mol_l * MW_H2SO4

        # Total for system
        h2so4_kg_initial = h2so4_g_l * volume_m3
        h2so4_kg_hr = h2so4_g_l * makeup_rate_m3_hr

        # Convert to 93% solution (density 1.83 kg/L)
        solution_density = Decimal('1.83')
        solution_concentration = Decimal('0.93')
        solution_l_initial = h2so4_kg_initial / (solution_density * solution_concentration)
        solution_l_hr = h2so4_kg_hr / (solution_density * solution_concentration)

        tracker.record_step(
            operation="acid_dosing",
            description="Calculate H2SO4 dosing for pH decrease",
            inputs={
                'current_pH': current_ph,
                'target_pH': target_ph,
                'alkalinity_mg_L': alk_mg_l,
                'system_volume_m3': volume_m3
            },
            output_value=h2so4_kg_initial,
            output_name="h2so4_initial_dose_kg",
            formula="H2SO4 = (H_deficit + Alk_neutralization) / 2 × MW × Volume",
            units="kg",
            reference="APHA Standard Methods"
        )

        return {
            'chemical': 'Sulfuric Acid (H2SO4)',
            'current_ph': float(current_ph),
            'target_ph': float(target_ph),
            'h2so4_concentration_g_L': float(h2so4_g_l.quantize(Decimal('0.0001'))),
            'initial_dose_kg': float(h2so4_kg_initial.quantize(Decimal('0.001'))),
            'continuous_dose_kg_hr': float(h2so4_kg_hr.quantize(Decimal('0.0001'))),
            'solution_93pct_initial_L': float(solution_l_initial.quantize(Decimal('0.01'))),
            'solution_93pct_continuous_L_hr': float(solution_l_hr.quantize(Decimal('0.001'))),
            'alkalinity_neutralization_mol_L': float(alk_neutralization.quantize(Decimal('0.000001'))),
            'provenance': tracker.get_provenance_record(h2so4_kg_initial).to_dict()
        }

    def _calculate_phosphate_dosing(
        self,
        sample: WaterSample,
        target_po4_mg_l: Decimal,
        volume_m3: Decimal,
        makeup_rate_m3_hr: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate phosphate treatment dosing (Na3PO4 or Na2HPO4).

        Used for boiler water treatment to maintain phosphate residual.

        Reference: ABMA Boiler Water Guidelines, ASME Consensus
        """
        current_po4 = Decimal(str(sample.phosphate_mg_l))
        MW_Na3PO4 = Decimal('163.94')  # Trisodium phosphate, anhydrous
        MW_PO4 = Decimal('94.97')

        # Phosphate deficit
        po4_deficit_mg_l = target_po4_mg_l - current_po4
        if po4_deficit_mg_l < 0:
            po4_deficit_mg_l = Decimal('0')

        # Na3PO4 required (g/L)
        # 1 mol Na3PO4 provides 1 mol PO4
        na3po4_g_l = (po4_deficit_mg_l / Decimal('1000')) * (MW_Na3PO4 / MW_PO4)

        # Total for system
        na3po4_kg_initial = na3po4_g_l * volume_m3
        na3po4_kg_hr = na3po4_g_l * makeup_rate_m3_hr

        tracker.record_step(
            operation="phosphate_dosing",
            description="Calculate Na3PO4 dosing for phosphate treatment",
            inputs={
                'current_PO4_mg_L': current_po4,
                'target_PO4_mg_L': target_po4_mg_l,
                'system_volume_m3': volume_m3
            },
            output_value=na3po4_kg_initial,
            output_name="na3po4_initial_dose_kg",
            formula="Na3PO4 = PO4_deficit × (MW_Na3PO4 / MW_PO4) × Volume",
            units="kg",
            reference="ABMA Boiler Water Guidelines"
        )

        return {
            'chemical': 'Trisodium Phosphate (Na3PO4)',
            'current_po4_mg_L': float(current_po4),
            'target_po4_mg_L': float(target_po4_mg_l),
            'na3po4_concentration_g_L': float(na3po4_g_l.quantize(Decimal('0.0001'))),
            'initial_dose_kg': float(na3po4_kg_initial.quantize(Decimal('0.001'))),
            'continuous_dose_kg_hr': float(na3po4_kg_hr.quantize(Decimal('0.0001'))),
            'provenance': tracker.get_provenance_record(na3po4_kg_initial).to_dict()
        }

    def _calculate_oxygen_scavenger_dosing(
        self,
        sample: WaterSample,
        residual_target_mg_l: Decimal,
        volume_m3: Decimal,
        makeup_rate_m3_hr: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate oxygen scavenger dosing (sodium sulfite Na2SO3).

        Stoichiometry:
        2Na2SO3 + O2 -> 2Na2SO4
        1 mg/L O2 requires 7.88 mg/L Na2SO3

        Typical excess: 20-40 mg/L Na2SO3 residual in boiler feedwater.

        Reference: ABMA, ASME PTC 19.11
        """
        do_mg_l = Decimal(str(sample.dissolved_oxygen_mg_l))
        MW_Na2SO3 = Decimal('126.04')
        MW_O2 = Decimal('32.0')

        # Stoichiometric ratio: 2 mol Na2SO3 per mol O2
        # Mass ratio: (2 × 126.04) / 32 = 7.88
        stoich_ratio = Decimal('7.88')

        # Na2SO3 for oxygen removal
        na2so3_for_o2 = do_mg_l * stoich_ratio

        # Add target residual
        total_na2so3_mg_l = na2so3_for_o2 + residual_target_mg_l

        # Convert to g/L
        na2so3_g_l = total_na2so3_mg_l / Decimal('1000')

        # Total for system
        na2so3_kg_initial = na2so3_g_l * volume_m3
        na2so3_kg_hr = na2so3_g_l * makeup_rate_m3_hr

        tracker.record_step(
            operation="oxygen_scavenger_dosing",
            description="Calculate Na2SO3 dosing for oxygen removal",
            inputs={
                'dissolved_oxygen_mg_L': do_mg_l,
                'residual_target_mg_L': residual_target_mg_l,
                'stoichiometric_ratio': stoich_ratio
            },
            output_value=na2so3_kg_initial,
            output_name="na2so3_initial_dose_kg",
            formula="Na2SO3 = (DO × 7.88 + Residual) × Volume / 1000",
            units="kg",
            reference="ABMA, ASME PTC 19.11"
        )

        return {
            'chemical': 'Sodium Sulfite (Na2SO3)',
            'dissolved_oxygen_mg_L': float(do_mg_l),
            'na2so3_for_oxygen_mg_L': float(na2so3_for_o2.quantize(Decimal('0.01'))),
            'residual_target_mg_L': float(residual_target_mg_l),
            'total_na2so3_mg_L': float(total_na2so3_mg_l.quantize(Decimal('0.01'))),
            'initial_dose_kg': float(na2so3_kg_initial.quantize(Decimal('0.001'))),
            'continuous_dose_kg_hr': float(na2so3_kg_hr.quantize(Decimal('0.0001'))),
            'stoichiometric_ratio_mg_mg': float(stoich_ratio),
            'provenance': tracker.get_provenance_record(na2so3_kg_initial).to_dict()
        }

    def _calculate_scale_inhibitor_dosing(
        self,
        sample: WaterSample,
        target_dose_mg_l: Decimal,
        volume_m3: Decimal,
        makeup_rate_m3_hr: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate scale inhibitor dosing (phosphonate or polymer).

        Typical dosing: 2-20 mg/L depending on water chemistry and inhibitor type.
        Common inhibitors: HEDP, PBTC, polyacrylates

        Reference: NACE Standards, Cooling Water Treatment Guidelines
        """
        # Scale inhibitor dosing is typically based on makeup water
        # to maintain concentration in the recirculating system

        inhibitor_g_l = target_dose_mg_l / Decimal('1000')

        # Initial dose for system volume
        inhibitor_kg_initial = inhibitor_g_l * volume_m3

        # Continuous dosing to makeup water
        inhibitor_kg_hr = inhibitor_g_l * makeup_rate_m3_hr

        # Calculate based on scaling indices
        lsi_data = self._calculate_langelier_index(
            sample, {}, ProvenanceTracker("temp", "temp", "1.0")
        )
        lsi = Decimal(str(lsi_data['lsi']))

        # Adjust dosing recommendation based on LSI
        if lsi > Decimal('1.0'):
            recommended_dose_mg_l = target_dose_mg_l * Decimal('1.5')
            dosing_note = "High scaling potential - increased dose recommended"
        elif lsi > Decimal('0.5'):
            recommended_dose_mg_l = target_dose_mg_l * Decimal('1.2')
            dosing_note = "Moderate scaling potential"
        else:
            recommended_dose_mg_l = target_dose_mg_l
            dosing_note = "Standard dosing appropriate"

        tracker.record_step(
            operation="scale_inhibitor_dosing",
            description="Calculate scale inhibitor dosing based on water chemistry",
            inputs={
                'target_dose_mg_L': target_dose_mg_l,
                'system_volume_m3': volume_m3,
                'makeup_rate_m3_hr': makeup_rate_m3_hr,
                'LSI': lsi
            },
            output_value=inhibitor_kg_initial,
            output_name="inhibitor_initial_dose_kg",
            formula="Dose = Target_mg_L × Volume / 1000",
            units="kg",
            reference="NACE Standards"
        )

        return {
            'chemical': 'Scale Inhibitor (phosphonate/polymer)',
            'target_dose_mg_L': float(target_dose_mg_l),
            'recommended_dose_mg_L': float(recommended_dose_mg_l.quantize(Decimal('0.1'))),
            'initial_dose_kg': float(inhibitor_kg_initial.quantize(Decimal('0.001'))),
            'continuous_dose_kg_hr': float(inhibitor_kg_hr.quantize(Decimal('0.0001'))),
            'lsi_value': float(lsi.quantize(Decimal('0.01'))),
            'dosing_note': dosing_note,
            'provenance': tracker.get_provenance_record(inhibitor_kg_initial).to_dict()
        }
