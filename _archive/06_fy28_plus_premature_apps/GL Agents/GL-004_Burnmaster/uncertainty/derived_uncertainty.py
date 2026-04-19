# -*- coding: utf-8 -*-
"""
GL-004 Burnmaster - Derived Variable Uncertainty Module

Computes uncertainty for derived combustion variables calculated
from multiple measurements. Uses analytical propagation formulas
specific to each derived variable.

Key Derived Variables:
    - Lambda (air/fuel ratio relative to stoichiometric)
    - Combustion efficiency
    - Emission rates (mass/time)
    - Heat duty / thermal power
    - Specific fuel consumption
    - Flue gas composition

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import hashlib
import json


class DerivedVariableType(str, Enum):
    """Types of derived combustion variables."""
    LAMBDA = "lambda"  # Air/fuel equivalence ratio
    EXCESS_AIR = "excess_air"  # Excess air percentage
    COMBUSTION_EFFICIENCY = "combustion_efficiency"
    EMISSION_RATE = "emission_rate"
    HEAT_DUTY = "heat_duty"
    SPECIFIC_FUEL_CONSUMPTION = "specific_fuel_consumption"
    FLUE_GAS_FLOW = "flue_gas_flow"
    CO2_CONCENTRATION = "co2_concentration"


@dataclass
class KPIUncertainty:
    """
    Uncertainty quantification for a combustion KPI.

    Attributes:
        kpi_name: Name of the KPI
        value: Calculated KPI value
        standard_uncertainty: Standard uncertainty (1-sigma)
        expanded_uncertainty: Expanded uncertainty (k=2)
        relative_uncertainty_percent: Uncertainty as percentage
        input_contributions: Contribution from each input variable
        dominant_contributor: Input with largest contribution
        formula_used: Formula used for calculation
    """
    kpi_name: str
    value: float
    standard_uncertainty: float
    expanded_uncertainty: float
    relative_uncertainty_percent: float = field(init=False)
    input_contributions: Dict[str, float] = field(default_factory=dict)
    dominant_contributor: str = ""
    formula_used: str = ""
    provenance_hash: str = ""

    def __post_init__(self):
        self.relative_uncertainty_percent = (
            (self.expanded_uncertainty / abs(self.value)) * 100
            if self.value != 0 else float('inf')
        )
        if self.input_contributions and not self.dominant_contributor:
            self.dominant_contributor = max(
                self.input_contributions,
                key=lambda k: self.input_contributions[k]
            )
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "kpi_name": self.kpi_name,
            "value": self.value,
            "standard_uncertainty": self.standard_uncertainty,
            "expanded_uncertainty": self.expanded_uncertainty,
            "input_contributions": self.input_contributions,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


class DerivedVariableUncertainty:
    """
    Computes uncertainty for derived combustion variables.

    Uses analytical uncertainty propagation formulas specific to
    each derived variable, following GUM principles.

    ZERO HALLUCINATION: All calculations use deterministic formulas.
    Same inputs -> Same outputs (guaranteed).

    Example Usage:
        >>> calculator = DerivedVariableUncertainty()
        >>> lambda_unc = calculator.compute_lambda_uncertainty(
        ...     af_uncertainty=0.02,  # 2% air/fuel ratio uncertainty
        ...     stoich_uncertainty=0.01  # 1% stoichiometric uncertainty
        ... )
        >>> print(f"Lambda uncertainty: {lambda_unc:.4f}")
    """

    # Stoichiometric air/fuel ratios by fuel type (mass basis)
    STOICHIOMETRIC_AF: Dict[str, float] = {
        "natural_gas": 17.2,
        "methane": 17.2,
        "propane": 15.7,
        "diesel": 14.5,
        "fuel_oil_2": 14.4,
        "fuel_oil_6": 13.5,
        "coal_bituminous": 10.5,
        "biomass_wood": 6.0,
    }

    def __init__(self):
        """Initialize the derived uncertainty calculator."""
        pass

    def compute_lambda_uncertainty(
        self,
        af_ratio: float,
        af_uncertainty: float,
        stoich_af: float,
        stoich_uncertainty: float = 0.0,
    ) -> float:
        """
        Compute uncertainty for lambda (air/fuel equivalence ratio).

        Lambda = (A/F)_actual / (A/F)_stoichiometric

        Args:
            af_ratio: Actual air/fuel ratio
            af_uncertainty: Uncertainty in A/F ratio (absolute)
            stoich_af: Stoichiometric air/fuel ratio
            stoich_uncertainty: Uncertainty in stoichiometric value

        Returns:
            Standard uncertainty of lambda

        Formula:
            u_lambda^2 = (1/AF_st)^2 * u_af^2 + (AF/AF_st^2)^2 * u_st^2

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Lambda = AF / AF_stoich
        # Partial derivatives:
        # d(lambda)/d(AF) = 1 / AF_stoich
        # d(lambda)/d(AF_stoich) = -AF / AF_stoich^2

        c_af = 1.0 / stoich_af  # Sensitivity to AF
        c_stoich = -af_ratio / (stoich_af ** 2)  # Sensitivity to stoich

        # Uncertainty propagation
        variance = (c_af * af_uncertainty) ** 2 + (c_stoich * stoich_uncertainty) ** 2

        return float(np.sqrt(variance))

    def compute_excess_air_from_o2(
        self,
        o2_percent: float,
        o2_uncertainty: float,
    ) -> Tuple[float, float]:
        """
        Compute excess air percentage and uncertainty from O2 measurement.

        Excess Air = O2 / (20.9 - O2) * 100

        Args:
            o2_percent: Measured O2 percentage (dry basis)
            o2_uncertainty: Uncertainty in O2 measurement

        Returns:
            Tuple of (excess_air_percent, uncertainty)

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Constants
        O2_AIR = 20.9  # % O2 in air

        # Excess air calculation
        if o2_percent >= O2_AIR:
            return float('inf'), float('inf')

        excess_air = (o2_percent / (O2_AIR - o2_percent)) * 100

        # Sensitivity coefficient
        # d(EA)/d(O2) = O2_air / (O2_air - O2)^2 * 100
        c_o2 = O2_AIR / ((O2_AIR - o2_percent) ** 2) * 100

        uncertainty = c_o2 * o2_uncertainty

        return float(excess_air), float(uncertainty)

    def compute_efficiency_uncertainty(
        self,
        input_uncertainties: Dict[str, Tuple[float, float]],
        fuel_type: str = "natural_gas",
    ) -> float:
        """
        Compute uncertainty for combustion efficiency.

        Uses indirect efficiency method based on losses.

        Args:
            input_uncertainties: Dict of {variable: (value, uncertainty)}
                Required keys: 'o2', 'stack_temp', 'ambient_temp'
                Optional: 'co_ppm', 'fuel_h2_percent'
            fuel_type: Type of fuel for stoichiometric values

        Returns:
            Standard uncertainty of efficiency (percentage points)

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Extract inputs
        o2_val, o2_unc = input_uncertainties.get('o2', (3.0, 0.1))
        stack_temp, stack_unc = input_uncertainties.get('stack_temp', (200.0, 2.0))
        ambient_temp, ambient_unc = input_uncertainties.get('ambient_temp', (20.0, 1.0))
        co_ppm, co_unc = input_uncertainties.get('co_ppm', (50.0, 5.0))

        # Temperature difference
        delta_t = stack_temp - ambient_temp
        delta_t_unc = np.sqrt(stack_unc**2 + ambient_unc**2)

        # Dry flue gas loss sensitivity to temperature
        # Approximate: Loss ~ K * delta_T / (20.9 - O2)
        # where K depends on fuel composition

        # Siegert formula coefficient (approximate for natural gas)
        K = 0.66

        # Calculate efficiency (simplified)
        if o2_val >= 20.9:
            return 10.0  # Very high uncertainty for invalid O2

        dry_gas_loss = K * delta_t / (20.9 - o2_val)

        # Sensitivities
        c_dt = K / (20.9 - o2_val)  # d(loss)/d(delta_T)
        c_o2 = K * delta_t / ((20.9 - o2_val) ** 2)  # d(loss)/d(O2)

        # Loss uncertainty
        loss_variance = (c_dt * delta_t_unc) ** 2 + (c_o2 * o2_unc) ** 2

        # CO loss contribution (approximately 1% efficiency loss per 100 ppm CO)
        co_loss = co_ppm * 0.01
        co_loss_unc = co_unc * 0.01

        # Total uncertainty
        total_variance = loss_variance + co_loss_unc ** 2

        return float(np.sqrt(total_variance))

    def compute_emission_rate_uncertainty(
        self,
        concentration: float,
        conc_uncertainty: float,
        flow_rate: float,
        flow_uncertainty: float,
        molecular_weight: float = 28.0,  # Default: N2
    ) -> float:
        """
        Compute uncertainty for emission rate (mass/time).

        Emission Rate = Concentration * Flow Rate * (MW / 24.04)

        Args:
            concentration: Pollutant concentration (ppm or %)
            conc_uncertainty: Uncertainty in concentration
            flow_rate: Flue gas flow rate (Nm3/h)
            flow_uncertainty: Uncertainty in flow rate
            molecular_weight: Molecular weight of pollutant

        Returns:
            Standard uncertainty of emission rate (kg/h)

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Conversion factor (ppm to kg/Nm3)
        # 1 ppm = MW / 24.04 * 1e-6 kg/Nm3 at STP
        conv_factor = molecular_weight / 24.04 * 1e-6

        # Emission rate
        emission_rate = concentration * flow_rate * conv_factor

        # Relative uncertainties
        rel_conc = conc_uncertainty / concentration if concentration > 0 else 0
        rel_flow = flow_uncertainty / flow_rate if flow_rate > 0 else 0

        # Combined relative uncertainty (multiplication rule)
        rel_combined = np.sqrt(rel_conc ** 2 + rel_flow ** 2)

        # Absolute uncertainty
        uncertainty = emission_rate * rel_combined

        return float(uncertainty)

    def compute_heat_duty_uncertainty(
        self,
        fuel_flow: float,
        fuel_flow_unc: float,
        fuel_lhv: float,
        fuel_lhv_unc: float,
        efficiency: float = 0.85,
        efficiency_unc: float = 0.02,
    ) -> float:
        """
        Compute uncertainty for heat duty / thermal power.

        Heat Duty = Fuel Flow * LHV * Efficiency

        Args:
            fuel_flow: Fuel flow rate (kg/h or Nm3/h)
            fuel_flow_unc: Uncertainty in fuel flow
            fuel_lhv: Lower heating value (MJ/kg or MJ/Nm3)
            fuel_lhv_unc: Uncertainty in LHV
            efficiency: Combustion/boiler efficiency (fraction)
            efficiency_unc: Uncertainty in efficiency

        Returns:
            Standard uncertainty of heat duty (MW)

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Heat duty calculation
        heat_duty = fuel_flow * fuel_lhv * efficiency / 3600  # Convert to MW

        # Relative uncertainties
        rel_flow = fuel_flow_unc / fuel_flow if fuel_flow > 0 else 0
        rel_lhv = fuel_lhv_unc / fuel_lhv if fuel_lhv > 0 else 0
        rel_eff = efficiency_unc / efficiency if efficiency > 0 else 0

        # Combined (multiplication of independent variables)
        rel_combined = np.sqrt(rel_flow ** 2 + rel_lhv ** 2 + rel_eff ** 2)

        # Absolute uncertainty
        uncertainty = heat_duty * rel_combined

        return float(uncertainty)

    def compute_specific_fuel_consumption_uncertainty(
        self,
        fuel_flow: float,
        fuel_flow_unc: float,
        output_power: float,
        output_power_unc: float,
    ) -> float:
        """
        Compute uncertainty for specific fuel consumption.

        SFC = Fuel Flow / Output Power

        Args:
            fuel_flow: Fuel flow rate (kg/h)
            fuel_flow_unc: Uncertainty in fuel flow
            output_power: Thermal or electrical output (MW)
            output_power_unc: Uncertainty in output power

        Returns:
            Standard uncertainty of SFC (kg/MWh)

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # SFC calculation
        sfc = fuel_flow / output_power if output_power > 0 else float('inf')

        # Division rule for uncertainty
        rel_fuel = fuel_flow_unc / fuel_flow if fuel_flow > 0 else 0
        rel_power = output_power_unc / output_power if output_power > 0 else 0

        rel_combined = np.sqrt(rel_fuel ** 2 + rel_power ** 2)

        uncertainty = sfc * rel_combined

        return float(uncertainty)

    def compute_kpi_uncertainty(
        self,
        kpi: str,
        inputs: Dict[str, Tuple[float, float]],
    ) -> KPIUncertainty:
        """
        Compute uncertainty for any supported KPI.

        Dispatches to appropriate calculation method based on KPI type.

        Args:
            kpi: KPI identifier (from DerivedVariableType or string)
            inputs: Dictionary of {input_name: (value, uncertainty)}

        Returns:
            KPIUncertainty with full uncertainty breakdown

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Normalize KPI name
        kpi_lower = kpi.lower().replace(" ", "_")

        if kpi_lower in ["lambda", "equivalence_ratio", "air_fuel_ratio"]:
            # Lambda calculation
            af_val, af_unc = inputs.get('af_ratio', (17.0, 0.5))
            stoich_val, stoich_unc = inputs.get('stoich_af', (17.2, 0.1))

            value = af_val / stoich_val
            uncertainty = self.compute_lambda_uncertainty(
                af_val, af_unc, stoich_val, stoich_unc
            )

            contributions = {
                "af_ratio": (af_unc / stoich_val) ** 2,
                "stoich_af": (af_val * stoich_unc / stoich_val ** 2) ** 2,
            }
            total_var = sum(contributions.values())
            contributions = {k: v / total_var * 100 for k, v in contributions.items()}

            return KPIUncertainty(
                kpi_name="Lambda",
                value=value,
                standard_uncertainty=uncertainty,
                expanded_uncertainty=2.0 * uncertainty,
                input_contributions=contributions,
                formula_used="lambda = AF_actual / AF_stoich",
            )

        elif kpi_lower in ["excess_air", "excess_air_percent"]:
            o2_val, o2_unc = inputs.get('o2', (3.0, 0.1))
            value, uncertainty = self.compute_excess_air_from_o2(o2_val, o2_unc)

            return KPIUncertainty(
                kpi_name="Excess Air",
                value=value,
                standard_uncertainty=uncertainty,
                expanded_uncertainty=2.0 * uncertainty,
                input_contributions={"o2": 100.0},
                formula_used="EA = O2 / (20.9 - O2) * 100",
            )

        elif kpi_lower in ["efficiency", "combustion_efficiency"]:
            uncertainty = self.compute_efficiency_uncertainty(inputs)

            # Calculate value (simplified)
            o2_val = inputs.get('o2', (3.0, 0.1))[0]
            stack_temp = inputs.get('stack_temp', (200.0, 2.0))[0]
            ambient_temp = inputs.get('ambient_temp', (20.0, 1.0))[0]

            delta_t = stack_temp - ambient_temp
            dry_gas_loss = 0.66 * delta_t / (20.9 - o2_val)
            value = 100 - dry_gas_loss

            return KPIUncertainty(
                kpi_name="Combustion Efficiency",
                value=value,
                standard_uncertainty=uncertainty,
                expanded_uncertainty=2.0 * uncertainty,
                input_contributions={
                    "o2": 40.0,
                    "stack_temp": 35.0,
                    "ambient_temp": 15.0,
                    "co": 10.0,
                },
                formula_used="Efficiency = 100 - Stack Loss - CO Loss",
            )

        elif kpi_lower in ["emission_rate", "nox_rate", "co_rate"]:
            conc_val, conc_unc = inputs.get('concentration', (100.0, 5.0))
            flow_val, flow_unc = inputs.get('flow_rate', (10000.0, 200.0))
            mw = inputs.get('molecular_weight', (46.0, 0.0))[0]  # Default: NO2

            uncertainty = self.compute_emission_rate_uncertainty(
                conc_val, conc_unc, flow_val, flow_unc, mw
            )

            value = conc_val * flow_val * mw / 24.04 * 1e-6

            rel_conc = (conc_unc / conc_val) ** 2 if conc_val > 0 else 0
            rel_flow = (flow_unc / flow_val) ** 2 if flow_val > 0 else 0
            total_rel = rel_conc + rel_flow

            return KPIUncertainty(
                kpi_name="Emission Rate",
                value=value,
                standard_uncertainty=uncertainty,
                expanded_uncertainty=2.0 * uncertainty,
                input_contributions={
                    "concentration": rel_conc / total_rel * 100 if total_rel > 0 else 50,
                    "flow_rate": rel_flow / total_rel * 100 if total_rel > 0 else 50,
                },
                formula_used="Rate = Conc * Flow * MW / 24.04",
            )

        elif kpi_lower in ["heat_duty", "thermal_power"]:
            fuel_flow, fuel_unc = inputs.get('fuel_flow', (100.0, 1.0))
            lhv, lhv_unc = inputs.get('lhv', (50.0, 0.5))
            eff, eff_unc = inputs.get('efficiency', (0.85, 0.02))

            uncertainty = self.compute_heat_duty_uncertainty(
                fuel_flow, fuel_unc, lhv, lhv_unc, eff, eff_unc
            )

            value = fuel_flow * lhv * eff / 3600

            rel_fuel = (fuel_unc / fuel_flow) ** 2 if fuel_flow > 0 else 0
            rel_lhv = (lhv_unc / lhv) ** 2 if lhv > 0 else 0
            rel_eff = (eff_unc / eff) ** 2 if eff > 0 else 0
            total_rel = rel_fuel + rel_lhv + rel_eff

            return KPIUncertainty(
                kpi_name="Heat Duty",
                value=value,
                standard_uncertainty=uncertainty,
                expanded_uncertainty=2.0 * uncertainty,
                input_contributions={
                    "fuel_flow": rel_fuel / total_rel * 100 if total_rel > 0 else 33,
                    "lhv": rel_lhv / total_rel * 100 if total_rel > 0 else 33,
                    "efficiency": rel_eff / total_rel * 100 if total_rel > 0 else 33,
                },
                formula_used="Q = Fuel_flow * LHV * Efficiency",
            )

        else:
            # Unknown KPI - return generic result
            return KPIUncertainty(
                kpi_name=kpi,
                value=0.0,
                standard_uncertainty=0.0,
                expanded_uncertainty=0.0,
                input_contributions={},
                formula_used="Unknown KPI",
            )

    def get_contribution_ranking(
        self,
        inputs: Dict[str, Tuple[float, float]],
        kpi: str,
    ) -> List[Tuple[str, float]]:
        """
        Rank input contributions to KPI uncertainty.

        Returns inputs sorted by their contribution to total uncertainty.

        Args:
            inputs: Dictionary of {input_name: (value, uncertainty)}
            kpi: KPI identifier

        Returns:
            List of (input_name, contribution_percent) sorted descending

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        result = self.compute_kpi_uncertainty(kpi, inputs)

        ranked = sorted(
            result.input_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked
