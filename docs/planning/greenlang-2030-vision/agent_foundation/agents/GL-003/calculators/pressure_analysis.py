# -*- coding: utf-8 -*-
"""
Pressure Drop and Flow Analysis Calculator - Zero Hallucination

Calculates pressure drops in steam piping using Darcy-Weisbach equation,
flow rates, velocities, and optimal pipe sizing.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ASME B31.1, Crane TP-410, ISO 5167
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional
from dataclasses import dataclass
import math
from .provenance import ProvenanceTracker
from greenlang.determinism import FinancialDecimal


@dataclass
class PipeFlowData:
    """Pipe flow input data."""
    flow_rate_kg_hr: float
    pipe_diameter_mm: float
    pipe_length_m: float
    pipe_roughness_mm: float
    steam_pressure_bar: float
    steam_temperature_c: float
    fittings: Dict[str, int]  # {'elbow_90': 4, 'valve_gate': 2, etc.}


@dataclass
class PressureAnalysisResult:
    """Pressure drop analysis results."""
    pressure_drop_bar: float
    pressure_drop_percent: float
    velocity_m_s: float
    reynolds_number: float
    friction_factor: float
    is_velocity_acceptable: bool
    is_pressure_drop_acceptable: bool
    recommended_pipe_size_mm: Optional[float]
    erosion_risk: str
    recommendations: List[str]
    provenance: Dict


class PressureAnalysisCalculator:
    """
    Calculate pressure drops and flow characteristics in steam piping.

    Zero Hallucination Guarantee:
    - Darcy-Weisbach equation (fundamental fluid mechanics)
    - Moody diagram correlations (Colebrook-White, Swamee-Jain)
    - No LLM inference
    """

    # Recommended steam velocities (m/s)
    VELOCITY_LIMITS = {
        'saturated_low_pressure': {'min': 15, 'max': 40, 'optimal': 25},
        'saturated_high_pressure': {'min': 25, 'max': 50, 'optimal': 35},
        'superheated': {'min': 30, 'max': 60, 'optimal': 40}
    }

    # K factors for fittings (resistance coefficients)
    FITTING_K_FACTORS = {
        'elbow_90': 0.90,
        'elbow_45': 0.40,
        'tee_line_flow': 0.60,
        'tee_branch_flow': 1.80,
        'valve_gate_open': 0.15,
        'valve_globe_open': 10.0,
        'valve_ball_open': 0.05,
        'valve_butterfly_open': 0.25,
        'entrance_sharp': 0.50,
        'entrance_rounded': 0.05,
        'exit': 1.00
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize calculator."""
        self.version = version

    def analyze_pressure_drop(
        self,
        data: PipeFlowData,
        steam_density_kg_m3: Optional[float] = None,
        steam_viscosity_pa_s: Optional[float] = None
    ) -> PressureAnalysisResult:
        """
        Analyze pressure drop and flow characteristics.

        Uses Darcy-Weisbach equation: ΔP = f * (L/D) * (ρ*v²/2)
        """
        tracker = ProvenanceTracker(
            calculation_id=f"pressure_analysis_{id(data)}",
            calculation_type="pressure_drop_analysis",
            version=self.version
        )

        tracker.record_inputs(data.__dict__)

        # Step 1: Calculate steam properties if not provided
        if steam_density_kg_m3 is None:
            rho = self._estimate_steam_density(
                data.steam_pressure_bar,
                data.steam_temperature_c,
                tracker
            )
        else:
            rho = Decimal(str(steam_density_kg_m3))

        if steam_viscosity_pa_s is None:
            mu = self._estimate_steam_viscosity(data.steam_temperature_c, tracker)
        else:
            mu = Decimal(str(steam_viscosity_pa_s))

        # Step 2: Calculate flow velocity
        velocity = self._calculate_velocity(
            data.flow_rate_kg_hr,
            data.pipe_diameter_mm,
            rho,
            tracker
        )

        # Step 3: Calculate Reynolds number
        Re = self._calculate_reynolds_number(
            velocity,
            data.pipe_diameter_mm,
            rho,
            mu,
            tracker
        )

        # Step 4: Calculate friction factor
        f = self._calculate_friction_factor(
            Re,
            data.pipe_diameter_mm,
            data.pipe_roughness_mm,
            tracker
        )

        # Step 5: Calculate pressure drop (Darcy-Weisbach)
        dp_straight = self._calculate_straight_pipe_pressure_drop(
            f,
            data.pipe_length_m,
            data.pipe_diameter_mm,
            velocity,
            rho,
            tracker
        )

        # Step 6: Calculate fitting losses
        dp_fittings = self._calculate_fitting_pressure_drop(
            data.fittings,
            velocity,
            rho,
            tracker
        )

        # Step 7: Total pressure drop
        dp_total = dp_straight + dp_fittings
        dp_percent = (dp_total / Decimal(str(data.steam_pressure_bar))) * Decimal('100')

        # Step 8: Check acceptability
        velocity_ok = self._check_velocity(float(velocity), data.steam_pressure_bar)
        pressure_ok = float(dp_percent) < 5.0  # <5% pressure drop is acceptable

        # Step 9: Recommend pipe size if current is inadequate
        recommended_size = None
        if not velocity_ok or not pressure_ok:
            recommended_size = self._recommend_pipe_size(
                data.flow_rate_kg_hr,
                float(rho),
                data.steam_pressure_bar,
                tracker
            )

        # Step 10: Assess erosion risk
        erosion_risk = self._assess_erosion_risk(FinancialDecimal.from_string(velocity), FinancialDecimal.from_string(dp_total))

        # Step 11: Generate recommendations
        recommendations = self._generate_recommendations(
            float(velocity),
            float(dp_percent),
            velocity_ok,
            pressure_ok,
            erosion_risk,
            recommended_size
        )

        return PressureAnalysisResult(
            pressure_drop_bar=FinancialDecimal.from_string(dp_total),
            pressure_drop_percent=float(dp_percent),
            velocity_m_s=float(velocity),
            reynolds_number=float(Re),
            friction_factor=FinancialDecimal.from_string(f),
            is_velocity_acceptable=velocity_ok,
            is_pressure_drop_acceptable=pressure_ok,
            recommended_pipe_size_mm=recommended_size,
            erosion_risk=erosion_risk,
            recommendations=recommendations,
            provenance=tracker.get_provenance_record(dp_total).to_dict()
        )

    def _estimate_steam_density(
        self,
        pressure_bar: float,
        temperature_c: float,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Estimate steam density using ideal gas law with compressibility.

        ρ = P / (Z * R * T)
        """
        P = Decimal(str(pressure_bar)) * Decimal('100')  # Convert to kPa
        T = Decimal(str(temperature_c)) + Decimal('273.15')  # Convert to K
        R = Decimal('0.4615')  # Specific gas constant for steam kJ/(kg·K)
        Z = Decimal('0.95')  # Compressibility factor (typical)

        rho = P / (Z * R * T)

        tracker.record_step(
            operation="steam_density",
            description="Estimate steam density",
            inputs={'pressure_bar': pressure_bar, 'temperature_c': temperature_c},
            output_value=rho,
            output_name="density_kg_m3",
            formula="ρ = P / (Z*R*T)",
            units="kg/m³"
        )

        return rho

    def _estimate_steam_viscosity(
        self,
        temperature_c: float,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Estimate steam viscosity using Sutherland's formula.

        μ = μ₀ * (T/T₀)^1.5 * (T₀ + S) / (T + S)
        """
        T = Decimal(str(temperature_c)) + Decimal('273.15')
        T0 = Decimal('373.15')  # Reference temperature (100°C)
        mu0 = Decimal('1.23e-5')  # Reference viscosity Pa·s
        S = Decimal('110.4')  # Sutherland constant

        mu = mu0 * ((T / T0) ** Decimal('1.5')) * (T0 + S) / (T + S)

        tracker.record_step(
            operation="steam_viscosity",
            description="Estimate steam viscosity",
            inputs={'temperature_c': temperature_c},
            output_value=mu,
            output_name="viscosity_pa_s",
            formula="Sutherland's formula",
            units="Pa·s"
        )

        return mu

    def _calculate_velocity(
        self,
        flow_rate_kg_hr: float,
        diameter_mm: float,
        density: Decimal,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate flow velocity.

        v = m_dot / (ρ * A)
        """
        m_dot = Decimal(str(flow_rate_kg_hr)) / Decimal('3600')  # kg/s
        D = Decimal(str(diameter_mm)) / Decimal('1000')  # m
        A = Decimal(str(math.pi)) * (D / Decimal('2')) ** 2  # m²

        v = m_dot / (density * A)

        tracker.record_step(
            operation="velocity",
            description="Calculate flow velocity",
            inputs={
                'flow_rate_kg_s': m_dot,
                'density_kg_m3': density,
                'area_m2': A
            },
            output_value=v,
            output_name="velocity_m_s",
            formula="v = m_dot / (ρ * A)",
            units="m/s"
        )

        return v

    def _calculate_reynolds_number(
        self,
        velocity: Decimal,
        diameter_mm: float,
        density: Decimal,
        viscosity: Decimal,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate Reynolds number.

        Re = ρ * v * D / μ
        """
        D = Decimal(str(diameter_mm)) / Decimal('1000')
        Re = (density * velocity * D) / viscosity

        tracker.record_step(
            operation="reynolds_number",
            description="Calculate Reynolds number",
            inputs={
                'velocity_m_s': velocity,
                'diameter_m': D,
                'density_kg_m3': density,
                'viscosity_pa_s': viscosity
            },
            output_value=Re,
            output_name="reynolds_number",
            formula="Re = ρ*v*D/μ",
            units="dimensionless"
        )

        return Re

    def _calculate_friction_factor(
        self,
        reynolds: Decimal,
        diameter_mm: float,
        roughness_mm: float,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate Darcy friction factor using Swamee-Jain equation.

        Explicit approximation of Colebrook-White equation:
        f = 0.25 / [log10(ε/3.7D + 5.74/Re^0.9)]²
        """
        D = Decimal(str(diameter_mm))
        epsilon = Decimal(str(roughness_mm))
        Re = reynolds

        # Relative roughness
        rel_roughness = epsilon / D

        # Swamee-Jain equation
        if Re > Decimal('4000'):  # Turbulent
            term = (rel_roughness / Decimal('3.7') +
                    Decimal('5.74') / (Re ** Decimal('0.9')))
            log_term = Decimal(str(math.log10(float(term))))
            f = Decimal('0.25') / (log_term ** 2)
        else:  # Laminar
            f = Decimal('64') / Re

        tracker.record_step(
            operation="friction_factor",
            description="Calculate Darcy friction factor",
            inputs={
                'reynolds_number': reynolds,
                'relative_roughness': rel_roughness
            },
            output_value=f,
            output_name="friction_factor",
            formula="Swamee-Jain equation",
            units="dimensionless"
        )

        return f

    def _calculate_straight_pipe_pressure_drop(
        self,
        friction_factor: Decimal,
        length_m: float,
        diameter_mm: float,
        velocity: Decimal,
        density: Decimal,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate pressure drop in straight pipe (Darcy-Weisbach).

        ΔP = f * (L/D) * (ρ*v²/2)
        """
        L = Decimal(str(length_m))
        D = Decimal(str(diameter_mm)) / Decimal('1000')
        f = friction_factor
        rho = density
        v = velocity

        dp_pa = f * (L / D) * (rho * v ** 2 / Decimal('2'))
        dp_bar = dp_pa / Decimal('100000')  # Pa to bar

        tracker.record_step(
            operation="pressure_drop_straight",
            description="Calculate pressure drop in straight pipe",
            inputs={
                'friction_factor': f,
                'length_m': L,
                'diameter_m': D,
                'velocity_m_s': v,
                'density_kg_m3': rho
            },
            output_value=dp_bar,
            output_name="pressure_drop_straight_bar",
            formula="ΔP = f*(L/D)*(ρ*v²/2)",
            units="bar"
        )

        return dp_bar

    def _calculate_fitting_pressure_drop(
        self,
        fittings: Dict[str, int],
        velocity: Decimal,
        density: Decimal,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate pressure drop due to fittings.

        ΔP = Σ(K * ρ*v²/2)
        """
        total_K = Decimal('0')

        for fitting_type, count in fittings.items():
            K = Decimal(str(self.FITTING_K_FACTORS.get(fitting_type, 0.5)))
            total_K += K * Decimal(str(count))

        dp_pa = total_K * (density * velocity ** 2 / Decimal('2'))
        dp_bar = dp_pa / Decimal('100000')

        tracker.record_step(
            operation="pressure_drop_fittings",
            description="Calculate pressure drop from fittings",
            inputs={
                'total_K_factor': total_K,
                'velocity_m_s': velocity,
                'density_kg_m3': density
            },
            output_value=dp_bar,
            output_name="pressure_drop_fittings_bar",
            formula="ΔP = Σ(K*ρ*v²/2)",
            units="bar"
        )

        return dp_bar

    def _check_velocity(self, velocity: float, pressure: float) -> bool:
        """Check if velocity is within acceptable range."""
        if pressure < 2.0:
            limits = self.VELOCITY_LIMITS['saturated_low_pressure']
        elif pressure < 20.0:
            limits = self.VELOCITY_LIMITS['saturated_high_pressure']
        else:
            limits = self.VELOCITY_LIMITS['superheated']

        return limits['min'] <= velocity <= limits['max']

    def _recommend_pipe_size(
        self,
        flow_rate: float,
        density: float,
        pressure: float,
        tracker: ProvenanceTracker
    ) -> float:
        """Recommend optimal pipe size based on target velocity."""
        if pressure < 2.0:
            target_v = self.VELOCITY_LIMITS['saturated_low_pressure']['optimal']
        elif pressure < 20.0:
            target_v = self.VELOCITY_LIMITS['saturated_high_pressure']['optimal']
        else:
            target_v = self.VELOCITY_LIMITS['superheated']['optimal']

        # Calculate required area: A = m_dot / (ρ * v)
        m_dot = flow_rate / 3600
        A_required = m_dot / (density * target_v)

        # Calculate diameter: D = sqrt(4*A/π)
        D = math.sqrt(4 * A_required / math.pi)
        D_mm = D * 1000

        # Round up to standard pipe size
        standard_sizes = [25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 400, 500, 600]
        recommended = next((size for size in standard_sizes if size >= D_mm), standard_sizes[-1])

        return float(recommended)

    def _assess_erosion_risk(self, velocity: float, pressure_drop: float) -> str:
        """Assess erosion risk based on velocity and pressure drop."""
        if velocity > 60:
            return "high"
        elif velocity > 50:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self,
        velocity: float,
        dp_percent: float,
        velocity_ok: bool,
        pressure_ok: bool,
        erosion_risk: str,
        recommended_size: Optional[float]
    ) -> List[str]:
        """Generate recommendations."""
        recommendations = []

        if not velocity_ok:
            if velocity < 15:
                recommendations.append(
                    f"Velocity ({velocity:.1f} m/s) is too low. Consider smaller pipe size to avoid condensate accumulation."
                )
            else:
                recommendations.append(
                    f"Velocity ({velocity:.1f} m/s) is too high. Risk of erosion and noise. "
                    f"Increase pipe size to {recommended_size}mm."
                )

        if not pressure_ok:
            recommendations.append(
                f"Pressure drop ({dp_percent:.1f}%) exceeds 5% limit. "
                f"This reduces system efficiency. Consider larger pipe size: {recommended_size}mm."
            )

        if erosion_risk == "high":
            recommendations.append(
                "HIGH erosion risk. Immediate action required: increase pipe size or use erosion-resistant materials."
            )
        elif erosion_risk == "medium":
            recommendations.append(
                "MEDIUM erosion risk. Monitor for erosion and plan pipe size increase."
            )

        if not recommendations:
            recommendations.append("Piping design is acceptable. No changes needed.")

        return recommendations
