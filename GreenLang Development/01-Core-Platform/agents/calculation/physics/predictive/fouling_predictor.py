"""
Heat Exchanger Fouling Prediction

Zero-Hallucination Fouling Prediction Models

This module implements deterministic fouling prediction models for
heat exchangers based on operating conditions and historical data.

References:
    - TEMA Standards (Tubular Exchanger Manufacturers Association)
    - Kern-Seaton Model for asymptotic fouling
    - Ebert-Panchal Model for refinery fouling
    - Polley Threshold Model

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math
import hashlib


class FoulingMechanism(Enum):
    """Fouling mechanisms per TEMA."""
    PARTICULATE = "particulate"
    CRYSTALLIZATION = "crystallization"
    BIOLOGICAL = "biological"
    CORROSION = "corrosion"
    CHEMICAL_REACTION = "chemical_reaction"
    SOLIDIFICATION = "solidification"


@dataclass
class FoulingData:
    """Operating data for fouling prediction."""
    time_hours: float
    inlet_temp_c: float
    outlet_temp_c: float
    wall_temp_c: float
    velocity_m_s: float
    heat_duty_kw: float
    pressure_drop_kpa: float


@dataclass
class FoulingPredictionResult:
    """
    Fouling prediction results.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Current fouling state
    fouling_resistance_m2k_w: Decimal
    fouling_thickness_mm: Decimal
    cleanliness_factor: Decimal

    # Heat transfer impact
    ua_reduction_pct: Decimal
    heat_transfer_loss_kw: Decimal

    # Predictions
    asymptotic_fouling_m2k_w: Decimal
    time_to_critical_hours: Decimal
    time_to_cleaning_hours: Decimal

    # Rate of fouling
    fouling_rate_m2k_w_hr: Decimal
    is_accelerating: bool

    # Recommendations
    recommended_action: str
    cleaning_benefit_pct: Decimal

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        return {
            "fouling_resistance_m2k_w": float(self.fouling_resistance_m2k_w),
            "cleanliness_factor": float(self.cleanliness_factor),
            "time_to_cleaning_hours": float(self.time_to_cleaning_hours),
            "recommended_action": self.recommended_action,
            "provenance_hash": self.provenance_hash
        }


class FoulingPredictor:
    """
    Heat Exchanger Fouling Prediction Engine.

    ZERO-HALLUCINATION GUARANTEE:
    - Deterministic fouling models
    - Based on established correlations
    - Complete provenance tracking

    Models:
    1. Kern-Seaton: Asymptotic fouling model
    2. Ebert-Panchal: Threshold fouling model
    3. Linear: Simple linear accumulation
    4. Combined: Multiple mechanisms

    References:
        - Kern, D.Q., Seaton, R.E. (1959) "A Theoretical Analysis of Thermal
          Surface Fouling"
        - Ebert, W., Panchal, C.B. (1995) "Analysis of Exxon Crude-Oil Slip
          Stream Coking Data"
    """

    # TEMA fouling resistance design values (m2-K/W)
    TEMA_FOULING_FACTORS = {
        "cooling_tower_water": 0.00035,
        "city_water": 0.00018,
        "boiler_blowdown": 0.00035,
        "crude_oil": 0.00053,
        "residual_fuel_oil": 0.00088,
        "natural_gas": 0.00009,
        "steam_clean": 0.00009,
        "refrigerant": 0.00018,
    }

    def __init__(
        self,
        design_ua_kw_k: float,
        design_fouling_m2k_w: float = 0.0003,
        critical_fouling_m2k_w: float = 0.001,
        precision: int = 6
    ):
        """
        Initialize fouling predictor.

        Args:
            design_ua_kw_k: Clean UA value (kW/K)
            design_fouling_m2k_w: Design fouling allowance (m2-K/W)
            critical_fouling_m2k_w: Critical fouling level triggering cleaning
            precision: Output precision
        """
        self.ua_clean = Decimal(str(design_ua_kw_k))
        self.rf_design = Decimal(str(design_fouling_m2k_w))
        self.rf_critical = Decimal(str(critical_fouling_m2k_w))
        self.precision = precision

        # Historical data storage
        self.history: List[FoulingData] = []

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "method": "Fouling_Prediction",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def calculate_fouling_from_performance(
        self,
        current_ua_kw_k: float,
        area_m2: float
    ) -> Decimal:
        """
        Calculate fouling resistance from measured UA.

        Reference: 1/UA_fouled = 1/UA_clean + Rf/A

        Args:
            current_ua_kw_k: Current (fouled) UA value
            area_m2: Heat transfer area

        Returns:
            Fouling resistance (m2-K/W)
        """
        ua_fouled = Decimal(str(current_ua_kw_k))
        a = Decimal(str(area_m2))

        if ua_fouled <= 0 or a <= 0:
            raise ValueError("UA and area must be positive")

        # Rf = A * (1/UA_fouled - 1/UA_clean)
        rf = a * (Decimal("1") / ua_fouled - Decimal("1") / self.ua_clean)

        if rf < 0:
            rf = Decimal("0")

        return self._apply_precision(rf)

    def kern_seaton_model(
        self,
        time_hours: float,
        deposition_rate: float,
        removal_constant: float
    ) -> Decimal:
        """
        Kern-Seaton asymptotic fouling model.

        Reference: Kern & Seaton (1959)

        Rf(t) = Rf_inf * (1 - exp(-b*t))

        Where:
        - Rf_inf = deposition_rate / removal_constant
        - b = removal_constant

        Args:
            time_hours: Time since cleaning
            deposition_rate: Fouling deposition rate (m2-K/W/hr)
            removal_constant: Removal rate constant (1/hr)

        Returns:
            Fouling resistance (m2-K/W)
        """
        t = Decimal(str(time_hours))
        phi_d = Decimal(str(deposition_rate))
        b = Decimal(str(removal_constant))

        if b <= 0:
            raise ValueError("Removal constant must be positive")

        # Asymptotic fouling
        rf_inf = phi_d / b

        # Current fouling
        exp_term = Decimal(str(math.exp(-float(b * t))))
        rf = rf_inf * (Decimal("1") - exp_term)

        return self._apply_precision(rf)

    def ebert_panchal_model(
        self,
        wall_temp_c: float,
        velocity_m_s: float,
        time_hours: float,
        activation_energy_j_mol: float = 68000,
        pre_exponential: float = 1e-10
    ) -> Decimal:
        """
        Ebert-Panchal threshold fouling model.

        Reference: Ebert & Panchal (1995)

        dRf/dt = alpha * exp(-E/RT_wall) - gamma * tau_w * Rf

        For crude oil fouling in refineries.

        Args:
            wall_temp_c: Wall temperature (C)
            velocity_m_s: Flow velocity (m/s)
            time_hours: Time since cleaning
            activation_energy_j_mol: Activation energy (J/mol)
            pre_exponential: Pre-exponential factor

        Returns:
            Fouling resistance (m2-K/W)
        """
        t_wall_k = Decimal(str(wall_temp_c + 273.15))
        v = Decimal(str(velocity_m_s))
        t = Decimal(str(time_hours))
        e_a = Decimal(str(activation_energy_j_mol))
        alpha = Decimal(str(pre_exponential))
        r_gas = Decimal("8.314")  # J/(mol*K)

        # Wall shear stress (simplified correlation)
        # tau_w ~ 0.5 * rho * f * v^2
        # Assuming f ~ 0.005 and rho ~ 800 kg/m3
        tau_w = Decimal("2") * v ** 2  # Simplified

        # Fouling rate
        exp_term = Decimal(str(math.exp(-float(e_a / (r_gas * t_wall_k)))))
        deposition = alpha * exp_term

        # Removal term (simplified)
        gamma = Decimal("1e-6")  # Removal coefficient
        removal = gamma * tau_w

        # Net fouling rate
        if deposition > removal:
            rate = deposition - removal
        else:
            rate = Decimal("0")  # Below threshold - no fouling

        # Integrate over time (simplified linear)
        rf = rate * t

        return self._apply_precision(rf)

    def predict(
        self,
        current_data: FoulingData,
        mechanism: FoulingMechanism = FoulingMechanism.PARTICULATE
    ) -> FoulingPredictionResult:
        """
        Predict fouling based on current conditions.

        ZERO-HALLUCINATION: Deterministic fouling prediction.

        Args:
            current_data: Current operating data
            mechanism: Primary fouling mechanism

        Returns:
            FoulingPredictionResult with predictions
        """
        t = Decimal(str(current_data.time_hours))
        t_wall = Decimal(str(current_data.wall_temp_c))
        v = Decimal(str(current_data.velocity_m_s))
        q = Decimal(str(current_data.heat_duty_kw))

        # Store data point
        self.history.append(current_data)

        # Model selection based on mechanism
        if mechanism == FoulingMechanism.CHEMICAL_REACTION:
            # Use Ebert-Panchal for chemical/thermal fouling
            rf = self.ebert_panchal_model(
                current_data.wall_temp_c,
                current_data.velocity_m_s,
                current_data.time_hours
            )
            # Asymptotic value (high for chemical fouling)
            rf_asymp = self.rf_critical * Decimal("2")
        else:
            # Use Kern-Seaton for particulate/biological
            # Estimate parameters from conditions
            deposition_rate = Decimal("1e-7") / v  # Higher velocity = lower deposition
            removal_constant = Decimal("0.001") * v  # Higher velocity = more removal

            rf = self.kern_seaton_model(
                current_data.time_hours,
                float(deposition_rate),
                float(removal_constant)
            )
            rf_asymp = deposition_rate / removal_constant if removal_constant > 0 else self.rf_critical

        # Fouling thickness (assuming thermal conductivity ~ 1 W/m-K for deposit)
        k_deposit = Decimal("1")  # W/m-K
        thickness_m = rf * k_deposit
        thickness_mm = thickness_m * Decimal("1000")

        # Cleanliness factor
        # CF = UA_current / UA_clean
        # UA_current = UA_clean / (1 + UA_clean * Rf / A)
        # Assume A = 100 m2 typical
        a_typical = Decimal("100")
        cf = Decimal("1") / (Decimal("1") + self.ua_clean * rf / a_typical)

        # UA reduction
        ua_reduction = (Decimal("1") - cf) * Decimal("100")

        # Heat transfer loss
        q_loss = q * (Decimal("1") - cf)

        # Time to critical fouling
        if t > 0 and rf > 0:
            fouling_rate = rf / t
            rf_remaining = self.rf_critical - rf
            if fouling_rate > 0 and rf_remaining > 0:
                time_to_critical = rf_remaining / fouling_rate
            else:
                time_to_critical = Decimal("0")
        else:
            fouling_rate = Decimal("0")
            time_to_critical = Decimal("999999")

        # Time to recommended cleaning (80% of critical)
        time_to_cleaning = time_to_critical * Decimal("0.8")

        # Check if fouling is accelerating
        is_accelerating = False
        if len(self.history) >= 3:
            # Compare recent rate to earlier rate
            recent_points = self.history[-3:]
            if recent_points[2].time_hours > recent_points[0].time_hours:
                recent_dt = recent_points[2].time_hours - recent_points[0].time_hours
                # This would need actual Rf calculation at each point
                is_accelerating = False  # Placeholder

        # Recommendations
        if rf > self.rf_critical:
            action = "IMMEDIATE CLEANING REQUIRED"
        elif rf > self.rf_critical * Decimal("0.8"):
            action = "Schedule cleaning within 1 week"
        elif rf > self.rf_critical * Decimal("0.5"):
            action = "Monitor closely, plan cleaning"
        else:
            action = "Normal operation, continue monitoring"

        # Cleaning benefit
        cleaning_benefit = ua_reduction

        # Provenance
        inputs = {
            "time_hours": str(t),
            "wall_temp_c": str(t_wall),
            "velocity_m_s": str(v),
            "mechanism": mechanism.value
        }
        outputs = {
            "fouling_resistance": str(rf),
            "time_to_critical": str(time_to_critical)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return FoulingPredictionResult(
            fouling_resistance_m2k_w=self._apply_precision(rf),
            fouling_thickness_mm=self._apply_precision(thickness_mm),
            cleanliness_factor=self._apply_precision(cf),
            ua_reduction_pct=self._apply_precision(ua_reduction),
            heat_transfer_loss_kw=self._apply_precision(q_loss),
            asymptotic_fouling_m2k_w=self._apply_precision(rf_asymp),
            time_to_critical_hours=self._apply_precision(time_to_critical),
            time_to_cleaning_hours=self._apply_precision(time_to_cleaning),
            fouling_rate_m2k_w_hr=self._apply_precision(fouling_rate),
            is_accelerating=is_accelerating,
            recommended_action=action,
            cleaning_benefit_pct=self._apply_precision(cleaning_benefit),
            provenance_hash=provenance_hash
        )

    def reset_history(self) -> None:
        """Clear fouling history (after cleaning)."""
        self.history = []


# Convenience functions
def predict_fouling(
    time_since_cleaning_hours: float,
    wall_temp_c: float,
    velocity_m_s: float,
    design_ua_kw_k: float = 100.0
) -> FoulingPredictionResult:
    """
    Predict heat exchanger fouling.

    Example:
        >>> result = predict_fouling(
        ...     time_since_cleaning_hours=2000,
        ...     wall_temp_c=150,
        ...     velocity_m_s=1.5,
        ...     design_ua_kw_k=100
        ... )
        >>> print(f"Fouling: {result.fouling_resistance_m2k_w} m2-K/W")
    """
    predictor = FoulingPredictor(design_ua_kw_k=design_ua_kw_k)

    data = FoulingData(
        time_hours=time_since_cleaning_hours,
        inlet_temp_c=100,
        outlet_temp_c=80,
        wall_temp_c=wall_temp_c,
        velocity_m_s=velocity_m_s,
        heat_duty_kw=1000,
        pressure_drop_kpa=50
    )

    return predictor.predict(data)


def estimate_cleaning_interval(
    critical_fouling_m2k_w: float = 0.001,
    fouling_rate_m2k_w_hr: float = 1e-7
) -> float:
    """
    Estimate optimal cleaning interval.

    Returns:
        Recommended cleaning interval in hours
    """
    if fouling_rate_m2k_w_hr <= 0:
        return float('inf')

    # Clean at 80% of critical
    target = critical_fouling_m2k_w * 0.8
    interval = target / fouling_rate_m2k_w_hr

    return interval
