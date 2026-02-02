"""
GL-017 CONDENSYNC - Fouling Calculator

Zero-hallucination, deterministic calculations for condenser tube fouling
analysis and cleaning optimization following HEI Standards and TEMA.

This module provides:
- Tube fouling factor calculation
- Fouling rate prediction
- Cleaning interval optimization
- Micro/macro fouling differentiation
- Biofouling risk assessment
- Tube plugging impact calculation

Standards Reference:
- HEI Standards for Steam Surface Condensers (11th Edition)
- TEMA (Tubular Exchanger Manufacturers Association) Standards
- EPRI Condenser Fouling Guidelines

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# TEMA standard fouling factors (m2-K/W)
TEMA_FOULING_FACTORS = {
    "seawater": {
        "below_43c": 0.000088,
        "above_43c": 0.000176,
    },
    "brackish_water": {
        "below_43c": 0.000176,
        "above_43c": 0.000352,
    },
    "river_water": {
        "minimum": 0.000176,
        "average": 0.000352,
    },
    "cooling_tower_water": {
        "treated": 0.000176,
        "untreated": 0.000528,
    },
    "city_water": 0.000176,
    "boiler_feedwater": 0.000088,
}

# Fouling rate constants (m2-K/W per day)
# Based on empirical correlations
FOULING_RATE_CONSTANTS = {
    "low": 0.000001,      # Well-treated water
    "moderate": 0.000005,  # Average conditions
    "high": 0.000015,      # Poor water quality
    "severe": 0.000030,    # Biofouling conditions
}

# Biofouling risk thresholds
BIOFOULING_RISK_THRESHOLDS = {
    "temperature_c": {
        "low_risk_max": 25,
        "moderate_risk_max": 35,
        "high_risk_above": 35,
    },
    "velocity_m_s": {
        "high_risk_below": 1.0,
        "moderate_risk_max": 2.0,
        "low_risk_above": 2.0,
    }
}

# Cleaning effectiveness factors
CLEANING_EFFECTIVENESS = {
    "mechanical_brushing": 0.90,
    "chemical_cleaning": 0.85,
    "high_pressure_water": 0.80,
    "online_ball_cleaning": 0.75,
    "thermal_shock": 0.60,
}


class WaterType(Enum):
    """Types of cooling water sources."""
    SEAWATER = "seawater"
    BRACKISH = "brackish_water"
    RIVER = "river_water"
    COOLING_TOWER = "cooling_tower_water"
    CITY = "city_water"


class FoulingType(Enum):
    """Types of fouling mechanisms."""
    BIOLOGICAL = "biological"
    PARTICULATE = "particulate"
    SCALE = "scale"
    CORROSION = "corrosion"
    MIXED = "mixed"


class BiofoulingRisk(Enum):
    """Biofouling risk categories."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class FoulingInput:
    """
    Input parameters for fouling calculations.

    Attributes:
        design_u_value_w_m2k: Design U-value (W/m2-K)
        actual_u_value_w_m2k: Current U-value (W/m2-K)
        cw_inlet_temp_c: Cooling water inlet temperature (C)
        cw_outlet_temp_c: Cooling water outlet temperature (C)
        cw_velocity_m_s: Cooling water velocity in tubes (m/s)
        water_type: Type of cooling water source
        num_tubes_total: Total number of tubes
        num_tubes_plugged: Number of plugged tubes
        days_since_cleaning: Days since last cleaning
        tube_id_mm: Tube inside diameter (mm)
        tube_material: Tube material
        water_treatment_quality: Treatment quality (0-1 scale)
    """
    design_u_value_w_m2k: float
    actual_u_value_w_m2k: float
    cw_inlet_temp_c: float
    cw_outlet_temp_c: float
    cw_velocity_m_s: float
    water_type: str
    num_tubes_total: int
    num_tubes_plugged: int
    days_since_cleaning: int
    tube_id_mm: float
    tube_material: str
    water_treatment_quality: float = 0.8


@dataclass(frozen=True)
class FoulingOutput:
    """
    Output results from fouling calculations.

    Attributes:
        cleanliness_factor: Current cleanliness factor (0-1)
        fouling_resistance_m2k_w: Calculated fouling resistance (m2-K/W)
        tema_reference_fouling_m2k_w: TEMA standard fouling factor (m2-K/W)
        fouling_severity_ratio: Ratio to TEMA standard
        fouling_rate_m2k_w_day: Estimated fouling rate (m2-K/W per day)
        predicted_cf_30d: Predicted CF in 30 days
        predicted_cf_60d: Predicted CF in 60 days
        predicted_cf_90d: Predicted CF in 90 days
        optimal_cleaning_interval_days: Recommended cleaning interval (days)
        biofouling_risk: Biofouling risk category
        tube_plugging_pct: Percentage of tubes plugged
        plugging_heat_loss_pct: Heat transfer loss from plugging (%)
        total_performance_loss_pct: Combined performance loss (%)
        cleaning_urgency_score: Cleaning urgency (0-100)
        recommended_cleaning_method: Suggested cleaning approach
    """
    cleanliness_factor: float
    fouling_resistance_m2k_w: float
    tema_reference_fouling_m2k_w: float
    fouling_severity_ratio: float
    fouling_rate_m2k_w_day: float
    predicted_cf_30d: float
    predicted_cf_60d: float
    predicted_cf_90d: float
    optimal_cleaning_interval_days: int
    biofouling_risk: str
    tube_plugging_pct: float
    plugging_heat_loss_pct: float
    total_performance_loss_pct: float
    cleaning_urgency_score: float
    recommended_cleaning_method: str


# =============================================================================
# FOULING CALCULATOR CLASS
# =============================================================================

class FoulingCalculator:
    """
    Zero-hallucination fouling calculator for steam condensers.

    Implements deterministic calculations following TEMA Standards and
    HEI guidelines for fouling analysis. All calculations produce
    bit-perfect reproducible results with complete provenance tracking.

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = FoulingCalculator()
        >>> inputs = FoulingInput(
        ...     design_u_value_w_m2k=3500.0,
        ...     actual_u_value_w_m2k=2800.0,
        ...     cw_inlet_temp_c=25.0,
        ...     cw_outlet_temp_c=35.0,
        ...     cw_velocity_m_s=2.0,
        ...     water_type="cooling_tower_water",
        ...     num_tubes_total=5000,
        ...     num_tubes_plugged=50,
        ...     days_since_cleaning=120,
        ...     tube_id_mm=22.0,
        ...     tube_material="admiralty_brass"
        ... )
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"Fouling Resistance: {result.fouling_resistance_m2k_w:.6f}")
    """

    VERSION = "1.0.0"
    NAME = "FoulingCalculator"

    def __init__(self):
        """Initialize the fouling calculator."""
        self._tracker: Optional[ProvenanceTracker] = None

    def calculate(
        self,
        inputs: FoulingInput
    ) -> Tuple[FoulingOutput, ProvenanceRecord]:
        """
        Perform complete fouling analysis.

        Args:
            inputs: FoulingInput with all required parameters

        Returns:
            Tuple of (FoulingOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid or out of range
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["TEMA Standards", "HEI Standards", "EPRI Guidelines"],
                "domain": "Condenser Fouling Analysis"
            }
        )

        # Set inputs for provenance
        input_dict = {
            "design_u_value_w_m2k": inputs.design_u_value_w_m2k,
            "actual_u_value_w_m2k": inputs.actual_u_value_w_m2k,
            "cw_inlet_temp_c": inputs.cw_inlet_temp_c,
            "cw_outlet_temp_c": inputs.cw_outlet_temp_c,
            "cw_velocity_m_s": inputs.cw_velocity_m_s,
            "water_type": inputs.water_type,
            "num_tubes_total": inputs.num_tubes_total,
            "num_tubes_plugged": inputs.num_tubes_plugged,
            "days_since_cleaning": inputs.days_since_cleaning,
            "tube_id_mm": inputs.tube_id_mm,
            "tube_material": inputs.tube_material,
            "water_treatment_quality": inputs.water_treatment_quality
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Step 1: Calculate cleanliness factor
        cleanliness_factor = self._calculate_cleanliness_factor(
            inputs.actual_u_value_w_m2k,
            inputs.design_u_value_w_m2k
        )

        # Step 2: Calculate fouling resistance
        fouling_resistance = self._calculate_fouling_resistance(
            inputs.actual_u_value_w_m2k,
            inputs.design_u_value_w_m2k
        )

        # Step 3: Get TEMA reference fouling factor
        tema_fouling = self._get_tema_fouling_factor(
            inputs.water_type,
            inputs.cw_outlet_temp_c
        )

        # Step 4: Calculate fouling severity ratio
        severity_ratio = self._calculate_severity_ratio(
            fouling_resistance,
            tema_fouling
        )

        # Step 5: Estimate fouling rate
        fouling_rate = self._estimate_fouling_rate(
            fouling_resistance,
            inputs.days_since_cleaning,
            inputs.water_treatment_quality
        )

        # Step 6: Predict future cleanliness factors
        cf_30d, cf_60d, cf_90d = self._predict_future_cf(
            cleanliness_factor,
            fouling_rate,
            inputs.design_u_value_w_m2k
        )

        # Step 7: Calculate optimal cleaning interval
        optimal_interval = self._calculate_optimal_cleaning_interval(
            fouling_rate,
            inputs.design_u_value_w_m2k
        )

        # Step 8: Assess biofouling risk
        biofouling_risk = self._assess_biofouling_risk(
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c,
            inputs.cw_velocity_m_s,
            inputs.water_type
        )

        # Step 9: Calculate tube plugging impact
        plugging_pct, plugging_loss = self._calculate_plugging_impact(
            inputs.num_tubes_plugged,
            inputs.num_tubes_total
        )

        # Step 10: Calculate total performance loss
        total_loss = self._calculate_total_performance_loss(
            cleanliness_factor,
            plugging_loss
        )

        # Step 11: Calculate cleaning urgency
        urgency_score = self._calculate_cleaning_urgency(
            cleanliness_factor,
            biofouling_risk,
            plugging_pct,
            inputs.days_since_cleaning
        )

        # Step 12: Recommend cleaning method
        cleaning_method = self._recommend_cleaning_method(
            biofouling_risk,
            severity_ratio,
            inputs.tube_material
        )

        # Create output
        output = FoulingOutput(
            cleanliness_factor=round(cleanliness_factor, 4),
            fouling_resistance_m2k_w=round(fouling_resistance, 7),
            tema_reference_fouling_m2k_w=round(tema_fouling, 7),
            fouling_severity_ratio=round(severity_ratio, 2),
            fouling_rate_m2k_w_day=round(fouling_rate, 9),
            predicted_cf_30d=round(cf_30d, 4),
            predicted_cf_60d=round(cf_60d, 4),
            predicted_cf_90d=round(cf_90d, 4),
            optimal_cleaning_interval_days=optimal_interval,
            biofouling_risk=biofouling_risk,
            tube_plugging_pct=round(plugging_pct, 2),
            plugging_heat_loss_pct=round(plugging_loss, 2),
            total_performance_loss_pct=round(total_loss, 2),
            cleaning_urgency_score=round(urgency_score, 1),
            recommended_cleaning_method=cleaning_method
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "cleanliness_factor": output.cleanliness_factor,
            "fouling_resistance_m2k_w": output.fouling_resistance_m2k_w,
            "tema_reference_fouling_m2k_w": output.tema_reference_fouling_m2k_w,
            "fouling_severity_ratio": output.fouling_severity_ratio,
            "fouling_rate_m2k_w_day": output.fouling_rate_m2k_w_day,
            "predicted_cf_30d": output.predicted_cf_30d,
            "predicted_cf_60d": output.predicted_cf_60d,
            "predicted_cf_90d": output.predicted_cf_90d,
            "optimal_cleaning_interval_days": output.optimal_cleaning_interval_days,
            "biofouling_risk": output.biofouling_risk,
            "tube_plugging_pct": output.tube_plugging_pct,
            "plugging_heat_loss_pct": output.plugging_heat_loss_pct,
            "total_performance_loss_pct": output.total_performance_loss_pct,
            "cleaning_urgency_score": output.cleaning_urgency_score,
            "recommended_cleaning_method": output.recommended_cleaning_method
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _validate_inputs(self, inputs: FoulingInput) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If any input is invalid
        """
        if inputs.design_u_value_w_m2k <= 0:
            raise ValueError("Design U-value must be positive")

        if inputs.actual_u_value_w_m2k <= 0:
            raise ValueError("Actual U-value must be positive")

        if inputs.actual_u_value_w_m2k > inputs.design_u_value_w_m2k * 1.1:
            raise ValueError(
                "Actual U-value cannot exceed design by more than 10%"
            )

        if inputs.cw_velocity_m_s <= 0 or inputs.cw_velocity_m_s > 5:
            raise ValueError(
                f"CW velocity {inputs.cw_velocity_m_s} m/s out of range (0-5)"
            )

        if inputs.num_tubes_plugged > inputs.num_tubes_total:
            raise ValueError(
                "Plugged tubes cannot exceed total tubes"
            )

        if inputs.water_treatment_quality < 0 or inputs.water_treatment_quality > 1:
            raise ValueError(
                "Water treatment quality must be between 0 and 1"
            )

    def _calculate_cleanliness_factor(
        self,
        actual_u: float,
        design_u: float
    ) -> float:
        """
        Calculate cleanliness factor.

        Formula:
            CF = U_actual / U_design

        Args:
            actual_u: Actual U-value (W/m2-K)
            design_u: Design U-value (W/m2-K)

        Returns:
            Cleanliness factor (0-1)
        """
        cf = actual_u / design_u
        cf = min(cf, 1.0)

        self._tracker.add_step(
            step_number=1,
            description="Calculate cleanliness factor",
            operation="divide",
            inputs={
                "actual_u_value_w_m2k": actual_u,
                "design_u_value_w_m2k": design_u
            },
            output_value=cf,
            output_name="cleanliness_factor",
            formula="CF = U_actual / U_design"
        )

        return cf

    def _calculate_fouling_resistance(
        self,
        actual_u: float,
        design_u: float
    ) -> float:
        """
        Calculate fouling resistance from U-value degradation.

        Formula:
            Rf = (1/U_actual) - (1/U_design)

        This assumes fouling resistance is the primary cause of
        U-value degradation.

        Args:
            actual_u: Actual U-value (W/m2-K)
            design_u: Design U-value (W/m2-K)

        Returns:
            Fouling resistance (m2-K/W)
        """
        rf = (1.0 / actual_u) - (1.0 / design_u)
        rf = max(0, rf)  # Cannot be negative

        self._tracker.add_step(
            step_number=2,
            description="Calculate fouling resistance",
            operation="fouling_resistance",
            inputs={
                "actual_u_value_w_m2k": actual_u,
                "design_u_value_w_m2k": design_u,
                "resistance_actual": 1.0 / actual_u,
                "resistance_design": 1.0 / design_u
            },
            output_value=rf,
            output_name="fouling_resistance_m2k_w",
            formula="Rf = (1/U_actual) - (1/U_design)"
        )

        return rf

    def _get_tema_fouling_factor(
        self,
        water_type: str,
        outlet_temp_c: float
    ) -> float:
        """
        Get TEMA standard fouling factor for water type.

        TEMA provides reference fouling factors based on water source
        and temperature conditions.

        Args:
            water_type: Type of cooling water
            outlet_temp_c: CW outlet temperature (C)

        Returns:
            TEMA fouling factor (m2-K/W)
        """
        water_type_lower = water_type.lower()

        if "seawater" in water_type_lower or "sea" in water_type_lower:
            if outlet_temp_c <= 43:
                tema_rf = TEMA_FOULING_FACTORS["seawater"]["below_43c"]
            else:
                tema_rf = TEMA_FOULING_FACTORS["seawater"]["above_43c"]
        elif "brackish" in water_type_lower:
            if outlet_temp_c <= 43:
                tema_rf = TEMA_FOULING_FACTORS["brackish_water"]["below_43c"]
            else:
                tema_rf = TEMA_FOULING_FACTORS["brackish_water"]["above_43c"]
        elif "river" in water_type_lower:
            tema_rf = TEMA_FOULING_FACTORS["river_water"]["average"]
        elif "tower" in water_type_lower or "cooling" in water_type_lower:
            tema_rf = TEMA_FOULING_FACTORS["cooling_tower_water"]["treated"]
        elif "city" in water_type_lower or "municipal" in water_type_lower:
            tema_rf = TEMA_FOULING_FACTORS["city_water"]
        else:
            # Default to cooling tower treated
            tema_rf = TEMA_FOULING_FACTORS["cooling_tower_water"]["treated"]

        self._tracker.add_step(
            step_number=3,
            description="Get TEMA reference fouling factor",
            operation="lookup",
            inputs={
                "water_type": water_type,
                "outlet_temp_c": outlet_temp_c
            },
            output_value=tema_rf,
            output_name="tema_fouling_factor_m2k_w",
            formula="TEMA lookup table"
        )

        return tema_rf

    def _calculate_severity_ratio(
        self,
        actual_fouling: float,
        tema_fouling: float
    ) -> float:
        """
        Calculate fouling severity ratio compared to TEMA standard.

        Ratio > 1.0 indicates fouling worse than TEMA standard.

        Args:
            actual_fouling: Calculated fouling resistance (m2-K/W)
            tema_fouling: TEMA reference value (m2-K/W)

        Returns:
            Severity ratio (dimensionless)
        """
        if tema_fouling <= 0:
            ratio = 0.0
        else:
            ratio = actual_fouling / tema_fouling

        self._tracker.add_step(
            step_number=4,
            description="Calculate fouling severity ratio",
            operation="divide",
            inputs={
                "actual_fouling_m2k_w": actual_fouling,
                "tema_fouling_m2k_w": tema_fouling
            },
            output_value=ratio,
            output_name="fouling_severity_ratio",
            formula="Severity = Rf_actual / Rf_TEMA"
        )

        return ratio

    def _estimate_fouling_rate(
        self,
        current_fouling: float,
        days_since_cleaning: int,
        treatment_quality: float
    ) -> float:
        """
        Estimate fouling rate from operating history.

        Assumes linear fouling accumulation (conservative estimate).

        Formula:
            Rate = Rf_current / days

        With correction for water treatment quality.

        Args:
            current_fouling: Current fouling resistance (m2-K/W)
            days_since_cleaning: Days since last cleaning
            treatment_quality: Treatment quality (0-1)

        Returns:
            Fouling rate (m2-K/W per day)
        """
        if days_since_cleaning <= 0:
            # Use reference rate for well-treated water
            base_rate = FOULING_RATE_CONSTANTS["low"]
        else:
            # Calculate from history
            base_rate = current_fouling / days_since_cleaning

        # Adjust for treatment quality (better treatment = lower rate)
        treatment_factor = 2.0 - treatment_quality  # 1.0 to 2.0
        adjusted_rate = base_rate * treatment_factor

        self._tracker.add_step(
            step_number=5,
            description="Estimate fouling rate from operating history",
            operation="fouling_rate",
            inputs={
                "current_fouling_m2k_w": current_fouling,
                "days_since_cleaning": days_since_cleaning,
                "treatment_quality": treatment_quality,
                "treatment_factor": treatment_factor
            },
            output_value=adjusted_rate,
            output_name="fouling_rate_m2k_w_day",
            formula="Rate = (Rf / days) * (2 - treatment_quality)"
        )

        return adjusted_rate

    def _predict_future_cf(
        self,
        current_cf: float,
        fouling_rate: float,
        design_u: float
    ) -> Tuple[float, float, float]:
        """
        Predict future cleanliness factors.

        Uses fouling rate to project CF at 30, 60, and 90 days.

        Args:
            current_cf: Current cleanliness factor
            fouling_rate: Fouling rate (m2-K/W per day)
            design_u: Design U-value (W/m2-K)

        Returns:
            Tuple of (CF at 30d, CF at 60d, CF at 90d)
        """
        # Current fouling resistance
        current_rf = (1.0 - current_cf) / (current_cf * design_u)

        # Predict future fouling
        rf_30d = current_rf + (fouling_rate * 30)
        rf_60d = current_rf + (fouling_rate * 60)
        rf_90d = current_rf + (fouling_rate * 90)

        # Convert back to CF
        # CF = 1 / (1 + Rf * U_design)
        cf_30d = 1.0 / (1.0 + rf_30d * design_u)
        cf_60d = 1.0 / (1.0 + rf_60d * design_u)
        cf_90d = 1.0 / (1.0 + rf_90d * design_u)

        # Ensure reasonable bounds
        cf_30d = max(0.5, min(1.0, cf_30d))
        cf_60d = max(0.4, min(1.0, cf_60d))
        cf_90d = max(0.3, min(1.0, cf_90d))

        self._tracker.add_step(
            step_number=6,
            description="Predict future cleanliness factors",
            operation="cf_projection",
            inputs={
                "current_cf": current_cf,
                "fouling_rate_m2k_w_day": fouling_rate,
                "design_u_value_w_m2k": design_u,
                "rf_30d": rf_30d,
                "rf_60d": rf_60d,
                "rf_90d": rf_90d
            },
            output_value={"cf_30d": cf_30d, "cf_60d": cf_60d, "cf_90d": cf_90d},
            output_name="predicted_cf",
            formula="CF = 1 / (1 + Rf * U_design)"
        )

        return cf_30d, cf_60d, cf_90d

    def _calculate_optimal_cleaning_interval(
        self,
        fouling_rate: float,
        design_u: float
    ) -> int:
        """
        Calculate optimal cleaning interval.

        Determines when CF would reach 0.85 (HEI threshold) from clean.

        Formula:
            Days = Rf_target / fouling_rate

        Where Rf_target corresponds to CF = 0.85

        Args:
            fouling_rate: Fouling rate (m2-K/W per day)
            design_u: Design U-value (W/m2-K)

        Returns:
            Optimal cleaning interval (days)
        """
        # Target CF threshold (HEI recommendation)
        target_cf = 0.85

        # Calculate target fouling resistance
        # From CF = 1 / (1 + Rf * U)
        # Rf = (1 - CF) / (CF * U)
        target_rf = (1.0 - target_cf) / (target_cf * design_u)

        # Calculate days to reach target
        if fouling_rate <= 0:
            interval = 365  # Default to annual
        else:
            interval = int(target_rf / fouling_rate)

        # Practical bounds
        interval = max(30, min(365, interval))

        self._tracker.add_step(
            step_number=7,
            description="Calculate optimal cleaning interval",
            operation="interval_calculation",
            inputs={
                "target_cf": target_cf,
                "target_rf_m2k_w": target_rf,
                "fouling_rate_m2k_w_day": fouling_rate
            },
            output_value=interval,
            output_name="optimal_cleaning_interval_days",
            formula="Days = Rf_target / Rate"
        )

        return interval

    def _assess_biofouling_risk(
        self,
        inlet_temp_c: float,
        outlet_temp_c: float,
        velocity_m_s: float,
        water_type: str
    ) -> str:
        """
        Assess biofouling risk based on operating conditions.

        Factors:
        - Temperature (biological growth rate increases with temp)
        - Velocity (low velocity promotes attachment)
        - Water source (some sources more prone to biofouling)

        Args:
            inlet_temp_c: CW inlet temperature (C)
            outlet_temp_c: CW outlet temperature (C)
            velocity_m_s: Tube velocity (m/s)
            water_type: Water source type

        Returns:
            Risk category string
        """
        risk_score = 0
        avg_temp = (inlet_temp_c + outlet_temp_c) / 2

        # Temperature risk
        if avg_temp > 35:
            risk_score += 3
        elif avg_temp > 25:
            risk_score += 2
        else:
            risk_score += 1

        # Velocity risk (lower is worse)
        if velocity_m_s < 1.0:
            risk_score += 3
        elif velocity_m_s < 2.0:
            risk_score += 2
        else:
            risk_score += 1

        # Water type risk
        water_lower = water_type.lower()
        if "river" in water_lower or "lake" in water_lower:
            risk_score += 2
        elif "tower" in water_lower:
            risk_score += 2
        elif "sea" in water_lower:
            risk_score += 1
        else:
            risk_score += 1

        # Determine risk category
        if risk_score >= 7:
            risk = BiofoulingRisk.CRITICAL.value
        elif risk_score >= 5:
            risk = BiofoulingRisk.HIGH.value
        elif risk_score >= 4:
            risk = BiofoulingRisk.MODERATE.value
        else:
            risk = BiofoulingRisk.LOW.value

        self._tracker.add_step(
            step_number=8,
            description="Assess biofouling risk",
            operation="risk_assessment",
            inputs={
                "avg_temp_c": avg_temp,
                "velocity_m_s": velocity_m_s,
                "water_type": water_type,
                "risk_score": risk_score
            },
            output_value=risk,
            output_name="biofouling_risk",
            formula="Score-based risk classification"
        )

        return risk

    def _calculate_plugging_impact(
        self,
        tubes_plugged: int,
        tubes_total: int
    ) -> Tuple[float, float]:
        """
        Calculate impact of tube plugging on performance.

        Plugged tubes reduce heat transfer area and increase
        velocity in remaining tubes.

        Args:
            tubes_plugged: Number of plugged tubes
            tubes_total: Total number of tubes

        Returns:
            Tuple of (plugging percentage, heat transfer loss %)
        """
        plugging_pct = (tubes_plugged / tubes_total) * 100.0

        # Heat transfer loss is approximately proportional to plugging
        # but slightly worse due to flow redistribution
        heat_loss_pct = plugging_pct * 1.1  # 10% penalty

        self._tracker.add_step(
            step_number=9,
            description="Calculate tube plugging impact",
            operation="plugging_analysis",
            inputs={
                "tubes_plugged": tubes_plugged,
                "tubes_total": tubes_total
            },
            output_value={"plugging_pct": plugging_pct, "heat_loss_pct": heat_loss_pct},
            output_name="plugging_impact",
            formula="Loss% = (plugged/total) * 100 * 1.1"
        )

        return plugging_pct, heat_loss_pct

    def _calculate_total_performance_loss(
        self,
        cleanliness_factor: float,
        plugging_loss_pct: float
    ) -> float:
        """
        Calculate total performance loss from all factors.

        Combines fouling (CF) and plugging impacts.

        Args:
            cleanliness_factor: Current CF (0-1)
            plugging_loss_pct: Heat loss from plugging (%)

        Returns:
            Total performance loss (%)
        """
        # Loss from fouling
        fouling_loss_pct = (1.0 - cleanliness_factor) * 100.0

        # Combined loss (not simply additive)
        # Using multiplicative model
        remaining_performance = cleanliness_factor * (1.0 - plugging_loss_pct / 100.0)
        total_loss = (1.0 - remaining_performance) * 100.0

        self._tracker.add_step(
            step_number=10,
            description="Calculate total performance loss",
            operation="combined_loss",
            inputs={
                "cleanliness_factor": cleanliness_factor,
                "fouling_loss_pct": fouling_loss_pct,
                "plugging_loss_pct": plugging_loss_pct,
                "remaining_performance": remaining_performance
            },
            output_value=total_loss,
            output_name="total_performance_loss_pct",
            formula="Loss = (1 - CF * (1 - plug_loss/100)) * 100"
        )

        return total_loss

    def _calculate_cleaning_urgency(
        self,
        cleanliness_factor: float,
        biofouling_risk: str,
        plugging_pct: float,
        days_since_cleaning: int
    ) -> float:
        """
        Calculate cleaning urgency score (0-100).

        Higher scores indicate more urgent need for cleaning.

        Args:
            cleanliness_factor: Current CF
            biofouling_risk: Biofouling risk category
            plugging_pct: Tube plugging percentage
            days_since_cleaning: Days since last cleaning

        Returns:
            Urgency score (0-100)
        """
        score = 0.0

        # CF contribution (0-40 points)
        # CF = 0.85 (threshold) = 20 points, CF = 0.65 = 40 points
        cf_score = max(0, (0.85 - cleanliness_factor) / 0.20) * 40
        score += min(40, cf_score)

        # Biofouling risk contribution (0-25 points)
        risk_scores = {
            BiofoulingRisk.LOW.value: 5,
            BiofoulingRisk.MODERATE.value: 10,
            BiofoulingRisk.HIGH.value: 20,
            BiofoulingRisk.CRITICAL.value: 25,
        }
        score += risk_scores.get(biofouling_risk, 10)

        # Plugging contribution (0-20 points)
        plug_score = min(20, plugging_pct * 2)
        score += plug_score

        # Time contribution (0-15 points)
        time_score = min(15, days_since_cleaning / 20)
        score += time_score

        score = min(100, max(0, score))

        self._tracker.add_step(
            step_number=11,
            description="Calculate cleaning urgency score",
            operation="urgency_scoring",
            inputs={
                "cleanliness_factor": cleanliness_factor,
                "cf_score": cf_score,
                "biofouling_risk": biofouling_risk,
                "plugging_pct": plugging_pct,
                "days_since_cleaning": days_since_cleaning
            },
            output_value=score,
            output_name="cleaning_urgency_score",
            formula="Score = CF_contrib + Risk_contrib + Plug_contrib + Time_contrib"
        )

        return score

    def _recommend_cleaning_method(
        self,
        biofouling_risk: str,
        severity_ratio: float,
        tube_material: str
    ) -> str:
        """
        Recommend cleaning method based on conditions.

        Args:
            biofouling_risk: Biofouling risk category
            severity_ratio: Fouling severity ratio
            tube_material: Tube material type

        Returns:
            Recommended cleaning method
        """
        # High biofouling risk or severe fouling needs aggressive cleaning
        if biofouling_risk in [BiofoulingRisk.HIGH.value, BiofoulingRisk.CRITICAL.value]:
            if "stainless" in tube_material.lower():
                method = "chemical_cleaning"
            else:
                method = "mechanical_brushing"
        elif severity_ratio > 2.0:
            method = "mechanical_brushing"
        elif severity_ratio > 1.5:
            method = "high_pressure_water"
        else:
            method = "online_ball_cleaning"

        self._tracker.add_step(
            step_number=12,
            description="Recommend cleaning method",
            operation="method_selection",
            inputs={
                "biofouling_risk": biofouling_risk,
                "severity_ratio": severity_ratio,
                "tube_material": tube_material
            },
            output_value=method,
            output_name="recommended_cleaning_method",
            formula="Rule-based selection"
        )

        return method


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_asymptotic_fouling(
    time_days: float,
    fouling_rate: float,
    asymptotic_limit: float
) -> float:
    """
    Calculate fouling using asymptotic model.

    The asymptotic model accounts for fouling approaching a limit
    as deposit removal balances deposition.

    Formula:
        Rf(t) = Rf_inf * (1 - exp(-beta * t))

    Where beta = initial_rate / Rf_inf

    Args:
        time_days: Time since cleaning (days)
        fouling_rate: Initial fouling rate (m2-K/W per day)
        asymptotic_limit: Maximum fouling resistance (m2-K/W)

    Returns:
        Fouling resistance at time t (m2-K/W)
    """
    if asymptotic_limit <= 0:
        return fouling_rate * time_days

    beta = fouling_rate / asymptotic_limit
    rf = asymptotic_limit * (1 - math.exp(-beta * time_days))

    return rf


def calculate_cleaning_benefit(
    current_cf: float,
    cleaning_method: str,
    tube_condition: str = "average"
) -> float:
    """
    Calculate expected cleanliness factor after cleaning.

    Args:
        current_cf: Current cleanliness factor
        cleaning_method: Cleaning method name
        tube_condition: Tube condition (good/average/poor)

    Returns:
        Expected CF after cleaning
    """
    effectiveness = CLEANING_EFFECTIVENESS.get(cleaning_method, 0.8)

    # Condition factor
    condition_factors = {
        "good": 1.0,
        "average": 0.95,
        "poor": 0.85,
    }
    condition_factor = condition_factors.get(tube_condition, 0.95)

    # Maximum achievable CF
    max_cf = effectiveness * condition_factor

    # Post-cleaning CF
    improvement = (max_cf - current_cf) * effectiveness
    post_cf = current_cf + improvement

    return min(max_cf, post_cf)


def estimate_cleaning_cost(
    num_tubes: int,
    cleaning_method: str,
    tube_length_m: float,
    labor_rate_usd_hr: float = 75.0
) -> float:
    """
    Estimate cleaning cost.

    Args:
        num_tubes: Number of tubes to clean
        cleaning_method: Cleaning method
        tube_length_m: Tube length (m)
        labor_rate_usd_hr: Labor rate (USD/hr)

    Returns:
        Estimated cost (USD)
    """
    # Time per tube (hours) by method
    time_per_tube = {
        "mechanical_brushing": 0.05,
        "chemical_cleaning": 0.02,  # Plus chemical cost
        "high_pressure_water": 0.03,
        "online_ball_cleaning": 0.01,
        "thermal_shock": 0.01,
    }

    base_time = time_per_tube.get(cleaning_method, 0.03)

    # Adjust for tube length
    length_factor = tube_length_m / 10.0  # Reference 10m tube

    total_time_hr = num_tubes * base_time * length_factor
    labor_cost = total_time_hr * labor_rate_usd_hr

    # Add materials/chemicals cost
    if cleaning_method == "chemical_cleaning":
        chemical_cost = num_tubes * 5.0  # $5 per tube
    else:
        chemical_cost = 0

    return labor_cost + chemical_cost
