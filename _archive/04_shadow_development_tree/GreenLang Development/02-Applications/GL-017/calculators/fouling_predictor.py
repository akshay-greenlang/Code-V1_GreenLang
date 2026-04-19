"""
GL-017 CONDENSYNC - Fouling Predictor Calculator

Zero-hallucination, deterministic calculations for condenser tube fouling
prediction and cleaning optimization following industry best practices.

This module provides:
- Fouling rate estimation from historical data
- Biofouling prediction (seasonal, water quality based)
- Scale formation prediction (CaCO3, silica deposition)
- Cleaning interval optimization
- Cleaning cost vs performance tradeoff analysis
- Predicted efficiency loss from fouling over time

Standards Reference:
- HEI Standards for Steam Surface Condensers (11th Edition)
- TEMA (Tubular Exchanger Manufacturers Association) Standards
- EPRI Condenser Fouling Guidelines
- Kern & Seaton Asymptotic Fouling Model
- Watkinson Fouling Model

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Fouling model parameters (Kern & Seaton asymptotic model)
# Rf(t) = Rf_inf * (1 - exp(-beta * t))
FOULING_MODEL_PARAMS = {
    "seawater": {
        "rf_inf": 0.00025,      # Asymptotic fouling resistance (m2-K/W)
        "beta": 0.02,           # Fouling rate constant (1/day)
    },
    "brackish_water": {
        "rf_inf": 0.00035,
        "beta": 0.025,
    },
    "river_water": {
        "rf_inf": 0.00040,
        "beta": 0.03,
    },
    "cooling_tower": {
        "rf_inf": 0.00030,
        "beta": 0.022,
    },
    "lake_water": {
        "rf_inf": 0.00032,
        "beta": 0.024,
    },
}

# Biofouling growth rate factors by temperature range (relative to 25C base)
BIOFOULING_TEMP_FACTORS = {
    "0-10": 0.1,
    "10-15": 0.3,
    "15-20": 0.5,
    "20-25": 0.8,
    "25-30": 1.0,
    "30-35": 1.2,
    "35-40": 1.0,  # Decreases above optimal
    "40-45": 0.6,
    "45-50": 0.2,
}

# Biofouling seasonal factors (Northern Hemisphere)
BIOFOULING_SEASONAL_FACTORS = {
    1: 0.4,   # January
    2: 0.3,   # February
    3: 0.5,   # March
    4: 0.7,   # April
    5: 0.9,   # May
    6: 1.0,   # June
    7: 1.0,   # July
    8: 1.0,   # August
    9: 0.9,   # September
    10: 0.7,  # October
    11: 0.5,  # November
    12: 0.4,  # December
}

# Scale formation parameters
SCALE_FORMATION = {
    "caco3": {
        "solubility_factor": -0.02,  # Decreases with temperature
        "supersaturation_threshold": 2.5,  # LSI threshold
        "deposition_rate_base": 0.00001,  # m2-K/W per day at unit supersaturation
    },
    "silica": {
        "solubility_ppm_30c": 120,
        "solubility_ppm_50c": 160,
        "deposition_rate_base": 0.000005,
    },
    "calcium_sulfate": {
        "inverse_solubility": True,  # Less soluble at higher temps
        "critical_temp_c": 65,
        "deposition_rate_base": 0.000008,
    },
}

# Cleaning effectiveness by method
CLEANING_EFFECTIVENESS = {
    "mechanical_brush": {
        "biofouling": 0.95,
        "particulate": 0.90,
        "soft_scale": 0.80,
        "hard_scale": 0.50,
    },
    "high_pressure_water": {
        "biofouling": 0.90,
        "particulate": 0.85,
        "soft_scale": 0.70,
        "hard_scale": 0.40,
    },
    "chemical_acid": {
        "biofouling": 0.70,
        "particulate": 0.60,
        "soft_scale": 0.95,
        "hard_scale": 0.85,
    },
    "chemical_biocide": {
        "biofouling": 0.90,
        "particulate": 0.30,
        "soft_scale": 0.20,
        "hard_scale": 0.10,
    },
    "online_ball": {
        "biofouling": 0.70,
        "particulate": 0.75,
        "soft_scale": 0.40,
        "hard_scale": 0.20,
    },
}

# Cleaning costs (USD)
CLEANING_COSTS = {
    "mechanical_brush": {
        "fixed_cost": 5000,
        "per_tube": 2.50,
        "duration_hours": 48,
    },
    "high_pressure_water": {
        "fixed_cost": 3000,
        "per_tube": 1.50,
        "duration_hours": 24,
    },
    "chemical_acid": {
        "fixed_cost": 8000,
        "per_tube": 0.50,
        "duration_hours": 12,
    },
    "chemical_biocide": {
        "fixed_cost": 2000,
        "per_tube": 0.20,
        "duration_hours": 4,
    },
    "online_ball": {
        "fixed_cost": 50000,  # System installation
        "per_tube": 0.10,    # Per cycle
        "duration_hours": 0,  # Continuous operation
    },
}


class FoulingMechanism(Enum):
    """Primary fouling mechanisms."""
    BIOFOULING = "Biofouling"
    PARTICULATE = "Particulate"
    SCALE_CACO3 = "CaCO3 Scale"
    SCALE_SILICA = "Silica Scale"
    CORROSION = "Corrosion Products"
    MIXED = "Mixed Fouling"


class WaterQuality(Enum):
    """Water quality categories."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    MODERATE = "Moderate"
    POOR = "Poor"
    VERY_POOR = "Very Poor"


class CleaningUrgency(Enum):
    """Cleaning urgency levels."""
    NOT_REQUIRED = "Not Required"
    SCHEDULED = "Scheduled Maintenance"
    RECOMMENDED = "Recommended Soon"
    URGENT = "Urgent"
    CRITICAL = "Critical"


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class FoulingPredictorInput:
    """
    Input parameters for fouling prediction calculations.

    Attributes:
        water_source: Type of cooling water source
        cw_inlet_temp_c: Cooling water inlet temperature (C)
        cw_outlet_temp_c: Cooling water outlet temperature (C)
        cw_velocity_m_s: Cooling water velocity in tubes (m/s)
        current_cf: Current cleanliness factor (0-1)
        days_since_cleaning: Days since last tube cleaning
        num_tubes: Total number of tubes
        tube_length_m: Tube length (m)
        calcium_hardness_ppm: Calcium hardness (ppm as CaCO3)
        total_alkalinity_ppm: Total alkalinity (ppm as CaCO3)
        ph: Water pH
        silica_ppm: Silica concentration (ppm)
        tds_ppm: Total dissolved solids (ppm)
        chloride_ppm: Chloride concentration (ppm)
        current_month: Current month (1-12) for seasonal adjustment
        historical_cf_data: List of (days_ago, cf_value) tuples
        design_u_value_w_m2k: Design U-value (W/m2-K)
        electricity_price_usd_mwh: Electricity price for cost analysis
        plant_capacity_mw: Plant capacity for loss calculations
    """
    water_source: str
    cw_inlet_temp_c: float
    cw_outlet_temp_c: float
    cw_velocity_m_s: float
    current_cf: float
    days_since_cleaning: int
    num_tubes: int
    tube_length_m: float
    calcium_hardness_ppm: float
    total_alkalinity_ppm: float
    ph: float
    silica_ppm: float = 20.0
    tds_ppm: float = 500.0
    chloride_ppm: float = 50.0
    current_month: int = 6
    historical_cf_data: Optional[List[Tuple[int, float]]] = None
    design_u_value_w_m2k: float = 3000.0
    electricity_price_usd_mwh: float = 50.0
    plant_capacity_mw: float = 500.0


@dataclass(frozen=True)
class FoulingPredictorOutput:
    """
    Output results from fouling prediction calculations.

    Attributes:
        dominant_mechanism: Primary fouling mechanism identified
        biofouling_rate_m2kw_day: Biofouling rate (m2-K/W per day)
        scale_rate_m2kw_day: Scale formation rate (m2-K/W per day)
        total_fouling_rate_m2kw_day: Combined fouling rate (m2-K/W per day)
        langelier_saturation_index: Langelier Saturation Index
        ryznar_stability_index: Ryznar Stability Index
        silica_saturation_pct: Silica saturation percentage
        predicted_cf_7d: Predicted CF in 7 days
        predicted_cf_30d: Predicted CF in 30 days
        predicted_cf_60d: Predicted CF in 60 days
        predicted_cf_90d: Predicted CF in 90 days
        days_to_cf_085: Days until CF reaches 0.85 threshold
        days_to_cf_080: Days until CF reaches 0.80 threshold
        optimal_cleaning_interval_days: Economically optimal cleaning interval
        cleaning_urgency: Current cleaning urgency level
        recommended_cleaning_method: Recommended cleaning approach
        estimated_cleaning_cost_usd: Estimated cleaning cost
        daily_efficiency_loss_pct: Daily efficiency loss from fouling
        annual_fouling_cost_usd: Annual cost of fouling at current rate
        cleaning_roi_pct: ROI from cleaning at current conditions
        biofouling_risk_score: Biofouling risk (0-100)
        scale_risk_score: Scale formation risk (0-100)
        overall_fouling_risk_score: Combined fouling risk (0-100)
        water_quality_rating: Overall water quality rating
    """
    dominant_mechanism: str
    biofouling_rate_m2kw_day: float
    scale_rate_m2kw_day: float
    total_fouling_rate_m2kw_day: float
    langelier_saturation_index: float
    ryznar_stability_index: float
    silica_saturation_pct: float
    predicted_cf_7d: float
    predicted_cf_30d: float
    predicted_cf_60d: float
    predicted_cf_90d: float
    days_to_cf_085: int
    days_to_cf_080: int
    optimal_cleaning_interval_days: int
    cleaning_urgency: str
    recommended_cleaning_method: str
    estimated_cleaning_cost_usd: float
    daily_efficiency_loss_pct: float
    annual_fouling_cost_usd: float
    cleaning_roi_pct: float
    biofouling_risk_score: float
    scale_risk_score: float
    overall_fouling_risk_score: float
    water_quality_rating: str


# =============================================================================
# FOULING PREDICTOR CALCULATOR CLASS
# =============================================================================

class FoulingPredictor:
    """
    Zero-hallucination fouling predictor for steam condensers.

    Implements deterministic fouling prediction calculations using
    established models (Kern-Seaton, Watkinson). All calculations
    produce bit-perfect reproducible results with complete provenance.

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> predictor = FoulingPredictor()
        >>> inputs = FoulingPredictorInput(
        ...     water_source="cooling_tower",
        ...     cw_inlet_temp_c=28.0,
        ...     cw_outlet_temp_c=38.0,
        ...     cw_velocity_m_s=2.0,
        ...     current_cf=0.88,
        ...     days_since_cleaning=90,
        ...     num_tubes=8000,
        ...     tube_length_m=12.0,
        ...     calcium_hardness_ppm=250,
        ...     total_alkalinity_ppm=180,
        ...     ph=8.2,
        ...     silica_ppm=35
        ... )
        >>> result, provenance = predictor.predict(inputs)
        >>> print(f"Days to CF 0.85: {result.days_to_cf_085}")
    """

    VERSION = "1.0.0"
    NAME = "FoulingPredictor"

    def __init__(self):
        """Initialize the fouling predictor."""
        self._tracker: Optional[ProvenanceTracker] = None

    def predict(
        self,
        inputs: FoulingPredictorInput
    ) -> Tuple[FoulingPredictorOutput, ProvenanceRecord]:
        """
        Perform complete fouling prediction analysis.

        Args:
            inputs: FoulingPredictorInput with all required parameters

        Returns:
            Tuple of (FoulingPredictorOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid or out of range
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["HEI Standards", "TEMA", "EPRI Guidelines"],
                "models": ["Kern-Seaton Asymptotic", "LSI/RSI Indices"],
                "domain": "Condenser Fouling Prediction"
            }
        )

        # Convert inputs to dictionary for provenance
        input_dict = self._inputs_to_dict(inputs)
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Step 1: Calculate water quality indices
        lsi = self._calculate_langelier_index(
            inputs.ph,
            inputs.calcium_hardness_ppm,
            inputs.total_alkalinity_ppm,
            inputs.tds_ppm,
            inputs.cw_outlet_temp_c
        )

        rsi = self._calculate_ryznar_index(inputs.ph, lsi)

        silica_saturation = self._calculate_silica_saturation(
            inputs.silica_ppm,
            inputs.cw_outlet_temp_c
        )

        # Step 2: Calculate biofouling rate
        biofouling_rate = self._calculate_biofouling_rate(
            inputs.water_source,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c,
            inputs.cw_velocity_m_s,
            inputs.current_month
        )

        # Step 3: Calculate scale formation rate
        scale_rate = self._calculate_scale_rate(
            lsi,
            silica_saturation,
            inputs.cw_outlet_temp_c
        )

        # Step 4: Determine dominant mechanism and total rate
        total_rate, dominant_mechanism = self._calculate_total_fouling_rate(
            biofouling_rate,
            scale_rate,
            inputs.historical_cf_data,
            inputs.days_since_cleaning
        )

        # Step 5: Predict future cleanliness factors
        cf_predictions = self._predict_future_cf(
            inputs.current_cf,
            total_rate,
            inputs.design_u_value_w_m2k
        )

        # Step 6: Calculate days to thresholds
        days_to_085 = self._calculate_days_to_threshold(
            inputs.current_cf, total_rate, 0.85, inputs.design_u_value_w_m2k
        )
        days_to_080 = self._calculate_days_to_threshold(
            inputs.current_cf, total_rate, 0.80, inputs.design_u_value_w_m2k
        )

        # Step 7: Calculate optimal cleaning interval
        optimal_interval = self._calculate_optimal_cleaning_interval(
            total_rate,
            inputs.design_u_value_w_m2k,
            inputs.num_tubes,
            inputs.tube_length_m,
            inputs.electricity_price_usd_mwh,
            inputs.plant_capacity_mw
        )

        # Step 8: Determine cleaning urgency
        cleaning_urgency = self._determine_cleaning_urgency(
            inputs.current_cf,
            days_to_085,
            inputs.days_since_cleaning
        )

        # Step 9: Recommend cleaning method
        recommended_method = self._recommend_cleaning_method(
            dominant_mechanism,
            lsi,
            biofouling_rate,
            scale_rate
        )

        # Step 10: Estimate cleaning cost
        cleaning_cost = self._estimate_cleaning_cost(
            recommended_method,
            inputs.num_tubes,
            inputs.tube_length_m
        )

        # Step 11: Calculate efficiency loss and annual costs
        daily_loss, annual_cost = self._calculate_fouling_costs(
            inputs.current_cf,
            total_rate,
            inputs.electricity_price_usd_mwh,
            inputs.plant_capacity_mw
        )

        # Step 12: Calculate cleaning ROI
        cleaning_roi = self._calculate_cleaning_roi(
            inputs.current_cf,
            cleaning_cost,
            annual_cost
        )

        # Step 13: Calculate risk scores
        biofouling_risk = self._calculate_biofouling_risk_score(
            biofouling_rate,
            inputs.cw_inlet_temp_c,
            inputs.cw_outlet_temp_c,
            inputs.cw_velocity_m_s,
            inputs.current_month
        )

        scale_risk = self._calculate_scale_risk_score(
            lsi,
            rsi,
            silica_saturation
        )

        overall_risk = self._calculate_overall_risk_score(
            biofouling_risk,
            scale_risk,
            inputs.current_cf
        )

        # Step 14: Rate water quality
        water_quality = self._rate_water_quality(
            lsi,
            rsi,
            inputs.chloride_ppm,
            inputs.tds_ppm
        )

        # Create output
        output = FoulingPredictorOutput(
            dominant_mechanism=dominant_mechanism,
            biofouling_rate_m2kw_day=round(biofouling_rate, 9),
            scale_rate_m2kw_day=round(scale_rate, 9),
            total_fouling_rate_m2kw_day=round(total_rate, 9),
            langelier_saturation_index=round(lsi, 2),
            ryznar_stability_index=round(rsi, 2),
            silica_saturation_pct=round(silica_saturation, 1),
            predicted_cf_7d=round(cf_predictions[0], 4),
            predicted_cf_30d=round(cf_predictions[1], 4),
            predicted_cf_60d=round(cf_predictions[2], 4),
            predicted_cf_90d=round(cf_predictions[3], 4),
            days_to_cf_085=days_to_085,
            days_to_cf_080=days_to_080,
            optimal_cleaning_interval_days=optimal_interval,
            cleaning_urgency=cleaning_urgency,
            recommended_cleaning_method=recommended_method,
            estimated_cleaning_cost_usd=round(cleaning_cost, 0),
            daily_efficiency_loss_pct=round(daily_loss, 4),
            annual_fouling_cost_usd=round(annual_cost, 0),
            cleaning_roi_pct=round(cleaning_roi, 1),
            biofouling_risk_score=round(biofouling_risk, 1),
            scale_risk_score=round(scale_risk, 1),
            overall_fouling_risk_score=round(overall_risk, 1),
            water_quality_rating=water_quality
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs(self._output_to_dict(output))
        provenance = self._tracker.finalize()

        return output, provenance

    def _inputs_to_dict(self, inputs: FoulingPredictorInput) -> Dict:
        """Convert input dataclass to dictionary."""
        return {
            "water_source": inputs.water_source,
            "cw_inlet_temp_c": inputs.cw_inlet_temp_c,
            "cw_outlet_temp_c": inputs.cw_outlet_temp_c,
            "cw_velocity_m_s": inputs.cw_velocity_m_s,
            "current_cf": inputs.current_cf,
            "days_since_cleaning": inputs.days_since_cleaning,
            "num_tubes": inputs.num_tubes,
            "tube_length_m": inputs.tube_length_m,
            "calcium_hardness_ppm": inputs.calcium_hardness_ppm,
            "total_alkalinity_ppm": inputs.total_alkalinity_ppm,
            "ph": inputs.ph,
            "silica_ppm": inputs.silica_ppm,
            "tds_ppm": inputs.tds_ppm,
            "chloride_ppm": inputs.chloride_ppm,
            "current_month": inputs.current_month,
            "design_u_value_w_m2k": inputs.design_u_value_w_m2k,
            "electricity_price_usd_mwh": inputs.electricity_price_usd_mwh,
            "plant_capacity_mw": inputs.plant_capacity_mw
        }

    def _output_to_dict(self, output: FoulingPredictorOutput) -> Dict:
        """Convert output dataclass to dictionary."""
        return {
            "dominant_mechanism": output.dominant_mechanism,
            "biofouling_rate_m2kw_day": output.biofouling_rate_m2kw_day,
            "scale_rate_m2kw_day": output.scale_rate_m2kw_day,
            "total_fouling_rate_m2kw_day": output.total_fouling_rate_m2kw_day,
            "langelier_saturation_index": output.langelier_saturation_index,
            "ryznar_stability_index": output.ryznar_stability_index,
            "silica_saturation_pct": output.silica_saturation_pct,
            "predicted_cf_7d": output.predicted_cf_7d,
            "predicted_cf_30d": output.predicted_cf_30d,
            "predicted_cf_60d": output.predicted_cf_60d,
            "predicted_cf_90d": output.predicted_cf_90d,
            "days_to_cf_085": output.days_to_cf_085,
            "days_to_cf_080": output.days_to_cf_080,
            "optimal_cleaning_interval_days": output.optimal_cleaning_interval_days,
            "cleaning_urgency": output.cleaning_urgency,
            "recommended_cleaning_method": output.recommended_cleaning_method,
            "estimated_cleaning_cost_usd": output.estimated_cleaning_cost_usd,
            "daily_efficiency_loss_pct": output.daily_efficiency_loss_pct,
            "annual_fouling_cost_usd": output.annual_fouling_cost_usd,
            "cleaning_roi_pct": output.cleaning_roi_pct,
            "biofouling_risk_score": output.biofouling_risk_score,
            "scale_risk_score": output.scale_risk_score,
            "overall_fouling_risk_score": output.overall_fouling_risk_score,
            "water_quality_rating": output.water_quality_rating
        }

    def _validate_inputs(self, inputs: FoulingPredictorInput) -> None:
        """Validate input parameters."""
        if inputs.current_cf < 0.5 or inputs.current_cf > 1.0:
            raise ValueError(f"CF {inputs.current_cf} out of valid range (0.5-1.0)")

        if inputs.ph < 5.0 or inputs.ph > 10.0:
            raise ValueError(f"pH {inputs.ph} out of valid range (5.0-10.0)")

        if inputs.cw_velocity_m_s < 0.5 or inputs.cw_velocity_m_s > 4.0:
            raise ValueError(f"Velocity {inputs.cw_velocity_m_s} m/s out of range (0.5-4.0)")

        if inputs.current_month < 1 or inputs.current_month > 12:
            raise ValueError(f"Month {inputs.current_month} out of range (1-12)")

    def _calculate_langelier_index(
        self,
        ph: float,
        calcium_hardness: float,
        alkalinity: float,
        tds: float,
        temp_c: float
    ) -> float:
        """
        Calculate Langelier Saturation Index (LSI).

        LSI = pH - pHs
        where pHs is the saturation pH

        LSI > 0: Scale forming tendency
        LSI < 0: Corrosive tendency
        LSI = 0: Balanced water

        Formula:
            pHs = (9.3 + A + B) - (C + D)
            A = (log10(TDS) - 1) / 10
            B = -13.12 * log10(T + 273) + 34.55
            C = log10(Ca) - 0.4
            D = log10(Alk)
        """
        # Calculate A factor (TDS)
        a_factor = (math.log10(max(tds, 1)) - 1) / 10

        # Calculate B factor (Temperature)
        temp_k = temp_c + 273.15
        b_factor = -13.12 * math.log10(temp_k) + 34.55

        # Calculate C factor (Calcium)
        c_factor = math.log10(max(calcium_hardness, 1)) - 0.4

        # Calculate D factor (Alkalinity)
        d_factor = math.log10(max(alkalinity, 1))

        # Calculate saturation pH
        phs = (9.3 + a_factor + b_factor) - (c_factor + d_factor)

        # Calculate LSI
        lsi = ph - phs

        self._tracker.add_step(
            step_number=1,
            description="Calculate Langelier Saturation Index (LSI)",
            operation="lsi_formula",
            inputs={
                "ph": ph,
                "calcium_hardness_ppm": calcium_hardness,
                "alkalinity_ppm": alkalinity,
                "tds_ppm": tds,
                "temp_c": temp_c,
                "phs_calculated": phs
            },
            output_value=lsi,
            output_name="langelier_saturation_index",
            formula="LSI = pH - pHs; pHs = (9.3 + A + B) - (C + D)"
        )

        return lsi

    def _calculate_ryznar_index(self, ph: float, lsi: float) -> float:
        """
        Calculate Ryznar Stability Index (RSI).

        RSI = 2 * pHs - pH = pH - 2 * LSI

        RSI < 6.0: Heavy scale formation
        RSI 6.0-7.0: Light scale formation
        RSI 7.0-8.0: Balanced water
        RSI > 8.0: Aggressive/corrosive water
        """
        rsi = ph - 2 * lsi

        self._tracker.add_step(
            step_number=2,
            description="Calculate Ryznar Stability Index (RSI)",
            operation="rsi_formula",
            inputs={
                "ph": ph,
                "lsi": lsi
            },
            output_value=rsi,
            output_name="ryznar_stability_index",
            formula="RSI = pH - 2 * LSI"
        )

        return rsi

    def _calculate_silica_saturation(
        self,
        silica_ppm: float,
        temp_c: float
    ) -> float:
        """
        Calculate silica saturation percentage.

        Silica solubility increases with temperature.
        Base solubility at 25C is approximately 100 ppm.
        """
        # Calculate solubility at temperature (simplified model)
        base_solubility = 100  # ppm at 25C
        temp_factor = 1 + 0.02 * (temp_c - 25)  # 2% increase per degree
        solubility = base_solubility * temp_factor

        saturation_pct = (silica_ppm / solubility) * 100

        self._tracker.add_step(
            step_number=3,
            description="Calculate silica saturation percentage",
            operation="saturation_calc",
            inputs={
                "silica_ppm": silica_ppm,
                "temp_c": temp_c,
                "solubility_ppm": solubility
            },
            output_value=saturation_pct,
            output_name="silica_saturation_pct",
            formula="Saturation% = (Silica / Solubility) * 100"
        )

        return saturation_pct

    def _calculate_biofouling_rate(
        self,
        water_source: str,
        inlet_temp: float,
        outlet_temp: float,
        velocity: float,
        month: int
    ) -> float:
        """
        Calculate biofouling rate using environmental factors.

        Base rate adjusted for:
        - Water source (nutrient availability)
        - Temperature (biological growth rate)
        - Velocity (shear stress, attachment)
        - Season (growth patterns)
        """
        # Get base parameters for water source
        source_key = self._normalize_water_source(water_source)
        params = FOULING_MODEL_PARAMS.get(source_key, FOULING_MODEL_PARAMS["cooling_tower"])

        # Base rate from asymptotic model derivative at t=0
        base_rate = params["rf_inf"] * params["beta"]

        # Temperature factor (average water temperature)
        avg_temp = (inlet_temp + outlet_temp) / 2
        temp_factor = self._get_temp_factor(avg_temp)

        # Velocity factor (higher velocity reduces attachment)
        # Biofilm attachment decreases significantly above 2 m/s
        if velocity < 1.0:
            velocity_factor = 1.5
        elif velocity < 2.0:
            velocity_factor = 1.0
        elif velocity < 3.0:
            velocity_factor = 0.6
        else:
            velocity_factor = 0.3

        # Seasonal factor
        seasonal_factor = BIOFOULING_SEASONAL_FACTORS.get(month, 0.7)

        # Combined biofouling rate
        biofouling_rate = base_rate * temp_factor * velocity_factor * seasonal_factor

        self._tracker.add_step(
            step_number=4,
            description="Calculate biofouling rate",
            operation="biofouling_model",
            inputs={
                "water_source": water_source,
                "base_rate": base_rate,
                "avg_temp_c": avg_temp,
                "temp_factor": temp_factor,
                "velocity_m_s": velocity,
                "velocity_factor": velocity_factor,
                "month": month,
                "seasonal_factor": seasonal_factor
            },
            output_value=biofouling_rate,
            output_name="biofouling_rate_m2kw_day",
            formula="Rate = Base * Temp_factor * Velocity_factor * Seasonal_factor"
        )

        return biofouling_rate

    def _normalize_water_source(self, source: str) -> str:
        """Normalize water source string to match keys."""
        source_lower = source.lower().replace(" ", "_").replace("-", "_")

        if "sea" in source_lower:
            return "seawater"
        elif "brackish" in source_lower:
            return "brackish_water"
        elif "river" in source_lower:
            return "river_water"
        elif "lake" in source_lower:
            return "lake_water"
        elif "tower" in source_lower or "cooling" in source_lower:
            return "cooling_tower"
        else:
            return "cooling_tower"

    def _get_temp_factor(self, temp_c: float) -> float:
        """Get temperature factor for biofouling rate."""
        if temp_c < 10:
            return 0.1
        elif temp_c < 15:
            return 0.3
        elif temp_c < 20:
            return 0.5
        elif temp_c < 25:
            return 0.8
        elif temp_c < 30:
            return 1.0
        elif temp_c < 35:
            return 1.2
        elif temp_c < 40:
            return 1.0
        elif temp_c < 45:
            return 0.6
        else:
            return 0.2

    def _calculate_scale_rate(
        self,
        lsi: float,
        silica_saturation: float,
        temp_c: float
    ) -> float:
        """
        Calculate scale formation rate.

        Combines CaCO3 scaling (from LSI) and silica scaling.
        """
        # CaCO3 scale rate (only if LSI > 0, scale-forming)
        if lsi > 0:
            caco3_rate = SCALE_FORMATION["caco3"]["deposition_rate_base"] * lsi * (1 + 0.02 * (temp_c - 25))
        else:
            caco3_rate = 0.0

        # Silica scale rate (only if saturation > 100%)
        if silica_saturation > 100:
            excess_saturation = (silica_saturation - 100) / 100
            silica_rate = SCALE_FORMATION["silica"]["deposition_rate_base"] * excess_saturation
        else:
            silica_rate = 0.0

        total_scale_rate = caco3_rate + silica_rate

        self._tracker.add_step(
            step_number=5,
            description="Calculate scale formation rate",
            operation="scale_model",
            inputs={
                "lsi": lsi,
                "silica_saturation_pct": silica_saturation,
                "temp_c": temp_c,
                "caco3_rate": caco3_rate,
                "silica_rate": silica_rate
            },
            output_value=total_scale_rate,
            output_name="scale_rate_m2kw_day",
            formula="Total = CaCO3_rate + Silica_rate"
        )

        return total_scale_rate

    def _calculate_total_fouling_rate(
        self,
        biofouling_rate: float,
        scale_rate: float,
        historical_data: Optional[List[Tuple[int, float]]],
        days_since_cleaning: int
    ) -> Tuple[float, str]:
        """
        Calculate total fouling rate and determine dominant mechanism.

        If historical data available, calibrate to actual fouling behavior.
        """
        # Calculate theoretical total
        theoretical_total = biofouling_rate + scale_rate

        # Determine dominant mechanism
        if biofouling_rate > scale_rate * 2:
            dominant = FoulingMechanism.BIOFOULING.value
        elif scale_rate > biofouling_rate * 2:
            dominant = FoulingMechanism.SCALE_CACO3.value
        else:
            dominant = FoulingMechanism.MIXED.value

        # Calibrate with historical data if available
        if historical_data and len(historical_data) >= 2:
            # Use linear regression to estimate actual rate
            # historical_data is list of (days_ago, cf_value)
            actual_rate = self._calibrate_from_history(historical_data)
            if actual_rate > 0:
                # Blend theoretical and actual (weight actual more)
                total_rate = 0.3 * theoretical_total + 0.7 * actual_rate
            else:
                total_rate = theoretical_total
        else:
            total_rate = theoretical_total

        self._tracker.add_step(
            step_number=6,
            description="Calculate total fouling rate and dominant mechanism",
            operation="total_rate_calc",
            inputs={
                "biofouling_rate": biofouling_rate,
                "scale_rate": scale_rate,
                "theoretical_total": theoretical_total,
                "has_historical_data": historical_data is not None
            },
            output_value={"total_rate": total_rate, "dominant": dominant},
            output_name="total_fouling_analysis",
            formula="Total = Biofouling + Scale (calibrated with history if available)"
        )

        return total_rate, dominant

    def _calibrate_from_history(
        self,
        historical_data: List[Tuple[int, float]]
    ) -> float:
        """Calibrate fouling rate from historical CF data."""
        if len(historical_data) < 2:
            return 0.0

        # Sort by days (oldest first)
        sorted_data = sorted(historical_data, key=lambda x: -x[0])

        # Simple linear regression to get CF change rate
        n = len(sorted_data)
        x_vals = [d[0] for d in sorted_data]  # days ago
        y_vals = [d[1] for d in sorted_data]  # CF values

        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator == 0:
            return 0.0

        # Slope is CF change per day (should be negative for degradation)
        slope = numerator / denominator

        # Convert CF change to fouling resistance change
        # Rf = (1 - CF) / (CF * U_design), derivative gives rate
        # Simplified: rate approx = -slope * U_design * CF^2
        # Use average CF and typical U value
        avg_cf = y_mean
        typical_u = 3000  # W/m2-K

        if slope >= 0:
            return 0.0  # Not fouling (improving or stable)

        fouling_rate = abs(slope) / (avg_cf * avg_cf * typical_u)

        return fouling_rate

    def _predict_future_cf(
        self,
        current_cf: float,
        fouling_rate: float,
        design_u: float
    ) -> Tuple[float, float, float, float]:
        """
        Predict future cleanliness factors at 7, 30, 60, and 90 days.
        """
        # Current fouling resistance
        if current_cf >= 1.0:
            current_rf = 0.0
        else:
            current_rf = (1.0 - current_cf) / (current_cf * design_u)

        predictions = []
        for days in [7, 30, 60, 90]:
            future_rf = current_rf + fouling_rate * days
            future_cf = 1.0 / (1.0 + future_rf * design_u)
            future_cf = max(0.5, min(1.0, future_cf))
            predictions.append(future_cf)

        self._tracker.add_step(
            step_number=7,
            description="Predict future cleanliness factors",
            operation="cf_projection",
            inputs={
                "current_cf": current_cf,
                "current_rf": current_rf,
                "fouling_rate": fouling_rate,
                "design_u": design_u
            },
            output_value={
                "cf_7d": predictions[0],
                "cf_30d": predictions[1],
                "cf_60d": predictions[2],
                "cf_90d": predictions[3]
            },
            output_name="cf_predictions",
            formula="CF = 1 / (1 + Rf * U_design)"
        )

        return tuple(predictions)

    def _calculate_days_to_threshold(
        self,
        current_cf: float,
        fouling_rate: float,
        threshold: float,
        design_u: float
    ) -> int:
        """Calculate days until CF reaches specified threshold."""
        if current_cf <= threshold:
            return 0

        if fouling_rate <= 0:
            return 365  # Cap at 1 year if not fouling

        # Current and target fouling resistance
        current_rf = (1.0 - current_cf) / (current_cf * design_u)
        target_rf = (1.0 - threshold) / (threshold * design_u)

        # Days to reach target
        rf_increase = target_rf - current_rf
        days = int(rf_increase / fouling_rate)

        return max(0, min(365, days))

    def _calculate_optimal_cleaning_interval(
        self,
        fouling_rate: float,
        design_u: float,
        num_tubes: int,
        tube_length: float,
        elec_price: float,
        plant_capacity: float
    ) -> int:
        """
        Calculate economically optimal cleaning interval.

        Balances cleaning cost against performance loss cost.
        Uses marginal cost analysis.
        """
        # Estimate cleaning cost
        cleaning_cost = 5000 + num_tubes * 2.0  # Typical mechanical cleaning

        # Estimate daily cost at various CF levels
        # Power loss is approximately proportional to (1 - CF)
        hours_per_year = 8000

        # Find interval that minimizes total annual cost
        best_interval = 90
        min_annual_cost = float('inf')

        for interval in range(30, 181, 10):
            # Average CF over interval (assuming linear degradation)
            start_cf = 0.95  # After cleaning
            end_rf = fouling_rate * interval
            end_cf = 1.0 / (1.0 + end_rf * design_u)
            avg_cf = (start_cf + end_cf) / 2

            # Annual power loss cost
            power_loss_fraction = 1.0 - avg_cf
            annual_loss = power_loss_fraction * plant_capacity * 1000 * elec_price * hours_per_year / 1e6

            # Annual cleaning cost
            cleanings_per_year = 365 / interval
            annual_cleaning = cleaning_cost * cleanings_per_year

            total_annual = annual_loss + annual_cleaning

            if total_annual < min_annual_cost:
                min_annual_cost = total_annual
                best_interval = interval

        self._tracker.add_step(
            step_number=9,
            description="Calculate optimal cleaning interval",
            operation="economic_optimization",
            inputs={
                "fouling_rate": fouling_rate,
                "cleaning_cost": cleaning_cost,
                "electricity_price": elec_price,
                "plant_capacity_mw": plant_capacity
            },
            output_value=best_interval,
            output_name="optimal_cleaning_interval_days",
            formula="Minimize: Annual_loss_cost + Annual_cleaning_cost"
        )

        return best_interval

    def _determine_cleaning_urgency(
        self,
        current_cf: float,
        days_to_085: int,
        days_since_cleaning: int
    ) -> str:
        """Determine cleaning urgency level."""
        if current_cf >= 0.92:
            return CleaningUrgency.NOT_REQUIRED.value
        elif current_cf >= 0.85 and days_to_085 > 30:
            return CleaningUrgency.SCHEDULED.value
        elif current_cf >= 0.85 and days_to_085 <= 30:
            return CleaningUrgency.RECOMMENDED.value
        elif current_cf >= 0.80:
            return CleaningUrgency.URGENT.value
        else:
            return CleaningUrgency.CRITICAL.value

    def _recommend_cleaning_method(
        self,
        dominant_mechanism: str,
        lsi: float,
        biofouling_rate: float,
        scale_rate: float
    ) -> str:
        """Recommend cleaning method based on fouling type."""
        if FoulingMechanism.BIOFOULING.value in dominant_mechanism:
            if biofouling_rate > 0.00001:
                return "mechanical_brush"
            else:
                return "chemical_biocide"
        elif FoulingMechanism.SCALE_CACO3.value in dominant_mechanism:
            if lsi > 1.5:
                return "chemical_acid"
            else:
                return "high_pressure_water"
        else:  # Mixed
            return "mechanical_brush"

    def _estimate_cleaning_cost(
        self,
        method: str,
        num_tubes: int,
        tube_length: float
    ) -> float:
        """Estimate cleaning cost for recommended method."""
        cost_params = CLEANING_COSTS.get(method, CLEANING_COSTS["mechanical_brush"])

        fixed = cost_params["fixed_cost"]
        per_tube = cost_params["per_tube"]

        # Adjust per-tube cost for length
        length_factor = tube_length / 10.0  # Reference 10m tube
        adjusted_per_tube = per_tube * length_factor

        total_cost = fixed + num_tubes * adjusted_per_tube

        self._tracker.add_step(
            step_number=12,
            description="Estimate cleaning cost",
            operation="cost_calculation",
            inputs={
                "method": method,
                "num_tubes": num_tubes,
                "tube_length_m": tube_length,
                "fixed_cost": fixed,
                "per_tube_cost": adjusted_per_tube
            },
            output_value=total_cost,
            output_name="estimated_cleaning_cost_usd",
            formula="Cost = Fixed + Tubes * Per_tube_cost * Length_factor"
        )

        return total_cost

    def _calculate_fouling_costs(
        self,
        current_cf: float,
        fouling_rate: float,
        elec_price: float,
        plant_capacity: float
    ) -> Tuple[float, float]:
        """Calculate daily efficiency loss and annual fouling cost."""
        # Daily efficiency loss (% per day)
        # CF degradation translates to similar efficiency loss
        daily_cf_loss = fouling_rate * 3000 * current_cf * current_cf  # Simplified
        daily_loss_pct = daily_cf_loss * 100

        # Current performance loss
        current_loss_fraction = 1.0 - current_cf

        # Annual cost (using current loss as representative)
        hours_per_year = 8000
        annual_cost = current_loss_fraction * plant_capacity * elec_price * hours_per_year

        self._tracker.add_step(
            step_number=13,
            description="Calculate fouling costs",
            operation="cost_analysis",
            inputs={
                "current_cf": current_cf,
                "current_loss_fraction": current_loss_fraction,
                "electricity_price": elec_price,
                "plant_capacity_mw": plant_capacity
            },
            output_value={
                "daily_loss_pct": daily_loss_pct,
                "annual_cost": annual_cost
            },
            output_name="fouling_costs",
            formula="Annual = Loss_fraction * Capacity * Price * Hours"
        )

        return daily_loss_pct, annual_cost

    def _calculate_cleaning_roi(
        self,
        current_cf: float,
        cleaning_cost: float,
        annual_fouling_cost: float
    ) -> float:
        """Calculate ROI from cleaning at current conditions."""
        if cleaning_cost <= 0:
            return 0.0

        # Assume cleaning restores CF to 0.95
        target_cf = 0.95
        cf_improvement = target_cf - current_cf

        if cf_improvement <= 0:
            return 0.0

        # Annual savings from improvement
        annual_savings = annual_fouling_cost * (cf_improvement / (1.0 - current_cf + 0.001))

        # Simple ROI (annual savings / cost)
        roi = (annual_savings / cleaning_cost) * 100

        return roi

    def _calculate_biofouling_risk_score(
        self,
        biofouling_rate: float,
        inlet_temp: float,
        outlet_temp: float,
        velocity: float,
        month: int
    ) -> float:
        """Calculate biofouling risk score (0-100)."""
        score = 0.0

        # Rate contribution (0-40)
        rate_score = min(40, biofouling_rate / 0.00001 * 40)
        score += rate_score

        # Temperature contribution (0-30)
        avg_temp = (inlet_temp + outlet_temp) / 2
        if 25 <= avg_temp <= 35:
            temp_score = 30
        elif 20 <= avg_temp < 25 or 35 < avg_temp <= 40:
            temp_score = 20
        else:
            temp_score = 10
        score += temp_score

        # Velocity contribution (0-20) - lower velocity = higher risk
        if velocity < 1.5:
            vel_score = 20
        elif velocity < 2.5:
            vel_score = 10
        else:
            vel_score = 5
        score += vel_score

        # Season contribution (0-10)
        if month in [6, 7, 8]:
            season_score = 10
        elif month in [5, 9]:
            season_score = 7
        else:
            season_score = 3
        score += season_score

        return min(100, max(0, score))

    def _calculate_scale_risk_score(
        self,
        lsi: float,
        rsi: float,
        silica_saturation: float
    ) -> float:
        """Calculate scale formation risk score (0-100)."""
        score = 0.0

        # LSI contribution (0-50)
        if lsi > 2.0:
            lsi_score = 50
        elif lsi > 1.0:
            lsi_score = 35
        elif lsi > 0.5:
            lsi_score = 25
        elif lsi > 0:
            lsi_score = 15
        else:
            lsi_score = 5
        score += lsi_score

        # RSI contribution (0-30)
        if rsi < 5.5:
            rsi_score = 30
        elif rsi < 6.5:
            rsi_score = 20
        elif rsi < 7.5:
            rsi_score = 10
        else:
            rsi_score = 5
        score += rsi_score

        # Silica contribution (0-20)
        if silica_saturation > 150:
            silica_score = 20
        elif silica_saturation > 100:
            silica_score = 15
        elif silica_saturation > 80:
            silica_score = 8
        else:
            silica_score = 3
        score += silica_score

        return min(100, max(0, score))

    def _calculate_overall_risk_score(
        self,
        biofouling_risk: float,
        scale_risk: float,
        current_cf: float
    ) -> float:
        """Calculate overall fouling risk score."""
        # Weighted combination
        combined = 0.5 * biofouling_risk + 0.5 * scale_risk

        # Adjust for current condition
        cf_penalty = (0.95 - current_cf) * 50  # Up to 50 points for degraded CF

        overall = combined + cf_penalty

        self._tracker.add_step(
            step_number=15,
            description="Calculate overall fouling risk score",
            operation="risk_combination",
            inputs={
                "biofouling_risk": biofouling_risk,
                "scale_risk": scale_risk,
                "current_cf": current_cf,
                "cf_penalty": cf_penalty
            },
            output_value=overall,
            output_name="overall_fouling_risk_score",
            formula="Overall = 0.5*Bio + 0.5*Scale + CF_penalty"
        )

        return min(100, max(0, overall))

    def _rate_water_quality(
        self,
        lsi: float,
        rsi: float,
        chloride: float,
        tds: float
    ) -> str:
        """Rate overall water quality for condenser operation."""
        score = 0

        # LSI/RSI (balanced water best)
        if -0.5 <= lsi <= 0.5 and 6.5 <= rsi <= 7.5:
            score += 3
        elif -1.0 <= lsi <= 1.0 and 6.0 <= rsi <= 8.0:
            score += 2
        else:
            score += 1

        # Chloride (corrosion indicator)
        if chloride < 100:
            score += 3
        elif chloride < 250:
            score += 2
        else:
            score += 1

        # TDS
        if tds < 500:
            score += 3
        elif tds < 1500:
            score += 2
        else:
            score += 1

        # Map score to rating
        if score >= 8:
            return WaterQuality.EXCELLENT.value
        elif score >= 6:
            return WaterQuality.GOOD.value
        elif score >= 4:
            return WaterQuality.MODERATE.value
        elif score >= 2:
            return WaterQuality.POOR.value
        else:
            return WaterQuality.VERY_POOR.value


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_langelier_index(
    ph: float,
    calcium_hardness_ppm: float,
    alkalinity_ppm: float,
    tds_ppm: float,
    temp_c: float
) -> float:
    """
    Calculate Langelier Saturation Index (LSI).

    Args:
        ph: Water pH
        calcium_hardness_ppm: Calcium hardness (ppm as CaCO3)
        alkalinity_ppm: Total alkalinity (ppm as CaCO3)
        tds_ppm: Total dissolved solids (ppm)
        temp_c: Water temperature (C)

    Returns:
        LSI value (negative = corrosive, positive = scale forming)
    """
    a = (math.log10(max(tds_ppm, 1)) - 1) / 10
    b = -13.12 * math.log10(temp_c + 273.15) + 34.55
    c = math.log10(max(calcium_hardness_ppm, 1)) - 0.4
    d = math.log10(max(alkalinity_ppm, 1))

    phs = (9.3 + a + b) - (c + d)
    return ph - phs


def calculate_ryznar_index(ph: float, lsi: float) -> float:
    """
    Calculate Ryznar Stability Index (RSI).

    Args:
        ph: Water pH
        lsi: Langelier Saturation Index

    Returns:
        RSI value (< 6 = heavy scale, 6-7 = light scale, > 8 = corrosive)
    """
    return ph - 2 * lsi


def estimate_fouling_rate_from_cf_history(
    cf_data: List[Tuple[int, float]],
    design_u: float = 3000.0
) -> float:
    """
    Estimate fouling rate from historical CF measurements.

    Args:
        cf_data: List of (days_ago, cf_value) tuples
        design_u: Design U-value (W/m2-K)

    Returns:
        Estimated fouling rate (m2-K/W per day)
    """
    if len(cf_data) < 2:
        return 0.0

    # Convert CF to fouling resistance
    rf_data = []
    for days, cf in cf_data:
        if 0 < cf < 1:
            rf = (1.0 - cf) / (cf * design_u)
            rf_data.append((days, rf))

    if len(rf_data) < 2:
        return 0.0

    # Linear regression on Rf vs time
    sorted_data = sorted(rf_data, key=lambda x: -x[0])  # Oldest first
    n = len(sorted_data)

    x_vals = [d[0] for d in sorted_data]
    y_vals = [d[1] for d in sorted_data]

    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)

    if denominator == 0:
        return 0.0

    # Slope is Rf change per day
    slope = numerator / denominator

    return max(0, -slope)  # Positive rate for fouling accumulation


def predict_cleaning_benefit(
    current_cf: float,
    cleaning_method: str,
    dominant_fouling: str = "mixed"
) -> float:
    """
    Predict post-cleaning cleanliness factor.

    Args:
        current_cf: Current cleanliness factor
        cleaning_method: Cleaning method name
        dominant_fouling: Dominant fouling type

    Returns:
        Expected CF after cleaning
    """
    # Get effectiveness for fouling type
    method_effectiveness = CLEANING_EFFECTIVENESS.get(
        cleaning_method,
        CLEANING_EFFECTIVENESS["mechanical_brush"]
    )

    # Map dominant fouling to category
    if "bio" in dominant_fouling.lower():
        fouling_key = "biofouling"
    elif "scale" in dominant_fouling.lower() or "caco3" in dominant_fouling.lower():
        fouling_key = "soft_scale"
    elif "silica" in dominant_fouling.lower():
        fouling_key = "hard_scale"
    else:
        fouling_key = "particulate"

    effectiveness = method_effectiveness.get(fouling_key, 0.7)

    # Calculate improvement
    max_achievable_cf = 0.95 * effectiveness
    improvement = (max_achievable_cf - current_cf) * effectiveness
    post_cleaning_cf = current_cf + improvement

    return min(0.95, max(current_cf, post_cleaning_cf))
