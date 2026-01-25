"""
EfficiencyPredictor - Combustion Efficiency Prediction for GL-004 BURNMASTER

This module implements combustion efficiency prediction using ML models
with physics-informed constraints. Predicts efficiency and identifies
optimization opportunities.

Key Features:
    - Combustion efficiency prediction (85-99%)
    - Stack loss estimation
    - Optimization opportunity identification
    - Trend analysis and degradation detection
    - Physics-constrained predictions
    - What-if scenario analysis

CRITICAL: Efficiency predictions are ADVISORY ONLY.
Control decisions use deterministic physics-based calculations.

Example:
    >>> predictor = EfficiencyPredictor()
    >>> features = EfficiencyFeatures(
    ...     o2_percent=3.5,
    ...     stack_temp_c=180,
    ...     fuel_type="natural_gas"
    ... )
    >>> prediction = predictor.predict(features)
    >>> print(f"Efficiency: {prediction.efficiency_percent:.1f}%")

Author: GreenLang ML Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# Optional imports with graceful degradation
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using physics-based fallback only")


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class FuelType(str, Enum):
    """Fuel types for efficiency calculation."""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    LPG = "lpg"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    DIESEL = "diesel"
    HYDROGEN = "hydrogen"
    MIXED = "mixed"


class EfficiencyTrend(str, Enum):
    """Efficiency trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


class OptimizationPotential(str, Enum):
    """Level of optimization potential."""
    HIGH = "high"  # >3% improvement possible
    MEDIUM = "medium"  # 1-3% improvement
    LOW = "low"  # 0.5-1% improvement
    OPTIMAL = "optimal"  # Near optimal


# Fuel-specific constants for Siegert formula
FUEL_CONSTANTS = {
    FuelType.NATURAL_GAS: {"K1": 0.37, "K2": 0.009, "CO2_max": 11.7, "H2_content": 23.0},
    FuelType.PROPANE: {"K1": 0.38, "K2": 0.008, "CO2_max": 13.8, "H2_content": 18.0},
    FuelType.LPG: {"K1": 0.38, "K2": 0.008, "CO2_max": 13.5, "H2_content": 18.0},
    FuelType.FUEL_OIL_2: {"K1": 0.48, "K2": 0.007, "CO2_max": 15.4, "H2_content": 12.0},
    FuelType.FUEL_OIL_6: {"K1": 0.52, "K2": 0.007, "CO2_max": 16.0, "H2_content": 10.0},
    FuelType.DIESEL: {"K1": 0.48, "K2": 0.007, "CO2_max": 15.3, "H2_content": 12.5},
    FuelType.HYDROGEN: {"K1": 0.21, "K2": 0.012, "CO2_max": 0.0, "H2_content": 100.0},
    FuelType.MIXED: {"K1": 0.40, "K2": 0.008, "CO2_max": 12.0, "H2_content": 20.0},
}

# Typical efficiency ranges by fuel type
EFFICIENCY_RANGES = {
    FuelType.NATURAL_GAS: (80.0, 95.0),
    FuelType.PROPANE: (78.0, 93.0),
    FuelType.LPG: (78.0, 93.0),
    FuelType.FUEL_OIL_2: (75.0, 90.0),
    FuelType.FUEL_OIL_6: (72.0, 88.0),
    FuelType.DIESEL: (75.0, 90.0),
    FuelType.HYDROGEN: (82.0, 96.0),
    FuelType.MIXED: (75.0, 92.0),
}


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================


class EfficiencyFeatures(BaseModel):
    """Input features for efficiency prediction."""

    # Flue gas composition
    o2_percent: float = Field(
        default=3.0, ge=0.0, le=21.0,
        description="O2 percentage in dry flue gas"
    )
    co_ppm: float = Field(
        default=50.0, ge=0.0,
        description="CO concentration in ppm"
    )
    co2_percent: Optional[float] = Field(
        default=None, ge=0.0, le=25.0,
        description="CO2 percentage (measured or calculated)"
    )

    # Temperature measurements
    stack_temp_c: float = Field(
        default=180.0, ge=50.0, le=500.0,
        description="Stack/flue gas temperature in Celsius"
    )
    ambient_temp_c: float = Field(
        default=25.0, ge=-40.0, le=50.0,
        description="Ambient air temperature in Celsius"
    )
    combustion_air_temp_c: Optional[float] = Field(
        default=None, ge=-40.0, le=400.0,
        description="Combustion air temperature (if preheated)"
    )

    # Fuel information
    fuel_type: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Type of fuel"
    )
    fuel_hhv_mj_m3: Optional[float] = Field(
        default=None, ge=0.0, le=150.0,
        description="Fuel higher heating value (MJ/m3)"
    )
    fuel_moisture_percent: Optional[float] = Field(
        default=None, ge=0.0, le=50.0,
        description="Fuel moisture content"
    )

    # Operating conditions
    load_percent: float = Field(
        default=80.0, ge=10.0, le=110.0,
        description="Burner load percentage"
    )
    lambda_value: Optional[float] = Field(
        default=None, ge=0.5, le=3.0,
        description="Lambda (air-fuel equivalence ratio)"
    )
    excess_air_percent: Optional[float] = Field(
        default=None, ge=-50.0, le=200.0,
        description="Excess air percentage"
    )

    # Loss estimates (if measured)
    radiation_loss_percent: Optional[float] = Field(
        default=None, ge=0.0, le=10.0,
        description="Surface radiation loss"
    )
    blowdown_loss_percent: Optional[float] = Field(
        default=None, ge=0.0, le=5.0,
        description="Boiler blowdown loss (if steam)"
    )

    # Metadata
    burner_id: str = Field(default="BNR-001", description="Burner ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )


class EfficiencyFactors(BaseModel):
    """Breakdown of efficiency losses."""

    # Major losses
    dry_flue_gas_loss_percent: float = Field(
        ..., ge=0.0, le=30.0,
        description="Dry flue gas sensible heat loss"
    )
    moisture_loss_percent: float = Field(
        ..., ge=0.0, le=15.0,
        description="Latent heat loss from H2O formation"
    )
    co_loss_percent: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Unburned CO loss"
    )

    # Minor losses
    radiation_loss_percent: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description="Surface radiation loss"
    )
    other_losses_percent: float = Field(
        default=0.5, ge=0.0, le=5.0,
        description="Other unaccounted losses"
    )

    # Totals
    total_losses_percent: float = Field(
        ..., ge=0.0, le=50.0,
        description="Total heat losses"
    )
    gross_efficiency_percent: float = Field(
        ..., ge=50.0, le=100.0,
        description="Gross combustion efficiency"
    )


class OptimizationOpportunity(BaseModel):
    """Identified efficiency optimization opportunity."""

    opportunity_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique opportunity ID"
    )
    category: str = Field(..., description="Opportunity category")
    description: str = Field(..., description="Detailed description")
    current_value: float = Field(..., description="Current value")
    target_value: float = Field(..., description="Recommended target")
    potential_gain_percent: float = Field(
        ..., ge=0.0, le=10.0,
        description="Potential efficiency gain"
    )
    implementation_difficulty: str = Field(
        default="medium",
        description="easy, medium, hard"
    )
    payback_months: Optional[float] = Field(
        default=None, ge=0.0,
        description="Estimated payback period"
    )


class EfficiencyPrediction(BaseModel):
    """Prediction result for efficiency model."""

    prediction_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique prediction identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Prediction timestamp"
    )

    # Core predictions
    efficiency_percent: float = Field(
        ..., ge=50.0, le=100.0,
        description="Predicted combustion efficiency"
    )
    efficiency_lower_bound: float = Field(
        ..., ge=50.0, le=100.0,
        description="Lower confidence bound (95%)"
    )
    efficiency_upper_bound: float = Field(
        ..., ge=50.0, le=100.0,
        description="Upper confidence bound (95%)"
    )

    # Loss breakdown
    efficiency_factors: EfficiencyFactors = Field(
        ..., description="Breakdown of efficiency losses"
    )

    # Optimization
    optimization_potential: OptimizationPotential = Field(
        ..., description="Level of optimization potential"
    )
    potential_improvement_percent: float = Field(
        default=0.0, ge=0.0, le=10.0,
        description="Potential efficiency improvement"
    )
    opportunities: List[OptimizationOpportunity] = Field(
        default_factory=list,
        description="Identified optimization opportunities"
    )

    # Trend analysis
    trend: EfficiencyTrend = Field(
        default=EfficiencyTrend.UNKNOWN,
        description="Efficiency trend direction"
    )
    trend_rate_per_day: Optional[float] = Field(
        default=None,
        description="Trend rate (% per day)"
    )

    # Benchmarking
    vs_design_efficiency: float = Field(
        default=0.0,
        description="Difference from design efficiency"
    )
    vs_optimal_efficiency: float = Field(
        default=0.0,
        description="Gap from optimal achievable efficiency"
    )
    percentile_rank: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Percentile rank vs. similar equipment"
    )

    # Confidence
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Prediction confidence"
    )

    # What-if recommendations
    what_if_scenarios: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="What-if scenario analysis"
    )

    # Provenance
    model_version: str = Field(default="1.0.0", description="Model version")
    is_physics_fallback: bool = Field(
        default=False,
        description="Whether physics fallback was used"
    )
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")
    computation_time_ms: float = Field(default=0.0, ge=0.0)


# =============================================================================
# EFFICIENCY PREDICTOR
# =============================================================================


class EfficiencyPredictor:
    """
    Combustion efficiency prediction model.

    Uses ML ensemble with physics-informed constraints.
    Predictions are bounded by thermodynamic limits.

    CRITICAL: Efficiency predictions are ADVISORY ONLY.

    Attributes:
        is_fitted: Whether model has been trained
        model_id: Unique model identifier
        design_efficiency: Equipment design efficiency

    Example:
        >>> predictor = EfficiencyPredictor(design_efficiency=90.0)
        >>> features = EfficiencyFeatures(o2_percent=4.0, stack_temp_c=200)
        >>> prediction = predictor.predict(features)
        >>> print(f"Efficiency: {prediction.efficiency_percent:.1f}%")
    """

    FEATURE_NAMES = [
        "o2_percent",
        "co_ppm",
        "stack_temp_c",
        "ambient_temp_c",
        "combustion_air_temp_c",
        "load_percent",
        "lambda_value",
        "excess_air_percent",
        "radiation_loss_percent",
    ]

    def __init__(
        self,
        model_path: Optional[Path] = None,
        design_efficiency: float = 90.0,
        random_seed: int = 42
    ):
        """
        Initialize EfficiencyPredictor.

        Args:
            model_path: Path to pre-trained model file
            design_efficiency: Equipment design efficiency
            random_seed: Random seed for reproducibility
        """
        self.design_efficiency = design_efficiency
        self.random_seed = random_seed
        self._model_id = f"efficiency_{uuid4().hex[:8]}"

        self._regressor: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._is_fitted = False
        self._feature_importance: Dict[str, float] = {}

        # Efficiency history for trend analysis
        self._efficiency_history: Deque[Tuple[datetime, float]] = deque(maxlen=1000)

        if model_path and model_path.exists():
            self._load_model(model_path)
        elif SKLEARN_AVAILABLE:
            self._initialize_default_models()

        logger.info(
            f"EfficiencyPredictor initialized: "
            f"id={self._model_id}, design_efficiency={design_efficiency}%"
        )

    def _initialize_default_models(self) -> None:
        """Initialize default model architecture."""
        if not SKLEARN_AVAILABLE:
            return

        self._regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_seed
        )

        self._scaler = StandardScaler()

    def predict(self, features: EfficiencyFeatures) -> EfficiencyPrediction:
        """
        Predict combustion efficiency.

        Args:
            features: Current operating features

        Returns:
            EfficiencyPrediction with detailed breakdown
        """
        start_time = time.time()

        # Extract feature vector
        feature_vector = self._extract_features(features)

        # Use ML if available, otherwise physics
        if self._is_fitted and SKLEARN_AVAILABLE:
            prediction = self._predict_with_ml(features, feature_vector)
        else:
            prediction = self._predict_with_physics(features)

        # Add to history for trend
        self._efficiency_history.append(
            (prediction.timestamp, prediction.efficiency_percent)
        )

        # Calculate trend
        prediction.trend, prediction.trend_rate_per_day = self._calculate_trend()

        # Compute provenance hash
        prediction.provenance_hash = self._compute_provenance_hash(
            features, prediction
        )
        prediction.computation_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Efficiency prediction: efficiency={prediction.efficiency_percent:.1f}%, "
            f"potential_gain={prediction.potential_improvement_percent:.2f}%"
        )

        return prediction

    def predict_what_if(
        self,
        features: EfficiencyFeatures,
        scenario: Dict[str, float]
    ) -> EfficiencyPrediction:
        """
        Predict efficiency for a what-if scenario.

        Args:
            features: Current operating features
            scenario: Dict of feature changes to apply

        Returns:
            EfficiencyPrediction for the scenario
        """
        # Apply scenario changes
        features_dict = features.model_dump()
        for key, value in scenario.items():
            if key in features_dict:
                features_dict[key] = value

        modified_features = EfficiencyFeatures(**features_dict)
        return self.predict(modified_features)

    def _predict_with_ml(
        self,
        features: EfficiencyFeatures,
        feature_vector: np.ndarray
    ) -> EfficiencyPrediction:
        """Make prediction using trained ML model."""
        # Scale features
        if self._scaler and hasattr(self._scaler, "mean_"):
            feature_vector_scaled = self._scaler.transform(
                feature_vector.reshape(1, -1)
            )
        else:
            feature_vector_scaled = feature_vector.reshape(1, -1)

        # ML prediction
        if self._regressor:
            efficiency = float(self._regressor.predict(feature_vector_scaled)[0])
        else:
            efficiency = self._calculate_physics_efficiency(features)

        # Apply physics constraints
        min_eff, max_eff = EFFICIENCY_RANGES.get(
            features.fuel_type,
            (70.0, 95.0)
        )
        efficiency = max(min_eff, min(max_eff, efficiency))

        # Calculate loss breakdown using physics
        factors = self._calculate_loss_breakdown(features, efficiency)

        # Calculate uncertainty
        std_estimate = 0.5 + (100 - efficiency) * 0.02
        lower_bound = max(min_eff, efficiency - 1.96 * std_estimate)
        upper_bound = min(max_eff, efficiency + 1.96 * std_estimate)

        # Identify optimization opportunities
        opportunities = self._identify_opportunities(features, factors)

        # Calculate optimization potential
        potential = sum(o.potential_gain_percent for o in opportunities)
        if potential > 3.0:
            opt_level = OptimizationPotential.HIGH
        elif potential > 1.0:
            opt_level = OptimizationPotential.MEDIUM
        elif potential > 0.5:
            opt_level = OptimizationPotential.LOW
        else:
            opt_level = OptimizationPotential.OPTIMAL

        # Benchmarking
        vs_design = efficiency - self.design_efficiency
        optimal_for_fuel = max_eff - 2  # Realistic optimal
        vs_optimal = efficiency - optimal_for_fuel

        # What-if scenarios
        what_ifs = self._generate_what_if_scenarios(features, efficiency)

        return EfficiencyPrediction(
            efficiency_percent=round(efficiency, 2),
            efficiency_lower_bound=round(lower_bound, 2),
            efficiency_upper_bound=round(upper_bound, 2),
            efficiency_factors=factors,
            optimization_potential=opt_level,
            potential_improvement_percent=round(potential, 2),
            opportunities=opportunities,
            vs_design_efficiency=round(vs_design, 2),
            vs_optimal_efficiency=round(vs_optimal, 2),
            confidence=0.85,
            what_if_scenarios=what_ifs,
            model_version="1.0.0",
            is_physics_fallback=False
        )

    def _predict_with_physics(
        self, features: EfficiencyFeatures
    ) -> EfficiencyPrediction:
        """
        Make prediction using physics-based model (DETERMINISTIC).

        Uses Siegert formula for stack loss calculation.
        """
        # Calculate efficiency using Siegert formula
        efficiency = self._calculate_physics_efficiency(features)

        # Calculate loss breakdown
        factors = self._calculate_loss_breakdown(features, efficiency)

        # Uncertainty (higher for physics model)
        min_eff, max_eff = EFFICIENCY_RANGES.get(
            features.fuel_type, (70.0, 95.0)
        )
        std_estimate = 1.0
        lower_bound = max(min_eff, efficiency - 1.96 * std_estimate)
        upper_bound = min(max_eff, efficiency + 1.96 * std_estimate)

        # Identify opportunities
        opportunities = self._identify_opportunities(features, factors)
        potential = sum(o.potential_gain_percent for o in opportunities)

        # Optimization level
        if potential > 3.0:
            opt_level = OptimizationPotential.HIGH
        elif potential > 1.0:
            opt_level = OptimizationPotential.MEDIUM
        elif potential > 0.5:
            opt_level = OptimizationPotential.LOW
        else:
            opt_level = OptimizationPotential.OPTIMAL

        # What-if scenarios
        what_ifs = self._generate_what_if_scenarios(features, efficiency)

        return EfficiencyPrediction(
            efficiency_percent=round(efficiency, 2),
            efficiency_lower_bound=round(lower_bound, 2),
            efficiency_upper_bound=round(upper_bound, 2),
            efficiency_factors=factors,
            optimization_potential=opt_level,
            potential_improvement_percent=round(potential, 2),
            opportunities=opportunities,
            vs_design_efficiency=round(efficiency - self.design_efficiency, 2),
            vs_optimal_efficiency=round(efficiency - (max_eff - 2), 2),
            confidence=0.75,
            what_if_scenarios=what_ifs,
            model_version="1.0.0",
            is_physics_fallback=True
        )

    def _calculate_physics_efficiency(
        self, features: EfficiencyFeatures
    ) -> float:
        """
        Calculate efficiency using Siegert formula.

        DETERMINISTIC: Physics-based calculation.

        Siegert formula:
        Stack Loss = K1 * (Tstack - Tambient) / CO2%

        Efficiency = 100 - Stack Loss - Moisture Loss - CO Loss - Radiation Loss
        """
        fuel_constants = FUEL_CONSTANTS.get(
            features.fuel_type,
            FUEL_CONSTANTS[FuelType.NATURAL_GAS]
        )

        K1 = fuel_constants["K1"]
        K2 = fuel_constants["K2"]
        CO2_max = fuel_constants["CO2_max"]
        H2_content = fuel_constants["H2_content"]

        # Temperature difference
        temp_diff = features.stack_temp_c - features.ambient_temp_c

        # Calculate CO2 from O2 if not provided
        if features.co2_percent is not None:
            co2_pct = features.co2_percent
        else:
            # CO2 = CO2_max * (21 - O2) / 21
            co2_pct = CO2_max * (21.0 - features.o2_percent) / 21.0
            co2_pct = max(0.1, co2_pct)  # Avoid division by zero

        # Dry flue gas loss (Siegert)
        dry_flue_loss = K1 * temp_diff / co2_pct

        # Moisture loss (latent heat of H2O from fuel hydrogen)
        # Approximately 9 kg H2O per kg H in fuel
        moisture_loss = K2 * H2_content * temp_diff / 10

        # CO loss (incomplete combustion)
        # Each ppm CO represents ~0.0001% loss
        co_loss = features.co_ppm * 0.0001 if features.co_ppm > 0 else 0.0

        # Radiation loss (use provided or default)
        radiation_loss = features.radiation_loss_percent or 1.0

        # Other losses
        other_loss = 0.5

        # Total efficiency
        total_loss = dry_flue_loss + moisture_loss + co_loss + radiation_loss + other_loss
        efficiency = 100.0 - total_loss

        # Apply physical limits
        min_eff, max_eff = EFFICIENCY_RANGES.get(
            features.fuel_type, (70.0, 95.0)
        )
        efficiency = max(min_eff, min(max_eff, efficiency))

        return efficiency

    def _calculate_loss_breakdown(
        self,
        features: EfficiencyFeatures,
        efficiency: float
    ) -> EfficiencyFactors:
        """Calculate detailed loss breakdown."""
        fuel_constants = FUEL_CONSTANTS.get(
            features.fuel_type,
            FUEL_CONSTANTS[FuelType.NATURAL_GAS]
        )

        K1 = fuel_constants["K1"]
        K2 = fuel_constants["K2"]
        CO2_max = fuel_constants["CO2_max"]
        H2_content = fuel_constants["H2_content"]

        temp_diff = features.stack_temp_c - features.ambient_temp_c

        # CO2 calculation
        if features.co2_percent is not None:
            co2_pct = features.co2_percent
        else:
            co2_pct = max(0.1, CO2_max * (21.0 - features.o2_percent) / 21.0)

        # Individual losses
        dry_flue_loss = K1 * temp_diff / co2_pct
        moisture_loss = K2 * H2_content * temp_diff / 10
        co_loss = features.co_ppm * 0.0001 if features.co_ppm > 0 else 0.0
        radiation_loss = features.radiation_loss_percent or 1.0
        other_loss = 0.5

        total_loss = dry_flue_loss + moisture_loss + co_loss + radiation_loss + other_loss

        return EfficiencyFactors(
            dry_flue_gas_loss_percent=round(dry_flue_loss, 2),
            moisture_loss_percent=round(moisture_loss, 2),
            co_loss_percent=round(co_loss, 2),
            radiation_loss_percent=round(radiation_loss, 2),
            other_losses_percent=round(other_loss, 2),
            total_losses_percent=round(total_loss, 2),
            gross_efficiency_percent=round(efficiency, 2)
        )

    def _identify_opportunities(
        self,
        features: EfficiencyFeatures,
        factors: EfficiencyFactors
    ) -> List[OptimizationOpportunity]:
        """Identify efficiency optimization opportunities."""
        opportunities = []

        # Opportunity 1: Reduce excess air
        if features.o2_percent > 3.5:
            current_o2 = features.o2_percent
            target_o2 = 3.0
            # Rule of thumb: 1% excess O2 reduction = 0.3% efficiency gain
            gain = (current_o2 - target_o2) * 0.3

            opportunities.append(OptimizationOpportunity(
                category="air_fuel_ratio",
                description=f"Reduce excess O2 from {current_o2:.1f}% to {target_o2:.1f}%",
                current_value=current_o2,
                target_value=target_o2,
                potential_gain_percent=round(gain, 2),
                implementation_difficulty="easy",
                payback_months=1.0
            ))

        # Opportunity 2: Reduce stack temperature
        if features.stack_temp_c > 200:
            current_temp = features.stack_temp_c
            target_temp = 180.0
            # Rule of thumb: 20C stack temp reduction = 1% efficiency gain
            gain = (current_temp - target_temp) / 20.0

            opportunities.append(OptimizationOpportunity(
                category="heat_recovery",
                description=f"Reduce stack temperature from {current_temp:.0f}C to {target_temp:.0f}C",
                current_value=current_temp,
                target_value=target_temp,
                potential_gain_percent=round(min(gain, 3.0), 2),
                implementation_difficulty="medium",
                payback_months=12.0
            ))

        # Opportunity 3: Reduce CO
        if features.co_ppm > 100:
            current_co = features.co_ppm
            target_co = 50.0
            gain = (current_co - target_co) * 0.0001 * 10  # Amplified for visibility

            opportunities.append(OptimizationOpportunity(
                category="combustion_quality",
                description=f"Reduce CO from {current_co:.0f} ppm to {target_co:.0f} ppm",
                current_value=current_co,
                target_value=target_co,
                potential_gain_percent=round(min(gain, 1.0), 2),
                implementation_difficulty="medium",
                payback_months=3.0
            ))

        # Opportunity 4: Combustion air preheat
        air_temp = features.combustion_air_temp_c or features.ambient_temp_c
        if air_temp < 100 and factors.dry_flue_gas_loss_percent > 8:
            current_air_temp = air_temp
            target_air_temp = 150.0
            # Rule of thumb: 50C air preheat = 1% efficiency gain
            gain = (target_air_temp - current_air_temp) / 50.0

            opportunities.append(OptimizationOpportunity(
                category="air_preheat",
                description=f"Install/improve combustion air preheater to {target_air_temp:.0f}C",
                current_value=current_air_temp,
                target_value=target_air_temp,
                potential_gain_percent=round(min(gain, 2.5), 2),
                implementation_difficulty="hard",
                payback_months=24.0
            ))

        # Opportunity 5: Load optimization
        if features.load_percent < 60:
            opportunities.append(OptimizationOpportunity(
                category="load_optimization",
                description="Low load operation reduces efficiency - consider scheduling",
                current_value=features.load_percent,
                target_value=80.0,
                potential_gain_percent=1.0,
                implementation_difficulty="medium",
                payback_months=0.0
            ))

        # Sort by potential gain
        opportunities.sort(
            key=lambda x: x.potential_gain_percent,
            reverse=True
        )

        return opportunities[:5]

    def _generate_what_if_scenarios(
        self,
        features: EfficiencyFeatures,
        current_efficiency: float
    ) -> List[Dict[str, Any]]:
        """Generate what-if scenario analyses."""
        scenarios = []

        # Scenario 1: Optimal O2
        if features.o2_percent > 3.0:
            optimal_o2_eff = current_efficiency + (features.o2_percent - 3.0) * 0.3
            scenarios.append({
                "name": "Optimal O2 Control",
                "changes": {"o2_percent": 3.0},
                "expected_efficiency": round(min(optimal_o2_eff, 97), 2),
                "improvement": round(optimal_o2_eff - current_efficiency, 2)
            })

        # Scenario 2: Reduced stack temperature
        if features.stack_temp_c > 160:
            reduced_stack_eff = current_efficiency + (features.stack_temp_c - 160) / 20
            scenarios.append({
                "name": "Economizer Installation",
                "changes": {"stack_temp_c": 160},
                "expected_efficiency": round(min(reduced_stack_eff, 97), 2),
                "improvement": round(reduced_stack_eff - current_efficiency, 2)
            })

        # Scenario 3: Air preheat
        air_temp = features.combustion_air_temp_c or features.ambient_temp_c
        if air_temp < 150:
            preheated_eff = current_efficiency + (150 - air_temp) / 50
            scenarios.append({
                "name": "Combustion Air Preheat",
                "changes": {"combustion_air_temp_c": 150},
                "expected_efficiency": round(min(preheated_eff, 97), 2),
                "improvement": round(preheated_eff - current_efficiency, 2)
            })

        # Scenario 4: Perfect combustion
        perfect_eff = current_efficiency + features.co_ppm * 0.0002
        scenarios.append({
            "name": "Perfect Combustion (CO=0)",
            "changes": {"co_ppm": 0},
            "expected_efficiency": round(min(perfect_eff, 97), 2),
            "improvement": round(perfect_eff - current_efficiency, 2)
        })

        return scenarios

    def _calculate_trend(self) -> Tuple[EfficiencyTrend, Optional[float]]:
        """Calculate efficiency trend from history."""
        if len(self._efficiency_history) < 10:
            return EfficiencyTrend.UNKNOWN, None

        # Get last 24 hours of data (assuming hourly samples)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)

        recent_data = [
            (ts, eff) for ts, eff in self._efficiency_history
            if ts > cutoff
        ]

        if len(recent_data) < 5:
            return EfficiencyTrend.UNKNOWN, None

        # Simple linear regression
        times = [(ts - recent_data[0][0]).total_seconds() / 86400 for ts, _ in recent_data]
        effs = [eff for _, eff in recent_data]

        n = len(times)
        sum_x = sum(times)
        sum_y = sum(effs)
        sum_xy = sum(t * e for t, e in zip(times, effs))
        sum_xx = sum(t * t for t in times)

        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return EfficiencyTrend.STABLE, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator  # % per day

        if slope > 0.05:
            trend = EfficiencyTrend.IMPROVING
        elif slope < -0.05:
            trend = EfficiencyTrend.DEGRADING
        else:
            trend = EfficiencyTrend.STABLE

        return trend, round(slope, 4)

    def _extract_features(self, features: EfficiencyFeatures) -> np.ndarray:
        """Extract feature vector from input."""
        return np.array([
            features.o2_percent,
            features.co_ppm,
            features.stack_temp_c,
            features.ambient_temp_c,
            features.combustion_air_temp_c or features.ambient_temp_c,
            features.load_percent,
            features.lambda_value or (1 + features.o2_percent / 21 * 5),
            features.excess_air_percent or (features.o2_percent / (21 - features.o2_percent) * 100),
            features.radiation_loss_percent or 1.0,
        ], dtype=np.float64)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the efficiency prediction model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Efficiency values (%)
            validation_split: Validation set fraction

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for training")

        start_time = time.time()

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Train regressor
        self._regressor.fit(X_scaled, y)

        # Store feature importance
        if hasattr(self._regressor, "feature_importances_"):
            self._feature_importance = dict(zip(
                self.FEATURE_NAMES,
                [float(v) for v in self._regressor.feature_importances_]
            ))

        self._is_fitted = True

        elapsed = time.time() - start_time

        # Calculate training metrics
        y_pred = self._regressor.predict(X_scaled)
        mae = float(np.mean(np.abs(y - y_pred)))
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

        return {
            "training_time_s": elapsed,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "mae": mae,
            "rmse": rmse,
            "feature_importance": self._feature_importance
        }

    def save_model(self, path: Path) -> None:
        """Save model to file."""
        data = {
            "regressor": self._regressor,
            "scaler": self._scaler,
            "feature_importance": self._feature_importance,
            "is_fitted": self._is_fitted,
            "model_id": self._model_id,
            "design_efficiency": self.design_efficiency
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {path}")

    def _load_model(self, path: Path) -> None:
        """Load model from file."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._regressor = data.get("regressor")
            self._scaler = data.get("scaler")
            self._feature_importance = data.get("feature_importance", {})
            self._is_fitted = data.get("is_fitted", False)
            self._model_id = data.get("model_id", self._model_id)
            self.design_efficiency = data.get("design_efficiency", self.design_efficiency)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if SKLEARN_AVAILABLE:
                self._initialize_default_models()

    def _compute_provenance_hash(
        self,
        features: EfficiencyFeatures,
        prediction: EfficiencyPrediction
    ) -> str:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "model_id": self._model_id,
            "features": {
                "o2_percent": features.o2_percent,
                "stack_temp_c": features.stack_temp_c,
                "co_ppm": features.co_ppm,
                "fuel_type": features.fuel_type.value
            },
            "efficiency_percent": prediction.efficiency_percent,
            "timestamp": prediction.timestamp.isoformat()
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    @property
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        return self._feature_importance.copy()
